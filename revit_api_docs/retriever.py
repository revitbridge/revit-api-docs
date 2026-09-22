"""
Two-tier retriever — ChromaDB (semantic index) + SQLite (content store).

Architecture:
    ChromaDB holds lightweight embeddings (summary-based) + document IDs.
    SQLite holds the full content (code, API docs, parameters, etc.).
    At query time:
        0. (Optional) LLM query rewriting: extract Revit API keywords from user query
        1. Embed the rewritten query and search ChromaDB → ranked IDs + scores
        2. Batch-fetch full records from SQLite by ID
        3. Assemble structured context dicts ready for prompt injection

Public API:
    retriever = RAGRetriever(config, api_db, sdk_db, chromadb_api_dir, chromadb_code_dir)
    results   = retriever.search(query, api_top_k=15, code_top_k=5)
    context   = retriever.build_context(results)

Degraded modes (no network keys needed for the MCP server to be useful):
    - No embedding API key  -> vector search is skipped, SQLite keyword search only.
    - No rerank API key     -> results keep vector/keyword order.
    - No LLM key            -> query rewriting is skipped.
"""
from __future__ import annotations

import json
import os
import re
import sqlite3
import threading
from dataclasses import dataclass, field
from typing import Any

from .prompts import load_prompt


def _escape_like(s: str) -> str:
    """Escape LIKE wildcards % and _ (used with ESCAPE '\\'); the backslash itself too."""
    return s.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")


@dataclass
class RetrievedItem:
    """A single retrieved item with full content from SQLite."""
    source: str               # "api" or "sdk"
    chromadb_id: str
    distance: float
    # API fields
    name: str = ""
    full_id: str = ""
    summary: str = ""
    info: str = ""
    syntax: str = ""
    parameters: str = ""
    remark: str = ""
    # SDK fields
    project: str = ""
    content: str = ""         # golden code snippet
    mentioned_apis: str = ""


@dataclass
class SearchResults:
    """Container for all retrieval results."""
    query: str
    rewritten_query: str = ""
    api_items: list[RetrievedItem] = field(default_factory=list)
    sdk_items: list[RetrievedItem] = field(default_factory=list)


class RAGRetriever:
    """
    Two-tier retriever: ChromaDB (semantic search) + SQLite (content store)
    with optional post-retrieval reranking.

    Works with both existing (pre-trained) API ChromaDB and new SDK ChromaDB
    without requiring re-embedding.
    """

    def __init__(
        self,
        config: dict[str, Any],
        api_db_path: str,
        sdk_db_path: str,
        chromadb_api_dir: str,
        chromadb_code_dir: str,
    ):
        import logging
        self._log = logging.getLogger("revit_api_docs.retriever")

        from .embedder.providers import create_embedding
        try:
            self._embedder = create_embedding(config)
        except (KeyError, ValueError, ImportError, NotImplementedError) as e:
            # No key / unknown provider / provider not implemented (local_hf,
            # zhipu) / SDK not installed: keyword search only.
            self._log.warning(f"embedding disabled ({e}); vector search is off, "
                              "keyword search on SQLite only")
            self._embedder = None
        self._config = config

        # Retrieval config
        ret_cfg = config.get("retrieval", {})
        self._api_top_k = ret_cfg.get("api", {}).get("top_k", 30)
        self._api_rerank_n = ret_cfg.get("api", {}).get("rerank_top_n", 15)
        self._code_top_k = ret_cfg.get("code", {}).get("top_k", 5)
        self._code_rerank_n = ret_cfg.get("code", {}).get("rerank_top_n", 3)

        # ChromaDB collections are opened on the first vector query (see
        # _open_collections); without an embedder they are never touched.
        self._chromadb_api_dir = chromadb_api_dir
        self._chromadb_code_dir = chromadb_code_dir
        self._api_collection = None
        self._code_collection = None
        self._collections_lock = threading.Lock()
        if self._embedder is not None:
            for label, d in [("api", chromadb_api_dir), ("code", chromadb_code_dir)]:
                if not os.path.isdir(d):
                    self._log.error(f"ChromaDB {label} dir missing: {d}")
                elif not os.path.exists(os.path.join(d, "chroma.sqlite3")):
                    self._log.error(f"ChromaDB {label} dir has no chroma.sqlite3: {d}")

        # SQLite paths (opened per-query to avoid threading issues)
        self._api_db = api_db_path
        self._sdk_db = sdk_db_path

        # Consistency check: record_count in the ChromaDB meta.json should equal
        # the SQLite revit_api row count; otherwise the two stores are out of sync.
        # Logged at INFO: the shipped v1.0-data has this property (28863 vectors,
        # 27596 rows) and the extra vectors are dropped at query time.
        try:
            meta_path = os.path.join(chromadb_api_dir, "meta.json")
            if os.path.exists(meta_path):
                with open(meta_path, "r", encoding="utf-8") as f:
                    meta_count = json.load(f).get("record_count")
                conn = sqlite3.connect(api_db_path)
                sqlite_count = conn.execute("SELECT COUNT(*) FROM revit_api").fetchone()[0]
                conn.close()
                if meta_count is not None and meta_count != sqlite_count:
                    self._log.info(
                        "index/content mismatch: ChromaDB meta.record_count=%s but SQLite "
                        "revit_api has %s rows; vectors without a row are dropped at query time "
                        "(rebuild the ChromaDB index after parse_chm rebuilds SQLite)",
                        meta_count, sqlite_count,
                    )
        except Exception as e:
            self._log.warning(f"index/content consistency check failed: {e}")

        # Detect SDK schema
        self._sdk_new_schema = self._detect_sdk_schema()

        # Vector search is switched off for the process after this many
        # consecutive embedding/index failures (expired key, unreachable API).
        self._embed_failures = 0
        self._max_embed_failures = 3

        # Query rewriting LLM (lazy init; stays None after a failed init)
        self._rewrite_client = None
        self._rewrite_unavailable = False

        # Reranker (lazy init from config)
        self._reranker = None
        self._reranker_initialized = False

    def _open_collections(self) -> None:
        """Open both ChromaDB collections (slow import, done once, lazily).

        Tool calls run in worker threads, so two first queries can race here;
        the lock makes the second one wait instead of seeing one collection
        set and the other still None. Both are published together, after the
        second open succeeded.
        """
        if self._code_collection is not None:
            return
        with self._collections_lock:
            if self._code_collection is not None:
                return
            import chromadb
            from chromadb.config import Settings

            settings = Settings(anonymized_telemetry=False)
            api = (
                chromadb.PersistentClient(path=self._chromadb_api_dir, settings=settings)
                .get_collection("revit_api")
            )
            code = (
                chromadb.PersistentClient(path=self._chromadb_code_dir, settings=settings)
                .get_collection("revit_sdk")
            )
            self._api_collection, self._code_collection = api, code

    @property
    def vector_search_enabled(self) -> bool:
        return self._embedder is not None

    def _detect_sdk_schema(self) -> bool:
        """Check if sdk_db uses the new sdk_info table."""
        if not os.path.exists(self._sdk_db):
            raise FileNotFoundError(f"SDK DB not found: {self._sdk_db}")
        conn = sqlite3.connect(self._sdk_db)
        try:
            conn.execute("SELECT id FROM sdk_info LIMIT 1")
            return True
        except sqlite3.OperationalError:
            # table missing / schema mismatch -> legacy schema
            return False
        finally:
            conn.close()

    def _get_reranker(self):
        """Lazy-init reranker from config."""
        if not self._reranker_initialized:
            self._reranker_initialized = True
            try:
                from .reranker import create_reranker
                self._reranker = create_reranker(self._config)
            except Exception as e:
                self._log.warning(f"[_get_reranker] failed to init: {e}")
                self._reranker = None
        return self._reranker

    # ------------------------------------------------------------------
    # Query rewriting
    # ------------------------------------------------------------------

    _REWRITE_PROMPT = load_prompt("pipeline.rewrite_query.md")
    _REWRITE_SYSTEM_PROMPT = load_prompt("pipeline.rewrite_query_system.md")

    def _get_rewrite_client(self):
        """Lazy-init the query rewriting LLM client (None when no LLM is configured)."""
        if self._rewrite_client is None and not self._rewrite_unavailable:
            from .llm_client import create_llm_client
            try:
                self._rewrite_client = create_llm_client(
                    self._config, provider_override="gemini_flash"
                )
                self._rewrite_client.max_tokens = 256
                self._rewrite_client.temperature = 0.1
            except Exception as e:
                self._rewrite_unavailable = True
                self._log.info(f"query rewriting disabled ({e})")
        return self._rewrite_client

    def rewrite_query(self, query: str) -> str:
        """
        Use LLM to extract Revit API keywords from user query.
        Returns enriched query string for embedding.
        Falls back to original query on any error.
        """
        client = self._get_rewrite_client()
        if client is None:
            return query
        try:
            raw = client.generate_text(
                self._REWRITE_PROMPT.format(query=query),
                system_prompt=self._REWRITE_SYSTEM_PROMPT,
            )
            raw = re.sub(r"^```(?:json)?\s*", "", raw.strip(), flags=re.IGNORECASE)
            raw = re.sub(r"\s*```$", "", raw.strip())
            result = json.loads(raw)
            keywords = result.get("keywords", "")
            api_terms = result.get("api_terms", [])
            # Combine: original query + extracted keywords + API terms
            enriched = f"{query} {keywords} {' '.join(api_terms)}"
            return enriched.strip()
        except Exception:
            return query

    # ------------------------------------------------------------------
    # Search
    # ------------------------------------------------------------------

    def search(
        self,
        query: str,
        api_top_k: int | None = None,
        code_top_k: int | None = None,
        rewrite: bool = True,
        rerank: bool | None = None,
    ) -> SearchResults:
        """
        Tier 0: (Optional) LLM query rewriting for better keyword alignment.
        Tier 1: Embed query → search both ChromaDB collections (broad recall).
        Tier 2: Fetch full records from SQLite by matched IDs.
        Tier 3: (Optional) Rerank results for precision using cross-encoder.

        `rerank=None` follows `rewrite` (the historical fast/full switch); pass
        `rerank=True` with `rewrite=False` to rerank without an LLM rewrite.
        Pass 0 as a top_k to skip that source entirely.
        """
        import time as _time
        t0 = _time.monotonic()

        # Use config defaults if not specified (0 means "none of this source")
        final_api_n = self._api_rerank_n if api_top_k is None else api_top_k
        final_code_n = self._code_rerank_n if code_top_k is None else code_top_k

        # In fast mode: retrieve exactly final_api_n, skip rewrite + rerank
        # In full mode: retrieve broad (top_k), then rerank down to final_api_n
        # A source whose final count is 0 is not queried at all.
        use_rerank = rewrite if rerank is None else rerank
        retrieve_api_k = (self._api_top_k if use_rerank else final_api_n) if final_api_n > 0 else 0
        retrieve_code_k = (self._code_top_k if use_rerank else final_code_n) if final_code_n > 0 else 0

        # Tier 0: Query rewriting
        search_query = self.rewrite_query(query) if rewrite else query
        t_rewrite = _time.monotonic() - t0

        # Tier 1: Embedding + Vector search (skipped without an embedding key;
        # a failing embedding API degrades to keyword search instead of erroring)
        api_items: list[RetrievedItem] = []
        sdk_items: list[RetrievedItem] = []
        t_embed = 0.0
        vector_ok = False
        if self._embedder is not None:
            try:
                self._open_collections()
                query_embedding = self._embedder.embed_query(search_query)
                t_embed = _time.monotonic() - t0 - t_rewrite

                if retrieve_api_k > 0:
                    api_raw = self._api_collection.query(
                        query_embeddings=[query_embedding],
                        n_results=retrieve_api_k,
                    )
                    api_items = self._hydrate_api(api_raw)
                if retrieve_code_k > 0:
                    code_raw = self._code_collection.query(
                        query_embeddings=[query_embedding],
                        n_results=retrieve_code_k,
                    )
                    sdk_items = self._hydrate_sdk(code_raw)
                vector_ok = True
                self._embed_failures = 0
            except Exception as e:
                self._embed_failures += 1
                self._log.warning(f"vector search failed ({type(e).__name__}: {str(e)[:200]}); "
                                  "using keyword search for this query")
                if self._embed_failures >= self._max_embed_failures:
                    self._log.warning("vector search disabled for this process after "
                                      f"{self._embed_failures} consecutive failures")
                    self._embedder = None
                api_items, sdk_items = [], []
        if not vector_ok and final_code_n > 0:
            sdk_items = self._keyword_search_sdk(query, limit=final_code_n)

        # Tier 2: Hydrate from SQLite (done above per source)
        t_hydrate = _time.monotonic() - t0 - t_rewrite - t_embed

        # Tier 2.5: Keyword search on SQLite — merge with vector results
        # This catches exact name/API matches that embedding search misses
        kw_items = self._keyword_search_api(query, limit=final_api_n) if final_api_n > 0 else []
        if kw_items:
            existing_ids = {item.chromadb_id for item in api_items}
            merged_count = 0
            for kw_item in kw_items:
                if kw_item.chromadb_id not in existing_ids:
                    api_items.append(kw_item)
                    existing_ids.add(kw_item.chromadb_id)
                    merged_count += 1
                else:
                    # Keyword match is also in vector results — boost its ranking
                    for ai in api_items:
                        if ai.chromadb_id == kw_item.chromadb_id:
                            ai.distance = min(ai.distance, kw_item.distance)
                            break
            if merged_count:
                self._log.info(f"[search] merged {merged_count} keyword results")
            # Re-sort by distance after merging
            api_items.sort(key=lambda x: x.distance)

        # Tier 3: Rerank (only in full mode)
        t_rerank = 0.0
        reranker = self._get_reranker() if use_rerank else None
        if reranker and api_items:
            t_rr_start = _time.monotonic()
            # Build document texts for reranking
            api_docs = []
            for item in api_items:
                doc = f"{item.full_id or item.name}: {item.summary}"
                if item.syntax:
                    doc += f" | {item.syntax}"
                api_docs.append(doc)

            reranked = reranker.rerank(query, api_docs, top_n=final_api_n)
            # Reorder api_items by rerank scores
            reranked_items = []
            for rr in reranked:
                if rr.index < len(api_items):
                    item = api_items[rr.index]
                    item.distance = 1.0 - rr.score  # convert score to distance
                    reranked_items.append(item)
            api_items = reranked_items
            t_rerank = _time.monotonic() - t_rr_start
            self._log.info(f"[search] reranked {len(api_docs)} → {len(api_items)} "
                           f"in {t_rerank:.2f}s")
        else:
            # No reranker — just trim to final count
            api_items = api_items[:final_api_n]

        sdk_items = sdk_items[:final_code_n]

        elapsed = _time.monotonic() - t0
        self._log.info(
            f"[search] query={query[:40]!r} rewrite={t_rewrite:.2f}s "
            f"embed={t_embed:.2f}s hydrate={t_hydrate:.2f}s "
            f"rerank={t_rerank:.2f}s total={elapsed:.2f}s "
            f"api={len(api_items)} sdk={len(sdk_items)}"
        )

        return SearchResults(
            query=query, rewritten_query=search_query,
            api_items=api_items, sdk_items=sdk_items,
        )

    # ------------------------------------------------------------------
    # Hydrate from SQLite
    # ------------------------------------------------------------------

    def _hydrate_api(self, raw: dict) -> list[RetrievedItem]:
        """Fetch full API records from SQLite by ChromaDB IDs."""
        ids = raw["ids"][0] if raw["ids"] else []
        distances = raw["distances"][0] if raw["distances"] else []
        if not ids:
            return []

        conn = sqlite3.connect(self._api_db)
        conn.row_factory = sqlite3.Row
        placeholders = ",".join("?" * len(ids))
        rows = conn.execute(
            f"SELECT id, name, full_id, summary, info, syntax, parameters, remark "
            f"FROM revit_api WHERE id IN ({placeholders})",
            [int(i) for i in ids],
        ).fetchall()
        conn.close()

        row_map = {str(r["id"]): r for r in rows}

        items = []
        for cid, dist in zip(ids, distances):
            r = row_map.get(cid)
            if r is None:
                continue
            items.append(RetrievedItem(
                source="api",
                chromadb_id=cid,
                distance=dist,
                name=r["name"] or "",
                full_id=r["full_id"] or "",
                summary=r["summary"] or "",
                info=r["info"] or "",
                syntax=r["syntax"] or "",
                parameters=r["parameters"] or "",
                remark=r["remark"] or "",
            ))
        return items

    def _hydrate_sdk(self, raw: dict) -> list[RetrievedItem]:
        """Fetch full SDK records from SQLite by ChromaDB IDs."""
        ids = raw["ids"][0] if raw["ids"] else []
        distances = raw["distances"][0] if raw["distances"] else []
        if not ids:
            return []

        conn = sqlite3.connect(self._sdk_db)
        conn.row_factory = sqlite3.Row

        placeholders = ",".join("?" * len(ids))
        int_ids = [int(i) for i in ids]

        if self._sdk_new_schema:
            rows = conn.execute(
                f"SELECT id, project, summary, content, mentioned_apis "
                f"FROM sdk_info WHERE id IN ({placeholders})",
                int_ids,
            ).fetchall()
            row_map = {str(r["id"]): r for r in rows}
        else:
            rows = conn.execute(
                f"SELECT id, project, filename, code, clean_code, description "
                f"FROM revit_sdk WHERE id IN ({placeholders})",
                int_ids,
            ).fetchall()
            row_map = {str(r["id"]): r for r in rows}

        conn.close()

        items = []
        for cid, dist in zip(ids, distances):
            r = row_map.get(cid)
            if r is None:
                continue
            if self._sdk_new_schema:
                items.append(RetrievedItem(
                    source="sdk",
                    chromadb_id=cid,
                    distance=dist,
                    project=r["project"] or "",
                    summary=r["summary"] or "",
                    content=r["content"] or "",
                    mentioned_apis=r["mentioned_apis"] or "",
                ))
            else:
                code = r["clean_code"] if r["clean_code"] else (r["code"] or "")
                items.append(RetrievedItem(
                    source="sdk",
                    chromadb_id=cid,
                    distance=dist,
                    project=r["project"] or "",
                    name=r["filename"] or "",
                    summary=r["description"] or "",
                    content=code,
                ))
        return items

    # ------------------------------------------------------------------
    # Keyword normalization
    # ------------------------------------------------------------------

    # Common stop words that add noise to Revit API keyword search
    _STOP_WORDS = frozenset({
        "i", "me", "my", "we", "our", "you", "your", "he", "she", "it",
        "the", "a", "an", "is", "am", "are", "was", "were", "be", "been",
        "do", "does", "did", "have", "has", "had", "will", "would", "can",
        "could", "should", "shall", "may", "might", "must",
        "want", "need", "like", "know", "think", "use", "using", "used",
        "get", "got", "getting", "let", "make", "making",
        "to", "of", "in", "for", "on", "at", "by", "with", "from",
        "about", "into", "through", "after", "before", "between",
        "and", "or", "but", "not", "no", "if", "then", "so", "than",
        "how", "what", "which", "who", "where", "when", "why",
        "all", "any", "some", "this", "that", "these", "those",
        "api", "apis", "revit", "method", "function", "class",
        "please", "help", "show", "find", "look", "looking", "search",
    })

    # Verb → base form mapping for common Revit-relevant verbs
    _VERB_STEMS = {
        "creates": "create", "created": "create", "creating": "create",
        "deletes": "delete", "deleted": "delete", "deleting": "delete",
        "moves": "move", "moved": "move", "moving": "move",
        "copies": "copy", "copied": "copy", "copying": "copy",
        "modifies": "modify", "modified": "modify", "modifying": "modify",
        "changes": "change", "changed": "change", "changing": "change",
        "sets": "set", "setting": "set",
        "gets": "get", "getting": "get",  # "get" itself is a stop word but stem kept for compounds
        "adds": "add", "added": "add", "adding": "add",
        "removes": "remove", "removed": "remove", "removing": "remove",
        "updates": "update", "updated": "update", "updating": "update",
        "opens": "open", "opened": "open", "opening": "open",
        "closes": "close", "closed": "close", "closing": "close",
        "selects": "select", "selected": "select", "selecting": "select",
        "filters": "filter", "filtered": "filter", "filtering": "filter",
        "places": "place", "placed": "place", "placing": "place",
        "splits": "split", "splitting": "split",
        "joins": "join", "joined": "join", "joining": "join",
        "exports": "export", "exported": "export", "exporting": "export",
        "imports": "import", "imported": "import", "importing": "import",
        "rotates": "rotate", "rotated": "rotate", "rotating": "rotate",
        "mirrors": "mirror", "mirrored": "mirror", "mirroring": "mirror",
    }

    # Tokens that are likely Revit entity nouns (boosted in scoring)
    _ENTITY_NOUNS = frozenset({
        "wall", "floor", "ceiling", "roof", "door", "window", "room",
        "column", "beam", "pipe", "duct", "family", "type", "level",
        "grid", "view", "sheet", "schedule", "parameter", "element",
        "group", "assembly", "material", "phase", "workset", "link",
        "rebar", "stair", "railing", "ramp", "curtain", "panel",
        "mullion", "opening", "dimension", "tag", "annotation",
        "detail", "section", "elevation", "plan", "legend",
        "connector", "fitting", "system", "zone", "space", "area",
        "part", "instance", "symbol", "category", "filter",
        "transaction", "collector", "reference", "curve", "line",
        "arc", "point", "solid", "face", "edge", "geometry",
    })

    def _normalize_search_tokens(self, query: str) -> list[tuple[str, float]]:
        """Extract and weight search tokens from a natural-language query.

        Returns list of (token, weight) tuples where:
        - Entity nouns (wall, floor, etc.) get weight 2.0
        - Stemmed action verbs (create, delete) get weight 1.0
        - Other surviving tokens get weight 0.5
        Stop words are removed, verbs are stemmed to base form.
        """
        raw_tokens = [t.strip().lower() for t in query.split() if len(t.strip()) >= 2]
        weighted: list[tuple[str, float]] = []
        seen = set()

        for token in raw_tokens:
            # Stem verbs first
            stemmed = self._VERB_STEMS.get(token, token)

            # Skip stop words only when BOTH the original token and its stem are
            # stop words (so verb forms like "gets"→"get" that are also action
            # stems can survive if the stem is meaningful elsewhere).
            if token in self._STOP_WORDS and stemmed in self._STOP_WORDS:
                continue

            if stemmed in seen:
                continue
            seen.add(stemmed)

            if stemmed in self._ENTITY_NOUNS:
                weighted.append((stemmed, 2.0))
            elif token in self._VERB_STEMS:
                # It was a verb form → stemmed action verb
                weighted.append((stemmed, 1.0))
            else:
                weighted.append((stemmed, 0.5))

        # Sort by weight descending so most important tokens come first
        weighted.sort(key=lambda x: x[1], reverse=True)
        return weighted

    # ------------------------------------------------------------------
    # Keyword search (SQLite direct)
    # ------------------------------------------------------------------

    # Candidates handed from SQL to the Python scorer. The cap is applied after
    # SQL has ranked every matching row, never to an arbitrary rowid subset.
    _KEYWORD_CANDIDATES = 300

    def _keyword_candidates(self, weighted_tokens: list[tuple[str, float]]) -> list[sqlite3.Row]:
        """All rows matching any token, ordered by a coarse SQL relevance score.

        Per token: name hit 5*w, full_id hit 1.5*w, summary hit 0.5*w (the same
        weights the Python scorer starts from), so rows with several tokens in
        the name come first and the LIMIT trims the long tail of summary-only
        hits instead of dropping exact name matches.
        """
        score_terms: list[str] = []
        params: list[str] = []
        for token, w in weighted_tokens:
            like = f"%{_escape_like(token)}%"
            score_terms.append(
                f"(LOWER(name) LIKE ? ESCAPE '\\') * {5.0 * w:.2f}"
                f" + (LOWER(full_id) LIKE ? ESCAPE '\\') * {1.5 * w:.2f}"
                f" + (LOWER(summary) LIKE ? ESCAPE '\\') * {0.5 * w:.2f}"
            )
            params.extend([like, like, like])
        sql = (
            "SELECT id, name, full_id, summary, info, syntax, parameters, remark, sql_score FROM ("
            "  SELECT id, name, full_id, summary, info, syntax, parameters, remark, "
            f"    ({' + '.join(score_terms)}) AS sql_score FROM revit_api"
            ") WHERE sql_score > 0 "
            "ORDER BY sql_score DESC, LENGTH(name) ASC, id ASC "
            f"LIMIT {self._KEYWORD_CANDIDATES}"
        )
        conn = sqlite3.connect(self._api_db)
        conn.row_factory = sqlite3.Row
        try:
            return conn.execute(sql, params).fetchall()
        finally:
            conn.close()

    def _keyword_search_api(self, query: str, limit: int = 15) -> list[RetrievedItem]:
        """Search API docs by keyword matching on name/full_id/summary.

        1. SQL ranks every row that matches any token by a coarse relevance
           score and hands over the top _KEYWORD_CANDIDATES (see
           _keyword_candidates).
        2. Python scoring ranks those by match quality (name > full_id >
           summary, segment and exact-identifier bonuses) and penalizes long
           compound names.
        Token weights from _normalize_search_tokens boost entity nouns over verbs.
        """
        weighted_tokens = self._normalize_search_tokens(query)
        if not weighted_tokens:
            return []

        tokens = [t for t, _ in weighted_tokens]
        token_weights = {t: w for t, w in weighted_tokens}

        all_rows = self._keyword_candidates(weighted_tokens)

        # Score each row in Python (token weights boost entity nouns)
        scored: list[tuple[float, dict]] = []
        for r in all_rows:
            name_l = (r["name"] or "").lower()
            fid_l = (r["full_id"] or "").lower()
            summary_l = (r["summary"] or "").lower()
            score = 0.0

            for token in tokens:
                tw = token_weights.get(token, 0.5)  # token importance weight
                # Name match is most valuable
                if token in name_l:
                    score += 5.0 * tw
                    # Bonus: token starts a name segment (Wall in "Wall.Create")
                    name_parts = re.split(r'[.\s(]', name_l)
                    if any(part == token or part.startswith(token) for part in name_parts):
                        score += 3.0 * tw
                    # Dotted tokens ("wall.create") never equal a segment; compare
                    # them with the identifier before the first space or "(" so
                    # Wall.Create and its overloads outrank Wall.CreateProfileSketch.
                    if "." in token:
                        head = re.split(r"[\s(]", name_l, maxsplit=1)[0]
                        if head == token:
                            score += 6.0 * tw
                        elif head.startswith(token):
                            score += 2.0 * tw
                # full_id match (less weight since full_id often contains namespace)
                elif token in fid_l:
                    score += 1.5 * tw
                # summary match
                if token in summary_l:
                    score += 0.5 * tw

            if score <= 0:
                continue

            # Penalize long/deeply-nested names
            name_len = len(r["name"] or "")
            dot_count = (r["name"] or "").count(".")
            if name_len > 80:
                score *= 0.3
            elif name_len > 60:
                score *= 0.5
            elif name_len > 40:
                score *= 0.7
            if dot_count >= 3:
                score *= 0.5
            # BuiltInFailures/BuiltInParameter are rarely the target API
            if name_l.startswith("builtinfailures."):
                score *= 0.3
            elif name_l.startswith("builtinparameter."):
                score *= 0.6

            # Strong bonus for ALL tokens matching in name
            name_matched = sum(1 for t in tokens if t in name_l)
            if name_matched == len(tokens):
                score *= 2.0  # all tokens in name — very relevant

            scored.append((score, dict(r)))

        scored.sort(key=lambda x: x[0], reverse=True)

        items = []
        for i, (sc, r) in enumerate(scored[:limit]):
            items.append(RetrievedItem(
                source="api",
                chromadb_id=str(r["id"]),
                distance=0.1 + i * 0.01,  # low distance = high relevance
                name=r["name"] or "",
                full_id=r["full_id"] or "",
                summary=r["summary"] or "",
                info=r["info"] or "",
                syntax=r["syntax"] or "",
                parameters=r["parameters"] or "",
                remark=r["remark"] or "",
            ))
        self._log.info(f"[keyword_search] query={query!r} "
                       f"tokens={[(t, f'{w:.1f}') for t, w in weighted_tokens]} "
                       f"found={len(items)} "
                       f"top3={[(s[1]['name'][:50], f'{s[0]:.1f}') for s in scored[:3]]}")
        return items

    def _keyword_search_sdk(self, query: str, limit: int = 3) -> list[RetrievedItem]:
        """Keyword fallback for SDK samples when no embedding key is configured.

        Matches tokens against project name, summary and mentioned_apis; a hit on
        the project name or the API list outranks one in the prose summary.
        Only the current sdk_info schema is supported (the legacy table has no
        API list to match against).
        """
        if not self._sdk_new_schema:
            return []
        weighted_tokens = self._normalize_search_tokens(query)
        if not weighted_tokens:
            return []
        tokens = [t for t, _ in weighted_tokens]
        token_weights = {t: w for t, w in weighted_tokens}

        conditions = []
        params: list[str] = []
        for token in tokens:
            like = f"%{_escape_like(token)}%"
            conditions.append(
                "(LOWER(project) LIKE ? ESCAPE '\\' OR LOWER(summary) LIKE ? ESCAPE '\\' "
                "OR LOWER(mentioned_apis) LIKE ? ESCAPE '\\')"
            )
            params.extend([like, like, like])
        conn = sqlite3.connect(self._sdk_db)
        conn.row_factory = sqlite3.Row
        rows = conn.execute(
            f"SELECT id, project, summary, content, mentioned_apis FROM sdk_info "
            f"WHERE {' OR '.join(conditions)} LIMIT 200",
            params,
        ).fetchall()
        conn.close()

        scored: list[tuple[float, sqlite3.Row]] = []
        for r in rows:
            project_l = (r["project"] or "").lower()
            summary_l = (r["summary"] or "").lower()
            apis_l = (r["mentioned_apis"] or "").lower()
            score = 0.0
            for token in tokens:
                tw = token_weights.get(token, 0.5)
                if token in project_l:
                    score += 5.0 * tw
                if token in apis_l:
                    score += 3.0 * tw
                if token in summary_l:
                    score += 1.0 * tw
            if score > 0:
                scored.append((score, r))
        scored.sort(key=lambda x: x[0], reverse=True)

        items = []
        for i, (_, r) in enumerate(scored[:limit]):
            items.append(RetrievedItem(
                source="sdk",
                chromadb_id=str(r["id"]),
                distance=0.1 + i * 0.01,
                project=r["project"] or "",
                summary=r["summary"] or "",
                content=r["content"] or "",
                mentioned_apis=r["mentioned_apis"] or "",
            ))
        self._log.info(f"[keyword_search_sdk] query={query!r} found={len(items)}")
        return items

    # ------------------------------------------------------------------
    # Context assembly
    # ------------------------------------------------------------------

    def build_context(
        self,
        results: SearchResults,
        api_max_chars: int = 600,
        code_max_chars: int = 2000,
    ) -> dict[str, str]:
        """
        Assemble structured context strings for LLM prompt injection.

        Returns:
            {"api_context": str, "code_context": str}
        """
        api_parts = []
        for item in results.api_items:
            header = item.full_id or item.name
            body_parts = []
            if item.summary:
                body_parts.append(item.summary)
            if item.syntax:
                body_parts.append(f"Syntax: {item.syntax}")
            if item.parameters:
                body_parts.append(f"Parameters: {item.parameters}")
            if item.remark:
                body_parts.append(f"Remarks: {item.remark}")
            if not body_parts and item.info:
                body_parts.append(item.info)
            body = "\n".join(body_parts)[:api_max_chars]
            api_parts.append(f"### {header}\n{body}")

        code_parts = []
        for item in results.sdk_items:
            header = f"Project: {item.project}"
            if item.mentioned_apis:
                header += f"  |  APIs: {item.mentioned_apis}"
            body = item.content[:code_max_chars] if item.content else item.summary
            code_parts.append(f"// {header}\n{body}")

        return {
            "api_context": "\n\n".join(api_parts),
            "code_context": "\n\n".join(code_parts),
        }

    # ------------------------------------------------------------------
    # Pretty-print
    # ------------------------------------------------------------------

    def format_results(self, results: SearchResults) -> str:
        """Format search results for display."""
        lines = []
        if results.rewritten_query and results.rewritten_query != results.query:
            lines.append(f"query:     {results.query}")
            lines.append(f"rewritten: {results.rewritten_query}")
            lines.append("")
        lines.append("=" * 60)
        lines.append("API results:")
        lines.append("=" * 60)
        for i, item in enumerate(results.api_items, 1):
            sim = 1 - item.distance
            lines.append(f"\n[{i}] similarity: {sim:.4f}")
            lines.append(f"    name:    {item.full_id or item.name}")
            lines.append(f"    summary: {item.summary[:150]}...")

        lines.append("\n" + "=" * 60)
        lines.append("SDK code results:")
        lines.append("=" * 60)
        for i, item in enumerate(results.sdk_items, 1):
            sim = 1 - item.distance
            lines.append(f"\n[{i}] similarity: {sim:.4f}")
            lines.append(f"    project: {item.project}")
            lines.append(f"    summary: {item.summary[:150]}...")
            if item.content:
                preview = item.content[:200].replace("\n", "\n    ")
                lines.append(f"    code:    {preview}...")

        return "\n".join(lines)
