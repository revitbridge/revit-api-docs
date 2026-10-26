"""
Two-tier retriever: ChromaDB (semantic index) + SQLite (content store).

Architecture:
    ChromaDB holds lightweight embeddings (summary-based); each record's
    metadata names its entry (`name` for the API reference, `project` for the
    SDK samples). SQLite holds the full content (code, API docs, parameters).
    At query time:
        0. (Optional) LLM query rewriting: extract Revit API keywords from the query
        1. Embed the (rewritten) query and search ChromaDB -> ranked records
        2. Match each record to its SQLite row by entry name, fetch the rows
        3. BM25 keyword search on SQLite, fused with the vector ranking by
           reciprocal rank (RRF)
        4. (Optional) rerank the fused candidates; assemble the context

Public API:
    retriever = RAGRetriever(config, api_db, sdk_db, chromadb_api_dir, chromadb_code_dir)
    results   = retriever.search(query, api_top_k=15, code_top_k=5)
    context   = retriever.build_context(results)

Vector records and SQLite rows: a record id need not be the id of the row the
record was built from. In the v1.0-data release it is not: its 28,863 API
records are numbered independently of the 27,596 rows of revit_api. A vector
hit is therefore matched to the row whose entry name equals the record's
metadata: the row with the record's own id when that row carries the name,
otherwise the row of that name; among rows sharing a name, the one whose
full_id or info equals the record's, else the lowest id. Records whose name is
not in SQLite are dropped. The name map is built on the first vector query and
kept in memory for the process (never written to disk).

Consistency probe: on the first vector query, 50 records of each collection
are sampled and compared with SQLite. When fewer than 90% of the sampled ids
name the same entry as the row with that id, the warning
`vector_store_misaligned` is logged and hits are matched by name as above (the
shipped data set is in this state). When fewer than 50% of the sampled names
are found in SQLite at all, the same warning says that vector search is off,
and it stays off for the process (vector_search_enabled becomes False; keyword
search only). The result is cached for the process; check_vector_stores()
runs the probe without an embedding key (`revit-api-docs download` prints it).

Hybrid ranking: the vector hits and the BM25 hits, each to the depth
retrieval.api.top_k, are fused by reciprocal rank, score = sum of
1 / (RRF_K + rank) with RRF_K = 60. The best top_k fused entries go to the
reranker; without one they are cut to the requested count. Without vector
search the BM25 order is kept as it is.

Degraded modes (no network keys needed for the MCP server to be useful):
    - No embedding API key  -> vector search is skipped, keyword search only
                               (BM25 over an SQLite FTS5 index, see bm25.py;
                               the LIKE scorer when FTS5 is missing).
    - Vector store fails the consistency probe -> keyword search only.
    - No rerank API key     -> results keep the fused (or keyword) order.
    - Jev call fails        -> Cohere reranks when its key is set, else no rerank;
                               SearchResults.rerank_used says which one ran.
    - No LLM key            -> query rewriting is skipped.

Abstention: with rerank provider "jev" and retrieval.api.min_relevance set,
the result is marked abstained when no candidate reaches the threshold (the
candidates stay for reference). The threshold decides only that: whether or
not the result abstains, every reranked candidate comes back (up to the
requested count) with its relevance, below the threshold or not. A low best
score says the data set has nothing relevant to the query, or words it
differently; it does not show that a member does not exist. Other providers
never abstain: their scores are not calibrated probabilities.
"""
from __future__ import annotations

import json
import logging
import os
import re
import sqlite3
import threading
from dataclasses import dataclass, field
from typing import Any

from .bm25 import Bm25Index, fts5_available
from .prompts import load_prompt


def _escape_like(s: str) -> str:
    """Escape LIKE wildcards % and _ (used with ESCAPE '\\'); the backslash itself too."""
    return s.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")


_log = logging.getLogger("revit_api_docs.retriever")

# Reciprocal rank fusion: score = sum over the rankings of 1 / (RRF_K + rank), rank from 1.
RRF_K = 60

# Consistency probe of the vector collections (see the module docstring).
PROBE_SAMPLE = 50     # records sampled per collection
PROBE_BLOCKS = 10     # read as 10 runs of consecutive records spread over the collection
PROBE_ALIGNED = 0.9   # below this share of sampled ids naming their row's entry: misaligned
PROBE_USABLE = 0.5    # below this share of sampled names found in SQLite: unusable


@dataclass
class VectorProbe:
    """Consistency probe result for one ChromaDB collection.

    status: "aligned" (record ids are row ids), "misaligned" (they are not,
    but the entry names are found: hits are matched to rows by name) or
    "unusable" (the names are not found either: vector search is off).
    """
    collection: str     # ChromaDB collection, e.g. "revit_api"
    table: str          # SQLite table its records describe
    key: str            # metadata field / column holding the entry name
    records: int        # records in the collection
    sampled: int
    id_matches: int     # sampled records whose id is a row carrying the same name
    name_matches: int   # sampled records whose name is found in the table
    status: str

    def describe(self) -> str:
        head = f"{self.collection} vectors ({self.records} records) vs SQLite {self.table}"
        if not self.sampled:
            return f"{head}: unusable, the collection is empty; vector search is off"
        ids = (f"{self.id_matches}/{self.sampled} sampled record ids name the same entry "
               "as the row with that id")
        names = f"{self.name_matches}/{self.sampled} sampled entry names ({self.key}) found"
        if self.status == "aligned":
            return f"{head}: aligned, {ids}"
        if self.status == "misaligned":
            return f"{head}: misaligned, {ids}; {names}, so vector hits are matched to rows by name"
        return f"{head}: unusable, {ids} and {names}; vector search is off"


@dataclass(frozen=True)
class _Source:
    """A vector collection and the SQLite table its records describe."""
    collection: str
    db_path: str
    table: str
    key: str                        # metadata field and column holding the entry name
    details: tuple[str, ...] = ()   # fields that tell rows sharing a name apart


def _api_source(api_db_path: str) -> _Source:
    return _Source("revit_api", api_db_path, "revit_api", "name", ("full_id", "info"))


def _sdk_source(sdk_db_path: str, new_schema: bool) -> _Source:
    return _Source("revit_sdk", sdk_db_path, "sdk_info" if new_schema else "revit_sdk", "project")


def _sdk_schema_is_new(sdk_db_path: str) -> bool:
    """True when the SDK database has the sdk_info table, False for the legacy revit_sdk."""
    if not os.path.exists(sdk_db_path):
        raise FileNotFoundError(f"SDK DB not found: {sdk_db_path}")
    conn = sqlite3.connect(sdk_db_path)
    try:
        conn.execute("SELECT id FROM sdk_info LIMIT 1")
        return True
    except sqlite3.OperationalError:
        # table missing / schema mismatch -> legacy schema
        return False
    finally:
        conn.close()


def _row_id(record_id: Any) -> int | None:
    try:
        return int(record_id)
    except (TypeError, ValueError):
        return None


class _RowIndex:
    """Entry name -> row ids of one SQLite table, to match vector records to rows."""

    def __init__(self, src: _Source):
        self.src = src
        self.name_by_id: dict[int, str] = {}
        self.ids_by_name: dict[str, list[int]] = {}
        self.details: dict[int, dict[str, str]] = {}   # only rows that share their name
        self._tie_logged = False
        conn = sqlite3.connect(src.db_path)
        try:
            for rid, name in conn.execute(f"SELECT id, {src.key} FROM {src.table} ORDER BY id"):
                name = name or ""
                self.name_by_id[rid] = name
                if name:
                    self.ids_by_name.setdefault(name, []).append(rid)
            shared = [rid for ids in self.ids_by_name.values() if len(ids) > 1 for rid in ids]
            if src.details and shared:
                cols = ", ".join(src.details)
                for i in range(0, len(shared), 500):
                    chunk = shared[i:i + 500]
                    rows = conn.execute(f"SELECT id, {cols} FROM {src.table} "
                                        f"WHERE id IN ({','.join('?' * len(chunk))})", chunk)
                    for row in rows:
                        self.details[row[0]] = {c: v or "" for c, v in zip(src.details, row[1:])}
        finally:
            conn.close()

    def id_agrees(self, record_id: Any, meta: dict | None) -> bool:
        """True when the row with the record's id carries the record's entry name."""
        name = (meta or {}).get(self.src.key)
        rid = _row_id(record_id)
        return isinstance(name, str) and bool(name) and self.name_by_id.get(rid) == name

    def resolve(self, record_id: Any, meta: dict | None, trust_ids: bool = False) -> int | None:
        """Row id for one vector record, or None when no row has its entry name.

        A record without a name in its metadata keeps its id only with
        `trust_ids` (the probe found the ids aligned); otherwise it is dropped.
        """
        meta = meta or {}
        name = meta.get(self.src.key)
        rid = _row_id(record_id)
        if not isinstance(name, str) or not name:
            return rid if trust_ids and rid in self.name_by_id else None
        if rid is not None and self.name_by_id.get(rid) == name:
            return rid
        ids = self.ids_by_name.get(name)
        if not ids:
            return None
        if len(ids) == 1:
            return ids[0]
        for col in self.src.details:
            want = meta.get(col)
            if isinstance(want, str) and want:
                same = [i for i in ids if self.details.get(i, {}).get(col) == want]
                if same:
                    return same[0]
        if not self._tie_logged:
            self._tie_logged = True
            _log.debug("vector record %s: %s %r names %d rows of %s and its metadata does not "
                       "tell them apart; using the lowest id %d (later cases are not logged)",
                       record_id, self.src.key, name, len(ids), self.src.table, ids[0])
        return ids[0]


# Built on the first vector query and kept for the process, never written to
# disk: the name map of each table and the probe result of each collection.
_ROW_INDEXES: dict[tuple, _RowIndex] = {}
_PROBES: dict[tuple, VectorProbe] = {}
_CACHE_LOCK = threading.Lock()


def _db_key(path: str) -> tuple:
    """A database file as a cache key: path, size and mtime (a replaced file is a new key)."""
    try:
        st = os.stat(path)
        return (os.path.realpath(path), st.st_size, st.st_mtime_ns)
    except OSError:
        return (os.path.realpath(path), None, None)


def _row_index(src: _Source) -> _RowIndex:
    key = (src, _db_key(src.db_path))
    with _CACHE_LOCK:
        index = _ROW_INDEXES.get(key)
        if index is None:
            index = _ROW_INDEXES[key] = _RowIndex(src)
    return index


def probe_vector_collection(collection: Any, index: _RowIndex) -> VectorProbe:
    """Sample up to PROBE_SAMPLE records of `collection` and compare their names with SQLite.

    The sample is PROBE_BLOCKS runs of consecutive records at evenly spaced
    offsets: deterministic, and 10 reads instead of 50 (an offset read costs
    about 25 ms on the shipped store).
    """
    total = int(collection.count())
    want = min(PROBE_SAMPLE, total)
    sample: dict[str, dict | None] = {}
    if want > 0:
        blocks = min(PROBE_BLOCKS, want)
        per = -(-want // blocks)
        for b in range(blocks):
            got = collection.get(limit=per, offset=b * total // blocks, include=["metadatas"])
            ids = got.get("ids") or []
            metas = got.get("metadatas") or []
            for i, cid in enumerate(ids):
                if len(sample) < want:
                    sample.setdefault(str(cid), metas[i] if i < len(metas) else None)
    n = len(sample)
    id_hits = sum(1 for cid, meta in sample.items() if index.id_agrees(cid, meta))
    name_hits = sum(1 for cid, meta in sample.items() if index.resolve(cid, meta) is not None)
    if n and id_hits >= PROBE_ALIGNED * n:
        status = "aligned"
    elif n and name_hits >= PROBE_USABLE * n:
        status = "misaligned"
    else:
        status = "unusable"
    src = index.src
    return VectorProbe(src.collection, src.table, src.key, total, n, id_hits, name_hits, status)


def _probe(src: _Source, collection: Any, chroma_dir: str) -> VectorProbe:
    """probe_vector_collection, once per collection and database in a process."""
    key = (src, _db_key(src.db_path), os.path.realpath(chroma_dir))
    with _CACHE_LOCK:
        probe = _PROBES.get(key)
    if probe is None:
        probe = probe_vector_collection(collection, _row_index(src))
        with _CACHE_LOCK:
            probe = _PROBES.setdefault(key, probe)
    return probe


def _open_collection(path: str, name: str) -> Any:
    """Open one ChromaDB collection.

    Each client gets its own Settings object: chromadb writes the client's
    path into the object it is given, so one object shared by two clients
    made a later client for the first path (a second retriever in the same
    process) fail with "An instance of Chroma already exists ... with
    different settings".
    """
    import chromadb
    from chromadb.config import Settings

    client = chromadb.PersistentClient(path=path, settings=Settings(anonymized_telemetry=False))
    return client.get_collection(name)


def check_vector_stores(api_db_path: str, sdk_db_path: str, chromadb_api_dir: str,
                        chromadb_code_dir: str) -> list[VectorProbe]:
    """Open both collections and run the consistency probe on each.

    Needs no embedding key; results are cached for the process. Used by
    `revit-api-docs download` to report the state of an installed data set.
    """
    sources = [(_api_source(api_db_path), chromadb_api_dir),
               (_sdk_source(sdk_db_path, _sdk_schema_is_new(sdk_db_path)), chromadb_code_dir)]
    return [_probe(src, _open_collection(d, src.collection), d) for src, d in sources]


@dataclass
class RetrievedItem:
    """A single retrieved item with full content from SQLite."""
    source: str               # "api" or "sdk"
    chromadb_id: str          # the SQLite row id (str); the field name is historical
    distance: float           # lower is better; orders candidates, not a calibrated score
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
    # Calibrated probability (0-1) from a calibrated rerank provider (jev);
    # None when the order came from any other source.
    relevance: float | None = None


@dataclass
class SearchResults:
    """Container for all retrieval results."""
    query: str
    rewritten_query: str = ""
    api_items: list[RetrievedItem] = field(default_factory=list)
    sdk_items: list[RetrievedItem] = field(default_factory=list)
    # Which provider ordered api_items: "jev" | "cohere" | "none".
    rerank_used: str = "none"
    # Providers tried before rerank_used that were unavailable or failed
    # (for example ["jev"] when Jev failed and Cohere took over).
    rerank_skipped: list[str] = field(default_factory=list)
    # True when Jev scored every candidate below retrieval.api.min_relevance.
    # The threshold only sets this flag: api_items keep every reranked
    # candidate, each with its relevance, whether or not it reaches it.
    abstained: bool = False
    abstain_reason: str = ""  # e.g. "best relevance 0.07 < 0.20"


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
        self._log = _log

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
        self._min_relevance = self._parse_min_relevance(ret_cfg.get("api", {}).get("min_relevance"))
        self._revit_version = str(config.get("revit_version", "2026"))

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

        # Keyword search over the API reference: BM25 on an FTS5 index kept in
        # its own file next to revit_api.db (built on first use), or the LIKE
        # scorer when this sqlite3 build has no FTS5. One of the two per process.
        self._bm25: Bm25Index | None = None
        if fts5_available():
            self._bm25 = Bm25Index(api_db_path)
        else:
            self._log.warning("SQLite FTS5 is not available in this Python build; keyword "
                              "search uses the slower LIKE scorer")

        # Size check: record_count in the ChromaDB meta.json against the SQLite
        # revit_api row count. Logged at INFO: the shipped v1.0-data differs
        # (28863 vectors, 27596 rows). Vector hits are matched to rows by entry
        # name, so the counts need not agree; the consistency probe on the
        # first vector query checks what matters (see the module docstring).
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
                        "revit_api has %s rows; vector hits are matched to rows by entry name "
                        "and records without a row are dropped",
                        meta_count, sqlite_count,
                    )
        except Exception as e:
            self._log.warning(f"index/content consistency check failed: {e}")

        # Detect SDK schema
        self._sdk_new_schema = self._detect_sdk_schema()

        # Vector records are matched to SQLite rows by entry name, checked by
        # the consistency probe on the first vector query (module docstring).
        self._sources = {
            "api": _api_source(api_db_path),
            "sdk": _sdk_source(sdk_db_path, self._sdk_new_schema),
        }
        self._probes: dict[str, VectorProbe] | None = None
        self._vector_off: str | None = None   # why the probe switched vector search off
        self._probe_lock = threading.Lock()

        # Vector search is switched off for the process after this many
        # consecutive embedding/index failures (expired key, unreachable API).
        self._embed_failures = 0
        self._max_embed_failures = 3

        # Query rewriting LLM (lazy init; stays None after a failed init)
        self._rewrite_client = None
        self._rewrite_unavailable = False

        # Rerank providers, tried in order (lazy init from config)
        self._rerank_chain: list = []
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
            api = _open_collection(self._chromadb_api_dir, "revit_api")
            code = _open_collection(self._chromadb_code_dir, "revit_sdk")
            self._api_collection, self._code_collection = api, code

    def _check_vector_stores(self) -> None:
        """Consistency probe of both collections; call after _open_collections().

        Each collection is probed once per process (cached). Logs the warning
        `vector_store_misaligned` when record ids are not row ids, and switches
        vector search off for the process when a collection's entry names are
        not found in SQLite either, or when the probe itself fails.
        """
        if self._probes is not None:
            return
        with self._probe_lock:
            if self._probes is not None:
                return
            probes: dict[str, VectorProbe] = {}
            try:
                for source, coll, chroma_dir in (
                        ("api", self._api_collection, self._chromadb_api_dir),
                        ("sdk", self._code_collection, self._chromadb_code_dir)):
                    probes[source] = _probe(self._sources[source], coll, chroma_dir)
            except Exception as e:
                self._vector_off = f"consistency probe failed ({type(e).__name__}: {str(e)[:200]})"
                self._log.warning(f"vector_store_misaligned: {self._vector_off}; vector search "
                                  "is off for this process, keyword search only")
            else:
                for p in probes.values():
                    if p.status == "aligned":
                        self._log.info(f"vector store check: {p.describe()}")
                    else:
                        self._log.warning(f"vector_store_misaligned: {p.describe()}")
                unusable = [p.describe() for p in probes.values() if p.status == "unusable"]
                if unusable:
                    self._vector_off = "; ".join(unusable)
            self._probes = probes

    @property
    def vector_search_enabled(self) -> bool:
        """True while queries go to the vector store: an embedding provider is
        configured and the consistency probe (run on the first vector query)
        has not switched vector search off."""
        return self._embedder is not None and self._vector_off is None

    @property
    def vector_probes(self) -> list[VectorProbe]:
        """Consistency probe results, once the first vector query has run the probe."""
        return list((self._probes or {}).values())

    @property
    def revit_version(self) -> str:
        """Revit release the data set was built from (config `revit_version`)."""
        return self._revit_version

    @property
    def min_relevance(self) -> float | None:
        return self._min_relevance

    def _parse_min_relevance(self, value: Any) -> float | None:
        """None / "none" / "" -> no threshold; otherwise a number in [0, 1]."""
        if value is None or (isinstance(value, str)
                             and value.strip().lower() in ("", "none", "null", "off")):
            return None
        try:
            threshold = float(value)
        except (TypeError, ValueError):
            self._log.warning(f"retrieval.api.min_relevance={value!r} is not a number; "
                              "abstention is off")
            return None
        if not 0.0 <= threshold <= 1.0:
            self._log.warning(f"retrieval.api.min_relevance={value!r} is outside 0-1; "
                              "abstention is off")
            return None
        return threshold

    def _detect_sdk_schema(self) -> bool:
        """Check if sdk_db uses the new sdk_info table."""
        return _sdk_schema_is_new(self._sdk_db)

    def _get_rerank_chain(self) -> list:
        """Lazy-init the rerank providers from config (empty list: no rerank)."""
        if not self._reranker_initialized:
            self._reranker_initialized = True
            try:
                from .reranker import configured_provider, create_rerank_chain
                self._rerank_chain = create_rerank_chain(self._config)
                configured = configured_provider(self._config)
                if self._min_relevance is not None and configured != "jev":
                    self._log.info("retrieval.api.min_relevance is ignored with rerank provider "
                                   f"{configured!r} (only jev returns calibrated scores)")
            except Exception as e:
                self._log.warning(f"[_get_rerank_chain] failed to init: {e}")
                self._rerank_chain = []
        return self._rerank_chain

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
        Tier 1: Embed query -> search both ChromaDB collections (broad recall).
        Tier 2: Match the records to SQLite rows by entry name, fetch the rows.
        Tier 2.5: BM25 keyword search, fused with the vector hits by RRF.
        Tier 3: (Optional) Rerank the fused candidates for precision.

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

        # Tier 1: Embedding + Vector search (skipped without an embedding key or
        # when the consistency probe switched it off; a failing embedding API
        # degrades to keyword search instead of erroring)
        api_items: list[RetrievedItem] = []
        vec_items: list[RetrievedItem] = []
        sdk_items: list[RetrievedItem] = []
        t_embed = 0.0
        vector_ok = False
        if self.vector_search_enabled:
            try:
                self._open_collections()
                self._check_vector_stores()   # first vector query only; may switch vectors off
                if self._vector_off is None:
                    query_embedding = self._embedder.embed_query(search_query)
                    t_embed = _time.monotonic() - t0 - t_rewrite

                    if retrieve_api_k > 0:
                        api_raw = self._api_collection.query(
                            query_embeddings=[query_embedding],
                            n_results=retrieve_api_k,
                            include=["metadatas", "distances"],
                        )
                        vec_items = self._hydrate_api(api_raw)
                    if retrieve_code_k > 0:
                        code_raw = self._code_collection.query(
                            query_embeddings=[query_embedding],
                            n_results=retrieve_code_k,
                            include=["metadatas", "distances"],
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
                vec_items, sdk_items = [], []
        if not vector_ok and final_code_n > 0:
            sdk_items = self._keyword_search_sdk(query, limit=final_code_n)

        # Tier 2: Hydrate from SQLite (done above per source)
        t_hydrate = _time.monotonic() - t0 - t_rewrite - t_embed

        # Tier 2.5: BM25 keyword search on SQLite, to the same depth as the
        # vector search (top_k when reranking, else the requested count). With
        # vector hits the two rankings are fused by reciprocal rank and the
        # best retrieve_api_k entries are the candidates; without them the
        # keyword order is used as it is. No bonus is added after fusion: an
        # entry in both rankings already scores higher.
        kw_items = self._keyword_search_api(query, limit=retrieve_api_k) if retrieve_api_k > 0 else []
        if vector_ok and retrieve_api_k > 0:
            api_items = rrf_fuse([kw_items, vec_items])[:retrieve_api_k]
            both = len({i.chromadb_id for i in kw_items} & {i.chromadb_id for i in vec_items})
            self._log.info(f"[search] RRF k={RRF_K}: {len(vec_items)} vector + {len(kw_items)} "
                           f"keyword ({both} in both) -> {len(api_items)} candidates")
        else:
            api_items = kw_items

        # Tier 3: Rerank (only in full mode); the first provider that answers wins
        t_rerank = 0.0
        rerank_used = "none"
        rerank_skipped: list[str] = []
        abstained = False
        abstain_reason = ""
        chain = self._get_rerank_chain() if use_rerank else []
        if use_rerank and api_items:
            from .reranker import configured_provider
            if configured_provider(self._config) == "jev" and not any(p.name == "jev" for p in chain):
                rerank_skipped.append("jev")  # selected, but no key or no config
        reranked_ok = False
        if chain and api_items:
            from .reranker import RerankError

            t_rr_start = _time.monotonic()
            # Build document texts for reranking
            api_docs = []
            for item in api_items:
                doc = f"{item.full_id or item.name}: {item.summary}"
                if item.syntax:
                    doc += f" | {item.syntax}"
                api_docs.append(doc)

            for provider in chain:
                try:
                    reranked = provider.rerank(query, api_docs, top_n=final_api_n, strict=True)
                except RerankError as e:
                    nxt = "trying the next one" if provider is not chain[-1] else "keeping the retrieval order"
                    self._log.warning(f"[search] rerank provider {provider.name} failed ({e}); {nxt}")
                    rerank_skipped.append(provider.name)
                    continue
                # Reorder api_items by rerank scores
                calibrated = bool(getattr(provider, "calibrated", False))
                reranked_items = []
                for rr in reranked:
                    if rr.index < len(api_items):
                        item = api_items[rr.index]
                        item.distance = 1.0 - rr.score  # convert score to distance
                        item.relevance = rr.score if calibrated else None
                        reranked_items.append(item)
                api_items = reranked_items
                rerank_used = provider.name
                reranked_ok = True

                # Abstention: only calibrated scores may be held to a fixed threshold.
                # The threshold decides whether to abstain and nothing else:
                # candidates below it stay in the result with their relevance.
                if calibrated and self._min_relevance is not None and api_items:
                    best = max(i.relevance or 0.0 for i in api_items)
                    if best < self._min_relevance:
                        abstained = True
                        abstain_reason = f"best relevance {best:.2f} < {self._min_relevance:.2f}"
                break
            t_rerank = _time.monotonic() - t_rr_start
            self._log.info(f"[search] rerank={rerank_used} {len(api_docs)} -> {len(api_items)} "
                           f"in {t_rerank:.2f}s"
                           + (f" abstained ({abstain_reason})" if abstained else ""))
        if not reranked_ok:
            # No reranker (or every provider failed): just trim to final count
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
            rerank_used=rerank_used, rerank_skipped=rerank_skipped,
            abstained=abstained, abstain_reason=abstain_reason,
        )

    # ------------------------------------------------------------------
    # Hydrate from SQLite
    # ------------------------------------------------------------------

    def _vector_rows(self, source: str, raw: dict) -> tuple[list[int], list[float]]:
        """SQLite row ids and ChromaDB distances for the records of a query result.

        Closest first. Each record is matched to a row by its entry name (see
        the module docstring); records without a matching row are dropped, and
        a row reached by two records is kept once.
        """
        ids = (raw.get("ids") or [[]])[0] or []
        distances = (raw.get("distances") or [[]])[0] or []
        metas = (raw.get("metadatas") or [[]])[0] or []
        index = _row_index(self._sources[source])
        probe = (self._probes or {}).get(source)
        trust_ids = probe is not None and probe.status == "aligned"
        rows: list[int] = []
        dists: list[float] = []
        for i, cid in enumerate(ids):
            rid = index.resolve(cid, metas[i] if i < len(metas) else None, trust_ids)
            if rid is None or rid in rows:
                continue
            rows.append(rid)
            dists.append(float(distances[i]) if i < len(distances) else 0.0)
        return rows, dists

    def _hydrate_api(self, raw: dict) -> list[RetrievedItem]:
        """Fetch the full API rows for the records of a ChromaDB query result."""
        ids, distances = self._vector_rows("api", raw)
        if not ids:
            return []

        conn = sqlite3.connect(self._api_db)
        conn.row_factory = sqlite3.Row
        placeholders = ",".join("?" * len(ids))
        rows = conn.execute(
            f"SELECT id, name, full_id, summary, info, syntax, parameters, remark "
            f"FROM revit_api WHERE id IN ({placeholders})",
            ids,
        ).fetchall()
        conn.close()

        row_map = {r["id"]: r for r in rows}

        items = []
        for rid, dist in zip(ids, distances):
            r = row_map.get(rid)
            if r is None:
                continue
            items.append(RetrievedItem(
                source="api",
                chromadb_id=str(rid),
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
        """Fetch the full SDK sample rows for the records of a ChromaDB query result."""
        ids, distances = self._vector_rows("sdk", raw)
        if not ids:
            return []

        conn = sqlite3.connect(self._sdk_db)
        conn.row_factory = sqlite3.Row

        placeholders = ",".join("?" * len(ids))

        if self._sdk_new_schema:
            rows = conn.execute(
                f"SELECT id, project, summary, content, mentioned_apis "
                f"FROM sdk_info WHERE id IN ({placeholders})",
                ids,
            ).fetchall()
        else:
            rows = conn.execute(
                f"SELECT id, project, filename, code, clean_code, description "
                f"FROM revit_sdk WHERE id IN ({placeholders})",
                ids,
            ).fetchall()
        row_map = {r["id"]: r for r in rows}

        conn.close()

        items = []
        for rid, dist in zip(ids, distances):
            r = row_map.get(rid)
            if r is None:
                continue
            cid = str(rid)
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

    # Verb -> base form mapping for common Revit-relevant verbs
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

    @classmethod
    def _normalize_search_tokens(cls, query: str) -> list[tuple[str, float]]:
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
            stemmed = cls._VERB_STEMS.get(token, token)

            # Skip stop words only when BOTH the original token and its stem are
            # stop words (so verb forms like "gets"->"get" that are also action
            # stems can survive if the stem is meaningful elsewhere).
            if token in cls._STOP_WORDS and stemmed in cls._STOP_WORDS:
                continue

            if stemmed in seen:
                continue
            seen.add(stemmed)

            if stemmed in cls._ENTITY_NOUNS:
                weighted.append((stemmed, 2.0))
            elif token in cls._VERB_STEMS:
                # It was a verb form -> stemmed action verb
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
        """Keyword candidates from the API reference, best first.

        BM25 over the FTS5 index (see bm25.py). When this sqlite3 has no FTS5,
        or the index can be neither read nor built (read-only data directory),
        the LIKE scorer below serves this and every later query instead.
        """
        if self._bm25 is not None:
            try:
                return self._bm25_search_api(query, limit)
            except (sqlite3.Error, OSError) as e:
                self._log.warning(f"keyword index unavailable ({type(e).__name__}: {e}); "
                                  "keyword search uses the slower LIKE scorer from now on")
                self._bm25 = None
        return self._like_search_api(query, limit)

    def _bm25_search_api(self, query: str, limit: int) -> list[RetrievedItem]:
        """Top `limit` rows by BM25.

        `distance` is 1 / (1 + score): monotonic in the BM25 score, lower is
        better. It orders and trims the candidates; it is not a probability,
        and `relevance` stays None.
        """
        hits = self._bm25.search(query, limit)
        if not hits:
            self._log.info(f"[keyword_search] bm25 query={query!r} found=0")
            return []
        ids = [rowid for rowid, _ in hits]
        conn = sqlite3.connect(self._api_db)
        conn.row_factory = sqlite3.Row
        try:
            rows = conn.execute(
                "SELECT id, name, full_id, summary, info, syntax, parameters, remark "
                f"FROM revit_api WHERE id IN ({','.join('?' * len(ids))})",
                ids,
            ).fetchall()
        finally:
            conn.close()
        by_id = {r["id"]: r for r in rows}
        items = []
        top3 = []
        for rowid, score in hits:
            r = by_id.get(rowid)
            if r is None:
                continue
            if len(top3) < 3:
                top3.append(((r["name"] or "")[:50], f"{score:.1f}"))
            items.append(RetrievedItem(
                source="api",
                chromadb_id=str(rowid),
                distance=1.0 / (1.0 + max(score, 0.0)),
                name=r["name"] or "",
                full_id=r["full_id"] or "",
                summary=r["summary"] or "",
                info=r["info"] or "",
                syntax=r["syntax"] or "",
                parameters=r["parameters"] or "",
                remark=r["remark"] or "",
            ))
        self._log.info(f"[keyword_search] bm25 query={query!r} found={len(items)} top3={top3}")
        return items

    def _like_search_api(self, query: str, limit: int = 15) -> list[RetrievedItem]:
        """Fallback keyword search without FTS5: LIKE matching on name/full_id/summary.

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
                score *= 2.0  # all tokens in name - very relevant

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
            if item.relevance is not None:
                # First line, outside the truncated body, so it is never cut off.
                body = f"Relevance: {item.relevance:.2f}" + ("\n" + body if body else "")
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
        lines = [f"rerank:    {rerank_label(results)}"]
        if results.abstained:
            lines.append(f"abstained: {results.abstain_reason}")
        if results.rewritten_query and results.rewritten_query != results.query:
            lines.append(f"query:     {results.query}")
            lines.append(f"rewritten: {results.rewritten_query}")
            lines.append("")
        lines.append("=" * 60)
        lines.append("API results:")
        lines.append("=" * 60)
        for i, item in enumerate(results.api_items, 1):
            sim = 1 - item.distance
            rel = f"  relevance: {item.relevance:.2f}" if item.relevance is not None else ""
            lines.append(f"\n[{i}] similarity: {sim:.4f}{rel}")
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


def rrf_fuse(rankings: list[list[RetrievedItem]], k: int = RRF_K) -> list[RetrievedItem]:
    """Reciprocal rank fusion of several rankings of RetrievedItem (best first).

    score = sum over the rankings that hold the entry of 1 / (k + rank), rank
    from 1; entries are keyed by chromadb_id (the SQLite row id). Ties go to
    the entry with the better single rank, then to the earlier ranking (the
    retriever passes [keyword, vector], so a keyword hit leads a vector hit of
    the same rank). `distance` becomes 1 - score / (len(rankings) / (k + 1)):
    0 for an entry ranked first everywhere, used for ordering only.
    """
    score: dict[str, float] = {}
    best: dict[str, tuple[int, int]] = {}   # (best rank, index of that ranking)
    item_of: dict[str, RetrievedItem] = {}
    for li, ranking in enumerate(rankings):
        seen: set[str] = set()
        for item in ranking:
            key = item.chromadb_id
            if key in seen:
                continue
            seen.add(key)
            rank = len(seen)
            score[key] = score.get(key, 0.0) + 1.0 / (k + rank)
            item_of.setdefault(key, item)
            if key not in best or rank < best[key][0]:
                best[key] = (rank, li)
    top = len(rankings) / (k + 1) if rankings else 1.0
    fused = []
    for key in sorted(score, key=lambda c: (-score[c], best[c][0], best[c][1])):
        item = item_of[key]
        item.distance = 1.0 - score[key] / top
        fused.append(item)
    return fused


def rerank_label(results: SearchResults) -> str:
    """`jev`, `cohere` or `none`, plus the providers that were skipped, if any."""
    if results.rerank_skipped:
        return f"{results.rerank_used} (unavailable: {', '.join(results.rerank_skipped)})"
    return results.rerank_used


def normalize_search_tokens(query: str) -> list[tuple[str, float]]:
    """The keyword search's (token, weight) list: stop words removed, verbs stemmed."""
    return RAGRetriever._normalize_search_tokens(query)
