"""BM25 keyword index (SQLite FTS5): query terms, the index file next to
revit_api.db, rebuilds, the LIKE fallback, and the candidate pool size.
Tiny fixtures only; no network, no data set."""
from __future__ import annotations

import hashlib
import logging
import sqlite3
import subprocess
import sys
import threading
import time
from pathlib import Path

import pytest

from revit_api_docs import bm25, data
from revit_api_docs import config as cfgmod
from revit_api_docs.retriever import RAGRetriever

ROOT = Path(__file__).resolve().parents[1]

needs_fts5 = pytest.mark.skipif(not bm25.fts5_available(), reason="this sqlite3 build has no FTS5")


def _retriever(api_db: Path, sdk_db: Path, tmp_path: Path) -> RAGRetriever:
    return RAGRetriever(cfgmod.load_config(), str(api_db), str(sdk_db), str(tmp_path / "a"), str(tmp_path / "b"))


@pytest.fixture
def no_embedding_key(monkeypatch):
    for name in ("REVIT_API_DOCS_EMBEDDING_API_KEY", "OPENROUTER_API_KEY", "COHERE_API_KEY"):
        monkeypatch.delenv(name, raising=False)


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


# -- query terms ---------------------------------------------------------------

@pytest.mark.parametrize("query, terms", [
    ("Wall.Create", ["wall", "create"]),
    ("FiltredElementColector", ["filtredelementcolector", "filtred", "element", "colector"]),
    ("how to get the ElementId of a wall in Revit", ["get", "elementid", "element", "id", "wall"]),
    ("UIDocument Selection", ["uidocument", "ui", "document", "selection"]),
    ("XYZ BasisZ", ["xyz", "basisz", "basis"]),   # one-letter parts are dropped
    ("Wall.Create Wall.Create", ["wall", "create"]),
    ("the of a", []),
    ("".join(map(chr, (0x5899, 0x20, 0x521B, 0x5EFA))), []),   # Chinese words carry no terms
])
def test_query_terms(query, terms):
    assert bm25.query_terms(query) == terms


def test_name_words_splits_compound_identifiers_only():
    assert bm25.name_words("FilteredElementCollector.OfCategory Method") == "Filtered Element Collector Of Category"
    assert bm25.name_words("Wall.Create Method") == ""
    assert bm25.name_words("BuiltInParameter.WALL_BASE_OFFSET Field") == "Built In Parameter"


# -- index file ------------------------------------------------------------------

@needs_fts5
def test_index_is_built_next_to_the_db_on_first_search(tiny_dbs, tmp_path, no_embedding_key):
    api_db, sdk_db = tiny_dbs
    before = _sha(api_db)
    r = _retriever(api_db, sdk_db, tmp_path)
    index = api_db.with_name("revit_api_fts.db")
    assert not index.exists()  # first use builds it, not the constructor

    names = [i.name for i in r.search("Wall.Create", api_top_k=3, code_top_k=0, rewrite=False).api_items]
    assert names[0] == "Wall.Create Method"
    assert index.is_file()
    assert bm25.Bm25Index.read_meta(index) == {
        "format_version": str(bm25.FORMAT_VERSION), "source_sha256": before, "rows": "6"}
    assert _sha(api_db) == before  # the source database is never written to
    assert sorted(p.name for p in api_db.parent.iterdir() if "fts" in p.name) == ["revit_api_fts.db"]


@needs_fts5
def test_index_is_reused_and_rebuilt_when_the_source_changes(tiny_dbs, tmp_path, no_embedding_key, monkeypatch):
    api_db, sdk_db = tiny_dbs
    builds: list[str] = []
    real_build = bm25.Bm25Index._build

    def counting_build(self, sha):
        builds.append(sha)
        return real_build(self, sha)

    monkeypatch.setattr(bm25.Bm25Index, "_build", counting_build)

    _retriever(api_db, sdk_db, tmp_path).search("wall", api_top_k=3, code_top_k=0, rewrite=False)
    _retriever(api_db, sdk_db, tmp_path).search("wall", api_top_k=3, code_top_k=0, rewrite=False)
    assert len(builds) == 1  # a second process (retriever) reuses the file

    conn = sqlite3.connect(api_db)
    conn.execute("INSERT INTO revit_api (name, full_id, summary) VALUES (?,?,?)",
                 ("Roof.Create Method", "Roof.Create", "Creates a roof."))
    conn.commit()
    conn.close()
    r = _retriever(api_db, sdk_db, tmp_path)
    assert [i.name for i in r.search("roof", api_top_k=3, code_top_k=0, rewrite=False).api_items] == [
        "Roof.Create Method"]
    assert len(builds) == 2 and builds[1] == _sha(api_db)

    monkeypatch.setattr(bm25, "FORMAT_VERSION", bm25.FORMAT_VERSION + 1)
    _retriever(api_db, sdk_db, tmp_path).search("roof", api_top_k=3, code_top_k=0, rewrite=False)
    assert len(builds) == 3  # a new index format rebuilds too


@needs_fts5
def test_a_damaged_index_file_is_rebuilt(tiny_dbs, tmp_path, no_embedding_key):
    api_db, sdk_db = tiny_dbs
    api_db.with_name("revit_api_fts.db").write_bytes(b"not a database")
    r = _retriever(api_db, sdk_db, tmp_path)
    assert r.search("Wall.Create", api_top_k=1, code_top_k=0, rewrite=False).api_items[0].name == "Wall.Create Method"
    assert bm25.Bm25Index.read_meta(api_db.with_name("revit_api_fts.db")) is not None


def test_source_sha_comes_from_the_manifest_when_it_describes_the_file(tiny_data_dir, monkeypatch):
    api_db = data.paths(tiny_data_dir).api_db
    real = _sha(api_db)
    assert bm25.source_sha256(api_db) == real  # fixture manifest has no file entries: hashed
    fake = "ab" * 32
    (tiny_data_dir / "manifest.json").write_text(
        '{"release": "%s", "files": {"revit_api.db": {"target": "sqlite/revit_api.db", '
        '"sha256": "%s", "size": %d}}}' % (data.RELEASE_TAG, fake, api_db.stat().st_size), encoding="utf-8")
    with monkeypatch.context() as m:
        m.setattr(data, "sha256_file", lambda p: pytest.fail("must not hash"))
        assert bm25.source_sha256(api_db) == fake
    # a manifest entry for a file of another size is not trusted
    (tiny_data_dir / "manifest.json").write_text(
        '{"release": "%s", "files": {"revit_api.db": {"target": "sqlite/revit_api.db", '
        '"sha256": "%s", "size": 1}}}' % (data.RELEASE_TAG, fake), encoding="utf-8")
    assert bm25.source_sha256(api_db) == real


@needs_fts5
def test_concurrent_first_queries_build_the_index_once(tiny_dbs, tmp_path, no_embedding_key, monkeypatch):
    api_db, sdk_db = tiny_dbs
    r = _retriever(api_db, sdk_db, tmp_path)
    builds: list[int] = []
    real_build = bm25.Bm25Index._build

    def slow_build(self, sha):
        builds.append(1)
        time.sleep(0.1)  # widen the race window
        return real_build(self, sha)

    monkeypatch.setattr(bm25.Bm25Index, "_build", slow_build)
    results: list[list[str]] = []

    def first_query():
        res = r.search("Wall.Create", api_top_k=2, code_top_k=0, rewrite=False)
        results.append([i.name for i in res.api_items])

    threads = [threading.Thread(target=first_query) for _ in range(4)]
    for th in threads:
        th.start()
    for th in threads:
        th.join()
    assert builds == [1]
    assert len(results) == 4 and all(names[0] == "Wall.Create Method" for names in results)


@needs_fts5
def test_installer_ignores_the_index_file(tiny_data_dir, no_key_env):
    """`missing`, `unexpected_sizes` and `download` neither miss nor flag it."""
    before = (data.missing(tiny_data_dir), [a.name for a, _ in data.unexpected_sizes(tiny_data_dir)])
    env = dict(no_key_env, REVIT_API_DOCS_DATA_DIR=str(tiny_data_dir),
               REVIT_API_DOCS_RELEASE_URL="http://127.0.0.1:9/unreachable")
    out = subprocess.run([sys.executable, "-m", "revit_api_docs", "search", "Wall.Create"],
                         capture_output=True, text=True, env=env, cwd=str(ROOT))
    assert out.returncode == 0, out.stderr
    assert (tiny_data_dir / "sqlite" / "revit_api_fts.db").is_file()
    assert (data.missing(tiny_data_dir), [a.name for a, _ in data.unexpected_sizes(tiny_data_dir)]) == before
    out = subprocess.run([sys.executable, "-m", "revit_api_docs", "download"],
                         capture_output=True, text=True, env=env, cwd=str(ROOT))
    assert out.returncode == 0, out.stderr
    assert "revit_api_fts" not in out.stderr and "data ready in" in out.stderr


# -- fallback and single path ------------------------------------------------------

def test_without_fts5_the_like_scorer_runs_and_warns_once(tiny_dbs, tmp_path, no_embedding_key,
                                                          monkeypatch, caplog):
    import revit_api_docs.retriever as retriever_mod

    api_db, sdk_db = tiny_dbs
    monkeypatch.setattr(retriever_mod, "fts5_available", lambda: False)
    caplog.set_level(logging.WARNING, logger="revit_api_docs.retriever")
    r = _retriever(api_db, sdk_db, tmp_path)
    for _ in range(2):
        names = [i.name for i in r.search("Wall.Create", api_top_k=2, code_top_k=0, rewrite=False).api_items]
        assert names == ["Wall.Create Method", "Wall.Create(Document, Curve, ElementId, Boolean) Method"]
    assert [rec.getMessage() for rec in caplog.records if "FTS5" in rec.getMessage()] == [
        "SQLite FTS5 is not available in this Python build; keyword search uses the slower LIKE scorer"]
    assert not api_db.with_name("revit_api_fts.db").exists()


@needs_fts5
def test_with_fts5_the_like_scorer_never_runs(tiny_dbs, tmp_path, no_embedding_key, monkeypatch):
    api_db, sdk_db = tiny_dbs
    r = _retriever(api_db, sdk_db, tmp_path)
    assert r._bm25 is not None
    monkeypatch.setattr(r, "_like_search_api", lambda *a, **k: pytest.fail("LIKE scorer ran"))
    monkeypatch.setattr(r, "_keyword_candidates", lambda *a, **k: pytest.fail("LIKE scorer ran"))
    assert r.search("wall", api_top_k=3, code_top_k=0, rewrite=False).api_items
    assert r.search("zzzz", api_top_k=3, code_top_k=0, rewrite=False).api_items == []


@needs_fts5
def test_an_unbuildable_index_switches_to_like_for_good(tiny_dbs, tmp_path, no_embedding_key, monkeypatch, caplog):
    api_db, sdk_db = tiny_dbs
    r = _retriever(api_db, sdk_db, tmp_path)
    attempts: list[int] = []

    def failing_build(self, sha):
        attempts.append(1)
        raise sqlite3.OperationalError("unable to open database file")

    monkeypatch.setattr(bm25.Bm25Index, "_build", failing_build)
    caplog.set_level(logging.WARNING, logger="revit_api_docs.retriever")
    for _ in range(2):
        names = [i.name for i in r.search("Wall.Create", api_top_k=2, code_top_k=0, rewrite=False).api_items]
        assert names[0] == "Wall.Create Method"
    assert attempts == [1] and r._bm25 is None
    assert caplog.text.count("keyword index unavailable") == 1


@needs_fts5
def test_a_failed_build_leaves_no_temporary_file(tiny_dbs, tmp_path, no_embedding_key, monkeypatch):
    api_db, sdk_db = tiny_dbs

    def broken(name):
        raise sqlite3.OperationalError("disk I/O error")

    monkeypatch.setattr(bm25, "name_words", broken)
    r = _retriever(api_db, sdk_db, tmp_path)
    assert r.search("Wall.Create", api_top_k=1, code_top_k=0, rewrite=False).api_items  # LIKE answered
    assert r._bm25 is None
    assert [p.name for p in api_db.parent.iterdir() if p.name.startswith("revit_api_fts")] == []


# -- candidate pool ------------------------------------------------------------------

def test_keyword_pool_is_top_k_without_vectors_and_unchanged_with_them(tiny_dbs, tmp_path, no_embedding_key,
                                                                       monkeypatch):
    api_db, sdk_db = tiny_dbs
    r = _retriever(api_db, sdk_db, tmp_path)
    limits: list[int] = []
    real = r._keyword_search_api

    def recording(query, limit=15):
        limits.append(limit)
        return real(query, limit)

    monkeypatch.setattr(r, "_keyword_search_api", recording)

    # No vector search, reranking on (what search_revit_api does): the keyword
    # search is the reranker's whole pool, retrieval.api.top_k.
    res = r.search("wall", api_top_k=3, code_top_k=0, rewrite=False, rerank=True)
    assert limits == [r._api_top_k] and r._api_top_k == 30
    assert len(res.api_items) == 3  # no reranker: BM25 order cut to the requested count
    # Reranking off: exactly the requested count.
    r.search("wall", api_top_k=3, code_top_k=0, rewrite=False, rerank=False)
    assert limits[-1] == 3

    # Next to vector results the keyword search adds final_api_n entries, as before.
    class _ZeroEmbedder:
        def embed_query(self, query):
            return [0.0]

    class _Collection:
        def query(self, query_embeddings, n_results):
            return {"ids": [["1"]], "distances": [[0.5]]}

    r._embedder = _ZeroEmbedder()
    r._api_collection = r._code_collection = _Collection()
    monkeypatch.setattr(r, "_open_collections", lambda: None)
    r.search("wall", api_top_k=3, code_top_k=0, rewrite=False, rerank=True)
    assert limits[-1] == 3
