"""Keyword-mode ranking against the shipped revit_api.db / revit_sdk.db (no API key).

The two SQLite files (34 MB) are looked up in REVIT_API_DOCS_DATA_DIR or the
default data directory. When REVIT_API_DOCS_TEST_DOWNLOAD=1 they are fetched
from the data release into that directory first (CI does this and caches the
files); otherwise the module is skipped (see conftest.find_shipped_dbs).
The vector store test also needs the installed ChromaDB directories (300 MB,
not fetched by the tests) and is skipped without them.
"""
from __future__ import annotations

import os
import sqlite3

import pytest

from revit_api_docs import data, server
from revit_api_docs.config import load_config
from revit_api_docs.retriever import RAGRetriever, check_vector_stores


@pytest.fixture(scope="module")
def shipped(shipped_dbs):
    api_db, sdk_db = shipped_dbs
    for name in ("REVIT_API_DOCS_EMBEDDING_API_KEY", "OPENROUTER_API_KEY", "COHERE_API_KEY"):
        os.environ.pop(name, None)
    r = RAGRetriever(load_config(), str(api_db), str(sdk_db), "unused", "unused")
    assert r.vector_search_enabled is False
    return r


def _names(r: RAGRetriever, query: str, n: int = 5) -> list[str]:
    results = r.search(query, api_top_k=n, code_top_k=0, rewrite=False)
    return [i.name for i in results.api_items]


def test_task_query_finds_floor_create_without_a_key(shipped):
    names = _names(shipped, "create floor from boundary", 3)
    assert any(n.startswith("Floor.Create") for n in names), names


def test_task_query_finds_element_transform_utils(shipped):
    # Missed entirely by the old arbitrary LIMIT 200 subset. BM25 puts
    # Location.Rotate first and ElementTransformUtils.RotateElement(s) within
    # the default search_revit_api top_k of 10.
    names = _names(shipped, "rotate element around axis", 10)
    assert names[0] == "Location.Rotate Method", names
    assert "ElementTransformUtils.RotateElement Method" in names, names


def test_api_name_queries(shipped):
    assert _names(shipped, "Wall.Create", 1) == ["Wall.Create Method"]
    names = _names(shipped, "FilteredElementCollector OfCategory", 2)
    assert names[0] == "FilteredElementCollector.OfCategory Method", names


def test_keyword_search_is_deterministic(shipped):
    first = _names(shipped, "create floor from boundary", 10)
    assert first == _names(shipped, "create floor from boundary", 10)


def test_misspelt_identifier_still_gets_candidates(shipped):
    """An unknown or misspelt identifier matches on its camel-case parts, so
    search_revit_api lists candidates (for a reranker to judge) instead of
    "No API entries found"."""
    text = server.search_api(shipped, "FiltredElementColector", 10)
    assert text.startswith("Rerank: none; data set: Revit 2026" + chr(10) * 2 + "### "), text[:200]


def test_wall_create_stays_first_through_the_tool(shipped):
    text = server.search_api(shipped, "Wall.Create", 10)
    assert text.startswith("Rerank: none; data set: Revit 2026" + chr(10) * 2 + "### Wall.Create" + chr(10)), text[:200]


def _installed_release(api_db) -> str:
    """The release named by the data directory's manifest; a tree without one
    is taken to be the release this package installs."""
    manifest = data.read_manifest(api_db.parent.parent) or {}
    return manifest.get("release") or data.RELEASE_TAG


def test_shipped_vector_store_is_matched_by_name(shipped_dbs):
    """The nearest records to the vector of `Wall.Create Method` come back as
    the rows that carry their names. No embedding call: the query vector is
    the record's own. v1.1-data: every record id is its revit_api row id.
    The older v1.0-data: the record ids are not row ids, the entry names are."""
    api_db, sdk_db = shipped_dbs
    p = data.paths(api_db.parent.parent)
    if not all((d / "chroma.sqlite3").is_file() for d in (p.chromadb_api, p.chromadb_code)):
        pytest.skip("the shipped vector store is not installed")
    old = _installed_release(api_db) == "v1.0-data"
    api, sdk = check_vector_stores(str(api_db), str(sdk_db), str(p.chromadb_api), str(p.chromadb_code))
    if old:
        assert (api.status, api.records, api.sampled) == ("misaligned", 28863, 50)
        assert api.id_matches <= 2 and api.name_matches >= 45, api
    else:
        conn = sqlite3.connect(api_db)
        try:
            rows = conn.execute("SELECT COUNT(*) FROM revit_api").fetchone()[0]
        finally:
            conn.close()
        assert (api.status, api.records, api.sampled, api.id_matches) == ("aligned", rows, 50, 50), api
    assert (sdk.status, sdk.records, sdk.id_matches) == ("aligned", 153, 50), sdk

    r = RAGRetriever(load_config(), str(api_db), str(sdk_db), str(p.chromadb_api), str(p.chromadb_code))
    r._open_collections()
    wall = r._api_collection.get(where={"name": "Wall.Create Method"}, include=["embeddings"])
    raw = r._api_collection.query(query_embeddings=[list(wall["embeddings"][0])], n_results=10,
                                  include=["metadatas", "distances"])
    record_names = [m["name"] for m in raw["metadatas"][0]]
    items = r._hydrate_api(raw)
    assert items[0].name == record_names[0] == "Wall.Create Method"
    assert len(items) >= 8 and all(i.name in record_names for i in items)
    conn = sqlite3.connect(api_db)
    try:
        by_id = conn.execute("SELECT name FROM revit_api WHERE id = ?", (int(raw["ids"][0][0]),)).fetchone()
    finally:
        conn.close()
    if old:
        assert by_id is None or by_id[0] != "Wall.Create Method"   # what the id lookup returned
    else:
        assert by_id is not None and by_id[0] == "Wall.Create Method"
        assert [i.chromadb_id for i in items] == [str(x) for x in raw["ids"][0]]   # rule 1: own id


# Pages added in v1.1-data: constructors and members whose C# signature has no
# `public` (interface members). The older v1.0-data has none of them.
NEW_IN_V1_1 = [
    ("XYZ constructor", "XYZ(Double, Double, Double) Constructor"),
    ("new Transaction(document, name)", "Transaction(Document, String) Constructor"),
    ("IExternalEventHandler.Execute", "IExternalEventHandler.Execute Method"),
]


@pytest.mark.parametrize("query,name", NEW_IN_V1_1)
def test_constructor_and_interface_member_pages_are_found(shipped, shipped_dbs, query, name):
    api_db, _ = shipped_dbs
    if _installed_release(api_db) == "v1.0-data":
        pytest.skip("v1.0-data has no constructor or interface member pages")
    conn = sqlite3.connect(api_db)
    try:
        rows = conn.execute("SELECT id, full_id FROM revit_api WHERE name = ?", (name,)).fetchall()
    finally:
        conn.close()
    assert len(rows) == 1 and rows[0][0] > 27596, rows   # appended after the v1.0-data rows
    text = server.search_api(shipped, query, 10)
    assert f"### {rows[0][1] or name}" + chr(10) in text, text[:600]


def test_keyword_index_sits_next_to_the_shipped_db(shipped, shipped_dbs):
    from revit_api_docs import bm25

    api_db, _ = shipped_dbs
    _names(shipped, "Wall.Create", 1)
    index = api_db.with_name(bm25.INDEX_FILENAME)
    assert index.is_file()
    meta = bm25.Bm25Index.read_meta(index)
    assert meta is not None and meta["format_version"] == str(bm25.FORMAT_VERSION)
    assert meta["source_sha256"] == bm25.source_sha256(api_db)
    assert int(meta["rows"]) > 0
