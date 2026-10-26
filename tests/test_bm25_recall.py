"""BM25 recall on the shipped API reference (no key, no network).

tests/fixtures/retrieval_queries.yaml holds 60 English queries with verified
gold rows: 50 that the reference answers (groups N, T, F) and 10 it does not
(group X). Without an embedding key the keyword search is the reranker's
whole candidate pool, so its top 30 bounds what any reranker can return.

Skipped unless the shipped data set is available (see
conftest.find_shipped_dbs) and this sqlite3 has FTS5.
"""
from __future__ import annotations

import os
from pathlib import Path

import pytest
import yaml

from revit_api_docs.bm25 import fts5_available
from revit_api_docs.config import load_config
from revit_api_docs.retriever import RAGRetriever

FIXTURE = Path(__file__).parent / "fixtures" / "retrieval_queries.yaml"
POOL = 30   # retrieval.api.top_k, the reranker's candidate pool

# Measured on v1.0-data: 45/50 gold in the pool, 38/50 hit@5. The margin
# absorbs small tokenizer differences between SQLite builds, nothing more.
MIN_GOLD_IN_POOL = 44
MIN_HIT_AT_5 = 36


def _queries() -> list[dict]:
    return yaml.safe_load(FIXTURE.read_text(encoding="utf-8"))["queries"]


@pytest.fixture(scope="module")
def pools(shipped_dbs) -> dict[str, list[int]]:
    """Top-30 BM25 row ids per query id."""
    if not fts5_available():
        pytest.skip("this sqlite3 build has no FTS5")
    api_db, sdk_db = shipped_dbs
    for name in ("REVIT_API_DOCS_EMBEDDING_API_KEY", "OPENROUTER_API_KEY", "COHERE_API_KEY"):
        os.environ.pop(name, None)
    r = RAGRetriever(load_config(), str(api_db), str(sdk_db), "unused", "unused")
    assert r.vector_search_enabled is False and r._bm25 is not None
    out = {}
    for q in _queries():
        res = r.search(q["query"], api_top_k=POOL, code_top_k=0, rewrite=False, rerank=False)
        out[q["id"]] = [int(i.chromadb_id) for i in res.api_items]
    assert r._bm25 is not None  # never fell back to the LIKE scorer
    return out


def test_fixture_shape():
    qs = _queries()
    groups = [q["group"] for q in qs]
    assert {g: groups.count(g) for g in "NTFX"} == {"N": 20, "T": 20, "F": 10, "X": 10}
    assert len({q["id"] for q in qs}) == 60
    assert all(q["verified"] for q in qs)
    assert all(q["gold"] for q in qs if q["group"] != "X")
    assert all("gold" not in q and q["absent"] for q in qs if q["group"] == "X")
    assert FIXTURE.read_bytes().isascii()


def test_bm25_recall_and_hit_at_5(pools):
    in_pool, hit5, missed = 0, 0, []
    for q in _queries():
        if q["group"] == "X":
            continue
        gold = {g["id"] for g in q["gold"]}
        ids = pools[q["id"]]
        if gold & set(ids):
            in_pool += 1
        else:
            missed.append(q["id"])
        if gold & set(ids[:5]):
            hit5 += 1
    print(f"\nBM25 top-{POOL}: gold in pool {in_pool}/50, hit@5 {hit5}/50, not in pool: {missed}")
    assert in_pool >= MIN_GOLD_IN_POOL, (in_pool, missed)
    assert hit5 >= MIN_HIT_AT_5, hit5


def test_unanswerable_queries_still_reach_the_reranker(pools):
    """Group X (2027-only types, members that do not exist, misspellings,
    non-Revit questions) must get candidates: judging them is the reranker's
    job, and an empty pool would skip it."""
    empty = [q["id"] for q in _queries() if q["group"] == "X" and not pools[q["id"]]]
    assert empty == []
