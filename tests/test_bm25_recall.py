"""BM25 recall on the shipped API reference (no key, no network).

Two query sets with verified gold rows, written separately:

- tests/fixtures/retrieval_queries.yaml: 60 English queries, 50 that the
  reference answers (groups N, T, F) and 10 it does not (group X).
- tests/fixtures/retrieval_queries_v2.yaml: 60 more, 40 answerable and 20 not;
  9 of its X queries sit next to a real API (a class name used as a member,
  a member of another class, a property that is really a parameter, a
  misspelling).

Without an embedding key the keyword search is the reranker's whole candidate
pool, so its top 30 bounds what any reranker can return.

Skipped unless the shipped data set is available (see
conftest.find_shipped_dbs) and this sqlite3 has FTS5.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

import pytest
import yaml

from revit_api_docs.bm25 import fts5_available
from revit_api_docs.config import load_config
from revit_api_docs.retriever import RAGRetriever

FIXTURES = Path(__file__).parent / "fixtures"
POOL = 30   # retrieval.api.top_k, the reranker's candidate pool


@dataclass(frozen=True)
class QuerySet:
    name: str
    path: Path
    groups: dict[str, int]
    # Floors for the measured numbers below. The margin absorbs small
    # tokenizer differences between SQLite builds, nothing more.
    min_gold_in_pool: int
    min_hit_at_5: int
    min_x_with_candidates: int

    def queries(self) -> list[dict]:
        return yaml.safe_load(self.path.read_text(encoding="utf-8"))["queries"]


QUERY_SETS = [
    # Measured on v1.0-data: 45/50 gold in the pool, 38/50 hit@5, 10/10 X
    # queries with candidates.
    QuerySet("v1", FIXTURES / "retrieval_queries.yaml", {"N": 20, "T": 20, "F": 10, "X": 10},
             min_gold_in_pool=44, min_hit_at_5=36, min_x_with_candidates=10),
    # Measured on v1.0-data: 34/40 gold in the pool, 27/40 hit@5, 19/20 X
    # queries with candidates (v2-X18 "Trasaction Comit": no word of it is in
    # the index).
    QuerySet("v2", FIXTURES / "retrieval_queries_v2.yaml", {"N": 15, "T": 15, "F": 10, "X": 20},
             min_gold_in_pool=33, min_hit_at_5=25, min_x_with_candidates=19),
]
by_name = pytest.mark.parametrize("qset", QUERY_SETS, ids=lambda s: s.name)


@pytest.fixture(scope="module")
def pools(shipped_dbs):
    """pools(qset) -> top-30 BM25 row ids per query id, computed once per set."""
    if not fts5_available():
        pytest.skip("this sqlite3 build has no FTS5")
    api_db, sdk_db = shipped_dbs
    for name in ("REVIT_API_DOCS_EMBEDDING_API_KEY", "OPENROUTER_API_KEY", "COHERE_API_KEY"):
        os.environ.pop(name, None)
    r = RAGRetriever(load_config(), str(api_db), str(sdk_db), "unused", "unused")
    assert r.vector_search_enabled is False and r._bm25 is not None
    cache: dict[str, dict[str, list[int]]] = {}

    def get(qset: QuerySet) -> dict[str, list[int]]:
        if qset.name not in cache:
            out = {}
            for q in qset.queries():
                res = r.search(q["query"], api_top_k=POOL, code_top_k=0, rewrite=False, rerank=False)
                out[q["id"]] = [int(i.chromadb_id) for i in res.api_items]
            assert r._bm25 is not None  # never fell back to the LIKE scorer
            cache[qset.name] = out
        return cache[qset.name]

    return get


@by_name
def test_fixture_shape(qset):
    qs = qset.queries()
    groups = [q["group"] for q in qs]
    assert {g: groups.count(g) for g in "NTFX"} == qset.groups
    assert len({q["id"] for q in qs}) == len(qs) == sum(qset.groups.values())
    assert all(q["verified"] for q in qs)
    assert all(q["gold"] for q in qs if q["group"] != "X")
    assert all("gold" not in q and q["absent"] for q in qs if q["group"] == "X")
    assert qset.path.read_bytes().isascii()


def test_query_sets_share_no_query():
    v1, v2 = ({q["query"].strip().lower() for q in s.queries()} for s in QUERY_SETS)
    assert v1.isdisjoint(v2)


@by_name
def test_bm25_recall_and_hit_at_5(pools, qset):
    ids_by_query = pools(qset)
    answerable = [q for q in qset.queries() if q["group"] != "X"]
    in_pool, hit5, missed = 0, 0, []
    for q in answerable:
        gold = {g["id"] for g in q["gold"]}
        ids = ids_by_query[q["id"]]
        if gold & set(ids):
            in_pool += 1
        else:
            missed.append(q["id"])
        if gold & set(ids[:5]):
            hit5 += 1
    n = len(answerable)
    print(f"\n{qset.name} BM25 top-{POOL}: gold in pool {in_pool}/{n}, hit@5 {hit5}/{n}, "
          f"not in pool: {missed}")
    assert in_pool >= qset.min_gold_in_pool, (in_pool, missed)
    assert hit5 >= qset.min_hit_at_5, hit5


@by_name
def test_unanswerable_queries_still_reach_the_reranker(pools, qset):
    """Group X (2027-only types, members that do not exist, misspellings,
    non-Revit questions) must get candidates: judging them is the reranker's
    job, and an empty pool would skip it."""
    ids_by_query = pools(qset)
    x = [q["id"] for q in qset.queries() if q["group"] == "X"]
    empty = [qid for qid in x if not ids_by_query[qid]]
    print(f"\n{qset.name} X queries with candidates: {len(x) - len(empty)}/{len(x)}, empty: {empty}")
    assert len(x) - len(empty) >= qset.min_x_with_candidates, empty
