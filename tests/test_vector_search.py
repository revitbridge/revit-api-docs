"""Vector search over the tiny data set: records matched to SQLite rows by entry
name, the consistency probe, and reciprocal rank fusion with BM25.

Stand-in collections (conftest.FakeCollection) for most tests; one test builds
tiny real ChromaDB collections (explicit embeddings, no embedding API).
"""
from __future__ import annotations

import logging
import sqlite3

import pytest

from revit_api_docs import config as cfgmod
from revit_api_docs import server
from revit_api_docs.bm25 import fts5_available
from revit_api_docs.reranker import RerankResult
from revit_api_docs.retriever import (
    PROBE_SAMPLE,
    RAGRetriever,
    RetrievedItem,
    check_vector_stores,
    rrf_fuse,
)

needs_fts5 = pytest.mark.skipif(not fts5_available(), reason="this sqlite3 build has no FTS5")


class _Embedder:
    """Counts calls; returns a fixed vector (the fake collections ignore it)."""

    def __init__(self, vector=(0.0, 0.0)):
        self.vector = list(vector)
        self.calls = 0

    def embed_query(self, query):
        self.calls += 1
        return self.vector


@pytest.fixture
def no_keys(monkeypatch):
    for name in ("REVIT_API_DOCS_EMBEDDING_API_KEY", "OPENROUTER_API_KEY"):
        monkeypatch.delenv(name, raising=False)


def _retriever(api_db, sdk_db, tmp_path, monkeypatch=None, api=None, code=None, embedder=None) -> RAGRetriever:
    r = RAGRetriever(cfgmod.load_config(), str(api_db), str(sdk_db), str(tmp_path / "a"), str(tmp_path / "b"))
    if api is not None:
        r._embedder = embedder or _Embedder()
        r._api_collection, r._code_collection = api, code
        monkeypatch.setattr(r, "_open_collections", lambda: None)
    return r


def _row_names(api_db) -> dict[int, str]:
    conn = sqlite3.connect(api_db)
    try:
        return dict(conn.execute("SELECT id, name FROM revit_api"))
    finally:
        conn.close()


# -- R1: records are matched to rows by entry name -----------------------------

def test_misaligned_ids_are_matched_by_name(tiny_dbs, tmp_path, monkeypatch, no_keys, fake_vectors, caplog):
    """Every record id names a different row than the record's entry (the
    shipped data set's state): the hits come back as the rows named in the
    records, with one vector_store_misaligned warning, and vectors stay on."""
    api_db, sdk_db = tiny_dbs
    collection, records = fake_vectors
    recs = records("api", shift=2)
    rows = _row_names(api_db)
    hits = ["1", "6"]
    assert all(rows[int(h)] != recs[h]["name"] for h in hits)   # an id lookup returns other rows
    api = collection(recs, hits=hits)
    r = _retriever(api_db, sdk_db, tmp_path, monkeypatch, api, collection(records("sdk")))
    caplog.set_level(logging.INFO, logger="revit_api_docs.retriever")
    caplog.clear()   # the "embedding disabled" warning of a retriever built without a key

    items = r._hydrate_api(api.query([0.0], n_results=2, include=["metadatas", "distances"]))
    assert [i.name for i in items] == [recs[h]["name"] for h in hits]
    assert [rows[int(i.chromadb_id)] for i in items] == [recs[h]["name"] for h in hits]

    res = r.search("floor", api_top_k=5, code_top_k=0, rewrite=False, rerank=False)
    assert recs["1"]["name"] in [i.name for i in res.api_items]
    assert r.vector_search_enabled is True
    probes = {p.collection: p for p in r.vector_probes}
    assert probes["revit_api"].status == "misaligned"
    assert (probes["revit_api"].sampled, probes["revit_api"].id_matches, probes["revit_api"].name_matches) == (6, 0, 6)
    assert probes["revit_sdk"].status == "aligned"
    warnings = [rec.getMessage() for rec in caplog.records if rec.levelno == logging.WARNING]
    assert len(warnings) == 1 and warnings[0].startswith("vector_store_misaligned: revit_api vectors")
    assert "matched to rows by name" in warnings[0]


def test_unmatched_store_switches_vector_search_off(tiny_dbs, tmp_path, monkeypatch, no_keys, fake_vectors, caplog):
    """No sampled entry name is in SQLite: vector search is switched off before
    any embedding call, the result is the keyword result, and the tool's first
    line is unchanged."""
    api_db, sdk_db = tiny_dbs
    collection, records = fake_vectors
    api = collection(records("api", rename=lambda n: "Nonexistent." + n), hits=["1", "2"])
    embedder = _Embedder()
    r = _retriever(api_db, sdk_db, tmp_path, monkeypatch, api, collection(records("sdk")), embedder)
    plain = _retriever(api_db, sdk_db, tmp_path)
    caplog.set_level(logging.WARNING, logger="revit_api_docs.retriever")
    caplog.clear()
    assert r.vector_search_enabled is True   # not probed yet

    res = r.search("Wall.Create", api_top_k=5, code_top_k=0, rewrite=False, rerank=True)

    assert r.vector_search_enabled is False
    assert embedder.calls == 0 and api.queries == []
    expected = plain.search("Wall.Create", api_top_k=5, code_top_k=0, rewrite=False, rerank=True)
    assert [i.chromadb_id for i in res.api_items] == [i.chromadb_id for i in expected.api_items]
    assert "vector_store_misaligned: revit_api vectors (6 records) vs SQLite revit_api: unusable" in caplog.text
    assert "0/6 sampled entry names (name) found; vector search is off" in caplog.text
    assert server.search_api(r, "Wall.Create", 3).splitlines()[0] == "Rerank: none; data set: Revit 2026"
    r.search("Wall.Create", api_top_k=5, code_top_k=0, rewrite=False)
    assert embedder.calls == 0   # stays off for the process


def test_aligned_store_is_used_without_a_warning(tiny_dbs, tmp_path, monkeypatch, no_keys, fake_vectors, caplog):
    api_db, sdk_db = tiny_dbs
    collection, records = fake_vectors
    api = collection(records("api"), hits=["5"])
    r = _retriever(api_db, sdk_db, tmp_path, monkeypatch, api, collection(records("sdk"), hits=["2"]))
    caplog.set_level(logging.WARNING, logger="revit_api_docs.retriever")
    caplog.clear()
    res = r.search("boundary loops", api_top_k=3, code_top_k=1, rewrite=False, rerank=False)
    assert res.api_items[0].name.startswith("Floor.Create(")
    assert [i.project for i in res.sdk_items] == ["FloorSlab"]
    assert [p.status for p in r.vector_probes] == ["aligned", "aligned"]
    assert caplog.text == ""


@pytest.mark.parametrize("valid, status", [(10, "aligned"), (6, "misaligned"), (5, "misaligned"), (4, "unusable")])
def test_probe_thresholds(tiny_dbs, tmp_path, monkeypatch, no_keys, fake_vectors, valid, status):
    """Ten records, ids 1-10 (only 1-6 are rows): `valid` of them carry an entry
    name found in SQLite, at its own id when it has one. Ids naming their row's
    entry below 90%: misaligned; names found below 50%: unusable."""
    api_db, sdk_db = tiny_dbs
    collection, records = fake_vectors
    names = list(_row_names(api_db).values())
    recs = {str(i): {"name": names[(i - 1) % 6] if i <= valid else f"Missing{i}.Member Method"}
            for i in range(1, 11)}
    r = _retriever(api_db, sdk_db, tmp_path, monkeypatch, collection(recs), collection(records("sdk")))
    if valid == 10:   # ids 7-10 are not rows: give them aligned stand-ins instead
        conn = sqlite3.connect(api_db)
        conn.executemany("INSERT INTO revit_api (id, name) VALUES (?, ?)",
                         [(i, recs[str(i)]["name"] + f" {i}") for i in range(7, 11)])
        conn.commit()
        conn.close()
        for i in range(7, 11):
            recs[str(i)]["name"] += f" {i}"
        r._api_collection = collection(recs)
    r.search("wall", api_top_k=3, code_top_k=0, rewrite=False)
    probe = r.vector_probes[0]
    assert probe.sampled == 10 and probe.status == status
    assert r.vector_search_enabled is (status != "unusable")


def test_probe_samples_fifty_records_in_ten_reads(tiny_dbs, tmp_path, monkeypatch, no_keys, fake_vectors):
    api_db, sdk_db = tiny_dbs
    collection, records = fake_vectors
    names = list(_row_names(api_db).values())
    big = collection({str(i): {"name": names[i % 6]} for i in range(1, 1001)})
    reads: list[tuple] = []
    real_get = big.get

    def get(**kw):
        reads.append((kw.get("limit"), kw.get("offset")))
        return real_get(**kw)

    big.get = get
    r = _retriever(api_db, sdk_db, tmp_path, monkeypatch, big, collection(records("sdk")))
    r.search("wall", api_top_k=3, code_top_k=0, rewrite=False)
    assert reads == [(5, o) for o in range(0, 1000, 100)]
    probe = r.vector_probes[0]
    assert (probe.records, probe.sampled, probe.name_matches) == (1000, PROBE_SAMPLE, 50)
    assert probe.status == "misaligned"


def test_probe_runs_once_per_process(tiny_dbs, tmp_path, monkeypatch, no_keys, fake_vectors):
    api_db, sdk_db = tiny_dbs
    collection, records = fake_vectors
    api = collection(records("api", shift=1), hits=["2"])
    counts: list[int] = []
    real_count = api.count
    api.count = lambda: counts.append(1) or real_count()
    first = _retriever(api_db, sdk_db, tmp_path, monkeypatch, api, collection(records("sdk")))
    second = _retriever(api_db, sdk_db, tmp_path, monkeypatch, api, collection(records("sdk")))
    for r in (first, second, first):
        r.search("wall", api_top_k=3, code_top_k=0, rewrite=False)
    assert len(counts) == 1
    assert first.vector_probes == second.vector_probes


def test_rows_sharing_a_name_are_told_apart(tiny_dbs, tmp_path, monkeypatch, no_keys, fake_vectors):
    """Same entry name on two rows: the record's full_id, then its info text,
    picks the row; with neither, the lowest id."""
    api_db, sdk_db = tiny_dbs
    conn = sqlite3.connect(api_db)
    conn.executemany("INSERT INTO revit_api (id, name, full_id, info) VALUES (?, ?, ?, ?)", [
        (20, "Selection.PickPoint Method", "Selection.PickPoint", "Overload list."),
        (21, "Selection.PickPoint Method", "M:Autodesk.Revit.UI.Selection.Selection.PickPoint", "Prompts."),
    ])
    conn.commit()
    conn.close()
    collection, records = fake_vectors
    recs = records("api")
    recs.update({
        "100": {"name": "Selection.PickPoint Method", "info": "Prompts."},
        "101": {"name": "Selection.PickPoint Method", "full_id": "Selection.PickPoint", "info": "Prompts."},
        "102": {"name": "Selection.PickPoint Method", "info": ""},
        "21": {"name": "Selection.PickPoint Method"},   # its own id carries the name
    })
    api = collection(recs)
    r = _retriever(api_db, sdk_db, tmp_path, monkeypatch, api, collection(records("sdk")))
    for hit, row in (("100", 21), ("101", 20), ("102", 20), ("21", 21)):
        raw = {"ids": [[hit]], "distances": [[0.5]], "metadatas": [[recs[hit]]]}
        assert r._vector_rows("api", raw) == ([row], [0.5]), hit


def test_records_without_a_row_are_dropped_and_rows_kept_once(tiny_dbs, tmp_path, monkeypatch, no_keys,
                                                               fake_vectors):
    api_db, sdk_db = tiny_dbs
    collection, records = fake_vectors
    recs = records("api", shift=3)
    recs["50"] = {"name": "Wall.Flip Method", "info": ""}            # a second record for row 4
    recs["51"] = {"name": "ElementId(Int64) Constructor", "info": ""}  # no such row
    r = _retriever(api_db, sdk_db, tmp_path, monkeypatch, collection(recs), collection(records("sdk")))
    flip_id = next(k for k, v in recs.items() if v["name"] == "Wall.Flip Method" and k != "50")
    raw = {"ids": [[flip_id, "51", "50"]], "distances": [[0.1, 0.2, 0.3]],
           "metadatas": [[recs[flip_id], recs["51"], recs["50"]]]}
    assert r._vector_rows("api", raw) == ([4], [0.1])
    # Without metadata the id is not trusted unless the probe found the ids aligned.
    assert r._vector_rows("api", {"ids": [["4"]], "distances": [[0.1]]}) == ([], [])


def test_two_retrievers_on_real_chromadb_collections(tiny_dbs, tmp_path, no_keys, fake_vectors):
    """Tiny ChromaDB stores (API ids misaligned, SDK ids aligned): the probe,
    vector queries through the real client, and a second retriever in the same
    process (each client gets its own Settings; a shared one made the second
    retriever fail with "An instance of Chroma already exists")."""
    chromadb = pytest.importorskip("chromadb")
    from chromadb.config import Settings

    api_db, sdk_db = tiny_dbs
    _, records = fake_vectors
    api_dir, code_dir = tmp_path / "chromadb_api", tmp_path / "chromadb_code"
    row_of = {name: rid for rid, name in _row_names(api_db).items()}

    def one_hot(i: int, n: int) -> list[float]:
        return [1.0 if j == i else 0.0 for j in range(n)]

    stores = ((api_dir, "revit_api", records("api", shift=2), lambda m: one_hot(row_of[m["name"]] - 1, 6)),
              (code_dir, "revit_sdk", records("sdk"), lambda m: one_hot(m["project"] == "FloorSlab", 2)))
    for path, name, recs, vector in stores:
        client = chromadb.PersistentClient(path=str(path), settings=Settings(anonymized_telemetry=False))
        coll = client.create_collection(name, embedding_function=None)
        coll.add(ids=list(recs), embeddings=[vector(m) for m in recs.values()], metadatas=list(recs.values()))

    probes = check_vector_stores(str(api_db), str(sdk_db), str(api_dir), str(code_dir))
    assert [(p.collection, p.status, p.sampled, p.id_matches, p.name_matches) for p in probes] == [
        ("revit_api", "misaligned", 6, 0, 6), ("revit_sdk", "aligned", 2, 2, 2)]

    floor = next(name for name in row_of if name.startswith("Floor.Create("))
    query_vec = one_hot(row_of[floor] - 1, 6)
    for _ in range(2):   # the second retriever opens the same paths again
        r = RAGRetriever(cfgmod.load_config(), str(api_db), str(sdk_db), str(api_dir), str(code_dir))
        r._embedder = _Embedder(query_vec)
        res = r.search("zzzz", api_top_k=3, code_top_k=0, rewrite=False, rerank=False)   # no keyword hit
        assert res.api_items[0].name == floor and res.api_items[0].chromadb_id == str(row_of[floor])
        raw = r._api_collection.query(query_embeddings=[query_vec], n_results=1, include=["metadatas"])
        assert raw["ids"][0][0] != str(row_of[floor])   # the record id names another row
        assert [p.status for p in r.vector_probes] == ["misaligned", "aligned"]


# -- R2: reciprocal rank fusion ----------------------------------------------------

def _item(cid: str) -> RetrievedItem:
    return RetrievedItem(source="api", chromadb_id=cid, distance=0.0, name=cid)


def test_rrf_fuse_scores_ties_and_distance():
    keyword = [_item(c) for c in ("k1", "both", "k3")]
    vector = [_item(c) for c in ("v1", "both", "v1", "v3")]   # a repeated entry counts once
    fused = rrf_fuse([keyword, vector])
    # "both" (1/62 + 1/62) first; then rank-1 ties, keyword before vector; then rank 3.
    assert [i.chromadb_id for i in fused] == ["both", "k1", "v1", "k3", "v3"]
    top = 2 / 61
    assert fused[0].distance == pytest.approx(1 - (2 / 62) / top)
    assert fused[1].distance == pytest.approx(1 - (1 / 61) / top) == pytest.approx(0.5)
    assert [i.distance for i in fused] == sorted(i.distance for i in fused)
    assert [i.chromadb_id for i in rrf_fuse([keyword, []])] == ["k1", "both", "k3"]
    assert rrf_fuse([]) == []


@needs_fts5
def test_hits_only_one_side_finds_both_reach_the_top_5(tiny_dbs, tmp_path, monkeypatch, no_keys, fake_vectors):
    """BM25 finds five wall entries, the vector search two entries BM25 does
    not find. Merged by distance (the old way) the vector hits ranked after
    every BM25 hit on the shipped data set (BM25 distances at most 0.12,
    vector distances near 1); fused by rank they alternate with the BM25 hits."""
    api_db, sdk_db = tiny_dbs
    conn = sqlite3.connect(api_db)
    conn.execute("INSERT INTO revit_api (id, name, full_id, summary) VALUES "
                 "(7, 'Level.Create Method', 'Level.Create', 'Creates a new level.')")
    conn.commit()
    conn.close()
    collection, records = fake_vectors
    recs = records("api")
    recs["7"] = {"name": "Level.Create Method", "info": ""}
    api = collection(recs, hits=["5", "7"])
    r = _retriever(api_db, sdk_db, tmp_path, monkeypatch, api, collection(records("sdk")))
    keyword = r._keyword_search_api("wall", limit=30)
    assert len(keyword) == 5
    assert {"5", "7"}.isdisjoint(i.chromadb_id for i in keyword)

    res = r.search("wall", api_top_k=5, code_top_k=0, rewrite=False, rerank=True)

    k = [i.chromadb_id for i in keyword]
    assert [i.chromadb_id for i in res.api_items] == [k[0], "5", k[1], "7", k[2]]
    assert [i.distance for i in res.api_items] == sorted(i.distance for i in res.api_items)


@needs_fts5
def test_without_vectors_the_result_is_the_bm25_order(tiny_dbs, tmp_path, no_keys):
    api_db, sdk_db = tiny_dbs
    r = _retriever(api_db, sdk_db, tmp_path)
    assert r.vector_search_enabled is False
    for query in ("wall", "Wall.Create", "create floor"):
        keyword = r._keyword_search_api(query, limit=30)
        res = r.search(query, api_top_k=30, code_top_k=0, rewrite=False, rerank=True)
        assert [(i.chromadb_id, i.distance) for i in res.api_items] == [
            (i.chromadb_id, i.distance) for i in keyword]


class _CountingProvider:
    name = "jev"
    calibrated = True

    def __init__(self):
        self.sizes: list[int] = []

    def rerank(self, query, documents, top_n=15, *, strict=False):
        self.sizes.append(len(documents))
        return [RerankResult(index=i, score=0.5, text=d) for i, d in enumerate(documents[:top_n])]


@needs_fts5
def test_the_reranker_gets_the_top_k_fused_candidates(tiny_dbs, tmp_path, monkeypatch, no_keys, fake_vectors):
    api_db, sdk_db = tiny_dbs
    collection, records = fake_vectors
    api = collection(records("api"), hits=["5", "1", "4"])
    r = _retriever(api_db, sdk_db, tmp_path, monkeypatch, api, collection(records("sdk")))
    r._api_top_k = 4
    provider = _CountingProvider()
    r._rerank_chain, r._reranker_initialized = [provider], True
    res = r.search("wall", api_top_k=3, code_top_k=0, rewrite=False, rerank=True)
    assert api.queries[-1]["n_results"] == 4 and provider.sizes == [4]   # 6 distinct entries fused, 4 kept
    assert len(res.api_items) == 3 and res.rerank_used == "jev"
