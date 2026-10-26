"""Rerank providers (Cohere, Jev), provider selection, fallback and abstention.

No network: every HTTP call goes to an httpx.MockTransport.
"""
from __future__ import annotations

import json
import logging
import threading
import time

import httpx
import pytest

from revit_api_docs import config as cfgmod
from revit_api_docs import server
from revit_api_docs.reranker import (
    JEV_URL,
    JevReranker,
    Reranker,
    RerankError,
    create_rerank_chain,
    create_reranker,
)
from revit_api_docs.retriever import RAGRetriever, rerank_label

COHERE_URL = "https://api.cohere.com/v2/rerank"


# -- helpers -----------------------------------------------------------------

def jev_transport(score, calls: list | None = None, *, status: int = 200):
    """MockTransport answering System One requests with score(doc_text) per noul question."""

    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        if calls is not None:
            calls.append((request, body))
        if status != 200:
            return httpx.Response(status, json={"detail": "error"})
        docs = body["state"]["documents"]
        answers = {qid: {"type": "noul", "noul": score(docs[qid])} for qid in body["questions"]}
        return httpx.Response(200, json={"model": "jev-1.14.0", "answers": answers,
                                         "usage": {"input_tokens": 100, "output_tokens": 0}})

    return httpx.MockTransport(handler)


def cohere_transport(scores: list[float], calls: list | None = None, *, status: int = 200):
    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        if calls is not None:
            calls.append((request, body))
        if status != 200:
            return httpx.Response(status, text="boom")
        n = len(body["documents"])
        results = [{"index": i, "relevance_score": scores[i] if i < len(scores) else 0.0} for i in range(n)]
        results.sort(key=lambda r: -r["relevance_score"])
        return httpx.Response(200, json={"results": results[:body["top_n"]]})

    return httpx.MockTransport(handler)


def jev_position_transport(scores: list[float]):
    """MockTransport giving the i-th candidate of the request the score scores[i]."""

    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        answers = {qid: {"type": "noul", "noul": scores[int(qid[1:])]} for qid in body["questions"]}
        return httpx.Response(200, json={"model": "jev-1.14.0", "answers": answers})

    return httpx.MockTransport(handler)


def jev(transport, **kw) -> JevReranker:
    kw.setdefault("sleep", lambda s: None)
    return JevReranker(transport=transport, **kw)


@pytest.fixture
def jev_key(monkeypatch):
    monkeypatch.setenv("TYPESAFE_API_KEY", "test-typesafe-key")


@pytest.fixture
def cohere_key(monkeypatch):
    monkeypatch.setenv("COHERE_API_KEY", "test-cohere-key")


# -- Jev provider ------------------------------------------------------------

def test_jev_request_follows_the_system_one_api(jev_key):
    calls: list = []
    docs = ["Wall.Create: Creates a wall", "Floor.Create: Creates a floor", "Wall.Flip: Flips"]
    r = jev(jev_transport(lambda d: 0.9 if d.startswith("Wall.Create") else 0.2, calls))

    out = r.rerank("create wall", docs, top_n=2)

    assert [(x.index, x.score) for x in out] == [(0, 0.9), (1, 0.2)]  # sorted, cut to top_n
    assert len(calls) == 1
    request, body = calls[0]
    assert str(request.url) == JEV_URL == "https://api.typesafe.ai/v1/systemone"
    assert request.method == "POST"
    assert request.headers["authorization"] == "Bearer test-typesafe-key"
    assert request.headers["content-type"] == "application/json"
    assert body["model"] == "jev-latest"
    assert body["state"]["query"] == "Revit 2026 API: create wall"
    assert body["state"]["documents"] == {"d0": docs[0], "d1": docs[1], "d2": docs[2]}
    assert set(body["questions"]) == {"d0", "d1", "d2"}
    for qid, q in body["questions"].items():
        assert q["type"] == "noul"
        assert f'"{qid}"' in q["instructions"]
        assert set(q["criteria"]) == {"true", "false"}


def test_jev_splits_31_documents_into_two_calls(jev_key):
    calls: list = []
    docs = [f"doc{i}" for i in range(31)]
    r = jev(jev_transport(lambda d: int(d[3:]) / 100, calls))

    out = r.rerank("q", docs, top_n=31)

    assert sorted(len(body["questions"]) for _, body in calls) == [1, 30]
    # indices map back to the input list across batches; doc30 scores highest
    assert out[0].index == 30 and out[0].text == "doc30" and out[0].score == pytest.approx(0.30)
    assert [x.index for x in out] == list(range(30, -1, -1))


def test_jev_keeps_at_most_max_concurrency_requests_in_flight(jev_key):
    lock = threading.Lock()
    state = {"now": 0, "peak": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        with lock:
            state["now"] += 1
            state["peak"] = max(state["peak"], state["now"])
        time.sleep(0.05)
        with lock:
            state["now"] -= 1
        body = json.loads(request.content)
        return httpx.Response(200, json={"answers": {q: {"type": "noul", "noul": 0.5} for q in body["questions"]}})

    r = jev(httpx.MockTransport(handler), max_concurrency=2)
    assert len(r.rerank("q", [f"d{i}" for i in range(120)], top_n=5)) == 5  # 4 batches
    assert state["peak"] == 2


def test_jev_defaults():
    r = JevReranker()
    assert (r._timeout, r._max_retries, r._max_concurrency, r._batch_size) == (20.0, 2, 2, 30)
    assert JevReranker(batch_size=100)._batch_size == 30  # never more than 30 per call


def test_jev_retries_with_backoff_then_succeeds(jev_key):
    statuses = iter([429, 529])
    sleeps: list[float] = []

    def handler(request: httpx.Request) -> httpx.Response:
        code = next(statuses, 200)
        if code != 200:
            return httpx.Response(code, text="busy")
        body = json.loads(request.content)
        return httpx.Response(200, json={"answers": {q: {"type": "noul", "noul": 0.4} for q in body["questions"]}})

    r = JevReranker(transport=httpx.MockTransport(handler), sleep=sleeps.append)
    out = r.rerank("q", ["a", "b"], top_n=2)
    assert [x.score for x in out] == [0.4, 0.4]
    assert sleeps == [0.5, 1.0]  # exponential backoff, two retries


def test_jev_honours_retry_after(jev_key):
    statuses = iter([429])
    sleeps: list[float] = []

    def handler(request: httpx.Request) -> httpx.Response:
        if next(statuses, 200) == 429:
            return httpx.Response(429, headers={"Retry-After": "3"})
        body = json.loads(request.content)
        return httpx.Response(200, json={"answers": {q: {"type": "noul", "noul": 0.4} for q in body["questions"]}})

    JevReranker(transport=httpx.MockTransport(handler), sleep=sleeps.append).rerank("q", ["a"])
    assert sleeps == [3.0]


def test_jev_gives_up_after_two_retries(jev_key):
    calls: list = []
    r = jev(jev_transport(lambda d: 0.5, calls, status=529))
    with pytest.raises(RerankError, match="529.*3 attempts"):
        r.rerank("q", ["a", "b"], strict=True)
    assert len(calls) == 3
    # non-strict keeps the historical behaviour: input order, no exception
    assert [x.index for x in r.rerank("q", ["a", "b"], top_n=2)] == [0, 1]


def test_jev_does_not_retry_client_errors(jev_key):
    calls: list = []
    r = jev(jev_transport(lambda d: 0.5, calls, status=401))
    with pytest.raises(RerankError, match="HTTP 401"):
        r.rerank("q", ["a"], strict=True)
    assert len(calls) == 1


def test_jev_retries_timeouts(jev_key):
    calls: list = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(request)
        raise httpx.ReadTimeout("timed out", request=request)

    with pytest.raises(RerankError, match="ReadTimeout"):
        jev(httpx.MockTransport(handler)).rerank("q", ["a"], strict=True)
    assert len(calls) == 3


@pytest.mark.parametrize("answer", [
    {},                                        # question missing
    {"d0": {"type": "noul"}},                  # no probability
    {"d0": {"type": "noul", "noul": 1.5}},     # not a probability
    {"d0": {"type": "noul", "noul": True}},    # not a number
])
def test_jev_rejects_malformed_answers(jev_key, answer):
    t = httpx.MockTransport(lambda req: httpx.Response(200, json={"answers": answer}))
    with pytest.raises(RerankError):
        jev(t).rerank("q", ["a"], strict=True)


class _CountingClient(httpx.Client):
    """httpx.Client that records each construction and close (the real transport is injected)."""

    created: list["_CountingClient"] = []

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        _CountingClient.created.append(self)
        self.closed_by_us = False

    def close(self):
        self.closed_by_us = True
        super().close()


@pytest.fixture
def counting_client(monkeypatch):
    _CountingClient.created = []
    monkeypatch.setattr(httpx, "Client", _CountingClient)
    return _CountingClient.created


def test_jev_reuses_one_client_across_calls(jev_key, counting_client):
    calls: list = []
    r = jev(jev_transport(lambda d: 0.5, calls))
    assert counting_client == []  # created lazily, not in __init__
    r.score("q", ["a", "b"])
    r.score("q", ["c"])
    r.rerank("q", [f"d{i}" for i in range(61)], top_n=3)  # 3 batches from worker threads
    assert len(counting_client) == 1
    assert len(calls) == 5  # the injected transport still serves every request
    r.close()
    assert counting_client[0].closed_by_us and counting_client[0].is_closed
    r.score("q", ["a"])  # a call after close() opens a new client
    assert len(counting_client) == 2 and not counting_client[1].is_closed
    r.close()
    r.close()  # idempotent


def test_cohere_reuses_one_client_across_calls(cohere_key, counting_client):
    calls: list = []
    r = Reranker(transport=cohere_transport([0.1, 0.8, 0.3], calls))
    r.rerank("q", ["a", "b", "c"], top_n=2)
    r.rerank("q", ["a", "b"], top_n=2)
    assert len(counting_client) == 1 and len(calls) == 2
    r.close()
    assert counting_client[0].is_closed


def test_jev_without_key_never_calls_out():
    calls: list = []
    r = jev(jev_transport(lambda d: 0.5, calls))
    with pytest.raises(RerankError, match="TYPESAFE_API_KEY"):
        r.rerank("q", ["a"], strict=True)
    assert [x.index for x in r.rerank("q", ["a", "b"], top_n=1)] == [0]
    assert calls == []


# -- Cohere provider keeps its behaviour ------------------------------------

def test_cohere_request_and_order(cohere_key):
    calls: list = []
    r = Reranker(transport=cohere_transport([0.1, 0.8, 0.3], calls))
    out = r.rerank("create wall", ["a", "b", "c"], top_n=2)
    assert [(x.index, x.score) for x in out] == [(1, 0.8), (2, 0.3)]
    request, body = calls[0]
    assert str(request.url) == COHERE_URL
    assert body["query"] == "create wall"  # no version prefix: unchanged request
    assert body["top_n"] == 2 and body["model"] == "rerank-v3.5"
    assert r.name == "cohere" and r.calibrated is False


def test_cohere_failure_falls_back_to_input_order_unless_strict(cohere_key):
    r = Reranker(transport=cohere_transport([], status=500))
    assert [x.index for x in r.rerank("q", ["a", "b", "c"], top_n=2)] == [0, 1]
    with pytest.raises(RerankError, match="HTTP 500"):
        r.rerank("q", ["a"], strict=True)


# -- provider selection -------------------------------------------------------

def _names(chain) -> list[str]:
    return [p.name for p in chain]


def test_default_provider_is_cohere(cohere_key):
    cfg = cfgmod.load_config()
    assert cfg["rerank"]["provider"] == "cohere"
    assert cfg["rerank"]["models"]["jev"] == {
        "model": "jev-latest", "api_key_env": "TYPESAFE_API_KEY",
        "batch_size": 30, "max_concurrency": 2, "max_retries": 2, "timeout": 20,
    }
    assert cfg["retrieval"]["api"]["min_relevance"] is None
    assert _names(create_rerank_chain(cfg)) == ["cohere"]


def test_no_key_means_no_rerank(caplog):
    caplog.set_level(logging.WARNING, logger="revit_api_docs.reranker")
    assert create_rerank_chain(cfgmod.load_config()) == []
    assert "COHERE_API_KEY not set" in caplog.text


@pytest.mark.parametrize("keys, expected", [
    ((), []),
    (("TYPESAFE_API_KEY",), ["jev"]),
    (("COHERE_API_KEY",), ["cohere"]),                 # jev unavailable -> cohere
    (("TYPESAFE_API_KEY", "COHERE_API_KEY"), ["jev", "cohere"]),
])
def test_jev_provider_chain(monkeypatch, keys, expected):
    monkeypatch.setenv("REVIT_API_DOCS_RERANK_PROVIDER", "jev")
    for k in keys:
        monkeypatch.setenv(k, "x")
    assert _names(create_rerank_chain(cfgmod.load_config())) == expected


def test_provider_none_and_unknown(monkeypatch, cohere_key, caplog):
    monkeypatch.setenv("REVIT_API_DOCS_RERANK_PROVIDER", "none")
    assert create_rerank_chain(cfgmod.load_config()) == []
    assert create_reranker(cfgmod.load_config()) is None
    monkeypatch.setenv("REVIT_API_DOCS_RERANK_PROVIDER", "voyage")
    caplog.set_level(logging.WARNING, logger="revit_api_docs.reranker")
    assert create_rerank_chain(cfgmod.load_config()) == []
    assert "unknown provider='voyage'" in caplog.text


def test_jev_settings_come_from_config(monkeypatch, jev_key):
    monkeypatch.setenv("REVIT_API_DOCS_RERANK_PROVIDER", "JEV")
    monkeypatch.setenv("REVIT_API_DOCS_JEV_MODEL", "jev-1.14.0")
    monkeypatch.setenv("REVIT_API_DOCS_MIN_RELEVANCE", "0.25")
    cfg = cfgmod.load_config()
    assert cfg["rerank"]["provider"] == "jev"
    assert cfg["retrieval"]["api"]["min_relevance"] == "0.25"
    provider = create_reranker(cfg)
    assert isinstance(provider, JevReranker)
    assert provider._model == "jev-1.14.0" and provider.calibrated is True
    assert provider.query_text("Wall.Create") == "Revit 2026 API: Wall.Create"


# -- retriever: rerank_used, fallback, abstention ----------------------------

def _retriever(tiny_dbs, tmp_path, monkeypatch, *, provider="jev", min_relevance=None, chain=()):
    for name in ("REVIT_API_DOCS_EMBEDDING_API_KEY", "OPENROUTER_API_KEY", "COHERE_API_KEY"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("REVIT_API_DOCS_RERANK_PROVIDER", provider)
    if min_relevance is not None:
        monkeypatch.setenv("REVIT_API_DOCS_MIN_RELEVANCE", min_relevance)
    api_db, sdk_db = tiny_dbs
    r = RAGRetriever(cfgmod.load_config(), str(api_db), str(sdk_db), str(tmp_path / "a"), str(tmp_path / "b"))
    r._rerank_chain = list(chain)
    r._reranker_initialized = True
    return r


def _wall_create_first(doc: str) -> float:
    return 0.91 if doc.startswith("Wall.Create:") else 0.12


def test_jev_orders_results_and_reports_relevance(tiny_dbs, tmp_path, monkeypatch, jev_key):
    r = _retriever(tiny_dbs, tmp_path, monkeypatch, chain=[jev(jev_transport(_wall_create_first))])
    res = r.search("wall", api_top_k=3, code_top_k=0, rewrite=False, rerank=True)
    assert res.rerank_used == "jev" and res.rerank_skipped == [] and not res.abstained
    assert res.api_items[0].full_id == "Wall.Create"
    assert res.api_items[0].relevance == pytest.approx(0.91)
    assert len(res.api_items) == 3

    text = server.search_api(r, "wall", 3)
    lines = text.splitlines()
    assert lines[0] == "Rerank: jev" and lines[1] == ""
    assert lines[2] == "### Wall.Create" and lines[3] == "Relevance: 0.91"
    assert r.format_results(res).splitlines()[0] == "rerank:    jev"


def test_jev_failure_falls_back_to_cohere(tiny_dbs, tmp_path, monkeypatch, jev_key, cohere_key, caplog):
    jev_calls: list = []
    chain = [jev(jev_transport(lambda d: 0.5, jev_calls, status=500)),
             Reranker(transport=cohere_transport([0.0, 0.0, 0.9, 0.1]))]
    r = _retriever(tiny_dbs, tmp_path, monkeypatch, min_relevance="0.5", chain=chain)
    caplog.set_level(logging.WARNING, logger="revit_api_docs.retriever")

    res = r.search("wall", api_top_k=2, code_top_k=0, rewrite=False, rerank=True)

    assert len(jev_calls) == 3  # first try + two retries
    assert res.rerank_used == "cohere" and res.rerank_skipped == ["jev"]
    assert rerank_label(res) == "cohere (unavailable: jev)"
    assert not res.abstained  # Cohere scores are not calibrated: no abstention
    assert all(i.relevance is None for i in res.api_items) and len(res.api_items) == 2
    assert "rerank provider jev failed" in caplog.text
    text = server.search_api(r, "wall", 2)
    assert text.splitlines()[0] == "Rerank: cohere (unavailable: jev)"
    assert "Relevance:" not in text


def test_jev_failure_without_cohere_keeps_retrieval_order(tiny_dbs, tmp_path, monkeypatch, jev_key):
    r = _retriever(tiny_dbs, tmp_path, monkeypatch, min_relevance="0.9",
                   chain=[jev(jev_transport(lambda d: 0.5, status=401))])
    plain = _retriever(tiny_dbs, tmp_path, monkeypatch, provider="none")
    res = r.search("Wall.Create", api_top_k=3, code_top_k=0, rewrite=False, rerank=True)
    expected = plain.search("Wall.Create", api_top_k=3, code_top_k=0, rewrite=False, rerank=True)
    assert res.rerank_used == "none" and res.rerank_skipped == ["jev"] and not res.abstained
    assert [i.chromadb_id for i in res.api_items] == [i.chromadb_id for i in expected.api_items]
    assert server.search_api(r, "Wall.Create", 3).splitlines()[0] == "Rerank: none (unavailable: jev)"


def test_jev_selected_without_a_key_is_reported(tiny_dbs, tmp_path, monkeypatch):
    api_db, sdk_db = tiny_dbs
    monkeypatch.setenv("REVIT_API_DOCS_RERANK_PROVIDER", "jev")
    monkeypatch.delenv("COHERE_API_KEY", raising=False)
    r = RAGRetriever(cfgmod.load_config(), str(api_db), str(sdk_db), str(tmp_path / "a"), str(tmp_path / "b"))
    res = r.search("Wall.Create", api_top_k=3, code_top_k=0, rewrite=False, rerank=True)
    assert res.rerank_used == "none" and res.rerank_skipped == ["jev"]


def test_abstains_when_every_candidate_is_below_the_threshold(tiny_dbs, tmp_path, monkeypatch, jev_key):
    r = _retriever(tiny_dbs, tmp_path, monkeypatch, min_relevance="0.2",
                   chain=[jev(jev_transport(lambda d: 0.07))])
    res = r.search("wall", api_top_k=3, code_top_k=0, rewrite=False, rerank=True)
    assert res.abstained is True
    assert res.abstain_reason == "best relevance 0.07 < 0.20"
    assert len(res.api_items) == 3  # low-scoring candidates stay for reference

    lines = server.search_api(r, "wall", 3).splitlines()
    # Neutral: a low best score may be a missing member or a recall miss.
    assert lines[0] == (
        "No entry in the Revit 2026 data set scored high enough for this query (best relevance "
        "0.07 < 0.20). This can mean the member is absent from this version, or that the data set "
        "has it under a different name or wording; check the name and version, or rephrase.")
    assert "absent from this version" in lines[0] and "different name or wording" in lines[0]
    assert "may not exist" not in lines[0]
    assert lines[1].startswith("Rerank: jev.") and "for reference only" in lines[1]
    assert lines.count("Relevance: 0.07") == 3
    assert r.format_results(res).splitlines()[1] == "abstained: best relevance 0.07 < 0.20"


def test_threshold_keeps_low_candidates_with_their_relevance(tiny_dbs, tmp_path, monkeypatch, jev_key):
    """The threshold only decides abstention: once one candidate reaches it,
    the ones below it are still returned, each with its relevance."""
    r = _retriever(tiny_dbs, tmp_path, monkeypatch, min_relevance="0.2",
                   chain=[jev(jev_transport(_wall_create_first))])
    res = r.search("wall", api_top_k=5, code_top_k=0, rewrite=False, rerank=True)
    assert not res.abstained and res.abstain_reason == ""
    assert res.api_items[0].full_id == "Wall.Create"
    assert res.api_items[0].relevance == pytest.approx(0.91)
    below = res.api_items[1:]
    assert len(below) == 4 and all(i.relevance == pytest.approx(0.12) for i in below)

    lines = server.search_api(r, "wall", 5).splitlines()
    assert lines[0] == "Rerank: jev" and lines[1] == ""
    assert lines[2] == "### Wall.Create" and lines[3] == "Relevance: 0.91"
    assert lines.count("Relevance: 0.12") == 4


@pytest.mark.parametrize("scores, abstains", [
    ([0.9, 0.3, 0.1], False),
    ([0.3, 0.1], True),
])
def test_threshold_decides_abstention_only(tiny_dbs, tmp_path, monkeypatch, jev_key, scores, abstains):
    """Threshold 0.5: [0.9, 0.3, 0.1] returns all three without abstaining;
    [0.3, 0.1] abstains and keeps both candidates for reference."""
    r = _retriever(tiny_dbs, tmp_path, monkeypatch, min_relevance="0.5",
                   chain=[jev(jev_position_transport(scores))])
    keyword = r._keyword_search_api
    monkeypatch.setattr(r, "_keyword_search_api", lambda q, limit: keyword(q, limit=limit)[:len(scores)])

    res = r.search("wall", api_top_k=5, code_top_k=0, rewrite=False, rerank=True)
    assert res.rerank_used == "jev" and res.abstained is abstains
    assert [i.relevance for i in res.api_items] == pytest.approx(scores)
    assert res.abstain_reason == ("best relevance 0.30 < 0.50" if abstains else "")

    lines = server.search_api(r, "wall", 5).splitlines()
    assert [ln for ln in lines if ln.startswith("Relevance: ")] == [f"Relevance: {s:.2f}" for s in scores]
    if abstains:
        assert lines[0].startswith("No entry in the Revit 2026 data set scored high enough")
        assert lines[1] == "Rerank: jev. Low-relevance candidates, for reference only:"
    else:
        assert lines[0] == "Rerank: jev" and lines[1] == ""


def test_no_threshold_never_abstains(tiny_dbs, tmp_path, monkeypatch, jev_key):
    r = _retriever(tiny_dbs, tmp_path, monkeypatch, chain=[jev(jev_transport(lambda d: 0.01))])
    res = r.search("wall", api_top_k=3, code_top_k=0, rewrite=False, rerank=True)
    assert res.rerank_used == "jev" and not res.abstained and len(res.api_items) == 3


def test_min_relevance_has_no_effect_with_cohere(tiny_dbs, tmp_path, monkeypatch, cohere_key):
    r = _retriever(tiny_dbs, tmp_path, monkeypatch, provider="cohere", min_relevance="0.9",
                   chain=[Reranker(transport=cohere_transport([0.01, 0.02, 0.03]))])
    res = r.search("wall", api_top_k=3, code_top_k=0, rewrite=False, rerank=True)
    assert res.rerank_used == "cohere" and not res.abstained
    assert len(res.api_items) == 3 and all(i.relevance is None for i in res.api_items)
    assert server.search_api(r, "wall", 3).splitlines()[0] == "Rerank: cohere"


@pytest.mark.parametrize("value, expected", [
    ("0.3", 0.3), ("0", 0.0), ("1", 1.0), ("none", None), ("null", None), ("abc", None), ("1.5", None),
])
def test_min_relevance_values(tiny_dbs, tmp_path, monkeypatch, value, expected):
    r = _retriever(tiny_dbs, tmp_path, monkeypatch, min_relevance=value)
    assert r.min_relevance == expected


def test_search_examples_never_reranks(tiny_dbs, tmp_path, monkeypatch, jev_key):
    calls: list = []
    r = _retriever(tiny_dbs, tmp_path, monkeypatch, min_relevance="0.9",
                   chain=[jev(jev_transport(lambda d: 0.0, calls))])
    assert "Project: CreateWall" in server.search_examples(r, "wall", 2)
    assert calls == []
