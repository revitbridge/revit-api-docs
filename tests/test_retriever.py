"""Retriever behaviour that needs no vector store: keyword search, context assembly, config."""
from __future__ import annotations

import logging

import pytest

from revit_api_docs import config as cfgmod
from revit_api_docs.retriever import RAGRetriever, SearchResults


@pytest.fixture
def keyword_only_retriever(tiny_dbs, monkeypatch, tmp_path):
    api_db, sdk_db = tiny_dbs
    # No embedding key anywhere: the retriever must fall back to SQLite keyword search.
    for name in ("REVIT_API_DOCS_EMBEDDING_API_KEY", "OPENROUTER_API_KEY", "COHERE_API_KEY"):
        monkeypatch.delenv(name, raising=False)
    cfg = cfgmod.load_config()
    return RAGRetriever(cfg, str(api_db), str(sdk_db), str(tmp_path / "no_api"), str(tmp_path / "no_code"))


def test_keyword_only_mode_without_key(keyword_only_retriever, caplog):
    r = keyword_only_retriever
    assert r.vector_search_enabled is False
    assert r._api_collection is None  # chromadb never opened


def test_search_wall_create_ranks_exact_identifier_first(keyword_only_retriever):
    results = keyword_only_retriever.search("Wall.Create", api_top_k=5, code_top_k=0, rewrite=False)
    names = [i.name for i in results.api_items]
    # exact identifier (overview + overloads) before the longer Wall.CreateProfileSketch
    assert names[:2] == ["Wall.Create Method", "Wall.Create(Document, Curve, ElementId, Boolean) Method"]
    assert names.index("Wall.CreateProfileSketch Method") > 1
    assert results.sdk_items == []


def test_top_k_zero_skips_a_source(keyword_only_retriever):
    results = keyword_only_retriever.search("wall", api_top_k=0, code_top_k=2, rewrite=False)
    assert results.api_items == []
    assert [i.project for i in results.sdk_items] == ["CreateWall"]


def test_sdk_keyword_fallback_matches_project_and_apis(keyword_only_retriever):
    results = keyword_only_retriever.search("floor boundary", api_top_k=0, code_top_k=3, rewrite=False)
    assert results.sdk_items[0].project == "FloorSlab"
    assert "Floor.Create" in results.sdk_items[0].content


def test_build_context_formats_api_and_code(keyword_only_retriever):
    r = keyword_only_retriever
    results = r.search("Wall.Create", api_top_k=2, code_top_k=1, rewrite=False)
    ctx = r.build_context(results)
    assert ctx["api_context"].startswith("### Wall.Create" + chr(10) + "Creates a wall within the project")
    assert chr(10) * 2 + "### M:Autodesk.Revit.DB.Wall.Create(" in ctx["api_context"]
    assert "Syntax: public static Wall Create(Document document" in ctx["api_context"]
    assert "Parameters: document:" in ctx["api_context"]
    assert ctx["code_context"].startswith("// Project: CreateWall  |  APIs: [")
    assert "Wall.Create(doc, line, levelId, false)" in ctx["code_context"]


def test_no_match_gives_empty_context(keyword_only_retriever):
    r = keyword_only_retriever
    results = r.search("zzzz qqqq", api_top_k=5, code_top_k=2, rewrite=False)
    assert results.api_items == [] and results.sdk_items == []
    assert r.build_context(results) == {"api_context": "", "code_context": ""}


def test_query_rewrite_is_skipped_without_llm(keyword_only_retriever, caplog):
    r = keyword_only_retriever
    caplog.set_level(logging.INFO, logger="revit_api_docs.retriever")
    assert r.rewrite_query("create wall") == "create wall"
    assert r._rewrite_unavailable is True
    # second call must not try again
    assert r.rewrite_query("create wall") == "create wall"


def test_like_wildcards_are_escaped(keyword_only_retriever):
    results = keyword_only_retriever.search("wall_%", api_top_k=5, code_top_k=0, rewrite=False)
    # "wall_%" must not act as a wildcard that matches everything
    assert all("wall" in i.name.lower() for i in results.api_items)


def test_search_results_dataclass_defaults():
    sr = SearchResults(query="q")
    assert sr.api_items == [] and sr.sdk_items == [] and sr.rewritten_query == ""


# ── config ──────────────────────────────────────────────────────────────────

def test_env_overrides_embedding(monkeypatch):
    monkeypatch.setenv("REVIT_API_DOCS_EMBEDDING_API_KEY", "x")
    monkeypatch.setenv("REVIT_API_DOCS_EMBEDDING_BASE_URL", "https://api.openai.com/v1")
    monkeypatch.setenv("REVIT_API_DOCS_EMBEDDING_MODEL", "text-embedding-3-large")
    cfg = cfgmod.load_config()
    m = cfg["embedding"]["models"]["openai"]
    assert m["api_key_env"] == "REVIT_API_DOCS_EMBEDDING_API_KEY"
    assert m["base_url"] == "https://api.openai.com/v1"
    assert m["model"] == "text-embedding-3-large"


def test_openrouter_key_is_a_fallback(monkeypatch):
    monkeypatch.delenv("REVIT_API_DOCS_EMBEDDING_API_KEY", raising=False)
    monkeypatch.setenv("OPENROUTER_API_KEY", "x")
    cfg = cfgmod.load_config()
    assert cfg["embedding"]["models"]["openai"]["api_key_env"] == "OPENROUTER_API_KEY"
    assert cfg["embedding"]["models"]["openai"]["base_url"] == "https://openrouter.ai/api/v1"


def test_default_config_has_pipeline_sections():
    cfg = cfgmod.load_config()
    assert set(cfg["llm"]["models"]) >= {"claude", "gemini_flash"}
    assert cfg["sdk"]["stage1_provider"] == "gemini_flash"
    assert cfg["rerank"]["models"]["cohere"]["api_key_env"] == "COHERE_API_KEY"
    # never a key value in the shipped config
    text = cfgmod._DEFAULT_CONFIG.read_text(encoding="utf-8")
    assert "sk-" not in text and "api_key:" not in text


# ── degraded vector search ──────────────────────────────────────────────────

class _BrokenEmbedder:
    calls = 0

    def embed_query(self, query):
        _BrokenEmbedder.calls += 1
        raise RuntimeError("401 API key expired")


def test_embedding_failure_degrades_to_keyword_search(keyword_only_retriever, monkeypatch, caplog):
    r = keyword_only_retriever
    r._embedder = _BrokenEmbedder()
    monkeypatch.setattr(r, "_open_collections", lambda: None)
    caplog.set_level(logging.WARNING, logger="revit_api_docs.retriever")

    for _ in range(3):
        results = r.search("Wall.Create", api_top_k=3, code_top_k=1, rewrite=False)
        assert results.api_items[0].name == "Wall.Create Method"
        assert results.sdk_items[0].project == "CreateWall"
    assert "vector search failed" in caplog.text
    assert r._embedder is None and "disabled for this process" in caplog.text
    assert _BrokenEmbedder.calls == 3
    # fourth query: keyword only, no further embedding attempts
    r.search("Wall.Create", api_top_k=3, code_top_k=0, rewrite=False)
    assert _BrokenEmbedder.calls == 3


# ── candidate selection ─────────────────────────────────────────────────────

def test_exact_name_match_is_found_beyond_the_candidate_window(tiny_dbs, monkeypatch, tmp_path):
    """Hundreds of summary-only hits with lower rowids must not push an exact
    name match out of the SQL candidate set (the old LIMIT had no ORDER BY)."""
    import sqlite3

    api_db, sdk_db = tiny_dbs
    conn = sqlite3.connect(api_db)
    conn.execute("DELETE FROM revit_api")
    conn.executemany(
        "INSERT INTO revit_api (name, full_id, summary) VALUES (?,?,?)",
        [(f"Thing{i}.Get Method", f"M:Thing{i}.Get", "Returns the floor of this thing.") for i in range(400)],
    )
    conn.execute(
        "INSERT INTO revit_api (name, full_id, summary) VALUES (?,?,?)",
        ("Floor.Create Method", "Floor.Create", "Creates a floor from a boundary."),
    )
    conn.commit()
    conn.close()
    for name in ("REVIT_API_DOCS_EMBEDDING_API_KEY", "OPENROUTER_API_KEY", "COHERE_API_KEY"):
        monkeypatch.delenv(name, raising=False)
    r = RAGRetriever(cfgmod.load_config(), str(api_db), str(sdk_db), str(tmp_path / "a"), str(tmp_path / "b"))
    monkeypatch.setattr(RAGRetriever, "_KEYWORD_CANDIDATES", 50)

    results = r.search("create floor from boundary", api_top_k=3, code_top_k=0, rewrite=False)
    assert [i.name for i in results.api_items][0] == "Floor.Create Method"
    # the window is a cap on ranked rows, not a scan limit
    assert len(r._keyword_candidates(r._normalize_search_tokens("floor"))) == 50
