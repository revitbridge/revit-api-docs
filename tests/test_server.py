"""MCP surface: exactly two read-only tools, status messages before the data is ready,
and a real stdio round trip with a keyword-only retriever."""
from __future__ import annotations

import asyncio
import json
import os
import sys
from pathlib import Path

import pytest

from revit_api_docs import server

ROOT = Path(__file__).resolve().parents[1]


def _tools() -> dict:
    return {t.name: t for t in asyncio.run(server.mcp.list_tools())}


def test_exactly_two_read_only_tools():
    tools = _tools()
    assert set(tools) == {"search_revit_api", "get_code_examples"}
    for t in tools.values():
        assert t.annotations is not None and t.annotations.read_only_hint is True
        assert set(t.input_schema["properties"]) == {"query", "top_k"}
        assert t.input_schema["required"] == ["query"]


def test_tool_defaults_and_clamping():
    tools = _tools()
    assert tools["search_revit_api"].input_schema["properties"]["top_k"]["default"] == 10
    assert tools["get_code_examples"].input_schema["properties"]["top_k"]["default"] == 3
    assert server._clamp_top_k(0, 10) == 1
    assert server._clamp_top_k(999, 10) == server.MAX_TOP_K
    assert server._clamp_top_k("x", 10) == 10  # type: ignore[arg-type]


def test_empty_query_is_rejected_without_touching_data(monkeypatch):
    monkeypatch.setattr(server._state, "start_background", lambda: pytest.fail("must not start"))
    assert "Give a query" in asyncio.run(server.search_revit_api("  "))
    assert "Give a query" in asyncio.run(server.get_code_examples(""))


def test_status_messages_while_not_ready(monkeypatch):
    st = server._State()
    started = []
    monkeypatch.setattr(st, "start_background", lambda: started.append(1))
    monkeypatch.setattr(server, "_state", st)

    st.status = "downloading"
    st.detail = "chromadb_api.tar.gz 42%"
    msg = asyncio.run(server.search_revit_api("Wall.Create"))
    assert "downloading" in msg and "42%" in msg and started

    st.status = "loading"
    assert "loading its index" in asyncio.run(server.get_code_examples("wall"))

    st.status = "failed"
    st.error = "RuntimeError: disk full"
    msg = asyncio.run(server.search_revit_api("Wall.Create"))
    assert "not available: RuntimeError: disk full" in msg


def test_search_helpers_use_keyword_mode(tiny_dbs, monkeypatch, tmp_path):
    from revit_api_docs.config import load_config
    from revit_api_docs.retriever import RAGRetriever

    for name in ("REVIT_API_DOCS_EMBEDDING_API_KEY", "OPENROUTER_API_KEY", "COHERE_API_KEY"):
        monkeypatch.delenv(name, raising=False)
    api_db, sdk_db = tiny_dbs
    r = RAGRetriever(load_config(), str(api_db), str(sdk_db), str(tmp_path / "a"), str(tmp_path / "b"))
    text = server.search_api(r, "Wall.Create", 5)
    assert text.startswith("### Wall.Create" + chr(10))
    text = server.search_examples(r, "wall", 2)
    assert "Project: CreateWall" in text
    assert server.search_api(r, "qqqq", 5).startswith("No API entries found")


@pytest.mark.anyio
async def test_stdio_round_trip_with_keyword_only_data(tiny_dbs, tmp_path):
    """Spawn the real server as an MCP stdio subprocess against a tiny data dir."""
    from mcp import ClientSession, StdioServerParameters
    from mcp.client.stdio import stdio_client

    api_db, sdk_db = tiny_dbs
    root = tmp_path / "data"
    (root / "sqlite").mkdir(parents=True)
    os.replace(api_db, root / "sqlite" / "revit_api.db")
    os.replace(sdk_db, root / "sqlite" / "revit_sdk.db")
    for d in ("chromadb_api", "chromadb_code"):
        (root / "chromadb" / d).mkdir(parents=True)
        (root / "chromadb" / d / "chroma.sqlite3").write_bytes(b"")
    # A manifest for the current release means "installed and verified": only existence is checked.
    (root / "manifest.json").write_text(json.dumps({"release": server.data.RELEASE_TAG}), encoding="utf-8")

    env = {k: v for k, v in os.environ.items()
           if k not in ("REVIT_API_DOCS_EMBEDDING_API_KEY", "OPENROUTER_API_KEY", "COHERE_API_KEY")}
    env["REVIT_API_DOCS_DATA_DIR"] = str(root)
    params = StdioServerParameters(command=sys.executable, args=["-m", "revit_api_docs"], env=env, cwd=str(ROOT))

    async with stdio_client(params) as (read, write):
        async with ClientSession(read, write) as session:
            init = await session.initialize()
            assert init.server_info.name == "revit-api-docs"
            names = {t.name for t in (await session.list_tools()).tools}
            assert names == {"search_revit_api", "get_code_examples"}

            # The retriever is built in the background; poll until it answers.
            text = ""
            for _ in range(100):
                res = await session.call_tool("search_revit_api", {"query": "Wall.Create", "top_k": 3})
                text = res.content[0].text
                if text.startswith("###"):
                    break
                await asyncio.sleep(0.2)
            assert text.startswith("### Wall.Create" + chr(10))

            res = await session.call_tool("get_code_examples", {"query": "wall"})
            assert "Project: CreateWall" in res.content[0].text
