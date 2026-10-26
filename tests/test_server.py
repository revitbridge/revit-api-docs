"""MCP surface: exactly two read-only tools, status messages before the data is ready,
and a real stdio round trip with a keyword-only retriever."""
from __future__ import annotations

import asyncio
import logging
import socket
import subprocess
import sys
import time
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


def test_tool_descriptions_say_queries_are_english():
    for t in _tools().values():
        assert "English" in t.description, t.name
    assert "English only" in server.SERVER_INSTRUCTIONS


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


def test_failed_state_claims_a_retry_only_when_one_was_started(monkeypatch):
    import threading

    st = server._State()
    monkeypatch.setattr(server, "_state", st)
    st.status = "failed"
    st.error = "RuntimeError: disk full"

    # A finished (dead) thread: this call starts a new one and says so.
    dead = threading.Thread(target=lambda: None)
    dead.start()
    dead.join()
    st._thread = dead
    started: list[threading.Thread] = []
    monkeypatch.setattr(threading.Thread, "start", lambda self: started.append(self))
    msg = asyncio.run(server.search_revit_api("Wall.Create"))
    assert "Retrying in the background" in msg and len(started) == 1
    assert st._thread is started[0]

    # A retry thread still alive: no new thread, no "retrying" claim.
    class _Alive:
        def is_alive(self):
            return True

    st._thread = _Alive()  # type: ignore[assignment]
    msg = asyncio.run(server.get_code_examples("wall"))
    assert "Retrying in the background" not in msg
    assert "A retry is already running" in msg and len(started) == 1


def test_search_helpers_use_keyword_mode(tiny_dbs, monkeypatch, tmp_path):
    from revit_api_docs.config import load_config
    from revit_api_docs.retriever import RAGRetriever

    for name in ("REVIT_API_DOCS_EMBEDDING_API_KEY", "OPENROUTER_API_KEY", "COHERE_API_KEY"):
        monkeypatch.delenv(name, raising=False)
    api_db, sdk_db = tiny_dbs
    r = RAGRetriever(load_config(), str(api_db), str(sdk_db), str(tmp_path / "a"), str(tmp_path / "b"))
    text = server.search_api(r, "Wall.Create", 5)
    # First line names the reranker (none without a key) and the data set, then the entries.
    assert text.startswith("Rerank: none; data set: Revit 2026" + chr(10) * 2 + "### Wall.Create" + chr(10))
    assert "Environment notes" not in text  # an API query gets no environment notes
    text = server.search_examples(r, "wall", 2)
    assert "Project: CreateWall" in text
    assert server.search_api(r, "qqqq", 5).startswith("No API entries found")



def test_the_first_line_names_the_data_set_of_the_retriever(tiny_dbs, monkeypatch, tmp_path):
    """The data set version on the Rerank line is the retriever's revit_version (config)."""
    from revit_api_docs.config import load_config
    from revit_api_docs.retriever import RAGRetriever

    for name in ("REVIT_API_DOCS_EMBEDDING_API_KEY", "OPENROUTER_API_KEY", "COHERE_API_KEY"):
        monkeypatch.delenv(name, raising=False)
    api_db, sdk_db = tiny_dbs
    config = load_config()
    config["revit_version"] = "2027"
    r = RAGRetriever(config, str(api_db), str(sdk_db), str(tmp_path / "a"), str(tmp_path / "b"))
    assert r.revit_version == "2027"
    first = server.search_api(r, "Wall.Create", 3).splitlines()[0]
    assert first == "Rerank: none; data set: Revit 2027"
    assert first.startswith("Rerank: ")

# -- CLI subcommands --------------------------------------------------------

def _cli(args: list[str], env: dict[str, str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run([sys.executable, "-m", "revit_api_docs", *args],
                          capture_output=True, text=True, env=env, cwd=str(ROOT))


def test_search_stderr_holds_only_the_warnings_that_matter(tiny_data_dir, no_key_env):
    """Plain `logger: message` lines at WARNING; the INFO chatter (data dir,
    keyword search details, httpx requests) and the index/content mismatch
    of the shipped data stay off the terminal."""
    env = dict(no_key_env, REVIT_API_DOCS_DATA_DIR=str(tiny_data_dir))
    out = _cli(["search", "Wall.Create", "--top-k", "3"], env)
    assert out.returncode == 0, out.stderr
    assert out.stdout.startswith("Rerank: none; data set: Revit 2026" + chr(10) * 2 + "### Wall.Create" + chr(10))
    lines = out.stderr.splitlines()
    # The two data warnings are the fixture's: its SQLite files are not the
    # release sizes, which ensure_data reports for any hand-copied tree.
    assert [ln.split(":")[0] for ln in lines] == [
        "revit_api_docs.data", "revit_api_docs.data", "revit_api_docs.retriever", "revit_api_docs.reranker"]
    assert all("using it as is" in ln for ln in lines[:2])
    assert "embedding disabled" in lines[2] and "COHERE_API_KEY" in lines[3]
    for noise in ("mismatch", "INFO", "[search]", "[keyword_search]", "ready: data in", "HTTP Request"):
        assert noise not in out.stderr


def test_search_verbose_shows_info_including_the_mismatch(tiny_data_dir, no_key_env):
    env = dict(no_key_env, REVIT_API_DOCS_DATA_DIR=str(tiny_data_dir))
    out = _cli(["-v", "search", "Wall.Create", "--top-k", "3"], env)
    assert out.returncode == 0, out.stderr
    assert "revit_api_docs.retriever: index/content mismatch" in out.stderr
    assert "revit_api_docs.server: ready: data in" in out.stderr
    assert "[keyword_search]" in out.stderr


def test_cli_logging_replaces_the_server_setup_and_quiets_httpx():
    """MCPServer() configures the root logger at import; the CLI must still
    end up with its own plain handler, WARNING by default, httpx at WARNING."""
    code = (
        "import logging, sys; from revit_api_docs.server import _configure_logging; "
        "_configure_logging(verbose=bool(int(sys.argv[1]))); root = logging.getLogger(); "
        "print(logging.getLevelName(root.level), logging.getLevelName(logging.getLogger('httpx').getEffectiveLevel()), "
        "[type(h).__name__ for h in root.handlers], [h.stream is sys.stderr for h in root.handlers])"
    )
    quiet = subprocess.run([sys.executable, "-c", code, "0"], capture_output=True, text=True, check=True, cwd=ROOT)
    assert quiet.stdout.strip() == "WARNING WARNING ['StreamHandler'] [True]"
    verbose = subprocess.run([sys.executable, "-c", code, "1"], capture_output=True, text=True, check=True, cwd=ROOT)
    assert verbose.stdout.strip() == "INFO WARNING ['StreamHandler'] [True]"


@pytest.mark.anyio
async def test_stdio_round_trip_with_keyword_only_data(tiny_data_dir, no_key_env):
    """Spawn the real server as an MCP stdio subprocess against a tiny data dir."""
    from mcp import ClientSession, StdioServerParameters
    from mcp.client.stdio import stdio_client

    env = dict(no_key_env, REVIT_API_DOCS_DATA_DIR=str(tiny_data_dir))
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
                if "### " in text:
                    break
                await asyncio.sleep(0.2)
            assert text.startswith("Rerank: none; data set: Revit 2026" + chr(10) * 2 + "### Wall.Create" + chr(10))

            res = await session.call_tool("get_code_examples", {"query": "wall"})
            assert "Project: CreateWall" in res.content[0].text


# -- transports ---------------------------------------------------------------

@pytest.fixture
def recorded_run(monkeypatch):
    """server.mcp.run replaced by a recorder; no data preparation thread."""
    calls: list[tuple[str, dict]] = []
    monkeypatch.setattr(server.mcp, "run", lambda transport="stdio", **kw: calls.append((transport, kw)))
    monkeypatch.setattr(server._state, "start_background", lambda: True)
    return calls


def test_stdio_stays_the_default_transport(recorded_run):
    assert server.main([]) == 0
    assert server.main(["serve"]) == 0
    assert server.main(["serve", "--transport", "stdio"]) == 0
    assert recorded_run == [("stdio", {})] * 3


def test_http_transport_options(recorded_run, caplog):
    caplog.set_level(logging.WARNING, logger="revit_api_docs.server")
    server.main(["serve", "--transport", "http"])
    server.main(["serve", "--transport", "http", "--host", "localhost", "--port", "9001"])
    server.main(["serve", "--transport", "http", "--host", "::1"])
    assert recorded_run == [
        ("streamable-http", {"host": "127.0.0.1", "port": 8765}),
        ("streamable-http", {"host": "localhost", "port": 9001}),
        ("streamable-http", {"host": "::1", "port": 8765}),
    ]
    assert "no built-in authentication" not in caplog.text


def test_http_on_a_non_loopback_host_warns(recorded_run, caplog):
    caplog.set_level(logging.WARNING, logger="revit_api_docs.server")
    server.main(["serve", "--transport", "http", "--host", "0.0.0.0", "--port", "8800"])
    assert recorded_run == [("streamable-http", {"host": "0.0.0.0", "port": 8800})]
    warnings = [r for r in caplog.records if r.levelno == logging.WARNING]
    assert len(warnings) == 1
    msg = warnings[0].getMessage()
    assert "0.0.0.0:8800" in msg and "no built-in authentication" in msg and "reverse proxy" in msg


@pytest.mark.parametrize("host, loopback", [
    ("127.0.0.1", True), ("127.1.2.3", True), ("localhost", True), ("::1", True), ("[::1]", True),
    ("0.0.0.0", False), ("::", False), ("192.168.1.20", False), ("revit-docs.example.com", False),
])
def test_is_loopback(host, loopback):
    assert server.is_loopback(host) is loopback


@pytest.mark.parametrize("argv", [
    ["serve", "--transport", "http", "--port", "70000"],
    ["serve", "--transport", "http", "--port", "abc"],
    ["serve", "--transport", "sse"],
])
def test_bad_serve_options_are_rejected(recorded_run, argv, capsys):
    with pytest.raises(SystemExit) as e:
        server.main(argv)
    assert e.value.code == 2
    assert recorded_run == []


def test_host_and_port_without_http_are_ignored_with_a_warning(recorded_run, caplog):
    caplog.set_level(logging.WARNING, logger="revit_api_docs.server")
    server.main(["serve", "--host", "0.0.0.0"])
    assert recorded_run == [("stdio", {})]
    assert "apply to --transport http only" in caplog.text


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


@pytest.mark.anyio
async def test_http_round_trip_with_keyword_only_data(tiny_data_dir, no_key_env, tmp_path):
    """Run the real server with --transport http on a free loopback port and talk MCP to it."""
    import httpx
    import httpx2
    from mcp import ClientSession
    from mcp.client.streamable_http import streamable_http_client

    port = _free_port()
    url = f"http://127.0.0.1:{port}/mcp"
    env = dict(no_key_env, REVIT_API_DOCS_DATA_DIR=str(tiny_data_dir))
    log_path = tmp_path / "server.log"
    with open(log_path, "wb") as log_file:
        proc = subprocess.Popen(
            [sys.executable, "-m", "revit_api_docs", "serve", "--transport", "http", "--port", str(port)],
            env=env, cwd=str(ROOT), stdout=log_file, stderr=subprocess.STDOUT)
        try:
            deadline = time.monotonic() + 30
            while True:  # any HTTP status means the server is listening
                try:
                    httpx.get(url, timeout=1.0, trust_env=False)
                    break
                except httpx.TransportError:
                    if proc.poll() is not None or time.monotonic() > deadline:
                        pytest.fail("HTTP server did not start: " + log_path.read_text(errors="replace"))
                    await asyncio.sleep(0.2)

            async with httpx2.AsyncClient(trust_env=False, timeout=30.0) as client:
                async with streamable_http_client(url, http_client=client) as (read, write):
                    async with ClientSession(read, write) as session:
                        init = await session.initialize()
                        assert init.server_info.name == "revit-api-docs"
                        names = {t.name for t in (await session.list_tools()).tools}
                        assert names == {"search_revit_api", "get_code_examples"}
                        text = ""
                        for _ in range(100):
                            res = await session.call_tool("search_revit_api", {"query": "Wall.Create", "top_k": 3})
                            text = res.content[0].text
                            if "### " in text:
                                break
                            await asyncio.sleep(0.2)
                        assert text.startswith("Rerank: none; data set: Revit 2026" + chr(10) * 2 + "### Wall.Create" + chr(10))
        finally:
            proc.terminate()
            proc.wait(timeout=15)
    # loopback bind: no authentication warning
    assert "no built-in authentication" not in log_path.read_text(errors="replace")
