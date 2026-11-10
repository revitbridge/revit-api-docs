"""Proxy settings: every client accepts the scheme-less form local proxy tools
write ("127.0.0.1:7890"). No network: requests go to a local stand-in proxy
that answers itself and never connects anywhere."""
from __future__ import annotations

import http.server
import json
import logging
import threading

import pytest

from revit_api_docs import config as cfgmod

PROXY_VARS = ("HTTPS_PROXY", "HTTP_PROXY", "ALL_PROXY", "NO_PROXY",
              "https_proxy", "http_proxy", "all_proxy", "no_proxy")


@pytest.fixture
def clean_proxy_env(monkeypatch):
    """No proxy variable from the developer's shell, and the bundled config (proxy off)."""
    for name in PROXY_VARS:
        monkeypatch.delenv(name, raising=False)
    monkeypatch.delenv("REVIT_API_DOCS_CONFIG", raising=False)
    monkeypatch.setattr(cfgmod, "_config", None)
    return monkeypatch


class _Proxy(http.server.ThreadingHTTPServer):
    """Records each request line; CONNECT gets 502, anything else `answer`."""

    def __init__(self):
        super().__init__(("127.0.0.1", 0), _ProxyHandler)
        self.seen: list[tuple[str, str]] = []
        self.answer = lambda method, url, body: (200, "application/octet-stream", b"")

    @property
    def address(self) -> str:
        return f"127.0.0.1:{self.server_address[1]}"   # no scheme, on purpose


class _ProxyHandler(http.server.BaseHTTPRequestHandler):
    def do_CONNECT(self):  # noqa: N802
        self.server.seen.append(("CONNECT", self.path))
        self.send_response(502)
        self.send_header("Content-Length", "0")
        self.end_headers()

    def _answer(self):
        body = self.rfile.read(int(self.headers.get("Content-Length") or 0))
        self.server.seen.append((self.command, self.path))
        status, ctype, payload = self.server.answer(self.command, self.path, body)
        self.send_response(status)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    do_GET = do_POST = _answer

    def log_message(self, *a):
        pass


@pytest.fixture
def proxy():
    server = _Proxy()
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    yield server
    server.shutdown()
    server.server_close()


# -- get_proxy_url -------------------------------------------------------------

@pytest.mark.parametrize("value, expected", [
    ("127.0.0.1:7890", "http://127.0.0.1:7890"),
    ("  127.0.0.1:7890 ", "http://127.0.0.1:7890"),
    ("localhost:7890", "http://localhost:7890"),
    ("user:secret@proxy.local:8080", "http://user:secret@proxy.local:8080"),
    ("http://127.0.0.1:7890", "http://127.0.0.1:7890"),
    ("https://proxy.local:443", "https://proxy.local:443"),
    ("socks5://127.0.0.1:1080", "socks5://127.0.0.1:1080"),
    ("socks5h://127.0.0.1:1080", "socks5h://127.0.0.1:1080"),
])
def test_proxy_url_forms(clean_proxy_env, value, expected):
    clean_proxy_env.setenv("HTTPS_PROXY", value)
    assert cfgmod.get_proxy_url() == expected
    assert cfgmod.normalize_proxy_url(value) == expected


def test_a_blank_proxy_variable_counts_as_unset(clean_proxy_env):
    clean_proxy_env.setenv("HTTPS_PROXY", "   ")
    assert cfgmod.get_proxy_url() is None
    clean_proxy_env.setenv("HTTP_PROXY", "10.0.0.1:3128")
    assert cfgmod.get_proxy_url() == "http://10.0.0.1:3128"
    assert cfgmod.normalize_proxy_url("") is None and cfgmod.normalize_proxy_url(None) is None


def test_config_proxy_values_are_normalised(clean_proxy_env):
    clean_proxy_env.setattr(cfgmod, "_config", {"proxy": {"enabled": True, "https": " ", "http": "10.0.0.1:3128"}})
    assert cfgmod.get_proxy_url() == "http://10.0.0.1:3128"
    clean_proxy_env.setattr(cfgmod, "_config", {"proxy": {"enabled": False, "https": "10.0.0.1:3128"}})
    assert cfgmod.get_proxy_url() is None


# -- embedding (explicit proxy from get_proxy_url) -------------------------------

def test_a_scheme_less_proxy_keeps_vector_search_on(clean_proxy_env, tiny_dbs, tmp_path, caplog):
    """httpx rejects proxy='127.0.0.1:7890' with "Unknown scheme for proxy URL";
    the retriever used to log "embedding disabled" and stay keyword-only."""
    from revit_api_docs.retriever import RAGRetriever

    clean_proxy_env.setenv("HTTPS_PROXY", "127.0.0.1:7890")
    clean_proxy_env.setenv("REVIT_API_DOCS_EMBEDDING_API_KEY", "test-embedding-key")
    caplog.set_level(logging.WARNING, logger="revit_api_docs.retriever")
    api_db, sdk_db = tiny_dbs
    r = RAGRetriever(cfgmod.load_config(), str(api_db), str(sdk_db), str(tmp_path / "a"), str(tmp_path / "b"))
    assert "embedding disabled" not in caplog.text
    assert r.vector_search_enabled is True


def test_embedding_requests_go_through_a_scheme_less_proxy(clean_proxy_env, proxy):
    from revit_api_docs.embedder.providers.openai import OpenAIEmbedding

    def answer(method, url, body):
        payload = {"object": "list", "model": "m", "usage": {"prompt_tokens": 1, "total_tokens": 1},
                   "data": [{"object": "embedding", "index": 0, "embedding": [0.5, 0.25]}]}
        return 200, "application/json", json.dumps(payload).encode()

    proxy.answer = answer
    clean_proxy_env.setenv("HTTPS_PROXY", proxy.address)
    clean_proxy_env.setenv("REVIT_API_DOCS_EMBEDDING_API_KEY", "test-embedding-key")
    emb = OpenAIEmbedding(api_key_env="REVIT_API_DOCS_EMBEDDING_API_KEY", base_url="http://embeddings.invalid/v1")
    try:
        assert emb.embed_query("wall") == [0.5, 0.25]
    finally:
        emb.close()
    assert proxy.seen == [("POST", "http://embeddings.invalid/v1/embeddings")]


# -- rerank (httpx's own environment handling, or the config's proxy) -------------

@pytest.mark.parametrize("kind", ["cohere", "jev"])
def test_rerank_clients_use_a_scheme_less_proxy_from_the_environment(clean_proxy_env, proxy, kind):
    """The rerank clients get no proxy argument from the environment: httpx
    reads HTTPS_PROXY itself and adds http:// to a value without a scheme."""
    from revit_api_docs.reranker import JevReranker, Reranker, RerankError

    clean_proxy_env.setenv("HTTPS_PROXY", proxy.address)
    if kind == "cohere":
        clean_proxy_env.setenv("COHERE_API_KEY", "test-cohere-key")
        r, host = Reranker(), "api.cohere.com:443"
    else:
        clean_proxy_env.setenv("TYPESAFE_API_KEY", "test-typesafe-key")
        r, host = JevReranker(max_retries=0), "api.typesafe.ai:443"
    try:
        with pytest.raises(RerankError):   # the stand-in proxy refuses the tunnel
            r.rerank("wall", ["a", "b"], top_n=2, strict=True)
    finally:
        r.close()
    assert proxy.seen == [("CONNECT", host)]


@pytest.mark.parametrize("kind", ["cohere", "jev"])
def test_rerank_clients_accept_a_scheme_less_proxy_from_the_config(clean_proxy_env, proxy, kind):
    from revit_api_docs.reranker import JevReranker, Reranker, RerankError, _proxy_url

    cfg = {"proxy": {"enabled": True, "https": proxy.address}}
    assert _proxy_url(cfg) == f"http://{proxy.address}"
    if kind == "cohere":
        clean_proxy_env.setenv("COHERE_API_KEY", "test-cohere-key")
        r, host = Reranker(proxy=proxy.address), "api.cohere.com:443"
    else:
        clean_proxy_env.setenv("TYPESAFE_API_KEY", "test-typesafe-key")
        r, host = JevReranker(proxy=proxy.address, max_retries=0), "api.typesafe.ai:443"
    try:
        with pytest.raises(RerankError):
            r.rerank("wall", ["a"], top_n=1, strict=True)
    finally:
        r.close()
    assert proxy.seen == [("CONNECT", host)]


# -- data download (httpx's own environment handling) -----------------------------

def test_the_download_uses_a_scheme_less_proxy_from_the_environment(clean_proxy_env, proxy, tmp_path):
    from revit_api_docs import data

    proxy.answer = lambda method, url, body: (200, "application/octet-stream", b"0123456789")
    clean_proxy_env.setenv("HTTP_PROXY", proxy.address)
    dest = tmp_path / "x.part"
    data.download_file("http://release.invalid/dl/revit_sdk.db", dest, 10)
    assert dest.read_bytes() == b"0123456789"
    assert proxy.seen == [("GET", "http://release.invalid/dl/revit_sdk.db")]


# -- LLM client (query rewriting, pipeline tools) ----------------------------------

def test_the_llm_client_accepts_a_scheme_less_proxy(clean_proxy_env, proxy):
    from revit_api_docs.llm_client import LLMClient, create_llm_client

    def answer(method, url, body):
        reply = {"choices": [{"finish_reason": "stop", "message": {"content": "ok"}}]}
        return 200, "application/json", json.dumps(reply).encode()

    proxy.answer = answer
    with LLMClient("http://llm.invalid/v1", "test-llm-key", "m", proxy=proxy.address) as client:
        assert client.generate_text("hi") == "ok"
    assert proxy.seen == [("POST", "http://llm.invalid/v1/chat/completions")]

    clean_proxy_env.setenv("OPENROUTER_API_KEY", "test-llm-key")
    cfg = {"llm": {"provider": "x", "models": {"x": {"model": "m", "base_url": "http://llm.invalid/v1"}}},
           "proxy": {"enabled": True, "https": proxy.address}}
    with create_llm_client(cfg) as client:
        assert client.generate_text("hi") == "ok"
    assert len(proxy.seen) == 2
