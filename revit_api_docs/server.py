"""MCP server: two read-only tools over the Revit API reference.

    search_revit_api(query, top_k)   -> API entries (signature, parameters, remarks)
    get_code_examples(query, top_k)  -> C# snippets from the official SDK samples

Everything is local once the data set is installed. On the first run the
server answers immediately and downloads the data set (about 335 MB) in the
background; tool calls made before it is ready get a short status message
instead of a result. `revit-api-docs download` fetches it ahead of time.

stdout is the MCP transport: all diagnostics go to stderr.
"""
from __future__ import annotations

import argparse
import asyncio
import logging
import sys
import threading
from typing import TYPE_CHECKING

from mcp.server.mcpserver import MCPServer
from mcp.types import ToolAnnotations

from revit_api_docs import __version__, data
from revit_api_docs.config import load_config

if TYPE_CHECKING:
    from revit_api_docs.retriever import RAGRetriever

log = logging.getLogger("revit_api_docs.server")

MAX_TOP_K = 50

SERVER_INSTRUCTIONS = """\
Revit API reference lookup (Revit 2026). Optional companion for models that do
not know the Revit API well; it never touches a Revit session.

- search_revit_api(query, top_k): find classes, methods, properties and enums
  in the API reference. Ask with API names when you have them
  ("Wall.Create", "FilteredElementCollector OfCategory") or with a short task
  description ("create floor from boundary"). Results carry the signature,
  parameters and remarks; trust them over memory.
- get_code_examples(query, top_k): real C# from the Revit SDK samples that use
  the APIs in question. Read one before writing code for an unfamiliar API.

Use these before generating code that calls an API you are not certain about
(exact method name, parameter order, required ElementIds, units).
"""

mcp = MCPServer(
    "revit-api-docs",
    version=__version__,
    instructions=SERVER_INSTRUCTIONS,
)

_READ_ONLY = ToolAnnotations(readOnlyHint=True)


# ── Retriever lifecycle ──────────────────────────────────────────────────────

class _State:
    """Data download + retriever construction, shared by the tools."""

    def __init__(self) -> None:
        self.lock = threading.Lock()
        self.status = "idle"          # idle | downloading | loading | ready | failed
        self.detail = ""
        self.error: str | None = None
        self.retriever: RAGRetriever | None = None
        self._thread: threading.Thread | None = None

    def _progress(self, name: str, done: int, total: int) -> None:
        pct = 100 * done // total if total else 0
        self.detail = f"{name} {pct}%"
        data.stderr_progress(name, done, total)

    def prepare(self) -> RAGRetriever:
        """Blocking: install the data set if needed and build the retriever."""
        with self.lock:
            if self.retriever is not None:
                return self.retriever
            try:
                self.status = "downloading" if data.missing() else "loading"
                paths = data.ensure_data(progress=self._progress)
                self.status = "loading"
                from revit_api_docs.retriever import RAGRetriever

                self.retriever = RAGRetriever(
                    config=load_config(),
                    api_db_path=str(paths.api_db),
                    sdk_db_path=str(paths.sdk_db),
                    chromadb_api_dir=str(paths.chromadb_api),
                    chromadb_code_dir=str(paths.chromadb_code),
                )
                self.status = "ready"
                self.error = None
                log.info("ready: data in %s", paths.root)
                return self.retriever
            except Exception as e:  # reported to the caller, not raised into MCP
                self.status = "failed"
                self.error = f"{type(e).__name__}: {e}"
                log.exception("preparing the data set failed")
                raise

    def start_background(self) -> bool:
        """Start the preparation thread unless one is running; True if started."""
        if self._thread is not None and self._thread.is_alive():
            return False
        self._thread = threading.Thread(target=self._prepare_quietly,
                                        name="revit-api-docs-prepare", daemon=True)
        self._thread.start()
        return True

    def _prepare_quietly(self) -> None:
        try:
            self.prepare()
        except Exception:
            pass

    def message_if_not_ready(self) -> str | None:
        """Text to return from a tool while the retriever is unavailable."""
        if self.retriever is not None:
            return None
        if self.status == "failed":
            # Let the next call retry (network hiccup, disk full and freed, ...),
            # but only claim a retry when this call actually started one.
            retrying = self.start_background()
            how = ("Retrying in the background; call again in a moment"
                   if retrying else "A retry is already running; call again in a moment")
            return (f"revit-api-docs is not available: {self.error}. {how}, or run "
                    f"`revit-api-docs download` in a terminal to see the full error.")
        self.start_background()
        if self.status == "downloading":
            return (f"revit-api-docs is downloading its data set on first run "
                    f"({data.TOTAL_DOWNLOAD_BYTES / 1e6:.0f} MB, currently {self.detail}). "
                    f"Call again in a minute.")
        return "revit-api-docs is loading its index. Call again in a few seconds."


_state = _State()


def _clamp_top_k(top_k: int, default: int) -> int:
    try:
        n = int(top_k)
    except (TypeError, ValueError):
        return default
    return max(1, min(n, MAX_TOP_K))


def search_api(retriever: RAGRetriever, query: str, top_k: int) -> str:
    results = retriever.search(query, api_top_k=top_k, code_top_k=0, rewrite=False, rerank=True)
    ctx = retriever.build_context(results)
    return ctx.get("api_context") or f"No API entries found for {query!r}."


def search_examples(retriever: RAGRetriever, query: str, top_k: int) -> str:
    results = retriever.search(query, api_top_k=0, code_top_k=top_k, rewrite=False, rerank=False)
    ctx = retriever.build_context(results)
    return ctx.get("code_context") or f"No SDK examples found for {query!r}."


# ── Tools ────────────────────────────────────────────────────────────────────

@mcp.tool(annotations=_READ_ONLY)
async def search_revit_api(query: str, top_k: int = 10) -> str:
    """Search the Revit 2026 API reference (27,596 classes, methods, properties,
    enums). Returns each match with its signature, parameters and remarks.
    Query with an API name ("Wall.Create") or a short task ("rotate element
    around axis"). top_k: 1-50, default 10."""
    if not query or not query.strip():
        return "Give a query: an API name such as 'Wall.Create' or a short task description."
    msg = _state.message_if_not_ready()
    if msg:
        return msg
    n = _clamp_top_k(top_k, 10)
    return await asyncio.to_thread(search_api, _state.retriever, query.strip(), n)


@mcp.tool(annotations=_READ_ONLY)
async def get_code_examples(query: str, top_k: int = 3) -> str:
    """Return C# examples from the official Revit SDK samples (153 projects)
    that use the APIs matching the query, each with the project name and the
    APIs it touches. top_k: 1-50, default 3."""
    if not query or not query.strip():
        return "Give a query: an API name or the task the sample should show."
    msg = _state.message_if_not_ready()
    if msg:
        return msg
    n = _clamp_top_k(top_k, 3)
    return await asyncio.to_thread(search_examples, _state.retriever, query.strip(), n)


# ── CLI ──────────────────────────────────────────────────────────────────────

def _configure_logging(verbose: bool) -> None:
    """Plain stderr logging for the `download` and `search` subcommands.

    MCPServer() above already configured the root logger at import time
    (INFO, rich handler), so this must replace that setup (force=True);
    `serve` keeps it. httpx's per-request INFO lines stay hidden even with
    -v: the data module logs what it downloads in its own words.
    """
    logging.basicConfig(
        stream=sys.stderr,
        level=logging.INFO if verbose else logging.WARNING,
        format="%(name)s: %(message)s",
        force=True,
    )
    logging.getLogger("httpx").setLevel(logging.WARNING)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="revit-api-docs",
        description="MCP server for Revit API reference search (stdio).",
    )
    parser.add_argument("--version", action="version", version=f"revit-api-docs {__version__}")
    parser.add_argument("-v", "--verbose", action="store_true", help="info-level logs on stderr")
    sub = parser.add_subparsers(dest="command")
    sub.add_parser("serve", help="run the MCP server on stdio (default)")
    sub.add_parser("download", help="install the data set now instead of on first use")
    p_search = sub.add_parser("search", help="query from the terminal (installs the data set if needed)")
    p_search.add_argument("query")
    p_search.add_argument("--examples", action="store_true", help="SDK code examples instead of API entries")
    p_search.add_argument("--top-k", type=int, default=None)
    args = parser.parse_args(argv)

    if args.command in ("download", "search"):
        _configure_logging(args.verbose)

    if args.command == "download":
        paths = data.ensure_data(progress=data.stderr_progress)
        print(f"data ready in {paths.root}", file=sys.stderr)
        return 0

    if args.command == "search":
        retriever = _state.prepare()
        if args.examples:
            text = search_examples(retriever, args.query, _clamp_top_k(args.top_k or 3, 3))
        else:
            text = search_api(retriever, args.query, _clamp_top_k(args.top_k or 10, 10))
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[union-attr]
        print(text)
        return 0

    _state.start_background()
    mcp.run(transport="stdio")
    return 0


if __name__ == "__main__":
    sys.exit(main())
