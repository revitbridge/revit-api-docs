"""MCP server: two read-only tools over the Revit API reference.

    search_revit_api(query, top_k)   -> API entries (signature, parameters, remarks)
                                        plus matching development-environment notes
    get_code_examples(query, top_k)  -> C# snippets from the official SDK samples

Everything is local once the data set is installed. On the first run the
server answers immediately and downloads the data set (about 335 MB) in the
background; tool calls made before it is ready get a short status message
instead of a result (followed by any matching environment notes, which ship
with the package). `revit-api-docs download` fetches it ahead of time and
reports whether the vector store's records line up with the SQLite rows.

Transports: stdio (default; stdout is the MCP transport, all diagnostics go to
stderr) or streamable HTTP (`serve --transport http`), which has no built-in
authentication.
"""
from __future__ import annotations

import argparse
import asyncio
import ipaddress
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
DEFAULT_HTTP_HOST = "127.0.0.1"
DEFAULT_HTTP_PORT = 8765

SERVER_INSTRUCTIONS = """\
Revit API reference lookup (Revit 2026). Optional companion for models that do
not know the Revit API well; it never touches a Revit session.

Queries are English only: the reference is the English API documentation.
Rewrite other languages into English API names or English task keywords
before calling a tool.

- search_revit_api(query, top_k): find classes, methods, properties and enums
  in the API reference. Ask with API names when you have them
  ("Wall.Create", "FilteredElementCollector OfCategory") or with a short task
  description ("create floor from boundary"). Results carry the signature,
  parameters and remarks; trust them over memory. The first line (the second
  after an abstention line) names the reranker and the data set ("Rerank:
  none; data set: Revit 2026"); cite that version with what you take from the
  results. When the output says no entry scored high enough, either the
  member is absent from this version or the data set has it under a different
  name or wording: check the name and the Revit version, or rephrase, instead
  of guessing. Questions about the development environment ("target framework
  for Revit 2027 add-in") also get curated environment notes with sources.
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


# -- Retriever lifecycle -------------------------------------------------------

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


def _with_environment_notes(text: str, query: str) -> str:
    """Append the matching environment notes (if any) after a blank line."""
    from revit_api_docs.environment import notes_for_query

    notes = notes_for_query(query)
    return f"{text}\n\n{notes}" if notes else text


def abstention_line(revit_version: str, reason: str) -> str:
    """First line of search_revit_api output when Jev scored every candidate too low.

    Neutral on purpose: a low best score cannot tell a member that does not
    exist from one the keyword or vector search did not bring up.
    """
    return (f"No entry in the Revit {revit_version} data set scored high enough for this query "
            f"({reason}). This can mean the member is absent from this version, or that the "
            "data set has it under a different name or wording; check the name and version, "
            "or rephrase.")


def rerank_line(results, revit_version: str) -> str:
    """The `Rerank:` line: the reranker that ran and the data set the entries come from.

    `Rerank: <provider>[ (unavailable: ...)]; data set: Revit <version>` - the
    first output line, or the second after an abstention line.
    """
    from revit_api_docs.retriever import rerank_label

    return f"Rerank: {rerank_label(results)}; data set: Revit {revit_version}"


def search_api(retriever: RAGRetriever, query: str, top_k: int) -> str:
    results = retriever.search(query, api_top_k=top_k, code_top_k=0, rewrite=False, rerank=True)
    ctx = retriever.build_context(results)
    api_context = ctx.get("api_context")
    if not api_context:
        return _with_environment_notes(f"No API entries found for {query!r}.", query)
    head = [rerank_line(results, retriever.revit_version)]
    if results.abstained:
        head = [abstention_line(retriever.revit_version, results.abstain_reason),
                f"{head[0]}. Low-relevance candidates, for reference only:"]
    return _with_environment_notes("\n".join(head) + "\n\n" + api_context, query)


def search_examples(retriever: RAGRetriever, query: str, top_k: int) -> str:
    results = retriever.search(query, api_top_k=0, code_top_k=top_k, rewrite=False, rerank=False)
    ctx = retriever.build_context(results)
    return ctx.get("code_context") or f"No SDK examples found for {query!r}."


# -- Tools ---------------------------------------------------------------------

@mcp.tool(annotations=_READ_ONLY)
async def search_revit_api(query: str, top_k: int = 10) -> str:
    """Search the Revit 2026 API reference (27,596 classes, methods, properties,
    enums). Returns each match with its signature, parameters and remarks.
    Query in English with an API name ("Wall.Create") or a short task
    ("rotate element around axis"); the reference is English only, so rewrite
    other languages into English API keywords first. The first line (the
    second after an abstention line) names the reranker and the data set
    ("Rerank: none; data set: Revit 2026"); cite that version. If the output
    says no entry scored high enough, the member is either absent from this
    version or listed under a different name or wording.
    Development-environment questions ("target framework for Revit 2027
    add-in") also get curated environment notes. top_k: 1-50, default 10."""
    if not query or not query.strip():
        return "Give a query: an API name such as 'Wall.Create' or a short task description."
    msg = _state.message_if_not_ready()
    if msg:
        return _with_environment_notes(msg, query.strip())
    n = _clamp_top_k(top_k, 10)
    return await asyncio.to_thread(search_api, _state.retriever, query.strip(), n)


@mcp.tool(annotations=_READ_ONLY)
async def get_code_examples(query: str, top_k: int = 3) -> str:
    """Return C# examples from the official Revit SDK samples (153 projects)
    that use the APIs matching the query, each with the project name and the
    APIs it touches. Query in English (API names or a short task); the samples
    and their summaries are English only. top_k: 1-50, default 3."""
    if not query or not query.strip():
        return "Give a query: an API name or the task the sample should show."
    msg = _state.message_if_not_ready()
    if msg:
        return msg
    n = _clamp_top_k(top_k, 3)
    return await asyncio.to_thread(search_examples, _state.retriever, query.strip(), n)


# -- CLI -----------------------------------------------------------------------

def _configure_logging(verbose: bool) -> None:
    """Plain stderr logging for the `download`, `search` and `environment` subcommands.

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


def is_loopback(host: str) -> bool:
    """True for localhost and loopback addresses (127.0.0.0/8, ::1)."""
    h = host.strip().strip("[]").lower()
    if h == "localhost":
        return True
    try:
        return ipaddress.ip_address(h).is_loopback
    except ValueError:
        return False


def _port(text: str) -> int:
    try:
        n = int(text)
    except ValueError:
        raise argparse.ArgumentTypeError(f"not a port number: {text!r}") from None
    if not 1 <= n <= 65535:
        raise argparse.ArgumentTypeError(f"port out of range 1-65535: {n}")
    return n


def _revit_release(text: str):
    from revit_api_docs.environment import parse_version

    try:
        return parse_version(text)
    except ValueError as e:
        raise argparse.ArgumentTypeError(str(e)) from None


def vector_store_report(paths) -> list[str]:
    """Lines for `download`: the consistency probe of both vector collections.

    The probe needs no embedding key. A collection whose entry names are not
    found in SQLite turns vector search off (keyword search still works).
    """
    from revit_api_docs.retriever import check_vector_stores

    try:
        probes = check_vector_stores(str(paths.api_db), str(paths.sdk_db),
                                     str(paths.chromadb_api), str(paths.chromadb_code))
    except Exception as e:
        return [f"vector store check failed ({type(e).__name__}: {str(e)[:200]}); vector search "
                "will be off when it fails the same way at query time"]
    lines = [f"vector store check: {p.describe()}" for p in probes]
    if any(p.status == "unusable" for p in probes):
        lines.append("vector search stays off for this data set, even with an embedding key; "
                     "keyword search is not affected")
    return lines


def serve(transport: str = "stdio", host: str = DEFAULT_HTTP_HOST, port: int = DEFAULT_HTTP_PORT) -> None:
    """Start data preparation in the background and run the MCP server."""
    if transport == "http":
        if not is_loopback(host):
            log.warning(
                "serving MCP over HTTP on %s:%d with no built-in authentication: anyone who can "
                "reach this address can call the tools. For remote use put it behind a reverse "
                "proxy that authenticates.", host, port)
        log.info("MCP streamable HTTP endpoint: http://%s:%d/mcp", host, port)
        _state.start_background()
        mcp.run(transport="streamable-http", host=host, port=port)
        return
    _state.start_background()
    mcp.run(transport="stdio")


def _print_environment(query: str, revit) -> int:
    from revit_api_docs import environment

    notes = (environment.match(query, revit=revit, limit=None) if query.strip()
             else environment.select(revit))
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[union-attr]
    for n in notes:
        print(f"[{n.id}] {environment.describe_range(n)} | tags: {', '.join(n.tags)}")
        print(f"  {n.text}")
        print(f"  source: {n.source}")
        print()
    where = f" for Revit {environment.format_version(revit)}" if revit else ""
    print(f"{len(notes)} environment note(s){where}", file=sys.stderr)
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="revit-api-docs",
        description="MCP server for Revit API reference search (stdio by default, or streamable HTTP).",
    )
    parser.add_argument("--version", action="version", version=f"revit-api-docs {__version__}")
    parser.add_argument("-v", "--verbose", action="store_true", help="info-level logs on stderr")
    sub = parser.add_subparsers(dest="command")
    p_serve = sub.add_parser("serve", help="run the MCP server (default; stdio unless --transport http)")
    p_serve.add_argument("--transport", choices=("stdio", "http"), default="stdio",
                         help="stdio (default) or http (MCP streamable HTTP at /mcp, no built-in auth)")
    p_serve.add_argument("--host", default=None,
                         help=f"bind address for --transport http (default {DEFAULT_HTTP_HOST})")
    p_serve.add_argument("--port", type=_port, default=None,
                         help=f"port for --transport http (default {DEFAULT_HTTP_PORT})")
    sub.add_parser("download", help="install the data set now instead of on first use, "
                                    "then check that its vector store lines up with the SQLite rows")
    p_search = sub.add_parser("search", help="query from the terminal (installs the data set if needed)")
    p_search.add_argument("query")
    p_search.add_argument("--examples", action="store_true", help="SDK code examples instead of API entries")
    p_search.add_argument("--top-k", type=int, default=None)
    p_env = sub.add_parser("environment",
                           help="print the curated development-environment notes (no data set needed)")
    p_env.add_argument("query", nargs="?", default="", help="only notes this query matches")
    p_env.add_argument("--revit", type=_revit_release, default=None, metavar="RELEASE",
                       help="only notes that apply to this Revit release, e.g. 2027 or 2026.5")
    args = parser.parse_args(argv)

    if args.command in ("download", "search", "environment"):
        _configure_logging(args.verbose)

    if args.command == "environment":
        return _print_environment(args.query, args.revit)

    if args.command == "download":
        paths = data.ensure_data(progress=data.stderr_progress)
        print(f"data ready in {paths.root}", file=sys.stderr)
        for line in vector_store_report(paths):
            print(line, file=sys.stderr)
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

    transport = getattr(args, "transport", "stdio")
    host = getattr(args, "host", None)
    port = getattr(args, "port", None)
    if transport != "http" and (host is not None or port is not None):
        log.warning("--host and --port apply to --transport http only; serving on stdio")
    serve(transport, host or DEFAULT_HTTP_HOST, port or DEFAULT_HTTP_PORT)
    return 0


if __name__ == "__main__":
    sys.exit(main())
