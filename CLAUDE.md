# revit-api-docs

Python package + MCP server for Revit API reference search. Public repository:
only `README.md`, `CHANGELOG.md` and this file are documentation; design notes
and logs live in the private notes repository.

## Build and test

```bash
uv sync                      # creates .venv, installs dev group (pytest)
uv run pytest                # unit tests; tiny SQLite fixtures, a local HTTP server for the
                             # download path, one real stdio round trip; no network, no data set
uv build                     # sdist + wheel; the wheel ships revit_api_docs/ only (pipeline/ is sdist-only)
uvx --from . revit-api-docs search "Wall.Create"   # installs the data set on first run, prints API entries
```

## Layout

- `revit_api_docs/server.py` — `MCPServer` with exactly two tools (`search_revit_api`, `get_code_examples`),
  background data preparation, `main()` (`serve` | `download` | `search`).
- `revit_api_docs/data.py` — data directory (`REVIT_API_DOCS_DATA_DIR`), release manifest (`ARTIFACTS` with
  SHA-256), download/resume/verify/extract.
- `revit_api_docs/config.py` + `default_config.yaml` — bundled defaults, `REVIT_API_DOCS_*` overrides.
- `revit_api_docs/retriever.py` — SQLite keyword search + optional ChromaDB vector search + optional rerank;
  degrades to keyword-only whenever a key is missing or an API call fails.
- `revit_api_docs/embedder/`, `reranker.py`, `llm_client.py`, `prompts/` — query-time providers.
- `pipeline/` — offline data build (CHM parsing, batch embedding, SDK extraction, demos, notebook).
  Runs from the repo root with `uv sync --extra pipeline`; imports the package, never the other way round.
- `tests/` — pytest.

## Hard constraints

- Never depend on `revit-bridge`, and `revit-bridge` never depends on this package; the two meet only
  in the MCP host. `tests/test_package.py` enforces the first half.
- The MCP server exposes only `search_revit_api` and `get_code_examples`. No code execution, no Revit
  connection, no model calls except the optional embedding/rerank requests.
- Runtime configuration comes from environment variables only; `default_config.yaml` holds no secrets,
  and the pipeline reads keys from the environment (`.env` is git-ignored).
- The data set is a GitHub Release (`v1.0-data`), never committed. Changing a data file means a new
  release tag and new entries in `data.ARTIFACTS`.
- stdout is the MCP transport: log to stderr only.
- No `sys.path` manipulation, no imports from other repositories. Files copied from the old
  `revit-api-rag` repository came in as working-tree files without git history.
- Workflow files and tool scripts are ASCII only.

## Commits

Prefix commit subjects with `revit-api-docs: `. Before committing run
`uv sync; uv run pytest; uv build` and make sure a `git grep` for the OpenRouter,
Anthropic, Google and GitHub key prefixes finds nothing (CI runs the same check).
