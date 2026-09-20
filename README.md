# revit-api-docs

Revit API reference search as an MCP server: two read-only tools over a local
copy of the Revit 2026 API documentation (27,596 classes, methods, properties
and enums) and the official SDK samples (153 projects).

**Optional.** It exists for models that do not know the Revit API well. If
your model already writes correct Revit code, you do not need it;
[revit-bridge](https://github.com/revitbridge/revit-bridge) works without it.
It never talks to Revit and it never calls a model itself.

## Install

Prerequisites: Python 3.11+ with [`uv`](https://docs.astral.sh/uv/) (for
`uvx`), about 700 MB of disk for the data set, network for the first run.

**Claude Desktop** — `claude_desktop_config.json`:

```json
{
  "mcpServers": {
    "revit-api-docs": {
      "command": "uvx",
      "args": ["revit-api-docs"]
    }
  }
}
```

**Claude Code**:

```bash
claude mcp add revit-api-docs -- uvx revit-api-docs
```

**Any MCP client** — run `uvx revit-api-docs` as a stdio server.

On the first start the server downloads the data set (335 MB, four files) from
this repository's [`v1.0-data` release](https://github.com/revitbridge/revit-api-docs/releases/tag/v1.0-data)
into `%LOCALAPPDATA%\revit-api-docs` (Windows) or `~/.local/share/revit-api-docs`
(Linux; `~/Library/Application Support/revit-api-docs` on macOS), verifies each
file's SHA-256 and unpacks the vector stores. Tool calls made while that runs
return a short progress message instead of results. To do it ahead of time:

```bash
uvx revit-api-docs download
```

From a checkout: `uv sync` then `uv run revit-api-docs`.

## Use

Two tools, both read-only:

- `search_revit_api(query, top_k=10)` — API entries with signature, parameters
  and remarks. Query with an API name (`Wall.Create`,
  `FilteredElementCollector OfCategory`) or a short task (`create floor from
  boundary`).
- `get_code_examples(query, top_k=3)` — C# from the SDK samples that use the
  matching APIs, with the project name and the APIs each sample touches.

Without any API key the server answers from keyword search over the SQLite
reference (exact and partial name matches, which is what a model usually
asks for once it knows roughly what it wants). With an embedding key it adds
semantic search over the vector store, and with a Cohere key it reranks the
merged candidates. See Configure.

Try it from a terminal (installs the data set if needed):

```bash
uvx revit-api-docs search "Wall.Create"
uvx revit-api-docs search "rotate element around axis" --examples
```

## Configure

All settings are environment variables; there is no config file to edit.

| Variable | Default | Meaning |
|---|---|---|
| `REVIT_API_DOCS_DATA_DIR` | `%LOCALAPPDATA%\revit-api-docs` (Windows), `~/.local/share/revit-api-docs` (Linux), `~/Library/Application Support/revit-api-docs` (macOS) | Where the data set lives. An existing tree with `sqlite/` and `chromadb/` from the old `revit-api-rag` checkout works as is |
| `REVIT_API_DOCS_EMBEDDING_API_KEY` | *(unset)* | Key for the embedding endpoint; turns on semantic (vector) search. `OPENROUTER_API_KEY` is used when this is unset |
| `REVIT_API_DOCS_EMBEDDING_BASE_URL` | `https://openrouter.ai/api/v1` | Any OpenAI-compatible `/embeddings` endpoint |
| `REVIT_API_DOCS_EMBEDDING_MODEL` | `openai/text-embedding-3-large` | Must produce the same 3072-dimensional space the shipped vectors were built with (`text-embedding-3-large`); another model makes vector search return noise |
| `COHERE_API_KEY` | *(unset)* | Enables reranking with Cohere `rerank-v3.5` (native REST API) |
| `REVIT_API_DOCS_RELEASE_URL` | this repository's `v1.0-data` release | Mirror for the data files; the SHA-256 checks still apply |
| `HTTPS_PROXY` / `HTTP_PROXY` | *(unset)* | Proxy for the embedding and rerank calls |

Pass them through the `env` block of your MCP host's server entry.

Data set: the SQLite databases and ChromaDB indexes were built from the Revit
2026 API CHM and the Revit 2026 SDK samples with the pipeline in `pipeline/`
(`uv sync --extra pipeline`; needs the CHM, the SDK and OpenRouter/Cohere keys
in the environment). The server does not need any of that.

Development:

```bash
uv sync
uv run pytest
uv build
uvx --from . revit-api-docs search "Wall.Create"
```

License: MIT. Issues and discussion: <https://github.com/revitbridge/revit-api-docs/issues>.
