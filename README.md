# revit-api-docs

Revit API reference search as an MCP server: two read-only tools over a local
copy of the Revit 2026 API documentation (28,862 classes, constructors,
methods, properties and enums) and the official SDK samples (153 projects).

**Optional.** It exists for models that do not know the Revit API well. If
your model already writes correct Revit code, you do not need it;
[revit-bridge](https://github.com/revitbridge/revit-bridge) works without it.
It never talks to Revit and it never calls a model itself.

Queries are **English only**: the corpus is the English API reference. A host
model that talks to its user in another language rewrites the request into
English API names or English task keywords before calling a tool.

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

**Any MCP client over HTTP** — run the server with the MCP streamable HTTP
transport and point the client at `/mcp`:

```bash
uvx revit-api-docs serve --transport http --host 127.0.0.1 --port 8765
```

```json
{
  "mcpServers": {
    "revit-api-docs": {
      "type": "http",
      "url": "http://127.0.0.1:8765/mcp"
    }
  }
}
```

(Claude Code: `claude mcp add --transport http revit-api-docs http://127.0.0.1:8765/mcp`.)
stdio stays the default. The HTTP transport has no built-in authentication:
keep it on a loopback address, or put it behind a reverse proxy that
authenticates; the server logs a warning when `--host` is not a loopback
address.

On the first start the server downloads the data set (267 MB, four files) from
this repository's [`v1.1-data` release](https://github.com/revitbridge/revit-api-docs/releases/tag/v1.1-data)
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
  boundary`). The first line names the reranker that ordered the entries
  (`jev`, `cohere` or `none`) and the data set they come from, for example
  `Rerank: none; data set: Revit 2026`; after an abstention line it is the
  second line. Development-environment questions (`target framework for
  Revit 2027 add-in`) also get up to five curated environment notes
  (framework, SDK, manifest, reference packages per Revit release, each with
  its source); these ship with the package and work before the data set is
  installed and without any key.
- `get_code_examples(query, top_k=3)` — C# from the SDK samples that use the
  matching APIs, with the project name and the APIs each sample touches.

Without any API key the server answers from keyword search over the SQLite
reference: BM25 over a full-text index of entry names, IDs, summaries and
signatures, where identifiers also match by their camel-case parts, so a
misspelt or unknown name still gets candidates. The index (11.8 MB) is built
next to the data set on first use (under a second). With an embedding key it
adds semantic search over the vector store and fuses the two rankings by rank,
which helps most with task descriptions that do not name the API. With a
TypeSafe key it reranks the candidates with Jev, which returns a calibrated
relevance per entry; with only a Cohere key it reranks with Cohere. With a Jev
threshold set and no entry reaching it, the output starts with an abstention
line and then lists the candidates with their relevance, for reference. See
Configure.

Try it from a terminal (installs the data set if needed):

```bash
uvx revit-api-docs search "Wall.Create"
uvx revit-api-docs search "rotate element around axis" --examples
uvx revit-api-docs environment --revit 2027          # environment notes, no data set needed
uvx revit-api-docs environment "target framework"   # notes a query matches
```

## Configure

All settings are environment variables; there is no config file to edit.

| Variable | Default | Meaning |
|---|---|---|
| `REVIT_API_DOCS_DATA_DIR` | `%LOCALAPPDATA%\revit-api-docs` (Windows), `~/.local/share/revit-api-docs` (Linux), `~/Library/Application Support/revit-api-docs` (macOS) | Where the data set lives. An existing tree with `sqlite/` and `chromadb/` from the old `revit-api-rag` checkout works as is |
| `REVIT_API_DOCS_EMBEDDING_API_KEY` | *(unset)* | Key for the embedding endpoint; turns on semantic (vector) search. `OPENROUTER_API_KEY` is used when this is unset |
| `REVIT_API_DOCS_EMBEDDING_BASE_URL` | `https://openrouter.ai/api/v1` | Any OpenAI-compatible `/embeddings` endpoint |
| `REVIT_API_DOCS_EMBEDDING_MODEL` | `openai/text-embedding-3-large` | Must produce the same 3072-dimensional space the shipped vectors were built with (`text-embedding-3-large`); another model makes vector search return noise |
| `TYPESAFE_API_KEY` | *(unset)* | Key for TypeSafe System One (Jev), the recommended reranker: with the default provider `auto` setting this key is enough. At most 30 candidates per request, 2 requests in flight, 2 retries, 20 s timeout |
| `COHERE_API_KEY` | *(unset)* | Optional. Cohere `rerank-v3.5` (native REST API, 20 s timeout): reranks when it is the only rerank key set, and is the fallback for a failed Jev call when both are set |
| `REVIT_API_DOCS_RERANK_PROVIDER` | `auto` | `auto`, `jev`, `cohere` or `none`. `auto` reranks with Jev when `TYPESAFE_API_KEY` is set, otherwise with Cohere when `COHERE_API_KEY` is set, otherwise not at all; an explicit value wins. With Jev, a failed or unconfigured call falls back to Cohere when `COHERE_API_KEY` is set, otherwise to no reranking; the first output line says which one ran |
| `REVIT_API_DOCS_JEV_MODEL` | `jev-latest` | Jev model or alias; pin a version such as `jev-1.14.0` for reproducible results |
| `REVIT_API_DOCS_MIN_RELEVANCE` | *(unset)* | Probability 0-1 (`retrieval.api.min_relevance`). Only honoured while Jev ranks the results, and it only decides whether to abstain: no entry is dropped, every result keeps its `Relevance` line. When no entry reaches it, `search_revit_api` starts with "No entry in the Revit 2026 data set scored high enough for this query", notes that the member may be absent from this version or listed under a different name or wording, and lists the low-scoring candidates for reference. No effect with Cohere or without reranking. See "Reranking with Jev" below |
| `REVIT_API_DOCS_RELEASE_URL` | this repository's `v1.1-data` release | Mirror for the data files; the SHA-256 checks still apply |
| `HTTPS_PROXY` / `HTTP_PROXY` | *(unset)* | Proxy for the embedding and rerank calls |

Pass them through the `env` block of your MCP host's server entry.

Recommended setup: a `TYPESAFE_API_KEY` (Jev reranking), plus an embedding key
when requests describe multi-step tasks rather than name APIs. Without any key
the server runs on BM25 keyword search alone.

**Reranking with Jev.** Jev is the recommended reranker, and the default
provider (`auto`) uses it whenever `TYPESAFE_API_KEY` is set. Cohere is
optional: `auto` uses it when only `COHERE_API_KEY` is set, and with both keys
it is the fallback for a failed Jev call. Measured on `v1.0-data`: on 60
English test queries (50 with a known answer) over the keyword (BM25)
candidates, with no embedding key, Jev raised nDCG@5 from 0.675 to 0.871 and
never turned a correct top-5 answer into a miss. On 223 further queries over
the same candidates Jev reached an nDCG@5 of 0.788 against Cohere's 0.674. Input tokens cost about $0.23 per 1,000
queries; Cohere bills $2 per 1,000 searches.
`REVIT_API_DOCS_MIN_RELEVANCE` stays unset by default. Set to 0.15 it filters
out clearly unrelated queries (questions that are not about Revit, types this
data set does not have), abstaining on about 2.5% of the queries that do have
an answer (measured on `v1.0-data`). It cannot spot a misspelt or misplaced
member name: such a query usually gets the closest real API as the first
result, to be confirmed with a compile check. **Do not use 0.5**: it
abstained on 14% of the queries that have an answer. What TypeSafe keeps: see the `jev-data-handling` note
(`uvx revit-api-docs environment typesafe`).

**Vector search.** An embedding key (`REVIT_API_DOCS_EMBEDDING_API_KEY`, or
`OPENROUTER_API_KEY`) turns it on; each query then costs one embedding request
(about 0.9 s). The vector hits and the BM25 hits, 30 of each, are fused by
reciprocal rank (score = sum of 1 / (60 + rank)) and the best 30 go to the
reranker; without a reranker they are cut to `top_k`. On 223 English test
queries without reranking (measured on `v1.1-data`), fusion raises nDCG@5
from 0.557 (BM25 alone) to 0.647, and on the 113 of them that describe a
multi-step task from 0.355 to 0.459, with an answer among the top 30 for 82%
of those instead of 72%. Each
vector hit is matched to its entry by name. On the first vector query the
server samples 50 records of each vector collection: when the record ids do
not line up with the rows it logs the warning `vector_store_misaligned` and
keeps matching by name (the older `v1.0-data` vector store numbers its records
differently from the rows of `revit_api.db`; `v1.1-data` is aligned and logs
nothing); when the names are not found either, it turns vector search off
until the server restarts and answers from keyword search. `revit-api-docs download`
prints the same check for the installed data set.

Privacy: with an embedding, Cohere or TypeSafe key the query text and the
candidate API entries (public documentation) are sent to that provider. Keep
queries to API names and generic task descriptions, in English.

Data set (`v1.1-data`): the API reference comes from `RevitAPI.chm` of the
Revit 2026.3 SDK (the same pages as the Revit 2026 reference) and includes
constructor pages and interface members such as
`IExternalEventHandler.Execute`: 28,862 entries, of which the 27,596 of
`v1.0-data` are unchanged under the same ids. The SDK samples (153 projects)
come from the Revit 2026 SDK. The SQLite databases and ChromaDB indexes were
built with the pipeline in `pipeline/` (`uv sync --extra pipeline`; needs the
CHM, the SDK and OpenRouter/Cohere keys in the environment). `v1.1-data` was
built from `v1.0-data` with `uv run python -m pipeline.build_data_v1_1
--html-dir <CHM html dir> --base-db <v1.0-data dir>/sqlite/revit_api.db`, which
appends the pages `v1.0-data` lacked and rebuilds the API vector store so that
every record id is its row id. The server does not need any of that.

Development:

```bash
uv sync
uv run pytest
uv build
uvx --from . revit-api-docs search "Wall.Create"
```

License: MIT. Issues and discussion: <https://github.com/revitbridge/revit-api-docs/issues>.
