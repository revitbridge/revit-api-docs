# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/); versions follow
[Semantic Versioning](https://semver.org/).

## [Unreleased]

Vector hits now come back as the entries they describe and count also
without a reranker; reranking picks its provider from the keys that are set.

**Upgrading**: with an embedding key the first vector query logs the warning
`vector_store_misaligned` for the `v1.0-data` release: its record ids do not
line up with the SQLite rows, and hits are now matched by entry name. Without
any key nothing changes. Reranker choice: with only `COHERE_API_KEY` set Cohere
still reranks; with `TYPESAFE_API_KEY` set Jev now reranks without
`REVIT_API_DOCS_RERANK_PROVIDER=jev` (set the provider to `cohere` to keep
Cohere).

### Fixed

- Vector search returned unrelated entries. The vector store of the
  `v1.0-data` release numbers its 28,863 API records independently of the
  27,596 rows of `revit_api.db` (2 ids name the same entry in both), and hits
  were looked up by record id. A hit is now matched to the row whose entry
  name equals the record's `name` metadata (the row with the record's own id
  when it carries that name; among rows sharing a name the one with the
  record's `full_id` or `info`, else the lowest id). 27,591 records (95.6%)
  reach 27,512 rows (99.7%); records without a row are dropped. The SDK sample
  store's ids match its rows, and matching by project name keeps them. The
  name map is built on the first vector query and kept in memory.
- Consistency probe: on the first vector query 50 records of each vector
  collection are compared with SQLite. When fewer than 90% of the sampled ids
  name the same entry as the row with that id, the warning
  `vector_store_misaligned` is logged and hits are matched by name (`v1.0-data`:
  0/50 ids, 49/50 names). When fewer than 50% of the sampled names are found
  either, vector search is off for the process and keyword search answers
  alone. `revit-api-docs download` prints the check for the installed data set.
- A second retriever in the same process could not open the vector store
  ("An instance of Chroma already exists ... with different settings"): each
  ChromaDB client now gets its own `Settings` object.

### Changed

- Hybrid ranking by reciprocal rank fusion: the vector hits and the BM25 hits,
  `retrieval.api.top_k` (30) of each, are fused with score = sum of
  1 / (60 + rank), and the best 30 go to the reranker; without a reranker they
  are cut to `top_k`. On equal scores a keyword hit comes first. Before, the
  two lists were merged by distances on different scales, so without a
  reranker vector-only hits never reached the top 5, and up to 60 candidates
  went to the reranker. Without an embedding key the BM25 order and results
  are unchanged. On 223 English test queries without reranking: nDCG@5 0.565
  -> 0.610 overall and 0.365 -> 0.438 on the 113 multi-step task requests,
  whose answer is among the top 30 for 84% instead of 73% of them.
- The default rerank provider is `auto`: Jev when `TYPESAFE_API_KEY` is set,
  otherwise Cohere when `COHERE_API_KEY` is set, otherwise none.
  `REVIT_API_DOCS_RERANK_PROVIDER` (now `auto`, `jev`, `cohere` or `none`)
  still wins, and the `Rerank:` line names the provider that ran. README:
  Jev is the recommended reranker, Cohere optional and the fallback for Jev.
- Cohere requests time out after 20 s instead of 10 s (the same as Jev); 10 s
  cut off about 1% of calls.
- `search_revit_api` and `get_code_examples` are unchanged: names,
  parameters and the first output line.

## [0.2.0] - 2026-10-26

Better candidates without any key, and an opt-in relevance check: BM25 keyword
search over an SQLite FTS5 index, optional Jev reranking with a calibrated
relevance per entry and abstention, a streamable HTTP transport, and curated
development-environment notes.

**Upgrading from 0.1.0** (`uvx --refresh revit-api-docs`): the first search
builds `sqlite/revit_api_fts.db` (an FTS5 index, 11.8 MB) next to the data set
in under a second. `search_revit_api` output starts with `Rerank: <reranker>;
data set: Revit 2026`. Abstention is new and opt-in:
`REVIT_API_DOCS_MIN_RELEVANCE` (Jev only) only decides whether to abstain and
drops no entry; the abstention line does not claim that the member is missing.

### Added

- Jev rerank provider (TypeSafe System One, `POST https://api.typesafe.ai/v1/systemone`,
  one `noul` question per candidate): `REVIT_API_DOCS_RERANK_PROVIDER=jev` with
  `TYPESAFE_API_KEY`. Up to 30 candidates per request, 2 requests in flight,
  2 retries with backoff on 429/529/5xx/timeouts, 20 s timeout. A failed or
  unconfigured Jev call falls back to Cohere (when `COHERE_API_KEY` is set) or
  to no reranking.
- `search_revit_api` output starts with the reranker that ran
  (`Rerank: jev | cohere | none`, plus `(unavailable: jev)` after a fallback);
  Jev-ranked entries carry `Relevance: 0.00-1.00`.
- Abstention: with Jev and `REVIT_API_DOCS_MIN_RELEVANCE`, when no entry
  reaches the threshold the output starts with "No entry in the Revit 2026
  data set scored high enough for this query (best relevance x < y)" followed
  by the low-scoring candidates. Other providers never abstain.
- Streamable HTTP transport: `revit-api-docs serve --transport http --host
  127.0.0.1 --port 8765` (endpoint `/mcp`); stdio remains the default. A
  non-loopback `--host` logs a warning: there is no built-in authentication.
- Development-environment notes: a curated English corpus shipped with the
  package (`environment.yaml`: target framework, SDK, manifest, reference
  packages for Revit 2024-2027, each with a release range and a source URL),
  matched by keywords. `search_revit_api` appends up to five matching notes,
  also while the data set is still downloading and without any key;
  `revit-api-docs environment [query] [--revit 2027]` prints them. The
  `jev-data-handling` note points at TypeSafe's data processing terms.
- Retrieval regression fixture `tests/fixtures/retrieval_queries.yaml`: 60
  English queries with verified gold rows in the v1.0-data reference (50
  answerable, 10 not). `tests/test_bm25_recall.py` runs them against the
  shipped data set when it is installed: gold in the BM25 top 30 for at least
  44 of 50 (measured 45), hit@5 at least 36 of 50 (measured 38), and a
  non-empty candidate list for every unanswerable query. No network.
- A second regression fixture, `tests/fixtures/retrieval_queries_v2.yaml`: 60
  more English queries (40 answerable, 20 not, 9 of those next to a real API,
  such as a class name used as a member or a misspelling), none shared with
  the first set. `tests/test_bm25_recall.py` runs both sets;
  for this one: gold in the BM25 top 30 for at least 33 of 40 (measured 34),
  hit@5 at least 25 of 40 (measured 27), and candidates for at least 19 of the
  20 unanswerable queries (measured 19).
- `close()` on the rerank providers releases their HTTP client.
- README: when to turn on Jev (`REVIT_API_DOCS_RERANK_PROVIDER=jev` with a
  TypeSafe key; nDCG@5 0.675 -> 0.871 on the test queries, about $0.23 per
  1,000 queries), what a `REVIT_API_DOCS_MIN_RELEVANCE` of 0.15 filters and
  what it does not, and why not 0.5. Defaults are unchanged.

### Changed

- The `Rerank:` line also names the data set the entries come from: `Rerank:
  <provider>[ (unavailable: ...)]; data set: Revit <version>`, the version
  taken from the configuration (`revit_version`, 2026 for the v1.0-data
  release). It stays the first line, or the second after an abstention line,
  so an answer can cite the data set version next to a result; the tool
  description and the server instructions say so. The `Rerank:` prefix is
  unchanged.
- `REVIT_API_DOCS_MIN_RELEVANCE` only decides whether to abstain. When at
  least one entry reaches it, the entries below it are no longer dropped:
  every reranked entry comes back (up to `top_k`) with its `Relevance`, so a
  correct entry scored just under the threshold stays in the list. Abstention
  itself, and the output format, are unchanged.
- README and the `jev-rerank-provider` environment note describe the
  threshold as a relevance filter: at 0.15 it filters out clearly unrelated
  queries (not about Revit, types the data set does not have) and abstains
  on about 2.5% of the queries that have an answer, but it cannot spot a
  misspelt or misplaced member name, for which the closest real API usually
  comes first and a compile check decides. "No entry scored high enough" is
  described as "nothing relevant, or worded differently", not as "the member
  does not exist"; the README no longer says the value is being validated.
- The rerank HTTP clients (Jev and Cohere) keep idle connections open for 5
  minutes (`keepalive_expiry=300`; httpx closes them after 5 s), so a query
  after a pause can reuse the connection instead of setting up TLS again.
  The other connection limits stay at httpx's defaults.
- Keyword search over the API reference ranks with BM25 on an SQLite FTS5
  index instead of LIKE substring scoring. The index is a separate file,
  `sqlite/revit_api_fts.db` next to `revit_api.db` (which is never written),
  built on first use in under a second and rebuilt when the source database's
  SHA-256 or the index format changes; the installer neither downloads nor
  flags it. Identifiers also match by their camel-case parts, so misspelt or
  unknown names (`FiltredElementColector`) get candidates instead of "No API
  entries found". Without vector search the keyword search now hands the
  reranker `retrieval.api.top_k` (30) candidates instead of the requested
  count; with vector search it adds the requested count, as before. A Python
  whose sqlite3 lacks FTS5 keeps the LIKE scorer and logs one warning.
- The Jev and Cohere rerank providers keep one HTTP client per instance, and
  with it the open connection, instead of building one per call. Against the
  hosted Jev API a Jev-reranked `search_revit_api` call took about 2.1 s
  before (10 candidates); now the first call of a process takes about 2.3 s
  and later ones about 0.4 s (30 candidates).
- The abstention line no longer claims the class or member may not exist: it
  says that no entry scored high enough and that either the member is absent
  from this version or the data set has it under a different name or wording.
  The tool description and server instructions say the same. A low best score
  cannot tell a missing member from one the candidate search did not bring up.
- The embedding providers look up their API key before importing the SDK, so
  keyword-only runs (no key) no longer import `openai`; `search` starts about
  2 s faster.
- `search` and `download` log plain `logger: message` lines to stderr at
  WARNING (`-v` for INFO), with `httpx` request lines hidden; the MCP server's
  logging is unchanged. The index/content mismatch of the shipped data set is
  logged at INFO instead of WARNING.
- Download progress: on a terminal the line is redrawn in place; in host logs
  and pipes one line per 10% instead of one per second.
- Tool descriptions and the server instructions state that queries are English
  only.
- The Cohere reranker shares a provider interface with Jev; its requests and
  its fall-back-to-input-order behaviour are unchanged.

## [0.1.0] - 2026-09-20

First release, split out of the `revit-api-rag` repository (working-tree copy,
no git history).

### Added

- MCP server `revit-api-docs` (stdio) with two read-only tools:
  `search_revit_api(query, top_k)` and `get_code_examples(query, top_k)`.
- Data set as GitHub Release `v1.0-data` (Revit 2026 API reference SQLite,
  SDK samples SQLite, two ChromaDB indexes); downloaded on first run into the
  per-user data directory, SHA-256 verified, resumable. `REVIT_API_DOCS_DATA_DIR`
  overrides the location; `revit-api-docs download` prefetches.
- Keyword search over SQLite works with no API key; an OpenAI-compatible
  embedding key adds vector search, `COHERE_API_KEY` adds reranking. Embedding
  or rerank failures fall back to keyword search instead of erroring.
- CLI: `serve` (default), `download`, `search <query> [--examples] [--top-k]`.
- Offline data pipeline (`pipeline/`, `uv sync --extra pipeline`) from the old
  repository, importing the package's shared retriever, LLM client and prompts.
- CI (tests + build on Linux/Windows, Python 3.11/3.13) and PyPI Trusted
  Publishing on version tags.

[Unreleased]: https://github.com/revitbridge/revit-api-docs/compare/v0.2.0...HEAD
[0.2.0]: https://github.com/revitbridge/revit-api-docs/compare/v0.1.0...v0.2.0
[0.1.0]: https://github.com/revitbridge/revit-api-docs/releases/tag/v0.1.0
