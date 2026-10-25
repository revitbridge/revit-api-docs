# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/); versions follow
[Semantic Versioning](https://semver.org/).

## [Unreleased]

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
- Abstention: with Jev and `REVIT_API_DOCS_MIN_RELEVANCE`, entries below the
  threshold are dropped and, when none reaches it, the output starts with
  "No API entry in the Revit 2026 data set matched this query (best relevance
  x < y)" followed by the low-scoring candidates. Other providers never abstain.
- Streamable HTTP transport: `revit-api-docs serve --transport http --host
  127.0.0.1 --port 8765` (endpoint `/mcp`); stdio remains the default. A
  non-loopback `--host` logs a warning: there is no built-in authentication.
- Development-environment notes: a curated English corpus shipped with the
  package (`environment.yaml`: target framework, SDK, manifest, reference
  packages for Revit 2024-2027, each with a release range and a source URL),
  matched by keywords. `search_revit_api` appends up to five matching notes,
  also while the data set is still downloading and without any key;
  `revit-api-docs environment [query] [--revit 2027]` prints them.

### Changed

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

[Unreleased]: https://github.com/revitbridge/revit-api-docs/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/revitbridge/revit-api-docs/releases/tag/v0.1.0
