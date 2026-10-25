"""
Rerank providers: post-retrieval reranking of API reference candidates.

Flow:
    1. The retriever gathers candidates (vector + keyword search, broad recall)
    2. A provider scores each candidate against the query (precision)
    3. The top_n best results come back sorted by relevance score

Providers (config `rerank.provider`, env REVIT_API_DOCS_RERANK_PROVIDER):
    cohere  Cohere native rerank API (https://api.cohere.com/v2/rerank), key in
            COHERE_API_KEY. Scores order the candidates; they are not
            calibrated probabilities.
    jev     TypeSafe System One (model `jev-latest`), key in TYPESAFE_API_KEY.
            One `noul` question per candidate; the answer is the calibrated
            probability (0-1) that the candidate is what the query looks for,
            which is what lets the retriever abstain (`min_relevance`).
    none    no reranking.

Every provider has the same call: `rerank(query, documents, top_n, strict=False)
-> list[RerankResult]`. With strict=False a failure returns the input order
(the historical Cohere behaviour); with strict=True it raises RerankError so
the caller can try the next provider. `create_rerank_chain` returns the
providers to try in order: [jev, cohere] for provider "jev" (Cohere only when
its key is set), [cohere] for "cohere", [] for "none" or when no key is set.

Configured in default_config.yaml:
    rerank:
      enabled: true
      provider: "cohere"
      models:
        cohere:
          model: "rerank-v3.5"
          api_key_env: "COHERE_API_KEY"
        jev:
          model: "jev-latest"
          api_key_env: "TYPESAFE_API_KEY"
"""
from __future__ import annotations

import logging
import math
import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Any, Callable

import httpx

_log = logging.getLogger("revit_api_docs.reranker")

# Cohere native rerank endpoint
_COHERE_RERANK_URL = "https://api.cohere.com/v2/rerank"

# TypeSafe System One endpoint (docs.typesafe.ai, API reference "System One")
JEV_URL = "https://api.typesafe.ai/v1/systemone"

PROVIDERS = ("cohere", "jev", "none")


@dataclass
class RerankResult:
    """A single reranked item with its relevance score."""
    index: int          # original index in the input list
    score: float        # relevance score (higher = better)
    text: str           # the document text that was scored


class RerankError(RuntimeError):
    """A provider could not score the documents (no key, HTTP error, bad answer)."""


class RerankProvider:
    """Interface shared by the providers.

    `name` is what search results report as `rerank_used`; `calibrated` is
    True when scores are calibrated probabilities that a fixed threshold
    (`retrieval.api.min_relevance`) may be applied to.
    """

    name = ""
    calibrated = False

    def rerank(
        self,
        query: str,
        documents: list[str],
        top_n: int = 15,
        *,
        strict: bool = False,
    ) -> list[RerankResult]:
        raise NotImplementedError

    @staticmethod
    def _fallback(documents: list[str], top_n: int) -> list[RerankResult]:
        """Return documents in original (vector similarity) order."""
        return [
            RerankResult(index=i, score=1.0 - i * 0.01, text=doc)
            for i, doc in enumerate(documents[:top_n])
        ]


# -- Cohere -----------------------------------------------------------------

class Reranker(RerankProvider):
    """Reranker using Cohere's native rerank API (fast, ~200ms)."""

    name = "cohere"
    calibrated = False

    def __init__(
        self,
        model: str = "rerank-v3.5",
        api_key_env: str = "COHERE_API_KEY",
        proxy: str | None = None,
        transport: httpx.BaseTransport | None = None,
    ):
        self._model = model.replace("cohere/", "")  # strip provider prefix
        self._api_key_env = api_key_env
        self._api_key = os.getenv(api_key_env, "")
        self._proxy = proxy
        self._transport = transport

        if self._api_key:
            _log.info(f"[Reranker] initialized model={self._model}")
        else:
            _log.warning(f"[Reranker] no API key ({api_key_env}), rerank will be skipped")

    def rerank(
        self,
        query: str,
        documents: list[str],
        top_n: int = 15,
        *,
        strict: bool = False,
    ) -> list[RerankResult]:
        """Rerank documents against query, return top_n sorted by relevance.

        Uses Cohere's native v2 rerank endpoint - typically <1s for 30 docs.
        Without a key, or when the call fails, the input order comes back
        (strict=False) or RerankError is raised (strict=True).
        """
        if not documents:
            return []
        if not self._api_key:
            if strict:
                raise RerankError(f"no API key ({self._api_key_env})")
            _log.debug("[rerank] no API key, skipping rerank")
            return self._fallback(documents, top_n)
        try:
            return self._call(query, documents, top_n)
        except RerankError as e:
            if strict:
                raise
            _log.warning(f"[rerank] {e}, using fallback order")
            return self._fallback(documents, top_n)

    def _call(self, query: str, documents: list[str], top_n: int) -> list[RerankResult]:
        t0 = time.monotonic()
        try:
            headers = {
                "Authorization": f"Bearer {self._api_key}",
                "Content-Type": "application/json",
            }
            payload = {
                "model": self._model,
                "query": query,
                "documents": [{"text": d} for d in documents],
                "top_n": min(top_n, len(documents)),
                "return_documents": False,
            }

            client_kwargs: dict[str, Any] = {"timeout": 10.0}
            if self._proxy:
                client_kwargs["proxy"] = self._proxy
            if self._transport is not None:
                client_kwargs["transport"] = self._transport

            with httpx.Client(**client_kwargs) as client:
                resp = client.post(_COHERE_RERANK_URL, headers=headers,
                                   json=payload)
                resp.raise_for_status()
                data = resp.json()

            results = []
            for item in data.get("results", []):
                idx = item.get("index", 0)
                score = item.get("relevance_score", 0.0)
                text = documents[idx] if idx < len(documents) else ""
                results.append(RerankResult(index=idx, score=score, text=text))
        except httpx.HTTPStatusError as e:
            raise RerankError(f"HTTP {e.response.status_code}: "
                              f"{e.response.text[:200]}") from e
        except Exception as e:
            raise RerankError(f"failed ({e})") from e

        results.sort(key=lambda r: r.score, reverse=True)
        elapsed = time.monotonic() - t0
        _log.info(f"[rerank] {len(documents)} docs -> top {len(results)} "
                  f"in {elapsed:.2f}s "
                  f"scores=[{', '.join(f'{r.score:.3f}' for r in results[:3])}...]")
        return results


CohereReranker = Reranker


# -- Jev (TypeSafe System One) ----------------------------------------------

# One noul question per candidate; every question sees the same state (the
# query and the batch of candidates keyed by id) and is answered on its own.
_JEV_INSTRUCTIONS = (
    "The state holds a developer query about the Revit API (\"query\") and candidate "
    "entries from the Revit API reference (\"documents\", keyed by id). Is document "
    "\"{id}\" the API entry the query is looking for, or one that directly answers it?"
)
_JEV_CRITERIA = {
    "true": ("The entry is the class, member, overload or enum the query names, or the "
             "API that performs the task the query describes."),
    "false": ("The entry is only on a related topic, is a different member with a similar "
              "name, or would not help write the code the query is about."),
}

# 429 rate limit and 529 overload are documented as retryable; timeouts and
# 5xx gateway errors are treated the same way.
_JEV_RETRYABLE = frozenset({408, 429, 500, 502, 503, 504, 529})
_JEV_MAX_DOC_CHARS = 2000
_JEV_MAX_BACKOFF = 10.0


class JevReranker(RerankProvider):
    """Reranker using TypeSafe System One (Jev) noul probabilities.

    At most 30 documents per request (larger inputs are split into batches),
    at most `max_concurrency` requests in flight per instance (default 2),
    `max_retries` retries with exponential backoff on 429/529/5xx/timeouts
    (default 2), `timeout` seconds per request (default 20). Scores are the
    calibrated probabilities returned by the model (0-1).
    """

    name = "jev"
    calibrated = True
    MAX_BATCH = 30

    def __init__(
        self,
        model: str = "jev-latest",
        api_key_env: str = "TYPESAFE_API_KEY",
        *,
        base_url: str = JEV_URL,
        revit_version: str = "2026",
        batch_size: int = MAX_BATCH,
        max_concurrency: int = 2,
        max_retries: int = 2,
        timeout: float = 20.0,
        backoff: float = 0.5,
        proxy: str | None = None,
        transport: httpx.BaseTransport | None = None,
        sleep: Callable[[float], None] = time.sleep,
    ):
        self._model = model
        self._api_key_env = api_key_env
        self._api_key = os.getenv(api_key_env, "")
        self._url = base_url
        self._revit_version = str(revit_version)
        self._batch_size = max(1, min(int(batch_size), self.MAX_BATCH))
        self._max_concurrency = max(1, int(max_concurrency))
        self._slots = threading.BoundedSemaphore(self._max_concurrency)
        self._max_retries = max(0, int(max_retries))
        self._timeout = float(timeout)
        self._backoff = max(0.0, float(backoff))
        self._proxy = proxy
        self._transport = transport
        self._sleep = sleep

        if self._api_key:
            _log.info(f"[JevReranker] initialized model={self._model}")
        else:
            _log.warning(f"[JevReranker] no API key ({api_key_env}), rerank will be skipped")

    def query_text(self, query: str) -> str:
        """The query as sent to the model: prefixed with the API version."""
        return f"Revit {self._revit_version} API: {query}"

    def rerank(
        self,
        query: str,
        documents: list[str],
        top_n: int = 15,
        *,
        strict: bool = False,
    ) -> list[RerankResult]:
        """Score every document, return the top_n by probability (highest first)."""
        if not documents:
            return []
        if not self._api_key:
            if strict:
                raise RerankError(f"no API key ({self._api_key_env})")
            return self._fallback(documents, top_n)
        t0 = time.monotonic()
        try:
            scores = self.score(query, documents)
        except RerankError as e:
            if strict:
                raise
            _log.warning(f"[rerank:jev] {e}, using fallback order")
            return self._fallback(documents, top_n)
        results = [RerankResult(index=i, score=s, text=documents[i]) for i, s in enumerate(scores)]
        results.sort(key=lambda r: (-r.score, r.index))
        _log.info(f"[rerank:jev] {len(documents)} docs in {time.monotonic() - t0:.2f}s "
                  f"scores=[{', '.join(f'{r.score:.3f}' for r in results[:3])}...]")
        return results[:max(0, top_n)]

    def score(self, query: str, documents: list[str]) -> list[float]:
        """Probability per document, in input order. Raises RerankError."""
        if not self._api_key:
            raise RerankError(f"no API key ({self._api_key_env})")
        q = self.query_text(query)
        bs = self._batch_size
        batches = [documents[i:i + bs] for i in range(0, len(documents), bs)]
        client_kwargs: dict[str, Any] = {"timeout": self._timeout}
        if self._proxy:
            client_kwargs["proxy"] = self._proxy
        if self._transport is not None:
            client_kwargs["transport"] = self._transport
        with httpx.Client(**client_kwargs) as client:
            if len(batches) == 1:
                per_batch = [self._score_batch(client, q, batches[0])]
            else:
                workers = min(self._max_concurrency, len(batches))
                with ThreadPoolExecutor(max_workers=workers,
                                        thread_name_prefix="revit-api-docs-jev") as pool:
                    per_batch = list(pool.map(lambda b: self._score_batch(client, q, b), batches))
        return [s for batch in per_batch for s in batch]

    def _score_batch(self, client: httpx.Client, query: str, docs: list[str]) -> list[float]:
        ids = [f"d{i}" for i in range(len(docs))]
        payload = {
            "model": self._model,
            "state": {
                "query": query,
                "documents": {qid: doc[:_JEV_MAX_DOC_CHARS] for qid, doc in zip(ids, docs)},
            },
            "questions": {
                qid: {
                    "type": "noul",
                    "instructions": _JEV_INSTRUCTIONS.format(id=qid),
                    "criteria": _JEV_CRITERIA,
                }
                for qid in ids
            },
        }
        data = self._post(client, payload)
        answers = data.get("answers") if isinstance(data, dict) else None
        if not isinstance(answers, dict):
            raise RerankError("response has no answers object")
        scores: list[float] = []
        for qid in ids:
            answer = answers.get(qid)
            value = answer.get("noul") if isinstance(answer, dict) else None
            if (isinstance(value, bool) or not isinstance(value, (int, float))
                    or not math.isfinite(value) or not 0.0 <= value <= 1.0):
                raise RerankError(f"answer {qid!r} has no noul probability")
            scores.append(float(value))
        return scores

    def _post(self, client: httpx.Client, payload: dict[str, Any]) -> Any:
        headers = {
            "Authorization": f"Bearer {self._api_key}",
            "Content-Type": "application/json",
        }
        attempt = 0
        while True:
            retry_after: float | None = None
            try:
                with self._slots:
                    resp = client.post(self._url, headers=headers, json=payload)
            except httpx.TransportError as e:   # connect errors and timeouts
                error = f"{type(e).__name__}: {e}"
            else:
                if resp.status_code == 200:
                    try:
                        return resp.json()
                    except ValueError as e:
                        raise RerankError("response is not JSON") from e
                error = f"HTTP {resp.status_code}: {resp.text[:200]}"
                if resp.status_code not in _JEV_RETRYABLE:
                    raise RerankError(error)
                retry_after = _retry_after_seconds(resp)
            if attempt >= self._max_retries:
                raise RerankError(f"{error} (gave up after {attempt + 1} attempts)")
            delay = retry_after if retry_after is not None else self._backoff * (2 ** attempt)
            _log.info(f"[rerank:jev] {error}; retry {attempt + 1}/{self._max_retries} in {delay:.1f}s")
            self._sleep(min(delay, _JEV_MAX_BACKOFF))
            attempt += 1


def _retry_after_seconds(resp: httpx.Response) -> float | None:
    value = resp.headers.get("retry-after", "")
    try:
        seconds = float(value)
    except ValueError:
        return None
    return seconds if math.isfinite(seconds) and seconds >= 0 else None


# -- Factory ----------------------------------------------------------------

def _proxy_url(config: dict) -> str | None:
    proxy_cfg = config.get("proxy", {}) or {}
    if proxy_cfg.get("enabled", False):
        return proxy_cfg.get("https") or proxy_cfg.get("http")
    return None


def configured_provider(config: dict) -> str:
    """The provider the config asks for: cohere | jev | none (unknown -> none)."""
    rerank_cfg = config.get("rerank", {}) or {}
    if not rerank_cfg.get("enabled", False):
        return "none"
    provider = str(rerank_cfg.get("provider") or "cohere").strip().lower()
    return provider if provider in PROVIDERS else "none"


def _make_cohere(config: dict, *, primary: bool) -> Reranker | None:
    model_cfg = (config.get("rerank", {}).get("models", {}) or {}).get("cohere", {})
    if not model_cfg:
        if primary:
            _log.warning("[create_reranker] no config for provider=cohere")
        return None
    api_key_env = model_cfg.get("api_key_env", "COHERE_API_KEY")
    if not os.getenv(api_key_env, ""):
        if primary:
            _log.warning(f"[create_reranker] env var {api_key_env} not set, "
                         f"rerank will be disabled")
        else:
            _log.info(f"[create_reranker] env var {api_key_env} not set, "
                      f"no Cohere fallback for jev")
        return None
    return Reranker(
        model=model_cfg.get("model", "rerank-v3.5"),
        api_key_env=api_key_env,
        proxy=_proxy_url(config),
    )


def _make_jev(config: dict) -> JevReranker | None:
    model_cfg = (config.get("rerank", {}).get("models", {}) or {}).get("jev", {})
    if not model_cfg:
        _log.warning("[create_reranker] no config for provider=jev")
        return None
    api_key_env = model_cfg.get("api_key_env", "TYPESAFE_API_KEY")
    if not os.getenv(api_key_env, ""):
        _log.warning(f"[create_reranker] env var {api_key_env} not set, "
                     f"jev rerank will be skipped")
        return None
    return JevReranker(
        model=model_cfg.get("model", "jev-latest"),
        api_key_env=api_key_env,
        base_url=model_cfg.get("base_url") or JEV_URL,
        revit_version=str(config.get("revit_version", "2026")),
        batch_size=model_cfg.get("batch_size", JevReranker.MAX_BATCH),
        max_concurrency=model_cfg.get("max_concurrency", 2),
        max_retries=model_cfg.get("max_retries", 2),
        timeout=model_cfg.get("timeout", 20.0),
        proxy=_proxy_url(config),
    )


def create_rerank_chain(config: dict) -> list[RerankProvider]:
    """Providers to try in order; empty when reranking is off or no key is set.

    provider "jev": Jev first, then Cohere when COHERE_API_KEY is set (the
    fallback for a failed or unconfigured Jev call). provider "cohere":
    Cohere only. provider "none" or rerank.enabled false: nothing.
    """
    rerank_cfg = config.get("rerank", {}) or {}
    if not rerank_cfg.get("enabled", False):
        _log.info("[create_reranker] rerank disabled in config")
        return []
    provider = str(rerank_cfg.get("provider") or "cohere").strip().lower()
    if provider == "none":
        _log.info("[create_reranker] rerank provider is none")
        return []
    if provider not in PROVIDERS:
        _log.warning(f"[create_reranker] unknown provider={provider!r} "
                     f"(use one of {', '.join(PROVIDERS)}), rerank will be disabled")
        return []

    chain: list[RerankProvider] = []
    if provider == "jev":
        jev = _make_jev(config)
        if jev is not None:
            chain.append(jev)
        cohere = _make_cohere(config, primary=False)
    else:
        cohere = _make_cohere(config, primary=True)
    if cohere is not None:
        chain.append(cohere)
    return chain


def create_reranker(config: dict) -> RerankProvider | None:
    """The first provider of `create_rerank_chain`, or None if there is none."""
    chain = create_rerank_chain(config)
    return chain[0] if chain else None
