"""Minimal chat-completions client (OpenAI-compatible, used through OpenRouter).

Used by the offline pipeline tools (SDK summarisation, quality agents) and by
the optional query rewriting in the retriever. Configuration comes from the
`llm` section of the config; the key from the environment variable named in
`api_key_env`. The MCP server never needs it.
"""
from __future__ import annotations

import json
import os
import sys
from typing import Any, Generator

import time

import httpx

# SDK summarisation calls run long; use a generous timeout.
_DEFAULT_TIMEOUT = 120.0

# Transient statuses (OpenRouter / upstream hiccups) that are retried.
# 403 has been observed as a transient refusal from Anthropic via OpenRouter.
_RETRYABLE_STATUS = frozenset({403, 408, 409, 429, 500, 502, 503, 504, 529})
_MAX_RETRIES = 4          # total attempts = 1 + retries
_BACKOFF_BASE = 1.5       # backoff base in seconds: 1.5, 3.0, 4.5 ...
_MAX_403_RETRIES = 1      # 403 is usually permissions/quota: retry once, then give up


class LLMClient:
    def __init__(
        self,
        base_url: str,
        api_key: str,
        model: str,
        temperature: float = 0.3,
        max_tokens: int = 4096,
        timeout: float = _DEFAULT_TIMEOUT,
        proxy: str | None = None,
    ):
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens

        # Configure HTTP proxy if provided (helps with Gemini routing in restricted networks)
        if proxy:
            self._client = httpx.Client(
                timeout=timeout,
                proxy=proxy,
            )
        else:
            self._client = httpx.Client(timeout=timeout)

    def close(self) -> None:
        """Close the underlying httpx.Client and its connection pool."""
        try:
            self._client.close()
        except Exception:
            pass

    def __enter__(self) -> "LLMClient":
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.close()

    def _build_request(
        self,
        prompt: str,
        system_prompt: str | None = None,
        stream: bool = False,
        messages: list[dict] | None = None,
    ) -> tuple[str, dict, dict]:
        """Build request URL, headers, and payload.

        If `messages` is provided it is used verbatim as the chat messages array
        (structured multi-turn: system + per-role turns). Otherwise a simple
        [system, user] pair is built from `system_prompt` / `prompt`.
        """
        url = f"{self.base_url}/chat/completions"
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "X-Title": "revit-api-docs",
        }

        if messages is not None:
            msgs = messages
        else:
            _system = system_prompt or (
                "You are an expert assistant for summarizing Revit SDK C# sample code. "
                "Always respond in English."
            )
            msgs = [
                {"role": "system", "content": _system},
                {"role": "user", "content": prompt},
            ]

        payload: dict[str, Any] = {
            "model": self.model,
            "messages": msgs,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "stream": stream,
        }
        return url, headers, payload

    def generate_text(self, prompt: str, system_prompt: str | None = None) -> str:
        """Chat-completions call; returns the first message's content.

        Retries transient statuses with a short backoff (see _RETRYABLE_STATUS).
        """
        url, headers, payload = self._build_request(prompt, system_prompt, stream=False)

        import logging
        _llm_log = logging.getLogger("revit_api_docs.llm_client")

        resp = None
        last_err: Exception | None = None
        consecutive_403 = 0
        for attempt in range(1, _MAX_RETRIES + 1):
            try:
                resp = self._client.post(url, headers=headers, json=payload)
                if resp.status_code in _RETRYABLE_STATUS and attempt < _MAX_RETRIES:
                    # 403 circuit breaker: stop after _MAX_403_RETRIES so a lasting
                    # permission/quota error is not retried as if transient.
                    if resp.status_code == 403:
                        consecutive_403 += 1
                        if consecutive_403 > _MAX_403_RETRIES:
                            _llm_log.warning(
                                f"[generate_text] {self.model} HTTP 403 x{consecutive_403} "
                                "— circuit break, no more retries."
                            )
                            resp.raise_for_status()
                    else:
                        consecutive_403 = 0
                    body = resp.text[:300].replace("\n", " ")
                    wait = _BACKOFF_BASE * attempt
                    _llm_log.warning(
                        f"[generate_text] {self.model} HTTP {resp.status_code} "
                        f"(attempt {attempt}/{_MAX_RETRIES}) — retrying in {wait:.1f}s. Body: {body}"
                    )
                    time.sleep(wait)
                    continue
                resp.raise_for_status()
                break
            except (httpx.TransportError, httpx.TimeoutException) as e:
                # transport-level hiccup: retry as well
                last_err = e
                if attempt < _MAX_RETRIES:
                    wait = _BACKOFF_BASE * attempt
                    _llm_log.warning(
                        f"[generate_text] {self.model} transport error "
                        f"(attempt {attempt}/{_MAX_RETRIES}): {e} — retrying in {wait:.1f}s"
                    )
                    time.sleep(wait)
                    continue
                raise
            except httpx.HTTPStatusError as e:
                # non-retryable status or retries exhausted: raise with the body for diagnosis
                body = e.response.text[:500] if e.response is not None else ""
                raise httpx.HTTPStatusError(
                    f"{e} | OpenRouter response: {body}",
                    request=e.request, response=e.response,
                ) from None

        if resp is None:  # unreachable in practice
            raise last_err or RuntimeError("generate_text: no response")

        data = resp.json()

        choices = data.get("choices") or []
        if not choices:
            return ""

        choice = choices[0]
        finish_reason = choice.get("finish_reason", "unknown")
        message = choice.get("message") or {}
        content = message.get("content") or ""

        _llm_log.info(
            f"[generate_text] finish_reason={finish_reason} "
            f"content_len={len(content)} max_tokens={self.max_tokens}"
        )
        if finish_reason == "length":
            _llm_log.warning(
                f"[generate_text] RESPONSE TRUNCATED — hit max_tokens={self.max_tokens}. "
                f"Increase max_tokens in config."
            )

        return str(content)

    def generate_stream(
        self, prompt: str, system_prompt: str | None = None,
        messages: list[dict] | None = None,
    ) -> Generator[str, None, None]:
        """Stream tokens over SSE. `messages` may carry a structured multi-turn
        conversation (system + per-role entries) instead of `prompt`."""
        url, headers, payload = self._build_request(
            prompt, system_prompt, stream=True, messages=messages,
        )

        with self._client.stream("POST", url, headers=headers, json=payload) as resp:
            resp.raise_for_status()
            for line in resp.iter_lines():
                if not line or not line.startswith("data: "):
                    continue
                data_str = line[6:]  # strip "data: "
                if data_str.strip() == "[DONE]":
                    break
                try:
                    chunk = json.loads(data_str)
                    delta = chunk.get("choices", [{}])[0].get("delta", {})
                    token = delta.get("content", "")
                    if token:
                        yield token
                except (json.JSONDecodeError, IndexError):
                    continue

    def stream_print(
        self, prompt: str, system_prompt: str | None = None
    ) -> str:
        """Stream to stdout as tokens arrive; return the full text."""
        full_text = []
        for token in self.generate_stream(prompt, system_prompt):
            print(token, end="", flush=True)
            full_text.append(token)
        print()  # final newline
        return "".join(full_text)


def create_llm_client(config: dict[str, Any], provider_override: str | None = None) -> LLMClient:
    """Build an LLMClient from config["llm"].

    Args:
        config:            the loaded config dict
        provider_override: use this llm.models entry instead of llm.provider
    """
    llm_cfg = config.get("llm", {})
    provider = provider_override or llm_cfg.get("provider", "claude")
    models_cfg = llm_cfg.get("models", {})
    model_cfg = models_cfg.get(provider, {})

    if not model_cfg:
        available = list(models_cfg.keys())
        raise RuntimeError(
            f"no llm.models entry for provider={provider!r} in the config; "
            f"available: {available}"
        )

    model = model_cfg.get("model")
    base_url = model_cfg.get("base_url") or config.get("openrouter", {}).get("base_url")
    api_key_env = (
        model_cfg.get("api_key_env")
        or config.get("openrouter", {}).get("api_key_env", "OPENROUTER_API_KEY")
    )
    api_key = os.getenv(api_key_env, "")

    if not model:
        raise RuntimeError(f"llm.models.{provider}.model is not set in the config")
    if not base_url:
        raise RuntimeError(f"llm.models.{provider}.base_url is not set in the config")
    if not api_key:
        raise RuntimeError(
            f"environment variable {api_key_env} is not set; it must hold the API key "
            f"for the {provider} model"
        )

    temperature = llm_cfg.get("temperature", 0.3)
    max_tokens  = llm_cfg.get("max_tokens", 4096)

    # Proxy: model-level override > global proxy config
    proxy_cfg   = config.get("proxy", {})
    proxy_url: str | None = None
    if proxy_cfg.get("enabled", False):
        proxy_url = proxy_cfg.get("https") or proxy_cfg.get("http")

    return LLMClient(
        base_url=base_url,
        api_key=api_key,
        model=model,
        temperature=temperature,
        max_tokens=max_tokens,
        proxy=proxy_url,
    )

