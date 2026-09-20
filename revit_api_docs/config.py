"""Configuration: bundled defaults plus environment overrides.

The MCP server needs no config file. `load_config()` reads the packaged
`default_config.yaml` (or the file named by `REVIT_API_DOCS_CONFIG`) and then
applies the `REVIT_API_DOCS_*` environment variables documented in the README.
The dict keeps the shape the retriever and the pipeline tools expect
(`embedding.models.<provider>`, `llm.models.<provider>`, `rerank`, `retrieval`,
`sdk`); API keys are looked up by the environment variable named in
`api_key_env`, never stored in the dict.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import yaml

ENV_CONFIG = "REVIT_API_DOCS_CONFIG"
ENV_EMBEDDING_API_KEY = "REVIT_API_DOCS_EMBEDDING_API_KEY"
ENV_EMBEDDING_BASE_URL = "REVIT_API_DOCS_EMBEDDING_BASE_URL"
ENV_EMBEDDING_MODEL = "REVIT_API_DOCS_EMBEDDING_MODEL"
ENV_EMBEDDING_PROVIDER = "REVIT_API_DOCS_EMBEDDING_PROVIDER"

# Keys tried, in order, when REVIT_API_DOCS_EMBEDDING_API_KEY is unset.
EMBEDDING_KEY_FALLBACKS = ("OPENROUTER_API_KEY",)

_DEFAULT_CONFIG = Path(__file__).with_name("default_config.yaml")


def load_config(config_path: str | os.PathLike[str] | None = None) -> dict[str, Any]:
    """Load the YAML config and apply environment overrides."""
    path = Path(config_path or os.getenv(ENV_CONFIG) or _DEFAULT_CONFIG)
    with open(path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f) or {}
    apply_env_overrides(config, os.environ)
    return config


def apply_env_overrides(config: dict[str, Any], env: os._Environ[str] | dict[str, str]) -> None:
    """Mutate `config` in place from REVIT_API_DOCS_* variables."""
    emb = config.setdefault("embedding", {})
    provider = env.get(ENV_EMBEDDING_PROVIDER) or emb.get("provider", "openai")
    emb["provider"] = provider
    model_cfg = emb.setdefault("models", {}).setdefault(provider, {})

    if env.get(ENV_EMBEDDING_MODEL):
        model_cfg["model"] = env[ENV_EMBEDDING_MODEL]
    if env.get(ENV_EMBEDDING_BASE_URL) and provider == "openai":
        model_cfg["base_url"] = env[ENV_EMBEDDING_BASE_URL]

    # Resolve which variable holds the embedding key so the provider can read it.
    if provider == "openai":
        if env.get(ENV_EMBEDDING_API_KEY):
            model_cfg["api_key_env"] = ENV_EMBEDDING_API_KEY
        else:
            for name in EMBEDDING_KEY_FALLBACKS:
                if env.get(name):
                    model_cfg["api_key_env"] = name
                    break
            else:
                model_cfg.setdefault("api_key_env", ENV_EMBEDDING_API_KEY)


def load_dotenv(root: str | Path | None = None) -> None:
    """Read `.env` in `root` (default: current directory) into os.environ.

    Existing variables are not overwritten. Used by the pipeline tools; the
    MCP server relies on the host's environment only.
    """
    env = Path(root or Path.cwd()) / ".env"
    if not env.exists():
        return
    for line in env.read_text(encoding="utf-8", errors="ignore").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))


def get_api_key(env_name: str) -> str:
    """Return the key held by `env_name`, or raise ValueError when unset."""
    key = os.getenv(env_name, "")
    if not key:
        raise ValueError(f"environment variable {env_name} is not set")
    return key


_config: dict[str, Any] | None = None


def get_config() -> dict[str, Any]:
    """Process-wide config singleton (first call loads it)."""
    global _config
    if _config is None:
        _config = load_config()
    return _config


def get_proxy_url() -> str | None:
    """Proxy URL from HTTPS_PROXY/HTTP_PROXY, else from the config, else None."""
    from_env = (
        os.getenv("HTTPS_PROXY") or os.getenv("HTTP_PROXY")
        or os.getenv("https_proxy") or os.getenv("http_proxy")
    )
    if from_env:
        return from_env
    try:
        proxy_cfg = get_config().get("proxy", {})
    except FileNotFoundError:
        return None
    if proxy_cfg.get("enabled"):
        return proxy_cfg.get("https") or proxy_cfg.get("http") or None
    return None
