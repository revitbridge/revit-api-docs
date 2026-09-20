"""Package-level constraints: entry point, dependency boundaries, no cross-repo imports."""
from __future__ import annotations

import subprocess
import sys
import tomllib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _dep_names(specs: list[str]) -> list[str]:
    return sorted(d.split(">")[0].split("<")[0].split("=")[0].split("[")[0].strip() for d in specs)


def test_declared_dependencies():
    data = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    assert data["project"]["name"] == "revit-api-docs"
    assert _dep_names(data["project"]["dependencies"]) == ["chromadb", "httpx", "mcp", "openai", "pyyaml"]
    assert data["project"]["scripts"] == {"revit-api-docs": "revit_api_docs.server:main"}


def test_never_depends_on_revit_bridge():
    """Repositories depend on each other only through published packages, and
    revit-bridge must stay installable without this package (no cycle)."""
    data = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    everything = list(data["project"]["dependencies"])
    for extra in data["project"].get("optional-dependencies", {}).values():
        everything.extend(extra)
    assert "revit-bridge" not in _dep_names(everything)
    for path in list((ROOT / "revit_api_docs").rglob("*.py")) + list((ROOT / "pipeline").rglob("*.py")):
        text = path.read_text(encoding="utf-8")
        assert "revit_bridge" not in text, path
        assert "sys.path.insert" not in text, path
        assert "mcp_bridge" not in text and "intent_bridge" not in text, path


def test_wheel_ships_runtime_only():
    data = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    assert data["tool"]["hatch"]["build"]["targets"]["wheel"]["packages"] == ["revit_api_docs"]
    assert "pipeline" in data["tool"]["hatch"]["build"]["targets"]["sdist"]["include"]
    assert (ROOT / "revit_api_docs" / "default_config.yaml").is_file()
    assert (ROOT / "revit_api_docs" / "prompts" / "pipeline.rewrite_query.md").is_file()


def test_importing_the_server_stays_light():
    """`revit-api-docs --version` must not pay for chromadb/openai imports."""
    code = (
        "import sys; import revit_api_docs.server; "
        "print(sorted(m for m in ('chromadb', 'openai', 'onnxruntime') if m in sys.modules))"
    )
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True, cwd=ROOT)
    assert out.stdout.strip() == "[]"


def test_cli_version_and_help():
    from revit_api_docs import __version__
    from revit_api_docs.server import main

    out = subprocess.run([sys.executable, "-m", "revit_api_docs", "--version"],
                         capture_output=True, text=True, check=True, cwd=ROOT)
    assert out.stdout.strip() == f"revit-api-docs {__version__}"
    assert main.__module__ == "revit_api_docs.server"
