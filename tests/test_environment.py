"""Environment notes: corpus shape, release ranges, lexical matching, and availability
without any key or data set (tool output while not ready, CLI subcommand)."""
from __future__ import annotations

import asyncio
import subprocess
import sys
from pathlib import Path

import pytest

from revit_api_docs import environment as env
from revit_api_docs import server

ROOT = Path(__file__).resolve().parents[1]


def _ids(notes) -> list[str]:
    return [n.id for n in notes]


def _texts(version: str) -> str:
    return " ".join(n.text for n in env.select(env.parse_version(version)))


# -- corpus -------------------------------------------------------------------

def test_corpus_is_small_curated_english():
    notes = env.load_notes()
    assert 20 <= len(notes) <= 40
    assert len(set(_ids(notes))) == len(notes)
    raw = env.CORPUS.read_text(encoding="utf-8")
    assert raw.isascii()
    for n in notes:
        assert n.source.startswith("https://") and n.tags and n.text
        assert n.text.isascii()


def test_corpus_covers_the_verified_facts():
    assert "net48" in _texts("2024")
    assert "net8.0-windows" in _texts("2025") and "net8.0-windows" in _texts("2026")
    assert "net10.0-windows" in _texts("2025.5") and "net10.0-windows" in _texts("2026.5")
    t2027 = _texts("2027")
    for fact in (".NET 10", "net10.0-windows", ".NET 10 SDK", "10.0.100", '"rollForward": "minor"',
                 "AddInDependencyBase", "ClientIdDependency", "ContextNameDependency",
                 "PublicAssemblies", "Dependencies", "UseAllContextsForDependencyResolution",
                 "Nice3point.Revit.Api.RevitAPI", "dotnet --list-sdks", "runtimeconfig.json",
                 "not a .NET SDK"):
        assert fact in t2027, fact
    assert "2026-11-10" in _texts("2025") and "2026-11-10" in _texts("2026")
    assert "2026.0" in _texts("2026.5")  # do not build 2026.5 against 2026.0 references
    everything = " ".join(n.text for n in env.load_notes())
    assert "jev-latest" in everything and "relevance_rerank" in everything and "30 documents" in everything


def test_corpus_rejects_bad_notes(tmp_path):
    bad = tmp_path / "env.yaml"
    bad.write_text("notes:\n  - {id: a, text: t, revit: '>=2025', tags: [x], source: 'http://x'}\n",
                   encoding="utf-8")
    with pytest.raises(ValueError, match="https"):
        env.load_notes(bad)
    bad.write_text("notes:\n  - {id: a, text: t, revit: 'soon', tags: [x], source: 'https://x'}\n",
                   encoding="utf-8")
    with pytest.raises(ValueError, match="bad Revit range"):
        env.load_notes(bad)
    bad.write_text("notes:\n  - {id: a, text: t, revit: '*', tags: [x], source: 'https://x'}\n"
                   "  - {id: a, text: u, revit: '*', tags: [y], source: 'https://y'}\n", encoding="utf-8")
    with pytest.raises(ValueError, match="duplicate"):
        env.load_notes(bad)


# -- release ranges -----------------------------------------------------------

def test_release_ranges():
    dot5 = env.RevitRange.parse(">=2025.5 <2026 || >=2026.5 <2027")
    assert dot5.contains((2025, 5)) and dot5.contains((2026, 5))
    assert not dot5.contains((2026, 0)) and not dot5.contains((2027, 0)) and not dot5.contains((2025, 0))
    assert env.RevitRange.parse("*").contains((2031, 0))
    assert env.RevitRange.parse("2027").contains((2027, 0))
    assert not env.RevitRange.parse("2027").contains((2027, 5))


def test_versions_in_queries():
    assert env.query_version("which .NET for Revit 2026.5") == (2026, 5)
    assert env.query_version("target framework for Revit 2027 add-in") == (2027, 0)
    assert env.query_version("ElementId 120270 and 20271") is None
    assert env.query_version("Wall.Create") is None
    assert env.parse_version("2026.5") == (2026, 5)
    with pytest.raises(ValueError):
        env.parse_version("26")


# -- matching -----------------------------------------------------------------

def test_target_framework_for_2027():
    notes = env.match("target framework for Revit 2027 add-in")
    assert notes[0].id == "revit-2027-framework"
    assert 1 <= len(notes) <= env.MAX_NOTES
    assert all(n.applies_to((2027, 0)) for n in notes)


def test_matching_filters_by_the_release_in_the_query():
    ids = _ids(env.match("Revit 2024 target framework"))
    assert ids[0] == "revit-2024-framework"
    assert "revit-2027-framework" not in ids
    assert _ids(env.match("which .NET does Revit 2026.5 use"))[0] == "revit-dot5-framework"
    assert env.match("target framework", revit=(2026, 0))[0].id == "revit-2026-framework"


@pytest.mark.parametrize("query", [
    "Wall.Create", "create floor from boundary", "FilteredElementCollector OfCategory",
    "rotate element around axis", "wall location curve", "get selected elements in my add-in",
    "Transaction commit", "zzzz qqqq",
])
def test_api_queries_get_no_notes(query):
    assert env.match(query) == []


def test_section_text():
    text = env.format_section(env.match("target framework for Revit 2027 add-in"))
    assert text.startswith("Environment notes (curated development-environment facts")
    assert "net10.0-windows" in text and "[Revit >=2027 <2028; source: https://" in text
    assert env.format_section([]) == ""


# -- available without data or keys ---------------------------------------------

def test_notes_come_with_the_not_ready_message(monkeypatch):
    st = server._State()
    monkeypatch.setattr(st, "start_background", lambda: None)
    monkeypatch.setattr(server, "_state", st)
    st.status = "downloading"
    st.detail = "revit_api.db 3%"

    text = asyncio.run(server.search_revit_api("target framework for Revit 2027 add-in"))
    status, _, notes = text.partition("\n\n")
    assert "downloading its data set" in status
    assert notes.startswith("Environment notes") and ".NET 10" in notes and "net10.0-windows" in notes
    # API questions keep the plain status message
    assert "Environment notes" not in asyncio.run(server.search_revit_api("Wall.Create"))
    # get_code_examples never adds notes
    assert "Environment notes" not in asyncio.run(server.get_code_examples("target framework 2027"))


def test_notes_follow_search_results(tiny_dbs, monkeypatch, tmp_path):
    from revit_api_docs.config import load_config
    from revit_api_docs.retriever import RAGRetriever

    for name in ("REVIT_API_DOCS_EMBEDDING_API_KEY", "OPENROUTER_API_KEY", "COHERE_API_KEY"):
        monkeypatch.delenv(name, raising=False)
    api_db, sdk_db = tiny_dbs
    r = RAGRetriever(load_config(), str(api_db), str(sdk_db), str(tmp_path / "a"), str(tmp_path / "b"))
    text = server.search_api(r, "target framework for Revit 2027 add-in", 5)
    assert text.startswith("No API entries found")
    assert "\n\nEnvironment notes" in text and "net10.0-windows" in text


def _cli(args: list[str], env_vars: dict[str, str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run([sys.executable, "-m", "revit_api_docs", *args],
                          capture_output=True, text=True, encoding="utf-8", env=env_vars, cwd=str(ROOT))


def test_environment_command_needs_no_data_set(no_key_env, tmp_path):
    empty = tmp_path / "no-data-here"
    env_vars = dict(no_key_env, REVIT_API_DOCS_DATA_DIR=str(empty))
    out = _cli(["environment", "--revit", "2027"], env_vars)
    assert out.returncode == 0, out.stderr
    assert "[revit-2027-framework]" in out.stdout and "Revit 2027 runs on .NET 10" in out.stdout
    assert "[revit-2024-framework]" not in out.stdout
    assert "environment note(s) for Revit 2027" in out.stderr
    assert not empty.exists()  # nothing was downloaded or created

    out = _cli(["environment", "target framework", "--revit", "2024"], env_vars)
    assert out.returncode == 0 and out.stdout.startswith("[revit-2024-framework]")

    out = _cli(["environment", "--revit", "twenty"], env_vars)
    assert out.returncode == 2 and "not a Revit release" in out.stderr
