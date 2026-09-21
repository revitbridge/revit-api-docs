"""Keyword-mode ranking against the shipped revit_api.db / revit_sdk.db (no API key).

The two SQLite files (34 MB) are looked up in REVIT_API_DOCS_DATA_DIR or the
default data directory. When REVIT_API_DOCS_TEST_DOWNLOAD=1 they are fetched
from the data release into that directory first (CI does this and caches the
files); otherwise the module is skipped.
"""
from __future__ import annotations

import os
from pathlib import Path

import pytest

from revit_api_docs import data
from revit_api_docs.config import load_config
from revit_api_docs.retriever import RAGRetriever


def _shipped_dbs() -> tuple[Path, Path] | None:
    env = os.environ.get("REVIT_API_DOCS_DATA_DIR")
    roots = [Path(env)] if env else []
    roots.append(data.default_data_dir())
    for root in roots:
        p = data.paths(root)
        if p.api_db.is_file() and p.sdk_db.is_file():
            return p.api_db, p.sdk_db
    if os.environ.get("REVIT_API_DOCS_TEST_DOWNLOAD") == "1":
        root = roots[0]
        for art in data.ARTIFACTS:
            if not art.extract and not (root / art.target).is_file():
                data.install_artifact(art, root, progress=data.stderr_progress)
        p = data.paths(root)
        return p.api_db, p.sdk_db
    return None


@pytest.fixture(scope="module")
def shipped():
    found = _shipped_dbs()
    if found is None:
        pytest.skip("shipped data set not available (set REVIT_API_DOCS_DATA_DIR or REVIT_API_DOCS_TEST_DOWNLOAD=1)")
    api_db, sdk_db = found
    for name in ("REVIT_API_DOCS_EMBEDDING_API_KEY", "OPENROUTER_API_KEY", "COHERE_API_KEY"):
        os.environ.pop(name, None)
    r = RAGRetriever(load_config(), str(api_db), str(sdk_db), "unused", "unused")
    assert r.vector_search_enabled is False
    return r


def _names(r: RAGRetriever, query: str, n: int = 5) -> list[str]:
    results = r.search(query, api_top_k=n, code_top_k=0, rewrite=False)
    return [i.name for i in results.api_items]


def test_task_query_finds_floor_create_without_a_key(shipped):
    names = _names(shipped, "create floor from boundary", 3)
    assert any(n.startswith("Floor.Create") for n in names), names


def test_task_query_finds_element_transform_utils(shipped):
    # Missed entirely by the old arbitrary LIMIT 200 subset.
    names = _names(shipped, "rotate element around axis", 3)
    assert any(n.startswith("ElementTransformUtils") for n in names), names


def test_api_name_queries(shipped):
    assert _names(shipped, "Wall.Create", 1) == ["Wall.Create Method"]
    names = _names(shipped, "FilteredElementCollector OfCategory", 2)
    assert names[0] == "FilteredElementCollector.OfCategory Method", names


def test_keyword_search_is_deterministic(shipped):
    first = _names(shipped, "create floor from boundary", 10)
    assert first == _names(shipped, "create floor from boundary", 10)
