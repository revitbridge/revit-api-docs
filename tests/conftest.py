"""Shared fixtures: a tiny SQLite data set so retriever tests need no ChromaDB or network."""
from __future__ import annotations

import json
import os
import sqlite3
from pathlib import Path

import pytest

from revit_api_docs import data

KEY_VARS = ("REVIT_API_DOCS_EMBEDDING_API_KEY", "OPENROUTER_API_KEY", "COHERE_API_KEY")

API_ROWS = [
    # (name, full_id, summary, syntax, parameters, remark)
    ("Wall.Create(Document, Curve, ElementId, Boolean) Method",
     "M:Autodesk.Revit.DB.Wall.Create(Autodesk.Revit.DB.Document,Autodesk.Revit.DB.Curve,Autodesk.Revit.DB.ElementId,System.Boolean)",
     "Creates a new rectangular profile wall within the project using the default wall style.",
     "public static Wall Create(Document document, Curve curve, ElementId levelId, bool structural)",
     "document: The document in which the new wall is created. curve: An arc or line.",
     "The wall's base offset is zero."),
    ("Wall.CreateProfileSketch Method", "M:Autodesk.Revit.DB.Wall.CreateProfileSketch",
     "Creates a new Wall profile Sketch.", "public Sketch CreateProfileSketch()", "", ""),
    ("Wall.Create Method", "Wall.Create",
     "Creates a wall within the project, with overloads.", "public static Wall Create(...)", "", ""),
    ("Wall.Flip Method", "M:Autodesk.Revit.DB.Wall.Flip",
     "Flips the wall orientation.", "public void Flip()", "", ""),
    ("Floor.Create(Document, IList<CurveLoop>, ElementId, ElementId) Method",
     "M:Autodesk.Revit.DB.Floor.Create(...)",
     "Creates a floor within the project with the given horizontal profile.",
     "public static Floor Create(Document document, IList<CurveLoop> profile, ElementId floorTypeId, ElementId levelId)",
     "profile: The boundary loops.", ""),
    ("BuiltInParameter.WALL_BASE_OFFSET Field", "F:Autodesk.Revit.DB.BuiltInParameter.WALL_BASE_OFFSET",
     "Base offset of a wall.", "", "", ""),
]

SDK_ROWS = [
    # (project, summary, content, mentioned_apis)
    ("CreateWall", "Demonstrates creating a wall from a line on a level.",
     "Wall wall = Wall.Create(doc, line, levelId, false);", '["Autodesk.Revit.DB.Wall.Create", "Line.CreateBound"]'),
    ("FloorSlab", "Demonstrates creating a floor from a boundary.",
     "Floor floor = Floor.Create(doc, loops, typeId, levelId);", '["Autodesk.Revit.DB.Floor.Create"]'),
]


@pytest.fixture
def tiny_dbs(tmp_path: Path) -> tuple[Path, Path]:
    api_db = tmp_path / "revit_api.db"
    conn = sqlite3.connect(api_db)
    conn.execute(
        "CREATE TABLE revit_api (id INTEGER PRIMARY KEY AUTOINCREMENT, name TEXT NOT NULL, full_id TEXT, "
        "namespace TEXT, content_type TEXT, keywords TEXT, info TEXT, summary TEXT, remark TEXT, "
        "parameters TEXT, exceptions TEXT, return_value TEXT, syntax TEXT, members TEXT, "
        "quality_score REAL, quality_issues TEXT, rewritten INTEGER DEFAULT 0)"
    )
    conn.executemany(
        "INSERT INTO revit_api (name, full_id, summary, syntax, parameters, remark) VALUES (?,?,?,?,?,?)",
        API_ROWS,
    )
    conn.commit()
    conn.close()

    sdk_db = tmp_path / "revit_sdk.db"
    conn = sqlite3.connect(sdk_db)
    conn.execute(
        "CREATE TABLE sdk_info (id INTEGER PRIMARY KEY AUTOINCREMENT, project TEXT, summary TEXT, "
        "content TEXT, mentioned_apis TEXT)"
    )
    conn.executemany(
        "INSERT INTO sdk_info (project, summary, content, mentioned_apis) VALUES (?,?,?,?)", SDK_ROWS
    )
    conn.commit()
    conn.close()
    return api_db, sdk_db


@pytest.fixture
def tiny_data_dir(tiny_dbs, tmp_path: Path) -> Path:
    """A data directory the installer accepts as complete (manifest for the
    current release, empty ChromaDB placeholders), so subprocesses start in
    keyword mode without downloading anything."""
    api_db, sdk_db = tiny_dbs
    root = tmp_path / "data"
    (root / "sqlite").mkdir(parents=True)
    os.replace(api_db, root / "sqlite" / "revit_api.db")
    os.replace(sdk_db, root / "sqlite" / "revit_sdk.db")
    for d in ("chromadb_api", "chromadb_code"):
        (root / "chromadb" / d).mkdir(parents=True)
        (root / "chromadb" / d / "chroma.sqlite3").write_bytes(b"")
    # record_count differs from the SQLite row count on purpose: the shipped
    # data set has the same property and it must not surface as a warning.
    (root / "chromadb" / "chromadb_api" / "meta.json").write_text(
        json.dumps({"record_count": 999}), encoding="utf-8")
    # A manifest for the current release means "installed and verified": only existence is checked.
    (root / "manifest.json").write_text(json.dumps({"release": data.RELEASE_TAG}), encoding="utf-8")
    return root


@pytest.fixture
def no_key_env() -> dict[str, str]:
    """os.environ without any embedding/rerank key, for subprocesses."""
    return {k: v for k, v in os.environ.items() if k not in KEY_VARS}


@pytest.fixture
def anyio_backend():
    """Run @pytest.mark.anyio tests on asyncio only (anyio ships the pytest plugin via mcp)."""
    return "asyncio"
