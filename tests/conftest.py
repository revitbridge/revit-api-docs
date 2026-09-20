"""Shared fixtures: a tiny SQLite data set so retriever tests need no ChromaDB or network."""
from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest

API_ROWS = [
    # (name, full_id, summary, syntax, parameters, remark)
    ("Wall.Create(Document, Curve, ElementId, Boolean) Method",
     "M:Autodesk.Revit.DB.Wall.Create(Autodesk.Revit.DB.Document,Autodesk.Revit.DB.Curve,Autodesk.Revit.DB.ElementId,System.Boolean)",
     "Creates a new rectangular profile wall within the project using the default wall style.",
     "public static Wall Create(Document document, Curve curve, ElementId levelId, bool structural)",
     "document: The document in which the new wall is created. curve: An arc or line.",
     "The wall's base offset is zero."),
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
def anyio_backend():
    """Run @pytest.mark.anyio tests on asyncio only (anyio ships the pytest plugin via mcp)."""
    return "asyncio"
