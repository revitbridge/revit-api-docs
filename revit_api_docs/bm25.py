"""BM25 keyword index over the API reference (SQLite FTS5).

The downloaded revit_api.db is checked against the release SHA-256 and is
never written to. The index is a separate file next to it,
`revit_api_fts.db`, built on first use (under a second for the 27,596 rows
of the shipped data set). Its `meta` table records the SHA-256 of the source
database and the index format version; when either differs from the current
ones the index is rebuilt. The installer's manifest.json supplies the
SHA-256 when it describes the same file, otherwise the file is hashed once
per process.

Index: one FTS5 table `api_fts` with the columns name, full_id, summary,
syntax and name_words (the camel-case parts of the identifiers in name, for
example "Element Id" for "ElementId"), tokenizer unicode61, rowid = the
revit_api row id. Ranking: bm25() with column weights 4 / 1 / 1 / 0.5 / 2.

Query: split on anything that is not an ASCII letter or digit; each token is
kept whole and also split at camel-case boundaries; terms are lower-cased,
terms shorter than two characters and a small set of English stop words are
dropped, and the rest are OR-ed, so a misspelt or unknown identifier still
matches on its parts.
"""
from __future__ import annotations

import logging
import os
import re
import sqlite3
import threading
import time
from pathlib import Path

log = logging.getLogger("revit_api_docs.bm25")

INDEX_FILENAME = "revit_api_fts.db"
# Bump when the table layout, the tokenizer or name_words() changes:
# existing index files are then rebuilt on first use.
FORMAT_VERSION = 1

COLUMNS = ("name", "full_id", "summary", "syntax", "name_words")
WEIGHTS = (4.0, 1.0, 1.0, 0.5, 2.0)   # same order as COLUMNS

_WORD = re.compile(r"[A-Za-z0-9]+")
_CAMEL = re.compile(r"[A-Z]+(?=[A-Z][a-z])|[A-Z]?[a-z]+|[A-Z]+|\d+")
STOP_WORDS = frozenset("""
a an the of to from its it in on for by and or with into is are be how what which all this that
these those do does my i we you revit api please
""".split())


def camel_parts(token: str) -> list[str]:
    """Split at camel-case boundaries: FilteredElementCollector -> Filtered, Element, Collector."""
    return _CAMEL.findall(token)


def name_words(name: str) -> str:
    """Camel-case parts of the compound identifiers in an entry name."""
    words: list[str] = []
    for tok in _WORD.findall(name or ""):
        parts = camel_parts(tok)
        if len(parts) > 1:
            words.extend(parts)
    return " ".join(words)


def query_terms(query: str) -> list[str]:
    """Distinct lower-case search terms of a query, in query order."""
    terms: list[str] = []
    for tok in _WORD.findall(query or ""):
        for candidate in [tok, *camel_parts(tok)]:
            c = candidate.lower()
            if len(c) < 2 or c in STOP_WORDS or c in terms:
                continue
            terms.append(c)
    return terms


def fts5_available() -> bool:
    """True when this Python's sqlite3 module was built with FTS5."""
    try:
        conn = sqlite3.connect(":memory:")
        try:
            conn.execute("CREATE VIRTUAL TABLE temp.fts5_probe USING fts5(x)")
        finally:
            conn.close()
    except sqlite3.Error:
        return False
    return True


def _read_only_uri(path: Path) -> str:
    return path.resolve().as_uri() + "?mode=ro"


def source_sha256(api_db: Path) -> str:
    """SHA-256 of the source database.

    Taken from the installer's manifest.json when it describes this file
    (same path, same size); hashed otherwise (hand-copied or locally built
    trees, test fixtures).
    """
    from . import data

    api_db = Path(api_db)
    size = api_db.stat().st_size
    root = api_db.parent.parent
    manifest = data.read_manifest(root) or {}
    files = manifest.get("files") if isinstance(manifest, dict) else None
    entry = files.get("revit_api.db") if isinstance(files, dict) else None
    if isinstance(entry, dict):
        target = entry.get("target")
        sha = entry.get("sha256")
        if (isinstance(sha, str) and len(sha) == 64 and entry.get("size") == size
                and isinstance(target, str) and (root / target).resolve() == api_db.resolve()):
            return sha
    return data.sha256_file(api_db)


class Bm25Index:
    """The FTS5 index file for one revit_api.db; built or rebuilt lazily."""

    def __init__(self, api_db_path: str | os.PathLike[str]):
        self._api_db = Path(api_db_path)
        self._path = self._api_db.with_name(INDEX_FILENAME)
        self._lock = threading.Lock()
        self._ready: Path | None = None   # index file checked (or built) by this process

    @property
    def path(self) -> Path:
        """Where the index lives: next to revit_api.db."""
        return self._path

    def ensure(self) -> Path:
        """Path of an index that matches the source; builds it when missing or stale.

        Raises sqlite3.Error or OSError when the index can be neither read nor
        built (no FTS5, read-only data directory, unreadable source).
        """
        if self._ready is not None:
            return self._ready
        with self._lock:
            if self._ready is None:
                sha = source_sha256(self._api_db)
                self._ready = self._path if self._is_current(self._path, sha) else self._build(sha)
            return self._ready

    @staticmethod
    def read_meta(path: Path) -> dict[str, str] | None:
        """The index's meta table, or None when the file is missing or unusable."""
        if not Path(path).is_file():
            return None
        try:
            conn = sqlite3.connect(_read_only_uri(Path(path)), uri=True)
            try:
                meta = dict(conn.execute("SELECT key, value FROM meta").fetchall())
                conn.execute("SELECT rowid FROM api_fts LIMIT 1").fetchall()
            finally:
                conn.close()
        except sqlite3.Error:
            return None
        return meta

    def _is_current(self, path: Path, sha: str) -> bool:
        meta = self.read_meta(path)
        return (meta is not None and meta.get("format_version") == str(FORMAT_VERSION)
                and meta.get("source_sha256") == sha)

    def _build(self, sha: str) -> Path:
        """Write the index to a temporary file, then move it into place."""
        t0 = time.monotonic()
        tmp = self._path.with_name(f"{self._path.name}.{os.getpid()}-{threading.get_ident()}.tmp")
        tmp.unlink(missing_ok=True)
        src = sqlite3.connect(_read_only_uri(self._api_db), uri=True)
        try:
            rows = src.execute("SELECT id, name, full_id, summary, syntax FROM revit_api").fetchall()
        finally:
            src.close()
        try:
            conn = sqlite3.connect(tmp)
            try:
                conn.execute(f"CREATE VIRTUAL TABLE api_fts USING fts5({', '.join(COLUMNS)}, "
                             "tokenize='unicode61')")
                conn.executemany(
                    "INSERT INTO api_fts (rowid, name, full_id, summary, syntax, name_words) "
                    "VALUES (?, ?, ?, ?, ?, ?)",
                    ((rid, n or "", f or "", sm or "", sy or "", name_words(n or ""))
                     for rid, n, f, sm, sy in rows),
                )
                conn.execute("CREATE TABLE meta (key TEXT PRIMARY KEY, value TEXT NOT NULL)")
                conn.executemany("INSERT INTO meta (key, value) VALUES (?, ?)", [
                    ("format_version", str(FORMAT_VERSION)),
                    ("source_sha256", sha),
                    ("rows", str(len(rows))),
                ])
                conn.commit()
            finally:
                conn.close()
        except BaseException:
            try:
                tmp.unlink(missing_ok=True)   # never leave a half-built file behind
            except OSError:
                pass
            raise
        try:
            os.replace(tmp, self._path)
        except OSError as e:
            # Windows refuses to replace a file another process has open.
            if self._is_current(self._path, sha):   # that process built the same index
                try:
                    tmp.unlink()
                except OSError:
                    pass
                return self._path
            log.warning("could not move the keyword index into place (%s); using %s for this "
                        "process", e, tmp)
            return tmp
        log.info("built keyword index %s: %d rows in %.2fs", self._path, len(rows),
                 time.monotonic() - t0)
        return self._path

    def search(self, query: str, k: int) -> list[tuple[int, float]]:
        """Top k (revit_api row id, BM25 score) pairs, best first; higher score is better."""
        terms = query_terms(query)
        if not terms or k <= 0:
            return []
        path = self.ensure()
        match = " OR ".join(f'"{t}"' for t in terms)
        weights = ", ".join(str(w) for w in WEIGHTS)
        conn = sqlite3.connect(_read_only_uri(path), uri=True)
        try:
            hits = conn.execute(
                f"SELECT rowid, bm25(api_fts, {weights}) AS s FROM api_fts "
                "WHERE api_fts MATCH ? ORDER BY s, rowid LIMIT ?",
                (match, int(k)),
            ).fetchall()
        finally:
            conn.close()
        # bm25() is lower-is-better (negative); flip the sign for a score.
        return [(int(rowid), -float(s)) for rowid, s in hits]
