"""Data set location and first-run download.

The API reference (SQLite) and the vector stores (ChromaDB directories) are
not in the wheel; they come from the GitHub Release `v1.0-data` of this
repository. `ensure_data()` downloads whatever is missing into the data
directory, verifies each file against the SHA-256 recorded below, unpacks the
tarballs and writes `manifest.json`.

Data directory (override with REVIT_API_DOCS_DATA_DIR):
    Windows  %LOCALAPPDATA%/revit-api-docs
    macOS    ~/Library/Application Support/revit-api-docs
    Linux    $XDG_DATA_HOME/revit-api-docs  (default ~/.local/share)

Layout inside it (same as the old repository's data/ tree, so an existing
copy can be pointed at directly):
    sqlite/revit_api.db
    sqlite/revit_sdk.db
    chromadb/chromadb_api/
    chromadb/chromadb_code/
    manifest.json
"""
from __future__ import annotations

import hashlib
import json
import logging
import os
import shutil
import sys
import tarfile
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import httpx

log = logging.getLogger("revit_api_docs.data")

ENV_DATA_DIR = "REVIT_API_DOCS_DATA_DIR"
ENV_RELEASE_URL = "REVIT_API_DOCS_RELEASE_URL"

RELEASE_TAG = "v1.0-data"
RELEASE_URL = f"https://github.com/revitbridge/revit-api-docs/releases/download/{RELEASE_TAG}"

ProgressFn = Callable[[str, int, int], None]


@dataclass(frozen=True)
class Artifact:
    """One release asset: where it lands and how to check it."""

    name: str
    sha256: str
    size: int
    target: str          # path inside the data dir (file, or directory for tarballs)
    extract: bool = False


ARTIFACTS: tuple[Artifact, ...] = (
    Artifact("revit_api.db", "38f78297b11b5fb4aa8a0c609e2a0535b5d2f6770ea895c8468742e6d4a7bf72",
             33_611_776, "sqlite/revit_api.db"),
    Artifact("revit_sdk.db", "f381f392485cd4910b2b826062414b5331c29f165269412ebb59569dd176e474",
             954_368, "sqlite/revit_sdk.db"),
    Artifact("chromadb_api.tar.gz", "2db7c2fafa45defb1cc56522c5c5455741549d42d56bbec918a93554c4b15c73",
             299_124_932, "chromadb/chromadb_api", extract=True),
    Artifact("chromadb_code.tar.gz", "3ee38a4e44e4b41154faeafd431c830ca54275d2738897971e5780cb493118ad",
             1_792_184, "chromadb/chromadb_code", extract=True),
)

TOTAL_DOWNLOAD_BYTES = sum(a.size for a in ARTIFACTS)


@dataclass(frozen=True)
class DataPaths:
    root: Path
    api_db: Path
    sdk_db: Path
    chromadb_api: Path
    chromadb_code: Path
    manifest: Path


def default_data_dir() -> Path:
    if sys.platform == "win32":
        base = os.environ.get("LOCALAPPDATA") or str(Path.home() / "AppData" / "Local")
    elif sys.platform == "darwin":
        base = str(Path.home() / "Library" / "Application Support")
    else:
        base = os.environ.get("XDG_DATA_HOME") or str(Path.home() / ".local" / "share")
    return Path(base) / "revit-api-docs"


def data_dir() -> Path:
    env = os.environ.get(ENV_DATA_DIR)
    return Path(env).expanduser() if env else default_data_dir()


def release_url() -> str:
    return os.environ.get(ENV_RELEASE_URL, RELEASE_URL).rstrip("/")


def paths(root: Path | None = None) -> DataPaths:
    root = Path(root) if root is not None else data_dir()
    return DataPaths(
        root=root,
        api_db=root / "sqlite" / "revit_api.db",
        sdk_db=root / "sqlite" / "revit_sdk.db",
        chromadb_api=root / "chromadb" / "chromadb_api",
        chromadb_code=root / "chromadb" / "chromadb_code",
        manifest=root / "manifest.json",
    )


def _installed(root: Path, art: Artifact, check_size: bool) -> bool:
    target = root / art.target
    if art.extract:
        return (target / "chroma.sqlite3").is_file()
    if not target.is_file():
        return False
    return not check_size or target.stat().st_size == art.size


def read_manifest(root: Path) -> dict | None:
    p = root / "manifest.json"
    if not p.is_file():
        return None
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def missing(root: Path | None = None) -> list[Artifact]:
    """Artifacts that must be (re)downloaded.

    A manifest for this release means the installer verified every file, so
    only presence is checked. A manifest from another release marks everything
    stale. Without a manifest (a hand-copied data tree) files are trusted when
    present with the expected size.
    """
    root = Path(root) if root is not None else data_dir()
    manifest = read_manifest(root)
    if manifest is not None and manifest.get("release") != RELEASE_TAG:
        return list(ARTIFACTS)
    return [a for a in ARTIFACTS if not _installed(root, a, check_size=manifest is None)]


def is_ready(root: Path | None = None) -> bool:
    return not missing(root)


def sha256_file(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            block = f.read(chunk)
            if not block:
                break
            h.update(block)
    return h.hexdigest()


def download_file(url: str, dest: Path, expected_size: int,
                  progress: ProgressFn | None = None, name: str = "",
                  timeout: float = 60.0) -> None:
    """Stream `url` to `dest`, resuming a partial `dest` with a Range request."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    have = dest.stat().st_size if dest.exists() else 0
    if have > expected_size:
        dest.unlink()
        have = 0
    if have == expected_size:
        return  # a previous run finished the download but not the checks
    headers = {"Range": f"bytes={have}-"} if have else {}
    mode = "ab" if have else "wb"

    with httpx.Client(follow_redirects=True, timeout=timeout) as client:
        with client.stream("GET", url, headers=headers) as resp:
            if have and resp.status_code == 200:
                # Server ignored the range: start over.
                have, mode = 0, "wb"
            elif resp.status_code not in (200, 206):
                resp.raise_for_status()
            done = have
            last_report = time.monotonic()
            with open(dest, mode) as out:
                for chunk in resp.iter_bytes(1 << 20):
                    out.write(chunk)
                    done += len(chunk)
                    now = time.monotonic()
                    if progress and (now - last_report >= 1.0 or done >= expected_size):
                        progress(name or dest.name, done, expected_size)
                        last_report = now
    if progress:
        progress(name or dest.name, dest.stat().st_size, expected_size)


def _extract_tarball(archive: Path, target: Path) -> None:
    """Unpack `archive` (which holds one top-level directory) to `target`."""
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp = Path(tempfile.mkdtemp(prefix=f".{target.name}-", dir=target.parent))
    try:
        with tarfile.open(archive, "r:gz") as tar:
            try:
                tar.extractall(tmp, filter="data")
            except TypeError:  # Python < 3.11.4 without the filter argument
                tar.extractall(tmp)
        entries = [p for p in tmp.iterdir()]
        src = entries[0] if len(entries) == 1 and entries[0].is_dir() else tmp
        if target.exists():
            shutil.rmtree(target)
        shutil.move(str(src), str(target))
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def install_artifact(art: Artifact, root: Path, progress: ProgressFn | None = None) -> None:
    """Download one artifact, verify it and put it in place."""
    url = f"{release_url()}/{art.name}"
    part = root / "downloads" / f"{art.name}.part"
    log.info("downloading %s (%.1f MB) from %s", art.name, art.size / 1e6, url)
    download_file(url, part, art.size, progress=progress, name=art.name)

    digest = sha256_file(part)
    if digest != art.sha256:
        part.unlink(missing_ok=True)
        raise RuntimeError(
            f"{art.name}: SHA-256 mismatch (got {digest[:16]}..., expected "
            f"{art.sha256[:16]}...); the partial download was discarded, run again"
        )

    target = root / art.target
    if art.extract:
        _extract_tarball(part, target)
        part.unlink(missing_ok=True)
    else:
        target.parent.mkdir(parents=True, exist_ok=True)
        os.replace(part, target)


def write_manifest(root: Path) -> None:
    root.mkdir(parents=True, exist_ok=True)
    manifest = {
        "release": RELEASE_TAG,
        "source": release_url(),
        "installed_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "files": {a.name: {"sha256": a.sha256, "size": a.size, "target": a.target} for a in ARTIFACTS},
    }
    (root / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")


def ensure_data(root: Path | None = None, progress: ProgressFn | None = None) -> DataPaths:
    """Make sure every artifact is installed under `root`; return its paths."""
    p = paths(root)
    todo = missing(p.root)
    if todo:
        p.root.mkdir(parents=True, exist_ok=True)
        for art in todo:
            install_artifact(art, p.root, progress=progress)
        write_manifest(p.root)
        shutil.rmtree(p.root / "downloads", ignore_errors=True)
    elif read_manifest(p.root) is None:
        log.info("using existing data in %s (no manifest)", p.root)
    return p


def stderr_progress(name: str, done: int, total: int) -> None:
    """Default progress reporter: one line per step, never on stdout."""
    pct = 100 * done // total if total else 0
    print(f"revit-api-docs: {name} {done / 1e6:.1f}/{total / 1e6:.1f} MB ({pct}%)",
          file=sys.stderr, flush=True)
