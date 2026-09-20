"""Data directory resolution and the download/verify/extract path, served from a local HTTP server."""
from __future__ import annotations

import hashlib
import http.server
import io
import json
import tarfile
import threading
from pathlib import Path

import pytest

from revit_api_docs import data


def test_data_dir_env_override(monkeypatch, tmp_path):
    monkeypatch.setenv("REVIT_API_DOCS_DATA_DIR", str(tmp_path / "custom"))
    assert data.data_dir() == tmp_path / "custom"
    p = data.paths()
    assert p.api_db == tmp_path / "custom" / "sqlite" / "revit_api.db"
    assert p.chromadb_api == tmp_path / "custom" / "chromadb" / "chromadb_api"


def test_default_data_dir_per_platform(monkeypatch):
    monkeypatch.delenv("REVIT_API_DOCS_DATA_DIR", raising=False)
    monkeypatch.setattr(data.sys, "platform", "win32")
    monkeypatch.setenv("LOCALAPPDATA", "C:/Users/x/AppData/Local")
    assert data.default_data_dir() == Path("C:/Users/x/AppData/Local") / "revit-api-docs"
    monkeypatch.setattr(data.sys, "platform", "linux")
    monkeypatch.setenv("XDG_DATA_HOME", "/xdg")
    assert data.default_data_dir() == Path("/xdg") / "revit-api-docs"


def test_artifact_table_matches_release_layout():
    names = [a.name for a in data.ARTIFACTS]
    assert names == ["revit_api.db", "revit_sdk.db", "chromadb_api.tar.gz", "chromadb_code.tar.gz"]
    assert all(len(a.sha256) == 64 for a in data.ARTIFACTS)
    assert data.RELEASE_URL.endswith("/releases/download/v1.0-data")
    assert data.TOTAL_DOWNLOAD_BYTES > 300_000_000


def test_missing_trusts_a_manifestless_tree_and_rejects_other_releases(tmp_path):
    root = tmp_path
    for art in data.ARTIFACTS:
        target = root / art.target
        if art.extract:
            target.mkdir(parents=True)
            (target / "chroma.sqlite3").write_bytes(b"x")
        else:
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(b"\0" * art.size)  # size is what _installed checks
    assert data.missing(root) == []
    assert data.is_ready(root)
    (root / "manifest.json").write_text(json.dumps({"release": "v0.9-data"}), encoding="utf-8")
    assert [a.name for a in data.missing(root)] == [a.name for a in data.ARTIFACTS]


# ── end-to-end against a local HTTP server ───────────────────────────────────

def _tar_with_dir(dirname: str, files: dict[str, bytes]) -> bytes:
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:gz") as tar:
        for name, content in files.items():
            info = tarfile.TarInfo(f"{dirname}/{name}")
            info.size = len(content)
            tar.addfile(info, io.BytesIO(content))
    return buf.getvalue()


@pytest.fixture
def fake_release(monkeypatch):
    """Serve four small assets and point ARTIFACTS at them."""
    payloads = {
        "revit_api.db": b"api-db-bytes" * 100,
        "revit_sdk.db": b"sdk-db-bytes" * 10,
        "chromadb_api.tar.gz": _tar_with_dir("chromadb_api", {"chroma.sqlite3": b"a", "meta.json": b"{}"}),
        "chromadb_code.tar.gz": _tar_with_dir("chromadb_code", {"chroma.sqlite3": b"c"}),
    }
    hits: list[str] = []

    class Handler(http.server.BaseHTTPRequestHandler):
        def do_GET(self):  # noqa: N802
            name = self.path.rsplit("/", 1)[-1]
            hits.append(self.headers.get("Range") or "full")
            body = payloads.get(name)
            if body is None:
                self.send_error(404)
                return
            rng = self.headers.get("Range")
            if rng:
                start = int(rng.split("=")[1].rstrip("-"))
                self.send_response(206)
                body = body[start:]
            else:
                self.send_response(200)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *a):  # keep pytest output clean
            pass

    server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    url = f"http://127.0.0.1:{server.server_address[1]}/dl"
    monkeypatch.setenv("REVIT_API_DOCS_RELEASE_URL", url)
    arts = tuple(
        data.Artifact(a.name, hashlib.sha256(payloads[a.name]).hexdigest(), len(payloads[a.name]), a.target, a.extract)
        for a in data.ARTIFACTS
    )
    monkeypatch.setattr(data, "ARTIFACTS", arts)
    yield payloads, hits
    server.shutdown()


def test_ensure_data_downloads_verifies_and_extracts(fake_release, tmp_path):
    payloads, hits = fake_release
    seen: list[tuple[str, int, int]] = []
    p = data.ensure_data(tmp_path, progress=lambda n, d, t: seen.append((n, d, t)))

    assert p.api_db.read_bytes() == payloads["revit_api.db"]
    assert p.sdk_db.read_bytes() == payloads["revit_sdk.db"]
    assert (p.chromadb_api / "chroma.sqlite3").read_bytes() == b"a"
    assert (p.chromadb_api / "meta.json").is_file()
    assert (p.chromadb_code / "chroma.sqlite3").read_bytes() == b"c"
    manifest = json.loads(p.manifest.read_text(encoding="utf-8"))
    assert manifest["release"] == data.RELEASE_TAG
    assert not (tmp_path / "downloads").exists()  # tarballs and temp files are gone
    assert data.missing(tmp_path) == []
    assert {n for n, _, _ in seen} == set(payloads)
    assert all(d == t for _, d, t in seen if d == t)  # final report reaches 100%

    # second call: nothing to do, no HTTP traffic
    before = len(hits)
    data.ensure_data(tmp_path)
    assert len(hits) == before


def test_partial_download_is_resumed(fake_release, tmp_path):
    payloads, hits = fake_release
    part = tmp_path / "downloads" / "revit_api.db.part"
    part.parent.mkdir(parents=True)
    part.write_bytes(payloads["revit_api.db"][:100])
    data.install_artifact(data.ARTIFACTS[0], tmp_path)
    assert (tmp_path / "sqlite" / "revit_api.db").read_bytes() == payloads["revit_api.db"]
    assert hits == ["bytes=100-"]


def test_corrupt_download_is_rejected_and_removed(fake_release, tmp_path, monkeypatch):
    bad = data.Artifact("revit_sdk.db", "0" * 64, data.ARTIFACTS[1].size, "sqlite/revit_sdk.db")
    with pytest.raises(RuntimeError, match="SHA-256 mismatch"):
        data.install_artifact(bad, tmp_path)
    assert not (tmp_path / "downloads" / "revit_sdk.db.part").exists()
    assert not (tmp_path / "sqlite" / "revit_sdk.db").exists()
