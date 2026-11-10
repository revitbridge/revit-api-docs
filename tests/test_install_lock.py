"""Two processes installing into one data directory: one downloads, the other
waits and uses its copy; a crashed installer does not block the next start.

Real child processes against a local release server whose first file stops
halfway until the test opens a gate. No network.
"""
from __future__ import annotations

import hashlib
import http.server
import io
import json
import os
import subprocess
import sys
import tarfile
import threading
import time
from pathlib import Path

import pytest

from revit_api_docs import data

GATED = "revit_api.db"
PROXY_VARS = ("HTTPS_PROXY", "HTTP_PROXY", "ALL_PROXY", "NO_PROXY",
              "https_proxy", "http_proxy", "all_proxy", "no_proxy")

CHILD = """\
import json, logging, sys
from pathlib import Path
from revit_api_docs import data

spec = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
data.ARTIFACTS = tuple(data.Artifact(**a) for a in spec["artifacts"])
data.TOTAL_DOWNLOAD_BYTES = sum(a.size for a in data.ARTIFACTS)
logging.basicConfig(stream=sys.stderr, level=logging.INFO, format="%(name)s: %(message)s")
logging.getLogger("httpx").setLevel(logging.WARNING)
data.ensure_data(Path(spec["root"]))
print("installed", flush=True)
"""


def _tar_with_dir(dirname: str, files: dict[str, bytes]) -> bytes:
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:gz") as tar:
        for name, content in files.items():
            info = tarfile.TarInfo(f"{dirname}/{name}")
            info.size = len(content)
            tar.addfile(info, io.BytesIO(content))
    return buf.getvalue()


class _Server(http.server.ThreadingHTTPServer):
    daemon_threads = True

    def handle_error(self, request, client_address):
        pass   # a killed client resets its connection; not a test failure


class _Release:
    """Serves the payloads; GATED stops after half its bytes until `gate` is set."""

    def __init__(self, payloads: dict[str, bytes]):
        self.payloads = payloads
        self.gate = threading.Event()
        self.at_gate = threading.Event()
        self.hits: list[tuple[str, str]] = []   # (file, Range header or "full")
        release = self

        class Handler(http.server.BaseHTTPRequestHandler):
            def do_GET(self):  # noqa: N802
                name = self.path.rsplit("/", 1)[-1]
                rng = self.headers.get("Range")
                release.hits.append((name, rng or "full"))
                body = release.payloads.get(name)
                if body is None:
                    self.send_error(404)
                    return
                start = int(rng.split("=")[1].rstrip("-")) if rng else 0
                rest = body[start:]
                self.send_response(206 if rng else 200)
                self.send_header("Content-Length", str(len(rest)))
                self.end_headers()
                try:
                    if name == GATED and not release.gate.is_set():
                        half = len(rest) // 2
                        self.wfile.write(rest[:half])
                        self.wfile.flush()
                        release.at_gate.set()
                        release.gate.wait(60)
                        rest = rest[half:]
                    self.wfile.write(rest)
                except OSError:
                    pass

            def log_message(self, *a):
                pass

        self.server = _Server(("127.0.0.1", 0), Handler)
        self.url = f"http://127.0.0.1:{self.server.server_address[1]}/dl"
        threading.Thread(target=self.server.serve_forever, daemon=True).start()

    def close(self):
        self.gate.set()
        self.server.shutdown()
        self.server.server_close()


@pytest.fixture
def release(monkeypatch):
    payloads = {
        # Over 1 MB before the gate, so the first process has written part of
        # the file when it stops (httpx hands the body over in 1 MB chunks).
        "revit_api.db": bytes(range(256)) * 12_288,   # 3 MiB
        "revit_sdk.db": b"sdk-db-bytes" * 10,
        "chromadb_api.tar.gz": _tar_with_dir("chromadb_api", {"chroma.sqlite3": b"a", "meta.json": b"{}"}),
        "chromadb_code.tar.gz": _tar_with_dir("chromadb_code", {"chroma.sqlite3": b"c"}),
    }
    rel = _Release(payloads)
    arts = tuple(data.Artifact(a.name, hashlib.sha256(payloads[a.name]).hexdigest(), len(payloads[a.name]),
                               a.target, a.extract) for a in data.ARTIFACTS)
    monkeypatch.setattr(data, "ARTIFACTS", arts)
    monkeypatch.setenv("REVIT_API_DOCS_RELEASE_URL", rel.url)
    yield rel
    rel.close()


def _start(tmp_path: Path, release: _Release, root: Path, label: str) -> tuple[subprocess.Popen, Path]:
    script = tmp_path / "child.py"
    script.write_text(CHILD, encoding="utf-8")
    spec = tmp_path / "spec.json"
    spec.write_text(json.dumps({"root": str(root), "artifacts": [
        {"name": a.name, "sha256": a.sha256, "size": a.size, "target": a.target, "extract": a.extract}
        for a in data.ARTIFACTS]}), encoding="utf-8")
    env = {k: v for k, v in os.environ.items() if k not in PROXY_VARS and k != "REVIT_API_DOCS_DATA_DIR"}
    env.update(NO_PROXY="*", REVIT_API_DOCS_RELEASE_URL=release.url)   # straight to the local server
    log = tmp_path / f"{label}.log"
    with open(log, "wb") as out:
        proc = subprocess.Popen([sys.executable, str(script), str(spec)], env=env,
                                stdout=out, stderr=subprocess.STDOUT)
    return proc, log


def _wait_until(condition, timeout: float = 30.0) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if condition():
            return True
        time.sleep(0.05)
    return False


def _assert_installed(root: Path, payloads: dict[str, bytes]) -> None:
    assert data.missing(root) == []
    assert (root / "sqlite" / "revit_api.db").read_bytes() == payloads["revit_api.db"]
    assert (root / "sqlite" / "revit_sdk.db").read_bytes() == payloads["revit_sdk.db"]
    assert (root / "chromadb" / "chromadb_api" / "chroma.sqlite3").read_bytes() == b"a"
    assert (root / "chromadb" / "chromadb_code" / "chroma.sqlite3").read_bytes() == b"c"
    manifest = data.read_manifest(root)
    assert manifest is not None and manifest["release"] == data.RELEASE_TAG
    assert all(e["verified"] is True for e in manifest["files"].values())
    assert not (root / "downloads").exists()


def test_two_processes_starting_together_download_once(release, tmp_path):
    root = tmp_path / "data"
    a, a_log = _start(tmp_path, release, root, "a")
    try:
        assert release.at_gate.wait(30), a_log.read_text(errors="replace")
        b, b_log = _start(tmp_path, release, root, "b")
        try:
            # B finds the lock held and waits, without a request of its own
            # (the unlocked code sent a Range request for the shared .part).
            _wait_until(lambda: "waiting for it to finish" in b_log.read_text(errors="replace")
                        or len(release.hits) > 1)
            assert release.hits == [(GATED, "full")]
            release.gate.set()
            assert b.wait(60) == 0, b_log.read_text(errors="replace")
        finally:
            b.kill()
        assert a.wait(60) == 0, a_log.read_text(errors="replace")
    finally:
        a.kill()
    assert sorted(release.hits) == sorted((name, "full") for name in release.payloads)   # each file once
    assert "waiting for it to finish" in b_log.read_text(errors="replace")
    assert "waiting" not in a_log.read_text(errors="replace")
    assert "installed by another process" in b_log.read_text(errors="replace")
    _assert_installed(root, release.payloads)


def test_a_crashed_installer_does_not_block_the_next_start(release, tmp_path):
    """The lock is the OS's: it goes with the killed process. The next start
    takes it at once and resumes the partial download."""
    root = tmp_path / "data"
    a, a_log = _start(tmp_path, release, root, "a")
    part = root / "downloads" / f"{GATED}.part"
    try:
        assert release.at_gate.wait(30), a_log.read_text(errors="replace")
        assert _wait_until(lambda: part.is_file() and part.stat().st_size >= 1 << 20)
    finally:
        a.kill()
        a.wait(30)
    kept = part.stat().st_size
    release.gate.set()
    t0 = time.monotonic()
    b, b_log = _start(tmp_path, release, root, "b")
    try:
        assert b.wait(60) == 0, b_log.read_text(errors="replace")
    finally:
        b.kill()
    assert time.monotonic() - t0 < 30
    assert (GATED, f"bytes={kept}-") in release.hits   # resumed, not started over
    _assert_installed(root, release.payloads)


def test_the_install_lock_is_exclusive_and_reports_the_other_download(tmp_path):
    """In one process too: a second holder waits, sees the other's .part, and
    gets the lock when the first lets go."""
    (tmp_path / "downloads").mkdir()
    (tmp_path / "downloads" / "revit_sdk.db.part").write_bytes(b"x" * 1000)
    seen: list[tuple] = []
    order: list[str] = []

    def second():
        with data.install_lock(tmp_path, waiting=lambda *a: seen.append(a), poll=0.02) as held:
            order.append(f"second {held}")

    with data.install_lock(tmp_path) as held:
        order.append(f"first {held}")
        t = threading.Thread(target=second)
        t.start()
        assert _wait_until(lambda: seen, timeout=10)
        assert order == ["first True"]
    t.join(10)
    assert order == ["first True", "second True"]
    size = next(a.size for a in data.ARTIFACTS if a.name == "revit_sdk.db")
    assert seen[0] == ("revit_sdk.db", 1000, size)
    assert (tmp_path / data.LOCK_FILENAME).is_file()   # kept: removing it would race with a waiter


def test_waiting_for_another_process_reports_downloading(monkeypatch, tmp_path):
    """A tool call while another process installs gets the "downloading" status."""
    from revit_api_docs import server

    monkeypatch.setenv("REVIT_API_DOCS_DATA_DIR", str(tmp_path))
    st = server._State()
    monkeypatch.setattr(st, "start_background", lambda: True)
    seen: dict[str, str] = {}

    def ensure_data(progress=None, waiting=None):
        waiting("chromadb_api.tar.gz", 42, 100)
        seen["msg"] = st.message_if_not_ready()
        waiting(None, 0, 0)
        seen["msg2"] = st.message_if_not_ready()
        raise RuntimeError("stop here")

    monkeypatch.setattr(server.data, "ensure_data", ensure_data)
    with pytest.raises(RuntimeError):
        st.prepare()
    assert seen["msg"].startswith("revit-api-docs is downloading its data set on first run")
    assert "currently chromadb_api.tar.gz 42%, in another revit-api-docs process" in seen["msg"]
    assert "currently in another revit-api-docs process" in seen["msg2"]
