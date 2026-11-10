"""Data directory resolution and the download/verify/extract path, served from a local HTTP server."""
from __future__ import annotations

import hashlib
import http.server
import io
import json
import logging
import os
import sys
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


def test_a_relative_data_dir_is_made_absolute(monkeypatch, tmp_path):
    """A relative REVIT_API_DOCS_DATA_DIR would name a different place for every
    working directory (each MCP client starts the server in its own)."""
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("REVIT_API_DOCS_DATA_DIR", os.path.join("rel", "data"))
    root = data.data_dir()
    assert root.is_absolute()
    assert root == Path(os.getcwd()) / "rel" / "data"
    assert data.paths().api_db == root / "sqlite" / "revit_api.db"


def test_a_relative_data_dir_is_pinned_once_at_start_up(monkeypatch, tmp_path, caplog):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("REVIT_API_DOCS_DATA_DIR", os.path.join("rel", "data"))
    caplog.set_level(logging.WARNING, logger="revit_api_docs.data")
    root = data.pin_data_dir()
    assert root == Path(os.getcwd()) / "rel" / "data"
    assert os.environ["REVIT_API_DOCS_DATA_DIR"] == str(root)
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    monkeypatch.chdir(elsewhere)
    assert data.data_dir() == root   # a later change of directory does not move the data set
    assert "is relative" in caplog.text and str(root) in caplog.text

    caplog.clear()   # an absolute value (or none) is left alone and logs nothing
    assert data.pin_data_dir() == root and caplog.text == ""
    monkeypatch.delenv("REVIT_API_DOCS_DATA_DIR")
    assert data.pin_data_dir() == data.default_data_dir() and caplog.text == ""


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
    assert data.RELEASE_URL.endswith("/releases/download/v1.1-data")
    assert data.TOTAL_DOWNLOAD_BYTES > 250_000_000   # v1.1-data: 267 MB


def test_missing_trusts_a_manifestless_tree_and_rejects_other_releases(tmp_path):
    root = tmp_path
    for art in data.ARTIFACTS:
        target = root / art.target
        if art.extract:
            target.mkdir(parents=True)
            (target / "chroma.sqlite3").write_bytes(b"x")
        else:
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(b"hand-copied")  # any size: presence is what counts
    assert data.missing(root) == []
    assert data.is_ready(root)
    assert [a.name for a, _ in data.unexpected_sizes(root)] == ["revit_api.db", "revit_sdk.db"]
    (root / "manifest.json").write_text(json.dumps({"release": "v0.9-data"}), encoding="utf-8")
    assert [a.name for a in data.missing(root)] == [a.name for a in data.ARTIFACTS]


def _complete_tree(root: Path) -> None:
    """Every artifact present (placeholder contents), no manifest."""
    for art in data.ARTIFACTS:
        target = root / art.target
        if art.extract:
            target.mkdir(parents=True, exist_ok=True)
            (target / "chroma.sqlite3").write_bytes(b"x")
        else:
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(b"hand-copied")


@pytest.mark.parametrize("text", ["[]", "null", "42", '"v1.1-data"', "{}", '{"release": 5}',
                                  '{"files": {}}', "not json", chr(0xFEFF) + "{"])
def test_a_manifest_that_is_not_an_object_with_a_release_counts_as_absent(tmp_path, text):
    """A manifest.json this installer did not write (a list, null, a number, an
    object without a release name) is treated as absent: the tree is used as
    it is instead of failing on every start or being downloaded again."""
    _complete_tree(tmp_path)
    (tmp_path / "manifest.json").write_text(text, encoding="utf-8")
    assert data.missing(tmp_path) == []
    assert data.read_manifest(tmp_path) is None
    assert data.ensure_data(tmp_path).root == tmp_path   # no download, no exception


# -- progress reporting -------------------------------------------------------

class _FakeStderr(io.StringIO):
    def __init__(self, tty: bool):
        super().__init__()
        self._tty = tty

    def isatty(self) -> bool:
        return self._tty


def test_progress_redraws_one_line_on_a_terminal(monkeypatch):
    err = _FakeStderr(tty=True)
    monkeypatch.setattr(sys, "stderr", err)
    report = data.StderrProgress()
    for done in (1_000_000, 2_500_000, 4_000_000, 4_000_000):  # the final report comes twice
        report("revit_api.db", done, 4_000_000)
    report("revit_sdk.db", 500_000, 500_000)
    out = err.getvalue()
    assert out == ("revit-api-docs: revit_api.db 1.0/4.0 MB (25%)\r"
                   "revit-api-docs: revit_api.db 2.5/4.0 MB (62%)\r"
                   "revit-api-docs: revit_api.db 4.0/4.0 MB (100%)\n"
                   "revit-api-docs: revit_sdk.db 0.5/0.5 MB (100%)\n")
    assert out.isascii()


def test_progress_prints_one_line_per_ten_percent_when_piped(monkeypatch):
    err = _FakeStderr(tty=False)
    monkeypatch.setattr(sys, "stderr", err)
    report = data.StderrProgress()
    total = 1_000_000
    for done in range(0, total + 1, 7_000):  # one report per "second", 143 of them
        report("chromadb_api.tar.gz", done, total)
        report("other.db", done, total)  # files are tracked independently
    report("chromadb_api.tar.gz", total, total)
    report("chromadb_api.tar.gz", total, total)
    out = err.getvalue()
    assert "\r" not in out and out.isascii()
    lines = [ln for ln in out.splitlines() if "chromadb_api.tar.gz" in ln]
    steps = [int(ln.rsplit("(", 1)[1].rstrip("%)")) // 10 for ln in lines]
    assert steps == list(range(11))  # 0%, 10%, ..., 100%, once each
    assert lines[-1].endswith("1.0/1.0 MB (100%)")
    assert sum("other.db" in ln for ln in out.splitlines()) == 10  # never reached 100%


def test_default_reporter_is_the_stateful_one():
    assert isinstance(data.stderr_progress, data.StderrProgress)


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
    assert all(e["verified"] is True and "trusted" not in e for e in manifest["files"].values())
    assert manifest["files"]["revit_api.db"]["sha256"] == data.ARTIFACTS[0].sha256
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


def test_existing_file_of_another_size_survives_ensure_data(fake_release, tmp_path, caplog):
    """A hand-copied DB (no manifest) is never replaced, whatever its size."""
    payloads, hits = fake_release
    mine = tmp_path / "sqlite" / "revit_api.db"
    mine.parent.mkdir(parents=True)
    mine.write_bytes(b"my own build of the api db")
    assert mine.stat().st_size != data.ARTIFACTS[0].size

    caplog.set_level(logging.WARNING, logger="revit_api_docs.data")
    p = data.ensure_data(tmp_path)

    assert mine.read_bytes() == b"my own build of the api db"
    assert len(hits) == 3  # only the three absent artifacts were requested
    assert p.sdk_db.read_bytes() == payloads["revit_sdk.db"]
    assert (p.chromadb_api / "chroma.sqlite3").is_file()
    assert "using it as is" in caplog.text and "revit_api.db" in caplog.text


# -- upgrading over a tree another process has open ------------------------------

@pytest.fixture
def open_elsewhere(monkeypatch):
    """Paths 'open in another process', with Windows semantics on every OS: such
    a file cannot be deleted, renamed or replaced, and a directory that holds
    one cannot be renamed or removed. Add paths to the returned set; clear it
    to 'close' them."""
    held: set[Path] = set()

    def blocked(path) -> bool:
        if not held or isinstance(path, int):
            return False
        p = Path(os.fspath(path)).resolve()
        return any(p == h or p in h.parents for h in held)

    def guard(fn, nargs):
        def inner(*args, **kw):
            if any(blocked(a) for a in args[:nargs]):
                raise PermissionError(13, "The process cannot access the file because it is being "
                                          "used by another process", str(args[0]))
            return fn(*args, **kw)
        return inner

    for name, nargs in (("unlink", 1), ("remove", 1), ("rmdir", 1), ("rename", 2), ("replace", 2)):
        monkeypatch.setattr(os, name, guard(getattr(os, name), nargs))
    monkeypatch.setattr(data, "RENAME_RETRY_DELAYS", (), raising=False)
    return held


def _old_release_tree(root: Path) -> dict[Path, bytes]:
    """An installed tree of another release, with a vector segment directory
    that sorts before chroma.sqlite3 (rmtree reaches it first)."""
    files = {
        root / "chromadb" / "chromadb_api" / "0a1b2c3d" / "data_level0.bin": b"old hnsw",
        root / "chromadb" / "chromadb_api" / "chroma.sqlite3": b"old api store",
        root / "chromadb" / "chromadb_code" / "chroma.sqlite3": b"old code store",
        root / "sqlite" / "revit_api.db": b"old api db",
        root / "sqlite" / "revit_sdk.db": b"old sdk db",
    }
    for path, content in files.items():
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(content)
    (root / "manifest.json").write_text(json.dumps({"release": "v1.0-data", "files": {}}), encoding="utf-8")
    return files


def _tree(root: Path) -> dict[str, bytes]:
    return {p.relative_to(root).as_posix(): p.read_bytes() for p in sorted(root.rglob("*")) if p.is_file()}


def test_an_upgrade_over_files_in_use_keeps_the_old_tree(fake_release, tmp_path, open_elsewhere):
    """The vector store is open in an older server: the new data set is
    downloaded and verified, nothing of the installed one is removed or
    replaced, and the next try (files closed) installs it without downloading."""
    payloads, hits = fake_release
    old = _old_release_tree(tmp_path)
    before = {k: v for k, v in _tree(tmp_path).items()}
    open_elsewhere.add((tmp_path / "chromadb" / "chromadb_api" / "chroma.sqlite3").resolve())

    with pytest.raises(Exception) as failed:
        data.ensure_data(tmp_path)

    after = _tree(tmp_path)
    for path, content in old.items():   # complete and unchanged, also the SQLite files swapped before
        assert after.get(path.relative_to(tmp_path).as_posix()) == content, path
    assert json.loads((tmp_path / "manifest.json").read_text(encoding="utf-8"))["release"] == "v1.0-data"
    assert isinstance(failed.value, data.DataInUseError)
    assert "close other revit-api-docs processes and retry" in str(failed.value)
    kept = sorted(p.name for p in (tmp_path / "downloads").iterdir() if p.name.endswith(".part"))
    assert kept == sorted(f"{name}.part" for name in payloads)   # verified downloads stay for the retry
    leftovers = [p for p in tmp_path.rglob(".*") if p.name != data.LOCK_FILENAME]
    assert leftovers == []   # no staging or backup copies left behind
    assert {k for k in after if not k.startswith("downloads/")} == set(before) | {data.LOCK_FILENAME}

    open_elsewhere.clear()   # the other server was closed
    requests = len(hits)
    p = data.ensure_data(tmp_path)
    assert len(hits) == requests   # nothing downloaded again
    assert p.api_db.read_bytes() == payloads["revit_api.db"]
    assert (p.chromadb_api / "chroma.sqlite3").read_bytes() == b"a"
    assert not (p.chromadb_api / "0a1b2c3d").exists()   # the old store was replaced as a whole
    assert data.read_manifest(tmp_path)["release"] == data.RELEASE_TAG and data.missing(tmp_path) == []
    assert not (tmp_path / "downloads").exists()
    assert [p for p in tmp_path.rglob(".*") if p.name != data.LOCK_FILENAME] == []


@pytest.mark.skipif(sys.platform != "win32", reason="Windows refuses to delete or rename open files")
def test_an_upgrade_over_a_really_open_store_keeps_it(fake_release, tmp_path):
    payloads, _ = fake_release
    old = _old_release_tree(tmp_path)
    with open(tmp_path / "chromadb" / "chromadb_api" / "chroma.sqlite3", "rb"):   # as an older server holds it
        with pytest.raises(Exception) as failed:
            data.ensure_data(tmp_path)
        after = _tree(tmp_path)
        for path, content in old.items():
            assert after.get(path.relative_to(tmp_path).as_posix()) == content, path
        assert isinstance(failed.value, data.DataInUseError)
    p = data.ensure_data(tmp_path)
    assert p.api_db.read_bytes() == payloads["revit_api.db"] and data.missing(tmp_path) == []


def test_the_server_keeps_working_on_the_installed_tree_when_files_are_in_use(tiny_data_dir, monkeypatch,
                                                                              no_key_env, caplog):
    from revit_api_docs import server

    monkeypatch.setenv("REVIT_API_DOCS_DATA_DIR", str(tiny_data_dir))
    for name in ("REVIT_API_DOCS_EMBEDDING_API_KEY", "OPENROUTER_API_KEY"):
        monkeypatch.delenv(name, raising=False)

    def in_use(progress=None, waiting=None):
        raise data.DataInUseError(tiny_data_dir / "chromadb" / "chromadb_api",
                                  PermissionError(13, "in use"))

    monkeypatch.setattr(data, "ensure_data", in_use)
    caplog.set_level(logging.WARNING, logger="revit_api_docs.server")
    st = server._State()
    retriever = st.prepare()
    assert st.status == "ready" and retriever is st.retriever
    assert server.search_api(retriever, "Wall.Create", 3).startswith("Rerank: none; data set: Revit 2026")
    assert "close other revit-api-docs processes and retry" in caplog.text
    assert str(tiny_data_dir) in caplog.text

    (tiny_data_dir / "sqlite" / "revit_api.db").unlink()   # nothing to fall back on: the error stands
    st = server._State()
    with pytest.raises(data.DataInUseError):
        st.prepare()
    assert st.status == "failed" and "close other revit-api-docs processes" in st.error


def test_download_reports_files_in_use_without_a_traceback(monkeypatch, tmp_path, capsys):
    from revit_api_docs import server

    monkeypatch.setenv("REVIT_API_DOCS_DATA_DIR", str(tmp_path))
    monkeypatch.setattr(server, "_configure_logging", lambda verbose: None)

    def in_use(progress=None, waiting=None):
        raise data.DataInUseError(tmp_path / "sqlite" / "revit_api.db", PermissionError(13, "in use"))

    monkeypatch.setattr(data, "ensure_data", in_use)
    assert server.main(["download"]) == 1
    err = capsys.readouterr().err
    assert err.startswith("revit-api-docs: could not replace ") and "Traceback" not in err
    assert "close other revit-api-docs processes and retry" in err


# -- names pipeline/ uses ------------------------------------------------------

def test_extract_tarball_unpacks_and_replaces_the_target(tmp_path):
    """pipeline/build_data_v1_1.py unpacks the code store with data._extract_tarball:
    the target becomes the archive's top-level directory, replacing what was there."""
    archive = tmp_path / "chromadb_code.tar.gz"
    archive.write_bytes(_tar_with_dir("chromadb_code", {"chroma.sqlite3": b"c", "seg/link_lists.bin": b"l"}))
    target = tmp_path / "out" / "chromadb" / "chromadb_code"
    data._extract_tarball(archive, target)   # (a) no target, no parent yet
    assert _tree(target) == {"chroma.sqlite3": b"c", "seg/link_lists.bin": b"l"}

    (target / "stale.bin").write_bytes(b"from an earlier build")
    archive.write_bytes(_tar_with_dir("chromadb_code", {"chroma.sqlite3": b"new"}))
    data._extract_tarball(archive, target)   # (b) an existing target is replaced as a whole
    assert _tree(target) == {"chroma.sqlite3": b"new"}
    assert [p.name for p in target.parent.iterdir()] == ["chromadb_code"]   # no staging or previous copy left


def test_every_package_name_the_pipeline_uses_exists():
    """The tests never run pipeline/; this catches a package name it imports,
    or reads from an imported module (data._extract_tarball), being removed."""
    import ast
    import importlib

    root = Path(__file__).resolve().parents[1]
    absent: list[str] = []
    checked = 0
    for path in sorted((root / "pipeline").rglob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        parents = {child: node for node in ast.walk(tree) for child in ast.iter_child_nodes(node)}
        for node in ast.walk(tree):
            if not (isinstance(node, ast.ImportFrom) and (node.module or "").split(".")[0] == "revit_api_docs"):
                continue
            scope = node   # the function the import is in, or the module
            while scope in parents and not isinstance(scope, (ast.FunctionDef, ast.AsyncFunctionDef)):
                scope = parents[scope]
            module = importlib.import_module(node.module)
            for alias in node.names:
                where = f"{path.relative_to(root).as_posix()}:{node.lineno}"
                if hasattr(module, alias.name):
                    value = getattr(module, alias.name)
                else:
                    try:
                        value = importlib.import_module(f"{node.module}.{alias.name}")
                    except ImportError:
                        absent.append(f"{where} {node.module}.{alias.name}")
                        continue
                checked += 1
                if not isinstance(value, type(ast)):   # a function or class: nothing more to check
                    continue
                bound = alias.asname or alias.name
                for use in ast.walk(scope):
                    if (isinstance(use, ast.Attribute) and isinstance(use.value, ast.Name)
                            and use.value.id == bound):
                        checked += 1
                        if not hasattr(value, use.attr):
                            absent.append(f"{path.relative_to(root).as_posix()}:{use.lineno} "
                                          f"{bound}.{use.attr}")
    assert checked >= 15   # the check saw the pipeline's imports
    assert absent == []


def test_manifest_never_claims_release_hashes_for_pre_existing_files(fake_release, tmp_path, caplog):
    payloads, _ = fake_release
    # a foreign DB, a DB that happens to be the release bytes, and a pre-existing index directory
    (tmp_path / "sqlite").mkdir(parents=True)
    (tmp_path / "sqlite" / "revit_api.db").write_bytes(b"my own build of the api db")
    (tmp_path / "sqlite" / "revit_sdk.db").write_bytes(payloads["revit_sdk.db"])
    (tmp_path / "chromadb" / "chromadb_code").mkdir(parents=True)
    (tmp_path / "chromadb" / "chromadb_code" / "chroma.sqlite3").write_bytes(b"old index")

    caplog.set_level(logging.WARNING, logger="revit_api_docs.data")
    p = data.ensure_data(tmp_path)
    files = json.loads(p.manifest.read_text(encoding="utf-8"))["files"]

    foreign = files["revit_api.db"]
    assert foreign["verified"] is False and foreign["trusted"] is True
    assert foreign["sha256"] == hashlib.sha256(b"my own build of the api db").hexdigest()
    assert foreign["sha256"] != foreign["release_sha256"]
    assert foreign["size"] == len(b"my own build of the api db")

    same = files["revit_sdk.db"]  # hashed now, matches the release
    assert same["verified"] is True and "trusted" not in same

    index = files["chromadb_code.tar.gz"]  # nothing to hash a directory against
    assert index["verified"] is False and index["trusted"] is True and "sha256" not in index

    downloaded = files["chromadb_api.tar.gz"]
    assert downloaded["verified"] is True and downloaded["sha256"] == downloaded["release_sha256"]
    assert "unverified pre-existing files: revit_api.db, chromadb_code.tar.gz" in caplog.text
