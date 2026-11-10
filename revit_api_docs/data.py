"""Data set location and first-run download.

The API reference (SQLite) and the vector stores (ChromaDB directories) are
not in the wheel; they come from the GitHub Release `v1.1-data` of this
repository. `ensure_data()` downloads whatever is missing into the data
directory, verifies each file against the SHA-256 recorded below, unpacks the
tarballs and writes `manifest.json`.

History: `v1.1-data` keeps the 27,596 API rows of `v1.0-data` unchanged (same
ids), appends the constructor and interface member pages from 27,597 on, and
rebuilds the API vector store so that every record id is its row id (in
`v1.0-data` they were numbered independently; the retriever matches by entry
name either way). Its records embed `name : summary` (`v1.0-data` embedded
mostly the entry name alone). The two SDK files are the `v1.0-data` ones. A
tree whose manifest names another release is downloaded again as a whole.

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

While installing, the directory also holds downloads/ (partial and verified
downloads, removed afterwards) and .install.lock (kept; see install_lock).
"""
from __future__ import annotations

import contextlib
import errno
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
from typing import Callable, Iterable, Iterator

import httpx

if sys.platform == "win32":
    import msvcrt
else:
    import fcntl

log = logging.getLogger("revit_api_docs.data")

ENV_DATA_DIR = "REVIT_API_DOCS_DATA_DIR"
ENV_RELEASE_URL = "REVIT_API_DOCS_RELEASE_URL"

RELEASE_TAG = "v1.1-data"
RELEASE_URL = f"https://github.com/revitbridge/revit-api-docs/releases/download/{RELEASE_TAG}"

ProgressFn = Callable[[str, int, int], None]
# Called while another process holds the install lock: (file it is
# downloading or None, bytes it has so far, size of that file).
WaitFn = Callable[[str | None, int, int], None]

LOCK_FILENAME = ".install.lock"
LOCK_POLL_SECONDS = 0.5


@dataclass(frozen=True)
class Artifact:
    """One release asset: where it lands and how to check it."""

    name: str
    sha256: str
    size: int
    target: str          # path inside the data dir (file, or directory for tarballs)
    extract: bool = False


ARTIFACTS: tuple[Artifact, ...] = (
    Artifact("revit_api.db", "e6e323ad6d00a76f0b068b843658d7295160054a10d04dc060cd6f9be509cf1b",
             34_631_680, "sqlite/revit_api.db"),
    Artifact("revit_sdk.db", "f381f392485cd4910b2b826062414b5331c29f165269412ebb59569dd176e474",
             954_368, "sqlite/revit_sdk.db"),
    Artifact("chromadb_api.tar.gz", "68e986e2e90b1e9167823925a0089697df2ab681eb182e35e51c62b1a03d26a1",
             230_005_566, "chromadb/chromadb_api", extract=True),
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
    """REVIT_API_DOCS_DATA_DIR as an absolute path, else the default location."""
    env = os.environ.get(ENV_DATA_DIR)
    return Path(os.path.abspath(Path(env).expanduser())) if env else default_data_dir()


def pin_data_dir() -> Path:
    """Resolve the data directory once, at start-up, and return it.

    A relative REVIT_API_DOCS_DATA_DIR is taken against the working
    directory, which differs between MCP clients, so the same setting could
    install the data set in several places. Its absolute form is written
    back to the environment (later lookups in this process agree even if the
    working directory changes) and a warning names it.
    """
    env = os.environ.get(ENV_DATA_DIR)
    root = data_dir()
    if env and not Path(env).expanduser().is_absolute():
        os.environ[ENV_DATA_DIR] = str(root)
        log.warning("%s=%r is relative to the working directory, which differs between MCP "
                    "clients; using %s for this process. Set an absolute path to keep one data set.",
                    ENV_DATA_DIR, env, root)
    return root


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


def _installed(root: Path, art: Artifact) -> bool:
    target = root / art.target
    if art.extract:
        return (target / "chroma.sqlite3").is_file()
    return target.is_file()


def unexpected_sizes(root: Path) -> list[tuple[Artifact, int]]:
    """Present files whose size differs from the release asset (hand-copied
    or built locally). They are used as they are, never replaced."""
    out = []
    for art in ARTIFACTS:
        if art.extract:
            continue
        target = root / art.target
        if target.is_file():
            size = target.stat().st_size
            if size != art.size:
                out.append((art, size))
    return out


def read_manifest(root: Path) -> dict | None:
    """The manifest this installer wrote, or None.

    None also for a file that is unreadable, is not JSON, or is not an
    object naming a release: such a file was not written by this installer
    and counts as absent (the tree is used as it is).
    """
    p = root / "manifest.json"
    if not p.is_file():
        return None
    try:
        manifest = json.loads(p.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    if not isinstance(manifest, dict) or not isinstance(manifest.get("release"), str):
        return None
    return manifest


def missing(root: Path | None = None) -> list[Artifact]:
    """Artifacts that must be downloaded: the absent ones.

    An existing file is never replaced, whatever its size (a hand-copied tree
    may hold a different build; see unexpected_sizes). The one exception is a
    manifest written by this installer for another release, which marks the
    whole tree stale.
    """
    root = Path(root) if root is not None else data_dir()
    manifest = read_manifest(root)
    if manifest is not None and manifest.get("release") != RELEASE_TAG:
        return list(ARTIFACTS)
    return [a for a in ARTIFACTS if not _installed(root, a)]


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


class DataInUseError(RuntimeError):
    """A verified download could not replace the installed copy: another
    process has a file of it open (Windows refuses to move such a file, or a
    directory that holds one). The installed copy is left as it was."""

    def __init__(self, target: Path, error: OSError):
        reason = error.strerror or str(error)
        super().__init__(
            f"could not replace {target} with the {RELEASE_TAG} data set ({reason}); another "
            "process has it open: close other revit-api-docs processes and retry (the installed "
            "data set is kept as it is, and the verified download is reused)")


# Waits before retrying a rename Windows refused: a virus scanner or the
# search indexer may hold a freshly written file for a moment.
RENAME_RETRY_DELAYS = (0.1, 0.25, 0.5, 1.0)


def _rename(src: Path, dst: Path) -> None:
    for delay in (*RENAME_RETRY_DELAYS, None):
        try:
            os.rename(src, dst)
            return
        except PermissionError:
            if delay is None:
                raise
            time.sleep(delay)


def _backup_name(target: Path) -> Path:
    return target.with_name(f".{target.name}.old-{os.getpid()}-{time.monotonic_ns()}")


def _remove(path: Path) -> None:
    """Best effort: a leftover is removed by the next install (_remove_leftovers)."""
    if path.is_dir() and not path.is_symlink():
        shutil.rmtree(path, ignore_errors=True)
    else:
        try:
            path.unlink()
        except OSError:
            pass


def _remove_leftovers(root: Path) -> None:
    """Previous copies and unpacked staging directories an interrupted install left next to the targets."""
    for art in ARTIFACTS:
        target = root / art.target
        if target.parent.is_dir():
            for pattern in (f".{target.name}.old-*", f".{target.name}-new-*"):
                for leftover in target.parent.glob(pattern):
                    _remove(leftover)


def fetch_artifact(art: Artifact, root: Path, progress: ProgressFn | None = None) -> Path:
    """Download `art` into root/downloads (resuming a partial file) and check
    its SHA-256; return the verified file. Nothing in place is touched."""
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
    return part


def _unpack_beside(archive: Path, target: Path) -> tuple[Path, Path]:
    """Unpack `archive` (one top-level directory) into a hidden directory next
    to `target`, so that moving it in is a rename on the same volume. Returns
    the unpacked tree and the scratch directory to remove afterwards."""
    target.parent.mkdir(parents=True, exist_ok=True)
    scratch = Path(tempfile.mkdtemp(prefix=f".{target.name}-new-", dir=target.parent))
    try:
        with tarfile.open(archive, "r:gz") as tar:
            try:
                tar.extractall(scratch, filter="data")
            except TypeError:  # Python < 3.11.4 without the filter argument
                tar.extractall(scratch)
        entries = list(scratch.iterdir())
        staged = entries[0] if len(entries) == 1 and entries[0].is_dir() else scratch
    except BaseException:
        shutil.rmtree(scratch, ignore_errors=True)
        raise
    return staged, scratch


def _stage(art: Artifact, part: Path, root: Path) -> tuple[Path, Path | None]:
    """The verified download ready to be moved into place, and a scratch
    directory to remove afterwards: a file as it is, a tarball unpacked next
    to its target (_unpack_beside)."""
    if not art.extract:
        return part, None
    return _unpack_beside(part, root / art.target)


def _swap_in(staged: Path, target: Path) -> Path | None:
    """Move `staged` to `target`; return where the previous copy went (None if there was none).

    Two renames and no deletion, so `target` holds the previous copy or the
    new one, each complete, at every moment. When the previous copy cannot
    be moved aside (another process has a file of it open) it stays where it
    is and DataInUseError is raised; when the new one cannot be moved in,
    the previous one is put back.
    """
    target.parent.mkdir(parents=True, exist_ok=True)
    backup = None
    if target.exists() or target.is_symlink():
        backup = _backup_name(target)
        try:
            _rename(target, backup)
        except PermissionError as e:
            raise DataInUseError(target, e) from e
    try:
        _rename(staged, target)
    except OSError as e:
        if backup is not None:
            _put_back(backup, target)
        if isinstance(e, PermissionError):
            raise DataInUseError(target, e) from e
        raise
    return backup


def _put_back(backup: Path, target: Path) -> None:
    try:
        _rename(backup, target)
    except OSError as e:
        log.error("could not move %s back to %s (%s); move it back by hand", backup, target, e)


def _undo_swap(target: Path, staged: Path, backup: Path | None) -> None:
    """Reverse _swap_in: the new copy back to where it was staged, the previous one back in place."""
    try:
        _rename(target, staged)
    except OSError as e:
        log.error("could not move the new %s out again (%s); the next start installs it again", target, e)
        return
    if backup is not None:
        _put_back(backup, target)


def _extract_tarball(archive: Path, target: Path) -> None:
    """Unpack `archive` (which holds one top-level directory) to `target`,
    replacing what is there (pipeline/build_data_v1_1.py uses it)."""
    staged, scratch = _unpack_beside(archive, target)
    try:
        backup = _swap_in(staged, target)
    finally:
        shutil.rmtree(scratch, ignore_errors=True)
    if backup is not None:
        _remove(backup)


def install_artifact(art: Artifact, root: Path, progress: ProgressFn | None = None) -> None:
    """Download one artifact, verify it and put it in place (see _swap_in)."""
    part = fetch_artifact(art, root, progress)
    staged, scratch = _stage(art, part, root)
    try:
        backup = _swap_in(staged, root / art.target)
    finally:
        if scratch is not None:
            shutil.rmtree(scratch, ignore_errors=True)
    if backup is not None:
        _remove(backup)
    if art.extract:
        part.unlink(missing_ok=True)


def describe_artifact(root: Path, art: Artifact, installed: bool) -> dict:
    """Manifest entry for one artifact, honest about what was checked.

    `installed` means this run downloaded and hash-checked it. Anything else
    was already in `root`: a file is hashed now and counts as verified only
    when it matches the release; an unpacked directory has nothing to hash
    against and is recorded as trusted.
    """
    entry: dict = {"target": art.target, "release_sha256": art.sha256, "release_size": art.size}
    target = root / art.target
    if installed:
        entry.update(sha256=art.sha256, size=art.size, verified=True)
    elif art.extract:
        entry.update(verified=False, trusted=(target / "chroma.sqlite3").is_file())
    elif target.is_file():
        digest = sha256_file(target)
        verified = digest == art.sha256
        entry.update(sha256=digest, size=target.stat().st_size, verified=verified)
        if not verified:
            entry["trusted"] = True
    else:
        entry.update(verified=False, trusted=False)
    return entry


def _install(root: Path, todo: list[Artifact], progress: ProgressFn | None) -> None:
    """Install `todo` into `root` and write the manifest; the caller holds the install lock.

    Every artifact is downloaded and verified, and the tarballs unpacked next
    to their targets, before anything in place changes. Then each target is
    swapped in by renames (_swap_in). When one cannot be (DataInUseError: a
    file of the installed copy is open elsewhere), the ones already swapped
    are swapped back, so the installed tree stays as it was, and the verified
    downloads stay in root/downloads for the next try.
    """
    _remove_leftovers(root)
    parts = [fetch_artifact(art, root, progress) for art in todo]
    staged: list[tuple[Artifact, Path, Path | None]] = []
    swapped: list[tuple[Path, Path, Path | None]] = []   # (target, staged copy, previous copy)
    try:
        for art, part in zip(todo, parts):
            staged.append((art, *_stage(art, part, root)))
        for art, path, _ in staged:
            target = root / art.target
            swapped.append((target, path, _swap_in(path, target)))
    except BaseException:
        for target, path, backup in reversed(swapped):
            _undo_swap(target, path, backup)
        raise
    finally:
        for _, _, scratch in staged:
            if scratch is not None:
                shutil.rmtree(scratch, ignore_errors=True)
    write_manifest(root, installed=[a.name for a in todo])
    files = (read_manifest(root) or {}).get("files") or {}
    trusted = [n for n, e in files.items() if e.get("trusted")]
    if trusted:
        log.warning("manifest written with unverified pre-existing files: %s", ", ".join(trusted))
    for _, _, backup in swapped:
        if backup is not None:
            _remove(backup)
    shutil.rmtree(root / "downloads", ignore_errors=True)


def usable_as_is(root: Path | None = None) -> bool:
    """True when `root` holds what keyword search needs (both SQLite files),
    whatever release they come from: the server can run on them while an
    update waits for files in use to be closed."""
    p = paths(root)
    return p.api_db.is_file() and p.sdk_db.is_file()


def write_manifest(root: Path, installed: Iterable[str] = ()) -> None:
    """Record the tree under `root`; `installed` names the artifacts this run
    downloaded and verified (see describe_artifact for the rest)."""
    root.mkdir(parents=True, exist_ok=True)
    done = set(installed)
    manifest = {
        "release": RELEASE_TAG,
        "source": release_url(),
        "installed_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "files": {a.name: describe_artifact(root, a, a.name in done) for a in ARTIFACTS},
    }
    (root / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")


if sys.platform == "win32":
    def _try_lock(fd: int) -> bool | None:
        """True: locked; False: another handle holds it; None: locking not possible here."""
        os.lseek(fd, 0, os.SEEK_SET)
        try:
            msvcrt.locking(fd, msvcrt.LK_NBLCK, 1)   # first byte; locking past the end is allowed
        except OSError as e:
            return False if e.errno in (errno.EACCES, errno.EDEADLK) else None
        return True

    def _unlock(fd: int) -> None:
        os.lseek(fd, 0, os.SEEK_SET)
        msvcrt.locking(fd, msvcrt.LK_UNLCK, 1)
else:
    def _try_lock(fd: int) -> bool | None:
        """True: locked; False: another handle holds it; None: locking not possible here."""
        try:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as e:
            return False if e.errno in (errno.EAGAIN, errno.EACCES, errno.EWOULDBLOCK) else None
        return True

    def _unlock(fd: int) -> None:
        fcntl.flock(fd, fcntl.LOCK_UN)


def _other_install_progress(root: Path) -> tuple[str | None, int, int]:
    """The download another process is writing: the .part written last, as
    (artifact name, bytes so far, artifact size); (None, 0, 0) when none is."""
    sizes = {a.name: a.size for a in ARTIFACTS}
    latest: tuple[float, str, int] | None = None
    try:
        parts = list((root / "downloads").glob("*.part"))
    except OSError:
        parts = []
    for part in parts:
        name = part.name[:-len(".part")]
        try:
            st = part.stat()
        except OSError:
            continue
        if name in sizes and (latest is None or st.st_mtime > latest[0]):
            latest = (st.st_mtime, name, st.st_size)
    if latest is None:
        return None, 0, 0
    return latest[1], min(latest[2], sizes[latest[1]]), sizes[latest[1]]


@contextlib.contextmanager
def install_lock(root: Path, waiting: WaitFn | None = None,
                 poll: float = LOCK_POLL_SECONDS) -> Iterator[bool]:
    """Hold the install lock of the data directory `root` for the block.

    While another process holds it this waits (logging once, and calling
    `waiting` about every `poll` seconds with that process's download). The
    lock is the operating system's, on root/.install.lock (LockFile on
    Windows, flock elsewhere): it is released when the holding process ends,
    however it ends, so a crashed install never blocks the next one. The
    file stays (removing it would race with a process about to lock it).
    Where the file cannot be opened or locked the block runs unlocked, with
    a warning. Yields True when the lock is held.
    """
    path = Path(root) / LOCK_FILENAME
    try:
        fd = os.open(path, os.O_RDWR | os.O_CREAT, 0o644)
    except OSError as e:
        log.warning("cannot open %s (%s); installing without the lock", path, e)
        yield False
        return
    locked = False
    try:
        announced = False
        while True:
            state = _try_lock(fd)
            if state is None:
                log.warning("cannot lock %s; installing without the lock", path)
                break
            if state:
                locked = True
                break
            if not announced:
                announced = True
                log.warning("another revit-api-docs process is installing the data set in %s; "
                            "waiting for it to finish", root)
            if waiting is not None:
                waiting(*_other_install_progress(Path(root)))
            time.sleep(poll)
        yield locked
    finally:
        if locked:
            try:
                _unlock(fd)
            except OSError:
                pass   # closing the descriptor releases it as well
        os.close(fd)


def ensure_data(root: Path | None = None, progress: ProgressFn | None = None,
                waiting: WaitFn | None = None) -> DataPaths:
    """Make sure every artifact is installed under `root`; return its paths.

    The install runs under the directory's install lock (install_lock). A
    process that finds the data set missing while another installs it waits
    (`waiting` is called meanwhile), then finds it installed and downloads
    nothing; the waiting process never writes to the directory.
    """
    p = paths(root)
    todo = missing(p.root)
    for art, size in unexpected_sizes(p.root):
        log.warning("%s: existing file is %d bytes, the %s release file is %d; using it as is "
                    "(delete it to re-download)", p.root / art.target, size, RELEASE_TAG, art.size)
    if todo:
        p.root.mkdir(parents=True, exist_ok=True)
        with install_lock(p.root, waiting=waiting):
            todo = missing(p.root)   # another process may have installed it while this one waited
            if todo:
                _install(p.root, todo, progress)
            else:
                log.info("data set installed by another process in %s", p.root)
    elif read_manifest(p.root) is None:
        log.info("using existing data in %s (no manifest)", p.root)
    return p


def _is_tty(stream) -> bool:
    try:
        return bool(stream.isatty())
    except (AttributeError, ValueError):  # no stderr, or a closed one
        return False


class StderrProgress:
    """Default progress reporter, never on stdout (ASCII only).

    On a terminal the current file's line is redrawn in place; anywhere else
    (MCP host logs, pipes) one line per 10% keeps the log readable.
    download_file reports every second and twice at the end, so the last
    report of each file is remembered.
    """

    def __init__(self) -> None:
        self._last: dict[str, int] = {}   # name -> last percentage (tty) or 10% step (pipe)

    def __call__(self, name: str, done: int, total: int) -> None:
        pct = min(100 * done // total, 100) if total else 0
        line = f"revit-api-docs: {name} {done / 1e6:.1f}/{total / 1e6:.1f} MB ({pct}%)"
        if _is_tty(sys.stderr):
            if self._last.get(name) == 100:
                return
            self._last[name] = pct
            print(line, end="\n" if pct == 100 else "\r", file=sys.stderr, flush=True)
        else:
            step = pct // 10 * 10
            if self._last.get(name) == step:
                return
            self._last[name] = step
            print(line, file=sys.stderr, flush=True)


stderr_progress: ProgressFn = StderrProgress()
