"""Build the v1.1-data release of the API data set by appending to v1.0.

The v1.0-data SQLite (27,596 rows, every column) is kept as it is; CHM pages
that are not in it (constructor pages and members whose C# signature has no
`public`, mostly interface members) are appended with ids from 27,597 on.
The API vector store is then rebuilt from the new SQLite so that every record
id is its row id. The SDK files (revit_sdk.db, chromadb_code.tar.gz) are the
v1.0 ones, unchanged.

    uv sync --extra pipeline
    uv run python -m pipeline.build_data_v1_1 --html-dir <CHM html dir> \
        --base-db <v1.0 data dir>/sqlite/revit_api.db [--out out/v1.1-data]

Steps (each one's output is kept under <out>/work/ so that --resume picks up
where a run stopped):
  1. parse   every page with keep_constructors=True
  2. classify against the base: (name, full_id) in the base -> skipped; the
             name in the base with another full_id, or the members table of
             a base row that matches no page (a name changed by the v1.0
             rewrite) -> reported, not appended (more than --max-suspects
             such pages stops the build); else a new row, sorted by
             (name, full_id)
  3. quality the two-stage quality agent on the new rows only (--no-quality
             to skip); a row whose audit failed keeps its parse and gets no
             quality_score. Stops early on --quality-budget, --stage1-cap or
             --max-consecutive-failures; the remaining rows are not audited
  4. db      copy of the base + the new rows (columns of save_to_sqlite),
             VACUUM; asserts the base rows are unchanged
  5. embed   every row into a fresh chromadb_api (record id = row id); the
             embedded text is --document-template (default NS, name : summary,
             as in the v1.1-data release)
  6. package release/ holds the four assets: revit_api.db,
             chromadb_api.tar.gz (one top-level dir chromadb_api/),
             revit_sdk.db (copied from next to --base-db) and
             chromadb_code.tar.gz (the v1.0-data asset via `gh release
             download`); both v1.0 files must match their v1.0 SHA-256.
             SHA256SUMS.txt and manifest.json go to <out>
  7. check   consistency probe of both vector stores (no API call) and 200
             random API records looked up by id
Writes <out>/build_report.json (counts, requests, cost, timings).

Keys come from the environment only: OPENROUTER_API_KEY for the quality agent
and REVIT_API_DOCS_EMBEDDING_API_KEY (or OPENROUTER_API_KEY) for embeddings.
They are never printed or written.

<out> is laid out as a data directory (sqlite/, chromadb/, manifest.json), so
REVIT_API_DOCS_DATA_DIR can point at it; <out>/release/ can be served as a
mirror (REVIT_API_DOCS_RELEASE_URL).
"""
from __future__ import annotations

import argparse
import datetime as _dt
import hashlib
import json
import os
import random
import re
import shutil
import sqlite3
import subprocess
import sys
import tarfile
import threading
import time
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]   # run as `python -m pipeline.build_data_v1_1`

RELEASE = "v1.1-data"
BASE_RELEASE = "v1.0-data"
# v1.0-data assets this build starts from or reuses (data.py of revit-api-docs 0.2.0).
V10_SHA = {
    "revit_api.db": "38f78297b11b5fb4aa8a0c609e2a0535b5d2f6770ea895c8468742e6d4a7bf72",
    "revit_sdk.db": "f381f392485cd4910b2b826062414b5331c29f165269412ebb59569dd176e474",
    "chromadb_code.tar.gz": "3ee38a4e44e4b41154faeafd431c830ca54275d2738897971e5780cb493118ad",
}
GH_REPO = "revitbridge/revit-api-docs"

COLUMNS = ("name", "full_id", "namespace", "content_type", "keywords", "info", "summary",
           "remark", "parameters", "exceptions", "return_value", "syntax", "members",
           "quality_score", "quality_issues", "rewritten")
# Fields a stage-2 rewrite may change; everything else stays as parsed.
REWRITABLE = ("summary", "info", "parameters", "syntax", "remark", "exceptions", "return_value")

# USD per million tokens, used only when a response carries no usage.cost.
PRICE_PER_M = {
    "google/gemini-3-flash-preview": (0.50, 3.00),
    "anthropic/claude-sonnet-4.6": (3.00, 15.00),
    "openai/text-embedding-3-large": (0.13, 0.0),
}


# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------

def log(msg: str) -> None:
    print(f"[build {time.strftime('%H:%M:%S')}] {msg}", flush=True)


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def ro_uri(path: Path) -> str:
    return "file:" + path.resolve().as_posix() + "?mode=ro"


def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")
    os.replace(tmp, path)


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def secret_values() -> list[str]:
    names = ("OPENROUTER_API_KEY", "REVIT_API_DOCS_EMBEDDING_API_KEY")
    return [v for v in (os.environ.get(n, "") for n in names) if len(v) >= 8]


def assert_no_secret(text: str, where: str) -> None:
    for v in secret_values():
        if v in text:
            raise SystemExit(f"refusing to write {where}: it contains an API key")


# --------------------------------------------------------------------------
# 1. parse
# --------------------------------------------------------------------------

def stage_parse(html_dir: Path, work: Path, resume: bool) -> tuple[list[dict], dict]:
    from pipeline.api_parser import parse_chm

    out_file, stats_file = work / "parsed.jsonl", work / "parse_stats.json"
    if resume and out_file.is_file() and stats_file.is_file():
        recs = [json.loads(line) for line in out_file.read_text(encoding="utf-8").splitlines() if line]
        log(f"parse: reusing {len(recs)} parsed pages from {out_file}")
        return recs, read_json(stats_file)

    files = sorted(p for p in html_dir.iterdir() if p.suffix.lower() in (".htm", ".html"))
    log(f"parse: {len(files)} pages in {html_dir}")
    recs: list[dict] = []
    skipped: list[dict] = []
    ctor = no_public = 0
    t0 = time.perf_counter()
    for i, f in enumerate(files):
        if i and i % 2000 == 0:
            log(f"parse: {i}/{len(files)} valid={len(recs)} skipped={len(skipped)}")
        rec = parse_chm.parse_single_html(str(f), keep_constructors=True)
        if rec is None:
            title, fid = parse_chm._peek_title_and_fullid(f)
            skipped.append({"file": f.name, "title": title, "full_id": fid})
            continue
        if parse_chm._is_constructor_page(rec["name"], rec["full_id"]):
            rec["_kind"] = "constructor"
            ctor += 1
        elif "public" not in (rec.get("syntax") or "") and not rec.get("members"):
            # The old parser required `public` in the signature and skipped a
            # page with neither such a signature nor a members table.
            rec["_kind"] = "no_public_signature"
            no_public += 1
        else:
            rec["_kind"] = "regular"
        recs.append(rec)
    stats = {
        "pages": len(files),
        "valid": len(recs),
        "constructor_pages": ctor,
        "no_public_signature_pages": no_public,
        "regular_pages": len(recs) - ctor - no_public,
        "skipped_noise": len(skipped),
        "skipped": skipped,
        "seconds": round(time.perf_counter() - t0, 1),
    }
    work.mkdir(parents=True, exist_ok=True)
    with open(out_file, "w", encoding="utf-8") as fh:
        for rec in recs:
            fh.write(json.dumps(rec, ensure_ascii=True) + "\n")
    write_json(stats_file, stats)
    log(f"parse: valid={len(recs)} constructors={ctor} no-public={no_public} "
        f"skipped={len(skipped)} in {stats['seconds']} s")
    return recs, stats


# --------------------------------------------------------------------------
# 2. classify
# --------------------------------------------------------------------------

def sort_key(rec: dict) -> tuple:
    return (rec["name"], rec.get("full_id") or "", Path(rec.get("_source_file", "")).name)


def stage_classify(recs: list[dict], base_db: Path) -> tuple[list[dict], dict]:
    """Split the parsed pages into: in the base, not appended (reported), new.

    Not appended: a page whose name is in the base with another full_id
    (suspected id change), and a page whose name is not in the base but whose
    members table (or signature) equals that of a base row that matches no
    page by (name, full_id): the same page under a name the v1.0 stage-2
    rewrite changed (those base rows have rewritten = 1).
    """
    conn = sqlite3.connect(ro_uri(base_db), uri=True)
    try:
        rows = conn.execute("SELECT id, name, full_id, members, syntax, rewritten FROM revit_api").fetchall()
    finally:
        conn.close()
    pairs = {(r[1], r[2]) for r in rows}
    names: dict[str, list[str]] = {}
    for r in rows:
        names.setdefault(r[1], []).append(r[2])
    page_pairs = {(rec["name"], rec["full_id"]) for rec in recs}
    unmatched = [r for r in rows if (r[1], r[2]) not in page_pairs]
    by_members = {r[3]: r for r in unmatched if r[3]}
    by_syntax = {r[4]: r for r in unmatched if r[4] and not r[3]}
    existing, suspects, renamed, new = 0, [], [], []
    for rec in recs:
        key = (rec["name"], rec["full_id"])
        if key in pairs:
            existing += 1
        elif rec["name"] in names:
            suspects.append({"name": rec["name"], "full_id": rec["full_id"],
                             "base_full_ids": names[rec["name"]],
                             "file": Path(rec["_source_file"]).name})
        else:
            twin = (by_members.get(rec.get("members")) if rec.get("members")
                    else by_syntax.get(rec.get("syntax")) if rec.get("syntax") else None)
            if twin is not None:
                renamed.append({"name": rec["name"], "full_id": rec["full_id"],
                                "file": Path(rec["_source_file"]).name,
                                "base_id": twin[0], "base_name": twin[1], "base_full_id": twin[2],
                                "base_rewritten": twin[5],
                                "same": "members" if rec.get("members") else "syntax"})
            else:
                new.append(rec)
    new.sort(key=sort_key)
    dup_new = len(new) - len({(r["name"], r["full_id"]) for r in new})
    kinds: dict[str, int] = {}
    for r in new:
        kinds[r["_kind"]] = kinds.get(r["_kind"], 0) + 1
    info = {
        "base_rows": len(rows),
        "pages_in_base": existing,
        "base_rows_without_page": [{"id": r[0], "name": r[1], "full_id": r[2], "rewritten": r[5]}
                                   for r in unmatched],
        "suspected_id_change": suspects,
        "renamed_in_base": renamed,
        "new_rows": len(new),
        "new_by_kind": kinds,
        "new_duplicate_pairs": dup_new,
    }
    return new, info


# --------------------------------------------------------------------------
# 3. quality agent (metered)
# --------------------------------------------------------------------------

class QualityStopped(RuntimeError):
    pass


class LlmMeter:
    """Counts the quality agent's calls, tokens and cost; stops it on a limit."""

    def __init__(self, budget_usd: float, caps: dict[str, int], max_consecutive_failures: int,
                 state: dict | None = None):
        self.lock = threading.Lock()
        self.budget = budget_usd
        self.caps = caps
        self.max_fail = max_consecutive_failures
        self.s = state or {
            "calls": {}, "attempts": {}, "failures": {}, "status": {}, "prompt_tokens": {},
            "completion_tokens": {}, "cost_usd": {}, "cost_reported": {}, "cost_estimated": {},
            "stopped": None,
        }
        self.consecutive = 0

    def _add(self, field: str, kind: str, n: float) -> None:
        d = self.s[field]
        d[kind] = d.get(kind, 0) + n

    def total_cost(self) -> float:
        return float(sum(self.s["cost_usd"].values()))

    def before_call(self, kind: str) -> None:
        with self.lock:
            if self.s["stopped"]:
                raise QualityStopped(f"quality agent stopped: {self.s['stopped']}")
            cap = self.caps.get(kind)
            if cap is not None and self.s["calls"].get(kind, 0) >= cap:
                self.s["stopped"] = f"{kind} call cap {cap} reached"
                raise QualityStopped(f"quality agent stopped: {self.s['stopped']}")
            self._add("calls", kind, 1)

    def after_call(self, kind: str, ok: bool) -> None:
        with self.lock:
            if ok:
                self.consecutive = 0
                return
            self._add("failures", kind, 1)
            self.consecutive += 1
            if self.consecutive >= self.max_fail and not self.s["stopped"]:
                self.s["stopped"] = f"{self.consecutive} consecutive failed calls"

    def response(self, kind: str, model: str, resp) -> None:
        usage: dict = {}
        try:
            if resp.status_code == 200:
                usage = (resp.json() or {}).get("usage") or {}
        except Exception:
            usage = {}
        with self.lock:
            self._add("attempts", kind, 1)
            code = str(resp.status_code)
            self.s["status"].setdefault(kind, {})
            self.s["status"][kind][code] = self.s["status"][kind].get(code, 0) + 1
            pt = int(usage.get("prompt_tokens") or 0)
            ct = int(usage.get("completion_tokens") or 0)
            self._add("prompt_tokens", kind, pt)
            self._add("completion_tokens", kind, ct)
            cost = usage.get("cost")
            if isinstance(cost, (int, float)):
                self._add("cost_usd", kind, float(cost))
                self._add("cost_reported", kind, 1)
            elif pt or ct:
                pin, pout = PRICE_PER_M.get(model, (3.0, 15.0))
                self._add("cost_usd", kind, (pt * pin + ct * pout) / 1e6)
                self._add("cost_estimated", kind, 1)
            if self.total_cost() >= self.budget and not self.s["stopped"]:
                self.s["stopped"] = f"budget ${self.budget:.2f} reached"

    def transport_error(self, kind: str) -> None:
        with self.lock:
            self._add("attempts", kind, 1)
            self.s["status"].setdefault(kind, {})
            self.s["status"][kind]["transport_error"] = self.s["status"][kind].get("transport_error", 0) + 1


def meter_client(client, kind: str, meter: LlmMeter):
    """Wrap one LLMClient: count calls and attempts, ask OpenRouter for usage.cost."""
    orig_generate = client.generate_text
    orig_build = client._build_request
    orig_post = client._client.post
    model = client.model

    def build(*a, **k):
        url, headers, payload = orig_build(*a, **k)
        payload["usage"] = {"include": True}
        return url, headers, payload

    def post(*a, **k):
        try:
            resp = orig_post(*a, **k)
        except Exception:
            meter.transport_error(kind)
            raise
        meter.response(kind, model, resp)
        return resp

    def generate_text(prompt, system_prompt=None):
        meter.before_call(kind)
        try:
            out = orig_generate(prompt, system_prompt)
        except Exception:
            meter.after_call(kind, False)
            raise
        meter.after_call(kind, True)
        return out

    client._build_request = build
    client._client.post = post
    client.generate_text = generate_text
    return client


def openrouter_key_usage() -> float | None:
    """Credits used by the OpenRouter key so far (GET /api/v1/key); None if unknown.
    Only the number is kept: the response also holds a key label."""
    key = os.environ.get("OPENROUTER_API_KEY", "")
    if not key:
        return None
    try:
        import httpx
        r = httpx.get("https://openrouter.ai/api/v1/key", headers={"Authorization": f"Bearer {key}"},
                      timeout=20)
        if r.status_code != 200:
            return None
        usage = (r.json().get("data") or {}).get("usage")
        return float(usage) if isinstance(usage, (int, float)) else None
    except Exception:
        return None


FAILURE_PREFIXES = ("audit_failed", "rewrite_failed", "stage1_thread_error", "stage2_thread_error")


def _short_issue(issue: str) -> str:
    """Failure issues lose the error text (it may quote a provider's response body)."""
    for prefix in FAILURE_PREFIXES:
        if issue.startswith(prefix):
            if "quality agent stopped" in issue:
                return f"{prefix} (quality agent stopped)"
            m = re.search(r"\b([45]\d\d)\b", issue[:200])
            return f"{prefix} (HTTP {m.group(1)})" if m else prefix
    return issue


def finish_quality_row(raw: dict, qa: dict | None) -> tuple[dict, dict]:
    """Merge one quality-agent result into the raw parse; returns (row, flags)."""
    row = {k: raw.get(k) for k in COLUMNS[:13]}
    flags = {"audited": False, "audit_failed": False, "stopped": False, "rewritten": False,
             "dropped_keys": [], "syntax_changed": False}
    if qa is None:
        row.update(quality_score=None, quality_issues=None, rewritten=0)
        return row, flags
    issues = [str(i) for i in (qa.get("_quality_issues") or [])]
    failed = any(i.startswith(("audit_failed", "stage1_thread_error")) for i in issues)
    stopped = failed and any("quality agent stopped" in i for i in issues)
    flags["audited"] = not failed
    flags["audit_failed"] = failed and not stopped
    flags["stopped"] = stopped
    if qa.get("_rewritten"):
        for k, v in qa.items():
            if k.startswith("_") or v == raw.get(k):
                continue
            if k in REWRITABLE and isinstance(v, str):
                row[k] = v
                flags["rewritten"] = True
                if k == "syntax":
                    flags["syntax_changed"] = True
            else:
                flags["dropped_keys"].append(k)
    short = [_short_issue(i) for i in issues]
    row["quality_score"] = None if failed else qa.get("_quality_score")
    row["quality_issues"] = "; ".join(short) if short else None
    row["rewritten"] = 1 if flags["rewritten"] else 0
    return row, flags


def stage_quality(new: list[dict], html_dir: Path, work: Path, args) -> tuple[list[dict], dict]:
    from pipeline.api_parser import quality_agent
    from revit_api_docs.config import load_config

    qdir = work / "quality"
    qdir.mkdir(parents=True, exist_ok=True)
    meter_file = qdir / "meter.json"
    if not args.resume:
        for p in qdir.glob("chunk_*.json"):
            p.unlink()
        meter_file.unlink(missing_ok=True)
    state = read_json(meter_file) if meter_file.is_file() else None
    meter = LlmMeter(args.quality_budget, {"stage1": args.stage1_cap, "stage2": args.max_rewrites},
                     args.max_consecutive_failures, state.get("meter") if state else None)
    elapsed = float(state.get("seconds", 0.0)) if state else 0.0
    usage_start = state.get("key_usage_start") if state else None
    if usage_start is None:
        usage_start = openrouter_key_usage()

    config = load_config()
    config.setdefault("proxy", {})["enabled"] = False
    real_create = quality_agent.create_llm_client

    def create(cfg, provider_override=None):
        client = real_create(cfg, provider_override=provider_override)
        kind = "stage1" if provider_override == "gemini" else "stage2"
        return meter_client(client, kind, meter)

    quality_agent.create_llm_client = create
    results: list[dict | None] = [None] * len(new)
    n_chunks = (len(new) + args.chunk_size - 1) // args.chunk_size
    ran_any = False
    try:
        for c in range(n_chunks):
            lo, hi = c * args.chunk_size, min(len(new), (c + 1) * args.chunk_size)
            cfile = qdir / f"chunk_{c:03d}.json"
            if cfile.is_file():
                saved = read_json(cfile)
                for i, r in enumerate(saved):
                    results[lo + i] = r
                continue
            if meter.s["stopped"]:
                break
            ran_any = True
            remaining2 = max(0, args.max_rewrites - int(meter.s["calls"].get("stage2", 0)))
            log(f"quality: chunk {c + 1}/{n_chunks} rows {lo + 1}-{hi}, stage-2 left {remaining2}, "
                f"cost so far ${meter.total_cost():.3f}")
            t0 = time.perf_counter()
            chunk = [dict(r) for r in new[lo:hi]]
            try:
                out = quality_agent.run_quality_agent(
                    chunk, config, str(html_dir), max_stage2=remaining2, verbose=args.verbose_quality,
                    failed_403_log_path=str(qdir / "stage2_403.log"))
            except RuntimeError as e:   # e.g. OpenRouter not reachable: no audit for the rest
                meter.s["stopped"] = f"quality agent error ({type(e).__name__}: {str(e)[:120]})"
                assert_no_secret(meter.s["stopped"], "the meter state")
                log(f"quality: {meter.s['stopped']}")
                break
            elapsed += time.perf_counter() - t0
            if len(out) != len(chunk):
                raise SystemExit(f"quality agent returned {len(out)} rows for {len(chunk)}")
            keep = [{k: v for k, v in r.items() if k not in ("_source_file", "_kind")} for r in out]
            text = json.dumps(keep, ensure_ascii=True)
            assert_no_secret(text, str(cfile))
            cfile.write_text(text, encoding="utf-8")
            for i, r in enumerate(keep):
                results[lo + i] = r
            write_json(meter_file, {"meter": meter.s, "seconds": round(elapsed, 1),
                                    "key_usage_start": usage_start})
            if meter.s["stopped"]:
                log(f"quality: stopping: {meter.s['stopped']}")
    finally:
        quality_agent.create_llm_client = real_create

    if ran_any:   # credits used by the key right after the last call of this step
        usage_end = openrouter_key_usage()
        write_json(meter_file, {"meter": meter.s, "seconds": round(elapsed, 1),
                                "key_usage_start": usage_start, "key_usage_end": usage_end})
    else:
        usage_end = state.get("key_usage_end") if state else None
    rows, n = [], {"audited": 0, "audit_failed": 0, "stopped_before_audit": 0, "not_audited": 0,
                   "rewritten": 0, "syntax_changed": 0, "dropped_keys": 0, "score_below_0_6": 0}
    scores = []
    for raw, qa in zip(new, results):
        row, flags = finish_quality_row(raw, qa)
        if qa is None:
            n["not_audited"] += 1
        n["audited"] += flags["audited"]
        n["audit_failed"] += flags["audit_failed"]
        n["stopped_before_audit"] += flags["stopped"]
        n["rewritten"] += flags["rewritten"]
        n["syntax_changed"] += flags["syntax_changed"]
        n["dropped_keys"] += len(flags["dropped_keys"])
        if row["quality_score"] is not None:
            scores.append(float(row["quality_score"]))
            n["score_below_0_6"] += float(row["quality_score"]) < 0.6
        rows.append(row)
    info = {
        "enabled": True,
        "models": {"stage1": config["llm"]["models"]["gemini"]["model"],
                   "stage2": config["llm"]["models"]["claude"]["model"]},
        "counts": n,
        "mean_score": round(sum(scores) / len(scores), 3) if scores else None,
        "meter": meter.s,
        "cost_usd_from_responses": round(meter.total_cost(), 4),
        "key_usage_delta_usd": (round(usage_end - usage_start, 4)
                                if usage_end is not None and usage_start is not None else None),
        "seconds": round(elapsed, 1),
    }
    return rows, info


def rows_without_quality(new: list[dict]) -> list[dict]:
    return [finish_quality_row(r, None)[0] for r in new]


# --------------------------------------------------------------------------
# 4. db
# --------------------------------------------------------------------------

def stage_db(base_db: Path, rows: list[dict], out_db: Path, reuse: bool = False) -> dict:
    """Copy of the base with `rows` appended. With `reuse` (--resume) an
    existing out_db is kept as it is (it may be packaged already) and only
    checked against the base and `rows`."""
    t0 = time.perf_counter()
    base_sha = sha256_file(base_db)
    if base_sha != V10_SHA["revit_api.db"]:
        raise SystemExit(f"--base-db is not the {BASE_RELEASE} revit_api.db (sha256 {base_sha[:16]}...)")
    conn = sqlite3.connect(ro_uri(base_db), uri=True)
    try:
        base_n, base_max = conn.execute("SELECT COUNT(*), MAX(id) FROM revit_api").fetchone()
        seq = conn.execute("SELECT seq FROM sqlite_sequence WHERE name = 'revit_api'").fetchone()[0]
    finally:
        conn.close()
    reused = reuse and out_db.is_file()
    if not reused:
        out_db.parent.mkdir(parents=True, exist_ok=True)
        for suffix in ("", "-journal", "-wal", "-shm"):
            Path(str(out_db) + suffix).unlink(missing_ok=True)
        shutil.copyfile(base_db, out_db)
        if sha256_file(out_db) != base_sha:
            raise SystemExit("the copy of the base db differs from the base")
        conn = sqlite3.connect(out_db)
        try:
            placeholders = ", ".join("?" * len(COLUMNS))
            conn.executemany(
                f"INSERT INTO revit_api ({', '.join(COLUMNS)}) VALUES ({placeholders})",
                [tuple(r[c] for c in COLUMNS) for r in rows])
            conn.commit()
            conn.execute("VACUUM")
        finally:
            conn.close()

    conn = sqlite3.connect(ro_uri(out_db), uri=True)
    try:
        total = conn.execute("SELECT COUNT(*) FROM revit_api").fetchone()[0]
        new_ids = [r[0] for r in conn.execute("SELECT id FROM revit_api WHERE id > ? ORDER BY id", (base_max,))]
        seq_after = conn.execute("SELECT seq FROM sqlite_sequence WHERE name = 'revit_api'").fetchone()[0]
        appended = conn.execute(f"SELECT {', '.join(COLUMNS)} FROM revit_api WHERE id > ? ORDER BY id",
                                (base_max,)).fetchall()
    finally:
        conn.close()

    # Assertions: base rows unchanged, new rows numbered after them and as built.
    assert total == base_n + len(rows), (total, base_n, len(rows))
    assert new_ids == list(range(seq + 1, seq + 1 + len(rows))), "new ids are not contiguous after the base"
    assert seq_after == seq + len(rows)
    if appended != [tuple(r[c] for c in COLUMNS) for r in rows]:
        raise SystemExit("the appended rows differ from this build's rows")
    a = sqlite3.connect(ro_uri(base_db), uri=True)
    b = sqlite3.connect(ro_uri(out_db), uri=True)
    try:
        cur_a = a.execute("SELECT * FROM revit_api ORDER BY id")
        cur_b = b.execute("SELECT * FROM revit_api WHERE id <= ? ORDER BY id", (base_max,))
        compared = 0
        while True:
            ra, rb = cur_a.fetchmany(2000), cur_b.fetchmany(2000)
            if ra != rb:
                raise SystemExit(f"base rows differ after row {compared}")
            if not ra:
                break
            compared += len(ra)
        typed = ("SELECT id, typeof(name), typeof(full_id), typeof(quality_score), typeof(rewritten) "
                 "FROM revit_api WHERE id <= ? ORDER BY id")
        if a.execute(typed.replace(" WHERE id <= ?", ""), ()).fetchall() != b.execute(typed, (base_max,)).fetchall():
            raise SystemExit("column types of the base rows differ")
        tables_a = a.execute("SELECT type, name, sql FROM sqlite_master ORDER BY name").fetchall()
        tables_b = b.execute("SELECT type, name, sql FROM sqlite_master ORDER BY name").fetchall()
        if tables_a != tables_b:
            raise SystemExit("schema differs from the base")
    finally:
        a.close()
        b.close()
    assert compared == base_n
    return {
        "base_rows": base_n, "base_max_id": base_max, "base_seq": seq,
        "appended": len(rows), "rows_total": total, "first_new_id": new_ids[0] if new_ids else None,
        "last_new_id": new_ids[-1] if new_ids else None, "base_rows_compared_equal": compared,
        "appended_rows_equal_build": True, "reused_existing": reused,
        "size": out_db.stat().st_size, "sha256": sha256_file(out_db),
        "seconds": round(time.perf_counter() - t0, 1),
    }


# --------------------------------------------------------------------------
# 5. embed (metered)
# --------------------------------------------------------------------------

class EmbedCapReached(RuntimeError):
    pass


def release_chroma_clients() -> None:
    """Stop every ChromaDB system of this process and forget it.

    embed_api_data's client stays registered in chromadb's process-wide cache:
    its files stay open, and a later client for the same path with other
    Settings (the retriever's) fails with "An instance of Chroma already
    exists ... with different settings". Stopping it also lets the store
    persist before it is packaged.
    """
    import gc

    from chromadb.api.shared_system_client import SharedSystemClient

    for system in list(SharedSystemClient._identifier_to_system.values()):
        try:
            system.stop()
        except Exception:
            pass
    SharedSystemClient.clear_system_cache()
    gc.collect()


def stage_embed(db: Path, chroma_dir: Path, work: Path, args) -> dict:
    """Embed every row (see embed_api_data); per-run request counts are kept in
    <work>/embed_stats.json so that --resume runs add up against the caps."""
    from pipeline.embedder import embed as embed_mod
    from revit_api_docs.config import load_config

    config = load_config()
    model_cfg = config["embedding"]["models"][config["embedding"]["provider"]]
    if not os.environ.get(model_cfg.get("api_key_env", "")):
        raise SystemExit("no embedding key in the environment (REVIT_API_DOCS_EMBEDDING_API_KEY)")
    stats_file = work / "embed_stats.json"
    history = read_json(stats_file) if args.resume and stats_file.is_file() else {"runs": []}
    prev_requests = sum(int(r.get("requests") or 0) for r in history["runs"])
    prev_texts = sum(int(r.get("texts") or 0) for r in history["runs"])
    if not args.resume and chroma_dir.exists():
        shutil.rmtree(chroma_dir)
    stats = {"requests": 0, "failed_requests": 0, "texts": 0, "prompt_tokens": 0, "cost_reported_usd": 0.0}
    lock = threading.Lock()
    real_create = embed_mod.create_embedding

    def create(cfg):
        emb = real_create(cfg)
        res = emb._client.embeddings
        orig = res.create

        def metered(*a, **k):
            n = len(k.get("input") or [])
            with lock:
                if (prev_requests + stats["requests"] >= args.embed_max_requests
                        or prev_texts + stats["texts"] + n > args.embed_max_texts):
                    raise EmbedCapReached(f"embedding cap reached ({stats['requests']} requests, "
                                          f"{stats['texts']} texts)")
                stats["requests"] += 1
                stats["texts"] += n
            try:
                resp = orig(*a, **k)
            except Exception:
                with lock:
                    stats["failed_requests"] += 1
                raise
            usage = getattr(resp, "usage", None)
            with lock:
                stats["prompt_tokens"] += int(getattr(usage, "prompt_tokens", 0) or 0)
                extra = getattr(usage, "model_extra", None) or {}
                if isinstance(extra.get("cost"), (int, float)):
                    stats["cost_reported_usd"] += float(extra["cost"])
            return resp

        res.create = metered
        return emb

    embed_mod.create_embedding = create
    t0 = time.perf_counter()
    try:
        embed_mod.embed_api_data(config, str(db), str(chroma_dir), batch_size=50, resume=args.resume,
                                 document_template=args.document_template)
    finally:
        embed_mod.create_embedding = real_create
        release_chroma_clients()
    pin = PRICE_PER_M[model_cfg["model"]][0] if model_cfg["model"] in PRICE_PER_M else 0.13
    stats["cost_estimated_usd"] = round(stats["prompt_tokens"] * pin / 1e6, 4)
    stats["cost_reported_usd"] = round(stats["cost_reported_usd"], 4)
    stats["model"] = model_cfg["model"]
    stats["document_template"] = args.document_template
    stats["seconds"] = round(time.perf_counter() - t0, 1)
    history["runs"].append(dict(stats, finished_at=_dt.datetime.now().isoformat(timespec="seconds")))
    write_json(stats_file, history)
    meta = read_json(chroma_dir / "meta.json")
    return {"this_run": stats, "runs": history["runs"],
            "requests_total": prev_requests + stats["requests"],
            "texts_total": prev_texts + stats["texts"],
            "meta_record_count": meta.get("record_count")}


# --------------------------------------------------------------------------
# 6. package
# --------------------------------------------------------------------------

def make_tarball(src_dir: Path, archive: Path) -> None:
    """chromadb_api.tar.gz with one top-level directory named after src_dir."""
    tmp = archive.with_suffix(".part")
    with tarfile.open(tmp, "w:gz", compresslevel=6) as tar:
        tar.add(src_dir, arcname=src_dir.name)
    os.replace(tmp, archive)


def fetch_code_tarball(dest_dir: Path) -> Path:
    target = dest_dir / "chromadb_code.tar.gz"
    if target.is_file() and sha256_file(target) == V10_SHA["chromadb_code.tar.gz"]:
        return target
    dest_dir.mkdir(parents=True, exist_ok=True)
    target.unlink(missing_ok=True)
    cmd = ["gh", "release", "download", BASE_RELEASE, "-R", GH_REPO, "-p", "chromadb_code.tar.gz",
           "-D", str(dest_dir)]
    log("package: " + " ".join(cmd))
    subprocess.run(cmd, check=True)
    return target


def stage_package(out: Path, base_db: Path, sdk_db: Path, code_tarball: Path | None) -> dict:
    from revit_api_docs import data

    t0 = time.perf_counter()
    rel = out / "release"
    rel.mkdir(parents=True, exist_ok=True)
    work = out / "work"

    # revit_sdk.db: the v1.0 file, unchanged.
    if sha256_file(sdk_db) != V10_SHA["revit_sdk.db"]:
        raise SystemExit(f"{sdk_db} is not the {BASE_RELEASE} revit_sdk.db")
    shutil.copyfile(sdk_db, rel / "revit_sdk.db")
    shutil.copyfile(sdk_db, out / "sqlite" / "revit_sdk.db")

    # chromadb_code.tar.gz: the v1.0 asset itself (repacking would change its hash).
    src = code_tarball if code_tarball else fetch_code_tarball(work / "gh")
    if sha256_file(src) != V10_SHA["chromadb_code.tar.gz"]:
        raise SystemExit(f"{src} is not the {BASE_RELEASE} chromadb_code.tar.gz")
    shutil.copyfile(src, rel / "chromadb_code.tar.gz")
    code_dir = out / "chromadb" / "chromadb_code"
    data._extract_tarball(rel / "chromadb_code.tar.gz", code_dir)

    # revit_api.db and chromadb_api.tar.gz: this build.
    shutil.copyfile(out / "sqlite" / "revit_api.db", rel / "revit_api.db")
    api_dir = out / "chromadb" / "chromadb_api"
    release_chroma_clients()   # no ChromaDB client may hold the store while it is packed
    layout = sorted(p.name for p in api_dir.iterdir())
    uuid_dirs = [p.name for p in api_dir.iterdir() if p.is_dir()]
    if sorted(set(layout) - set(uuid_dirs)) != ["chroma.sqlite3", "meta.json"] or len(uuid_dirs) != 1:
        raise SystemExit(f"unexpected chromadb_api layout (v1.0: chroma.sqlite3, meta.json and one "
                         f"segment dir): {layout}; rerun with --resume in a new process")
    make_tarball(api_dir, rel / "chromadb_api.tar.gz")
    with tarfile.open(rel / "chromadb_api.tar.gz", "r:gz") as tar:
        tops = sorted({m.name.split("/")[0] for m in tar.getmembers()})
    if tops != ["chromadb_api"]:
        raise SystemExit(f"tarball top level is {tops}")

    names = ("revit_api.db", "revit_sdk.db", "chromadb_api.tar.gz", "chromadb_code.tar.gz")
    assets = {n: {"sha256": sha256_file(rel / n), "size": (rel / n).stat().st_size} for n in names}
    (out / "SHA256SUMS.txt").write_text(
        "".join(f"{assets[n]['sha256']}  {n}\n" for n in names), encoding="ascii")

    targets = {"revit_api.db": "sqlite/revit_api.db", "revit_sdk.db": "sqlite/revit_sdk.db",
               "chromadb_api.tar.gz": "chromadb/chromadb_api", "chromadb_code.tar.gz": "chromadb/chromadb_code"}
    manifest = {
        "release": RELEASE,
        "source": "local build (pipeline/build_data_v1_1.py)",
        "installed_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "files": {n: {"target": targets[n], "release_sha256": assets[n]["sha256"],
                      "release_size": assets[n]["size"], "sha256": assets[n]["sha256"],
                      "size": assets[n]["size"], "verified": True} for n in names},
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return {"assets": assets, "chromadb_api_layout": layout, "tar_top_level": tops,
            "seconds": round(time.perf_counter() - t0, 1)}


# --------------------------------------------------------------------------
# 7. check (no API call)
# --------------------------------------------------------------------------

def stage_check(out: Path, seed: int = 20261026) -> dict:
    from revit_api_docs.retriever import _open_collection, check_vector_stores

    db = out / "sqlite" / "revit_api.db"
    sdk = out / "sqlite" / "revit_sdk.db"
    api_dir, code_dir = out / "chromadb" / "chromadb_api", out / "chromadb" / "chromadb_code"
    release_chroma_clients()
    probes = check_vector_stores(str(db), str(sdk), str(api_dir), str(code_dir))
    conn = sqlite3.connect(ro_uri(db), uri=True)
    try:
        names = dict(conn.execute("SELECT id, name FROM revit_api"))
    finally:
        conn.close()
    coll = _open_collection(str(api_dir), "revit_api")
    ids = sorted(names)
    sample = random.Random(seed).sample(ids, min(200, len(ids)))
    got = coll.get(ids=[str(i) for i in sample], include=["metadatas"])
    found = dict(zip(got["ids"], got["metadatas"]))
    ok = sum(1 for i in sample if (found.get(str(i)) or {}).get("name") == names[i])
    # Every record id is a row id and every row has one record.
    all_ids = set()
    total = coll.count()
    for off in range(0, total, 5000):
        all_ids.update(coll.get(limit=5000, offset=off, include=[])["ids"])
    rows_ids = {str(i) for i in ids}
    return {
        "probes": [p.__dict__ | {"describe": p.describe()} for p in probes],
        "random_records": len(sample), "random_records_name_equal": ok,
        "records": total, "rows": len(ids),
        "records_without_row": len(all_ids - rows_ids), "rows_without_record": len(rows_ids - all_ids),
    }


# --------------------------------------------------------------------------
# main
# --------------------------------------------------------------------------

def parse_args(argv: list[str] | None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--html-dir", required=True, type=Path, help="unpacked RevitAPI.chm (the .htm pages)")
    ap.add_argument("--base-db", required=True, type=Path, help="the v1.0-data revit_api.db (opened read-only)")
    ap.add_argument("--sdk-db", type=Path, help="the v1.0-data revit_sdk.db (default: next to --base-db)")
    ap.add_argument("--code-tarball", type=Path,
                    help="the v1.0-data chromadb_code.tar.gz (default: gh release download)")
    ap.add_argument("--out", type=Path, default=REPO_ROOT / "out" / "v1.1-data")
    ap.add_argument("--quality", action=argparse.BooleanOptionalAction, default=True,
                    help="run the quality agent on the new rows (default: on)")
    ap.add_argument("--max-rewrites", type=int, default=500, help="stage-2 (Claude) call cap")
    ap.add_argument("--stage1-cap", type=int, default=1300, help="stage-1 (Gemini) call cap")
    ap.add_argument("--quality-budget", type=float, default=15.0, help="USD; the quality agent stops there")
    ap.add_argument("--max-consecutive-failures", type=int, default=20)
    ap.add_argument("--max-suspects", type=int, default=20,
                    help="stop when more pages than this have a base name with another full_id")
    ap.add_argument("--chunk-size", type=int, default=100, help="rows per quality-agent run (checkpoint)")
    ap.add_argument("--verbose-quality", action="store_true", help="the quality agent's own progress output")
    ap.add_argument("--embed-max-texts", type=int, default=29500)
    ap.add_argument("--embed-max-requests", type=int, default=600)
    ap.add_argument("--document-template", choices=("NS", "FS", "N"), default="NS",
                    help="embedded text per API row: NS name : summary (default, the v1.1-data "
                         "store), FS full_id : summary, N name only (pipeline/embedder/embed.py)")
    ap.add_argument("--limit", type=int, help="smoke test: keep only the first N new rows")
    ap.add_argument("--resume", action="store_true", help="reuse the finished steps under <out>/work")
    ap.add_argument("--skip-embed", action="store_true", help="stop after the SQLite step")
    return ap.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    out = args.out.resolve()
    work = out / "work"
    work.mkdir(parents=True, exist_ok=True)
    base_db = args.base_db.resolve()
    sdk_db = (args.sdk_db or base_db.with_name("revit_sdk.db")).resolve()
    if out in base_db.parents:
        raise SystemExit("--out must not contain --base-db")
    import chromadb

    report: dict[str, Any] = {
        "release": RELEASE, "base_release": BASE_RELEASE,
        "started_at": _dt.datetime.now().isoformat(timespec="seconds"),
        "args": {"html_dir": str(args.html_dir), "base_db": str(base_db), "out": str(out),
                 "quality": args.quality, "max_rewrites": args.max_rewrites, "stage1_cap": args.stage1_cap,
                 "quality_budget": args.quality_budget, "limit": args.limit, "resume": args.resume,
                 "skip_embed": args.skip_embed, "document_template": args.document_template},
        "python": sys.version.split()[0], "chromadb": chromadb.__version__,
    }
    t_all = time.perf_counter()

    recs, pstats = stage_parse(args.html_dir, work, args.resume)
    skipped = pstats.pop("skipped")
    report["parse"] = pstats
    report["parse"]["skipped_sample"] = random.Random(8).sample(skipped, min(20, len(skipped)))
    report["parse"]["skipped_titles_in_base"] = None

    new, cinfo = stage_classify(recs, base_db)
    conn = sqlite3.connect(ro_uri(base_db), uri=True)
    try:
        base_names = {n for (n,) in conn.execute("SELECT name FROM revit_api")}
    finally:
        conn.close()
    report["parse"]["skipped_titles_in_base"] = sum(1 for s in skipped if s["title"] in base_names)
    report["classify"] = cinfo
    held = len(cinfo["suspected_id_change"]) + len(cinfo["renamed_in_base"])
    log(f"classify: in base {cinfo['pages_in_base']}, new {cinfo['new_rows']} {cinfo['new_by_kind']}, "
        f"suspected id change {len(cinfo['suspected_id_change'])}, "
        f"renamed in base {len(cinfo['renamed_in_base'])}")
    if held > args.max_suspects:
        report["blocked"] = (f"{held} pages are held back as probably the same page as a base row "
                             f"under another full_id or name (limit {args.max_suspects})")
        write_json(out / "build_report.json", report)
        log("BLOCKED: " + report["blocked"])
        return 3
    if args.limit:
        new = new[:args.limit]
        report["classify"]["limited_to"] = len(new)

    if args.quality and new:
        rows, qinfo = stage_quality(new, args.html_dir, work, args)
    else:
        rows, qinfo = rows_without_quality(new), {"enabled": False}
    report["quality"] = qinfo
    for r in rows:
        assert_no_secret(json.dumps(r, ensure_ascii=True), "the SQLite rows")

    report["db"] = stage_db(base_db, rows, out / "sqlite" / "revit_api.db", reuse=args.resume)
    log(f"db: {report['db']['rows_total']} rows, ids {report['db']['first_new_id']}-"
        f"{report['db']['last_new_id']} appended"
        + (" (existing file kept and checked)" if report["db"]["reused_existing"] else ""))

    if not args.skip_embed:
        report["embed"] = stage_embed(out / "sqlite" / "revit_api.db", out / "chromadb" / "chromadb_api",
                                      work, args)
        log(f"embed: {report['embed']['this_run']['requests']} requests this run, "
            f"{report['embed']['requests_total']} in all; {report['embed']['texts_total']} texts")
        report["package"] = stage_package(out, base_db, sdk_db, args.code_tarball)
        report["check"] = stage_check(out)
        for p in report["check"]["probes"]:
            log("check: " + p["describe"])
    report["seconds_total"] = round(time.perf_counter() - t_all, 1)
    report["finished_at"] = _dt.datetime.now().isoformat(timespec="seconds")
    text = json.dumps(report, indent=2, ensure_ascii=True)
    assert_no_secret(text, "build_report.json")
    (out / "build_report.json").write_text(text + "\n", encoding="utf-8")
    log(f"done in {report['seconds_total']} s; report {out / 'build_report.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
