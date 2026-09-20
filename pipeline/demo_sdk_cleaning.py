"""
SDK 代码清洗 Agent — before/after 演示生成器
=============================================

展示 SDK 清洗 Agent（pipeline/sdk_parser/extract.py，V2 五阶段）把
**原始 SDK 样例工程（数百~数千行噪音）** 提炼成 **一段 golden code（聚焦、
参数化、带 XML 文档、API 锚定）** 的效果。

Agent 工作流程（与 extract.py 完全一致）
---------------------------------------
    Phase 0  项目发现        os.walk 找 CS/ 目录，过滤 boilerplate
                             （AssemblyInfo / *.Designer.cs / obj / bin / Properties）
    Phase 1  ReadMe 分析     Gemini Flash 读 RTF readme → target_files / key_classes / mentioned_apis
    Phase 1b Tree-sitter 提取 只从目标类里抽方法体（丢掉无关代码）
    Phase 2  Golden 生成     Claude：选核心逻辑块 → 删 UI/TaskDialog/Execute 样板
                             → 硬编码改参数 → Transaction 包裹 → 加 XML 文档 → {summary, content}
    Phase 3  落库            写入 SQLite sdk_info(project, summary, content, mentioned_apis)

数据源
------
- BEFORE: F:/Revit 2026.3 SDK/Samples/<Project>/  下的原始 .cs（可用 --sdk-root 覆盖）
- AFTER : data/sqlite/revit_sdk.db  表 sdk_info 的 content（golden code）

输出
----
- docs/demo/sdk-cleaning-before-after.md
- docs/demo/<project>.raw.cs        原始主文件（初始材料）
- docs/demo/<project>.golden.cs      清洗后的 golden code

运行
----
    python -m pipeline.demo_sdk_cleaning
    python -m pipeline.demo_sdk_cleaning --project NewRebar
"""
from __future__ import annotations

import argparse
import os
import re
import sqlite3
from pathlib import Path
from typing import Any

_ROOT = Path(__file__).resolve().parent.parent
_SDK_DB = _ROOT / "data" / "sqlite" / "revit_sdk.db"
_DEFAULT_SDK_ROOT = Path(os.getenv("REVIT_SDK_ROOT", "F:/Revit 2026.3 SDK/Samples"))
_OUT_DIR = _ROOT / "out" / "demo"

_C = {"head": "\033[1;36m", "ok": "\033[32m", "bad": "\033[31m",
      "warn": "\033[33m", "dim": "\033[90m", "b": "\033[1m", "x": "\033[0m"}


def c(k: str, s: str) -> str:
    return f"{_C[k]}{s}{_C['x']}"


# Phase 0 过滤规则（与 extract.py 一致）
def _is_boilerplate(p: Path) -> bool:
    s = str(p).replace("\\", "/").lower()
    if "/obj/" in s or "/bin/" in s or "/properties/" in s:
        return True
    n = p.name.lower()
    return n == "assemblyinfo.cs" or n.endswith(".designer.cs") or n.endswith("assemblyattributes.cs")


def collect_raw(project_dir: Path) -> list[Path]:
    """收集工程里参与清洗的 .cs（已按 Phase 0 规则过滤 boilerplate）。"""
    return sorted(
        p for p in project_dir.rglob("*.cs") if not _is_boilerplate(p)
    )


def analyze_noise(files: list[Path]) -> dict[str, Any]:
    """统计原始代码里的『噪音』，量化清洗前的负担。"""
    total_lines = 0
    license_lines = 0
    usings: set[str] = set()
    ui_lines = 0          # WinForms / 控件 / 对话框
    hardcoded: list[str] = []
    taskdialog = msgbox = console = 0
    in_license = False

    ui_pat = re.compile(r"\b(TaskDialog|MessageBox|System\.Windows\.Forms|\.Text\s*=|\.Checked|"
                        r"Form\b|Button|TextBox|ComboBox|DataGridView)\b")
    using_pat = re.compile(r"^\s*using\s+([\w.]+)\s*;")
    # 硬编码：.First/FirstOrDefault(x => x.Name == "literal")
    hard_pat = re.compile(r'\.First(?:OrDefault)?\([^)]*==\s*"([^"]+)"')

    for fp in files:
        for i, line in enumerate(fp.open(encoding="utf-8", errors="ignore")):
            total_lines += 1
            stripped = line.strip()
            # license / 文件头注释块（开头连续的 // 或 /* */）
            if i < 60 and (stripped.startswith("//") or stripped.startswith("/*")
                           or in_license or stripped.startswith("*")):
                if "copyright" in stripped.lower() or "license" in stripped.lower() \
                        or in_license or stripped.startswith("*") or i < 30 and stripped.startswith("//"):
                    license_lines += 1
                if stripped.startswith("/*"):
                    in_license = True
                if "*/" in stripped:
                    in_license = False
            m = using_pat.match(line)
            if m:
                usings.add(m.group(1))
            if ui_pat.search(line):
                ui_lines += 1
            taskdialog += line.count("TaskDialog")
            msgbox += line.count("MessageBox")
            console += line.count("Console.")
            for hm in hard_pat.findall(line):
                hardcoded.append(hm)

    return {
        "total_lines": total_lines,
        "license_lines": license_lines,
        "using_count": len(usings),
        "usings": sorted(usings),
        "ui_lines": ui_lines,
        "taskdialog": taskdialog,
        "msgbox": msgbox,
        "console": console,
        "hardcoded": hardcoded,
    }


def load_golden(project: str) -> dict[str, Any] | None:
    if not _SDK_DB.exists():
        raise FileNotFoundError(f"SDK DB 不存在: {_SDK_DB}")
    conn = sqlite3.connect(str(_SDK_DB))
    conn.row_factory = sqlite3.Row
    try:
        row = conn.execute(
            "SELECT project, summary, content, mentioned_apis FROM sdk_info "
            "WHERE project = ? OR project LIKE ? ORDER BY length(content) DESC LIMIT 1",
            (project, project + ".%"),
        ).fetchone()
        return dict(row) if row else None
    finally:
        conn.close()


def main() -> None:
    ap = argparse.ArgumentParser(description="SDK 代码清洗 Agent before/after 演示")
    ap.add_argument("--project", default="CreateBeamsColumnsBraces",
                    help="工程名（默认 CreateBeamsColumnsBraces；其他如 AutoRoute / Loads / NewRebar）")
    ap.add_argument("--sdk-root", default=str(_DEFAULT_SDK_ROOT), help="Revit SDK Samples 根目录")
    args = ap.parse_args()

    project = args.project
    sdk_root = Path(args.sdk_root)
    project_dir = sdk_root / project

    print(c("head", "═" * 70))
    print(c("head", f" SDK 代码清洗 Agent · 前 vs 后  —  工程：{project}"))
    print(c("head", "═" * 70))

    # ── Agent 工作流程 ──
    print(c("head", "\n【Agent 工作流程】(pipeline/sdk_parser/extract.py · V2 五阶段)"))
    for ph in [
        "Phase 0  项目发现       过滤 AssemblyInfo / *.Designer.cs / obj / bin / Properties",
        "Phase 1  ReadMe 分析    Gemini Flash → target_files / key_classes / mentioned_apis",
        "Phase 1b Tree-sitter    只从目标类抽方法体",
        "Phase 2  Golden 生成    Claude：删 UI/TaskDialog/Execute、硬编码→参数、加 XML 文档",
        "Phase 3  落库           写入 sqlite sdk_info(project, summary, content, mentioned_apis)",
    ]:
        print(c("dim", "  " + ph))

    golden = load_golden(project)
    if not golden:
        raise SystemExit(f"sdk_info 里找不到工程 '{project}'。换一个 --project（如 NewRebar / SpanDirection）。")

    # ── BEFORE ──
    if not project_dir.exists():
        raise SystemExit(f"原始工程目录不存在: {project_dir}（用 --sdk-root 指定 SDK Samples 路径）")
    files = collect_raw(project_dir)
    noise = analyze_noise(files)
    main_file = max(files, key=lambda p: p.stat().st_size) if files else None

    print(c("head", "\n【BEFORE — 原始 SDK 工程（初始材料）】"))
    print(f"  参与清洗的 .cs 文件 : {len(files)}")
    print(f"  代码总行数          : {c('bad', str(noise['total_lines']))} 行")
    print(f"  主文件              : {main_file.name if main_file else '?'}")
    print(c("warn", "  —— 其中属于『噪音』的部分 ——"))
    print(f"    license/文件头注释 : ~{noise['license_lines']} 行")
    print(f"    using 引用         : {noise['using_count']} 个")
    print(f"    UI/表单/控件 相关行: {noise['ui_lines']} 行")
    print(f"    TaskDialog 调用    : {noise['taskdialog']}    MessageBox: {noise['msgbox']}    Console: {noise['console']}")
    if noise["hardcoded"]:
        print(f"    硬编码字面量       : {c('bad', str(len(noise['hardcoded'])))} 处，例如 "
              + ", ".join(f'\"{h}\"' for h in noise["hardcoded"][:4]))

    # ── AFTER ──
    golden_code = golden["content"] or ""
    golden_lines = len(golden_code.splitlines())
    apis = golden["mentioned_apis"] or "[]"
    print(c("head", "\n【AFTER — Golden Code（清洗后）】"))
    print(f"  代码行数   : {c('ok', str(golden_lines))} 行   "
          f"（{c('bad', str(noise['total_lines']))} → {c('ok', str(golden_lines))}，"
          f"压缩 {100 - golden_lines * 100 // max(1, noise['total_lines'])}%）")
    print(f"  summary    : {golden['summary']}")
    print(f"  API 锚定   : {apis[:140]}")
    print(c("dim", "\n  ── golden code 预览（前 28 行）──"))
    for ln in golden_code.splitlines()[:28]:
        print(c("dim", "    " + ln))
    if golden_lines > 28:
        print(c("dim", f"    … 余 {golden_lines - 28} 行"))

    # ── 写文件 ──
    _OUT_DIR.mkdir(parents=True, exist_ok=True)
    safe = re.sub(r"[^A-Za-z0-9_.-]", "_", project)
    raw_out = _OUT_DIR / f"{safe}.raw.cs"
    golden_out = _OUT_DIR / f"{safe}.golden.cs"
    if main_file:
        raw_out.write_text(main_file.read_text(encoding="utf-8", errors="ignore"), encoding="utf-8")
    golden_out.write_text(golden_code, encoding="utf-8")

    md = _render_md(project, files, main_file, noise, golden, golden_lines)
    (_OUT_DIR / "sdk-cleaning-before-after.md").write_text(md, encoding="utf-8")

    print(c("ok", f"\n✓ 已生成："))
    print(f"  {_OUT_DIR / 'sdk-cleaning-before-after.md'}")
    print(f"  {raw_out}")
    print(f"  {golden_out}")


def _render_md(project, files, main_file, noise, golden, golden_lines) -> str:
    apis = golden["mentioned_apis"] or "[]"
    pct = 100 - golden_lines * 100 // max(1, noise["total_lines"])
    L = []
    L.append(f"# SDK 代码清洗 · 前 vs 后 —— `{project}`")
    L.append("")
    L.append(f"> {noise['total_lines']} 行原始噪音  →  {golden_lines} 行高信息密度可复用代码"
             f"（压缩 {pct}%）")
    L.append("")
    L.append("## Agent 工作流程（pipeline/sdk_parser/extract.py · V2 五阶段）")
    L.append("")
    L.append("| 阶段 | 模型 | 做什么 |")
    L.append("|------|------|--------|")
    L.append("| Phase 0 项目发现 | — | os.walk 找 CS/ 目录，过滤 AssemblyInfo / *.Designer.cs / obj / bin |")
    L.append("| Phase 1 ReadMe 分析 | Gemini Flash | 读 RTF readme → target_files / key_classes / mentioned_apis |")
    L.append("| Phase 1b Tree-sitter 提取 | — | 只从目标类里抽方法体 |")
    L.append("| Phase 2 Golden 生成 | Claude | 删 UI/TaskDialog/Execute 样板、硬编码→参数、Transaction 包裹、加 XML 文档 |")
    L.append("| Phase 3 落库 | — | 写入 sqlite `sdk_info(project, summary, content, mentioned_apis)` |")
    L.append("")
    L.append("## BEFORE — 原始 SDK 工程（初始材料）")
    L.append("")
    L.append(f"- 参与清洗的 `.cs` 文件：**{len(files)}** 个，主文件 `{main_file.name if main_file else '?'}`")
    L.append(f"- 代码总行数：**{noise['total_lines']}** 行")
    L.append(f"- 噪音：license/文件头注释 ~{noise['license_lines']} 行、`using` {noise['using_count']} 个、"
             f"UI/表单相关 {noise['ui_lines']} 行、TaskDialog×{noise['taskdialog']}、MessageBox×{noise['msgbox']}")
    if noise["hardcoded"]:
        L.append(f"- 硬编码字面量 **{len(noise['hardcoded'])}** 处："
                 + ", ".join(f"`\"{h}\"`" for h in noise["hardcoded"][:6]))
    L.append("")
    L.append(f"原始主文件见同目录 `{re.sub(r'[^A-Za-z0-9_.-]', '_', project)}.raw.cs`。")
    L.append("")
    L.append("## AFTER — Golden Code（清洗后）")
    L.append("")
    L.append(f"**summary**：{golden['summary']}")
    L.append("")
    L.append(f"**API 锚定**：`{apis}`")
    L.append("")
    L.append("```csharp")
    L.append(golden["content"] or "")
    L.append("```")
    L.append("")
    L.append(f"**一句话**：原始工程是『能跑的完整 demo』——夹着 license、using、UI 表单、TaskDialog、"
             f"硬编码值；Agent 把它提炼成一段聚焦核心 API、参数化、带文档、可直接进 RAG 检索的 golden code。")
    return "\n".join(L)


if __name__ == "__main__":
    main()
