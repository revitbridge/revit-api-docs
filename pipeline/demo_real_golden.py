"""
SDK 代码清洗 Agent — 真实 LLM 现场跑一次 Golden 生成
=====================================================

不同于 demo_sdk_cleaning.py（读库里已存的 golden），本脚本**真实调用**
extract.py 的各阶段函数，对单个工程现场跑完整流程：

    Phase 0  discover_sdk_projects   找到目标工程
    Phase 1  analyze_readme          真实 Gemini Flash 解析 readme
    Phase 1b match_and_extract       提取目标类方法体（无 tree-sitter 时走 regex 回退）
    Phase 2  generate_golden_code    真实 Claude 生成 {summary, content}

证明库里的 golden code 是真·LLM 产出，而非写死。

前置：.env 里要有 OPENROUTER_API_KEY（脚本自动读取注入环境）。
走代理：设置环境变量 HTTPS_PROXY / HTTP_PROXY（见文末运行命令），或在 config.yaml 开 proxy.enabled。

运行：
    python -m pipeline.demo_real_golden --project CreateBeamsColumnsBraces
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
_DEFAULT_SDK_ROOT = Path(os.getenv("REVIT_SDK_ROOT", "F:/Revit 2026.3 SDK/Samples"))
_OUT_DIR = _ROOT / "out" / "demo"

_C = {"head": "\033[1;36m", "ok": "\033[32m", "bad": "\033[31m",
      "warn": "\033[33m", "dim": "\033[90m", "b": "\033[1m", "x": "\033[0m"}


def c(k: str, s: str) -> str:
    return f"{_C[k]}{s}{_C['x']}"


def main() -> None:
    ap = argparse.ArgumentParser(description="真实 LLM 现场跑 SDK Golden 生成")
    ap.add_argument("--project", default="CreateBeamsColumnsBraces", help="工程名")
    ap.add_argument("--sdk-root", default=str(_DEFAULT_SDK_ROOT))
    args = ap.parse_args()

    from revit_api_docs.config import load_config, load_dotenv
    load_dotenv()
    if not os.getenv("OPENROUTER_API_KEY"):
        raise SystemExit("OPENROUTER_API_KEY 未配置（.env 或系统环境）。")

    from pipeline.sdk_parser.extract import (
        discover_sdk_projects, analyze_readme, match_and_extract,
        generate_golden_code, create_llm_client,
    )

    cfg = load_config()
    proxy = os.getenv("HTTPS_PROXY") or os.getenv("HTTP_PROXY") or "(none / direct)"

    print(c("head", "═" * 70))
    print(c("head", f" SDK Golden 生成 · 真实 LLM 现场跑  —  工程：{args.project}"))
    print(c("head", "═" * 70))
    print(c("dim", f"  代理: {proxy}"))

    # ── Phase 0 ──
    print(c("head", "\n【Phase 0】discover_sdk_projects"))
    projects = discover_sdk_projects(str(Path(args.sdk_root)))
    target = next(
        (p for p in projects if p["project_name"].split(".")[-1].lower() == args.project.lower()
         or p["project_name"].lower() == args.project.lower()),
        None,
    )
    if not target:
        raise SystemExit(f"未发现工程 {args.project}（在 {args.sdk_root} 下）。")
    print(f"  project   : {target['project_name']}")
    print(f"  .cs 文件  : {len(target['all_cs_files'])}")
    print(f"  readme    : {target['readme_path'].name if target['readme_path'] else '(none)'}")
    raw_total = 0
    for p in target["all_cs_files"]:
        raw_total += sum(1 for _ in p.open(encoding="utf-8", errors="ignore"))
    print(f"  原始总行数: {c('bad', str(raw_total))} 行")

    sdk_cfg = cfg.get("sdk", {})
    gemini = create_llm_client(cfg, provider_override=sdk_cfg.get("stage1_provider", "gemini_flash"))
    gemini.max_tokens = 1024
    claude = create_llm_client(cfg, provider_override=sdk_cfg.get("stage2_provider", "claude"))
    claude.max_tokens = 4096

    # ── Phase 1 (real Gemini) ──
    print(c("head", f"\n【Phase 1】analyze_readme — 真实调用 {gemini.model}"))
    readme_analysis = analyze_readme(target, gemini)
    if readme_analysis is None:
        raise SystemExit("ReadMe 分析失败（无 readme 或 LLM 出错）。")
    print(f"  target_files          : {readme_analysis.get('target_files')}")
    print(f"  key_classes_and_methods: {readme_analysis.get('key_classes_and_methods')}")
    print(f"  mentioned_apis        : {str(readme_analysis.get('mentioned_apis'))[:140]}")
    if readme_analysis.get("_error"):
        print(c("bad", f"  _error: {readme_analysis['_error']}"))

    # ── Phase 1b (tree-sitter / regex fallback) ──
    print(c("head", "\n【Phase 1b】match_and_extract — 提取目标类方法体"))
    matched = match_and_extract(target, readme_analysis, verbose=True)
    if matched is None:
        raise SystemExit("没有匹配到任何目标类，无法生成 golden。")
    details = matched["code_details"]
    extracted_chars = sum(len(m) for d in details for m in d["methods"])
    print(f"  匹配到的类/文件: {len(details)}")
    for d in details:
        print(c("dim", f"    - {d['class_name']} @ {d['filename']}  ({len(d['methods'])} methods)"))
    print(f"  提取代码字符数 : {extracted_chars}")

    # ── Phase 2 (real Claude) ──
    print(c("head", f"\n【Phase 2】generate_golden_code — 真实调用 {claude.model} …"))
    golden = generate_golden_code(matched, claude)
    if not golden:
        raise SystemExit("Golden 生成失败（LLM 错误或输出不可用）。")

    content = golden["content"]
    golden_lines = len(content.splitlines())
    pct = 100 - golden_lines * 100 // max(1, raw_total)
    print(c("ok", f"\n  ✓ 生成成功  {c('bad', str(raw_total))} 行 → {c('ok', str(golden_lines))} 行（压缩 {pct}%）"))
    print(f"  summary       : {golden['summary']}")
    print(f"  mentioned_apis: {str(golden['mentioned_apis'])[:140]}")
    print(c("dim", "\n  ── 现场生成的 golden code（前 30 行）──"))
    for ln in content.splitlines()[:30]:
        print(c("dim", "    " + ln))
    if golden_lines > 30:
        print(c("dim", f"    … 余 {golden_lines - 30} 行"))

    # ── 保存现场产物 ──
    _OUT_DIR.mkdir(parents=True, exist_ok=True)
    import re as _re
    safe = _re.sub(r"[^A-Za-z0-9_.-]", "_", args.project)
    live_out = _OUT_DIR / f"{safe}.golden.live.cs"
    live_out.write_text(content, encoding="utf-8")
    print(c("ok", f"\n✓ 现场产物已存：{live_out}"))


if __name__ == "__main__":
    main()
