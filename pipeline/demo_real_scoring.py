"""
数据打分 Agent — 真实 LLM 打分演示（接通 OpenRouter）
=====================================================

不同于 demo_scoring.py（按规则模拟），本脚本**真实调用** quality_agent.py 里的
两阶段函数，对 `Floor.Create` 跑一次真打分：

    Stage-1  google/gemini-3-flash-preview   审核 + 打分
    Stage-2  anthropic/claude-sonnet-4.6      以 HTML 为依据重写（仅当低于阈值）

控制台会打印每一步的真实 LLM 返回（quality_score / issues / 重写后的字段）。

前置：.env 里要有 OPENROUTER_API_KEY（脚本会自动读取注入环境）。

运行：
    python -m pipeline.demo_real_scoring
"""
from __future__ import annotations

import json
import os
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
_RAW_HTML = _ROOT / "out" / "demo" / "floor-create.raw.htm"

_C = {"head": "\033[1;36m", "ok": "\033[32m", "bad": "\033[31m",
      "warn": "\033[33m", "dim": "\033[90m", "b": "\033[1m", "x": "\033[0m"}


def c(k: str, s: str) -> str:
    return f"{_C[k]}{s}{_C['x']}"


def main() -> None:
    from revit_api_docs.config import load_config, load_dotenv
    load_dotenv()
    if not os.getenv("OPENROUTER_API_KEY"):
        raise SystemExit("OPENROUTER_API_KEY 未配置（.env 或系统环境）。")

    from pipeline.api_parser.quality_agent import (
        check_connectivity, _stage1_audit, _stage2_rewrite, create_llm_client,
        _QUALITY_THRESHOLD,
    )

    cfg = load_config()

    print(c("head", "═" * 64))
    print(c("head", " 数据打分 Agent · 真实 LLM 打分  —  记录：Floor.Create"))
    print(c("head", "═" * 64))

    # ── 连通性 ──
    print(c("dim", "\n检查 OpenRouter 连通性 …"))
    conn = check_connectivity(cfg)
    print(f"  proxy : {conn['proxy_addr']}")
    print(f"  api   : {c('ok','OK') if conn['api_ok'] else c('bad','FAILED: '+str(conn['error']))}")
    if not conn["api_ok"]:
        raise SystemExit("OpenRouter 不可达，请检查网络/代理（config.proxy.enabled）。")

    # ── 待审记录（朴素解析结果：关键字段空，指向真实 HTML 源文件）──
    item = {
        "name": "Floor.Create Method",
        "full_id": "Floor.Create",
        "namespace": "Autodesk.Revit.DB",
        "summary": "",
        "info": "",
        "syntax": "",
        "parameters": "",
        "remark": "",
        "_source_file": str(_RAW_HTML),   # 让 agent 读取真实原始 HTML
    }
    print(c("head", "\n【待审记录】（朴素解析，关键字段为空）"))
    for k in ("name", "full_id", "summary", "syntax", "parameters"):
        print(f"  {k:11s}: {item[k] if item[k] else c('warn','(空)')}")
    print(c("dim", f"  原始 HTML 源 : {_RAW_HTML.name}"))

    gemini = create_llm_client(cfg, provider_override="gemini")
    claude = create_llm_client(cfg, provider_override="claude")
    claude.max_tokens = 4096

    # ── Stage-1（真实 Gemini）──
    print(c("head", f"\n【Stage-1 · 真实调用 {gemini.model}】审核 + 打分 …"))
    audit = _stage1_audit(item, gemini, html_dir=None)
    score = audit.get("quality_score")
    issues = audit.get("issues", [])
    needs = audit.get("needs_rewrite", score < _QUALITY_THRESHOLD if score is not None else False)
    print(f"  quality_score : {c('bad', str(score)) if needs else c('ok', str(score))}  （阈值 {_QUALITY_THRESHOLD}）")
    print(f"  html_found    : {audit.get('_html_found')}")
    print(f"  needs_rewrite : {c('bad','True') if needs else c('ok','False')}")
    print("  issues :")
    for i in issues:
        print(c("dim", f"    - {i}"))

    if not needs:
        print(c("ok", "\n✓ Gemini 判定质量达标，无需修复。"))
        return

    # ── Stage-2（真实 Claude）──
    print(c("head", f"\n【Stage-2 · 真实调用 {claude.model}】以 HTML 为依据重写 …"))
    patch = _stage2_rewrite(item, issues, None, claude)
    if "_rewrite_error" in patch:
        print(c("bad", f"  重写失败：{patch['_rewrite_error']}"))
        return

    for k, v in patch.items():
        if k.startswith("_"):
            continue
        old = item.get(k) or "(空)"
        print(f"  {c('b', k)}:")
        print(f"    {c('bad','旧')} {old}")
        print(f"    {c('ok','新')} {v}")

    fixed = {**item, **{k: v for k, v in patch.items() if not k.startswith('_')}}
    print(c("head", "\n【最终修复结果】（真实 LLM 产出）"))
    print(json.dumps(
        {k: fixed[k] for k in ("name", "full_id", "summary", "info", "syntax", "parameters")},
        ensure_ascii=False, indent=2,
    ))
    print(c("ok", f"\n✓ Floor.Create 由 {score} 分的不可用记录，经 Claude 重写为完整结构化数据。"))


if __name__ == "__main__":
    main()
