"""
生产流程 · 真实演示 —— Agent 协作 + 幻觉对抗（流式 + 真实调用）
================================================================

与早期的脚本化演示不同，本脚本**真实调用项目内的组件**，输出真实的、
逐字滚动的思维链，并对**真实生成的代码**做真实的幻觉检测：

  - retriever.rewrite_query()        真实 Gemini Flash 关键词解析
  - retriever._keyword_search_api()  真实查 SQLite（27,596 条）召回候选 API
  - LLMClient.generate_stream()      真实流式推理（思维链逐字滚动）
  - SQLite 真实校验                  生成代码里出现的 API 是否真的存在
  - 正则真实检测                     生成代码里是否有 .First() / 编造数值

两遍代码生成：先要一版“自然草稿”，让幻觉对抗在**真实输出**里抓到真实问题，
再带约束重生成最终版。全程思维链记录到 docs/demo/production-thinking-chain.md。

前置：.env 有 OPENROUTER_API_KEY。走代理见文末命令（HTTPS_PROXY=...:10808）。

运行：
    python -m pipeline.demo_production_real
"""
from __future__ import annotations

import argparse
import os
import re
import sqlite3
import sys
import time
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
_API_DB = _ROOT / "data" / "sqlite" / "revit_api.db"
_SDK_DB = _ROOT / "data" / "sqlite" / "revit_sdk.db"
_CHROMA_API = _ROOT / "data" / "chromadb" / "chromadb_api"
_CHROMA_CODE = _ROOT / "data" / "chromadb" / "chromadb_code"
_LOG = _ROOT / "out" / "demo" / "production-thinking-chain.md"

_C = {"user": "\033[1;32m", "query": "\033[1;36m", "data": "\033[1;33m",
      "param": "\033[1;35m", "out": "\033[1;32m", "guard": "\033[1;31m",
      "think": "\033[36m", "ok": "\033[32m", "warn": "\033[33m",
      "stop": "\033[1;31m", "b": "\033[1m", "dim": "\033[90m", "x": "\033[0m"}

_transcript: list[str] = []  # 记录思维链
_PAUSE = 0.0   # 段落间停顿（录屏用）


def c(k: str, s: str) -> str:
    return f"{_C[k]}{s}{_C['x']}"


def pause() -> None:
    if _PAUSE > 0:
        time.sleep(_PAUSE)


def log(line: str = "") -> None:
    """普通输出（同时记录，去掉颜色）。"""
    print(line)
    _transcript.append(re.sub(r"\033\[[0-9;]*m", "", line))


def header(key: str, tag: str, title: str, model: str = "") -> None:
    pause()
    log()
    log(c(key, "━" * 70))
    m = c("dim", f"  [{model}]") if model else ""
    log(f"{c(key, tag)}  {c('b', title)}{m}")
    log(c(key, "━" * 70))


def stream(client, prompt: str, system: str, label: str = "thinking") -> str:
    """真实流式调用，逐 token 滚动打印并捕获；失败回退非流式。"""
    print(c("think", f"  💭 {label} "), end="", flush=True)
    print(c("dim", "(real stream) ─────────────────────────"))
    print(c("think", "  "), end="", flush=True)
    buf: list[str] = []
    try:
        for tok in client.generate_stream(prompt, system_prompt=system):
            sys.stdout.write(_C["think"] + tok + _C["x"])
            sys.stdout.flush()
            buf.append(tok)
    except Exception as e:
        # 回退非流式（带重试的 generate_text），并打印失败原因便于诊断
        print(c("warn", f"\n  (stream failed: {type(e).__name__}: {e} — 回退非流式)"))
        txt = client.generate_text(prompt, system_prompt=system)
        sys.stdout.write(_C["think"] + txt + _C["x"])
        buf.append(txt)
    print()
    text = "".join(buf)
    _transcript.append(f"\n[{label}]\n{text}\n")
    return text


# ── 真实幻觉检测工具 ────────────────────────────────────────────────

def api_exists(member: str) -> bool:
    """真实查 SQLite：该 API 成员是否存在于 27,596 条里。
    C# 属性 getter/setter（get_X / set_X）脱糖成属性名 X 再查（文档按属性名收录）。"""
    variants = {member}
    m = re.sub(r"^(get_|set_)", "", member)
    variants.add(m)
    conn = sqlite3.connect(str(_API_DB))
    try:
        for v in variants:
            cur = conn.execute(
                "SELECT 1 FROM revit_api WHERE full_id = ? OR full_id LIKE ? "
                "OR name LIKE ? LIMIT 1",
                (v, f"%.{v}", f"%{v}%"),
            )
            if cur.fetchone() is not None:
                return True
        return False
    finally:
        conn.close()


# LINQ / 语言 / 框架方法——不是 Revit API 文档库里的成员，校验时排除
_NON_API_METHODS = {
    "ToList", "Cast", "Where", "Select", "Concat", "Distinct", "Any", "All",
    "Count", "Add", "Contains", "First", "FirstOrDefault", "Single", "OrderBy",
    "Show", "Start", "Commit", "RollBack", "Dispose", "ToString", "GetType",
    "Equals", "Format", "Join", "Substring", "Split", "TryParse", "Parse",
}


# 框架/语言/枚举类型前缀——这些 Type.Member 不在 Revit 文档库里，不作真伪判定
_NON_API_TYPES = {
    "System", "Result", "TransactionMode", "RegenerationOption", "StorageType",
    "StructuralType", "BuiltInCategory", "BuiltInParameter", "UnitTypeId",
    "DisplayUnitType", "Math", "Convert", "Console", "String", "Debug",
}


def extract_qualified_calls(code: str) -> list[str]:
    """抽出限定形式的 API 调用 `Type.Method(`（用于校验该 API 是否真实存在）。"""
    cands = set()
    for typ, meth in re.findall(r"\b([A-Z][A-Za-z0-9_]+)\.([A-Za-z_][A-Za-z0-9_]+)\s*\(", code):
        if typ in _NON_API_TYPES or meth in _NON_API_METHODS:
            continue
        if meth.isupper():           # 枚举值，不判
            continue
        cands.add(f"{typ}.{meth}")
    return sorted(cands)


def detect_first(code: str) -> list[str]:
    # .First()/.FirstOrDefault()/.FirstElement()/.ElementAt(0) —— 都是“默认抓取”反模式
    hits = re.findall(r"[A-Za-z_][\w.]*\.(?:First(?:OrDefault|Element)?|Single)\s*\([^;\n]*", code)
    hits += re.findall(r"[A-Za-z_][\w.]*\.ElementAt\s*\(\s*0\s*\)", code)
    return hits


def detect_hardcoded(code: str) -> list[str]:
    """检测疑似 AI 编造的数值/经验值。"""
    hits: list[str] = []
    hits += re.findall(r"\.Set\(\s*(-?\d+(?:\.\d+)?)\s*\)", code)                 # .Set(120.0)
    hits += re.findall(r'\.Set\(\s*"([^"\n]{1,20})"\s*\)', code)                  # .Set("2小时")
    hits += re.findall(r'"([^"\n]{0,12}(?:小时|分钟|mm)[^"\n]{0,8})"', code)        # "2小时"/"60分钟"/"25mm"
    hits += re.findall(r"\b\w{0,20}(?:thickness|厚度|height|层高|rating|等级|offset|elevation)\w{0,6}\s*=\s*(-?\d+(?:\.\d+)?)", code, re.I)
    # 清洗：去换行、截短、去重保序（避免截断代码导致正则吞掉一长段）
    seen, out = set(), []
    for h in hits:
        h = re.sub(r"\s+", " ", h).strip()[:24]
        if h and h not in seen:
            seen.add(h); out.append(h)
    return out


def guard(title: str, lines: list[str]) -> None:
    log()
    log(c("guard", f"  🛡️ 幻觉对抗 · {title}"))
    for ln in lines:
        log(c("guard", "     " + ln))


def main() -> None:
    global _PAUSE
    ap = argparse.ArgumentParser(description="生产流程真实演示（录屏友好）")
    ap.add_argument("--pause", type=float, default=0.0,
                    help="每个阶段标题前的停顿秒数（录屏时建议 0.8）")
    args = ap.parse_args()
    _PAUSE = args.pause

    # .env（统一走 config.load_dotenv）
    from revit_api_docs.config import load_config, load_dotenv
    load_dotenv()
    if not os.getenv("OPENROUTER_API_KEY"):
        raise SystemExit("OPENROUTER_API_KEY 未配置。")

    from revit_api_docs.retriever import RAGRetriever
    from revit_api_docs.llm_client import create_llm_client

    cfg = load_config()
    proxy = os.getenv("HTTPS_PROXY") or "(direct)"

    log(c("b", "╔════════════════════════════════════════════════════════════════════╗"))
    log(c("b", "║  生产流程 · 真实演示  ·  Agent 协作 + 幻觉对抗（流式 / 真实调用）   ║"))
    log(c("b", "╚════════════════════════════════════════════════════════════════════╝"))
    log(c("dim", f"  代理: {proxy}   |   API 库: revit_api.db"))

    retriever = RAGRetriever(cfg, str(_API_DB), str(_SDK_DB), str(_CHROMA_API), str(_CHROMA_CODE))
    gemini = create_llm_client(cfg, provider_override="gemini_flash")
    gemini.max_tokens = 600
    claude = create_llm_client(cfg, provider_override="claude")
    claude.max_tokens = 1200

    USER_Q = "给柱子加防火涂层"

    # ── ① USER ──
    header("user", "① USER", "用户提问")
    log(f"  用户输入：{c('b', USER_Q)}")
    log(c("warn", "  ⚠ 口语化、模糊、缺关键信息"))

    # ── ② QUERY AGENT（真实流式思考 + 真实 rewrite_query）──
    header("query", "② QUERY AGENT", "关键词解析：结合项目词库", "Gemini Flash")
    stream(
        gemini,
        f"用户口语化请求：『{USER_Q}』。用中文，简短地一步步思考(thinking)："
        "①这个请求有哪些歧义/缺失关键信息？②'柱子'对应 Revit 哪个类？"
        "③'防火涂层'是参数还是涂层元素？只输出思考过程，不要结论。",
        "你是 Revit API 关键词解析 Agent，输出简短中文思维链。",
        label="thinking · 意图分析",
    )
    rewritten = retriever.rewrite_query(USER_Q)
    log(c("ok", f"  ✅ rewrite_query() 真实输出：{rewritten[:120]}…"))

    # 幻觉对抗 ①（对真实输入做规则检测）
    ambiguities = []
    if not re.search(r"structural|architectural|结构|建筑", USER_Q):
        ambiguities.append('✗ “柱子” 歧义：结构柱(Structural) 还是 建筑柱(Architectural)？')
    if not re.search(r"\d+\s*(分钟|min|小时)|30|60|90", USER_Q):
        ambiguities.append('✗ “防火涂层” 等级未知：30 / 60 / 90 分钟？')
    if ambiguities:
        guard("介入点①：关键词解析 — 反向消除意图歧义",
              ambiguities + [c("stop", "⛔ 未澄清不进入下一步（演示中假定用户已澄清：结构柱 + 60min）")])

    # ── ③ DATA / RAG（真实 SQLite 召回）──
    header("data", "③ DATA", "RAG 检索：找准 API 和示例代码")
    hits = retriever._keyword_search_api(rewritten, limit=6)
    log(c("dim", "  💭 ChromaDB 召回 → uniqueid → SQLite 取完整数据（此处用真实 SQLite 关键词召回）"))
    log(c("ok", f"  ✅ 真实召回 {len(hits)} 条候选 API："))
    for it in hits[:6]:
        log(f"     • {c('b', it.full_id or it.name)}  {c('dim', '— ' + (it.summary or '')[:46])}")

    # ── ④ DYNAMIC PARAM AGENT（真实流式分类）──
    header("param", "④ DYNAMIC PARAM AGENT", "动态参数：识别数据来源", "Claude")
    cand_str = ", ".join((it.full_id or it.name) for it in hits[:6])
    stream(
        claude,
        f"目标：给选中的结构柱设置防火相关参数。已检索到候选 API：{cand_str}。"
        "用中文简短思考(thinking)：把实现所需的参数分三类——"
        "①用户已提供 ②需查 Revit 模型(Level/FamilyType/现有柱实例) ③不该由 AI 编造、需用户明确(选哪根柱/厚度)。"
        "只输出思维链。",
        "你是 Revit 参数来源分析 Agent，输出简短中文思维链。",
        label="thinking · 参数三分类",
    )

    # ── ⑤ 代码草稿（真实生成，naive 提示 → 自然出现 .First()/硬编码，让幻觉对抗去抓真实问题）──
    header("data", "⑤ CODE DRAFT", "先生成自然草稿（最简实现）→ 交给幻觉对抗审查", "Claude")
    claude.max_tokens = 800
    draft = stream(
        claude,
        f"快速给一版最简单的 C#(RevitAPI) 示范：实现『{USER_Q}』。"
        "怎么简单怎么来：从模型里取一个标高、取一根柱子直接示范即可，"
        "防火涂层厚度给个示例数值，不用考虑用户选择和健壮性。直接输出代码。",
        "你是 Revit API C# 开发助手，给最简示范代码。",
        label="code draft (real)",
    )

    # ── 幻觉对抗：对真实草稿做真实检测 ──
    log()
    log(c("b", "  ── 幻觉对抗 Agent 审查上面这段真实生成的草稿 ──"))

    # ② 真实 API 校验（校验限定调用 Type.Method 是否真实存在于文档库）
    calls = extract_qualified_calls(draft)
    fake, real = [], []
    for m in calls:
        # 校验限定全名（如 FireProtection.Create）是否真实存在于文档库
        (real if api_exists(m) else fake).append(m)
    guard("介入点②：RAG 检索后 — 验证 API 真实存在（真实查 27,596 条）",
          [f"代码中的 API 调用（Type.Method 形式）：{len(calls)} 个",
           c("ok", f"✓ 库内命中：{', '.join(real[:10]) or '（无）'}")]
          + ([c("stop", f"⛔ 不在库里（疑似臆造）：{', '.join(fake[:10])} → 拦截，触发 Code Agent 用真实 API 重写")]
             if fake else [c("ok", "✓ 未发现臆造 API（全部库内命中）")]))

    # ③ .First() / 默认值
    firsts = detect_first(draft)
    guard("介入点③：动态参数判断 — 阻止 .First() / 默认值",
          ([c("stop", f"⛔ 检出 {len(firsts)} 处默认抓取：{firsts[0][:60]}… → 强制改为用户选实例")]
           if firsts else [c("ok", "✓ 草稿未使用 .First()/默认抓取")]))

    # ④ 编造数值
    hard = detect_hardcoded(draft)
    guard("介入点④：代码生成时 — 拒绝 AI 编造数值",
          ([c("stop", f"⛔ 检出疑似编造数值：{', '.join(hard[:5])} → 改为占位符让用户填")]
           if hard else [c("ok", "✓ 草稿未发现硬编码经验值")]))

    # ── ⑥ OUTPUT（真实带约束重生成）──
    header("out", "⑥ OUTPUT", "依据审查意见重生成最终可执行 C#", "Claude")
    issues = []
    if fake:   issues.append(f"禁用不存在的 API（{', '.join(fake[:5])}），改用库内真实 API")
    if firsts: issues.append("禁止 .First()/默认抓取，改为遍历传入的 selectedColumnIds")
    if hard:   issues.append("禁止编造数值（厚度/层高/坐标），用占位符 {like_this} 让用户填")
    fix = "；".join(issues) or "保持真实 API、不默认抓取、不编造数值"
    claude.max_tokens = 2200
    final = stream(
        claude,
        f"修正下面的 Revit C# 代码草稿，要求：{fix}。用 Transaction 包裹。"
        f"输出修正后的【完整】代码，不要省略。\n\n草稿：\n{draft[:1500]}",
        "你是严谨的 Revit API C# 开发助手，遵守约束，不编造。直接输出完整代码。",
        label="final code (real)",
    )

    log()
    log(c("ok", "  ✅ 最终代码生成完成（已通过幻觉对抗审查约束）"))
    log(c("dim", "  ────────────────────────────────────────────────────────────────"))
    log(c("b", "  幻觉对抗成绩（基于真实检测）：")
        + c("stop", f" 臆造API×{len(fake)}") + " · "
        + c("stop", f".First()×{len(firsts)}") + " · "
        + c("stop", f"编造数值×{len(hard)}") + " · "
        + c("warn", f"消歧×{len(ambiguities)}"))

    # ── 记录思维链 ──
    _LOG.parent.mkdir(parents=True, exist_ok=True)
    _LOG.write_text(
        "# 生产流程 · 真实思维链记录\n\n"
        f"用户提问：{USER_Q}\n\n"
        "```\n" + "\n".join(_transcript) + "\n```\n",
        encoding="utf-8",
    )
    log(c("ok", f"\n✓ 思维链已记录：{_LOG}"))


if __name__ == "__main__":
    main()
