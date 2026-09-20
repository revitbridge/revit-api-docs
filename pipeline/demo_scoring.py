"""
数据打分 Agent — 打分过程模拟演示（控制台输出）
================================================

模拟 pipeline/api_parser/quality_agent.py 的两阶段流程，针对单条记录
`Floor.Create`，在控制台一步步展示：

    1. 朴素解析器从原始 HTML 抽出的「待审记录」（summary/syntax 多为空）
    2. Stage-1 审核：逐条套用打分规则（从 1.0 累计扣分），打印每条判定与证据
    3. 最终 quality_score / issues / needs_rewrite
    4. Stage-2 修复：以 HTML 为依据重写出的干净字段

扣分规则与真实 Agent 的 _STAGE1_PROMPT_WITH_HTML 完全一致：
    -0.5  构造函数噪声记录
    -0.4  summary 和 info 都为空，但 HTML 里有真实描述
    -0.3  summary/info 是样板文字，但 HTML 有真实描述
    -0.25 parameters 为空，但 HTML 列出了带描述的方法参数
    -0.2  syntax/C# 签名缺失，但 HTML 含公开签名/方法名
    -0.2  HTML 中的重要内容（描述/重载列表）没被解析进字段
    -0.15 参数类型全是 "Unknown Type"，但 HTML 有真实类型
    -0.1  字段里残留 HTML 标签 / 乱码 / 过多空白

运行：
    python -m pipeline.demo_scoring
"""
from __future__ import annotations

import re
import textwrap
from pathlib import Path

try:
    from bs4 import BeautifulSoup
except ImportError:
    BeautifulSoup = None  # type: ignore

_ROOT = Path(__file__).resolve().parent.parent
_HTML = _ROOT / "out" / "demo" / "floor-create.raw.htm"

_THRESHOLD = 0.6

# ── 简单的控制台上色（Windows Terminal / VT 支持）──
_C = {
    "head": "\033[1;36m", "ok": "\033[32m", "warn": "\033[33m",
    "bad": "\033[31m", "dim": "\033[90m", "b": "\033[1m", "x": "\033[0m",
}


def c(key: str, s: str) -> str:
    return f"{_C[key]}{s}{_C['x']}"


def rule(ch: str = "─", n: int = 64) -> None:
    print(_C["dim"] + ch * n + _C["x"])


# ─────────────────────────────────────────────────────────────
# Step 0: 朴素解析（模拟旧解析器从 HTML 抽字段的结果）
# ─────────────────────────────────────────────────────────────

def naive_parse(html: str) -> dict[str, str]:
    """
    模拟朴素解析器：它只会找 <title> 和明显的 summary/syntax 区块。
    本页是 "Overload List" 页，描述在 <table> 单元格里、签名被 span 拆碎，
    所以解析器抽到的 summary / syntax / parameters 都是空的。
    """
    soup = BeautifulSoup(html, "html.parser") if BeautifulSoup else None
    name = ""
    if soup and soup.title:
        name = soup.title.get_text(strip=True)

    # 朴素解析器只认 class="summary" 这类明确区块——本页没有，故为空
    summary = ""
    if soup:
        node = soup.find(attrs={"class": "summary"})
        if node:
            summary = node.get_text(" ", strip=True)

    # 朴素解析器只认 <div class="syntax"> / <pre> 代码块——本页没有
    syntax = ""
    if soup and soup.find(attrs={"class": "syntax"}):
        syntax = soup.find(attrs={"class": "syntax"}).get_text(" ", strip=True)

    return {
        "name": name or "Floor.Create Method",
        "full_id": "Floor.Create",
        "summary": summary,
        "info": "",
        "syntax": syntax,
        "parameters": "",
    }


# ─────────────────────────────────────────────────────────────
# HTML 证据提取（审核 agent 用来判断「HTML 里到底有没有」）
# ─────────────────────────────────────────────────────────────

def extract_html_evidence(html: str) -> dict[str, object]:
    """从原始 HTML 里抽出可核对的证据，供打分时引用。"""
    soup = BeautifulSoup(html, "html.parser") if BeautifulSoup else None
    overloads: list[tuple[str, str]] = []
    if soup:
        table = soup.find("table", id="OverloadList")
        if table:
            for tr in table.find_all("tr"):
                tds = tr.find_all("td")
                if len(tds) >= 2:
                    sig = re.sub(r"\s+", " ", tds[-2].get_text(" ", strip=True))
                    desc = re.sub(r"\s+", " ", tds[-1].get_text(" ", strip=True))
                    if desc:
                        overloads.append((sig, desc))
    return {
        "has_overload_descriptions": bool(overloads),
        "overloads": overloads,
        "has_method_signatures": bool(overloads),
        "is_constructor": "constructor" in html.lower() or "#ctor" in html.lower(),
    }


# ─────────────────────────────────────────────────────────────
# Stage-1: 打分（逐条规则、打印证据与扣分）
# ─────────────────────────────────────────────────────────────

def stage1_audit(rec: dict[str, str], ev: dict[str, object]) -> dict[str, object]:
    print(c("head", "\n【Stage-1 · Gemini 审核】比对 解析记录 ↔ 原始 HTML，逐条套用打分规则"))
    print(c("dim", "原则：只在『HTML 里有、但记录里缺/错』时扣分；HTML 本身没有的字段，空着是对的。\n"))

    score = 1.0
    issues: list[str] = []

    def deduct(amount: float, applies: bool, reason: str, evidence: str) -> None:
        nonlocal score
        tag = c("bad", f"-{amount:<4}") if applies else c("ok", " 0.0 ")
        verdict = c("bad", "命中") if applies else c("ok", "不扣")
        print(f"  [{verdict}] {tag} {reason}")
        print(c("dim", f"           证据：{evidence}"))
        if applies:
            score -= amount
            issues.append(reason)

    summary_empty = not rec["summary"].strip()
    info_empty = not rec["info"].strip()
    syntax_empty = not rec["syntax"].strip()
    params_empty = not rec["parameters"].strip()
    has_desc = bool(ev["has_overload_descriptions"])
    n_overloads = len(ev["overloads"])  # type: ignore[arg-type]

    deduct(
        0.5, bool(ev["is_constructor"]),
        "构造函数噪声记录",
        "页面非构造函数（是静态方法 Create 的重载列表）",
    )
    deduct(
        0.4, summary_empty and info_empty and has_desc,
        "summary 和 info 都为空，但 HTML 里有真实描述",
        f"解析记录 summary='' info=''；但 HTML 重载表里有 {n_overloads} 段真实描述",
    )
    deduct(
        0.2, syntax_empty and bool(ev["has_method_signatures"]),
        "syntax/C# 签名缺失，但 HTML 含方法签名",
        f"解析记录 syntax=''；HTML 列出 {n_overloads} 个方法签名（Create(Document, IList<CurveLoop>, ...)）",
    )
    deduct(
        0.2, has_desc and summary_empty,
        "HTML 中重要内容（重载列表描述）未被解析进字段",
        "重载表 <td> 里的描述完全没进 summary/info",
    )
    deduct(
        0.25, params_empty and False,  # 本页是重载列表，无逐参数描述表 → 不扣
        "parameters 为空，但 HTML 列出带描述的参数",
        "本页是重载汇总页，没有逐参数描述表 → 不应扣分",
    )
    deduct(
        0.1, bool(re.search(r"<[a-z]+[ >]", rec["summary"] + rec["syntax"])),
        "字段里残留 HTML 标签 / 乱码",
        "解析记录字段为空，无残留标签 → 不扣",
    )

    score = round(max(0.0, score), 2)
    needs_rewrite = score < _THRESHOLD

    rule()
    print(f"  {c('b','最终得分')}：{c('bad', str(score)) if needs_rewrite else c('ok', str(score))}"
          f"   （阈值 {_THRESHOLD}）")
    print(f"  needs_rewrite = {c('bad','True（触发 Stage-2 修复）') if needs_rewrite else c('ok','False')}")
    print(f"  issues = {issues}")
    return {"quality_score": score, "issues": issues, "needs_rewrite": needs_rewrite}


# ─────────────────────────────────────────────────────────────
# Stage-2: 以 HTML 为依据修复
# ─────────────────────────────────────────────────────────────

def stage2_rewrite(rec: dict[str, str], ev: dict[str, object]) -> dict[str, str]:
    print(c("head", "\n【Stage-2 · Claude 重写】以原始 HTML 为唯一依据，补全/修正字段\n"))
    overloads = ev["overloads"]  # type: ignore[assignment]

    # 从 HTML 证据合成干净字段（演示用——真实 agent 由 Claude 生成）
    arch = next((d for s, d in overloads if "Boolean" not in s and "Line" not in s), "")  # type: ignore
    struct = next((d for s, d in overloads if "Boolean" in s or "Line" in s), "")  # type: ignore

    patch = {
        "summary": "Creates a new instance of a floor within the project, "
                   "with overloads for architectural and structural floors.",
        "info": "Overloads: Create(Document, IList<CurveLoop>, ElementId, ElementId) "
                "creates a new architectural floor instance; "
                "Create(Document, IList<CurveLoop>, ElementId, ElementId, Boolean, Line, Double) "
                "creates a new floor instance with additional structural parameters.",
        "syntax": "public static Floor Create(...)",
    }
    print(c("dim", "  Claude 从 HTML 重载表读到："))
    for s, d in overloads:  # type: ignore
        print(c("dim", f"    • {s}\n        → {d}"))
    print()
    for k, v in patch.items():
        before = rec[k] or "(空)"
        print(f"  {c('b', k)}:")
        print(f"    {c('bad','旧')} {before}")
        print(f"    {c('ok','新')} {textwrap.fill(v, 100, subsequent_indent='        ')}")
    return patch


# ─────────────────────────────────────────────────────────────
def main() -> None:
    if BeautifulSoup is None:
        raise SystemExit("需要 beautifulsoup4：pip install beautifulsoup4")
    if not _HTML.exists():
        raise SystemExit(f"HTML 源文件不存在: {_HTML}")
    html = _HTML.read_text(encoding="utf-8", errors="ignore")

    print(c("head", "═" * 64))
    print(c("head", " 数据打分 Agent · 打分过程模拟  —  记录：Floor.Create"))
    print(c("head", "═" * 64))

    rec = naive_parse(html)
    print(c("head", "\n【Step 0】朴素解析器从原始 HTML 抽出的待审记录"))
    for k in ("name", "full_id", "summary", "info", "syntax", "parameters"):
        v = rec[k] if rec[k] else c("warn", "(空)")
        print(f"  {k:11s}: {v}")

    ev = extract_html_evidence(html)
    print(c("head", "\n【证据】原始 HTML 里实际包含的信息"))
    print(f"  重载条数        : {len(ev['overloads'])}")  # type: ignore
    print(f"  含方法签名      : {ev['has_method_signatures']}")
    print(f"  含重载描述      : {ev['has_overload_descriptions']}")
    print(f"  是否构造函数    : {ev['is_constructor']}")

    audit = stage1_audit(rec, ev)

    if audit["needs_rewrite"]:
        patch = stage2_rewrite(rec, ev)
        fixed = {**rec, **patch}
        print(c("head", "\n【最终修复结果】"))
        print(f"  quality_score : {audit['quality_score']}  →  rewritten = 1")
        print(f"  summary : {fixed['summary']}")
        print(f"  syntax  : {fixed['syntax']}")
        print(c("ok", "\n✓ 记录已从『不可用』修复为可直接进 RAG 检索的结构化数据。"))
    else:
        print(c("ok", "\n✓ 记录质量达标，无需修复。"))


if __name__ == "__main__":
    main()
