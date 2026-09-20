"""
数据双库映射演示 —— ChromaDB ↔ SQLite（一页 PPT 用，向量缩略）
==============================================================

强调对照：同一 `uniqueid`（= SQLite 主键 id）把两库连起来。
  - ChromaDB（向量库）：只存 `id → embedding(of "full_id : summary")`，承担语义召回
  - SQLite（结构化库）：存 `id → 完整字段`，承担精确查询

真实数据来自 data/sqlite/revit_api.db（27,596 条）。
向量按需求**缩略**（只标维度，不堆浮点数），以突出映射关系。

嵌入文本规则与 pipeline/embedder/embed.py 一致：document = f"{full_id} : {summary}"
关联键与 pipeline/retriever.py 一致：ChromaDB id == SQLite id（_hydrate_api: WHERE id IN ...）

运行：
    python -m pipeline.demo_dual_store
    python -m pipeline.demo_dual_store --ids 23245,869,71,23034
"""
from __future__ import annotations

import argparse
import sqlite3
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
_API_DB = _ROOT / "data" / "sqlite" / "revit_api.db"
_OUT = _ROOT / "out" / "demo" / "dual-store-mapping.md"
_EMBED_DIM = 3072  # OpenAI text-embedding-3-large（config: embedding.models.openai.dimension）

_C = {"head": "\033[1;36m", "ok": "\033[32m", "warn": "\033[33m",
      "dim": "\033[90m", "b": "\033[1m", "x": "\033[0m"}


def c(k: str, s: str) -> str:
    return f"{_C[k]}{s}{_C['x']}"


def _short(s: str | None, n: int) -> str:
    s = (s or "").strip().replace("\n", " ")
    return (s[:n].rstrip() + "…") if len(s) > n else s


def load(ids: list[int]) -> list[dict]:
    conn = sqlite3.connect(str(_API_DB))
    conn.row_factory = sqlite3.Row
    out = []
    for i in ids:
        r = conn.execute(
            "SELECT id, full_id, name, namespace, summary, syntax, remark, exceptions "
            "FROM revit_api WHERE id = ?", (i,),
        ).fetchone()
        if r:
            out.append(dict(r))
    conn.close()
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description="ChromaDB ↔ SQLite 双库映射演示（向量缩略）")
    ap.add_argument("--ids", default="23245,869,71,23034",
                    help="逗号分隔的 SQLite id（默认 wall/create 主题 4 条）")
    args = ap.parse_args()
    ids = [int(x) for x in args.ids.split(",") if x.strip()]

    rows = load(ids)
    _tconn = sqlite3.connect(str(_API_DB))
    try:
        total = _tconn.execute("SELECT COUNT(*) FROM revit_api").fetchone()[0]
    finally:
        _tconn.close()
    vec = f"[ …{_EMBED_DIM} 维浮点… ]"

    print(c("head", "═" * 74))
    print(c("head", " 数据双库映射 · ChromaDB ↔ SQLite   （uniqueid = SQLite 主键 id）"))
    print(c("head", "═" * 74))
    print(c("dim", f"  共 {total:,} 条；嵌入文本 = \"full_id : summary\"；向量缩略示意"))

    print(c("head", "\n  ChromaDB（向量库 · 仅语义召回）            SQLite（结构化库 · 完整字段）"))
    print(c("dim", "  " + "─" * 70))
    for r in rows:
        uid = c("b", f"{r['id']:>6}")
        print(f"  {uid}  summary_embedding {vec}")
        print(f"          ↳ embed(\"{_short(r['full_id'] + ' : ' + (r['summary'] or ''), 58)}\")")
        print(f"          {c('ok','→ SQLite[' + str(r['id']) + ']')}: "
              f"{{ name:{_short(r['name'],22)}, ns:{r['namespace']}, "
              f"syntax:{_short(r['syntax'],18)},")
        extra = []
        extra.append(f"remark:{'有' if (r['remark'] or '').strip() else '∅'}")
        extra.append(f"exceptions:{'有' if (r['exceptions'] or '').strip() else '∅'} }}")
        print(f"            { ', '.join(extra) }")
        print()

    _OUT.parent.mkdir(parents=True, exist_ok=True)
    _OUT.write_text(_render_md(rows, total, vec), encoding="utf-8")
    print(c("ok", f"✓ 已写出（可直接贴 PPT）：{_OUT}"))


def _render_md(rows: list[dict], total: int, vec: str) -> str:
    L = []
    L.append("# 数据双库映射 · ChromaDB ↔ SQLite")
    L.append("")
    L.append(f"共 {total:,} 条 · uniqueid 连接两库 · 向量 {_EMBED_DIM} 维（text-embedding-3-large）")
    L.append("")
    L.append("| uniqueid | ChromaDB（向量库） | SQLite（结构化库） |")
    L.append("|:--------:|--------------------|--------------------|")
    for r in rows:
        embed_text = f"`{r['full_id']} : {_short(r['summary'], 44)}`"
        full = (f"`{{ name, namespace, syntax, "
                f"remark{'✓' if (r['remark'] or '').strip() else '∅'}, "
                f"exceptions{'✓' if (r['exceptions'] or '').strip() else '∅'}, … }}`")
        L.append(f"| **{r['id']}** | summary_embedding {vec} ← {embed_text} | {full} |")
    L.append("")
    L.append("用户提问「create wall」 → Embedding → ChromaDB top-k → uniqueid 列表 "
             "→ SQLite 按 id 取完整 JSON → Code Agent")
    L.append("")
    L.append("改 syntax / remark / exceptions → 只更新 SQLite（0 次 embedding）；"
             "改 summary → 重新 embedding，两库同步")
    return "\n".join(L)


if __name__ == "__main__":
    main()
