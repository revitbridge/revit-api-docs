
"""
向量化模块 — 将 API/SDK 数据 embedding 后存入 ChromaDB
"""
import os
import json
import sqlite3
import time
from pathlib import Path
from datetime import datetime

import chromadb

from revit_api_docs.embedder.providers import create_embedding


def _embed_with_retry(embedder, texts: list[str], retries: int = 3, backoff: float = 2.0):
    """对 embedder.embed_texts 做 3 次退避重试，抵御 OpenRouter/上游瞬时错误。"""
    last_err: Exception | None = None
    for attempt in range(retries):
        try:
            return embedder.embed_texts(texts)
        except Exception as e:  # noqa: BLE001 — 网络/上游错误统一退避重试
            last_err = e
            if attempt < retries - 1:
                time.sleep(backoff * (attempt + 1))
    raise last_err  # type: ignore[misc]


def _write_meta(output_dir: str, config: dict, record_count: int, source: str,
                document_template: str | None = None):
    """写入 meta.json，记录 embedding 模型信息（API 库另记嵌入文本模板）"""
    provider = config["embedding"]["provider"]
    model_config = config["embedding"]["models"][provider]

    meta = {
        "revit_version": config.get("revit_version", "unknown"),
        "embedding_provider": provider,
        "embedding_model": model_config.get("model", "unknown"),
        "embedding_dimension": model_config.get("dimension", 0),
        "created_at": datetime.now().isoformat(),
        "record_count": record_count,
        "source": source,
    }
    if document_template:
        meta["document_template"] = document_template

    meta_path = os.path.join(output_dir, "meta.json")
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)
    print(f"已写入 {meta_path}")


# API 记录的嵌入文本模板（metadata 不受影响，始终含 name、full_id、summary、info）：
#   FS（默认）：full_id : summary；无 full_id 时 name : summary；无 summary 时 name - info；
#              再无则把可用字段用 " - " 连起来
#   NS：name : summary；无 summary 时 name - info；再无则 name
#   N ：只有 name
DOCUMENT_TEMPLATES = ("FS", "NS", "N")


def api_document_text(row_id, name, full_id, summary, info, template: str = "FS") -> str:
    """一条 API 记录的嵌入文本（见 DOCUMENT_TEMPLATES）。"""
    if template == "N":
        text = name or ""
    elif template == "NS":
        if name and summary:
            text = f"{name} : {summary}"
        elif name and info:
            text = f"{name} - {info}"
        else:
            text = name or ""
    elif template == "FS":
        if full_id and summary:
            text = f"{full_id} : {summary}"
        elif name and summary:
            text = f"{name} : {summary}"
        elif name and info:
            text = f"{name} - {info}"
        else:
            # 最后回退：合并所有可用字段
            parts = [p for p in [full_id, name, summary, info] if p]
            text = " - ".join(parts) if parts else ""
    else:
        raise ValueError(f"unknown document_template {template!r}; one of {DOCUMENT_TEMPLATES}")
    # 空文本会导致 embedding 报错——回退为 name 或 id
    if not text.strip():
        text = name or str(row_id)
    return text


def embed_api_data(config: dict, api_db_path: str, chromadb_dir: str, batch_size: int = 50,
                   resume: bool = False, document_template: str = "FS"):
    """
    将 API 数据向量化并存入 ChromaDB（记录 id = SQLite 行 id）

    resume=False（默认）：删除旧 collection 后全量重建。
    resume=True：保留已有 collection，只补嵌入缺失的行 id（中断后续跑）。
    document_template：嵌入文本模板 FS（默认，行为不变）/ NS / N，见 DOCUMENT_TEMPLATES；
    写入 meta.json 的 document_template。
    """
    if document_template not in DOCUMENT_TEMPLATES:
        raise ValueError(f"unknown document_template {document_template!r}; one of {DOCUMENT_TEMPLATES}")
    os.makedirs(chromadb_dir, exist_ok=True)
    embedder = create_embedding(config)

    conn = sqlite3.connect(api_db_path)
    cursor = conn.cursor()
    cursor.execute(
        "SELECT id, name, full_id, summary, info "
        "FROM revit_api "
        "WHERE name IS NOT NULL"
    )
    rows = cursor.fetchall()
    conn.close()

    print(f"从 {api_db_path} 读取 {len(rows)} 条 API 数据")

    client = chromadb.PersistentClient(path=chromadb_dir)
    # 全量重建：先删除旧 collection，保证与 SQLite 完全对齐（resume 时保留，只补缺）
    if not resume:
        try:
            client.delete_collection("revit_api")
        except Exception:
            pass
    collection = client.get_or_create_collection(
        name="revit_api",
        metadata={"description": "Revit API documentation embeddings"}
    )

    # 断点续传：已存在的 id 直接跳过（upsert 幂等，重复运行安全）
    existing = set(collection.get(include=[])["ids"])

    for i in range(0, len(rows), batch_size):
        batch = [row for row in rows[i:i + batch_size] if str(row[0]) not in existing]
        if not batch:
            continue
        ids = [str(row[0]) for row in batch]

        texts = []
        metadatas = []
        for row in batch:
            _id, name, full_id, summary, info = row
            texts.append(api_document_text(_id, name, full_id, summary, info, document_template))
            metadatas.append(
                {
                    "name": name or "",
                    "full_id": full_id or "",
                    "summary": (summary or "")[:500],
                    "info": (info or "")[:500],
                }
            )

        embeddings = _embed_with_retry(embedder, texts)

        collection.upsert(
            ids=ids,
            documents=texts,
            embeddings=embeddings,
            metadatas=metadatas,
        )

        if (i // batch_size) % 10 == 0:
            print(f"  API embedding 进度: {i + len(batch)}/{len(rows)}")

    _write_meta(chromadb_dir, config, len(rows), "RevitAPI CHM", document_template=document_template)
    print(f"API 向量化完成，共 {len(rows)} 条，存入 {chromadb_dir}")


def embed_code_data(config: dict, sdk_db_path: str, chromadb_dir: str, batch_size: int = 20):
    """
    将 SDK 代码数据向量化并存入 ChromaDB。

    读取新版 sdk_info 表（project, summary, content, mentioned_apis），
    仅使用 summary 字段作为 embedding 文本，大幅节省 token。
    兼容旧版 revit_sdk 表（fallback）。
    """
    os.makedirs(chromadb_dir, exist_ok=True)
    embedder = create_embedding(config)

    conn = sqlite3.connect(sdk_db_path)
    cursor = conn.cursor()

    # Prefer new sdk_info table; fall back to legacy revit_sdk
    try:
        cursor.execute("""
            SELECT id, project, summary, content, mentioned_apis
            FROM sdk_info
            WHERE summary IS NOT NULL AND summary != ''
        """)
        rows = cursor.fetchall()
        use_new_schema = True
    except sqlite3.OperationalError:
        try:
            cursor.execute("""
                SELECT id, project, filename, code, clean_code, description,
                       project_summary, file_purpose, use_case_category
                FROM revit_sdk
                WHERE code IS NOT NULL AND code != ''
            """)
            rows = cursor.fetchall()
            use_new_schema = False
        except sqlite3.OperationalError:
            rows = []
            use_new_schema = False

    conn.close()

    schema_label = "sdk_info (summary-only)" if use_new_schema else "revit_sdk (legacy)"
    print(f"从 {sdk_db_path} 读取 {len(rows)} 条 SDK 数据 [{schema_label}]")

    client = chromadb.PersistentClient(path=chromadb_dir)
    # 全量重建：先删除旧 collection，保证与 SQLite 完全对齐
    try:
        client.delete_collection("revit_sdk")
    except Exception:
        pass
    collection = client.get_or_create_collection(
        name="revit_sdk",
        metadata={"description": "Revit SDK sample code embeddings"}
    )

    # 断点续传：已存在的 id 直接跳过（upsert 幂等，重复运行安全）
    existing = set(collection.get(include=[])["ids"])

    for i in range(0, len(rows), batch_size):
        batch = [row for row in rows[i:i + batch_size] if str(row[0]) not in existing]
        if not batch:
            continue
        ids = [str(row[0]) for row in batch]

        texts = []
        metadatas = []

        for row in batch:
            if use_new_schema:
                # row: id, project, summary, content, mentioned_apis
                project = row[1] or ""
                summary = row[2] or ""
                mentioned_apis = row[4] or ""
                # Embedding text = summary only (concise, semantic-rich, minimal tokens)
                text = summary if summary else (row[3] or "")[:500]
                # 空文本会导致 embedding 报错——回退为 project 或 id
                if not text.strip():
                    text = project or str(row[0])
                texts.append(text)
                metadatas.append({
                    "project": project,
                    "mentioned_apis": mentioned_apis,
                })
            else:
                # Legacy schema fallback
                # row: id, project, filename, code, clean_code, description, project_summary, file_purpose, use_case_category
                code = row[4] if len(row) > 4 and row[4] else (row[3] or "")
                desc = row[5] if len(row) > 5 else ""
                proj_summary = row[6] if len(row) > 6 else ""
                file_purpose = row[7] if len(row) > 7 else ""
                prefix = " | ".join(filter(None, [proj_summary or "", file_purpose or "", desc or ""]))
                text = f"{prefix}\n{code[:800]}" if prefix else code[:1000]
                if not text.strip():
                    text = (row[1] or "") or str(row[0])
                texts.append(text)
                metadatas.append({"project": row[1] or "", "filename": row[2] or ""})

        embeddings = _embed_with_retry(embedder, texts)

        collection.upsert(
            ids=ids,
            documents=texts,
            embeddings=embeddings,
            metadatas=metadatas,
        )

        if (i // batch_size) % 10 == 0:
            print(f"  Code embedding 进度: {i + len(batch)}/{len(rows)}")

    _write_meta(chromadb_dir, config, len(rows), "RevitSDK Samples")
    print(f"Code 向量化完成，共 {len(rows)} 条，存入 {chromadb_dir}")
