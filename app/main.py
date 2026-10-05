# -*- coding: utf-8 -*-
"""FastAPI 接口：上传 / 检索 / 问答 / 知识库管理。端口 8040。Swagger /docs。"""
import threading
from pathlib import Path

from fastapi import FastAPI, HTTPException, Query, UploadFile
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field

from .core import VectorStore, TfEmbedder, answer, split_markdown

BASE = Path(__file__).resolve().parent.parent
DOCS = BASE / "data" / "docs"
INDEX = BASE / "data" / "index"

ALLOWED_EXTS = (".md", ".markdown", ".txt")   # 上传白名单与索引 glob 必须一致（2026-09 修复）
MAX_UPLOAD_BYTES = 5 * 1024 * 1024            # 5MB 上限，防止一次性读入打爆内存

# 2026-09 修复：upload 的"全量重建"是读-算-写三步，并发请求会互相覆盖甚至写坏
# 索引文件；用可重入锁把"重建 + 换引用"整体串行化（演示规模下重建 <1s，代价可忽略）。
_store_lock = threading.RLock()

app = FastAPI(title="课程资料知识库问答系统", version="1.0")
store = VectorStore(TfEmbedder())
store.load(INDEX)


class AskReq(BaseModel):
    query: str
    # top_k 无下界时：0 会让正常问题也拒答（空切片），负数触发负切片多返回 n-1 条 sources
    top_k: int = Field(3, ge=1, le=20)


def _safe_docs_path(name: str) -> Path:
    """净化文件名：只允许落在 DOCS 目录下的单段文件名（防 ../ 与绝对路径穿越）。

    2026-09 修复：原实现把 multipart 的 filename 直接拼进 DOCS/name，
    传 "../../evil.md" 可写任意目录。
    """
    safe = Path(name).name
    if not safe or safe != name or safe in (".", ".."):
        raise HTTPException(400, "非法文件名")
    target = DOCS / safe
    if target.parent.resolve() != DOCS.resolve():
        raise HTTPException(400, "非法文件名")
    return target


@app.get("/health")
def health():
    return {"status": "ok", "documents": len({c.file for c in store.chunks}), "chunks": len(store.chunks)}


_DEMO_PAGE = Path(__file__).parent / "static" / "index.html"


@app.get("/demo", response_class=HTMLResponse)
def demo_page():
    """单文件问答演示台：引用溯源 + 拒答可视化（产品入口——评审看产品，不看 curl）。"""
    return _DEMO_PAGE.read_text(encoding="utf-8")


@app.post("/api/upload")
def upload(file: UploadFile):
    """上传 Markdown/TXT 资料：切分 → 向量化 → 入库（同名覆盖）。"""
    name = file.filename or ""
    target = _safe_docs_path(name)
    if not name.lower().endswith(ALLOWED_EXTS):
        raise HTTPException(400, "仅支持 .md/.markdown/.txt 文本资料")
    data = file.file.read(MAX_UPLOAD_BYTES + 1)
    if len(data) > MAX_UPLOAD_BYTES:
        raise HTTPException(413, f"文件超过 {MAX_UPLOAD_BYTES // 1024 // 1024}MB 上限")
    try:
        text = data.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise HTTPException(400, "文件不是合法 UTF-8 文本，请先转存为 UTF-8") from exc
    chunks = split_markdown(text, target.name)
    if not chunks:
        raise HTTPException(422, "文件切分后为空（无有效内容），未入库")
    with _store_lock:
        DOCS.mkdir(parents=True, exist_ok=True)
        target.write_text(text, encoding="utf-8")
        rebuild_store()
    return {"message": "已入库", "file": target.name, "chunks": len([c for c in store.chunks if c.file == target.name])}


@app.get("/api/documents")
def documents():
    return sorted({c.file for c in store.chunks})


@app.delete("/api/documents/{name}")
def delete_document(name: str):
    with _store_lock:
        target = _safe_docs_path(name)   # 删除同样要防穿越
        if target.exists():
            target.unlink()
        before = len(store.chunks)
        rebuild_store()
        return {"message": "已删除", "removed_chunks": before - len(store.chunks)}


@app.get("/api/search")
def search(query: str, top_k: int = Query(3, ge=1, le=20)):
    hits = store.search(query, top_k)
    return {"query": query,
            "results": [{"file": c.file, "section": c.section, "score": round(s, 4),
                         "snippet": c.text[:120]} for c, s in hits]}


@app.post("/api/ask")
def ask(req: AskReq):
    return answer(req.query, store, req.top_k)


def rebuild_store():
    """全量重建索引（演示规模下 <1s；生产应增量）。同名文档覆盖后旧块自动清除。

    调用方需持有 _store_lock；收录范围与上传白名单共用 ALLOWED_EXTS。
    2026-09 二次审计修复：原来按扩展名逐个 glob（Linux 下大小写敏感），上传放行的
    "X.MD" 会落盘却永不入索引；且 build_index 漏 .markdown 导致"上传能进、重建即丢"。
    现统一为 iterdir + 小写后缀匹配，与上传白名单严格一致。
    """
    global store
    new_store = VectorStore(TfEmbedder())
    if DOCS.exists():
        files = sorted(f for f in DOCS.iterdir()
                       if f.is_file() and f.name.lower().endswith(ALLOWED_EXTS))
        for f in files:
            new_store.add(split_markdown(f.read_text(encoding="utf-8"), f.name))
    new_store.save(INDEX)
    store = new_store


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8040)
