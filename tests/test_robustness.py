# -*- coding: utf-8 -*-
"""持久化/切块/上传安全 的防御性测试（2026-09 审计修复回归）。

运行：PYTHONPATH=vendor python -m pytest tests/test_robustness.py -v
"""
import sys

import pytest
from fastapi import HTTPException

from app.core import TfEmbedder, VectorStore, split_markdown
from app.main import _safe_docs_path, DOCS

DOC = """## 节A
节A 内容，包含一段代码：
```python
# 这不是标题
x = 1
```
更多内容。

## 节B
节B 的内容。
"""


def test_code_fence_lines_are_not_headings():
    """代码围栏内以 # 开头的行不得被当作标题（审计 P2 90）。"""
    chunks = split_markdown(DOC, "t.md")
    sections = [c.section for c in chunks]
    assert "节A" in sections and "节B" in sections
    assert not any("这不是标题" in s for s in sections), f"围栏内注释被误判为标题: {sections}"
    assert not any("x = 1" in c.section for c in chunks)


def test_save_load_roundtrip_keeps_retrieval(tmp_path):
    """save → load 后检索结果必须一致（原子写 + 形状校验回归）。"""
    store = VectorStore(TfEmbedder())
    store.add(split_markdown(DOC, "g.md", 200))
    q = "节B 的内容"
    top_before = store.search(q, 1)[0][0].section

    store.save(tmp_path)
    store2 = VectorStore(TfEmbedder())
    store2.load(tmp_path)
    assert store2.embedder.idf is not None, "idf.npy 应随索引持久化"
    top_after = store2.search(q, 1)[0][0].section
    assert top_after == top_before
    assert len(store2.chunks) == len(store.chunks)
    assert store2.matrix.shape[0] == len(store2.chunks)


def test_load_rejects_inconsistent_index(tmp_path):
    """matrix 行数与 chunks 数不一致时必须拒载（旧行为会静默张冠李戴）。"""
    import numpy as np
    store = VectorStore(TfEmbedder())
    store.add(split_markdown(DOC, "g.md", 200))
    store.save(tmp_path)
    # 人为制造"新 meta + 旧 matrix"的错位
    (tmp_path / "matrix.npy").write_bytes((tmp_path / "matrix.npy").read_bytes()[:-8])
    store2 = VectorStore(TfEmbedder())
    with pytest.raises(ValueError):
        store2.load(tmp_path)


def test_safe_docs_path_blocks_traversal():
    """上传/删除文件名净化：../、绝对路径、子目录一律 400（审计 P0 92）。"""
    bad = ["../../evil.md", "/etc/passwd.md", "sub/dir/x.md", "..", "."]
    if sys.platform == "win32":
        # 反斜杠仅在 Windows 路径语义下是分隔符；POSIX 把它当作合法文件名字符，
        # 断言写死会在 Linux/容器内挂（DID NOT RAISE）
        bad.append("a\\b.md")
    for b in bad:
        with pytest.raises(HTTPException) as ei:
            _safe_docs_path(b)
        assert ei.value.status_code == 400, f"{b!r} 应被拒绝"
    ok = _safe_docs_path("合法资料.md")
    assert ok == DOCS / "合法资料.md"
