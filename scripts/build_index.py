# -*- coding: utf-8 -*-
"""构建知识库索引：把 data/docs 下全部资料切分入库并持久化到 data/index/。"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.core import TfEmbedder, VectorStore, split_markdown

BASE = Path(__file__).resolve().parent.parent
DOCS = BASE / "data" / "docs"
INDEX = BASE / "data" / "index"


# 与 app/main.py 的 ALLOWED_EXTS 保持一致（逐个 glob 在 Linux 大小写敏感且漏 .markdown）
ALLOWED_EXTS = (".md", ".markdown", ".txt")


def main():
    store = VectorStore(TfEmbedder())
    files = sorted(f for f in DOCS.iterdir()
                   if f.is_file() and f.name.lower().endswith(ALLOWED_EXTS))
    for f in files:
        store.add(split_markdown(f.read_text(encoding="utf-8"), f.name))
    store.save(INDEX)
    print(f"索引构建完成：{len(files)} 篇资料 / {len(store.chunks)} 块 → data/index/")


if __name__ == "__main__":
    main()
