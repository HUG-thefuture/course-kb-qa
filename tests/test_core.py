# -*- coding: utf-8 -*-
"""RAG 核心用例：切分 / 检索 / 拒答 / 问答引用。"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.core import TfEmbedder, VectorStore, answer, split_markdown

DOC = """# 高数讲义
## 极限的定义
极限是描述变量无限逼近过程的数学工具，ε-δ 语言给出严格定义。
## 导数
导数是函数增量的极限。"""


def make_store(tmp_path):
    store = VectorStore(TfEmbedder())
    store.add(split_markdown(DOC, "g.md", 300))
    return store


def test_split_by_heading():
    chunks = split_markdown("# A\n## s1\n内容一\n## s2\n内容二", "a.md", 300)
    assert [c.section for c in chunks][:3] == ["A", "s1", "s2"]


def test_search_hit_right_section(tmp_path):
    store = make_store(tmp_path)
    hits = store.search("ε-δ 语言", 3)
    assert hits and hits[0][0].section == "极限的定义"


def test_answer_has_sources(tmp_path):
    res = answer("极限的定义", make_store(tmp_path))
    assert res["refused"] is False and res["sources"] and res["sources"][0]["file"] == "g.md"


def test_refusal_on_offtopic(tmp_path):
    res = answer("学校食堂今天菜单", make_store(tmp_path))
    assert res["refused"] is True and res["sources"] == []


def test_demo_page_served():
    """产品入口：/demo 返回单文件问答演示台（HTML），无外部资源依赖。"""
    from fastapi.testclient import TestClient
    from app.main import app

    with TestClient(app) as c:
        r = c.get("/demo")
        assert r.status_code == 200 and "text/html" in r.headers["content-type"]
        assert "课程知识库问答" in r.text and "/api/ask" in r.text
