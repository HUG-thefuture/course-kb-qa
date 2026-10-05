# -*- coding: utf-8 -*-
"""生成 8 篇模拟课程资料 Markdown（固定 seed），供知识库与评测使用。"""
import numpy as np
from pathlib import Path

RNG = np.random.default_rng(9)
BASE = Path(__file__).resolve().parent.parent / "data" / "docs"

DOCS = {
    "高等数学_极限与连续.md": ["极限的定义", "ε-δ 语言", "两个重要极限", "连续与间断点"],
    "线性代数_矩阵与行列式.md": ["矩阵的运算", "行列式性质", "逆矩阵的求法", "秩与线性方程组"],
    "概率论与数理统计.md": ["古典概型", "条件概率与独立性", "随机变量与分布", "期望与方差"],
    "数据结构_线性表与树.md": ["顺序表与链表", "栈和队列", "二叉树的遍历", "二叉搜索树"],
    "数据结构_图与排序.md": ["图的存储", "最短路 Dijkstra", "快速排序", "归并排序"],
    "操作系统_进程与内存.md": ["进程与线程", "进程调度算法", "死锁的四个条件", "虚拟内存与页面置换"],
    "计算机网络_体系结构.md": ["OSI 七层模型", "TCP 与 UDP", "三次握手", "HTTP 与 HTTPS"],
    "数据库_索引与事务.md": ["B+树索引", "最左前缀原则", "事务的 ACID", "隔离级别与幻读"],
}
FACTS = [
    "定义是本节的基础，考试中常以选择题形式考查概念的准确表述。",
    "本节给出两个典型例题，解题步骤必须先写明所用定理的前提条件。",
    "常见错误是把充分条件当作充要条件使用，需要通过反例加深理解。",
    "课后习题第 3 题与本节例题同构，建议先独立完成再对照解析。",
    "本知识点与后续章节的 {next} 存在直接依赖关系，建议同步复习。",
    "历年真题中本节平均占比约 8 分，重点掌握推导过程而非结论本身。",
]


def main():
    BASE.mkdir(parents=True, exist_ok=True)
    total_sections = 0
    for fname, sections in DOCS.items():
        lines = [f"# {fname.replace('.md', '').replace('_', '·')} 讲义", ""]
        for i, sec in enumerate(sections, 1):
            lines.append(f"## {i}. {sec}")
            lines.append("")
            for k in range(int(RNG.integers(2, 4))):
                nxt = sections[(i) % len(sections)]
                lines.append(FACTS[int(RNG.integers(len(FACTS)))].format(next=nxt))
                lines.append(f"补充说明：{sec} 的核心思想可以概括为「从特殊到一般再到特殊」"
                             f"的归纳演绎过程，这一过程在第 {i} 节与第 {min(i + 1, len(sections))} 节之间反复出现。")
                lines.append("")
        (BASE / fname).write_text("\n".join(lines), encoding="utf-8")
        total_sections += len(sections)
    print(f"生成 {len(DOCS)} 篇课程资料 / {total_sections} 节 → data/docs/")


if __name__ == "__main__":
    main()
