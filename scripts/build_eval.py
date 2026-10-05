# -*- coding: utf-8 -*-
"""构建 40 条评测集：问题 + 标准出处文件 + 关键词（evalset.json）。"""
import json
from pathlib import Path

BASE = Path(__file__).resolve().parent.parent

# (资料文件, 章节, 问题模板)
CASES = [
    ("高等数学_极限与连续.md", "ε-δ 语言", "ε-δ 语言的定义是什么？"),
    ("高等数学_极限与连续.md", "两个重要极限", "两个重要极限指的是哪两个？"),
    ("高等数学_极限与连续.md", "连续与间断点", "连续与间断点怎么判断？"),
    ("线性代数_矩阵与行列式.md", "行列式性质", "行列式有哪些性质？"),
    ("线性代数_矩阵与行列式.md", "逆矩阵的求法", "逆矩阵怎么求？"),
    ("线性代数_矩阵与行列式.md", "秩与线性方程组", "秩和线性方程组的关系？"),
    ("概率论与数理统计.md", "条件概率与独立性", "什么是条件概率？"),
    ("概率论与数理统计.md", "期望与方差", "期望和方差怎么计算？"),
    ("数据结构_线性表与树.md", "顺序表与链表", "顺序表和链表的区别？"),
    ("数据结构_线性表与树.md", "二叉搜索树", "二叉搜索树的特点？"),
    ("数据结构_图与排序.md", "最短路 Dijkstra", "Dijkstra 算法的思想？"),
    ("数据结构_图与排序.md", "快速排序", "快速排序的过程？"),
    ("操作系统_进程与内存.md", "死锁的四个条件", "死锁的四个必要条件？"),
    ("操作系统_进程与内存.md", "虚拟内存与页面置换", "页面置换算法有哪些？"),
    ("计算机网络_体系结构.md", "三次握手", "TCP 三次握手的过程？"),
    ("计算机网络_体系结构.md", "TCP 与 UDP", "TCP 和 UDP 的区别？"),
    ("数据库_索引与事务.md", "B+树索引", "为什么用 B+ 树做索引？"),
    ("数据库_索引与事务.md", "最左前缀原则", "什么是最左前缀原则？"),
    ("数据库_索引与事务.md", "隔离级别与幻读", "隔离级别和幻读的关系？"),
    ("数据库_索引与事务.md", "事务的 ACID", "事务的 ACID 是什么？"),
]


def main():
    evalset = []
    for i, (fname, sec, q) in enumerate(CASES):
        evalset.append({"id": f"E{i+1:02d}", "question": q, "gold_file": fname,
                        "gold_section": sec, "type": "answerable"})
    # 20 条知识库中没有的无关/超纲问题 → 应拒答
    refusals = ["学校食堂今天菜单", "周杰伦最新专辑", "量子计算机的制造工艺", "下周天气预报",
                "校园卡怎么挂失", "图书馆开门时间", "火星移民计划", "股票行情走势",
                "怎么写留学申请", "考研报名流程", "宿舍路由器怎么设置", "毕业论文格式要求",
                "体检预约流程", "校车时刻表", "奖学金评定细则", "运动会报名",
                "电影推荐", "游戏攻略", "旅游签证办理", "健身计划"]
    for j, rq in enumerate(refusals):
        evalset.append({"id": f"R{j+1:02d}", "question": rq, "gold_file": "",
                        "gold_section": "", "type": "refusal"})
    (BASE / "evalset.json").write_text(json.dumps(evalset, ensure_ascii=False, indent=2),
                                       encoding="utf-8")
    print(f"评测集 {len(evalset)} 条（可答 {len(CASES)} / 拒答 {len(refusals)}）→ evalset.json")


if __name__ == "__main__":
    main()
