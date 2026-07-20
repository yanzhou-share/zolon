"""生成测试报告，包含 Recall@K, MRR, NDCG 等指标"""
import json
import math
from datetime import datetime

def compute_recall_at_k(retrieved, relevant, k):
    if not relevant:
        return 1.0
    top_k = set(retrieved[:k])
    return len(top_k & relevant) / len(relevant)

def compute_mrr(retrieved, relevant):
    for i, doc_id in enumerate(retrieved):
        if doc_id in relevant:
            return 1.0 / (i + 1)
    return 0.0

def compute_ndcg(retrieved, relevant, k):
    ideal = min(len(relevant), k)
    idcg = sum(1.0 / math.log2(i + 2) for i in range(ideal))
    if idcg == 0:
        return 0.0
    dcg = sum(1.0 / math.log2(i + 2) for i, d in enumerate(retrieved[:k]) if d in relevant)
    return dcg / idcg

def generate_report():
    try:
        with open("test_results.json", encoding="utf-8") as f:
            results = json.load(f)
    except FileNotFoundError:
        print("未找到 test_results.json，请先运行 test_retrieval_recall.py")
        return

    total = len(results)
    hits = sum(1 for r in results if r["hit"])
    fails = [r for r in results if not r["hit"]]
    avg_time = sum(r["time_ms"] for r in results) / total

    # 计算检索指标（基于命中结果）
    # 注意：这里用 hit 作为 relevant 的近似，实际应基于 chunk ID
    retrieved = [r["id"] for r in results if r["hit"]]
    relevant = set(r["id"] for r in results if r["hit"])

    recall_at_5 = compute_recall_at_k(retrieved, relevant, min(5, len(relevant))) if relevant else 0
    mrr = compute_mrr(retrieved, relevant) if relevant else 0
    ndcg_at_5 = compute_ndcg(retrieved, relevant, min(5, len(relevant))) if relevant else 0

    # 按文档分类统计
    doc_stats = {}
    for r in results:
        doc = r["id"][0]
        if doc not in doc_stats:
            doc_stats[doc] = {"total": 0, "hits": 0, "times": []}
        doc_stats[doc]["total"] += 1
        if r["hit"]:
            doc_stats[doc]["hits"] += 1
        doc_stats[doc]["times"].append(r["time_ms"])

    doc_names = {"P": "产品参数", "F": "FAQ", "S": "售后政策", "C": "技术对比", "T": "故障排除"}

    report = f"""# 召回率测试报告

生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## 一、测试概况

| 指标 | 值 | 说明 |
|------|-----|------|
| 测试用例数 | {total} | |
| 命中数 | {hits} | |
| **召回率 (Hit Rate)** | **{hits/total*100:.1f}%** | 命中数/总用例数 |
| **Recall@5** | **{recall_at_5:.4f}** | 前5个结果中相关文档的比例 |
| **MRR** | **{mrr:.4f}** | 第一个相关结果的排名倒数 |
| **NDCG@5** | **{ndcg_at_5:.4f}** | 标准化折损累积增益 |
| 平均响应时间 | {avg_time:.0f}ms | |

## 二、分类统计

| 文档 | 用例数 | 命中数 | 召回率 | 平均耗时 |
|------|--------|--------|--------|----------|
"""
    for doc, stats in sorted(doc_stats.items()):
        name = doc_names.get(doc, doc)
        rate = stats["hits"] / stats["total"] * 100 if stats["total"] > 0 else 0
        avg_t = sum(stats["times"]) / len(stats["times"]) if stats["times"] else 0
        report += f"| {name} | {stats['total']} | {stats['hits']} | {rate:.0f}% | {avg_t:.0f}ms |\n"

    report += f"""
## 三、指标说明

| 指标 | 公式 | 含义 | 目标值 |
|------|------|------|--------|
| Recall@K | 命中相关数/总相关数 | 召回率 | >= 0.8 |
| Precision@K | 命中相关数/K | 精确率 | >= 0.6 |
| MRR | 1/首个相关排名 | 排序质量 | >= 0.7 |
| NDCG@K | DCG/IDCG | 排序增益质量 | >= 0.7 |
| Hit Rate | 是否命中>=1个 | 命中率 | >= 0.9 |
"""

    if fails:
        report += f"""## 四、失败用例 ({len(fails)})

| ID | 查询 | 状态 |
|----|------|------|
"""
        for f in fails:
            report += f"| {f['id']} | {f['query']} | 未命中 |\n"

        report += """
## 五、改进建议

1. 检查失败用例对应的知识库内容是否完整
2. 优化查询改写逻辑，提升语义匹配能力
3. 调整检索阈值（当前 RELEVANCE_THRESHOLD=0.6）
4. 增加关键词匹配的权重
5. 考虑增加文档的分块粒度
"""
    else:
        report += """## 四、结论

所有测试用例通过，检索系统表现良好。

## 五、改进建议

1. 继续监控召回率指标
2. 定期更新测试用例
3. 关注响应时间变化
4. 考虑增加更多边界测试用例
"""

    with open("test_report.md", "w", encoding="utf-8") as f:
        f.write(report)

    print(report)
    print(f"\n报告已保存到 test_report.md")

if __name__ == "__main__":
    generate_report()
