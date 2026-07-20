"""
评估调度器 - 自动化评估任务

功能：
1. 每日自动评估：运行测试数据集，生成评估报告
2. 每周综合评估：生成趋势报告，对比历史数据
3. 手动触发评估：支持 API 调用触发评估
"""

import asyncio
import json
import logging
import os
from datetime import datetime, timedelta

from eval.eval_pipeline import batch_evaluate, get_eval_stats

logger = logging.getLogger("eval.scheduler")

# 评估数据集路径
DATASET_PATH = os.path.join(os.path.dirname(__file__), "eval_dataset.json")

# 评估报告存储目录
REPORTS_DIR = os.path.join(os.path.dirname(__file__), "eval_metrics")
os.makedirs(REPORTS_DIR, exist_ok=True)


def load_dataset() -> list:
    """加载评估测试数据集"""
    if not os.path.exists(DATASET_PATH):
        logger.warning(f"评估数据集不存在: {DATASET_PATH}")
        return []
    with open(DATASET_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data.get("test_cases", [])


async def daily_eval_task(llm_client, knowledge_collection=None):
    """
    每日自动评估任务

    执行流程：
    1. 加载测试数据集
    2. 运行批量评估
    3. 保存评估报告
    4. 检查指标是否下降
    """
    logger.info("开始每日评估任务...")
    start_time = datetime.now()

    # 加载测试数据集
    test_cases = load_dataset()
    if not test_cases:
        logger.warning("测试数据集为空，跳过评估")
        return None

    # 运行批量评估
    report = await batch_evaluate(
        client=llm_client,
        test_cases=test_cases,
        knowledge_collection=knowledge_collection,
    )

    # 生成报告文件名
    report_date = start_time.strftime("%Y-%m-%d")
    report_path = os.path.join(REPORTS_DIR, f"daily_report_{report_date}.json")

    # 保存报告
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report.to_dict(), f, ensure_ascii=False, indent=2)

    duration = (datetime.now() - start_time).total_seconds()
    logger.info(f"每日评估完成: {report.total_queries} 条查询, 耗时 {duration:.1f}s")

    # 检查指标是否下降
    from eval.eval_alerts import check_and_alert
    check_and_alert({
        "faithfulness": report.avg_generation_score,
        "hallucination": 1.0 - report.avg_generation_score,  # 近似计算
        "avg_latency_ms": report.avg_latency_ms,
    })

    return report


async def weekly_eval_task(llm_client, knowledge_collection=None):
    """
    每周综合评估任务

    执行流程：
    1. 运行完整数据集评估
    2. 生成趋势报告
    3. 对比上周数据
    """
    logger.info("开始每周评估任务...")
    start_time = datetime.now()

    # 运行批量评估
    test_cases = load_dataset()
    if not test_cases:
        return None

    report = await batch_evaluate(
        client=llm_client,
        test_cases=test_cases,
        knowledge_collection=knowledge_collection,
    )

    # 生成周报
    report_date = start_time.strftime("%Y-W%W")
    report_path = os.path.join(REPORTS_DIR, f"weekly_report_{report_date}.json")

    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report.to_dict(), f, ensure_ascii=False, indent=2)

    # 对比上周数据
    last_week_path = os.path.join(REPORTS_DIR, f"weekly_report_{(start_time - timedelta(weeks=1)).strftime('%Y-W%W')}.json")
    comparison = None
    if os.path.exists(last_week_path):
        with open(last_week_path, "r", encoding="utf-8") as f:
            last_week = json.load(f)
        comparison = {
            "faithfulness_change": report.avg_generation_score - last_week.get("avg_generation_score", 0),
            "latency_change": report.avg_latency_ms - last_week.get("avg_latency_ms", 0),
        }

    logger.info(f"每周评估完成: {report.total_queries} 条查询")
    return {"report": report, "comparison": comparison}


def get_latest_report(report_type: str = "daily") -> dict:
    """获取最新的评估报告"""
    import glob
    pattern = os.path.join(REPORTS_DIR, f"{report_type}_report_*.json")
    files = sorted(glob.glob(pattern), reverse=True)
    if not files:
        return {}
    with open(files[0], "r", encoding="utf-8") as f:
        return json.load(f)


def get_report_history(report_type: str = "daily", days: int = 30) -> list:
    """获取历史评估报告"""
    import glob
    pattern = os.path.join(REPORTS_DIR, f"{report_type}_report_*.json")
    files = sorted(glob.glob(pattern), reverse=True)[:days]
    reports = []
    for f in files:
        with open(f, "r", encoding="utf-8") as fp:
            reports.append(json.load(fp))
    return reports
