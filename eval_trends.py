"""
评估趋势追踪 - 追踪指标变化趋势

功能：
1. 记录每日指标到 SQLite
2. 获取指标趋势数据
3. 检测指标下降
"""

import sqlite3
import os
from datetime import datetime, timedelta
from typing import Optional

DB_PATH = os.path.join(os.path.dirname(__file__), "saas.db")


def get_conn():
    """获取数据库连接"""
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_eval_tables():
    """初始化评估相关的数据库表"""
    conn = get_conn()
    c = conn.cursor()

    # 每日指标表
    c.execute("""CREATE TABLE IF NOT EXISTS daily_metrics (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        date TEXT NOT NULL,
        api_key TEXT,
        faithfulness REAL DEFAULT 0,
        hallucination REAL DEFAULT 0,
        relevancy REAL DEFAULT 0,
        avg_latency_ms REAL DEFAULT 0,
        total_requests INTEGER DEFAULT 0,
        total_tokens INTEGER DEFAULT 0,
        recall_at_k REAL DEFAULT 0,
        precision_at_k REAL DEFAULT 0,
        mrr REAL DEFAULT 0,
        hit_rate REAL DEFAULT 0,
        answer_relevancy REAL DEFAULT 0,
        context_relevancy REAL DEFAULT 0,
        correctness REAL DEFAULT 0,
        response_quality REAL DEFAULT 0,
        satisfaction REAL DEFAULT 0,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    )""")

    # 评估报告表
    c.execute("""CREATE TABLE IF NOT EXISTS eval_reports (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        report_id TEXT UNIQUE NOT NULL,
        report_type TEXT,
        api_key TEXT,
        metrics_json TEXT,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    )""")

    # 告警记录表
    c.execute("""CREATE TABLE IF NOT EXISTS eval_alerts (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        alert_type TEXT,
        severity TEXT,
        message TEXT,
        api_key TEXT,
        resolved BOOLEAN DEFAULT 0,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    )""")

    # 创建索引
    c.execute("CREATE INDEX IF NOT EXISTS idx_daily_metrics_date ON daily_metrics(date)")
    c.execute("CREATE INDEX IF NOT EXISTS idx_daily_metrics_api_key ON daily_metrics(api_key)")
    c.execute("CREATE INDEX IF NOT EXISTS idx_eval_alerts_created ON eval_alerts(created_at)")

    conn.commit()
    conn.close()


def record_daily_metrics(
    faithfulness: float,
    hallucination: float,
    relevancy: float,
    avg_latency_ms: float,
    total_requests: int,
    total_tokens: int,
    api_key: str = None,
    recall_at_k: float = 0,
    precision_at_k: float = 0,
    mrr: float = 0,
    hit_rate: float = 0,
    answer_relevancy: float = 0,
    context_relevancy: float = 0,
    correctness: float = 0,
    response_quality: float = 0,
    satisfaction: float = 0,
):
    """
    记录每日指标

    参数：
        faithfulness: 忠实度（0-1，越高越好）
        hallucination: 幻觉率（0-1，越低越好）
        relevancy: 相关性（0-1，越高越好）
        avg_latency_ms: 平均延迟（毫秒）
        total_requests: 总请求数
        total_tokens: 总 Token 数
        api_key: 商户 API Key（可选，用于商户级别统计）
        recall_at_k: 召回率
        precision_at_k: 精确率
        mrr: 平均倒数排名
        hit_rate: 命中率
        answer_relevancy: 回答相关性
        context_relevancy: 上下文相关性
        correctness: 正确性
        response_quality: 回答质量
        satisfaction: 满意度
    """
    conn = get_conn()
    c = conn.cursor()
    today = datetime.now().strftime("%Y-%m-%d")

    c.execute("""INSERT INTO daily_metrics
        (date, api_key, faithfulness, hallucination, relevancy, avg_latency_ms, total_requests, total_tokens,
         recall_at_k, precision_at_k, mrr, hit_rate, answer_relevancy, context_relevancy,
         correctness, response_quality, satisfaction)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
        (today, api_key, faithfulness, hallucination, relevancy, avg_latency_ms, total_requests, total_tokens,
         recall_at_k, precision_at_k, mrr, hit_rate, answer_relevancy, context_relevancy,
         correctness, response_quality, satisfaction)
    )
    conn.commit()
    conn.close()


def get_trend(days: int = 30, api_key: str = None) -> list:
    """
    获取指标趋势

    参数：
        days: 获取最近几天的数据
        api_key: 商户 API Key（可选）

    返回：
        每日指标列表
    """
    conn = get_conn()
    c = conn.cursor()
    since = (datetime.now() - timedelta(days=days)).strftime("%Y-%m-%d")

    if api_key:
        c.execute("""SELECT date, faithfulness, hallucination, relevancy, avg_latency_ms, total_requests, total_tokens,
            recall_at_k, precision_at_k, mrr, hit_rate, answer_relevancy, context_relevancy,
            correctness, response_quality, satisfaction
            FROM daily_metrics WHERE date >= ? AND api_key = ? ORDER BY date""",
            (since, api_key))
    else:
        c.execute("""SELECT date, AVG(faithfulness) as faithfulness, AVG(hallucination) as hallucination,
            AVG(relevancy) as relevancy, AVG(avg_latency_ms) as avg_latency_ms,
            SUM(total_requests) as total_requests, SUM(total_tokens) as total_tokens,
            AVG(recall_at_k) as recall_at_k, AVG(precision_at_k) as precision_at_k,
            AVG(mrr) as mrr, AVG(hit_rate) as hit_rate,
            AVG(answer_relevancy) as answer_relevancy, AVG(context_relevancy) as context_relevancy,
            AVG(correctness) as correctness, AVG(response_quality) as response_quality,
            AVG(satisfaction) as satisfaction
            FROM daily_metrics WHERE date >= ? GROUP BY date ORDER BY date""",
            (since,))

    rows = [dict(row) for row in c.fetchall()]
    conn.close()
    return rows


def detect_degradation(current: dict, baseline: dict, threshold: float = 0.1) -> list:
    """
    检测指标下降

    参数：
        current: 当前指标
        baseline: 基线指标
        threshold: 下降阈值（默认 10%）

    返回：
        告警列表
    """
    alerts = []

    # Faithfulness 下降（越高越好）
    if baseline.get("faithfulness", 1) > 0:
        change = (current.get("faithfulness", 0) - baseline["faithfulness"]) / baseline["faithfulness"]
        if change < -threshold:
            alerts.append({
                "type": "faithfulness_degraded",
                "severity": "warning",
                "message": f"忠实度下降 {abs(change)*100:.1f}%: {baseline['faithfulness']:.2f} → {current['faithfulness']:.2f}",
                "current": current.get("faithfulness"),
                "baseline": baseline["faithfulness"],
            })

    # Hallucination 上升（越低越好）
    if baseline.get("hallucination", 0) > 0:
        change = (current.get("hallucination", 0) - baseline["hallucination"]) / baseline["hallucination"]
        if change > threshold:
            alerts.append({
                "type": "hallucination_increased",
                "severity": "warning",
                "message": f"幻觉率上升 {change*100:.1f}%: {baseline['hallucination']:.2f} → {current['hallucination']:.2f}",
                "current": current.get("hallucination"),
                "baseline": baseline["hallucination"],
            })

    # 延迟增加
    if baseline.get("avg_latency_ms", 0) > 0:
        change = (current.get("avg_latency_ms", 0) - baseline["avg_latency_ms"]) / baseline["avg_latency_ms"]
        if change > threshold:
            alerts.append({
                "type": "latency_increased",
                "severity": "info",
                "message": f"延迟增加 {change*100:.1f}%: {baseline['avg_latency_ms']:.0f}ms → {current['avg_latency_ms']:.0f}ms",
                "current": current.get("avg_latency_ms"),
                "baseline": baseline["avg_latency_ms"],
            })

    return alerts


def get_baseline(days: int = 7, api_key: str = None) -> dict:
    """获取基线指标（最近 N 天的平均值）"""
    trend = get_trend(days, api_key)
    if not trend:
        return {}

    return {
        "faithfulness": sum(t.get("faithfulness", 0) for t in trend) / len(trend),
        "hallucination": sum(t.get("hallucination", 0) for t in trend) / len(trend),
        "relevancy": sum(t.get("relevancy", 0) for t in trend) / len(trend),
        "avg_latency_ms": sum(t.get("avg_latency_ms", 0) for t in trend) / len(trend),
    }


# 初始化数据库表
init_eval_tables()
