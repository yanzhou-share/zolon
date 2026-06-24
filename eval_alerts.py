"""
评估告警系统 - 检测指标异常并发送告警

功能：
1. 检测指标异常（忠实度低、幻觉高、延迟高）
2. 记录告警到数据库
3. 发送告警通知
"""

import sqlite3
import logging
import os
from datetime import datetime
from typing import Optional

logger = logging.getLogger("eval.alerts")

DB_PATH = os.path.join(os.path.dirname(__file__), "saas.db")


def get_conn():
    """获取数据库连接"""
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def check_and_alert(metrics: dict, api_key: str = None):
    """
    检查指标并发送告警

    参数：
        metrics: 当前指标
        api_key: 商户 API Key（可选）

    告警规则：
        - faithfulness < 0.5: 忠实度过低
        - hallucination > 0.3: 幻觉率过高
        - avg_latency_ms > 10000: 延迟过高
    """
    alerts = []

    # 检查 Faithfulness（忠实度）
    faithfulness = metrics.get("faithfulness", 1.0)
    if faithfulness < 0.5:
        alerts.append({
            "type": "faithfulness_low",
            "severity": "critical" if faithfulness < 0.3 else "warning",
            "message": f"忠实度过低: {faithfulness:.2f} (阈值: 0.5)",
            "value": faithfulness,
        })

    # 检查 Hallucination（幻觉率）
    hallucination = metrics.get("hallucination", 0.0)
    if hallucination > 0.3:
        alerts.append({
            "type": "hallucination_high",
            "severity": "critical" if hallucination > 0.5 else "warning",
            "message": f"幻觉率过高: {hallucination:.2f} (阈值: 0.3)",
            "value": hallucination,
        })

    # 检查延迟
    latency = metrics.get("avg_latency_ms", 0)
    if latency > 10000:
        alerts.append({
            "type": "latency_high",
            "severity": "warning" if latency < 15000 else "critical",
            "message": f"延迟过高: {latency:.0f}ms (阈值: 10000ms)",
            "value": latency,
        })

    # 记录告警
    for alert in alerts:
        save_alert(alert, api_key)
        logger.warning(f"评估告警: {alert['message']}")

    return alerts


def save_alert(alert: dict, api_key: str = None):
    """
    保存告警到数据库

    参数：
        alert: 告警信息
        api_key: 商户 API Key（可选）
    """
    conn = get_conn()
    c = conn.cursor()
    c.execute("""INSERT INTO eval_alerts (alert_type, severity, message, api_key)
        VALUES (?, ?, ?, ?)""",
        (alert["type"], alert["severity"], alert["message"], api_key)
    )
    conn.commit()
    conn.close()


def get_recent_alerts(days: int = 7, api_key: str = None, resolved: bool = None) -> list:
    """
    获取最近的告警

    参数：
        days: 最近几天
        api_key: 商户 API Key（可选）
        resolved: 是否已解决（可选）

    返回：
        告警列表
    """
    conn = get_conn()
    c = conn.cursor()
    since = (datetime.now() - __import__("datetime").timedelta(days=days)).isoformat()

    query = "SELECT * FROM eval_alerts WHERE created_at >= ?"
    params = [since]

    if api_key:
        query += " AND api_key = ?"
        params.append(api_key)

    if resolved is not None:
        query += " AND resolved = ?"
        params.append(1 if resolved else 0)

    query += " ORDER BY created_at DESC"

    c.execute(query, params)
    alerts = [dict(row) for row in c.fetchall()]
    conn.close()
    return alerts


def resolve_alert(alert_id: int):
    """标记告警为已解决"""
    conn = get_conn()
    c = conn.cursor()
    c.execute("UPDATE eval_alerts SET resolved = 1 WHERE id = ?", (alert_id,))
    conn.commit()
    conn.close()


def get_alert_stats(days: int = 7) -> dict:
    """
    获取告警统计

    返回：
        告警统计信息
    """
    conn = get_conn()
    c = conn.cursor()
    since = (datetime.now() - __import__("datetime").timedelta(days=days)).isoformat()

    c.execute("SELECT COUNT(*) as total FROM eval_alerts WHERE created_at >= ?", (since,))
    total = c.fetchone()["total"]

    c.execute("SELECT COUNT(*) as unresolved FROM eval_alerts WHERE created_at >= ? AND resolved = 0", (since,))
    unresolved = c.fetchone()["unresolved"]

    c.execute("""SELECT alert_type, COUNT(*) as count FROM eval_alerts
        WHERE created_at >= ? GROUP BY alert_type ORDER BY count DESC""", (since,))
    by_type = [dict(row) for row in c.fetchall()]

    c.execute("""SELECT severity, COUNT(*) as count FROM eval_alerts
        WHERE created_at >= ? GROUP BY severity ORDER BY count DESC""", (since,))
    by_severity = [dict(row) for row in c.fetchall()]

    conn.close()
    return {
        "total": total,
        "unresolved": unresolved,
        "by_type": by_type,
        "by_severity": by_severity,
    }
