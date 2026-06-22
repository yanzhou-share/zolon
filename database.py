import sqlite3
import os
from datetime import datetime, timedelta
from typing import Optional

DB_PATH = os.path.join(os.path.dirname(__file__), "saas.db")


def get_conn():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    conn = get_conn()
    c = conn.cursor()

    c.execute("""CREATE TABLE IF NOT EXISTS usage_logs (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        api_key TEXT NOT NULL,
        merchant_name TEXT DEFAULT '',
        tokens_in INTEGER DEFAULT 0,
        tokens_out INTEGER DEFAULT 0,
        latency_ms REAL DEFAULT 0,
        status TEXT DEFAULT 'normal',
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    )""")

    c.execute("""CREATE TABLE IF NOT EXISTS billing_config (
        api_key TEXT PRIMARY KEY,
        price_per_1k_tokens REAL DEFAULT 0.01,
        monthly_free_quota INTEGER DEFAULT 10000,
        billing_cycle TEXT DEFAULT 'monthly'
    )""")

    c.execute("""CREATE INDEX IF NOT EXISTS idx_usage_api_key ON usage_logs(api_key)""")
    c.execute("""CREATE INDEX IF NOT EXISTS idx_usage_created ON usage_logs(created_at)""")

    conn.commit()
    conn.close()


def log_usage(api_key: str, merchant_name: str, tokens_in: int, tokens_out: int, latency_ms: float, status: str = "normal"):
    conn = get_conn()
    c = conn.cursor()
    c.execute(
        "INSERT INTO usage_logs (api_key, merchant_name, tokens_in, tokens_out, latency_ms, status) VALUES (?, ?, ?, ?, ?, ?)",
        (api_key, merchant_name, tokens_in, tokens_out, latency_ms, status)
    )
    conn.commit()
    conn.close()


def get_dashboard_stats(days: int = 30) -> dict:
    conn = get_conn()
    c = conn.cursor()

    since = (datetime.now() - timedelta(days=days)).isoformat()

    c.execute("SELECT COUNT(*) as total FROM usage_logs WHERE created_at >= ?", (since,))
    total_requests = c.fetchone()["total"]

    c.execute("SELECT COALESCE(SUM(tokens_in + tokens_out), 0) as total FROM usage_logs WHERE created_at >= ?", (since,))
    total_tokens = c.fetchone()["total"]

    c.execute("SELECT COUNT(DISTINCT api_key) as total FROM usage_logs WHERE created_at >= ?", (since,))
    active_merchants = c.fetchone()["total"]

    c.execute("SELECT COALESCE(AVG(latency_ms), 0) as avg FROM usage_logs WHERE created_at >= ?", (since,))
    avg_latency = c.fetchone()["avg"]

    c.execute("""SELECT DATE(created_at) as date, COUNT(*) as requests, COALESCE(SUM(tokens_in + tokens_out), 0) as tokens
        FROM usage_logs WHERE created_at >= ? GROUP BY DATE(created_at) ORDER BY date""", (since,))
    daily_stats = [dict(row) for row in c.fetchall()]

    c.execute("""SELECT api_key, merchant_name, COUNT(*) as requests, COALESCE(SUM(tokens_in + tokens_out), 0) as tokens
        FROM usage_logs WHERE created_at >= ? GROUP BY api_key ORDER BY requests DESC""", (since,))
    merchant_ranking = [dict(row) for row in c.fetchall()]

    import json
    merchants_file = os.path.join(os.path.dirname(__file__), "merchants.json")
    if os.path.exists(merchants_file):
        with open(merchants_file, "r", encoding="utf-8") as f:
            merchants_data = json.load(f)
        for key, info in merchants_data.get("merchants", {}).items():
            if not any(r["api_key"] == key for r in merchant_ranking):
                merchant_ranking.append({"api_key": key, "merchant_name": info.get("name", ""), "requests": 0, "tokens": 0})
    merchant_ranking.sort(key=lambda x: x["requests"], reverse=True)

    conn.close()

    return {
        "total_requests": total_requests,
        "total_tokens": total_tokens,
        "active_merchants": active_merchants,
        "total_merchants": len(merchant_ranking),
        "avg_latency_ms": round(avg_latency, 2),
        "daily_stats": daily_stats,
        "merchant_ranking": merchant_ranking,
    }


def get_merchant_stats(api_key: str, days: int = 30) -> dict:
    conn = get_conn()
    c = conn.cursor()

    since = (datetime.now() - timedelta(days=days)).isoformat()

    c.execute("SELECT COUNT(*) as total FROM usage_logs WHERE api_key = ? AND created_at >= ?", (api_key, since))
    total_requests = c.fetchone()["total"]

    c.execute("SELECT COALESCE(SUM(tokens_in + tokens_out), 0) as total FROM usage_logs WHERE api_key = ? AND created_at >= ?", (api_key, since))
    total_tokens = c.fetchone()["total"]

    c.execute("SELECT COALESCE(AVG(latency_ms), 0) as avg FROM usage_logs WHERE api_key = ? AND created_at >= ?", (api_key, since))
    avg_latency = c.fetchone()["avg"]

    c.execute("""SELECT DATE(created_at) as date, COUNT(*) as requests, COALESCE(SUM(tokens_in + tokens_out), 0) as tokens
        FROM usage_logs WHERE api_key = ? AND created_at >= ? GROUP BY DATE(created_at) ORDER BY date""", (api_key, since))
    daily_stats = [dict(row) for row in c.fetchall()]

    c.execute("SELECT COALESCE(SUM(tokens_in), 0) as total FROM usage_logs WHERE api_key = ? AND created_at >= ?", (api_key, since))
    total_tokens_in = c.fetchone()["total"]

    c.execute("SELECT COALESCE(SUM(tokens_out), 0) as total FROM usage_logs WHERE api_key = ? AND created_at >= ?", (api_key, since))
    total_tokens_out = c.fetchone()["total"]

    conn.close()

    return {
        "total_requests": total_requests,
        "total_tokens": total_tokens,
        "tokens_in": total_tokens_in,
        "tokens_out": total_tokens_out,
        "avg_latency_ms": round(avg_latency, 2),
        "daily_stats": daily_stats,
    }


def calculate_billing(api_key: str, month: Optional[str] = None) -> dict:
    if not month:
        month = datetime.now().strftime("%Y-%m")

    conn = get_conn()
    c = conn.cursor()

    c.execute("SELECT * FROM billing_config WHERE api_key = ?", (api_key,))
    config = c.fetchone()
    if not config:
        config = {"price_per_1k_tokens": 0.01, "monthly_free_quota": 10000}

    since = f"{month}-01"
    until = f"{month}-32"

    c.execute("SELECT COALESCE(SUM(tokens_in + tokens_out), 0) as total FROM usage_logs WHERE api_key = ? AND created_at >= ? AND created_at < ?",
              (api_key, since, until))
    total_tokens = c.fetchone()["total"]

    c.execute("SELECT COUNT(*) as total FROM usage_logs WHERE api_key = ? AND created_at >= ? AND created_at < ?",
              (api_key, since, until))
    total_requests = c.fetchone()["total"]

    conn.close()

    free_quota = config["monthly_free_quota"]
    price_per_1k = config["price_per_1k_tokens"]
    billable_tokens = max(0, total_tokens - free_quota)
    amount = round(billable_tokens / 1000 * price_per_1k, 4)

    return {
        "api_key": api_key,
        "month": month,
        "total_requests": total_requests,
        "total_tokens": total_tokens,
        "free_quota": free_quota,
        "billable_tokens": billable_tokens,
        "price_per_1k_tokens": price_per_1k,
        "amount": amount,
    }


init_db()
