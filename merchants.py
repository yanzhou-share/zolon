"""
商户管理模块

负责 API Key 验证、域名白名单、限流等商户相关功能。
"""

import json
import os
import secrets
import time
from collections import defaultdict
from typing import Optional

from config import merchant

# 限流缓存: {api_key: [timestamp, ...]}
_request_counts = defaultdict(list)


def load_merchants() -> dict:
    """加载商户配置"""
    if os.path.exists(merchant.MERCHANTS_FILE):
        with open(merchant.MERCHANTS_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    return {"merchants": {}}


def save_merchants(data: dict):
    """保存商户配置"""
    with open(merchant.MERCHANTS_FILE, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def generate_api_key() -> str:
    """生成随机 API Key"""
    return "zk_" + secrets.token_hex(6)


def validate_api_key(api_key: str) -> Optional[dict]:
    """验证 API Key，返回商户信息"""
    merchants = load_merchants()
    merchant_info = merchants.get("merchants", {}).get(api_key)
    if merchant_info and merchant_info.get("status") == "active":
        return merchant_info
    return None


def validate_origin(api_key: str, origin: str, referer: str) -> bool:
    """验证请求来源域名是否在白名单中"""
    merchants = load_merchants()
    merchant_info = merchants.get("merchants", {}).get(api_key)
    if not merchant_info:
        return False
    allowed = merchant_info.get("allowed_origins", [])
    if not allowed:
        return True
    source = origin or referer or ""
    for allowed_origin in allowed:
        if source.startswith(allowed_origin):
            return True
    return False


def check_rate_limit(api_key: str, limit: int = 100) -> bool:
    """检查频率限制"""
    now = time.time()
    _request_counts[api_key] = [t for t in _request_counts[api_key] if now - t < 60]
    if len(_request_counts[api_key]) >= limit:
        return False
    _request_counts[api_key].append(now)
    return True
