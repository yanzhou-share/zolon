"""
配置管理模块

集中管理所有可配置参数，支持环境变量和 .env 文件。
"""

import os
from dotenv import load_dotenv

load_dotenv()


class LLMConfig:
    """LLM 模型配置"""
    API_KEY: str = os.getenv("DEEPSEEK_API_KEY", "")
    MODEL: str = os.getenv("LLM_MODEL", "deepseek-chat")
    BASE_URL: str = os.getenv("LLM_BASE_URL", "https://api.deepseek.com")
    TEMPERATURE: float = float(os.getenv("LLM_TEMPERATURE", "0.3"))
    MAX_TOKENS: int = int(os.getenv("LLM_MAX_TOKENS", "800"))


class EmbeddingConfig:
    """Embedding 模型配置"""
    MODEL_NAME: str = os.getenv("EMBED_MODEL_NAME", "BAAI/bge-small-zh")
    CACHE_DIR: str = os.getenv("HF_HOME", os.path.join(os.path.dirname(__file__), "hf_cache"))


class ChromaConfig:
    """ChromaDB 配置"""
    PERSIST_DIR: str = os.getenv("CHROMA_DIR", os.path.join(os.path.dirname(__file__), "chroma_db"))
    DEFAULT_COLLECTION: str = "uploaded_knowledge"
    HNSW_SPACE: str = "cosine"


class UploadConfig:
    """文件上传配置"""
    UPLOAD_DIR: str = os.getenv("UPLOAD_DIR", os.path.join(os.path.dirname(__file__), "uploads"))
    FILE_SIZE_LIMIT: int = int(os.getenv("FILE_SIZE_LIMIT", str(10 * 1024 * 1024)))  # 10MB
    ALLOWED_FORMATS: list = ["txt", "docx", "md", "pdf"]


class ChunkConfig:
    """文本分块配置"""
    CHUNK_SIZE: int = int(os.getenv("CHUNK_SIZE", "500"))
    CHUNK_OVERLAP: int = int(os.getenv("CHUNK_OVERLAP", "50"))


class MerchantConfig:
    """商户配置"""
    MERCHANTS_FILE: str = os.getenv("MERCHANTS_FILE", os.path.join(os.path.dirname(__file__), "merchants.json"))
    ADMIN_SECRET: str = os.getenv("ADMIN_SECRET", "admin123")


class ChatConfig:
    """聊天配置"""
    MAX_HISTORY: int = int(os.getenv("MAX_HISTORY", "10"))
    CONTEXT_MERGE_WINDOW: float = float(os.getenv("CONTEXT_MERGE_WINDOW", "5.0"))
    GREETING_TEMPLATE: str = "我是您的AI智能助手{assistant_name}，随时为您服务！"
    CLOSING: str = "请问还有什么可以帮助您的吗？"


class SentimentConfig:
    """情感分析配置"""
    INSULT_KEYWORDS: list = [
        "傻逼", "操你", "他妈", "去死", "废物", "白痴", "智障",
        "脑残", "煞笔", "混蛋", "王八蛋", "fuck", "shit", "stupid",
        "idiot", "damn"
    ]
    ANGRY_KEYWORDS: list = ["生气", "愤怒", "投诉", "差评", "垃圾", "退货", "退款", "骗子", "骗人"]


class EvalConfig:
    """评估配置"""
    METRICS_DIR: str = os.getenv("EVAL_DIR", os.path.join(os.path.dirname(__file__), "eval_metrics"))
    DATASET_PATH: str = os.path.join(os.path.dirname(__file__), "eval_dataset.json")


class DashboardConfig:
    """仪表盘配置"""
    DB_PATH: str = os.getenv("SAAS_DB_PATH", os.path.join(os.path.dirname(__file__), "saas.db"))


# 快捷访问
llm = LLMConfig()
embed = EmbeddingConfig()
chroma = ChromaConfig()
upload = UploadConfig()
chunk = ChunkConfig()
merchant = MerchantConfig()
chat = ChatConfig()
sentiment = SentimentConfig()
eval_cfg = EvalConfig()
dashboard = DashboardConfig()
