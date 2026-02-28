"""
项目配置文件 - 从环境变量加载 API Keys、数据库、Qdrant 等配置
"""
import os
from dotenv import load_dotenv

load_dotenv()

# API Keys（请勿将 .env 提交至 Git）
DASHSCOPE_API_KEY = os.getenv("DASHSCOPE_API_KEY", "")
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY", "")
KIMI_API_KEY = os.getenv("KIMI_API_KEY", "")

# PostgreSQL 配置
DATABASE_URL = os.getenv(
    "DATABASE_URL",
    "postgresql://user:password@localhost:5432/postgres?client_encoding=utf8"
)

# Qdrant 向量数据库配置
QDRANT_URL = os.getenv("QDRANT_URL", "http://localhost:6333")
QDRANT_COLLECTION = os.getenv("QDRANT_COLLECTION", "business_reports")

# 文件处理配置
TEMP_DIR = os.getenv("TEMP_DIR", "temp")
UPLOAD_DIR = os.getenv("UPLOAD_DIR", "uploads")
MAX_FILE_SIZE = int(os.getenv("MAX_FILE_SIZE", "10485760"))  # 10MB
