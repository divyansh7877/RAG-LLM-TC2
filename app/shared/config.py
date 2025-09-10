"""
Configuration settings for the concurrent RAG system.
"""
import os
from typing import Dict, Any
import torch


class Config:
    """Application configuration."""
    
    # GPU/CUDA configuration
    HAS_CUDA = torch.cuda.is_available()
    
    # Redis configuration
    REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379/0")
    REDIS_SESSION_DB = os.getenv("REDIS_SESSION_DB", "1")
    
    # Celery configuration
    CELERY_BROKER_URL = REDIS_URL
    CELERY_RESULT_BACKEND = REDIS_URL
    CELERY_TASK_SOFT_TIME_LIMIT = int(os.getenv("CELERY_TASK_SOFT_TIME_LIMIT", "300"))  # seconds
    CELERY_TASK_TIME_LIMIT = int(os.getenv("CELERY_TASK_TIME_LIMIT", "900"))  # seconds
    
    # Security configuration
    SECRET_KEY = os.getenv("SECRET_KEY", "your-secret-key-change-in-production")
    ACCESS_TOKEN_EXPIRE_MINUTES = int(os.getenv("ACCESS_TOKEN_EXPIRE_MINUTES", "1440"))  # 24 hours
    
    # Database configuration
    LANCEDB_PATH = os.getenv("LANCEDB_PATH", "./multi_user_db.lance")
    EMBEDDING_MODEL_PATH = os.getenv("EMBEDDING_MODEL_PATH", "./models/gte-large-en-v1.5")
    
    # Resource limits
    RESOURCE_LIMITS = {
        "max_concurrent_embeddings": int(os.getenv("MAX_CONCURRENT_EMBEDDINGS", "2")),
        "max_concurrent_queries": int(os.getenv("MAX_CONCURRENT_QUERIES", "10")),
        "max_memory_per_worker": os.getenv("MAX_MEMORY_PER_WORKER", "8GB"),
        "max_queue_size": int(os.getenv("MAX_QUEUE_SIZE", "100")),
        "worker_timeout": int(os.getenv("WORKER_TIMEOUT", "6000"))
    }
    
    # Rate limiting
    RATE_LIMITS = {
        "query_max_requests": int(os.getenv("QUERY_MAX_REQUESTS", "20")),
        "query_window_seconds": int(os.getenv("QUERY_WINDOW_SECONDS", "60")),
        "upload_max_requests": int(os.getenv("UPLOAD_MAX_REQUESTS", "10")),
        "upload_window_seconds": int(os.getenv("UPLOAD_WINDOW_SECONDS", "300"))
    }
    
    # File upload settings
    MAX_FILE_SIZE = int(os.getenv("MAX_FILE_SIZE", "50")) * 1024 * 1024  # 50MB default
    ALLOWED_EXTENSIONS = {
        ".pdf",    # PDF documents
        ".docx",   # Word documents
        ".pptx",   # PowerPoint presentations
        ".xlsx",   # Excel spreadsheets
        ".xls",    # Excel spreadsheets (legacy)
        ".html",   # HTML documents
        ".md",     # Markdown files
        ".csv"     # CSV files
    }
    TEMP_UPLOAD_DIR = os.getenv("TEMP_UPLOAD_DIR", "./temp_uploads")
    
    # Session settings
    SESSION_EXPIRE_HOURS = int(os.getenv("SESSION_EXPIRE_HOURS", "24"))

    # Keycloak configuration
    KEYCLOAK_SERVER_URL = os.getenv("KEYCLOAK_SERVER_URL", "http://192.168.1.117:8080/")
    KEYCLOAK_CLIENT_ID = os.getenv("KEYCLOAK_CLIENT_ID", "fastapi-client")
    KEYCLOAK_REALM = os.getenv("KEYCLOAK_REALM", "rag_app")
    KEYCLOAK_ALGORITHM = os.getenv("KEYCLOAK_ALGORITHM", "RS256")


# Global config instance
config = Config()