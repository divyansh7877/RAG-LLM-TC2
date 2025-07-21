"""
Configuration settings for the concurrent RAG system.
"""
import os
from typing import Dict, Any


class Config:
    """Application configuration."""
    
    # Redis configuration
    REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379/0")
    REDIS_SESSION_DB = os.getenv("REDIS_SESSION_DB", "1")
    
    # Celery configuration
    CELERY_BROKER_URL = REDIS_URL
    CELERY_RESULT_BACKEND = REDIS_URL
    
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
        "max_memory_per_worker": os.getenv("MAX_MEMORY_PER_WORKER", "2GB"),
        "max_queue_size": int(os.getenv("MAX_QUEUE_SIZE", "100")),
        "worker_timeout": int(os.getenv("WORKER_TIMEOUT", "300"))
    }
    
    # Rate limiting
    RATE_LIMITS = {
        "query_max_requests": int(os.getenv("QUERY_MAX_REQUESTS", "20")),
        "query_window_seconds": int(os.getenv("QUERY_WINDOW_SECONDS", "60")),
        "upload_max_requests": int(os.getenv("UPLOAD_MAX_REQUESTS", "5")),
        "upload_window_seconds": int(os.getenv("UPLOAD_WINDOW_SECONDS", "300"))
    }
    
    # File upload settings
    MAX_FILE_SIZE = int(os.getenv("MAX_FILE_SIZE", "50")) * 1024 * 1024  # 50MB default
    ALLOWED_EXTENSIONS = {".pdf"}
    TEMP_UPLOAD_DIR = os.getenv("TEMP_UPLOAD_DIR", "./temp_uploads")
    
    # Session settings
    SESSION_EXPIRE_HOURS = int(os.getenv("SESSION_EXPIRE_HOURS", "24"))
    
    # User management (prototype - move to database in production)
    USERS = {
        "assistant1": {"password": "password1", "groups": ["assistance", "common_rules"]},
        "assistant2": {"password": "password2", "groups": ["assistance"]},
        "guest": {"password": "password", "groups": ["common_rules"]},
    }


# Global config instance
config = Config()