"""
Celery worker for maintenance tasks.
"""
from .celery_app import celery_app
from ..shared.redis_client import redis_client
import os
import shutil
import time


@celery_app.task(name="cleanup_expired_sessions")
def cleanup_expired_sessions():
    """Periodic cleanup of expired sessions and temporary files."""
    try:
        # Clean up expired sessions
        cleaned_sessions = redis_client.cleanup_expired_sessions()
        
        # Clean up temporary upload files older than 1 hour
        temp_dir = "./temp_uploads"
        cleaned_files = 0
        
        if os.path.exists(temp_dir):
            current_time = time.time()
            for filename in os.listdir(temp_dir):
                file_path = os.path.join(temp_dir, filename)
                if os.path.isfile(file_path):
                    # Check if file is older than 1 hour
                    if current_time - os.path.getmtime(file_path) > 3600:
                        try:
                            os.remove(file_path)
                            cleaned_files += 1
                        except OSError:
                            pass
        
        return {
            "cleaned_sessions": cleaned_sessions,
            "cleaned_files": cleaned_files,
            "timestamp": current_time
        }
        
    except Exception as e:
        print(f"Maintenance task error: {e}")
        raise


@celery_app.task(name="system_health_check")
def system_health_check():
    """Perform system health checks."""
    try:
        health_status = {
            "redis": redis_client.health_check(),
            "temp_dir_exists": os.path.exists("./temp_uploads"),
            "db_accessible": os.path.exists("./multi_user_db.lance"),
            "timestamp": time.time()
        }
        
        # Store health status in Redis
        redis_client.set_json("system:health", health_status, expire_seconds=300)  # 5 minutes
        
        return health_status
        
    except Exception as e:
        print(f"Health check error: {e}")
        raise