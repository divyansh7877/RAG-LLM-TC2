"""
Celery worker for maintenance tasks.
"""
from .celery_app import celery_app
from ..shared.redis_client import redis_client
from ..shared.config import config
import os
import shutil
import time
import psutil
import socket
import logging

logger = logging.getLogger(__name__)


def check_resource_limits():
    """Check if worker is approaching resource limits."""
    try:
        process = psutil.Process()
        memory_mb = process.memory_info().rss / 1024 / 1024
        cpu_percent = process.cpu_percent()
        
        # Parse memory limit (e.g., "2GB" -> 2048 MB)
        memory_limit_str = config.RESOURCE_LIMITS["max_memory_per_worker"]
        if memory_limit_str.endswith("GB"):
            memory_limit_mb = float(memory_limit_str[:-2]) * 1024
        elif memory_limit_str.endswith("MB"):
            memory_limit_mb = float(memory_limit_str[:-2])
        else:
            memory_limit_mb = 2048  # Default 2GB
        
        warnings = []
        if memory_mb > memory_limit_mb * 0.8:  # 80% threshold
            warnings.append(f"Memory usage high: {memory_mb:.1f}MB / {memory_limit_mb}MB")
        
        if cpu_percent > 80:  # 80% CPU threshold
            warnings.append(f"CPU usage high: {cpu_percent:.1f}%")
        
        return {
            "memory_mb": memory_mb,
            "memory_limit_mb": memory_limit_mb,
            "cpu_percent": cpu_percent,
            "warnings": warnings,
            "healthy": len(warnings) == 0
        }
    except Exception as e:
        logger.error(f"Error checking resource limits: {e}")
        return {"healthy": False, "error": str(e)}


@celery_app.task(name="cleanup_expired_sessions")
def cleanup_expired_sessions():
    """Periodic cleanup of expired sessions and temporary files."""
    try:
        # Clean up expired sessions
        cleaned_sessions = redis_client.cleanup_expired_sessions()
        
        # Clean up temporary upload files older than 1 hour
        temp_dir = "./temp_uploads"
        cleaned_files = 0
        current_time = time.time()
        
        if os.path.exists(temp_dir):
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


@celery_app.task(name="worker_health_report")
def worker_health_report():
    """Report worker health metrics to Redis."""
    try:
        # Get current process info
        process = psutil.Process()
        hostname = socket.gethostname()
        worker_name = f"{hostname}-{os.getpid()}"
        
        # Collect worker metrics
        worker_metrics = {
            "hostname": hostname,
            "pid": os.getpid(),
            "status": "healthy",
            "last_heartbeat": time.time(),
            "memory_usage": process.memory_info().rss,
            "memory_percent": process.memory_percent(),
            "cpu_percent": process.cpu_percent(),
            "num_threads": process.num_threads(),
            "create_time": process.create_time(),
            "uptime": time.time() - process.create_time()
        }
        
        # Check resource limits
        resource_check = check_resource_limits()
        worker_metrics.update({
            "resource_check": resource_check,
            "healthy": resource_check.get("healthy", True)
        })
        
        # Store worker metrics in Redis
        redis_client.set_json(f"worker:{worker_name}", worker_metrics, expire_seconds=120)
        
        # Also update global worker registry
        worker_registry = redis_client.get_json("workers:registry") or {}
        worker_registry[worker_name] = {
            "last_seen": time.time(),
            "status": "healthy" if worker_metrics["healthy"] else "warning"
        }
        redis_client.set_json("workers:registry", worker_registry, expire_seconds=300)
        
        return worker_metrics
        
    except Exception as e:
        print(f"Worker health report error: {e}")
        # Still try to report that we're alive but unhealthy
        try:
            hostname = socket.gethostname()
            worker_name = f"{hostname}-{os.getpid()}"
            error_metrics = {
                "hostname": hostname,
                "pid": os.getpid(),
                "status": "error",
                "last_heartbeat": time.time(),
                "error": str(e),
                "healthy": False
            }
            redis_client.set_json(f"worker:{worker_name}", error_metrics, expire_seconds=120)
        except:
            pass
        raise

@celery_app.task(name="cleanup_query_cache")
def cleanup_query_cache():
    """
    Periodically clean up the query engine cache to free up resources.
    """
    from ..shared.query_engine_factory import query_engine_factory
    
    try:
        query_engine_factory.cleanup()
        logger.info("Query engine cache cleanup successful.")
        return {"status": "SUCCESS"}
    except Exception as e:
        logger.error(f"Failed to cleanup query cache: {e}", exc_info=True)
        raise


@celery_app.task(name="cleanup_stuck_jobs")
def cleanup_stuck_jobs():
    """
    Automatically clean up stuck jobs that have been running too long.
    
    Jobs are considered stuck if:
    - Processing jobs running for more than 2 hours
    - Pending jobs waiting for more than 1 hour
    """
    from ..shared.job_manager import job_manager
    from ..shared.models import JobStatus
    from datetime import datetime, timedelta
    
    try:
        now = datetime.now()
        cleaned_jobs = 0
        
        logger.info("Starting automatic stuck job cleanup")
        
        # Clean up stuck processing jobs (running > 2 hours)
        processing_jobs = job_manager.get_jobs_by_status(JobStatus.PROCESSING)
        for job in processing_jobs:
            if job.started_at:
                duration = now - job.started_at
                if duration > timedelta(hours=2):
                    logger.warning(f"Cancelling stuck processing job {job.job_id} for user {job.user_id} (running for {duration})")
                    success = job_manager.cancel_job(
                        job.job_id, 
                        f"Automatically cancelled - stuck for {duration} (started at {job.started_at})"
                    )
                    if success:
                        cleaned_jobs += 1
        
        # Clean up old pending jobs (pending > 1 hour)
        pending_jobs = job_manager.get_jobs_by_status(JobStatus.PENDING)
        for job in pending_jobs:
            age = now - job.created_at
            if age > timedelta(hours=1):
                logger.warning(f"Cancelling old pending job {job.job_id} for user {job.user_id} (pending for {age})")
                success = job_manager.cancel_job(
                    job.job_id,
                    f"Automatically cancelled - pending for {age} (created at {job.created_at})"
                )
                if success:
                    cleaned_jobs += 1
        
        result = {
            "cleaned_jobs": cleaned_jobs,
            "timestamp": time.time(),
            "status": "SUCCESS"
        }
        
        if cleaned_jobs > 0:
            logger.info(f"Automatic stuck job cleanup completed: cleaned {cleaned_jobs} jobs")
        else:
            logger.debug("Automatic stuck job cleanup completed: no stuck jobs found")
        
        return result
        
    except Exception as e:
        logger.error(f"Error in automatic stuck job cleanup: {e}", exc_info=True)
        return {
            "status": "ERROR",
            "error": str(e),
            "timestamp": time.time()
        }
