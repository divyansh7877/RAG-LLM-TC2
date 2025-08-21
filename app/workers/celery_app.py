"""
Celery application configuration for background task processing.
"""
import multiprocessing

# Set the start method to 'spawn' for fork-safety BEFORE any other imports
# that might initialize multiprocessing. This is crucial for libraries like
# PyTorch (CUDA) and LanceDB that are not fork-safe.
try:
    multiprocessing.set_start_method("spawn", force=True)
except RuntimeError:
    # This will fail if the start method has already been set.
    # We can safely ignore this if it's already set to 'spawn'.
    if multiprocessing.get_start_method() != "spawn":
        # Re-raise the error if it's already been set to something else.
        raise

from celery import Celery
from celery.signals import worker_ready, worker_shutdown, task_prerun, task_postrun
from ..shared.config import config
from ..shared.redis_client import redis_client
import logging
import os
import psutil
import time

# Configure logging as early as possible
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Create Celery app
celery_app = Celery(
    "rag_workers",
    broker=config.CELERY_BROKER_URL,
    backend=config.CELERY_RESULT_BACKEND,
    include=[
        "app.workers.embedding_worker",
        "app.workers.query_worker",
        "app.workers.maintenance_worker"
    ]
)

# Celery configuration
celery_app.conf.update(
    # Task routing with priority queues
    task_routes={
        "app.workers.embedding_worker.*": {"queue": "embedding", "routing_key": "embedding"},
        "app.workers.query_worker.*": {"queue": "query", "routing_key": "query"},
        "app.workers.maintenance_worker.*": {"queue": "maintenance", "routing_key": "maintenance"},
        # Explicit task name routing for tasks with custom names
        "process_user_query": {"queue": "query", "routing_key": "query"},
        "process_document_embedding": {"queue": "embedding", "routing_key": "embedding"},
        "get_query_status": {"queue": "query", "routing_key": "query"},
        "cleanup_query_cache": {"queue": "maintenance", "routing_key": "maintenance"},
        "cleanup_expired_sessions": {"queue": "maintenance", "routing_key": "maintenance"},
        "system_health_check": {"queue": "maintenance", "routing_key": "maintenance"},
        "worker_health_report": {"queue": "maintenance", "routing_key": "maintenance"},
    },
    
    # Worker configuration for resource management
    worker_prefetch_multiplier=1,  # Prevent worker from prefetching too many tasks
    task_acks_late=True,  # Acknowledge task only after completion
    worker_disable_rate_limits=False,
    
    # Task time limits with graceful handling (configurable)
    task_soft_time_limit=config.CELERY_TASK_SOFT_TIME_LIMIT,
    task_time_limit=config.CELERY_TASK_TIME_LIMIT,
    task_reject_on_worker_lost=True,
    
    # Result backend settings
    result_expires=3600,  # Results expire after 1 hour
    result_persistent=True,
    result_compression="gzip",  # Compress results to save memory
    
    # Serialization
    task_serializer="json",
    accept_content=["json"],
    result_serializer="json",
    
    # Timezone
    timezone="UTC",
    enable_utc=True,
    
    # Monitoring and events
    worker_send_task_events=True,
    task_send_sent_event=True,
    task_track_started=True,
    
    # Queue configuration with priorities
    task_default_queue="default",
    task_default_exchange="default",
    task_default_exchange_type="direct",
    task_default_routing_key="default",
    
    # Advanced queue settings
    task_queue_max_priority=10,
    task_default_priority=5,
    worker_direct=True,  # Enable direct queue routing
    
    # Resource limits and worker lifecycle
    worker_max_tasks_per_child=10,  # Restart worker after 100 tasks to prevent memory leaks
    worker_max_memory_per_child=6048000,  # 2GB memory limit per worker
    worker_autoscaler="celery.worker.autoscale:Autoscaler",
    
    # Retry configuration
    task_default_retry_delay=60,  # 1 minute default retry delay
    task_max_retries=3,
    
    # Beat schedule for maintenance tasks
    beat_schedule={
        "cleanup-expired-sessions": {
            "task": "cleanup_expired_sessions",
            "schedule": 300.0,  # Every 5 minutes
            "options": {"queue": "maintenance"}
        },
        "cleanup-query-cache": {
            "task": "cleanup_query_cache",
            "schedule": 660.0,  # Every 10 minutes
            "options": {"queue": "maintenance"}
        },
        "system-health-check": {
            "task": "system_health_check", 
            "schedule": 150.0,  # Every 
            "options": {"queue": "maintenance"}
        },
        "worker-health-report": {
            "task": "worker_health_report",
            "schedule": 200.0,  # Every 200 seconds
            "options": {"queue": "maintenance"}
        }
    },
)


# Worker health monitoring signals
@worker_ready.connect
def worker_ready_handler(sender=None, **kwargs):
    """Handle worker ready event."""
    worker_name = sender.hostname
    logger.info(f"Worker {worker_name} is ready")
    
    # Register worker in Redis
    worker_info = {
        "hostname": worker_name,
        "pid": os.getpid(),
        "status": "ready",
        "started_at": time.time(),
        "last_heartbeat": time.time(),
        "memory_usage": psutil.Process().memory_info().rss,
        "cpu_percent": psutil.Process().cpu_percent()
    }
    redis_client.set_json(f"worker:{worker_name}", worker_info, expire_seconds=120)


@worker_shutdown.connect
def worker_shutdown_handler(sender=None, **kwargs):
    """Handle worker shutdown event."""
    worker_name = sender.hostname
    logger.info(f"Worker {worker_name} is shutting down")
    
    # Update worker status in Redis
    try:
        worker_key = f"worker:{worker_name}"
        worker_info = redis_client.get_json(worker_key)
        if worker_info:
            worker_info.update({
                "status": "shutdown",
                "shutdown_at": time.time()
            })
            redis_client.set_json(worker_key, worker_info, expire_seconds=300)
    except Exception as e:
        logger.error(f"Error updating worker shutdown status: {e}")


@task_prerun.connect
def task_prerun_handler(sender=None, task_id=None, task=None, args=None, kwargs=None, **kwds):
    """Handle task pre-run event for monitoring."""
    logger.info(f"Task {task.name} [{task_id}] starting")
    
    # Update task metrics
    try:
        task_info = {
            "task_id": task_id,
            "task_name": task.name,
            "status": "running",
            "started_at": time.time(),
            "worker_pid": os.getpid(),
            "memory_before": psutil.Process().memory_info().rss
        }
        redis_client.set_json(f"task:{task_id}", task_info, expire_seconds=3600)
    except Exception as e:
        logger.error(f"Error recording task start: {e}")


@task_postrun.connect
def task_postrun_handler(sender=None, task_id=None, task=None, args=None, kwargs=None, 
                        retval=None, state=None, **kwds):
    """Handle task post-run event for monitoring."""
    logger.info(f"Task {task.name} [{task_id}] completed with state: {state}")
    
    # Update task metrics
    try:
        task_key = f"task:{task_id}"
        task_info = redis_client.get_json(task_key) or {}
        task_info.update({
            "status": state.lower() if state else "unknown",
            "completed_at": time.time(),
            "memory_after": psutil.Process().memory_info().rss,
            "result_size": len(str(retval)) if retval else 0
        })
        
        # Calculate execution time if start time exists
        if "started_at" in task_info:
            task_info["execution_time"] = time.time() - task_info["started_at"]
        
        redis_client.set_json(task_key, task_info, expire_seconds=3600)
    except Exception as e:
        logger.error(f"Error recording task completion: {e}")


def get_worker_stats():
    """Get current worker statistics."""
    try:
        process = psutil.Process()
        return {
            "pid": os.getpid(),
            "memory_usage": process.memory_info().rss,
            "memory_percent": process.memory_percent(),
            "cpu_percent": process.cpu_percent(),
            "num_threads": process.num_threads(),
            "create_time": process.create_time(),
            "status": process.status()
        }
    except Exception as e:
        logger.error(f"Error getting worker stats: {e}")
        return {}


