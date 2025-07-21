"""
Celery application configuration for background task processing.
"""
from celery import Celery
from ..shared.config import config

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
    # Task routing
    task_routes={
        "app.workers.embedding_worker.*": {"queue": "embedding"},
        "app.workers.query_worker.*": {"queue": "query"},
        "app.workers.maintenance_worker.*": {"queue": "maintenance"},
    },
    
    # Worker configuration
    worker_prefetch_multiplier=1,  # Prevent worker from prefetching too many tasks
    task_acks_late=True,  # Acknowledge task only after completion
    worker_disable_rate_limits=False,
    
    # Task time limits
    task_soft_time_limit=config.RESOURCE_LIMITS["worker_timeout"],
    task_time_limit=config.RESOURCE_LIMITS["worker_timeout"] + 60,
    
    # Result backend settings
    result_expires=3600,  # Results expire after 1 hour
    result_persistent=True,
    
    # Serialization
    task_serializer="json",
    accept_content=["json"],
    result_serializer="json",
    
    # Timezone
    timezone="UTC",
    enable_utc=True,
    
    # Monitoring
    worker_send_task_events=True,
    task_send_sent_event=True,
    
    # Queue configuration
    task_default_queue="default",
    task_default_exchange="default",
    task_default_exchange_type="direct",
    task_default_routing_key="default",
    
    # Resource limits
    worker_max_tasks_per_child=100,  # Restart worker after 100 tasks to prevent memory leaks
    worker_max_memory_per_child=2048000,  # 2GB memory limit per worker
)