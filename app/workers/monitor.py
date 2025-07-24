"""
Worker monitoring utilities for Celery workers.
"""
from ..shared.redis_client import redis_client
from ..shared.config import config
import time
import json
from typing import Dict, List, Any
from dataclasses import dataclass
from enum import Enum


class WorkerStatus(Enum):
    HEALTHY = "healthy"
    WARNING = "warning"
    ERROR = "error"
    OFFLINE = "offline"


@dataclass
class WorkerInfo:
    hostname: str
    pid: int
    status: WorkerStatus
    last_heartbeat: float
    memory_usage: int
    memory_percent: float
    cpu_percent: float
    uptime: float
    warnings: List[str]
    queue: str = "unknown"


class WorkerMonitor:
    """Monitor and manage Celery worker health."""
    
    def __init__(self):
        self.redis = redis_client
        self.heartbeat_timeout = 120  # 2 minutes
    
    def get_all_workers(self) -> Dict[str, WorkerInfo]:
        """Get information about all registered workers."""
        workers = {}
        
        # Get worker registry
        registry = self.redis.get_json("workers:registry") or {}
        
        for worker_name in registry.keys():
            worker_data = self.redis.get_json(f"worker:{worker_name}")
            if worker_data:
                # Determine worker status
                last_heartbeat = worker_data.get("last_heartbeat", 0)
                time_since_heartbeat = time.time() - last_heartbeat
                
                if time_since_heartbeat > self.heartbeat_timeout:
                    status = WorkerStatus.OFFLINE
                elif not worker_data.get("healthy", True):
                    status = WorkerStatus.ERROR
                elif worker_data.get("resource_check", {}).get("warnings", []):
                    status = WorkerStatus.WARNING
                else:
                    status = WorkerStatus.HEALTHY
                
                # Determine queue from worker name
                queue = "unknown"
                if "embedding" in worker_name:
                    queue = "embedding"
                elif "query" in worker_name:
                    queue = "query"
                elif "maintenance" in worker_name:
                    queue = "maintenance"
                
                workers[worker_name] = WorkerInfo(
                    hostname=worker_data.get("hostname", "unknown"),
                    pid=worker_data.get("pid", 0),
                    status=status,
                    last_heartbeat=last_heartbeat,
                    memory_usage=worker_data.get("memory_usage", 0),
                    memory_percent=worker_data.get("memory_percent", 0.0),
                    cpu_percent=worker_data.get("cpu_percent", 0.0),
                    uptime=worker_data.get("uptime", 0.0),
                    warnings=worker_data.get("resource_check", {}).get("warnings", []),
                    queue=queue
                )
        
        return workers
    
    def get_queue_stats(self) -> Dict[str, Dict[str, Any]]:
        """Get statistics for each queue."""
        from .celery_app import celery_app
        
        stats = {}
        queues = ["embedding", "query", "maintenance", "default"]
        
        for queue in queues:
            try:
                # Get queue length (approximate)
                queue_length = self.redis.client.llen(queue)
                
                # Get active tasks for this queue
                active_tasks = []
                inspect = celery_app.control.inspect()
                active = inspect.active()
                
                if active:
                    for worker, tasks in active.items():
                        for task in tasks:
                            if task.get("delivery_info", {}).get("routing_key") == queue:
                                active_tasks.append({
                                    "task_id": task["id"],
                                    "task_name": task["name"],
                                    "worker": worker,
                                    "time_start": task.get("time_start")
                                })
                
                stats[queue] = {
                    "queue_length": queue_length,
                    "active_tasks": len(active_tasks),
                    "active_task_details": active_tasks
                }
            except Exception as e:
                stats[queue] = {
                    "error": str(e),
                    "queue_length": 0,
                    "active_tasks": 0,
                    "active_task_details": []
                }
        
        return stats
    
    def get_system_health(self) -> Dict[str, Any]:
        """Get overall system health status."""
        health_data = self.redis.get_json("system:health") or {}
        workers = self.get_all_workers()
        queue_stats = self.get_queue_stats()
        
        # Count workers by status
        worker_counts = {status.value: 0 for status in WorkerStatus}
        for worker in workers.values():
            worker_counts[worker.status.value] += 1
        
        # Calculate total queue length
        total_queue_length = sum(
            stats.get("queue_length", 0) for stats in queue_stats.values()
        )
        
        # Determine overall system status
        if worker_counts[WorkerStatus.ERROR.value] > 0:
            system_status = "error"
        elif worker_counts[WorkerStatus.WARNING.value] > 0:
            system_status = "warning"
        elif worker_counts[WorkerStatus.OFFLINE.value] > 0:
            system_status = "degraded"
        else:
            system_status = "healthy"
        
        return {
            "system_status": system_status,
            "timestamp": time.time(),
            "workers": {
                "total": len(workers),
                "by_status": worker_counts,
                "details": {name: {
                    "status": worker.status.value,
                    "queue": worker.queue,
                    "memory_percent": worker.memory_percent,
                    "cpu_percent": worker.cpu_percent,
                    "warnings": worker.warnings
                } for name, worker in workers.items()}
            },
            "queues": {
                "total_length": total_queue_length,
                "by_queue": queue_stats
            },
            "redis_health": health_data.get("redis", False),
            "database_accessible": health_data.get("db_accessible", False)
        }
    
    def restart_unhealthy_workers(self) -> Dict[str, Any]:
        """Restart workers that are in error state (placeholder for future implementation)."""
        # This would require integration with process management
        # For now, just return information about unhealthy workers
        workers = self.get_all_workers()
        unhealthy_workers = [
            name for name, worker in workers.items() 
            if worker.status in [WorkerStatus.ERROR, WorkerStatus.OFFLINE]
        ]
        
        return {
            "unhealthy_workers": unhealthy_workers,
            "action": "manual_restart_required",
            "message": "Use ./stop_workers.sh && ./start_workers.sh to restart workers"
        }
    
    def cleanup_stale_worker_data(self):
        """Clean up data for workers that haven't reported in a while."""
        current_time = time.time()
        registry = self.redis.get_json("workers:registry") or {}
        
        stale_workers = []
        for worker_name, worker_info in registry.items():
            last_seen = worker_info.get("last_seen", 0)
            if current_time - last_seen > self.heartbeat_timeout * 2:  # 4 minutes
                stale_workers.append(worker_name)
        
        # Remove stale workers from registry
        for worker_name in stale_workers:
            del registry[worker_name]
            # Also remove worker data
            self.redis.client.delete(f"worker:{worker_name}")
        
        if registry != (self.redis.get_json("workers:registry") or {}):
            self.redis.set_json("workers:registry", registry, expire_seconds=300)
        
        return {"cleaned_workers": stale_workers}


# Global monitor instance
worker_monitor = WorkerMonitor()