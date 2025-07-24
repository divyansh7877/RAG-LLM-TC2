"""
Resource management system for monitoring and controlling system resources.
"""
import psutil
import time
import logging
import threading
from typing import Dict, Any, Optional, List, Callable
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from contextlib import contextmanager
from .config import config
from .redis_client import redis_client

logger = logging.getLogger(__name__)


class ResourceStatus(Enum):
    """Resource status levels."""
    HEALTHY = "healthy"
    WARNING = "warning"
    CRITICAL = "critical"
    OVERLOADED = "overloaded"


class ResourceType(Enum):
    """Types of resources to monitor."""
    MEMORY = "memory"
    CPU = "cpu"
    DISK = "disk"
    QUEUE = "queue"
    WORKERS = "workers"


@dataclass
class ResourceMetrics:
    """Resource metrics data structure."""
    timestamp: datetime
    memory_usage: float  # Percentage
    memory_available: int  # Bytes
    cpu_usage: float  # Percentage
    disk_usage: float  # Percentage
    disk_available: int  # Bytes
    queue_lengths: Dict[str, int] = field(default_factory=dict)
    active_workers: int = 0
    active_tasks: int = 0
    load_average: List[float] = field(default_factory=list)


@dataclass
class ResourceLimits:
    """Resource limit configuration."""
    memory_warning_threshold: float = 75.0  # Percentage
    memory_critical_threshold: float = 85.0  # Percentage
    cpu_warning_threshold: float = 80.0  # Percentage
    cpu_critical_threshold: float = 90.0  # Percentage
    disk_warning_threshold: float = 80.0  # Percentage
    disk_critical_threshold: float = 90.0  # Percentage
    max_queue_length: int = 100
    max_concurrent_embeddings: int = 2
    max_concurrent_queries: int = 10
    worker_timeout: int = 300


class ResourceManager:
    """
    Comprehensive resource management system that monitors system resources,
    enforces limits, and provides dynamic scaling capabilities.
    """
    
    def __init__(self):
        """Initialize the resource manager."""
        self.limits = ResourceLimits(
            memory_warning_threshold=config.RESOURCE_LIMITS.get("memory_warning_threshold", 75.0),
            memory_critical_threshold=config.RESOURCE_LIMITS.get("memory_critical_threshold", 85.0),
            cpu_warning_threshold=config.RESOURCE_LIMITS.get("cpu_warning_threshold", 80.0),
            cpu_critical_threshold=config.RESOURCE_LIMITS.get("cpu_critical_threshold", 90.0),
            disk_warning_threshold=config.RESOURCE_LIMITS.get("disk_warning_threshold", 80.0),
            disk_critical_threshold=config.RESOURCE_LIMITS.get("disk_critical_threshold", 90.0),
            max_queue_length=config.RESOURCE_LIMITS.get("max_queue_size", 100),
            max_concurrent_embeddings=config.RESOURCE_LIMITS.get("max_concurrent_embeddings", 2),
            max_concurrent_queries=config.RESOURCE_LIMITS.get("max_concurrent_queries", 10),
            worker_timeout=config.RESOURCE_LIMITS.get("worker_timeout", 300)
        )
        
        self._monitoring_active = False
        self._monitor_thread = None
        self._lock = threading.RLock()
        self._callbacks: Dict[ResourceType, List[Callable]] = {
            resource_type: [] for resource_type in ResourceType
        }
        self._last_metrics: Optional[ResourceMetrics] = None
        self._metrics_history: List[ResourceMetrics] = []
        self._max_history_size = 1000  # Keep last 1000 metrics entries
        
        logger.info("ResourceManager initialized")
    
    def start_monitoring(self, interval: float = 30.0):
        """
        Start continuous resource monitoring.
        
        Args:
            interval: Monitoring interval in seconds
        """
        with self._lock:
            if self._monitoring_active:
                logger.warning("Resource monitoring is already active")
                return
            
            self._monitoring_active = True
            self._monitor_thread = threading.Thread(
                target=self._monitoring_loop,
                args=(interval,),
                daemon=True,
                name="ResourceMonitor"
            )
            self._monitor_thread.start()
            logger.info(f"Resource monitoring started with {interval}s interval")
    
    def stop_monitoring(self):
        """Stop continuous resource monitoring."""
        with self._lock:
            if not self._monitoring_active:
                return
            
            self._monitoring_active = False
            if self._monitor_thread and self._monitor_thread.is_alive():
                self._monitor_thread.join(timeout=5.0)
            
            logger.info("Resource monitoring stopped")
    
    def _monitoring_loop(self, interval: float):
        """Main monitoring loop."""
        while self._monitoring_active:
            try:
                metrics = self.collect_metrics()
                self._update_metrics_history(metrics)
                self._check_resource_limits(metrics)
                self._store_metrics_in_redis(metrics)
                
                # Trigger callbacks for resource changes
                self._trigger_callbacks(metrics)
                
            except Exception as e:
                logger.error(f"Error in monitoring loop: {e}")
            
            time.sleep(interval)
    
    def collect_metrics(self) -> ResourceMetrics:
        """
        Collect current system resource metrics.
        
        Returns:
            ResourceMetrics object with current system state
        """
        try:
            # Memory metrics
            memory = psutil.virtual_memory()
            memory_usage = memory.percent
            memory_available = memory.available
            
            # CPU metrics
            cpu_usage = psutil.cpu_percent(interval=1)
            
            # Disk metrics
            disk = psutil.disk_usage('/')
            disk_usage = (disk.used / disk.total) * 100
            disk_available = disk.free
            
            # Load average (Unix-like systems)
            try:
                load_average = list(psutil.getloadavg())
            except AttributeError:
                # Windows doesn't have load average
                load_average = [cpu_usage / 100.0] * 3
            
            # Queue metrics
            queue_lengths = self._get_queue_lengths()
            
            # Worker metrics
            active_workers = self._get_active_worker_count()
            active_tasks = self._get_active_task_count()
            
            metrics = ResourceMetrics(
                timestamp=datetime.now(),
                memory_usage=memory_usage,
                memory_available=memory_available,
                cpu_usage=cpu_usage,
                disk_usage=disk_usage,
                disk_available=disk_available,
                queue_lengths=queue_lengths,
                active_workers=active_workers,
                active_tasks=active_tasks,
                load_average=load_average
            )
            
            self._last_metrics = metrics
            return metrics
            
        except Exception as e:
            logger.error(f"Error collecting metrics: {e}")
            # Return empty metrics on error
            return ResourceMetrics(
                timestamp=datetime.now(),
                memory_usage=0.0,
                memory_available=0,
                cpu_usage=0.0,
                disk_usage=0.0,
                disk_available=0,
                queue_lengths={},
                active_workers=0,
                active_tasks=0,
                load_average=[0.0, 0.0, 0.0]
            )
    
    def _get_queue_lengths(self) -> Dict[str, int]:
        """Get current queue lengths from Celery."""
        try:
            from celery import current_app
            
            # Get queue lengths from Celery
            inspect = current_app.control.inspect()
            active_queues = inspect.active_queues()
            
            queue_lengths = {}
            if active_queues:
                for worker, queues in active_queues.items():
                    for queue_info in queues:
                        queue_name = queue_info.get('name', 'unknown')
                        # This is an approximation - Celery doesn't provide exact queue lengths easily
                        queue_lengths[queue_name] = queue_lengths.get(queue_name, 0) + 1
            
            return queue_lengths
            
        except Exception as e:
            logger.warning(f"Could not get queue lengths: {e}")
            return {}
    
    def _get_active_worker_count(self) -> int:
        """Get count of active workers."""
        try:
            # Count active workers from Redis
            worker_keys = redis_client.client.keys("worker:*")
            active_count = 0
            
            for key in worker_keys:
                worker_info = redis_client.get_json(key)
                if worker_info and worker_info.get("status") == "ready":
                    # Check if worker is still alive (heartbeat within last 2 minutes)
                    last_heartbeat = worker_info.get("last_heartbeat", 0)
                    if time.time() - last_heartbeat < 120:
                        active_count += 1
            
            return active_count
            
        except Exception as e:
            logger.warning(f"Could not get active worker count: {e}")
            return 0
    
    def _get_active_task_count(self) -> int:
        """Get count of currently running tasks."""
        try:
            # Count active tasks from Redis
            task_keys = redis_client.client.keys("task:*")
            active_count = 0
            
            for key in task_keys:
                task_info = redis_client.get_json(key)
                if task_info and task_info.get("status") == "running":
                    active_count += 1
            
            return active_count
            
        except Exception as e:
            logger.warning(f"Could not get active task count: {e}")
            return 0
    
    def _update_metrics_history(self, metrics: ResourceMetrics):
        """Update metrics history with size limit."""
        with self._lock:
            self._metrics_history.append(metrics)
            
            # Keep only the most recent metrics
            if len(self._metrics_history) > self._max_history_size:
                self._metrics_history = self._metrics_history[-self._max_history_size:]
    
    def _check_resource_limits(self, metrics: ResourceMetrics):
        """Check resource limits and log warnings/errors."""
        # Memory checks
        if metrics.memory_usage >= self.limits.memory_critical_threshold:
            logger.critical(f"Memory usage critical: {metrics.memory_usage:.1f}%")
        elif metrics.memory_usage >= self.limits.memory_warning_threshold:
            logger.warning(f"Memory usage high: {metrics.memory_usage:.1f}%")
        
        # CPU checks
        if metrics.cpu_usage >= self.limits.cpu_critical_threshold:
            logger.critical(f"CPU usage critical: {metrics.cpu_usage:.1f}%")
        elif metrics.cpu_usage >= self.limits.cpu_warning_threshold:
            logger.warning(f"CPU usage high: {metrics.cpu_usage:.1f}%")
        
        # Disk checks
        if metrics.disk_usage >= self.limits.disk_critical_threshold:
            logger.critical(f"Disk usage critical: {metrics.disk_usage:.1f}%")
        elif metrics.disk_usage >= self.limits.disk_warning_threshold:
            logger.warning(f"Disk usage high: {metrics.disk_usage:.1f}%")
        
        # Queue length checks
        for queue_name, length in metrics.queue_lengths.items():
            if length >= self.limits.max_queue_length:
                logger.warning(f"Queue '{queue_name}' length high: {length}")
    
    def _store_metrics_in_redis(self, metrics: ResourceMetrics):
        """Store metrics in Redis for monitoring and alerting."""
        try:
            metrics_data = {
                "timestamp": metrics.timestamp.isoformat(),
                "memory_usage": metrics.memory_usage,
                "memory_available": metrics.memory_available,
                "cpu_usage": metrics.cpu_usage,
                "disk_usage": metrics.disk_usage,
                "disk_available": metrics.disk_available,
                "queue_lengths": metrics.queue_lengths,
                "active_workers": metrics.active_workers,
                "active_tasks": metrics.active_tasks,
                "load_average": metrics.load_average
            }
            
            # Store current metrics
            redis_client.set_json("system:metrics:current", metrics_data, expire_seconds=300)
            
            # Store in time series (keep last 24 hours)
            timestamp_key = int(metrics.timestamp.timestamp())
            redis_client.set_json(
                f"system:metrics:history:{timestamp_key}",
                metrics_data,
                expire_seconds=86400  # 24 hours
            )
            
        except Exception as e:
            logger.error(f"Error storing metrics in Redis: {e}")
    
    def _trigger_callbacks(self, metrics: ResourceMetrics):
        """Trigger registered callbacks for resource changes."""
        try:
            # Determine resource status
            status = self.get_resource_status(metrics)
            
            # Trigger callbacks based on resource types
            for resource_type, callbacks in self._callbacks.items():
                for callback in callbacks:
                    try:
                        callback(resource_type, metrics, status)
                    except Exception as e:
                        logger.error(f"Error in resource callback: {e}")
                        
        except Exception as e:
            logger.error(f"Error triggering callbacks: {e}")
    
    def get_resource_status(self, metrics: Optional[ResourceMetrics] = None) -> ResourceStatus:
        """
        Get overall resource status.
        
        Args:
            metrics: Optional metrics to evaluate, uses last collected if None
            
        Returns:
            ResourceStatus indicating system health
        """
        if metrics is None:
            metrics = self._last_metrics
        
        if metrics is None:
            return ResourceStatus.HEALTHY
        
        # Check for critical conditions
        if (metrics.memory_usage >= self.limits.memory_critical_threshold or
            metrics.cpu_usage >= self.limits.cpu_critical_threshold or
            metrics.disk_usage >= self.limits.disk_critical_threshold):
            return ResourceStatus.CRITICAL
        
        # Check for overloaded conditions
        max_queue_length = max(metrics.queue_lengths.values()) if metrics.queue_lengths else 0
        if max_queue_length >= self.limits.max_queue_length:
            return ResourceStatus.OVERLOADED
        
        # Check for warning conditions
        if (metrics.memory_usage >= self.limits.memory_warning_threshold or
            metrics.cpu_usage >= self.limits.cpu_warning_threshold or
            metrics.disk_usage >= self.limits.disk_warning_threshold):
            return ResourceStatus.WARNING
        
        return ResourceStatus.HEALTHY
    
    def can_accept_task(self, task_type: str) -> bool:
        """
        Check if system can accept a new task of given type.
        
        Args:
            task_type: Type of task ('embedding' or 'query')
            
        Returns:
            True if task can be accepted, False otherwise
        """
        try:
            # Use last metrics if available, otherwise collect new ones
            metrics = self._last_metrics
            if metrics is None:
                metrics = self.collect_metrics()
            
            status = self.get_resource_status(metrics)
            
            # Don't accept new tasks if system is critical or overloaded
            if status in [ResourceStatus.CRITICAL, ResourceStatus.OVERLOADED]:
                logger.warning(f"Rejecting {task_type} task due to resource status: {status.value}")
                return False
            
            # Check specific task type limits
            if task_type == "embedding":
                current_embeddings = self._count_active_tasks("embedding")
                if current_embeddings >= self.limits.max_concurrent_embeddings:
                    logger.info(f"Rejecting embedding task: {current_embeddings}/{self.limits.max_concurrent_embeddings} active")
                    return False
            
            elif task_type == "query":
                current_queries = self._count_active_tasks("query")
                if current_queries >= self.limits.max_concurrent_queries:
                    logger.info(f"Rejecting query task: {current_queries}/{self.limits.max_concurrent_queries} active")
                    return False
            
            return True
            
        except Exception as e:
            logger.error(f"Error checking task acceptance: {e}")
            # Default to accepting tasks on error to avoid blocking
            return True
    
    def _count_active_tasks(self, task_type: str) -> int:
        """Count active tasks of a specific type."""
        try:
            task_keys = redis_client.client.keys("task:*")
            count = 0
            
            for key in task_keys:
                task_info = redis_client.get_json(key)
                if (task_info and 
                    task_info.get("status") == "running" and
                    task_type in task_info.get("task_name", "").lower()):
                    count += 1
            
            return count
            
        except Exception as e:
            logger.error(f"Error counting active {task_type} tasks: {e}")
            return 0
    
    @contextmanager
    def resource_guard(self, task_type: str, task_id: str):
        """
        Context manager for resource-guarded task execution.
        
        Args:
            task_type: Type of task being executed
            task_id: Unique identifier for the task
        """
        if not self.can_accept_task(task_type):
            raise ResourceError(f"Cannot accept {task_type} task due to resource constraints")
        
        start_time = time.time()
        try:
            logger.info(f"Starting resource-guarded {task_type} task: {task_id}")
            yield
            
        except Exception as e:
            logger.error(f"Error in resource-guarded task {task_id}: {e}")
            raise
            
        finally:
            execution_time = time.time() - start_time
            logger.info(f"Completed resource-guarded {task_type} task: {task_id} in {execution_time:.2f}s")
    
    def register_callback(self, resource_type: ResourceType, callback: Callable):
        """
        Register a callback for resource changes.
        
        Args:
            resource_type: Type of resource to monitor
            callback: Function to call when resource changes
        """
        with self._lock:
            self._callbacks[resource_type].append(callback)
            logger.info(f"Registered callback for {resource_type.value}")
    
    def unregister_callback(self, resource_type: ResourceType, callback: Callable):
        """
        Unregister a callback for resource changes.
        
        Args:
            resource_type: Type of resource to stop monitoring
            callback: Function to remove from callbacks
        """
        with self._lock:
            if callback in self._callbacks[resource_type]:
                self._callbacks[resource_type].remove(callback)
                logger.info(f"Unregistered callback for {resource_type.value}")
    
    def get_metrics_history(self, hours: int = 1) -> List[ResourceMetrics]:
        """
        Get metrics history for the specified time period.
        
        Args:
            hours: Number of hours of history to retrieve
            
        Returns:
            List of ResourceMetrics from the specified time period
        """
        cutoff_time = datetime.now() - timedelta(hours=hours)
        
        with self._lock:
            return [
                metrics for metrics in self._metrics_history
                if metrics.timestamp >= cutoff_time
            ]
    
    def get_current_metrics(self) -> Optional[ResourceMetrics]:
        """Get the most recent metrics."""
        return self._last_metrics
    
    def force_cleanup(self):
        """Force cleanup of resources and expired data."""
        try:
            logger.info("Starting forced resource cleanup")
            
            # Clean up expired sessions
            cleaned_sessions = redis_client.cleanup_expired_sessions()
            logger.info(f"Cleaned up {cleaned_sessions} expired sessions")
            
            # Clean up old task data
            task_keys = redis_client.client.keys("task:*")
            cleaned_tasks = 0
            current_time = time.time()
            
            for key in task_keys:
                try:
                    task_info = redis_client.get_json(key)
                    if task_info:
                        # Remove tasks older than 1 hour that are completed or failed
                        task_age = current_time - task_info.get("started_at", current_time)
                        if (task_age > 3600 and 
                            task_info.get("status") in ["completed", "failed", "cancelled"]):
                            redis_client.delete(key.replace("task:", ""))
                            cleaned_tasks += 1
                except Exception as e:
                    logger.warning(f"Error cleaning task {key}: {e}")
            
            logger.info(f"Cleaned up {cleaned_tasks} old task records")
            
            # Clean up old metrics
            metric_keys = redis_client.client.keys("system:metrics:history:*")
            cleaned_metrics = 0
            cutoff_time = current_time - 86400  # 24 hours ago
            
            for key in metric_keys:
                try:
                    timestamp = int(key.split(":")[-1])
                    if timestamp < cutoff_time:
                        redis_client.delete(key)
                        cleaned_metrics += 1
                except Exception as e:
                    logger.warning(f"Error cleaning metric {key}: {e}")
            
            logger.info(f"Cleaned up {cleaned_metrics} old metric records")
            logger.info("Forced resource cleanup completed")
            
        except Exception as e:
            logger.error(f"Error during forced cleanup: {e}")
    
    def get_health_status(self) -> Dict[str, Any]:
        """
        Get comprehensive health status of the resource management system.
        
        Returns:
            Dictionary containing health status information
        """
        try:
            metrics = self.collect_metrics()
            status = self.get_resource_status(metrics)
            
            health_info = {
                "status": status.value,
                "timestamp": datetime.now().isoformat(),
                "monitoring_active": self._monitoring_active,
                "metrics": {
                    "memory_usage": f"{metrics.memory_usage:.1f}%",
                    "cpu_usage": f"{metrics.cpu_usage:.1f}%",
                    "disk_usage": f"{metrics.disk_usage:.1f}%",
                    "active_workers": metrics.active_workers,
                    "active_tasks": metrics.active_tasks,
                    "queue_lengths": metrics.queue_lengths,
                    "load_average": metrics.load_average
                },
                "limits": {
                    "memory_warning": f"{self.limits.memory_warning_threshold}%",
                    "memory_critical": f"{self.limits.memory_critical_threshold}%",
                    "cpu_warning": f"{self.limits.cpu_warning_threshold}%",
                    "cpu_critical": f"{self.limits.cpu_critical_threshold}%",
                    "max_concurrent_embeddings": self.limits.max_concurrent_embeddings,
                    "max_concurrent_queries": self.limits.max_concurrent_queries,
                    "max_queue_length": self.limits.max_queue_length
                },
                "task_capacity": {
                    "can_accept_embedding": self.can_accept_task("embedding"),
                    "can_accept_query": self.can_accept_task("query"),
                    "active_embeddings": self._count_active_tasks("embedding"),
                    "active_queries": self._count_active_tasks("query")
                }
            }
            
            return health_info
            
        except Exception as e:
            logger.error(f"Error getting health status: {e}")
            return {
                "status": "error",
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }


class ResourceError(Exception):
    """Exception raised when resource constraints are violated."""
    pass


# Global resource manager instance
resource_manager = ResourceManager()