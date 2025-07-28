"""
System monitoring and alerting for the concurrent RAG application.
"""
import time
import psutil
import threading
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional, Callable
from dataclasses import dataclass, asdict
from enum import Enum
import logging
from collections import deque, defaultdict

from .redis_client import redis_client
from .config import config
from .error_handling import error_handler, StructuredLogger, ErrorSeverity


class AlertLevel(Enum):
    """Alert severity levels."""
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


class MetricType(Enum):
    """Types of metrics to monitor."""
    COUNTER = "counter"
    GAUGE = "gauge"
    HISTOGRAM = "histogram"
    TIMER = "timer"


@dataclass
class SystemMetrics:
    """System resource metrics."""
    timestamp: float
    cpu_usage: float
    memory_usage: float
    memory_available: int
    disk_usage: float
    disk_available: int
    active_connections: int
    active_workers: int
    active_tasks: int
    queue_lengths: Dict[str, int]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        result = asdict(self)
        result['datetime'] = datetime.fromtimestamp(self.timestamp).isoformat()
        return result


@dataclass
class Alert:
    """System alert."""
    alert_id: str
    timestamp: float
    level: AlertLevel
    title: str
    message: str
    metric_name: str
    current_value: Any
    threshold_value: Any
    resolved: bool = False
    resolved_timestamp: Optional[float] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        result = asdict(self)
        result['level'] = self.level.value
        result['datetime'] = datetime.fromtimestamp(self.timestamp).isoformat()
        if self.resolved_timestamp:
            result['resolved_datetime'] = datetime.fromtimestamp(self.resolved_timestamp).isoformat()
        return result


class MetricCollector:
    """Collects and stores system metrics."""
    
    def __init__(self):
        self.logger = StructuredLogger(__name__)
        self.metrics_history = deque(maxlen=1000)  # Keep last 1000 metric snapshots
        self._lock = threading.Lock()
        
    def collect_system_metrics(self) -> SystemMetrics:
        """Collect current system metrics."""
        try:
            # CPU and memory metrics
            cpu_percent = psutil.cpu_percent(interval=1)
            memory = psutil.virtual_memory()
            disk = psutil.disk_usage('/')
            
            # Network connections (approximate)
            connections = len(psutil.net_connections())
            
            # Get queue lengths from Redis
            queue_lengths = self._get_queue_lengths()
            
            # Get worker/task counts from Celery
            active_workers, active_tasks = self._get_celery_stats()
            
            metrics = SystemMetrics(
                timestamp=time.time(),
                cpu_usage=cpu_percent,
                memory_usage=memory.percent,
                memory_available=memory.available,
                disk_usage=disk.percent if hasattr(disk, 'percent') else 0,
                disk_available=disk.free,
                active_connections=connections,
                active_workers=active_workers,
                active_tasks=active_tasks,
                queue_lengths=queue_lengths
            )
            
            # Store metrics
            with self._lock:
                self.metrics_history.append(metrics)
            
            # Store in Redis for persistence
            self._store_metrics(metrics)
            
            return metrics
            
        except Exception as e:
            self.logger.error(f"Failed to collect system metrics: {e}")
            # Return empty metrics on failure
            return SystemMetrics(
                timestamp=time.time(),
                cpu_usage=0.0,
                memory_usage=0.0,
                memory_available=0,
                disk_usage=0.0,
                disk_available=0,
                active_connections=0,
                active_workers=0,
                active_tasks=0,
                queue_lengths={}
            )
    
    def _get_queue_lengths(self) -> Dict[str, int]:
        """Get current queue lengths from Redis."""
        try:
            queue_lengths = {}
            
            # Check Celery queues
            from ..workers.celery_app import celery_app
            inspect = celery_app.control.inspect()
            
            # Get active tasks
            active = inspect.active()
            if active:
                for worker, tasks in active.items():
                    queue_lengths[f"active_{worker}"] = len(tasks)
            
            # Get reserved tasks
            reserved = inspect.reserved()
            if reserved:
                for worker, tasks in reserved.items():
                    queue_lengths[f"reserved_{worker}"] = len(tasks)
            
            return queue_lengths
            
        except Exception as e:
            self.logger.warning(f"Failed to get queue lengths: {e}")
            return {}
    
    def _get_celery_stats(self) -> tuple[int, int]:
        """Get Celery worker and task statistics."""
        try:
            from ..workers.celery_app import celery_app
            inspect = celery_app.control.inspect()
            
            # Get active workers
            stats = inspect.stats()
            active_workers = len(stats) if stats else 0
            
            # Get active tasks
            active = inspect.active()
            active_tasks = sum(len(tasks) for tasks in active.values()) if active else 0
            
            return active_workers, active_tasks
            
        except Exception as e:
            self.logger.warning(f"Failed to get Celery stats: {e}")
            return 0, 0
    
    def _store_metrics(self, metrics: SystemMetrics):
        """Store metrics in Redis for persistence."""
        try:
            # Store current metrics
            redis_client.set_json("metrics:current", metrics.to_dict(), expire_seconds=300)
            
            # Store historical metrics (hourly aggregation)
            hour_key = f"metrics:hourly:{time.strftime('%Y-%m-%d:%H', time.localtime(metrics.timestamp))}"
            hourly_data = redis_client.get_json(hour_key) or {
                "hour": time.strftime('%Y-%m-%d:%H', time.localtime(metrics.timestamp)),
                "samples": 0,
                "avg_cpu": 0.0,
                "avg_memory": 0.0,
                "avg_disk": 0.0,
                "max_cpu": 0.0,
                "max_memory": 0.0,
                "max_disk": 0.0,
                "avg_connections": 0,
                "avg_workers": 0,
                "avg_tasks": 0
            }
            
            # Update averages
            samples = hourly_data["samples"]
            hourly_data["avg_cpu"] = (hourly_data["avg_cpu"] * samples + metrics.cpu_usage) / (samples + 1)
            hourly_data["avg_memory"] = (hourly_data["avg_memory"] * samples + metrics.memory_usage) / (samples + 1)
            hourly_data["avg_disk"] = (hourly_data["avg_disk"] * samples + metrics.disk_usage) / (samples + 1)
            hourly_data["avg_connections"] = (hourly_data["avg_connections"] * samples + metrics.active_connections) / (samples + 1)
            hourly_data["avg_workers"] = (hourly_data["avg_workers"] * samples + metrics.active_workers) / (samples + 1)
            hourly_data["avg_tasks"] = (hourly_data["avg_tasks"] * samples + metrics.active_tasks) / (samples + 1)
            
            # Update maximums
            hourly_data["max_cpu"] = max(hourly_data["max_cpu"], metrics.cpu_usage)
            hourly_data["max_memory"] = max(hourly_data["max_memory"], metrics.memory_usage)
            hourly_data["max_disk"] = max(hourly_data["max_disk"], metrics.disk_usage)
            
            hourly_data["samples"] += 1
            
            redis_client.set_json(hour_key, hourly_data, expire_seconds=86400 * 7)  # Keep for 7 days
            
        except Exception as e:
            self.logger.warning(f"Failed to store metrics: {e}")
    
    def get_metrics_history(self, hours: int = 24) -> List[Dict[str, Any]]:
        """Get historical metrics for the specified number of hours."""
        try:
            history = []
            current_time = time.time()
            
            for i in range(hours):
                hour_timestamp = current_time - (i * 3600)
                hour_key = f"metrics:hourly:{time.strftime('%Y-%m-%d:%H', time.localtime(hour_timestamp))}"
                
                hourly_data = redis_client.get_json(hour_key)
                if hourly_data:
                    history.append(hourly_data)
            
            return list(reversed(history))  # Return in chronological order
            
        except Exception as e:
            self.logger.error(f"Failed to get metrics history: {e}")
            return []


class AlertManager:
    """Manages system alerts and notifications."""
    
    def __init__(self):
        self.logger = StructuredLogger(__name__)
        self.active_alerts: Dict[str, Alert] = {}
        self.alert_history = deque(maxlen=500)  # Keep last 500 alerts
        self._lock = threading.Lock()
        
        # Alert thresholds
        self.thresholds = {
            "cpu_usage": {"warning": 70.0, "critical": 90.0},
            "memory_usage": {"warning": 80.0, "critical": 95.0},
            "disk_usage": {"warning": 85.0, "critical": 95.0},
            "active_tasks": {"warning": 50, "critical": 100},
            "error_rate": {"warning": 10, "critical": 25},  # errors per minute
            "queue_length": {"warning": 20, "critical": 50}
        }
        
        # Alert callbacks
        self.alert_callbacks: List[Callable[[Alert], None]] = []
    
    def add_alert_callback(self, callback: Callable[[Alert], None]):
        """Add callback function to be called when alerts are triggered."""
        self.alert_callbacks.append(callback)
    
    def check_metrics_alerts(self, metrics: SystemMetrics):
        """Check metrics against thresholds and trigger alerts."""
        try:
            # Check CPU usage
            self._check_threshold_alert(
                "cpu_usage", metrics.cpu_usage, metrics.timestamp,
                "High CPU Usage", f"CPU usage is at {metrics.cpu_usage:.1f}%"
            )
            
            # Check memory usage
            self._check_threshold_alert(
                "memory_usage", metrics.memory_usage, metrics.timestamp,
                "High Memory Usage", f"Memory usage is at {metrics.memory_usage:.1f}%"
            )
            
            # Check disk usage
            self._check_threshold_alert(
                "disk_usage", metrics.disk_usage, metrics.timestamp,
                "High Disk Usage", f"Disk usage is at {metrics.disk_usage:.1f}%"
            )
            
            # Check active tasks
            self._check_threshold_alert(
                "active_tasks", metrics.active_tasks, metrics.timestamp,
                "High Task Load", f"Active tasks: {metrics.active_tasks}"
            )
            
            # Check queue lengths
            for queue_name, length in metrics.queue_lengths.items():
                self._check_threshold_alert(
                    f"queue_length_{queue_name}", length, metrics.timestamp,
                    f"High Queue Length: {queue_name}", f"Queue {queue_name} has {length} items"
                )
            
            # Check error rates
            self._check_error_rate_alerts(metrics.timestamp)
            
        except Exception as e:
            self.logger.error(f"Failed to check metric alerts: {e}")
    
    def _check_threshold_alert(self, metric_name: str, current_value: float, 
                             timestamp: float, title: str, message: str):
        """Check if metric exceeds thresholds and trigger alerts."""
        # For queue metrics, use the full metric name, otherwise use the base metric
        if metric_name.startswith("queue_length_"):
            lookup_key = "queue_length"
        else:
            # Use the full metric name for exact matching
            lookup_key = metric_name
        
        thresholds = self.thresholds.get(lookup_key, {})
        
        if not thresholds:
            return
        
        alert_level = None
        threshold_value = None
        
        # Check critical threshold first, then warning
        critical_threshold = thresholds.get("critical", float('inf'))
        warning_threshold = thresholds.get("warning", float('inf'))
        
        if current_value >= critical_threshold:
            alert_level = AlertLevel.CRITICAL
            threshold_value = critical_threshold
        elif current_value >= warning_threshold:
            alert_level = AlertLevel.WARNING
            threshold_value = warning_threshold
        
        if alert_level:
            self._trigger_alert(
                metric_name, alert_level, title, message,
                current_value, threshold_value, timestamp
            )
        else:
            # Check if we should resolve an existing alert
            self._resolve_alert(metric_name, timestamp)
    
    def _check_error_rate_alerts(self, timestamp: float):
        """Check error rates and trigger alerts if necessary."""
        try:
            # Get recent error statistics
            error_stats = error_handler.get_error_statistics(days=1)
            
            # Calculate error rate for the last hour
            current_hour = time.strftime('%Y-%m-%d:%H', time.localtime(timestamp))
            hourly_errors = 0
            
            for daily_stat in error_stats.get("daily_stats", []):
                if daily_stat.get("date") == time.strftime('%Y-%m-%d', time.localtime(timestamp)):
                    # This is a simplified calculation - in practice, you'd want hourly granularity
                    hourly_errors = daily_stat.get("total_errors", 0) / 24
                    break
            
            # Check against thresholds
            self._check_threshold_alert(
                "error_rate", hourly_errors, timestamp,
                "High Error Rate", f"Error rate: {hourly_errors:.1f} errors/hour"
            )
            
        except Exception as e:
            self.logger.warning(f"Failed to check error rate alerts: {e}")
    
    def _trigger_alert(self, metric_name: str, level: AlertLevel, title: str, 
                      message: str, current_value: Any, threshold_value: Any, timestamp: float):
        """Trigger an alert."""
        alert_id = f"{metric_name}_{level.value}"
        
        with self._lock:
            # Check if alert already exists and is active
            if alert_id in self.active_alerts and not self.active_alerts[alert_id].resolved:
                return  # Don't spam the same alert
            
            # Create new alert
            alert = Alert(
                alert_id=alert_id,
                timestamp=timestamp,
                level=level,
                title=title,
                message=message,
                metric_name=metric_name,
                current_value=current_value,
                threshold_value=threshold_value
            )
            
            self.active_alerts[alert_id] = alert
            self.alert_history.append(alert)
        
        # Log alert
        if level == AlertLevel.CRITICAL:
            self.logger.critical(f"ALERT: {title} - {message}")
        elif level == AlertLevel.ERROR:
            self.logger.error(f"ALERT: {title} - {message}")
        elif level == AlertLevel.WARNING:
            self.logger.warning(f"ALERT: {title} - {message}")
        else:
            self.logger.info(f"ALERT: {title} - {message}")
        
        # Store alert in Redis
        self._store_alert(alert)
        
        # Call alert callbacks
        for callback in self.alert_callbacks:
            try:
                callback(alert)
            except Exception as e:
                self.logger.error(f"Alert callback failed: {e}")
    
    def _resolve_alert(self, metric_name: str, timestamp: float):
        """Resolve alerts for a metric."""
        with self._lock:
            alerts_to_resolve = [
                alert_id for alert_id, alert in self.active_alerts.items()
                if alert.metric_name == metric_name and not alert.resolved
            ]
            
            for alert_id in alerts_to_resolve:
                alert = self.active_alerts[alert_id]
                alert.resolved = True
                alert.resolved_timestamp = timestamp
                
                self.logger.info(f"RESOLVED: {alert.title}")
                
                # Update in Redis
                self._store_alert(alert)
    
    def _store_alert(self, alert: Alert):
        """Store alert in Redis."""
        try:
            alert_key = f"alert:{alert.alert_id}:{int(alert.timestamp)}"
            redis_client.set_json(alert_key, alert.to_dict(), expire_seconds=86400 * 7)
            
            # Update alert statistics
            date_key = f"alert_stats:daily:{time.strftime('%Y-%m-%d', time.localtime(alert.timestamp))}"
            stats = redis_client.get_json(date_key) or {
                "date": time.strftime('%Y-%m-%d', time.localtime(alert.timestamp)),
                "total_alerts": 0,
                "by_level": {},
                "by_metric": {}
            }
            
            stats["total_alerts"] += 1
            stats["by_level"][alert.level.value] = stats["by_level"].get(alert.level.value, 0) + 1
            stats["by_metric"][alert.metric_name] = stats["by_metric"].get(alert.metric_name, 0) + 1
            
            redis_client.set_json(date_key, stats, expire_seconds=86400 * 30)
            
        except Exception as e:
            self.logger.warning(f"Failed to store alert: {e}")
    
    def get_active_alerts(self) -> List[Dict[str, Any]]:
        """Get all active alerts."""
        with self._lock:
            return [
                alert.to_dict() for alert in self.active_alerts.values()
                if not alert.resolved
            ]
    
    def get_alert_history(self, hours: int = 24) -> List[Dict[str, Any]]:
        """Get alert history for the specified number of hours."""
        try:
            history = []
            cutoff_time = time.time() - (hours * 3600)
            
            # Get from Redis
            for i in range(hours):
                hour_timestamp = time.time() - (i * 3600)
                date = time.strftime('%Y-%m-%d', time.localtime(hour_timestamp))
                
                # This is a simplified approach - in practice, you'd want more granular storage
                date_key = f"alert_stats:daily:{date}"
                daily_stats = redis_client.get_json(date_key)
                if daily_stats:
                    history.append(daily_stats)
            
            return history
            
        except Exception as e:
            self.logger.error(f"Failed to get alert history: {e}")
            return []


class HealthChecker:
    """Performs health checks on system components."""
    
    def __init__(self):
        self.logger = StructuredLogger(__name__)
        self.health_checks: Dict[str, Callable[[], bool]] = {}
        self.last_health_status: Dict[str, bool] = {}
    
    def register_health_check(self, name: str, check_func: Callable[[], bool]):
        """Register a health check function."""
        self.health_checks[name] = check_func
    
    def run_health_checks(self) -> Dict[str, Any]:
        """Run all registered health checks."""
        results = {
            "timestamp": time.time(),
            "overall_healthy": True,
            "checks": {}
        }
        
        for name, check_func in self.health_checks.items():
            try:
                start_time = time.time()
                is_healthy = check_func()
                check_time = time.time() - start_time
                
                results["checks"][name] = {
                    "healthy": is_healthy,
                    "response_time": check_time,
                    "last_check": time.time()
                }
                
                if not is_healthy:
                    results["overall_healthy"] = False
                    
                    # Trigger alert if health status changed
                    if self.last_health_status.get(name, True) != is_healthy:
                        self.logger.error(f"Health check failed: {name}")
                
                self.last_health_status[name] = is_healthy
                
            except Exception as e:
                self.logger.error(f"Health check error for {name}: {e}")
                results["checks"][name] = {
                    "healthy": False,
                    "error": str(e),
                    "last_check": time.time()
                }
                results["overall_healthy"] = False
        
        # Store health check results
        redis_client.set_json("health:current", results, expire_seconds=300)
        
        return results


# Global instances
metric_collector = MetricCollector()
alert_manager = AlertManager()
health_checker = HealthChecker()


def setup_default_health_checks():
    """Setup default health checks for common components."""
    
    def redis_health_check() -> bool:
        """Check Redis connectivity."""
        return redis_client.health_check()
    
    def celery_health_check() -> bool:
        """Check Celery worker availability."""
        try:
            from ..workers.celery_app import celery_app
            inspect = celery_app.control.inspect()
            stats = inspect.stats()
            return stats is not None and len(stats) > 0
        except Exception:
            return False
    
    def disk_space_check() -> bool:
        """Check available disk space."""
        try:
            disk = psutil.disk_usage('/')
            return disk.percent < 90  # Alert if disk usage > 90%
        except Exception:
            return False
    
    def memory_check() -> bool:
        """Check available memory."""
        try:
            memory = psutil.virtual_memory()
            return memory.percent < 90  # Alert if memory usage > 90%
        except Exception:
            return False
    
    # Register health checks
    health_checker.register_health_check("redis", redis_health_check)
    health_checker.register_health_check("celery", celery_health_check)
    health_checker.register_health_check("disk_space", disk_space_check)
    health_checker.register_health_check("memory", memory_check)


# Initialize default health checks
setup_default_health_checks()


def start_monitoring_thread(interval: int = 60):
    """Start background monitoring thread."""
    def monitoring_loop():
        while True:
            try:
                # Collect metrics
                metrics = metric_collector.collect_system_metrics()
                
                # Check for alerts
                alert_manager.check_metrics_alerts(metrics)
                
                # Run health checks
                health_checker.run_health_checks()
                
                time.sleep(interval)
                
            except Exception as e:
                logging.error(f"Monitoring loop error: {e}")
                time.sleep(interval)
    
    monitoring_thread = threading.Thread(target=monitoring_loop, daemon=True)
    monitoring_thread.start()
    return monitoring_thread