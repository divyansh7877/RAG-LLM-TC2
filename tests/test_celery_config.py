"""
Tests for Celery configuration and worker infrastructure.
"""
import pytest
import time
import os
from unittest.mock import patch, MagicMock
from app.workers.celery_app import celery_app, get_worker_stats
from app.workers.maintenance_worker import check_resource_limits
from app.workers.monitor import WorkerMonitor, WorkerStatus


class TestCeleryConfiguration:
    """Test Celery app configuration."""
    
    def test_celery_app_creation(self):
        """Test that Celery app is created with correct configuration."""
        assert celery_app.main == "rag_workers"
        assert celery_app.conf.broker_url is not None
        assert celery_app.conf.result_backend is not None
    
    def test_task_routing_configuration(self):
        """Test that task routing is configured correctly."""
        routes = celery_app.conf.task_routes
        
        assert "app.workers.embedding_worker.*" in routes
        assert "app.workers.query_worker.*" in routes
        assert "app.workers.maintenance_worker.*" in routes
        
        assert routes["app.workers.embedding_worker.*"]["queue"] == "embedding"
        assert routes["app.workers.query_worker.*"]["queue"] == "query"
        assert routes["app.workers.maintenance_worker.*"]["queue"] == "maintenance"
    
    def test_worker_configuration(self):
        """Test worker configuration settings."""
        conf = celery_app.conf
        
        assert conf.worker_prefetch_multiplier == 1
        assert conf.task_acks_late is True
        assert conf.task_serializer == "json"
        assert conf.result_serializer == "json"
        assert conf.timezone == "UTC"
        assert conf.enable_utc is True
    
    def test_resource_limits_configuration(self):
        """Test resource limits are properly configured."""
        conf = celery_app.conf
        
        assert conf.worker_max_tasks_per_child == 100
        assert conf.worker_max_memory_per_child == 2048000
        assert conf.task_soft_time_limit > 0
        assert conf.task_time_limit > conf.task_soft_time_limit
    
    def test_beat_schedule_configuration(self):
        """Test that beat schedule is configured correctly."""
        schedule = celery_app.conf.beat_schedule
        
        assert "cleanup-expired-sessions" in schedule
        assert "system-health-check" in schedule
        assert "worker-health-report" in schedule
        
        # Check task names and schedules
        assert schedule["cleanup-expired-sessions"]["task"] == "cleanup_expired_sessions"
        assert schedule["system-health-check"]["task"] == "system_health_check"
        assert schedule["worker-health-report"]["task"] == "worker_health_report"


class TestWorkerStats:
    """Test worker statistics functions."""
    
    @patch('psutil.Process')
    def test_get_worker_stats(self, mock_process):
        """Test worker statistics collection."""
        # Mock process info
        mock_proc = MagicMock()
        mock_proc.memory_info.return_value.rss = 1024 * 1024 * 100  # 100MB
        mock_proc.memory_percent.return_value = 5.0
        mock_proc.cpu_percent.return_value = 10.0
        mock_proc.num_threads.return_value = 4
        mock_proc.create_time.return_value = time.time() - 3600  # 1 hour ago
        mock_proc.status.return_value = "running"
        mock_process.return_value = mock_proc
        
        stats = get_worker_stats()
        
        assert "pid" in stats
        assert "memory_usage" in stats
        assert "memory_percent" in stats
        assert "cpu_percent" in stats
        assert "num_threads" in stats
        assert "create_time" in stats
        assert "status" in stats
        
        assert stats["memory_usage"] == 1024 * 1024 * 100
        assert stats["memory_percent"] == 5.0
        assert stats["cpu_percent"] == 10.0
        assert stats["num_threads"] == 4
        assert stats["status"] == "running"
    
    @patch('psutil.Process')
    def test_check_resource_limits_healthy(self, mock_process):
        """Test resource limit checking when healthy."""
        # Mock healthy process
        mock_proc = MagicMock()
        mock_proc.memory_info.return_value.rss = 512 * 1024 * 1024  # 512MB
        mock_proc.cpu_percent.return_value = 30.0
        mock_process.return_value = mock_proc
        
        result = check_resource_limits()
        
        assert result["healthy"] is True
        assert len(result["warnings"]) == 0
        assert result["memory_mb"] == 512
        assert result["cpu_percent"] == 30.0
    
    @patch('psutil.Process')
    def test_check_resource_limits_memory_warning(self, mock_process):
        """Test resource limit checking with memory warning."""
        # Mock high memory usage
        mock_proc = MagicMock()
        mock_proc.memory_info.return_value.rss = 1800 * 1024 * 1024  # 1800MB (>80% of 2GB)
        mock_proc.cpu_percent.return_value = 30.0
        mock_process.return_value = mock_proc
        
        result = check_resource_limits()
        
        assert result["healthy"] is False
        assert len(result["warnings"]) > 0
        assert any("Memory usage high" in warning for warning in result["warnings"])
    
    @patch('psutil.Process')
    def test_check_resource_limits_cpu_warning(self, mock_process):
        """Test resource limit checking with CPU warning."""
        # Mock high CPU usage
        mock_proc = MagicMock()
        mock_proc.memory_info.return_value.rss = 512 * 1024 * 1024  # 512MB
        mock_proc.cpu_percent.return_value = 85.0  # >80%
        mock_process.return_value = mock_proc
        
        result = check_resource_limits()
        
        assert result["healthy"] is False
        assert len(result["warnings"]) > 0
        assert any("CPU usage high" in warning for warning in result["warnings"])


class TestWorkerMonitor:
    """Test worker monitoring functionality."""
    
    @patch('app.workers.monitor.redis_client')
    def test_worker_monitor_initialization(self, mock_redis):
        """Test worker monitor initialization."""
        monitor = WorkerMonitor()
        assert monitor.redis == mock_redis
        assert monitor.heartbeat_timeout == 120
    
    @patch('app.workers.monitor.redis_client')
    def test_get_all_workers_empty(self, mock_redis):
        """Test getting workers when none exist."""
        mock_redis.get_json.return_value = {}
        
        monitor = WorkerMonitor()
        workers = monitor.get_all_workers()
        
        assert workers == {}
    
    @patch('app.workers.monitor.redis_client')
    def test_get_all_workers_with_data(self, mock_redis):
        """Test getting workers with sample data."""
        # Mock registry
        registry = {"worker1": {"last_seen": time.time()}}
        
        # Mock worker data
        worker_data = {
            "hostname": "test-host",
            "pid": 12345,
            "last_heartbeat": time.time(),
            "memory_usage": 1024 * 1024 * 100,
            "memory_percent": 5.0,
            "cpu_percent": 10.0,
            "uptime": 3600,
            "healthy": True,
            "resource_check": {"warnings": []}
        }
        
        def mock_get_json(key):
            if key == "workers:registry":
                return registry
            elif key == "worker:worker1":
                return worker_data
            return None
        
        mock_redis.get_json.side_effect = mock_get_json
        
        monitor = WorkerMonitor()
        workers = monitor.get_all_workers()
        
        assert len(workers) == 1
        assert "worker1" in workers
        
        worker = workers["worker1"]
        assert worker.hostname == "test-host"
        assert worker.pid == 12345
        assert worker.status == WorkerStatus.HEALTHY
        assert worker.memory_usage == 1024 * 1024 * 100
    
    @patch('app.workers.monitor.redis_client')
    def test_cleanup_stale_worker_data(self, mock_redis):
        """Test cleanup of stale worker data."""
        current_time = time.time()
        stale_time = current_time - 300  # 5 minutes ago (stale)
        
        registry = {
            "worker1": {"last_seen": current_time},  # Fresh
            "worker2": {"last_seen": stale_time}     # Stale
        }
        
        mock_redis.get_json.return_value = registry
        mock_redis.client.delete = MagicMock()
        
        monitor = WorkerMonitor()
        result = monitor.cleanup_stale_worker_data()
        
        assert "worker2" in result["cleaned_workers"]
        assert "worker1" not in result["cleaned_workers"]
        mock_redis.client.delete.assert_called_with("worker:worker2")


if __name__ == "__main__":
    pytest.main([__file__])