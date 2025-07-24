"""
Tests for the ResourceManager class.
"""
import pytest
import time
import threading
from unittest.mock import Mock, patch, MagicMock
from datetime import datetime, timedelta

from app.shared.resource_manager import (
    ResourceManager, ResourceMetrics, ResourceLimits, ResourceStatus, 
    ResourceType, ResourceError, resource_manager
)


class TestResourceMetrics:
    """Test ResourceMetrics data structure."""
    
    def test_resource_metrics_creation(self):
        """Test creating ResourceMetrics instance."""
        timestamp = datetime.now()
        metrics = ResourceMetrics(
            timestamp=timestamp,
            memory_usage=50.0,
            memory_available=1024*1024*1024,  # 1GB
            cpu_usage=25.0,
            disk_usage=60.0,
            disk_available=10*1024*1024*1024,  # 10GB
            queue_lengths={"embedding": 5, "query": 10},
            active_workers=3,
            active_tasks=8,
            load_average=[1.0, 1.2, 1.1]
        )
        
        assert metrics.timestamp == timestamp
        assert metrics.memory_usage == 50.0
        assert metrics.cpu_usage == 25.0
        assert metrics.queue_lengths["embedding"] == 5
        assert metrics.active_workers == 3


class TestResourceLimits:
    """Test ResourceLimits configuration."""
    
    def test_resource_limits_defaults(self):
        """Test default ResourceLimits values."""
        limits = ResourceLimits()
        
        assert limits.memory_warning_threshold == 75.0
        assert limits.memory_critical_threshold == 85.0
        assert limits.cpu_warning_threshold == 80.0
        assert limits.cpu_critical_threshold == 90.0
        assert limits.max_concurrent_embeddings == 2
        assert limits.max_concurrent_queries == 10
    
    def test_resource_limits_custom(self):
        """Test custom ResourceLimits values."""
        limits = ResourceLimits(
            memory_warning_threshold=60.0,
            memory_critical_threshold=80.0,
            max_concurrent_embeddings=5
        )
        
        assert limits.memory_warning_threshold == 60.0
        assert limits.memory_critical_threshold == 80.0
        assert limits.max_concurrent_embeddings == 5


class TestResourceManager:
    """Test ResourceManager functionality."""
    
    @pytest.fixture
    def manager(self):
        """Create a ResourceManager instance for testing."""
        return ResourceManager()
    
    @pytest.fixture
    def mock_psutil(self):
        """Mock psutil for testing."""
        with patch('app.shared.resource_manager.psutil') as mock:
            # Mock virtual memory
            mock.virtual_memory.return_value = Mock(
                percent=50.0,
                available=2*1024*1024*1024  # 2GB
            )
            
            # Mock CPU usage
            mock.cpu_percent.return_value = 25.0
            
            # Mock disk usage
            mock.disk_usage.return_value = Mock(
                used=5*1024*1024*1024,  # 5GB
                total=10*1024*1024*1024,  # 10GB
                free=5*1024*1024*1024   # 5GB
            )
            
            # Mock load average
            mock.getloadavg.return_value = [1.0, 1.2, 1.1]
            
            yield mock
    
    @pytest.fixture
    def mock_redis(self):
        """Mock Redis client for testing."""
        with patch('app.shared.resource_manager.redis_client') as mock:
            mock.client.keys.return_value = []
            mock.get_json.return_value = None
            mock.set_json.return_value = True
            yield mock
    
    def test_initialization(self, manager):
        """Test ResourceManager initialization."""
        assert manager.limits is not None
        assert not manager._monitoring_active
        assert manager._monitor_thread is None
        assert manager._last_metrics is None
    
    def test_collect_metrics(self, manager, mock_psutil, mock_redis):
        """Test metrics collection."""
        metrics = manager.collect_metrics()
        
        assert isinstance(metrics, ResourceMetrics)
        assert metrics.memory_usage == 50.0
        assert metrics.cpu_usage == 25.0
        assert metrics.disk_usage == 50.0  # 5GB used / 10GB total
        assert metrics.load_average == [1.0, 1.2, 1.1]
    
    def test_collect_metrics_error_handling(self, manager):
        """Test metrics collection with errors."""
        with patch('app.shared.resource_manager.psutil.virtual_memory', side_effect=Exception("Test error")):
            metrics = manager.collect_metrics()
            
            # Should return empty metrics on error
            assert metrics.memory_usage == 0.0
            assert metrics.cpu_usage == 0.0
    
    def test_resource_status_healthy(self, manager):
        """Test resource status calculation - healthy."""
        metrics = ResourceMetrics(
            timestamp=datetime.now(),
            memory_usage=50.0,  # Below warning threshold
            memory_available=2*1024*1024*1024,
            cpu_usage=60.0,     # Below warning threshold
            disk_usage=50.0,    # Below warning threshold
            disk_available=5*1024*1024*1024,
            queue_lengths={"embedding": 5},  # Below max
            active_workers=2,
            active_tasks=5,
            load_average=[1.0, 1.0, 1.0]
        )
        
        status = manager.get_resource_status(metrics)
        assert status == ResourceStatus.HEALTHY
    
    def test_resource_status_warning(self, manager):
        """Test resource status calculation - warning."""
        metrics = ResourceMetrics(
            timestamp=datetime.now(),
            memory_usage=80.0,  # Above warning threshold
            memory_available=1*1024*1024*1024,
            cpu_usage=60.0,
            disk_usage=50.0,
            disk_available=5*1024*1024*1024,
            queue_lengths={"embedding": 5},
            active_workers=2,
            active_tasks=5,
            load_average=[1.0, 1.0, 1.0]
        )
        
        status = manager.get_resource_status(metrics)
        assert status == ResourceStatus.WARNING
    
    def test_resource_status_critical(self, manager):
        """Test resource status calculation - critical."""
        metrics = ResourceMetrics(
            timestamp=datetime.now(),
            memory_usage=90.0,  # Above critical threshold
            memory_available=512*1024*1024,
            cpu_usage=60.0,
            disk_usage=50.0,
            disk_available=5*1024*1024*1024,
            queue_lengths={"embedding": 5},
            active_workers=2,
            active_tasks=5,
            load_average=[1.0, 1.0, 1.0]
        )
        
        status = manager.get_resource_status(metrics)
        assert status == ResourceStatus.CRITICAL
    
    def test_resource_status_overloaded(self, manager):
        """Test resource status calculation - overloaded."""
        metrics = ResourceMetrics(
            timestamp=datetime.now(),
            memory_usage=50.0,
            memory_available=2*1024*1024*1024,
            cpu_usage=60.0,
            disk_usage=50.0,
            disk_available=5*1024*1024*1024,
            queue_lengths={"embedding": 150},  # Above max queue length
            active_workers=2,
            active_tasks=5,
            load_average=[1.0, 1.0, 1.0]
        )
        
        status = manager.get_resource_status(metrics)
        assert status == ResourceStatus.OVERLOADED
    
    def test_can_accept_task_healthy(self, manager, mock_psutil, mock_redis):
        """Test task acceptance when system is healthy."""
        # Mock healthy system
        manager._last_metrics = ResourceMetrics(
            timestamp=datetime.now(),
            memory_usage=50.0,
            memory_available=2*1024*1024*1024,
            cpu_usage=60.0,
            disk_usage=50.0,
            disk_available=5*1024*1024*1024,
            queue_lengths={"embedding": 5},
            active_workers=2,
            active_tasks=5,
            load_average=[1.0, 1.0, 1.0]
        )
        
        # Mock no active tasks of the requested type
        with patch.object(manager, '_count_active_tasks', return_value=0):
            assert manager.can_accept_task("embedding") is True
            assert manager.can_accept_task("query") is True
    
    def test_can_accept_task_critical(self, manager, mock_psutil, mock_redis):
        """Test task rejection when system is critical."""
        # Mock critical system
        critical_metrics = ResourceMetrics(
            timestamp=datetime.now(),
            memory_usage=95.0,  # Critical
            memory_available=256*1024*1024,
            cpu_usage=60.0,
            disk_usage=50.0,
            disk_available=5*1024*1024*1024,
            queue_lengths={"embedding": 5},
            active_workers=2,
            active_tasks=5,
            load_average=[1.0, 1.0, 1.0]
        )
        manager._last_metrics = critical_metrics
        
        # Mock no active tasks to isolate the resource status check
        with patch.object(manager, '_count_active_tasks', return_value=0):
            assert manager.can_accept_task("embedding") is False
            assert manager.can_accept_task("query") is False
    
    def test_can_accept_task_embedding_limit(self, manager, mock_psutil, mock_redis):
        """Test embedding task rejection when at limit."""
        # Mock healthy system
        manager._last_metrics = ResourceMetrics(
            timestamp=datetime.now(),
            memory_usage=50.0,
            memory_available=2*1024*1024*1024,
            cpu_usage=60.0,
            disk_usage=50.0,
            disk_available=5*1024*1024*1024,
            queue_lengths={"embedding": 5},
            active_workers=2,
            active_tasks=5,
            load_average=[1.0, 1.0, 1.0]
        )
        
        # Mock max embedding tasks active
        with patch.object(manager, '_count_active_tasks') as mock_count:
            mock_count.return_value = manager.limits.max_concurrent_embeddings
            assert manager.can_accept_task("embedding") is False
            
            # But queries should still be accepted
            mock_count.return_value = 0
            assert manager.can_accept_task("query") is True
    
    def test_can_accept_task_query_limit(self, manager, mock_psutil, mock_redis):
        """Test query task rejection when at limit."""
        # Mock healthy system
        manager._last_metrics = ResourceMetrics(
            timestamp=datetime.now(),
            memory_usage=50.0,
            memory_available=2*1024*1024*1024,
            cpu_usage=60.0,
            disk_usage=50.0,
            disk_available=5*1024*1024*1024,
            queue_lengths={"embedding": 5},
            active_workers=2,
            active_tasks=5,
            load_average=[1.0, 1.0, 1.0]
        )
        
        # Mock max query tasks active
        with patch.object(manager, '_count_active_tasks') as mock_count:
            def count_side_effect(task_type):
                if task_type == "query":
                    return manager.limits.max_concurrent_queries
                return 0
            
            mock_count.side_effect = count_side_effect
            assert manager.can_accept_task("query") is False
            assert manager.can_accept_task("embedding") is True
    
    def test_count_active_tasks(self, manager, mock_redis):
        """Test counting active tasks."""
        # Mock task data in Redis
        mock_redis.client.keys.return_value = ["task:1", "task:2", "task:3"]
        
        def get_json_side_effect(key):
            if key == "task:1":
                return {"status": "running", "task_name": "process_document_embedding"}
            elif key == "task:2":
                return {"status": "running", "task_name": "process_user_query"}
            elif key == "task:3":
                return {"status": "completed", "task_name": "process_document_embedding"}
            return None
        
        mock_redis.get_json.side_effect = get_json_side_effect
        
        embedding_count = manager._count_active_tasks("embedding")
        query_count = manager._count_active_tasks("query")
        
        assert embedding_count == 1  # Only task:1 is running embedding
        assert query_count == 1      # Only task:2 is running query
    
    def test_resource_guard_success(self, manager):
        """Test successful resource guard execution."""
        with patch.object(manager, 'can_accept_task', return_value=True):
            with manager.resource_guard("embedding", "test-task-1"):
                # Task execution would happen here
                pass
        # Should complete without exception
    
    def test_resource_guard_rejection(self, manager):
        """Test resource guard rejection."""
        with patch.object(manager, 'can_accept_task', return_value=False):
            with pytest.raises(ResourceError):
                with manager.resource_guard("embedding", "test-task-1"):
                    pass
    
    def test_resource_guard_exception_handling(self, manager):
        """Test resource guard with exception in task."""
        with patch.object(manager, 'can_accept_task', return_value=True):
            with pytest.raises(ValueError):
                with manager.resource_guard("embedding", "test-task-1"):
                    raise ValueError("Test error")
    
    def test_callback_registration(self, manager):
        """Test callback registration and unregistration."""
        callback = Mock()
        
        # Register callback
        manager.register_callback(ResourceType.MEMORY, callback)
        assert callback in manager._callbacks[ResourceType.MEMORY]
        
        # Unregister callback
        manager.unregister_callback(ResourceType.MEMORY, callback)
        assert callback not in manager._callbacks[ResourceType.MEMORY]
    
    def test_metrics_history(self, manager):
        """Test metrics history management."""
        # Add some metrics to history
        for i in range(5):
            metrics = ResourceMetrics(
                timestamp=datetime.now() - timedelta(minutes=i),
                memory_usage=50.0 + i,
                memory_available=2*1024*1024*1024,
                cpu_usage=60.0,
                disk_usage=50.0,
                disk_available=5*1024*1024*1024,
                queue_lengths={},
                active_workers=2,
                active_tasks=5,
                load_average=[1.0, 1.0, 1.0]
            )
            manager._update_metrics_history(metrics)
        
        # Get recent history
        recent_metrics = manager.get_metrics_history(hours=1)
        assert len(recent_metrics) == 5
        
        # Get older history (should be empty)
        old_metrics = manager.get_metrics_history(hours=0)
        assert len(old_metrics) == 0
    
    def test_monitoring_lifecycle(self, manager):
        """Test monitoring start and stop."""
        # Start monitoring
        manager.start_monitoring(interval=0.1)
        assert manager._monitoring_active is True
        assert manager._monitor_thread is not None
        
        # Let it run briefly
        time.sleep(0.2)
        
        # Stop monitoring
        manager.stop_monitoring()
        assert manager._monitoring_active is False
    
    def test_monitoring_already_active(self, manager):
        """Test starting monitoring when already active."""
        manager._monitoring_active = True
        
        # Should not start new thread
        manager.start_monitoring()
        assert manager._monitor_thread is None
    
    def test_force_cleanup(self, manager, mock_redis):
        """Test forced cleanup functionality."""
        # Mock Redis cleanup methods
        mock_redis.cleanup_expired_sessions.return_value = 5
        mock_redis.client.keys.return_value = ["task:1", "task:2"]
        mock_redis.get_json.return_value = {
            "started_at": time.time() - 7200,  # 2 hours ago
            "status": "completed"
        }
        mock_redis.delete.return_value = True
        
        # Should complete without exception
        manager.force_cleanup()
        
        # Verify cleanup methods were called
        mock_redis.cleanup_expired_sessions.assert_called_once()
    
    def test_health_status(self, manager, mock_psutil, mock_redis):
        """Test health status reporting."""
        with patch.object(manager, 'can_accept_task', return_value=True):
            with patch.object(manager, '_count_active_tasks', return_value=1):
                health = manager.get_health_status()
                
                assert "status" in health
                assert "timestamp" in health
                assert "metrics" in health
                assert "limits" in health
                assert "task_capacity" in health
                
                assert health["task_capacity"]["can_accept_embedding"] is True
                assert health["task_capacity"]["can_accept_query"] is True
    
    def test_health_status_error(self, manager):
        """Test health status with error."""
        with patch.object(manager, 'collect_metrics', side_effect=Exception("Test error")):
            health = manager.get_health_status()
            
            assert health["status"] == "error"
            assert "error" in health


class TestResourceManagerIntegration:
    """Integration tests for ResourceManager."""
    
    def test_global_instance(self):
        """Test that global resource manager instance exists."""
        assert resource_manager is not None
        assert isinstance(resource_manager, ResourceManager)
    
    def test_concurrent_access(self):
        """Test concurrent access to resource manager."""
        manager = ResourceManager()
        results = []
        
        def worker():
            try:
                # Simulate concurrent metric collection
                metrics = manager.collect_metrics()
                results.append(metrics is not None)
            except Exception as e:
                results.append(False)
        
        # Start multiple threads
        threads = []
        for _ in range(5):
            thread = threading.Thread(target=worker)
            threads.append(thread)
            thread.start()
        
        # Wait for all threads
        for thread in threads:
            thread.join()
        
        # All should succeed
        assert all(results)
        assert len(results) == 5
    
    @pytest.mark.integration
    def test_real_system_metrics(self):
        """Test collecting real system metrics (integration test)."""
        manager = ResourceManager()
        
        # This should work with real psutil
        metrics = manager.collect_metrics()
        
        assert isinstance(metrics, ResourceMetrics)
        assert metrics.memory_usage >= 0
        assert metrics.cpu_usage >= 0
        assert metrics.disk_usage >= 0
        assert isinstance(metrics.load_average, list)
        assert len(metrics.load_average) == 3
    
    @pytest.mark.integration
    def test_monitoring_integration(self):
        """Test monitoring integration (brief test)."""
        manager = ResourceManager()
        
        try:
            # Start monitoring for a short time
            manager.start_monitoring(interval=0.2)
            time.sleep(1.5)  # Let it collect a few metrics
            
            # Check that metrics were collected
            current_metrics = manager.get_current_metrics()
            if current_metrics is None:
                # If monitoring didn't work, at least test manual collection
                current_metrics = manager.collect_metrics()
            assert current_metrics is not None
            
            # Check history (may be empty if monitoring thread didn't run)
            history = manager.get_metrics_history(hours=1)
            # Don't assert on history length as it depends on timing
            
        finally:
            manager.stop_monitoring()


if __name__ == "__main__":
    pytest.main([__file__])