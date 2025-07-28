"""
Tests for system monitoring and alerting.
"""
import pytest
import time
import threading
from unittest.mock import Mock, patch, MagicMock
from datetime import datetime

from app.shared.monitoring import (
    MetricCollector, AlertManager, HealthChecker, SystemMetrics, Alert,
    AlertLevel, metric_collector, alert_manager, health_checker,
    setup_default_health_checks, start_monitoring_thread
)


class TestSystemMetrics:
    """Test cases for SystemMetrics dataclass."""
    
    def test_system_metrics_creation(self):
        """Test SystemMetrics creation and serialization."""
        timestamp = time.time()
        metrics = SystemMetrics(
            timestamp=timestamp,
            cpu_usage=50.0,
            memory_usage=60.0,
            memory_available=1024*1024*1024,  # 1GB
            disk_usage=70.0,
            disk_available=10*1024*1024*1024,  # 10GB
            active_connections=100,
            active_workers=5,
            active_tasks=20,
            queue_lengths={"embedding": 5, "query": 10}
        )
        
        # Test serialization
        metrics_dict = metrics.to_dict()
        assert metrics_dict['cpu_usage'] == 50.0
        assert metrics_dict['memory_usage'] == 60.0
        assert metrics_dict['active_workers'] == 5
        assert 'datetime' in metrics_dict
        assert isinstance(metrics_dict['datetime'], str)


class TestAlert:
    """Test cases for Alert dataclass."""
    
    def test_alert_creation(self):
        """Test Alert creation and serialization."""
        timestamp = time.time()
        alert = Alert(
            alert_id="test_alert",
            timestamp=timestamp,
            level=AlertLevel.WARNING,
            title="Test Alert",
            message="This is a test alert",
            metric_name="cpu_usage",
            current_value=85.0,
            threshold_value=80.0
        )
        
        # Test serialization
        alert_dict = alert.to_dict()
        assert alert_dict['alert_id'] == "test_alert"
        assert alert_dict['level'] == "warning"
        assert alert_dict['title'] == "Test Alert"
        assert alert_dict['current_value'] == 85.0
        assert 'datetime' in alert_dict
        assert isinstance(alert_dict['datetime'], str)
    
    def test_alert_resolution(self):
        """Test alert resolution functionality."""
        alert = Alert(
            alert_id="test_alert",
            timestamp=time.time(),
            level=AlertLevel.WARNING,
            title="Test Alert",
            message="Test message",
            metric_name="cpu_usage",
            current_value=85.0,
            threshold_value=80.0
        )
        
        assert not alert.resolved
        assert alert.resolved_timestamp is None
        
        # Resolve alert
        resolved_time = time.time()
        alert.resolved = True
        alert.resolved_timestamp = resolved_time
        
        assert alert.resolved
        assert alert.resolved_timestamp == resolved_time


class TestMetricCollector:
    """Test cases for MetricCollector class."""
    
    def setup_method(self):
        """Setup test environment."""
        self.collector = MetricCollector()
    
    @patch('app.shared.monitoring.psutil')
    @patch('app.shared.monitoring.redis_client')
    def test_collect_system_metrics_success(self, mock_redis, mock_psutil):
        """Test successful system metrics collection."""
        # Setup mocks
        mock_psutil.cpu_percent.return_value = 50.0
        mock_psutil.virtual_memory.return_value = Mock(percent=60.0, available=1024*1024*1024)
        mock_psutil.disk_usage.return_value = Mock(percent=70.0, free=10*1024*1024*1024)
        mock_psutil.net_connections.return_value = [Mock()] * 100
        
        mock_redis.set_json.return_value = True
        
        # Mock Celery stats
        with patch.object(self.collector, '_get_celery_stats', return_value=(5, 20)):
            with patch.object(self.collector, '_get_queue_lengths', return_value={"test": 10}):
                metrics = self.collector.collect_system_metrics()
        
        # Assertions
        assert isinstance(metrics, SystemMetrics)
        assert metrics.cpu_usage == 50.0
        assert metrics.memory_usage == 60.0
        assert metrics.disk_usage == 70.0
        assert metrics.active_connections == 100
        assert metrics.active_workers == 5
        assert metrics.active_tasks == 20
        assert metrics.queue_lengths == {"test": 10}
    
    @patch('app.shared.monitoring.psutil')
    def test_collect_system_metrics_failure(self, mock_psutil):
        """Test system metrics collection failure handling."""
        # Setup mock to raise exception
        mock_psutil.cpu_percent.side_effect = Exception("Test error")
        
        metrics = self.collector.collect_system_metrics()
        
        # Should return empty metrics on failure
        assert isinstance(metrics, SystemMetrics)
        assert metrics.cpu_usage == 0.0
        assert metrics.memory_usage == 0.0
        assert metrics.active_workers == 0
    
    @patch('app.shared.monitoring.redis_client')
    def test_store_metrics(self, mock_redis):
        """Test metrics storage in Redis."""
        mock_redis.set_json.return_value = True
        mock_redis.get_json.return_value = None
        
        metrics = SystemMetrics(
            timestamp=time.time(),
            cpu_usage=50.0,
            memory_usage=60.0,
            memory_available=1024*1024*1024,
            disk_usage=70.0,
            disk_available=10*1024*1024*1024,
            active_connections=100,
            active_workers=5,
            active_tasks=20,
            queue_lengths={}
        )
        
        self.collector._store_metrics(metrics)
        
        # Verify Redis calls
        assert mock_redis.set_json.call_count >= 2  # Current + hourly
        
        # Check current metrics storage
        current_call = None
        for call in mock_redis.set_json.call_args_list:
            if call[0][0] == "metrics:current":
                current_call = call
                break
        
        assert current_call is not None
        assert current_call[0][1]['cpu_usage'] == 50.0
    
    @patch('app.shared.monitoring.redis_client')
    def test_get_metrics_history(self, mock_redis):
        """Test metrics history retrieval."""
        # Setup mock data
        mock_data = {
            "hour": "2024-01-01:12",
            "samples": 10,
            "avg_cpu": 45.0,
            "avg_memory": 55.0,
            "max_cpu": 80.0
        }
        mock_redis.get_json.return_value = mock_data
        
        history = self.collector.get_metrics_history(hours=2)
        
        # Should return data for requested hours
        assert len(history) <= 2
        if history:
            assert history[0] == mock_data


class TestAlertManager:
    """Test cases for AlertManager class."""
    
    def setup_method(self):
        """Setup test environment."""
        self.alert_manager = AlertManager()
    
    def test_check_threshold_alert_warning(self):
        """Test threshold alert triggering for warning level."""
        # Create fresh alert manager to avoid state from other tests
        alert_manager = AlertManager()
        timestamp = time.time()
        
        # Should trigger warning alert
        alert_manager._check_threshold_alert(
            "cpu_usage", 75.0, timestamp,
            "High CPU Usage", "CPU usage is at 75.0%"
        )
        
        # Check that alert was created
        alert_id = "cpu_usage_warning"
        assert alert_id in alert_manager.active_alerts
        alert = alert_manager.active_alerts[alert_id]
        assert alert.level == AlertLevel.WARNING
        assert alert.current_value == 75.0
    
    def test_check_threshold_alert_critical(self):
        """Test threshold alert triggering for critical level."""
        timestamp = time.time()
        
        # Should trigger critical alert
        self.alert_manager._check_threshold_alert(
            "cpu_usage", 95.0, timestamp,
            "Critical CPU Usage", "CPU usage is at 95.0%"
        )
        
        # Check that alert was created
        alert_id = "cpu_usage_critical"
        assert alert_id in self.alert_manager.active_alerts
        alert = self.alert_manager.active_alerts[alert_id]
        assert alert.level == AlertLevel.CRITICAL
        assert alert.current_value == 95.0
    
    def test_check_threshold_alert_no_trigger(self):
        """Test that no alert is triggered when below threshold."""
        # Create fresh alert manager to avoid state from other tests
        alert_manager = AlertManager()
        timestamp = time.time()
        
        # Should not trigger any alert
        alert_manager._check_threshold_alert(
            "cpu_usage", 50.0, timestamp,
            "CPU Usage", "CPU usage is at 50.0%"
        )
        
        # Check that no alerts were created
        assert len(alert_manager.active_alerts) == 0
    
    def test_resolve_alert(self):
        """Test alert resolution."""
        timestamp = time.time()
        
        # Create an alert
        self.alert_manager._trigger_alert(
            "cpu_usage", AlertLevel.WARNING, "High CPU",
            "CPU usage high", 80.0, 70.0, timestamp
        )
        
        alert_id = "cpu_usage_warning"
        assert alert_id in self.alert_manager.active_alerts
        assert not self.alert_manager.active_alerts[alert_id].resolved
        
        # Resolve the alert
        self.alert_manager._resolve_alert("cpu_usage", timestamp + 60)
        
        # Check that alert was resolved
        assert self.alert_manager.active_alerts[alert_id].resolved
        assert self.alert_manager.active_alerts[alert_id].resolved_timestamp == timestamp + 60
    
    def test_alert_callback(self):
        """Test alert callback functionality."""
        callback_called = []
        
        def test_callback(alert):
            callback_called.append(alert)
        
        self.alert_manager.add_alert_callback(test_callback)
        
        # Trigger an alert
        timestamp = time.time()
        self.alert_manager._trigger_alert(
            "cpu_usage", AlertLevel.CRITICAL, "Critical CPU",
            "CPU usage critical", 95.0, 90.0, timestamp
        )
        
        # Check that callback was called
        assert len(callback_called) == 1
        assert callback_called[0].level == AlertLevel.CRITICAL
    
    def test_get_active_alerts(self):
        """Test getting active alerts."""
        timestamp = time.time()
        
        # Create some alerts
        self.alert_manager._trigger_alert(
            "cpu_usage", AlertLevel.WARNING, "High CPU",
            "CPU usage high", 80.0, 70.0, timestamp
        )
        
        self.alert_manager._trigger_alert(
            "memory_usage", AlertLevel.CRITICAL, "Critical Memory",
            "Memory usage critical", 95.0, 90.0, timestamp
        )
        
        # Resolve one alert
        self.alert_manager._resolve_alert("cpu_usage", timestamp + 60)
        
        # Get active alerts
        active_alerts = self.alert_manager.get_active_alerts()
        
        # Should only return unresolved alerts
        assert len(active_alerts) == 1
        assert active_alerts[0]['metric_name'] == 'memory_usage'
    
    @patch('app.shared.monitoring.redis_client')
    def test_store_alert(self, mock_redis):
        """Test alert storage in Redis."""
        mock_redis.set_json.return_value = True
        mock_redis.get_json.return_value = None
        
        alert = Alert(
            alert_id="test_alert",
            timestamp=time.time(),
            level=AlertLevel.WARNING,
            title="Test Alert",
            message="Test message",
            metric_name="cpu_usage",
            current_value=80.0,
            threshold_value=70.0
        )
        
        self.alert_manager._store_alert(alert)
        
        # Verify Redis calls
        assert mock_redis.set_json.call_count >= 2  # Alert + stats
        
        # Check alert storage
        alert_call = None
        for call in mock_redis.set_json.call_args_list:
            if call[0][0].startswith("alert:"):
                alert_call = call
                break
        
        assert alert_call is not None
        assert alert_call[0][1]['alert_id'] == "test_alert"


class TestHealthChecker:
    """Test cases for HealthChecker class."""
    
    def setup_method(self):
        """Setup test environment."""
        self.health_checker = HealthChecker()
    
    def test_register_health_check(self):
        """Test health check registration."""
        def test_check():
            return True
        
        self.health_checker.register_health_check("test_service", test_check)
        
        assert "test_service" in self.health_checker.health_checks
        assert self.health_checker.health_checks["test_service"] == test_check
    
    def test_run_health_checks_all_healthy(self):
        """Test running health checks when all are healthy."""
        def healthy_check():
            return True
        
        def another_healthy_check():
            return True
        
        self.health_checker.register_health_check("service1", healthy_check)
        self.health_checker.register_health_check("service2", another_healthy_check)
        
        results = self.health_checker.run_health_checks()
        
        assert results["overall_healthy"] is True
        assert len(results["checks"]) == 2
        assert results["checks"]["service1"]["healthy"] is True
        assert results["checks"]["service2"]["healthy"] is True
    
    def test_run_health_checks_some_unhealthy(self):
        """Test running health checks when some are unhealthy."""
        def healthy_check():
            return True
        
        def unhealthy_check():
            return False
        
        self.health_checker.register_health_check("service1", healthy_check)
        self.health_checker.register_health_check("service2", unhealthy_check)
        
        results = self.health_checker.run_health_checks()
        
        assert results["overall_healthy"] is False
        assert results["checks"]["service1"]["healthy"] is True
        assert results["checks"]["service2"]["healthy"] is False
    
    def test_run_health_checks_with_exception(self):
        """Test running health checks when one throws exception."""
        def healthy_check():
            return True
        
        def failing_check():
            raise Exception("Health check failed")
        
        self.health_checker.register_health_check("service1", healthy_check)
        self.health_checker.register_health_check("service2", failing_check)
        
        results = self.health_checker.run_health_checks()
        
        assert results["overall_healthy"] is False
        assert results["checks"]["service1"]["healthy"] is True
        assert results["checks"]["service2"]["healthy"] is False
        assert "error" in results["checks"]["service2"]
    
    @patch('app.shared.monitoring.redis_client')
    def test_health_check_storage(self, mock_redis):
        """Test health check results storage."""
        mock_redis.set_json.return_value = True
        
        def test_check():
            return True
        
        self.health_checker.register_health_check("test_service", test_check)
        results = self.health_checker.run_health_checks()
        
        # Verify Redis storage
        mock_redis.set_json.assert_called_once()
        call_args = mock_redis.set_json.call_args
        assert call_args[0][0] == "health:current"
        assert call_args[0][1]["overall_healthy"] is True


class TestDefaultHealthChecks:
    """Test cases for default health checks."""
    
    @patch('app.shared.monitoring.redis_client')
    def test_redis_health_check(self, mock_redis):
        """Test Redis health check."""
        mock_redis.health_check.return_value = True
        
        # Setup default health checks
        setup_default_health_checks()
        
        # Run health checks
        results = health_checker.run_health_checks()
        
        assert "redis" in results["checks"]
        assert results["checks"]["redis"]["healthy"] is True
    
    @patch('app.shared.monitoring.psutil')
    def test_disk_space_check(self, mock_psutil):
        """Test disk space health check."""
        # Mock disk usage below threshold
        mock_psutil.disk_usage.return_value = Mock(percent=85.0)
        
        # Setup default health checks
        setup_default_health_checks()
        
        # Run health checks
        results = health_checker.run_health_checks()
        
        assert "disk_space" in results["checks"]
        assert results["checks"]["disk_space"]["healthy"] is True
    
    @patch('app.shared.monitoring.psutil')
    def test_memory_check(self, mock_psutil):
        """Test memory health check."""
        # Mock memory usage below threshold
        mock_psutil.virtual_memory.return_value = Mock(percent=85.0)
        
        # Setup default health checks
        setup_default_health_checks()
        
        # Run health checks
        results = health_checker.run_health_checks()
        
        assert "memory" in results["checks"]
        assert results["checks"]["memory"]["healthy"] is True


class TestMonitoringIntegration:
    """Integration tests for monitoring system."""
    
    @patch('app.shared.monitoring.psutil')
    @patch('app.shared.monitoring.redis_client')
    def test_metrics_to_alerts_integration(self, mock_redis, mock_psutil):
        """Test integration between metrics collection and alerting."""
        # Setup mocks for high CPU usage
        mock_psutil.cpu_percent.return_value = 95.0  # Above critical threshold
        mock_psutil.virtual_memory.return_value = Mock(percent=60.0, available=1024*1024*1024)
        mock_psutil.disk_usage.return_value = Mock(percent=70.0, free=10*1024*1024*1024)
        mock_psutil.net_connections.return_value = []
        
        mock_redis.set_json.return_value = True
        mock_redis.get_json.return_value = None
        
        collector = MetricCollector()
        alert_mgr = AlertManager()
        
        # Mock Celery stats
        with patch.object(collector, '_get_celery_stats', return_value=(0, 0)):
            with patch.object(collector, '_get_queue_lengths', return_value={}):
                # Collect metrics
                metrics = collector.collect_system_metrics()
                
                # Check for alerts
                alert_mgr.check_metrics_alerts(metrics)
        
        # Should have triggered critical CPU alert
        active_alerts = alert_mgr.get_active_alerts()
        cpu_alerts = [alert for alert in active_alerts if alert['metric_name'] == 'cpu_usage']
        
        assert len(cpu_alerts) > 0
        assert any(alert['level'] == 'critical' for alert in cpu_alerts)
    
    def test_concurrent_monitoring(self):
        """Test monitoring system under concurrent access."""
        collector = MetricCollector()
        alert_mgr = AlertManager()
        
        results = []
        
        def monitoring_thread():
            try:
                # Simulate monitoring loop
                for _ in range(5):
                    with patch('app.shared.monitoring.psutil') as mock_psutil:
                        mock_psutil.cpu_percent.return_value = 50.0
                        mock_psutil.virtual_memory.return_value = Mock(percent=60.0, available=1024*1024*1024)
                        mock_psutil.disk_usage.return_value = Mock(percent=70.0, free=10*1024*1024*1024)
                        mock_psutil.net_connections.return_value = []
                        
                        with patch.object(collector, '_get_celery_stats', return_value=(0, 0)):
                            with patch.object(collector, '_get_queue_lengths', return_value={}):
                                metrics = collector.collect_system_metrics()
                                alert_mgr.check_metrics_alerts(metrics)
                                results.append(metrics.cpu_usage)
                    
                    time.sleep(0.01)  # Small delay
            except Exception as e:
                pytest.fail(f"Monitoring thread failed: {e}")
        
        # Start multiple monitoring threads
        threads = []
        for _ in range(3):
            thread = threading.Thread(target=monitoring_thread)
            threads.append(thread)
            thread.start()
        
        # Wait for completion
        for thread in threads:
            thread.join()
        
        # Verify all threads completed successfully
        assert len(results) == 15  # 3 threads * 5 iterations each
        assert all(cpu == 50.0 for cpu in results)
    
    def test_start_monitoring_thread(self):
        """Test monitoring thread startup."""
        with patch('app.shared.monitoring.metric_collector') as mock_collector:
            with patch('app.shared.monitoring.alert_manager') as mock_alert_mgr:
                with patch('app.shared.monitoring.health_checker') as mock_health:
                    mock_collector.collect_system_metrics.return_value = Mock()
                    mock_alert_mgr.check_metrics_alerts.return_value = None
                    mock_health.run_health_checks.return_value = {}
                    
                    # Start monitoring thread
                    thread = start_monitoring_thread(interval=1)
                    
                    # Wait a bit for thread to start
                    import time
                    time.sleep(0.1)
                    
                    # Verify thread is running
                    assert thread.is_alive()
                    assert thread.daemon  # Should be daemon thread
                    
                    # Verify the monitoring functions were called
                    # Note: Due to threading timing, we can't guarantee exact call counts
                    # but we can verify the thread started successfully


if __name__ == "__main__":
    pytest.main([__file__])