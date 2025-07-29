"""
Performance benchmarking and load testing.
Tests system performance under various load conditions.
"""
import pytest
import time
import threading
import statistics
from unittest.mock import Mock, patch, MagicMock
from concurrent.futures import ThreadPoolExecutor, as_completed
import asyncio

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.shared.session_manager import SessionManager
from app.shared.auth import AuthenticationManager
from app.shared.job_manager import JobManager
from app.shared.resource_manager import ResourceManager
from app.shared.models import Job, JobStatus, JobType, UserSession


class TestPerformanceBenchmarks:
    """Performance benchmarking tests."""
    
    def test_authentication_performance_benchmark(self):
        """Benchmark authentication performance."""
        auth_manager = AuthenticationManager()
        performance_metrics = []
        
        with patch('app.shared.auth.config') as mock_config:
            mock_config.SECRET_KEY = "test-secret"
            mock_config.ACCESS_TOKEN_EXPIRE_MINUTES = 30
            mock_config.USERS = {
                f"user_{i}": {"password": f"pass_{i}", "groups": [f"group_{i}"]}
                for i in range(100)
            }
            
            with patch('app.shared.auth.session_manager') as mock_session_manager:
                def create_session_side_effect(user_id, groups, permissions):
                    return UserSession(
                        session_id=f"session_{user_id}_{time.time()}",
                        user_id=user_id,
                        groups=groups,
                        permissions=permissions
                    )
                
                mock_session_manager.create_session.side_effect = create_session_side_effect
                mock_session_manager.validate_session.return_value = True
                mock_session_manager.update_session_activity.return_value = True
                
                # Benchmark single-threaded authentication
                single_thread_times = []
                for i in range(50):
                    start_time = time.time()
                    
                    user_id = f"user_{i}"
                    password = f"pass_{i}"
                    
                    auth_result = auth_manager.authenticate_user(user_id, password)
                    user_info = auth_manager.validate_token(auth_result["access_token"])
                    
                    end_time = time.time()
                    single_thread_times.append(end_time - start_time)
                
                # Benchmark multi-threaded authentication
                multi_thread_times = []
                
                def auth_benchmark_worker(user_num):
                    start_time = time.time()
                    
                    user_id = f"user_{user_num + 50}"  # Use different users
                    password = f"pass_{user_num + 50}"
                    
                    auth_result = auth_manager.authenticate_user(user_id, password)
                    user_info = auth_manager.validate_token(auth_result["access_token"])
                    
                    end_time = time.time()
                    return end_time - start_time
                
                with ThreadPoolExecutor(max_workers=10) as executor:
                    futures = [executor.submit(auth_benchmark_worker, i) for i in range(50)]
                    
                    for future in as_completed(futures):
                        multi_thread_times.append(future.result())
        
        # Calculate performance metrics
        single_thread_avg = statistics.mean(single_thread_times)
        single_thread_p95 = statistics.quantiles(single_thread_times, n=20)[18]  # 95th percentile
        
        multi_thread_avg = statistics.mean(multi_thread_times)
        multi_thread_p95 = statistics.quantiles(multi_thread_times, n=20)[18]
        
        performance_metrics = {
            "single_thread": {
                "average_time": single_thread_avg,
                "p95_time": single_thread_p95,
                "min_time": min(single_thread_times),
                "max_time": max(single_thread_times)
            },
            "multi_thread": {
                "average_time": multi_thread_avg,
                "p95_time": multi_thread_p95,
                "min_time": min(multi_thread_times),
                "max_time": max(multi_thread_times)
            }
        }
        
        # Performance assertions
        assert single_thread_avg < 0.1, f"Single-thread auth too slow: {single_thread_avg:.3f}s"
        assert single_thread_p95 < 0.2, f"Single-thread p95 too slow: {single_thread_p95:.3f}s"
        assert multi_thread_avg < 0.15, f"Multi-thread auth too slow: {multi_thread_avg:.3f}s"
        assert multi_thread_p95 < 0.3, f"Multi-thread p95 too slow: {multi_thread_p95:.3f}s"
        
        print(f"Authentication Performance Metrics: {performance_metrics}")
    
    def test_session_management_performance_benchmark(self):
        """Benchmark session management performance."""
        session_manager = SessionManager()
        performance_metrics = []
        
        with patch('app.shared.session_manager.redis_client') as mock_redis:
            # Mock Redis operations with realistic delays
            def mock_set_session_with_delay(session):
                time.sleep(0.001)  # 1ms Redis write delay
                return True
            
            def mock_get_session_with_delay(session_id):
                time.sleep(0.0005)  # 0.5ms Redis read delay
                return UserSession(
                    session_id=session_id,
                    user_id="test_user",
                    groups=["test_group"],
                    permissions=["upload", "query"]
                )
            
            mock_redis.set_session.side_effect = mock_set_session_with_delay
            mock_redis.get_session.side_effect = mock_get_session_with_delay
            mock_redis.delete_session.return_value = True
            
            # Benchmark session operations
            create_times = []
            get_times = []
            update_times = []
            
            # Test session creation performance
            for i in range(100):
                start_time = time.time()
                
                session = session_manager.create_session(
                    user_id=f"user_{i}",
                    groups=[f"group_{i}"],
                    permissions=["upload", "query"]
                )
                
                end_time = time.time()
                create_times.append(end_time - start_time)
            
            # Test session retrieval performance
            for i in range(100):
                start_time = time.time()
                
                session = session_manager.get_session(f"session_{i}")
                
                end_time = time.time()
                get_times.append(end_time - start_time)
            
            # Test session update performance
            for i in range(100):
                start_time = time.time()
                
                success = session_manager.update_session_activity(f"session_{i}")
                
                end_time = time.time()
                update_times.append(end_time - start_time)
        
        # Calculate metrics
        performance_metrics = {
            "create_session": {
                "average_time": statistics.mean(create_times),
                "p95_time": statistics.quantiles(create_times, n=20)[18],
                "operations_per_second": 1 / statistics.mean(create_times)
            },
            "get_session": {
                "average_time": statistics.mean(get_times),
                "p95_time": statistics.quantiles(get_times, n=20)[18],
                "operations_per_second": 1 / statistics.mean(get_times)
            },
            "update_session": {
                "average_time": statistics.mean(update_times),
                "p95_time": statistics.quantiles(update_times, n=20)[18],
                "operations_per_second": 1 / statistics.mean(update_times)
            }
        }
        
        # Performance assertions
        assert performance_metrics["create_session"]["average_time"] < 0.01
        assert performance_metrics["get_session"]["average_time"] < 0.005
        assert performance_metrics["update_session"]["average_time"] < 0.01
        
        assert performance_metrics["create_session"]["operations_per_second"] > 100
        assert performance_metrics["get_session"]["operations_per_second"] > 200
        
        print(f"Session Management Performance Metrics: {performance_metrics}")
    
    def test_job_processing_performance_benchmark(self):
        """Benchmark job processing performance."""
        job_manager = JobManager()
        performance_metrics = {}
        
        with patch('app.shared.job_manager.redis_client') as mock_redis:
            mock_redis.get_user_jobs.return_value = []
            mock_redis.set_job.return_value = True
            mock_redis.get_job.return_value = None
            
            # Benchmark job creation
            create_times = []
            for i in range(200):
                start_time = time.time()
                
                job = job_manager.create_job(
                    user_id=f"user_{i % 10}",  # 10 different users
                    job_type=JobType.EMBEDDING,
                    metadata={"document": f"doc_{i}.pdf"}
                )
                
                end_time = time.time()
                create_times.append(end_time - start_time)
            
            # Benchmark job status updates
            update_times = []
            for i in range(200):
                mock_job = Job(
                    user_id=f"user_{i % 10}",
                    job_type=JobType.EMBEDDING,
                    status=JobStatus.PENDING
                )
                mock_redis.get_job.return_value = mock_job
                
                start_time = time.time()
                
                success = job_manager.update_job_status(
                    mock_job.job_id,
                    JobStatus.PROCESSING,
                    result={"progress": 0.5}
                )
                
                end_time = time.time()
                update_times.append(end_time - start_time)
            
            # Benchmark concurrent job operations
            concurrent_times = []
            
            def concurrent_job_worker():
                start_time = time.time()
                
                # Create job
                job = job_manager.create_job(
                    user_id="concurrent_user",
                    job_type=JobType.QUERY,
                    metadata={"query": "test query"}
                )
                
                # Update job
                mock_job = Job(
                    user_id="concurrent_user",
                    job_type=JobType.QUERY,
                    status=JobStatus.PENDING
                )
                mock_redis.get_job.return_value = mock_job
                
                job_manager.update_job_status(job.job_id, JobStatus.COMPLETED)
                
                end_time = time.time()
                return end_time - start_time
            
            with ThreadPoolExecutor(max_workers=20) as executor:
                futures = [executor.submit(concurrent_job_worker) for _ in range(100)]
                
                for future in as_completed(futures):
                    concurrent_times.append(future.result())
        
        # Calculate metrics
        performance_metrics = {
            "job_creation": {
                "average_time": statistics.mean(create_times),
                "p95_time": statistics.quantiles(create_times, n=20)[18],
                "operations_per_second": 1 / statistics.mean(create_times)
            },
            "job_updates": {
                "average_time": statistics.mean(update_times),
                "p95_time": statistics.quantiles(update_times, n=20)[18],
                "operations_per_second": 1 / statistics.mean(update_times)
            },
            "concurrent_operations": {
                "average_time": statistics.mean(concurrent_times),
                "p95_time": statistics.quantiles(concurrent_times, n=20)[18],
                "operations_per_second": 1 / statistics.mean(concurrent_times)
            }
        }
        
        # Performance assertions
        assert performance_metrics["job_creation"]["average_time"] < 0.01
        assert performance_metrics["job_updates"]["average_time"] < 0.01
        assert performance_metrics["concurrent_operations"]["average_time"] < 0.05
        
        assert performance_metrics["job_creation"]["operations_per_second"] > 100
        assert performance_metrics["job_updates"]["operations_per_second"] > 100
        
        print(f"Job Processing Performance Metrics: {performance_metrics}")
    
    def test_resource_monitoring_performance_benchmark(self):
        """Benchmark resource monitoring performance."""
        resource_manager = ResourceManager()
        performance_metrics = {}
        
        with patch('app.shared.resource_manager.psutil') as mock_psutil, \
             patch('app.shared.resource_manager.redis_client') as mock_redis:
            
            # Mock system metrics with realistic delays
            def mock_cpu_percent():
                time.sleep(0.001)  # 1ms to get CPU info
                return 50.0
            
            def mock_virtual_memory():
                time.sleep(0.0005)  # 0.5ms to get memory info
                return Mock(percent=60.0, available=2*1024*1024*1024)
            
            def mock_disk_usage(path):
                time.sleep(0.001)  # 1ms to get disk info
                return Mock(percent=70.0, free=10*1024*1024*1024)
            
            mock_psutil.cpu_percent.side_effect = mock_cpu_percent
            mock_psutil.virtual_memory.side_effect = mock_virtual_memory
            mock_psutil.disk_usage.side_effect = mock_disk_usage
            mock_psutil.getloadavg.return_value = [1.0, 1.2, 1.1]
            mock_psutil.net_connections.return_value = []
            
            mock_redis.client.keys.return_value = []
            mock_redis.get_json.return_value = None
            mock_redis.set_json.return_value = True
            
            # Benchmark metrics collection
            collection_times = []
            for i in range(100):
                start_time = time.time()
                
                metrics = resource_manager.collect_metrics()
                
                end_time = time.time()
                collection_times.append(end_time - start_time)
            
            # Benchmark resource checks
            check_times = []
            for i in range(200):
                start_time = time.time()
                
                can_accept = resource_manager.can_accept_task("embedding")
                
                end_time = time.time()
                check_times.append(end_time - start_time)
            
            # Benchmark concurrent monitoring
            concurrent_times = []
            
            def concurrent_monitoring_worker():
                start_time = time.time()
                
                # Collect metrics
                metrics = resource_manager.collect_metrics()
                
                # Check resource availability
                can_accept_embedding = resource_manager.can_accept_task("embedding")
                can_accept_query = resource_manager.can_accept_task("query")
                
                end_time = time.time()
                return end_time - start_time
            
            with ThreadPoolExecutor(max_workers=10) as executor:
                futures = [executor.submit(concurrent_monitoring_worker) for _ in range(50)]
                
                for future in as_completed(futures):
                    concurrent_times.append(future.result())
        
        # Calculate metrics
        performance_metrics = {
            "metrics_collection": {
                "average_time": statistics.mean(collection_times),
                "p95_time": statistics.quantiles(collection_times, n=20)[18],
                "operations_per_second": 1 / statistics.mean(collection_times)
            },
            "resource_checks": {
                "average_time": statistics.mean(check_times),
                "p95_time": statistics.quantiles(check_times, n=20)[18],
                "operations_per_second": 1 / statistics.mean(check_times)
            },
            "concurrent_monitoring": {
                "average_time": statistics.mean(concurrent_times),
                "p95_time": statistics.quantiles(concurrent_times, n=20)[18],
                "operations_per_second": 1 / statistics.mean(concurrent_times)
            }
        }
        
        # Performance assertions
        assert performance_metrics["metrics_collection"]["average_time"] < 0.01
        assert performance_metrics["resource_checks"]["average_time"] < 0.005
        assert performance_metrics["concurrent_monitoring"]["average_time"] < 0.02
        
        assert performance_metrics["metrics_collection"]["operations_per_second"] > 100
        assert performance_metrics["resource_checks"]["operations_per_second"] > 200
        
        print(f"Resource Monitoring Performance Metrics: {performance_metrics}")


class TestLoadTesting:
    """Load testing under various scenarios."""
    
    def test_high_concurrency_load_test(self):
        """Test system under high concurrency load."""
        # Test parameters
        num_users = 100
        operations_per_user = 10
        max_workers = 20
        
        results = {
            "successful_operations": 0,
            "failed_operations": 0,
            "total_time": 0,
            "average_response_time": 0,
            "errors": []
        }
        
        operation_times = []
        
        def load_test_worker(user_id, operation_id):
            try:
                start_time = time.time()
                
                # Simulate complex operation (auth + job creation + status check)
                with patch('app.shared.auth.config') as mock_config, \
                     patch('app.shared.auth.session_manager') as mock_session_manager, \
                     patch('app.shared.job_manager.redis_client') as mock_job_redis:
                    
                    # Setup mocks
                    mock_config.SECRET_KEY = "test-secret"
                    mock_config.USERS = {user_id: {"password": "pass", "groups": ["group"]}}
                    
                    mock_session = UserSession(
                        session_id=f"session_{user_id}_{operation_id}",
                        user_id=user_id,
                        groups=["group"],
                        permissions=["upload", "query"]
                    )
                    mock_session_manager.create_session.return_value = mock_session
                    mock_session_manager.validate_session.return_value = True
                    
                    mock_job_redis.get_user_jobs.return_value = []
                    mock_job_redis.set_job.return_value = True
                    
                    # Perform operations
                    auth_manager = AuthenticationManager()
                    job_manager = JobManager()
                    
                    # Authenticate
                    auth_result = auth_manager.authenticate_user(user_id, "pass")
                    
                    # Create job
                    job = job_manager.create_job(
                        user_id=user_id,
                        job_type=JobType.EMBEDDING,
                        metadata={"operation_id": operation_id}
                    )
                    
                    # Update job status
                    mock_job = Job(user_id=user_id, job_type=JobType.EMBEDDING, status=JobStatus.PENDING)
                    mock_job_redis.get_job.return_value = mock_job
                    
                    job_manager.update_job_status(job.job_id, JobStatus.COMPLETED)
                    
                    end_time = time.time()
                    operation_time = end_time - start_time
                    
                    operation_times.append(operation_time)
                    results["successful_operations"] += 1
                    
            except Exception as e:
                results["failed_operations"] += 1
                results["errors"].append(f"User {user_id}, Op {operation_id}: {str(e)}")
        
        # Run load test
        start_time = time.time()
        
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = []
            
            for user_num in range(num_users):
                for op_num in range(operations_per_user):
                    user_id = f"load_user_{user_num}"
                    future = executor.submit(load_test_worker, user_id, op_num)
                    futures.append(future)
            
            # Wait for all operations to complete
            for future in as_completed(futures):
                future.result()
        
        end_time = time.time()
        results["total_time"] = end_time - start_time
        
        # Calculate metrics
        if operation_times:
            results["average_response_time"] = statistics.mean(operation_times)
            results["p95_response_time"] = statistics.quantiles(operation_times, n=20)[18]
            results["p99_response_time"] = statistics.quantiles(operation_times, n=100)[98]
        
        total_operations = num_users * operations_per_user
        results["throughput"] = total_operations / results["total_time"]
        results["success_rate"] = results["successful_operations"] / total_operations
        
        # Performance assertions
        assert results["success_rate"] > 0.95, f"Success rate too low: {results['success_rate']:.2%}"
        assert results["average_response_time"] < 0.1, f"Average response time too high: {results['average_response_time']:.3f}s"
        assert results["p95_response_time"] < 0.2, f"P95 response time too high: {results['p95_response_time']:.3f}s"
        assert results["throughput"] > 100, f"Throughput too low: {results['throughput']:.1f} ops/sec"
        
        print(f"Load Test Results: {results}")
    
    def test_memory_pressure_load_test(self):
        """Test system behavior under memory pressure."""
        resource_manager = ResourceManager()
        results = {
            "operations_under_normal_load": 0,
            "operations_under_high_load": 0,
            "operations_under_critical_load": 0,
            "rejections_under_critical_load": 0
        }
        
        with patch('app.shared.resource_manager.psutil') as mock_psutil, \
             patch('app.shared.resource_manager.redis_client') as mock_redis:
            
            mock_redis.client.keys.return_value = []
            mock_redis.get_json.return_value = None
            mock_redis.set_json.return_value = True
            
            # Test under different memory pressure levels
            memory_levels = [
                (50.0, "normal"),    # Normal load
                (80.0, "high"),      # High load
                (95.0, "critical")   # Critical load
            ]
            
            for memory_usage, load_level in memory_levels:
                mock_psutil.virtual_memory.return_value = Mock(
                    percent=memory_usage,
                    available=max(256*1024*1024, 8*1024*1024*1024 * (100-memory_usage)/100)
                )
                mock_psutil.cpu_percent.return_value = memory_usage * 0.8
                mock_psutil.disk_usage.return_value = Mock(percent=50.0)
                mock_psutil.getloadavg.return_value = [memory_usage/100*4] * 3
                
                # Test task acceptance under this load level
                accepted_tasks = 0
                rejected_tasks = 0
                
                for _ in range(50):
                    if resource_manager.can_accept_task("embedding"):
                        accepted_tasks += 1
                    else:
                        rejected_tasks += 1
                
                if load_level == "normal":
                    results["operations_under_normal_load"] = accepted_tasks
                    assert accepted_tasks > 40, f"Too many rejections under normal load: {rejected_tasks}"
                elif load_level == "high":
                    results["operations_under_high_load"] = accepted_tasks
                    # Some rejections expected under high load
                elif load_level == "critical":
                    results["operations_under_critical_load"] = accepted_tasks
                    results["rejections_under_critical_load"] = rejected_tasks
                    assert rejected_tasks > 40, f"Not enough rejections under critical load: {rejected_tasks}"
        
        print(f"Memory Pressure Test Results: {results}")
    
    def test_sustained_load_test(self):
        """Test system under sustained load over time."""
        duration_seconds = 10  # 10 second test
        operations_per_second = 50
        
        results = {
            "total_operations": 0,
            "successful_operations": 0,
            "failed_operations": 0,
            "response_times": [],
            "errors": []
        }
        
        def sustained_load_worker():
            try:
                start_time = time.time()
                
                # Simulate lightweight operation
                with patch('app.shared.session_manager.redis_client') as mock_redis:
                    mock_redis.set_session.return_value = True
                    mock_redis.get_session.return_value = UserSession(
                        session_id="test_session",
                        user_id="test_user",
                        groups=["test_group"],
                        permissions=["upload", "query"]
                    )
                    
                    session_manager = SessionManager()
                    
                    # Create and retrieve session
                    session = session_manager.create_session(
                        user_id="sustained_user",
                        groups=["test_group"],
                        permissions=["upload", "query"]
                    )
                    
                    retrieved_session = session_manager.get_session(session.session_id)
                    
                    end_time = time.time()
                    response_time = end_time - start_time
                    
                    results["response_times"].append(response_time)
                    results["successful_operations"] += 1
                    
            except Exception as e:
                results["failed_operations"] += 1
                results["errors"].append(str(e))
            
            results["total_operations"] += 1
        
        # Run sustained load test
        start_time = time.time()
        
        with ThreadPoolExecutor(max_workers=10) as executor:
            futures = []
            
            while time.time() - start_time < duration_seconds:
                # Submit operations at target rate
                for _ in range(operations_per_second):
                    future = executor.submit(sustained_load_worker)
                    futures.append(future)
                
                time.sleep(1)  # Wait 1 second before next batch
            
            # Wait for all operations to complete
            for future in as_completed(futures):
                future.result()
        
        # Calculate final metrics
        if results["response_times"]:
            results["average_response_time"] = statistics.mean(results["response_times"])
            results["p95_response_time"] = statistics.quantiles(results["response_times"], n=20)[18]
        
        results["success_rate"] = results["successful_operations"] / results["total_operations"]
        results["actual_throughput"] = results["total_operations"] / duration_seconds
        
        # Performance assertions
        assert results["success_rate"] > 0.98, f"Success rate degraded: {results['success_rate']:.2%}"
        assert results["average_response_time"] < 0.05, f"Response time degraded: {results['average_response_time']:.3f}s"
        assert results["actual_throughput"] > operations_per_second * 0.8, f"Throughput too low: {results['actual_throughput']:.1f} ops/sec"
        
        print(f"Sustained Load Test Results: {results}")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])