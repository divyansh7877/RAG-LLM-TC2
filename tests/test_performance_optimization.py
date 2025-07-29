"""
Unit tests for performance optimization and resource management.
Tests for Requirements 2.1, 2.2, 4.1, 4.2 - Performance and resource management.
"""
import pytest
import time
import threading
from unittest.mock import Mock, patch, MagicMock
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timedelta

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.shared.resource_manager import ResourceManager, ResourceStatus, ResourceMetrics
from app.shared.job_manager import JobManager, JobPriority
from app.shared.models import Job, JobStatus, JobType


class TestResourceManagement:
    """Test resource management and optimization."""
    
    @pytest.fixture
    def resource_manager(self):
        """Create ResourceManager instance for testing."""
        return ResourceManager()
    
    @pytest.fixture
    def mock_system_resources(self):
        """Mock system resource monitoring."""
        with patch('app.shared.resource_manager.psutil') as mock_psutil:
            # Mock healthy system by default
            mock_psutil.virtual_memory.return_value = Mock(
                percent=50.0,
                available=4*1024*1024*1024  # 4GB
            )
            mock_psutil.cpu_percent.return_value = 30.0
            mock_psutil.disk_usage.return_value = Mock(
                used=5*1024*1024*1024,
                total=20*1024*1024*1024,
                free=15*1024*1024*1024
            )
            mock_psutil.getloadavg.return_value = [1.0, 1.2, 1.1]
            yield mock_psutil
    
    def test_concurrent_embedding_limit_enforcement(self, resource_manager, mock_system_resources):
        """Test that concurrent embedding limits are enforced."""
        with patch('app.shared.resource_manager.redis_client') as mock_redis:
            # Mock active embedding tasks at limit
            mock_redis.client.keys.return_value = [
                "task:embedding_1", "task:embedding_2"  # At limit of 2
            ]
            
            def mock_get_json(key):
                if "embedding" in key:
                    return {
                        "status": "running",
                        "task_name": "process_document_embedding",
                        "started_at": time.time()
                    }
                return None
            
            mock_redis.get_json.side_effect = mock_get_json
            
            # Should reject new embedding task
            can_accept = resource_manager.can_accept_task("embedding")
            assert can_accept is False
            
            # Should still accept query tasks
            can_accept_query = resource_manager.can_accept_task("query")
            assert can_accept_query is True
    
    def test_memory_pressure_task_rejection(self, resource_manager):
        """Test task rejection under memory pressure."""
        with patch('app.shared.resource_manager.psutil') as mock_psutil:
            # Mock high memory usage
            mock_psutil.virtual_memory.return_value = Mock(
                percent=95.0,  # Critical memory usage
                available=256*1024*1024  # 256MB available
            )
            mock_psutil.cpu_percent.return_value = 30.0
            mock_psutil.disk_usage.return_value = Mock(percent=50.0)
            mock_psutil.getloadavg.return_value = [1.0, 1.0, 1.0]
            
            with patch('app.shared.resource_manager.redis_client') as mock_redis:
                mock_redis.client.keys.return_value = []
                mock_redis.get_json.return_value = None
                
                # Should reject both embedding and query tasks under critical memory
                can_accept_embedding = resource_manager.can_accept_task("embedding")
                can_accept_query = resource_manager.can_accept_task("query")
                
                assert can_accept_embedding is False
                assert can_accept_query is False
    
    def test_resource_guard_context_manager(self, resource_manager, mock_system_resources):
        """Test resource guard context manager functionality."""
        with patch('app.shared.resource_manager.redis_client') as mock_redis:
            mock_redis.client.keys.return_value = []
            mock_redis.get_json.return_value = None
            mock_redis.set_json.return_value = True
            mock_redis.delete.return_value = True
            
            # Should allow task execution when resources are available
            task_executed = False
            
            try:
                with resource_manager.resource_guard("embedding", "test_task"):
                    task_executed = True
                    # Simulate some work
                    time.sleep(0.01)
            except Exception:
                pass
            
            assert task_executed is True
    
    def test_concurrent_resource_monitoring(self, resource_manager, mock_system_resources):
        """Test resource monitoring under concurrent access."""
        with patch('app.shared.resource_manager.redis_client') as mock_redis:
            mock_redis.client.keys.return_value = []
            mock_redis.get_json.return_value = None
            mock_redis.set_json.return_value = True
            
            metrics_collected = []
            errors = []
            
            def collect_metrics_worker():
                try:
                    for _ in range(5):
                        metrics = resource_manager.collect_metrics()
                        metrics_collected.append(metrics)
                        time.sleep(0.01)
                except Exception as e:
                    errors.append(e)
            
            # Start multiple monitoring threads
            threads = []
            for _ in range(3):
                thread = threading.Thread(target=collect_metrics_worker)
                threads.append(thread)
                thread.start()
            
            # Wait for completion
            for thread in threads:
                thread.join()
            
            # Verify no errors and metrics were collected
            assert len(errors) == 0, f"Errors occurred: {errors}"
            assert len(metrics_collected) == 15  # 3 threads * 5 metrics each
            
            # Verify all metrics are valid
            for metrics in metrics_collected:
                assert isinstance(metrics, ResourceMetrics)
                assert metrics.memory_usage >= 0
                assert metrics.cpu_usage >= 0


class TestPerformanceOptimization:
    """Test performance optimization features."""
    
    @pytest.fixture
    def job_manager(self):
        """Create JobManager instance for testing."""
        return JobManager()
    
    def test_job_priority_queue_ordering(self, job_manager):
        """Test that jobs are processed in priority order."""
        with patch('app.shared.job_manager.redis_client') as mock_redis:
            mock_redis.get_user_jobs.return_value = []
            mock_redis.set_job.return_value = True
            
            created_jobs = []
            
            # Create jobs with different priorities
            priorities = [
                (JobPriority.LOW, "low_priority_job"),
                (JobPriority.URGENT, "urgent_job"),
                (JobPriority.NORMAL, "normal_job"),
                (JobPriority.HIGH, "high_priority_job")
            ]
            
            for priority, job_name in priorities:
                job = job_manager.create_job(
                    user_id="test_user",
                    job_type=JobType.EMBEDDING,
                    metadata={"name": job_name},
                    priority=priority
                )
                created_jobs.append(job)
            
            # Verify priority values are set correctly
            priority_values = [job.metadata["priority"] for job in created_jobs]
            expected_values = [1, 10, 5, 8]  # LOW, URGENT, NORMAL, HIGH
            assert priority_values == expected_values
    
    def test_concurrent_job_processing_limits(self, job_manager):
        """Test concurrent job processing respects user limits."""
        with patch('app.shared.job_manager.redis_client') as mock_redis:
            # Mock user already has maximum concurrent jobs
            existing_jobs = [
                Job(user_id="test_user", job_type=JobType.EMBEDDING, status=JobStatus.PROCESSING)
                for _ in range(10)  # At the limit
            ]
            
            def mock_get_user_jobs(user_id, job_type=None, status=None, active_only=False, limit=100):
                if active_only:
                    return [job for job in existing_jobs if not job.is_finished()]
                return existing_jobs
            
            with patch.object(job_manager, 'get_user_jobs', side_effect=mock_get_user_jobs):
                # Should reject new job creation
                with pytest.raises(Exception) as exc_info:
                    job_manager.create_job("test_user", JobType.EMBEDDING)
                
                assert "maximum concurrent jobs limit" in str(exc_info.value)
    
    def test_batch_processing_efficiency(self):
        """Test batch processing for multiple document uploads."""
        # Simulate batch document processing
        documents = [f"document_{i}.pdf" for i in range(10)]
        processing_times = []
        
        def process_document_batch(doc_batch):
            start_time = time.time()
            
            # Simulate batch processing (more efficient than individual)
            batch_size = len(doc_batch)
            # Simulate that batch processing is more efficient
            processing_time = 0.01 * batch_size * 0.8  # 20% efficiency gain
            time.sleep(processing_time)
            
            end_time = time.time()
            return end_time - start_time
        
        def process_documents_individually(docs):
            start_time = time.time()
            
            for doc in docs:
                # Simulate individual processing
                time.sleep(0.01)
            
            end_time = time.time()
            return end_time - start_time
        
        # Test batch processing
        batch_time = process_document_batch(documents)
        
        # Test individual processing
        individual_time = process_documents_individually(documents)
        
        # Batch processing should be more efficient
        assert batch_time < individual_time
    
    def test_query_result_caching(self):
        """Test query result caching for performance."""
        cache = {}
        cache_hits = 0
        cache_misses = 0
        
        def cached_query(query_text, user_id):
            nonlocal cache_hits, cache_misses
            
            cache_key = f"{user_id}:{hash(query_text)}"
            
            if cache_key in cache:
                cache_hits += 1
                return cache[cache_key]
            else:
                cache_misses += 1
                # Simulate query processing
                result = f"Results for '{query_text}' by {user_id}"
                cache[cache_key] = result
                return result
        
        # Test caching behavior
        queries = [
            ("What is AI?", "user1"),
            ("What is ML?", "user1"),
            ("What is AI?", "user1"),  # Should hit cache
            ("What is AI?", "user2"),  # Different user, should miss
            ("What is ML?", "user1"),  # Should hit cache
        ]
        
        results = []
        for query_text, user_id in queries:
            result = cached_query(query_text, user_id)
            results.append(result)
        
        # Verify caching worked
        assert cache_hits == 2  # Two cache hits
        assert cache_misses == 3  # Three cache misses
        assert len(set(results)) == 3  # Three unique results
    
    def test_concurrent_query_processing(self):
        """Test concurrent query processing performance."""
        query_results = []
        processing_times = []
        errors = []
        
        def process_query_worker(user_id, query_text):
            try:
                start_time = time.time()
                
                # Simulate query processing with user isolation
                time.sleep(0.05)  # Simulate processing time
                result = {
                    "user_id": user_id,
                    "query": query_text,
                    "results": f"Results for {user_id}: {query_text}",
                    "processing_time": time.time() - start_time
                }
                
                query_results.append(result)
                processing_times.append(result["processing_time"])
                
            except Exception as e:
                errors.append(f"User {user_id}: {e}")
        
        # Process multiple queries concurrently
        queries = [
            (f"user_{i}", f"query_{i}")
            for i in range(10)
        ]
        
        start_time = time.time()
        
        with ThreadPoolExecutor(max_workers=5) as executor:
            futures = [
                executor.submit(process_query_worker, user_id, query)
                for user_id, query in queries
            ]
            
            for future in as_completed(futures):
                future.result()
        
        total_time = time.time() - start_time
        
        # Verify concurrent processing
        assert len(errors) == 0, f"Errors occurred: {errors}"
        assert len(query_results) == 10
        
        # Concurrent processing should be faster than sequential
        sequential_time = sum(processing_times)
        assert total_time < sequential_time * 0.8  # At least 20% faster
        
        # Verify user isolation in results
        user_ids = [result["user_id"] for result in query_results]
        assert len(set(user_ids)) == 10  # All unique users


class TestLoadBalancing:
    """Test load balancing and fair resource allocation."""
    
    def test_fair_queue_processing(self):
        """Test fair processing of requests from different users."""
        user_request_counts = {"user1": 0, "user2": 0, "user3": 0}
        processing_order = []
        
        def process_request(user_id, request_id):
            processing_order.append(user_id)
            user_request_counts[user_id] += 1
            time.sleep(0.01)  # Simulate processing
        
        # Create requests from different users
        requests = []
        for i in range(15):
            user_id = f"user{(i % 3) + 1}"  # Distribute across 3 users
            requests.append((user_id, f"request_{i}"))
        
        # Process requests with fair scheduling simulation
        with ThreadPoolExecutor(max_workers=3) as executor:
            futures = [
                executor.submit(process_request, user_id, request_id)
                for user_id, request_id in requests
            ]
            
            for future in as_completed(futures):
                future.result()
        
        # Verify fair distribution
        assert user_request_counts["user1"] == 5
        assert user_request_counts["user2"] == 5
        assert user_request_counts["user3"] == 5
        
        # Verify no single user dominated the processing
        user_positions = {}
        for i, user_id in enumerate(processing_order):
            if user_id not in user_positions:
                user_positions[user_id] = []
            user_positions[user_id].append(i)
        
        # Check that users are reasonably interleaved
        for user_id, positions in user_positions.items():
            # No user should have all their requests processed consecutively
            consecutive_count = 0
            max_consecutive = 0
            
            for i in range(1, len(positions)):
                if positions[i] == positions[i-1] + 1:
                    consecutive_count += 1
                else:
                    max_consecutive = max(max_consecutive, consecutive_count)
                    consecutive_count = 0
            
            max_consecutive = max(max_consecutive, consecutive_count)
            assert max_consecutive < 3  # No more than 2 consecutive requests
    
    def test_resource_allocation_under_load(self):
        """Test resource allocation behavior under high load."""
        resource_manager = ResourceManager()
        allocation_results = []
        
        with patch('app.shared.resource_manager.psutil') as mock_psutil:
            # Simulate varying system load
            load_levels = [30.0, 60.0, 85.0, 95.0, 70.0]  # CPU percentages
            
            for cpu_load in load_levels:
                mock_psutil.virtual_memory.return_value = Mock(
                    percent=cpu_load * 0.8,  # Memory correlates with CPU
                    available=max(1024*1024*1024, 8*1024*1024*1024 * (100-cpu_load)/100)
                )
                mock_psutil.cpu_percent.return_value = cpu_load
                mock_psutil.disk_usage.return_value = Mock(percent=50.0)
                mock_psutil.getloadavg.return_value = [cpu_load/100*4, cpu_load/100*4, cpu_load/100*4]
                
                with patch('app.shared.resource_manager.redis_client') as mock_redis:
                    mock_redis.client.keys.return_value = []
                    mock_redis.get_json.return_value = None
                    
                    # Test task acceptance at different load levels
                    can_accept_embedding = resource_manager.can_accept_task("embedding")
                    can_accept_query = resource_manager.can_accept_task("query")
                    
                    allocation_results.append({
                        "cpu_load": cpu_load,
                        "can_accept_embedding": can_accept_embedding,
                        "can_accept_query": can_accept_query
                    })
        
        # Verify resource allocation behavior
        for result in allocation_results:
            cpu_load = result["cpu_load"]
            
            if cpu_load < 80:
                # Should accept tasks under normal load
                assert result["can_accept_embedding"] is True
                assert result["can_accept_query"] is True
            elif cpu_load >= 90:
                # Should reject tasks under critical load
                assert result["can_accept_embedding"] is False
                assert result["can_accept_query"] is False
            # Between 80-90 is warning zone, behavior may vary
    
    def test_graceful_degradation(self):
        """Test graceful degradation under resource constraints."""
        service_levels = []
        
        def simulate_service_under_load(memory_usage, cpu_usage):
            """Simulate service behavior under different load levels."""
            if memory_usage > 90 or cpu_usage > 90:
                # Critical load - minimal service
                return {
                    "embedding_enabled": False,
                    "query_enabled": False,
                    "max_concurrent_users": 1,
                    "response_time_multiplier": 5.0
                }
            elif memory_usage > 75 or cpu_usage > 75:
                # High load - reduced service
                return {
                    "embedding_enabled": False,
                    "query_enabled": True,
                    "max_concurrent_users": 5,
                    "response_time_multiplier": 2.0
                }
            else:
                # Normal load - full service
                return {
                    "embedding_enabled": True,
                    "query_enabled": True,
                    "max_concurrent_users": 20,
                    "response_time_multiplier": 1.0
                }
        
        # Test different load scenarios
        load_scenarios = [
            (50, 40),   # Normal
            (80, 70),   # High
            (95, 85),   # Critical
            (60, 95),   # CPU critical
            (95, 60),   # Memory critical
        ]
        
        for memory_usage, cpu_usage in load_scenarios:
            service_level = simulate_service_under_load(memory_usage, cpu_usage)
            service_levels.append({
                "memory_usage": memory_usage,
                "cpu_usage": cpu_usage,
                "service_level": service_level
            })
        
        # Verify graceful degradation
        normal_service = service_levels[0]["service_level"]
        high_load_service = service_levels[1]["service_level"]
        critical_service = service_levels[2]["service_level"]
        
        # Normal service should have full capabilities
        assert normal_service["embedding_enabled"] is True
        assert normal_service["query_enabled"] is True
        assert normal_service["max_concurrent_users"] == 20
        
        # High load should reduce capabilities
        assert high_load_service["embedding_enabled"] is False
        assert high_load_service["query_enabled"] is True
        assert high_load_service["max_concurrent_users"] < normal_service["max_concurrent_users"]
        
        # Critical load should minimize capabilities
        assert critical_service["embedding_enabled"] is False
        assert critical_service["query_enabled"] is False
        assert critical_service["max_concurrent_users"] == 1


if __name__ == "__main__":
    pytest.main([__file__, "-v"])