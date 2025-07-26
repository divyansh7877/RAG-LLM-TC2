"""
Performance tests and benchmarks for the concurrent RAG optimization system.
"""
import pytest
import time
import tempfile
import os
import shutil
import threading
import statistics
from concurrent.futures import ThreadPoolExecutor, as_completed
from unittest.mock import Mock, patch, MagicMock
from typing import List, Dict, Any, Tuple

from app.workers.embedding_worker import (
    create_nodes_from_pdfs_batch,
    record_embedding_performance_metrics,
    get_embedding_performance_stats,
    process_document_embedding
)
from app.workers.query_worker import (
    process_user_query,
    record_query_performance_metrics,
    get_query_performance_stats
)
from app.shared.query_engine_factory import (
    ThreadSafeQueryEngineFactory,
    DatabaseConnectionPool,
    QueryResultCache
)
from app.shared.redis_client import redis_client


class TestEmbeddingPerformance:
    """Performance tests for document embedding operations."""
    
    def setup_method(self):
        """Set up test environment."""
        self.test_user_id = "perf_test_user"
        self.test_group_id = "perf_test_group"
        
        # Clear Redis test data
        try:
            redis_client.redis_client.flushdb()
        except:
            pass
    
    def teardown_method(self):
        """Clean up test environment."""
        try:
            redis_client.redis_client.flushdb()
        except:
            pass
    
    def create_mock_pdf_files(self, count: int, size_kb: int = 100) -> List[str]:
        """Create mock PDF files for testing."""
        temp_dir = tempfile.mkdtemp()
        file_paths = []
        
        for i in range(count):
            file_path = os.path.join(temp_dir, f"test_doc_{i}.pdf")
            # Create a file with specified size
            content = b"Mock PDF content " * (size_kb * 1024 // 20)
            with open(file_path, "wb") as f:
                f.write(content)
            file_paths.append(file_path)
        
        return file_paths, temp_dir
    
    @patch('app.workers.embedding_worker.extract_pages_from_pdf')
    @patch('app.workers.embedding_worker.check_duplicate_document')
    def test_batch_processing_performance(self, mock_check_dup, mock_extract):
        """Test performance of batch processing vs individual processing."""
        # Mock PDF extraction
        mock_extract.return_value = [("Sample text content", 1)]
        mock_check_dup.return_value = False
        
        file_paths, temp_dir = self.create_mock_pdf_files(20, 50)
        
        try:
            # Test different batch sizes
            batch_sizes = [1, 5, 10, 20]
            results = {}
            
            for batch_size in batch_sizes:
                start_time = time.time()
                
                nodes, processed, failed = create_nodes_from_pdfs_batch(
                    file_paths, self.test_user_id, self.test_group_id, batch_size
                )
                
                processing_time = time.time() - start_time
                results[batch_size] = {
                    "processing_time": processing_time,
                    "nodes_created": len(nodes),
                    "files_processed": len(processed),
                    "files_failed": len(failed)
                }
                
                print(f"Batch size {batch_size}: {processing_time:.2f}s, {len(nodes)} nodes")
            
            # Verify that larger batch sizes are generally more efficient
            # (allowing some variance due to test environment)
            assert results[1]["processing_time"] >= results[5]["processing_time"] * 0.8
            assert all(r["files_failed"] == 0 for r in results.values())
            assert all(r["nodes_created"] > 0 for r in results.values())
            
        finally:
            shutil.rmtree(temp_dir, ignore_errors=True)
    
    def test_embedding_performance_metrics_recording(self):
        """Test embedding performance metrics recording and aggregation."""
        job_id = "test_job_123"
        
        # Record multiple performance metrics
        test_metrics = [
            (job_id + "_1", 5, 10.5, 150, 3, True, None),
            (job_id + "_2", 3, 8.2, 90, 2, True, None),
            (job_id + "_3", 8, 25.0, 240, 4, False, "Processing error"),
            (job_id + "_4", 2, 4.1, 60, 1, True, None),
        ]
        
        for job_id, file_count, proc_time, chunks, batch_size, success, error in test_metrics:
            record_embedding_performance_metrics(
                job_id, self.test_user_id, file_count, proc_time, chunks, batch_size, success, error
            )
        
        # Get performance statistics
        stats = get_embedding_performance_stats(days=1)
        
        # Verify statistics
        assert stats["summary"]["total_jobs"] == 4
        assert stats["summary"]["successful_jobs"] == 3
        assert stats["summary"]["total_files"] == 18  # 5+3+8+2
        assert stats["summary"]["total_chunks"] == 540  # 150+90+240+60
        assert stats["summary"]["success_rate"] == 0.75  # 3/4
        assert stats["summary"]["avg_chunks_per_file"] == 30.0  # 540/18
        
        # Check for slow processing detection
        slow_jobs = [m for m in test_metrics if m[2] / m[1] > 30.0]  # avg > 30s per file
        assert stats["summary"]["slow_jobs"] == len(slow_jobs)
    
    @patch('app.workers.embedding_worker.HuggingFaceEmbedding')
    @patch('app.workers.embedding_worker.lancedb')
    @patch('app.workers.embedding_worker.VectorStoreIndex')
    def test_concurrent_embedding_jobs_performance(self, mock_index, mock_lancedb, mock_embedding):
        """Test performance of concurrent embedding jobs."""
        # Mock dependencies
        mock_embedding.return_value = Mock()
        mock_db = Mock()
        mock_table = Mock()
        mock_table.count_rows.return_value = 1000
        mock_db.open_table.return_value = mock_table
        mock_lancedb.connect.return_value = mock_db
        mock_index.return_value = Mock()
        
        # Create test files
        file_paths, temp_dir = self.create_mock_pdf_files(5, 30)
        
        try:
            with patch('app.workers.embedding_worker.extract_pages_from_pdf') as mock_extract, \
                 patch('app.workers.embedding_worker.check_duplicate_document') as mock_check_dup, \
                 patch('app.workers.embedding_worker.job_manager') as mock_job_manager:
                
                mock_extract.return_value = [("Sample content", 1)]
                mock_check_dup.return_value = False
                mock_job_manager.update_job_status.return_value = True
                mock_job_manager.update_job_progress.return_value = True
                
                # Test concurrent processing
                def run_embedding_job(job_id: str) -> Tuple[str, float, bool]:
                    start_time = time.time()
                    try:
                        result = process_document_embedding(
                            job_id, f"user_{job_id}", self.test_group_id, file_paths
                        )
                        processing_time = time.time() - start_time
                        return job_id, processing_time, True
                    except Exception as e:
                        processing_time = time.time() - start_time
                        return job_id, processing_time, False
                
                # Run jobs concurrently
                num_concurrent_jobs = 3
                with ThreadPoolExecutor(max_workers=num_concurrent_jobs) as executor:
                    futures = [
                        executor.submit(run_embedding_job, f"job_{i}")
                        for i in range(num_concurrent_jobs)
                    ]
                    
                    results = [future.result() for future in as_completed(futures)]
                
                # Analyze results
                processing_times = [r[1] for r in results]
                success_count = sum(1 for r in results if r[2])
                
                print(f"Concurrent embedding jobs:")
                print(f"  Success rate: {success_count}/{num_concurrent_jobs}")
                print(f"  Avg processing time: {statistics.mean(processing_times):.2f}s")
                print(f"  Processing time range: {min(processing_times):.2f}s - {max(processing_times):.2f}s")
                
                # Verify all jobs completed successfully
                assert success_count == num_concurrent_jobs
                assert all(t > 0 for t in processing_times)
                
        finally:
            shutil.rmtree(temp_dir, ignore_errors=True)


class TestQueryPerformance:
    """Performance tests for query processing operations."""
    
    def setup_method(self):
        """Set up test environment."""
        self.test_user_id = "perf_query_user"
        self.test_group_ids = ["perf_group1", "perf_group2"]
        
        # Clear Redis test data
        try:
            redis_client.redis_client.flushdb()
        except:
            pass
    
    def teardown_method(self):
        """Clean up test environment."""
        try:
            redis_client.redis_client.flushdb()
        except:
            pass
    
    def test_query_performance_metrics_recording(self):
        """Test query performance metrics recording and aggregation."""
        # Record multiple query performance metrics
        test_queries = [
            ("query_1", "What is AI?", 1.2, 5, False, None),
            ("query_2", "Explain machine learning", 0.3, 3, True, None),  # Cached
            ("query_3", "Deep learning concepts", 2.8, 8, False, None),
            ("query_4", "Neural networks", 0.5, 4, True, None),  # Cached
            ("query_5", "Complex query about algorithms", 6.2, 2, False, None),  # Slow query
        ]
        
        for query_id, query_text, proc_time, result_count, cached, error in test_queries:
            record_query_performance_metrics(
                query_id, self.test_user_id, query_text, proc_time, result_count, cached, error
            )
        
        # Get performance statistics
        stats = get_query_performance_stats(days=1)
        
        # Verify statistics
        assert stats["summary"]["total_queries"] == 5
        assert stats["summary"]["successful_queries"] == 5
        assert stats["summary"]["cached_queries"] == 2
        assert stats["summary"]["slow_queries"] == 1  # query_5 > 5.0s threshold
        assert stats["summary"]["cache_hit_rate"] == 0.4  # 2/5
        assert stats["summary"]["success_rate"] == 1.0  # 5/5
        assert stats["summary"]["slow_query_rate"] == 0.2  # 1/5
        
        # Check average processing time
        expected_avg = sum(q[2] for q in test_queries) / len(test_queries)
        assert abs(stats["summary"]["avg_processing_time"] - expected_avg) < 0.01
    
    @patch('app.workers.query_worker.query_engine_factory')
    @patch('app.workers.query_worker.job_manager')
    def test_query_caching_performance_impact(self, mock_job_manager, mock_factory):
        """Test the performance impact of query result caching."""
        mock_job_manager.update_job_status.return_value = True
        mock_job_manager.update_job_progress.return_value = True
        
        # Mock query engine and response
        mock_engine = Mock()
        mock_response = Mock()
        mock_response.response = "This is a test answer"
        mock_response.source_nodes = []
        mock_engine.query.return_value = mock_response
        mock_factory.create_query_engine.return_value = mock_engine
        
        query_text = "What is the main topic?"
        
        # First query (no cache)
        mock_factory.get_cached_query_result.return_value = None
        
        start_time = time.time()
        result1 = process_user_query(
            "query_1", self.test_user_id, self.test_group_ids, query_text
        )
        first_query_time = time.time() - start_time
        
        # Verify caching was called
        mock_factory.cache_query_result.assert_called_once()
        
        # Second query (cached)
        cached_result = {
            "answer": "This is a test answer",
            "sources": [],
            "processing_time": first_query_time,
            "cached": False
        }
        mock_factory.get_cached_query_result.return_value = cached_result
        
        start_time = time.time()
        result2 = process_user_query(
            "query_2", self.test_user_id, self.test_group_ids, query_text
        )
        cached_query_time = time.time() - start_time
        
        # Verify cached query is significantly faster
        print(f"First query time: {first_query_time:.3f}s")
        print(f"Cached query time: {cached_query_time:.3f}s")
        print(f"Cache speedup: {first_query_time / cached_query_time:.1f}x")
        
        assert result2["cached"] is True
        assert cached_query_time < first_query_time * 0.5  # At least 2x faster
    
    @patch('app.workers.query_worker.query_engine_factory')
    @patch('app.workers.query_worker.job_manager')
    def test_concurrent_query_processing_performance(self, mock_job_manager, mock_factory):
        """Test performance of concurrent query processing."""
        mock_job_manager.update_job_status.return_value = True
        mock_job_manager.update_job_progress.return_value = True
        
        # Mock query engine with realistic processing delay
        def mock_query_with_delay(query_text):
            time.sleep(0.1)  # Simulate processing time
            mock_response = Mock()
            mock_response.response = f"Answer for: {query_text}"
            mock_response.source_nodes = []
            return mock_response
        
        mock_engine = Mock()
        mock_engine.query.side_effect = mock_query_with_delay
        mock_factory.create_query_engine.return_value = mock_engine
        mock_factory.get_cached_query_result.return_value = None
        
        # Test queries
        test_queries = [
            f"Query {i}: What is topic {i}?" for i in range(10)
        ]
        
        def run_query(query_id: str, query_text: str) -> Tuple[str, float, bool]:
            start_time = time.time()
            try:
                result = process_user_query(
                    query_id, f"user_{query_id}", self.test_group_ids, query_text
                )
                processing_time = time.time() - start_time
                return query_id, processing_time, True
            except Exception as e:
                processing_time = time.time() - start_time
                return query_id, processing_time, False
        
        # Test sequential processing
        sequential_start = time.time()
        sequential_results = []
        for i, query_text in enumerate(test_queries):
            result = run_query(f"seq_{i}", query_text)
            sequential_results.append(result)
        sequential_total_time = time.time() - sequential_start
        
        # Test concurrent processing
        concurrent_start = time.time()
        with ThreadPoolExecutor(max_workers=5) as executor:
            futures = [
                executor.submit(run_query, f"conc_{i}", query_text)
                for i, query_text in enumerate(test_queries)
            ]
            concurrent_results = [future.result() for future in as_completed(futures)]
        concurrent_total_time = time.time() - concurrent_start
        
        # Analyze results
        seq_success_count = sum(1 for r in sequential_results if r[2])
        conc_success_count = sum(1 for r in concurrent_results if r[2])
        
        print(f"Sequential processing:")
        print(f"  Total time: {sequential_total_time:.2f}s")
        print(f"  Success rate: {seq_success_count}/{len(test_queries)}")
        
        print(f"Concurrent processing:")
        print(f"  Total time: {concurrent_total_time:.2f}s")
        print(f"  Success rate: {conc_success_count}/{len(test_queries)}")
        print(f"  Speedup: {sequential_total_time / concurrent_total_time:.1f}x")
        
        # Verify concurrent processing is faster and successful
        assert conc_success_count == len(test_queries)
        assert seq_success_count == len(test_queries)
        assert concurrent_total_time < sequential_total_time * 0.8  # At least 25% faster


class TestQueryEngineFactoryPerformance:
    """Performance tests for the query engine factory."""
    
    def setup_method(self):
        """Set up test environment."""
        self.factory = ThreadSafeQueryEngineFactory()
    
    def test_connection_pool_performance(self):
        """Test database connection pool performance under load."""
        with patch('app.shared.query_engine_factory.lancedb') as mock_lancedb, \
             patch('app.shared.query_engine_factory.LanceDBVectorStore') as mock_vector_store:
            
            mock_db = Mock()
            mock_table = Mock()
            mock_db.open_table.return_value = mock_table
            mock_lancedb.connect.return_value = mock_db
            mock_vector_store.return_value = Mock()
            
            pool = DatabaseConnectionPool(
                db_path="test_db",
                table_name="test_table",
                max_connections=5
            )
            
            # Test concurrent connection access
            def get_connection_worker(worker_id: int) -> Tuple[int, float, bool]:
                start_time = time.time()
                try:
                    with pool.get_connection() as conn_data:
                        # Simulate work
                        time.sleep(0.05)
                        processing_time = time.time() - start_time
                        return worker_id, processing_time, True
                except Exception as e:
                    processing_time = time.time() - start_time
                    return worker_id, processing_time, False
            
            # Run concurrent workers
            num_workers = 20
            with ThreadPoolExecutor(max_workers=10) as executor:
                futures = [
                    executor.submit(get_connection_worker, i)
                    for i in range(num_workers)
                ]
                results = [future.result() for future in as_completed(futures)]
            
            # Analyze results
            processing_times = [r[1] for r in results]
            success_count = sum(1 for r in results if r[2])
            
            print(f"Connection pool performance:")
            print(f"  Success rate: {success_count}/{num_workers}")
            print(f"  Avg processing time: {statistics.mean(processing_times):.3f}s")
            print(f"  Max processing time: {max(processing_times):.3f}s")
            
            # Verify all connections succeeded
            assert success_count == num_workers
            assert max(processing_times) < 1.0  # Should not take too long to get connection
    
    def test_query_cache_performance(self):
        """Test query result cache performance under load."""
        cache = QueryResultCache(max_size=100)
        
        # Test data
        users = [f"user_{i}" for i in range(10)]
        groups = [["group1", "group2"], ["group2", "group3"], ["group1", "group3"]]
        queries = [f"Query {i} about topic {i}" for i in range(50)]
        
        # Populate cache
        populate_start = time.time()
        for i, query in enumerate(queries):
            user = users[i % len(users)]
            group = groups[i % len(groups)]
            result = {"answer": f"Answer {i}", "sources": [f"doc_{i}.pdf"]}
            cache.set(user, group, query, result)
        populate_time = time.time() - populate_start
        
        # Test cache retrieval performance
        def cache_retrieval_worker(worker_id: int) -> Tuple[int, int, float]:
            start_time = time.time()
            hits = 0
            
            # Perform multiple cache lookups
            for i in range(20):
                query_idx = (worker_id * 20 + i) % len(queries)
                user = users[query_idx % len(users)]
                group = groups[query_idx % len(groups)]
                query = queries[query_idx]
                
                result = cache.get(user, group, query)
                if result is not None:
                    hits += 1
            
            processing_time = time.time() - start_time
            return worker_id, hits, processing_time
        
        # Run concurrent cache access
        num_workers = 10
        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            futures = [
                executor.submit(cache_retrieval_worker, i)
                for i in range(num_workers)
            ]
            results = [future.result() for future in as_completed(futures)]
        
        # Analyze results
        total_hits = sum(r[1] for r in results)
        total_lookups = num_workers * 20
        processing_times = [r[2] for r in results]
        
        print(f"Cache performance:")
        print(f"  Populate time: {populate_time:.3f}s for {len(queries)} entries")
        print(f"  Hit rate: {total_hits}/{total_lookups} ({total_hits/total_lookups:.1%})")
        print(f"  Avg retrieval time: {statistics.mean(processing_times):.3f}s per worker")
        print(f"  Lookups per second: {total_lookups / sum(processing_times):.0f}")
        
        # Verify cache performance
        assert total_hits > total_lookups * 0.8  # At least 80% hit rate
        assert max(processing_times) < 0.5  # Fast retrieval


class TestSystemPerformanceBenchmarks:
    """System-wide performance benchmarks."""
    
    def setup_method(self):
        """Set up test environment."""
        # Clear Redis test data
        try:
            redis_client.redis_client.flushdb()
        except:
            pass
    
    def teardown_method(self):
        """Clean up test environment."""
        try:
            redis_client.redis_client.flushdb()
        except:
            pass
    
    @pytest.mark.slow
    def test_system_throughput_benchmark(self):
        """Benchmark overall system throughput with mixed workload."""
        # This test simulates a realistic mixed workload of embedding and query operations
        
        with patch('app.workers.embedding_worker.HuggingFaceEmbedding'), \
             patch('app.workers.embedding_worker.lancedb'), \
             patch('app.workers.embedding_worker.VectorStoreIndex'), \
             patch('app.workers.embedding_worker.extract_pages_from_pdf') as mock_extract, \
             patch('app.workers.embedding_worker.check_duplicate_document') as mock_check_dup, \
             patch('app.workers.query_worker.query_engine_factory') as mock_factory, \
             patch('app.workers.embedding_worker.job_manager') as mock_embed_job_mgr, \
             patch('app.workers.query_worker.job_manager') as mock_query_job_mgr:
            
            # Setup mocks
            mock_extract.return_value = [("Sample content", 1)]
            mock_check_dup.return_value = False
            mock_embed_job_mgr.update_job_status.return_value = True
            mock_embed_job_mgr.update_job_progress.return_value = True
            mock_query_job_mgr.update_job_status.return_value = True
            mock_query_job_mgr.update_job_progress.return_value = True
            
            # Mock query processing
            mock_engine = Mock()
            mock_response = Mock()
            mock_response.response = "Test answer"
            mock_response.source_nodes = []
            mock_engine.query.return_value = mock_response
            mock_factory.create_query_engine.return_value = mock_engine
            mock_factory.get_cached_query_result.return_value = None
            
            # Create test files
            file_paths = []
            temp_dir = tempfile.mkdtemp()
            try:
                for i in range(5):
                    file_path = os.path.join(temp_dir, f"test_doc_{i}.pdf")
                    with open(file_path, "wb") as f:
                        f.write(b"Mock PDF content " * 1000)
                    file_paths.append(file_path)
                
                # Mixed workload simulation
                def embedding_task(task_id: int) -> Tuple[str, float, bool]:
                    start_time = time.time()
                    try:
                        result = process_document_embedding(
                            f"embed_{task_id}", f"user_{task_id}", "test_group", file_paths[:2]
                        )
                        processing_time = time.time() - start_time
                        return f"embed_{task_id}", processing_time, True
                    except Exception as e:
                        processing_time = time.time() - start_time
                        return f"embed_{task_id}", processing_time, False
                
                def query_task(task_id: int) -> Tuple[str, float, bool]:
                    start_time = time.time()
                    try:
                        result = process_user_query(
                            f"query_{task_id}", f"user_{task_id}", ["test_group"], 
                            f"What is the topic of document {task_id}?"
                        )
                        processing_time = time.time() - start_time
                        return f"query_{task_id}", processing_time, True
                    except Exception as e:
                        processing_time = time.time() - start_time
                        return f"query_{task_id}", processing_time, False
                
                # Run mixed workload
                benchmark_start = time.time()
                with ThreadPoolExecutor(max_workers=8) as executor:
                    # Submit embedding tasks
                    embed_futures = [
                        executor.submit(embedding_task, i) for i in range(3)
                    ]
                    
                    # Submit query tasks
                    query_futures = [
                        executor.submit(query_task, i) for i in range(10)
                    ]
                    
                    # Collect results
                    all_futures = embed_futures + query_futures
                    results = [future.result() for future in as_completed(all_futures)]
                
                benchmark_total_time = time.time() - benchmark_start
                
                # Analyze results
                embed_results = [r for r in results if r[0].startswith("embed_")]
                query_results = [r for r in results if r[0].startswith("query_")]
                
                embed_success = sum(1 for r in embed_results if r[2])
                query_success = sum(1 for r in query_results if r[2])
                
                embed_avg_time = statistics.mean([r[1] for r in embed_results])
                query_avg_time = statistics.mean([r[1] for r in query_results])
                
                print(f"\nSystem Throughput Benchmark Results:")
                print(f"  Total benchmark time: {benchmark_total_time:.2f}s")
                print(f"  Embedding tasks: {embed_success}/{len(embed_results)} successful")
                print(f"  Query tasks: {query_success}/{len(query_results)} successful")
                print(f"  Avg embedding time: {embed_avg_time:.2f}s")
                print(f"  Avg query time: {query_avg_time:.2f}s")
                print(f"  Total throughput: {len(results) / benchmark_total_time:.1f} tasks/second")
                
                # Performance assertions
                assert embed_success == len(embed_results)
                assert query_success == len(query_results)
                assert benchmark_total_time < 60  # Should complete within 1 minute
                assert len(results) / benchmark_total_time > 0.2  # At least 0.2 tasks/second
                
            finally:
                shutil.rmtree(temp_dir, ignore_errors=True)
    
    def test_memory_usage_benchmark(self):
        """Benchmark memory usage during operations."""
        import psutil
        import gc
        
        process = psutil.Process()
        
        # Baseline memory
        gc.collect()
        baseline_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        # Test cache memory usage
        cache = QueryResultCache(max_size=1000)
        
        # Fill cache with test data
        for i in range(1000):
            user_id = f"user_{i % 10}"
            groups = [f"group_{i % 5}"]
            query = f"Query {i} with some content to test memory usage"
            result = {
                "answer": f"Answer {i} " * 50,  # Larger result
                "sources": [f"doc_{j}.pdf" for j in range(5)],
                "metadata": {"processing_time": 1.5, "chunks": 10}
            }
            cache.set(user_id, groups, query, result)
        
        gc.collect()
        cache_memory = process.memory_info().rss / 1024 / 1024  # MB
        cache_overhead = cache_memory - baseline_memory
        
        print(f"\nMemory Usage Benchmark:")
        print(f"  Baseline memory: {baseline_memory:.1f} MB")
        print(f"  Cache memory (1000 entries): {cache_memory:.1f} MB")
        print(f"  Cache overhead: {cache_overhead:.1f} MB")
        print(f"  Memory per cache entry: {cache_overhead / 1000 * 1024:.1f} KB")
        
        # Memory usage should be reasonable
        assert cache_overhead < 100  # Less than 100MB for 1000 entries
        assert cache_overhead / 1000 < 0.1  # Less than 100KB per entry


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])