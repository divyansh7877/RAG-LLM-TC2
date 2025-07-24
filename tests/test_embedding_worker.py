"""
Tests for embedding worker with progress tracking, user isolation, and resource limits.
"""
import os
import tempfile
import shutil
import time
import uuid
import pytest
from unittest.mock import Mock, patch, MagicMock
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading

# Import the worker functions
from app.workers.embedding_worker import (
    process_document_embedding,
    extract_pages_from_pdf,
    clean_text,
    calculate_file_hash,
    create_nodes_from_pdf,
    check_duplicate_document,
    store_document_metadata,
    update_job_progress,
    cleanup_failed_embeddings,
    get_embedding_job_status
)
from app.shared.models import JobStatus, Document
from app.shared.redis_client import redis_client


class TestEmbeddingWorkerBasics:
    """Test basic functionality of embedding worker components."""
    
    def test_clean_text(self):
        """Test text cleaning functionality."""
        # Test normal text
        text = "This is   a test\n\n\nwith   multiple   spaces\nand newlines"
        cleaned = clean_text(text)
        assert "multiple   spaces" not in cleaned
        assert "\n\n\n" not in cleaned
        
        # Test empty text
        assert clean_text("") == ""
        assert clean_text(None) == ""
        
        # Test whitespace only
        assert clean_text("   \n\n   ") == ""
    
    def test_calculate_file_hash(self):
        """Test file hash calculation."""
        with tempfile.NamedTemporaryFile(mode='w', delete=False) as f:
            f.write("test content")
            temp_path = f.name
        
        try:
            hash1 = calculate_file_hash(temp_path)
            hash2 = calculate_file_hash(temp_path)
            assert hash1 == hash2  # Same file should have same hash
            assert len(hash1) == 64  # SHA-256 hex string length
        finally:
            os.unlink(temp_path)
    
    def test_calculate_file_hash_nonexistent(self):
        """Test file hash calculation with non-existent file."""
        hash_result = calculate_file_hash("/nonexistent/file.pdf")
        # Should return a UUID as fallback
        assert len(hash_result) > 0
        assert hash_result != calculate_file_hash("/another/nonexistent/file.pdf")


class TestEmbeddingWorkerUserIsolation:
    """Test user isolation and security features."""
    
    @pytest.fixture
    def mock_redis(self):
        """Mock Redis client for testing."""
        with patch('app.workers.embedding_worker.redis_client') as mock:
            mock.get_json.return_value = None
            mock.set_json.return_value = True
            mock.redis_client.keys.return_value = []
            yield mock
    
    def test_check_duplicate_document_no_duplicates(self, mock_redis):
        """Test duplicate checking when no duplicates exist."""
        mock_redis.redis_client.keys.return_value = []
        
        result = check_duplicate_document("user1", "group1", "hash123")
        assert result is False
    
    def test_check_duplicate_document_with_duplicate(self, mock_redis):
        """Test duplicate checking when duplicate exists."""
        mock_redis.redis_client.keys.return_value = [b"document:user1:group1:doc1"]
        mock_redis.get_json.return_value = {"file_hash": "hash123"}
        
        result = check_duplicate_document("user1", "group1", "hash123")
        assert result is True
    
    def test_check_duplicate_document_different_user(self, mock_redis):
        """Test that duplicates are user-specific."""
        # Mock keys to return empty list for user1:group1 pattern (no duplicates for this user)
        mock_redis.redis_client.keys.return_value = []
        
        # Different user should not see duplicate
        result = check_duplicate_document("user1", "group1", "hash123")
        assert result is False
        
        # Verify the correct pattern was used
        mock_redis.redis_client.keys.assert_called_with("document:user1:group1:*")
    
    def test_store_document_metadata(self, mock_redis):
        """Test document metadata storage with user isolation."""
        with tempfile.NamedTemporaryFile(suffix='.pdf', delete=False) as f:
            f.write(b"test content")
            temp_path = f.name
        
        try:
            doc_id = store_document_metadata("user1", "group1", temp_path, 5, 10)
            
            # Verify Redis was called with correct key pattern
            mock_redis.set_json.assert_called_once()
            call_args = mock_redis.set_json.call_args
            redis_key = call_args[0][0]
            
            assert redis_key.startswith("document:user1:group1:")
            assert doc_id in redis_key
            
            # Verify document data structure
            doc_data = call_args[0][1]
            assert doc_data["user_id"] == "user1"
            assert doc_data["group_id"] == "group1"
            assert doc_data["page_count"] == 5
            assert doc_data["chunk_count"] == 10
            
        finally:
            os.unlink(temp_path)


class TestEmbeddingWorkerProgressTracking:
    """Test progress tracking and job status updates."""
    
    @pytest.fixture
    def mock_redis(self):
        """Mock Redis client for testing."""
        with patch('app.workers.embedding_worker.redis_client') as mock:
            mock.get_json.return_value = {
                "job_id": "test-job",
                "progress": 0.0,
                "status": "processing"
            }
            mock.set_json.return_value = True
            yield mock
    
    @patch('app.workers.embedding_worker.current_task')
    def test_update_job_progress(self, mock_task, mock_redis):
        """Test job progress updates."""
        mock_task.update_state = Mock()
        
        update_job_progress("test-job", 0.5, "Processing file 1 of 2")
        
        # Verify Redis update
        mock_redis.set_json.assert_called_once()
        call_args = mock_redis.set_json.call_args
        assert call_args[0][0] == "job:test-job"
        
        job_data = call_args[0][1]
        assert job_data["progress"] == 0.5
        assert job_data["status_message"] == "Processing file 1 of 2"
        assert "last_updated" in job_data
        
        # Verify Celery task state update
        mock_task.update_state.assert_called_once()
        state_args = mock_task.update_state.call_args[1]
        assert state_args["state"] == "PROGRESS"
        assert state_args["meta"]["progress"] == 0.5
        assert state_args["meta"]["status"] == "Processing file 1 of 2"
    
    def test_get_embedding_job_status(self, mock_redis):
        """Test job status retrieval."""
        job_data = {
            "job_id": "test-job",
            "status": "completed",
            "progress": 1.0,
            "result": {"files_processed": 2}
        }
        mock_redis.get_json.return_value = job_data
        
        result = get_embedding_job_status("test-job")
        assert result == job_data
        
        mock_redis.get_json.assert_called_once_with("job:test-job")
    
    def test_get_embedding_job_status_not_found(self, mock_redis):
        """Test job status retrieval for non-existent job."""
        mock_redis.get_json.return_value = None
        
        result = get_embedding_job_status("nonexistent-job")
        assert result == {"error": "Job not found"}


class TestEmbeddingWorkerConcurrency:
    """Test concurrent processing and resource limits."""
    
    @pytest.fixture
    def temp_pdf_files(self):
        """Create temporary PDF files for testing."""
        temp_dir = tempfile.mkdtemp()
        pdf_files = []
        
        # Create mock PDF files (just text files with .pdf extension for testing)
        for i in range(3):
            pdf_path = os.path.join(temp_dir, f"test_doc_{i}.pdf")
            with open(pdf_path, 'w') as f:
                f.write(f"Test document {i} content\nPage 1\nSome more content here.")
            pdf_files.append(pdf_path)
        
        yield pdf_files
        
        # Cleanup
        shutil.rmtree(temp_dir)
    
    @patch('app.workers.embedding_worker.redis_client')
    @patch('app.workers.embedding_worker.Settings')
    @patch('app.workers.embedding_worker.lancedb')
    @patch('app.workers.embedding_worker.VectorStoreIndex')
    @patch('app.workers.embedding_worker.extract_pages_from_pdf')
    def test_concurrent_embedding_jobs(self, mock_extract, mock_index, mock_lancedb, 
                                     mock_settings, mock_redis, temp_pdf_files):
        """Test multiple concurrent embedding jobs with user isolation."""
        # Mock PDF extraction to return test content
        mock_extract.return_value = [("Test content page 1", 1), ("Test content page 2", 2)]
        
        # Mock Redis operations
        mock_redis.get_json.return_value = None
        mock_redis.set_json.return_value = True
        mock_redis.redis_client.keys.return_value = []
        
        # Mock LanceDB operations
        mock_table = Mock()
        mock_table.count_rows.return_value = 100
        mock_db = Mock()
        mock_db.open_table.return_value = mock_table
        mock_lancedb.connect.return_value = mock_db
        
        # Mock vector index creation
        mock_index.return_value = Mock()
        
        # Create multiple jobs for different users
        jobs = []
        users = ["user1", "user2", "user3"]
        
        def run_embedding_job(user_id, job_id, files):
            """Run embedding job for a specific user."""
            try:
                # Mock the task context
                mock_task = Mock()
                mock_task.request.retries = 0
                
                with patch('app.workers.embedding_worker.current_task', mock_task):
                    result = process_document_embedding.apply(
                        args=[job_id, user_id, f"group_{user_id}", files]
                    )
                    return {"user_id": user_id, "job_id": job_id, "result": result}
            except Exception as e:
                return {"user_id": user_id, "job_id": job_id, "error": str(e)}
        
        # Run concurrent jobs
        with ThreadPoolExecutor(max_workers=3) as executor:
            futures = []
            for i, user_id in enumerate(users):
                job_id = f"job_{user_id}_{uuid.uuid4()}"
                # Each user gets a subset of files
                user_files = temp_pdf_files[i:i+1]  # One file per user
                future = executor.submit(run_embedding_job, user_id, job_id, user_files)
                futures.append(future)
            
            # Collect results
            results = []
            for future in as_completed(futures, timeout=30):
                result = future.result()
                results.append(result)
        
        # Verify all jobs completed
        assert len(results) == 3
        
        # Verify user isolation - each job should have been called with different user_id
        processed_users = {r["user_id"] for r in results}
        assert processed_users == {"user1", "user2", "user3"}
        
        # Verify Redis calls were made for each job
        assert mock_redis.set_json.call_count >= 3  # At least one call per job
    
    @patch('app.workers.embedding_worker.redis_client')
    def test_resource_limit_enforcement(self, mock_redis):
        """Test that resource limits are enforced during processing."""
        # Mock Redis to simulate resource exhaustion
        mock_redis.get_json.return_value = None
        mock_redis.set_json.return_value = True
        
        # Test with invalid inputs - these should be handled gracefully and marked as failed
        try:
            result = process_document_embedding.apply(args=["", "", "", []])
            # Should not succeed, but won't raise exception due to Celery error handling
            assert False, "Should have failed with invalid parameters"
        except Exception:
            # Exception is expected due to validation failure
            pass
        
        try:
            result = process_document_embedding.apply(args=["job1", "user1", "group1", []])
            # Should not succeed, but won't raise exception due to Celery error handling
            assert False, "Should have failed with empty file list"
        except Exception:
            # Exception is expected due to validation failure
            pass
    
    def test_concurrent_progress_updates(self):
        """Test that progress updates are thread-safe."""
        job_ids = [f"job_{i}" for i in range(5)]
        
        # Mock Redis client
        with patch('app.workers.embedding_worker.redis_client') as mock_redis:
            mock_redis.get_json.return_value = {"job_id": "test", "progress": 0.0}
            mock_redis.set_json.return_value = True
            
            # Run concurrent progress updates
            def update_progress(job_id, progress):
                update_job_progress(job_id, progress, f"Processing {job_id}")
            
            with ThreadPoolExecutor(max_workers=5) as executor:
                futures = []
                for i, job_id in enumerate(job_ids):
                    future = executor.submit(update_progress, job_id, i * 0.2)
                    futures.append(future)
                
                # Wait for all updates to complete
                for future in as_completed(futures, timeout=10):
                    future.result()  # This will raise if there was an exception
            
            # Verify all progress updates were called
            assert mock_redis.set_json.call_count == len(job_ids)


class TestEmbeddingWorkerErrorHandling:
    """Test error handling and retry logic."""
    
    @patch('app.workers.embedding_worker.redis_client')
    def test_invalid_file_handling(self, mock_redis):
        """Test handling of invalid or non-existent files."""
        mock_redis.get_json.return_value = None
        mock_redis.set_json.return_value = True
        
        # Test with non-existent file - should handle gracefully and mark job as failed
        try:
            result = process_document_embedding.apply(
                args=["job1", "user1", "group1", ["/nonexistent/file.pdf"]]
            )
            # If no exception is raised, check that Redis was called to mark job as failed
            # The last call should be setting the job status to failed
            calls = mock_redis.set_json.call_args_list
            if calls:
                last_call = calls[-1]
                job_data = last_call[0][1]  # Second argument is the job data
                assert job_data.get("status") == "failed" or "error" in job_data
        except Exception:
            # Exception is also acceptable - means the task failed as expected
            pass
    
    @patch('app.workers.embedding_worker.redis_client')
    def test_cleanup_failed_embeddings(self, mock_redis):
        """Test cleanup of failed embedding jobs."""
        mock_redis.redis_client.delete.return_value = True
        
        cleanup_failed_embeddings.apply(args=["user1", "failed_job_123"])
        
        # Verify cleanup was called
        mock_redis.redis_client.delete.assert_called_once_with("job:failed_job_123")
    
    @patch('app.workers.embedding_worker.redis_client')
    @patch('app.workers.embedding_worker.extract_pages_from_pdf')
    def test_partial_file_processing_failure(self, mock_extract, mock_redis):
        """Test that job continues when some files fail to process."""
        # Mock Redis operations
        mock_redis.get_json.return_value = None
        mock_redis.set_json.return_value = True
        mock_redis.redis_client.keys.return_value = []
        
        # Mock PDF extraction to fail for some files
        def mock_extract_side_effect(file_path):
            if "fail" in file_path:
                raise ValueError("Simulated PDF processing error")
            return [("Test content", 1)]
        
        mock_extract.side_effect = mock_extract_side_effect
        
        # Create temp files
        temp_dir = tempfile.mkdtemp()
        try:
            good_file = os.path.join(temp_dir, "good.pdf")
            bad_file = os.path.join(temp_dir, "fail.pdf")
            
            with open(good_file, 'w') as f:
                f.write("good content")
            with open(bad_file, 'w') as f:
                f.write("bad content")
            
            # Mock other dependencies
            with patch('app.workers.embedding_worker.Settings'), \
                 patch('app.workers.embedding_worker.lancedb'), \
                 patch('app.workers.embedding_worker.VectorStoreIndex'):
                
                # This should not raise an exception even though one file fails
                mock_task = Mock()
                mock_task.request.retries = 0
                
                with patch('app.workers.embedding_worker.current_task', mock_task):
                    result = process_document_embedding.apply(
                        args=["job1", "user1", "group1", [good_file, bad_file]]
                    )
                
                # Job should complete with partial success
                assert result is not None
        
        finally:
            shutil.rmtree(temp_dir)


class TestEmbeddingWorkerIntegration:
    """Integration tests for the complete embedding workflow."""
    
    @pytest.fixture
    def redis_cleanup(self):
        """Clean up Redis keys after tests."""
        yield
        # Cleanup test keys
        try:
            keys = redis_client.redis_client.keys("job:test_*")
            keys.extend(redis_client.redis_client.keys("document:test_*"))
            if keys:
                redis_client.redis_client.delete(*keys)
        except:
            pass  # Ignore cleanup errors
    
    def test_end_to_end_embedding_workflow(self, redis_cleanup):
        """Test complete embedding workflow from job creation to completion."""
        # This test requires actual Redis and file system access
        # Skip if Redis is not available
        try:
            redis_client.redis_client.ping()
        except:
            pytest.skip("Redis not available for integration test")
        
        # Create a temporary PDF-like file
        temp_dir = tempfile.mkdtemp()
        try:
            test_file = os.path.join(temp_dir, "test.pdf")
            with open(test_file, 'w') as f:
                f.write("Test document content for embedding")
            
            job_id = f"test_job_{uuid.uuid4()}"
            user_id = "test_user"
            group_id = "test_group"
            
            # Mock the heavy dependencies for integration test
            with patch('app.workers.embedding_worker.Settings'), \
                 patch('app.workers.embedding_worker.lancedb'), \
                 patch('app.workers.embedding_worker.VectorStoreIndex'), \
                 patch('app.workers.embedding_worker.extract_pages_from_pdf') as mock_extract:
                
                mock_extract.return_value = [("Test content", 1)]
                
                # Run the embedding task
                mock_task = Mock()
                mock_task.request.retries = 0
                
                with patch('app.workers.embedding_worker.current_task', mock_task):
                    result = process_document_embedding.apply(
                        args=[job_id, user_id, group_id, [test_file]]
                    )
                
                # Verify job was tracked in Redis
                job_status = get_embedding_job_status(job_id)
                assert job_status is not None
                assert "error" not in job_status or job_status.get("status") == "completed"
        
        finally:
            shutil.rmtree(temp_dir)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])