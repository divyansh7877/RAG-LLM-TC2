"""
Tests for the JobManager class and job lifecycle management.
"""
import pytest
import time
import threading
from datetime import datetime, timedelta
from unittest.mock import Mock, patch, MagicMock
from contextlib import contextmanager

from app.shared.job_manager import JobManager, JobPriority, JobManagerError
from app.shared.models import Job, JobStatus, JobType


class TestJobPriority:
    """Test JobPriority enumeration."""
    
    def test_priority_values(self):
        """Test priority enum values."""
        assert JobPriority.LOW.value == 1
        assert JobPriority.NORMAL.value == 5
        assert JobPriority.HIGH.value == 8
        assert JobPriority.URGENT.value == 10


class TestJobManagerError:
    """Test JobManagerError exception."""
    
    def test_job_manager_error(self):
        """Test JobManagerError creation."""
        error = JobManagerError("Test error")
        assert str(error) == "Test error"
        assert isinstance(error, Exception)


class TestJobManager:
    """Test JobManager functionality."""
    
    @pytest.fixture
    def manager(self):
        """Create a JobManager instance for testing."""
        return JobManager()
    
    @pytest.fixture
    def mock_redis(self):
        """Mock Redis client for testing."""
        with patch('app.shared.job_manager.redis_client') as mock:
            mock.set_job.return_value = True
            mock.get_job.return_value = None
            mock.delete_job.return_value = True
            mock.get_user_jobs.return_value = []
            mock.get_jobs_by_status.return_value = []
            yield mock
    
    @pytest.fixture
    def sample_job(self):
        """Create a sample job for testing."""
        return Job(
            user_id="test_user",
            job_type=JobType.EMBEDDING,
            status=JobStatus.PENDING,
            metadata={"test": "data"}
        )
    
    def test_initialization(self, manager):
        """Test JobManager initialization."""
        assert manager._lock is not None
        assert isinstance(manager._job_callbacks, dict)
        assert isinstance(manager._status_callbacks, dict)
        assert manager._cleanup_thread is None
        assert not manager._cleanup_active
        assert manager._job_history_days == 7
        assert manager._max_concurrent_jobs_per_user == 10
        
        # Check status callbacks are initialized for all statuses
        for status in JobStatus:
            assert status in manager._status_callbacks
            assert isinstance(manager._status_callbacks[status], list)
    
    def test_create_job_success(self, manager, mock_redis):
        """Test successful job creation."""
        # Mock no existing jobs for user
        mock_redis.get_user_jobs.return_value = []
        
        job = manager.create_job(
            user_id="test_user",
            job_type=JobType.EMBEDDING,
            metadata={"test": "data"},
            priority=JobPriority.HIGH
        )
        
        assert isinstance(job, Job)
        assert job.user_id == "test_user"
        assert job.job_type == JobType.EMBEDDING
        assert job.status == JobStatus.PENDING
        assert job.metadata["test"] == "data"
        assert job.metadata["priority"] == JobPriority.HIGH.value
        assert job.metadata["created_by"] == "job_manager"
        assert job.metadata["version"] == "1.0"
        
        # Verify Redis was called
        mock_redis.set_job.assert_called_once_with(job)
    
    def test_create_job_user_limit_exceeded(self, manager, mock_redis):
        """Test job creation when user has reached limit."""
        # Mock user already has max jobs - need to create actual Job objects that are not finished
        existing_jobs = [
            Job(user_id="test_user", job_type=JobType.EMBEDDING, status=JobStatus.PENDING)
            for _ in range(manager._max_concurrent_jobs_per_user)
        ]
        
        # Mock the get_user_jobs method to return these jobs when active_only=True
        def mock_get_user_jobs(user_id, job_type=None, status=None, active_only=False, limit=100):
            if active_only:
                return [job for job in existing_jobs if not job.is_finished()]
            return existing_jobs
        
        with patch.object(manager, 'get_user_jobs', side_effect=mock_get_user_jobs):
            with pytest.raises(JobManagerError) as exc_info:
                manager.create_job("test_user", JobType.EMBEDDING)
            
            assert "maximum concurrent jobs limit" in str(exc_info.value)
    
    def test_create_job_redis_failure(self, manager, mock_redis):
        """Test job creation when Redis fails."""
        mock_redis.get_user_jobs.return_value = []
        mock_redis.set_job.return_value = False
        
        with pytest.raises(JobManagerError) as exc_info:
            manager.create_job("test_user", JobType.EMBEDDING)
        
        assert "Failed to store job" in str(exc_info.value)
    
    def test_create_job_with_callbacks(self, manager, mock_redis):
        """Test job creation triggers callbacks."""
        mock_redis.get_user_jobs.return_value = []
        
        callback_called = []
        
        def test_callback(job_id, event, job):
            callback_called.append((job_id, event, job))
        
        job = manager.create_job("test_user", JobType.EMBEDDING)
        manager.register_job_callback(job.job_id, test_callback)
        
        # Create another job to trigger callback
        job2 = manager.create_job("test_user", JobType.QUERY)
        
        # First job callback should not be triggered for second job
        assert len(callback_called) == 0
    
    def test_get_job_success(self, manager, mock_redis, sample_job):
        """Test successful job retrieval."""
        mock_redis.get_job.return_value = sample_job
        
        result = manager.get_job(sample_job.job_id)
        
        assert result == sample_job
        mock_redis.get_job.assert_called_once_with(sample_job.job_id)
    
    def test_get_job_not_found(self, manager, mock_redis):
        """Test job retrieval when job not found."""
        mock_redis.get_job.return_value = None
        
        result = manager.get_job("nonexistent_job")
        
        assert result is None
    
    def test_get_job_error(self, manager, mock_redis):
        """Test job retrieval with error."""
        mock_redis.get_job.side_effect = Exception("Redis error")
        
        result = manager.get_job("test_job")
        
        assert result is None
    
    def test_update_job_status_success(self, manager, mock_redis, sample_job):
        """Test successful job status update."""
        mock_redis.get_job.return_value = sample_job
        
        result = manager.update_job_status(
            sample_job.job_id,
            JobStatus.PROCESSING,
            error=None,
            result={"test": "result"}
        )
        
        assert result is True
        assert sample_job.status == JobStatus.PROCESSING
        assert sample_job.result == {"test": "result"}
        assert sample_job.started_at is not None
        mock_redis.set_job.assert_called_once_with(sample_job)
    
    def test_update_job_status_completed(self, manager, mock_redis, sample_job):
        """Test job status update to completed."""
        mock_redis.get_job.return_value = sample_job
        
        result = manager.update_job_status(sample_job.job_id, JobStatus.COMPLETED)
        
        assert result is True
        assert sample_job.status == JobStatus.COMPLETED
        assert sample_job.completed_at is not None
    
    def test_update_job_status_failed(self, manager, mock_redis, sample_job):
        """Test job status update to failed."""
        mock_redis.get_job.return_value = sample_job
        
        result = manager.update_job_status(
            sample_job.job_id,
            JobStatus.FAILED,
            error="Test error"
        )
        
        assert result is True
        assert sample_job.status == JobStatus.FAILED
        assert sample_job.error == "Test error"
        assert sample_job.completed_at is not None
    
    def test_update_job_status_not_found(self, manager, mock_redis):
        """Test job status update when job not found."""
        mock_redis.get_job.return_value = None
        
        result = manager.update_job_status("nonexistent", JobStatus.COMPLETED)
        
        assert result is False
    
    def test_update_job_status_redis_failure(self, manager, mock_redis, sample_job):
        """Test job status update when Redis fails."""
        mock_redis.get_job.return_value = sample_job
        mock_redis.set_job.return_value = False
        
        result = manager.update_job_status(sample_job.job_id, JobStatus.COMPLETED)
        
        assert result is False
    
    def test_update_job_status_with_callbacks(self, manager, mock_redis, sample_job):
        """Test job status update triggers callbacks."""
        mock_redis.get_job.return_value = sample_job
        
        job_callback_called = []
        status_callback_called = []
        
        def job_callback(job_id, event, job):
            job_callback_called.append((job_id, event, job))
        
        def status_callback(status, job):
            status_callback_called.append((status, job))
        
        manager.register_job_callback(sample_job.job_id, job_callback)
        manager.register_status_callback(JobStatus.PROCESSING, status_callback)
        
        manager.update_job_status(sample_job.job_id, JobStatus.PROCESSING)
        
        assert len(job_callback_called) == 1
        assert job_callback_called[0][0] == sample_job.job_id
        assert job_callback_called[0][1] == "status_changed"
        
        assert len(status_callback_called) == 1
        assert status_callback_called[0][0] == JobStatus.PROCESSING
    
    def test_update_job_progress_success(self, manager, mock_redis, sample_job):
        """Test successful job progress update."""
        mock_redis.get_job.return_value = sample_job
        
        result = manager.update_job_progress(
            sample_job.job_id,
            0.5,
            message="Half complete"
        )
        
        assert result is True
        assert sample_job.progress == 0.5
        assert sample_job.metadata["progress_message"] == "Half complete"
        assert "last_progress_update" in sample_job.metadata
        mock_redis.set_job.assert_called_once_with(sample_job)
    
    def test_update_job_progress_not_found(self, manager, mock_redis):
        """Test job progress update when job not found."""
        mock_redis.get_job.return_value = None
        
        result = manager.update_job_progress("nonexistent", 0.5)
        
        assert result is False
    
    def test_update_job_progress_redis_failure(self, manager, mock_redis, sample_job):
        """Test job progress update when Redis fails."""
        mock_redis.get_job.return_value = sample_job
        mock_redis.set_job.return_value = False
        
        result = manager.update_job_progress(sample_job.job_id, 0.5)
        
        assert result is False
    
    def test_update_job_progress_with_callback(self, manager, mock_redis, sample_job):
        """Test job progress update triggers callback."""
        mock_redis.get_job.return_value = sample_job
        
        callback_called = []
        
        def progress_callback(job_id, event, job):
            callback_called.append((job_id, event, job))
        
        manager.register_job_callback(sample_job.job_id, progress_callback)
        manager.update_job_progress(sample_job.job_id, 0.75)
        
        assert len(callback_called) == 1
        assert callback_called[0][1] == "progress_updated"
    
    def test_cancel_job_success(self, manager, mock_redis, sample_job):
        """Test successful job cancellation."""
        mock_redis.get_job.return_value = sample_job
        
        result = manager.cancel_job(sample_job.job_id, reason="User requested")
        
        assert result is True
        assert sample_job.status == JobStatus.CANCELLED
        assert "Job cancelled: User requested" in sample_job.error
        assert "cancelled_at" in sample_job.metadata
        assert sample_job.metadata["cancellation_reason"] == "User requested"
        mock_redis.set_job.assert_called_once_with(sample_job)
    
    def test_cancel_job_processing(self, manager, mock_redis, sample_job):
        """Test cancelling a processing job."""
        sample_job.status = JobStatus.PROCESSING
        mock_redis.get_job.return_value = sample_job
        
        result = manager.cancel_job(sample_job.job_id)
        
        assert result is True
        assert sample_job.status == JobStatus.CANCELLED
    
    def test_cancel_job_already_completed(self, manager, mock_redis, sample_job):
        """Test cancelling an already completed job."""
        sample_job.status = JobStatus.COMPLETED
        mock_redis.get_job.return_value = sample_job
        
        result = manager.cancel_job(sample_job.job_id)
        
        assert result is False
        assert sample_job.status == JobStatus.COMPLETED  # Unchanged
    
    def test_cancel_job_not_found(self, manager, mock_redis):
        """Test cancelling a non-existent job."""
        mock_redis.get_job.return_value = None
        
        result = manager.cancel_job("nonexistent")
        
        assert result is False
    
    def test_cancel_job_with_callbacks(self, manager, mock_redis, sample_job):
        """Test job cancellation triggers callbacks."""
        mock_redis.get_job.return_value = sample_job
        
        job_callback_called = []
        status_callback_called = []
        
        def job_callback(job_id, event, job):
            job_callback_called.append((job_id, event, job))
        
        def status_callback(status, job):
            status_callback_called.append((status, job))
        
        manager.register_job_callback(sample_job.job_id, job_callback)
        manager.register_status_callback(JobStatus.CANCELLED, status_callback)
        
        manager.cancel_job(sample_job.job_id)
        
        assert len(job_callback_called) == 1
        assert job_callback_called[0][1] == "cancelled"
        
        assert len(status_callback_called) == 1
        assert status_callback_called[0][0] == JobStatus.CANCELLED
    
    def test_delete_job_success(self, manager, mock_redis, sample_job):
        """Test successful job deletion."""
        sample_job.status = JobStatus.COMPLETED
        mock_redis.get_job.return_value = sample_job
        
        result = manager.delete_job(sample_job.job_id)
        
        assert result is True
        mock_redis.delete_job.assert_called_once_with(sample_job.job_id)
    
    def test_delete_job_active(self, manager, mock_redis, sample_job):
        """Test deleting an active job."""
        sample_job.status = JobStatus.PROCESSING
        mock_redis.get_job.return_value = sample_job
        
        result = manager.delete_job(sample_job.job_id)
        
        assert result is False
        mock_redis.delete_job.assert_not_called()
    
    def test_delete_job_not_found(self, manager, mock_redis):
        """Test deleting a non-existent job."""
        mock_redis.get_job.return_value = None
        
        result = manager.delete_job("nonexistent")
        
        assert result is False
    
    def test_delete_job_redis_failure(self, manager, mock_redis, sample_job):
        """Test job deletion when Redis fails."""
        sample_job.status = JobStatus.COMPLETED
        mock_redis.get_job.return_value = sample_job
        mock_redis.delete_job.return_value = False
        
        result = manager.delete_job(sample_job.job_id)
        
        assert result is False
    
    def test_delete_job_with_callback(self, manager, mock_redis, sample_job):
        """Test job deletion triggers callback."""
        sample_job.status = JobStatus.COMPLETED
        mock_redis.get_job.return_value = sample_job
        
        callback_called = []
        
        def delete_callback(job_id, event, job):
            callback_called.append((job_id, event, job))
        
        manager.register_job_callback(sample_job.job_id, delete_callback)
        manager.delete_job(sample_job.job_id)
        
        assert len(callback_called) == 1
        assert callback_called[0][1] == "deleted"
    
    def test_get_user_jobs_no_filters(self, manager, mock_redis):
        """Test getting user jobs without filters."""
        jobs = [
            Job(user_id="test_user", job_type=JobType.EMBEDDING, status=JobStatus.PENDING),
            Job(user_id="test_user", job_type=JobType.QUERY, status=JobStatus.COMPLETED)
        ]
        mock_redis.get_user_jobs.return_value = jobs
        
        result = manager.get_user_jobs("test_user")
        
        assert len(result) == 2
        assert result == jobs
        mock_redis.get_user_jobs.assert_called_once_with("test_user")
    
    def test_get_user_jobs_with_type_filter(self, manager, mock_redis):
        """Test getting user jobs with type filter."""
        jobs = [
            Job(user_id="test_user", job_type=JobType.EMBEDDING, status=JobStatus.PENDING),
            Job(user_id="test_user", job_type=JobType.QUERY, status=JobStatus.COMPLETED)
        ]
        mock_redis.get_user_jobs.return_value = jobs
        
        result = manager.get_user_jobs("test_user", job_type=JobType.EMBEDDING)
        
        assert len(result) == 1
        assert result[0].job_type == JobType.EMBEDDING
    
    def test_get_user_jobs_with_status_filter(self, manager, mock_redis):
        """Test getting user jobs with status filter."""
        jobs = [
            Job(user_id="test_user", job_type=JobType.EMBEDDING, status=JobStatus.PENDING),
            Job(user_id="test_user", job_type=JobType.QUERY, status=JobStatus.COMPLETED)
        ]
        mock_redis.get_user_jobs.return_value = jobs
        
        result = manager.get_user_jobs("test_user", status=JobStatus.COMPLETED)
        
        assert len(result) == 1
        assert result[0].status == JobStatus.COMPLETED
    
    def test_get_user_jobs_active_only(self, manager, mock_redis):
        """Test getting only active user jobs."""
        jobs = [
            Job(user_id="test_user", job_type=JobType.EMBEDDING, status=JobStatus.PENDING),
            Job(user_id="test_user", job_type=JobType.QUERY, status=JobStatus.COMPLETED)
        ]
        mock_redis.get_user_jobs.return_value = jobs
        
        result = manager.get_user_jobs("test_user", active_only=True)
        
        assert len(result) == 1
        assert result[0].status == JobStatus.PENDING
    
    def test_get_user_jobs_with_limit(self, manager, mock_redis):
        """Test getting user jobs with limit."""
        jobs = [Job(user_id="test_user", job_type=JobType.EMBEDDING) for _ in range(5)]
        mock_redis.get_user_jobs.return_value = jobs
        
        result = manager.get_user_jobs("test_user", limit=3)
        
        assert len(result) == 3
    
    def test_get_user_jobs_error(self, manager, mock_redis):
        """Test getting user jobs with error."""
        mock_redis.get_user_jobs.side_effect = Exception("Redis error")
        
        result = manager.get_user_jobs("test_user")
        
        assert result == []
    
    def test_get_jobs_by_status(self, manager, mock_redis):
        """Test getting jobs by status."""
        jobs = [Job(user_id="user1", job_type=JobType.EMBEDDING, status=JobStatus.PENDING)]
        mock_redis.get_jobs_by_status.return_value = jobs
        
        result = manager.get_jobs_by_status(JobStatus.PENDING)
        
        assert result == jobs
        mock_redis.get_jobs_by_status.assert_called_once_with(JobStatus.PENDING.value)
    
    def test_get_jobs_by_status_with_limit(self, manager, mock_redis):
        """Test getting jobs by status with limit."""
        jobs = [Job(user_id=f"user{i}", job_type=JobType.EMBEDDING) for i in range(200)]
        mock_redis.get_jobs_by_status.return_value = jobs
        
        result = manager.get_jobs_by_status(JobStatus.PENDING, limit=50)
        
        assert len(result) == 50
    
    def test_get_jobs_by_status_error(self, manager, mock_redis):
        """Test getting jobs by status with error."""
        mock_redis.get_jobs_by_status.side_effect = Exception("Redis error")
        
        result = manager.get_jobs_by_status(JobStatus.PENDING)
        
        assert result == []
    
    def test_get_job_statistics_user_specific(self, manager, mock_redis):
        """Test getting job statistics for specific user."""
        jobs = [
            Job(user_id="test_user", job_type=JobType.EMBEDDING, status=JobStatus.COMPLETED),
            Job(user_id="test_user", job_type=JobType.QUERY, status=JobStatus.FAILED),
            Job(user_id="test_user", job_type=JobType.EMBEDDING, status=JobStatus.PENDING)
        ]
        
        # Set completion times for duration calculation
        jobs[0].started_at = datetime.now() - timedelta(seconds=10)
        jobs[0].completed_at = datetime.now()
        jobs[1].started_at = datetime.now() - timedelta(seconds=5)
        jobs[1].completed_at = datetime.now()
        
        mock_redis.get_user_jobs.return_value = jobs
        
        stats = manager.get_job_statistics("test_user")
        
        assert stats["total_jobs"] == 3
        assert stats["by_status"][JobStatus.COMPLETED.value] == 1
        assert stats["by_status"][JobStatus.FAILED.value] == 1
        assert stats["by_status"][JobStatus.PENDING.value] == 1
        assert stats["by_type"][JobType.EMBEDDING.value] == 2
        assert stats["by_type"][JobType.QUERY.value] == 1
        assert stats["active_jobs"] == 1
        assert stats["completed_jobs"] == 1
        assert stats["failed_jobs"] == 1
        assert stats["success_rate"] == 0.5  # 1 completed / 2 finished
        assert stats["average_duration"] > 0
    
    def test_get_job_statistics_all_jobs(self, manager, mock_redis):
        """Test getting job statistics for all jobs."""
        jobs = [
            Job(user_id="user1", job_type=JobType.EMBEDDING, status=JobStatus.COMPLETED),
            Job(user_id="user2", job_type=JobType.QUERY, status=JobStatus.FAILED)
        ]
        
        # Mock get_jobs_by_status to return different jobs for different statuses
        def mock_get_jobs_by_status(status, limit=100):
            if status == JobStatus.COMPLETED:
                return [jobs[0]]
            elif status == JobStatus.FAILED:
                return [jobs[1]]
            else:
                return []
        
        with patch.object(manager, 'get_jobs_by_status', side_effect=mock_get_jobs_by_status):
            stats = manager.get_job_statistics()
        
        assert stats["total_jobs"] == 2
        assert stats["completed_jobs"] == 1
        assert stats["failed_jobs"] == 1
    
    def test_get_job_statistics_error(self, manager, mock_redis):
        """Test getting job statistics with error."""
        # Mock get_user_jobs to raise exception, but the method catches it and returns empty list
        # So we need to patch the method itself to raise an exception that gets caught
        with patch.object(manager, 'get_user_jobs', side_effect=Exception("Redis error")):
            stats = manager.get_job_statistics("test_user")
            
            assert "error" in stats
            assert stats["error"] == "Redis error"
    
    def test_job_context_success(self, manager, mock_redis, sample_job):
        """Test successful job context execution."""
        mock_redis.get_job.return_value = sample_job
        
        with manager.job_context(sample_job.job_id) as job:
            assert job == sample_job
            assert job.status == JobStatus.PROCESSING
        
        # Should be completed after context
        assert sample_job.status == JobStatus.COMPLETED
    
    def test_job_context_failure(self, manager, mock_redis, sample_job):
        """Test job context with exception."""
        mock_redis.get_job.return_value = sample_job
        
        with pytest.raises(ValueError):
            with manager.job_context(sample_job.job_id):
                raise ValueError("Test error")
        
        # Should be failed after exception
        assert sample_job.status == JobStatus.FAILED
        assert sample_job.error == "Test error"
    
    def test_job_context_not_found(self, manager, mock_redis):
        """Test job context with non-existent job."""
        mock_redis.get_job.return_value = None
        
        with pytest.raises(JobManagerError):
            with manager.job_context("nonexistent"):
                pass
    
    def test_register_job_callback(self, manager):
        """Test registering job callback."""
        callback = Mock()
        job_id = "test_job"
        
        manager.register_job_callback(job_id, callback)
        
        assert job_id in manager._job_callbacks
        assert callback in manager._job_callbacks[job_id]
    
    def test_unregister_job_callback(self, manager):
        """Test unregistering job callback."""
        callback = Mock()
        job_id = "test_job"
        
        manager.register_job_callback(job_id, callback)
        manager.unregister_job_callback(job_id, callback)
        
        assert job_id not in manager._job_callbacks
    
    def test_register_status_callback(self, manager):
        """Test registering status callback."""
        callback = Mock()
        
        manager.register_status_callback(JobStatus.COMPLETED, callback)
        
        assert callback in manager._status_callbacks[JobStatus.COMPLETED]
    
    def test_unregister_status_callback(self, manager):
        """Test unregistering status callback."""
        callback = Mock()
        
        manager.register_status_callback(JobStatus.COMPLETED, callback)
        manager.unregister_status_callback(JobStatus.COMPLETED, callback)
        
        assert callback not in manager._status_callbacks[JobStatus.COMPLETED]
    
    def test_trigger_job_callbacks_error_handling(self, manager):
        """Test job callback error handling."""
        def failing_callback(job_id, event, job):
            raise Exception("Callback error")
        
        def working_callback(job_id, event, job):
            working_callback.called = True
        
        working_callback.called = False
        
        job = Job(user_id="test", job_type=JobType.EMBEDDING)
        manager.register_job_callback(job.job_id, failing_callback)
        manager.register_job_callback(job.job_id, working_callback)
        
        # Should not raise exception, but working callback should still be called
        manager._trigger_job_callbacks(job.job_id, "test", job)
        
        assert working_callback.called is True
    
    def test_trigger_status_callbacks_error_handling(self, manager):
        """Test status callback error handling."""
        def failing_callback(status, job):
            raise Exception("Callback error")
        
        def working_callback(status, job):
            working_callback.called = True
        
        working_callback.called = False
        
        job = Job(user_id="test", job_type=JobType.EMBEDDING)
        manager.register_status_callback(JobStatus.COMPLETED, failing_callback)
        manager.register_status_callback(JobStatus.COMPLETED, working_callback)
        
        # Should not raise exception, but working callback should still be called
        manager._trigger_status_callbacks(JobStatus.COMPLETED, job)
        
        assert working_callback.called is True
    
    def test_start_cleanup_service(self, manager):
        """Test starting cleanup service."""
        manager.start_cleanup_service(interval=0.1)
        
        assert manager._cleanup_active is True
        assert manager._cleanup_thread is not None
        assert manager._cleanup_thread.is_alive()
        
        # Clean up
        manager.stop_cleanup_service()
    
    def test_start_cleanup_service_already_active(self, manager):
        """Test starting cleanup service when already active."""
        manager._cleanup_active = True
        
        manager.start_cleanup_service()
        
        # Should not create new thread
        assert manager._cleanup_thread is None
    
    def test_stop_cleanup_service(self, manager):
        """Test stopping cleanup service."""
        manager.start_cleanup_service(interval=0.1)
        time.sleep(0.05)  # Let it start
        
        manager.stop_cleanup_service()
        
        assert manager._cleanup_active is False
    
    def test_cleanup_old_jobs(self, manager, mock_redis):
        """Test cleaning up old jobs."""
        old_job = Job(user_id="test", job_type=JobType.EMBEDDING, status=JobStatus.COMPLETED)
        old_job.completed_at = datetime.now() - timedelta(days=10)
        
        recent_job = Job(user_id="test", job_type=JobType.EMBEDDING, status=JobStatus.COMPLETED)
        recent_job.completed_at = datetime.now() - timedelta(hours=1)
        
        # Mock get_jobs_by_status to return different jobs for different statuses
        def mock_get_jobs_by_status(status, limit=1000):
            if status == JobStatus.COMPLETED:
                return [old_job, recent_job]
            else:
                return []
        
        with patch.object(manager, 'get_jobs_by_status', side_effect=mock_get_jobs_by_status):
            with patch.object(manager, 'delete_job') as mock_delete:
                mock_delete.return_value = True
                
                cleaned_count = manager.cleanup_old_jobs(days=7)
                
                assert cleaned_count == 1
                mock_delete.assert_called_once_with(old_job.job_id)
    
    def test_cleanup_old_jobs_custom_days(self, manager, mock_redis):
        """Test cleaning up old jobs with custom days."""
        old_job = Job(user_id="test", job_type=JobType.EMBEDDING, status=JobStatus.COMPLETED)
        old_job.completed_at = datetime.now() - timedelta(days=2)
        
        # Mock get_jobs_by_status to return different jobs for different statuses
        def mock_get_jobs_by_status(status, limit=1000):
            if status == JobStatus.COMPLETED:
                return [old_job]
            else:
                return []
        
        with patch.object(manager, 'get_jobs_by_status', side_effect=mock_get_jobs_by_status):
            with patch.object(manager, 'delete_job') as mock_delete:
                mock_delete.return_value = True
                
                cleaned_count = manager.cleanup_old_jobs(days=1)
                
                assert cleaned_count == 1
    
    def test_cleanup_old_jobs_error(self, manager, mock_redis):
        """Test cleanup old jobs with error."""
        mock_redis.get_jobs_by_status.side_effect = Exception("Redis error")
        
        cleaned_count = manager.cleanup_old_jobs()
        
        assert cleaned_count == 0
    
    def test_cancel_user_jobs(self, manager, mock_redis):
        """Test cancelling all user jobs."""
        active_jobs = [
            Job(user_id="test_user", job_type=JobType.EMBEDDING, status=JobStatus.PENDING),
            Job(user_id="test_user", job_type=JobType.QUERY, status=JobStatus.PROCESSING)
        ]
        
        with patch.object(manager, 'get_user_jobs', return_value=active_jobs):
            with patch.object(manager, 'cancel_job', return_value=True) as mock_cancel:
                cancelled_count = manager.cancel_user_jobs("test_user", reason="System shutdown")
                
                assert cancelled_count == 2
                assert mock_cancel.call_count == 2
                mock_cancel.assert_any_call(active_jobs[0].job_id, "System shutdown")
                mock_cancel.assert_any_call(active_jobs[1].job_id, "System shutdown")
    
    def test_cancel_user_jobs_error(self, manager):
        """Test cancelling user jobs with error."""
        with patch.object(manager, 'get_user_jobs', side_effect=Exception("Error")):
            cancelled_count = manager.cancel_user_jobs("test_user")
            
            assert cancelled_count == 0
    
    def test_get_health_status_healthy(self, manager):
        """Test getting health status when healthy."""
        stats = {
            "total_jobs": 10,
            "active_jobs": 2,
            "completed_jobs": 7,
            "failed_jobs": 1
        }
        
        with patch.object(manager, 'get_job_statistics', return_value=stats):
            health = manager.get_health_status()
            
            assert health["status"] == "healthy"
            assert "timestamp" in health
            assert health["cleanup_active"] == manager._cleanup_active
            assert health["job_statistics"] == stats
            assert "system_limits" in health
            assert "active_callbacks" in health
    
    def test_get_health_status_with_warnings(self, manager):
        """Test getting health status with warnings."""
        stats = {
            "total_jobs": 200,
            "active_jobs": 150,  # High number
            "completed_jobs": 30,
            "failed_jobs": 50   # High failure rate
        }
        
        with patch.object(manager, 'get_job_statistics', return_value=stats):
            health = manager.get_health_status()
            
            assert health["status"] == "warning"
            assert "warnings" in health
            assert len(health["warnings"]) == 2
            assert "High number of active jobs" in health["warnings"]
            assert "High failure rate" in health["warnings"]
    
    def test_get_health_status_error(self, manager):
        """Test getting health status with error."""
        with patch.object(manager, 'get_job_statistics', side_effect=Exception("Test error")):
            health = manager.get_health_status()
            
            assert health["status"] == "error"
            assert health["error"] == "Test error"
            assert "timestamp" in health


class TestJobManagerConcurrency:
    """Test JobManager concurrent access and thread safety."""
    
    @pytest.fixture
    def manager(self):
        """Create a JobManager instance for testing."""
        return JobManager()
    
    @pytest.fixture
    def mock_redis(self):
        """Mock Redis client for testing."""
        with patch('app.shared.job_manager.redis_client') as mock:
            mock.set_job.return_value = True
            mock.get_job.return_value = None
            mock.delete_job.return_value = True
            mock.get_user_jobs.return_value = []
            mock.get_jobs_by_status.return_value = []
            yield mock
    
    def test_concurrent_job_creation(self, manager, mock_redis):
        """Test concurrent job creation."""
        results = []
        errors = []
        
        def create_jobs(user_id, count):
            try:
                for i in range(count):
                    job = manager.create_job(f"{user_id}_{i}", JobType.EMBEDDING)
                    results.append(job.job_id)
            except Exception as e:
                errors.append(str(e))
        
        # Start multiple threads creating jobs
        threads = []
        for i in range(5):
            thread = threading.Thread(target=create_jobs, args=(f"user_{i}", 3))
            threads.append(thread)
            thread.start()
        
        # Wait for all threads
        for thread in threads:
            thread.join()
        
        # Should have created 15 jobs total (5 users * 3 jobs each)
        assert len(results) == 15
        assert len(errors) == 0
        assert len(set(results)) == 15  # All job IDs should be unique
    
    def test_concurrent_status_updates(self, manager, mock_redis):
        """Test concurrent status updates."""
        job = Job(user_id="test_user", job_type=JobType.EMBEDDING)
        mock_redis.get_job.return_value = job
        
        results = []
        
        def update_status(status):
            try:
                result = manager.update_job_status(job.job_id, status)
                results.append(result)
            except Exception as e:
                results.append(False)
        
        # Start multiple threads updating status
        threads = []
        statuses = [JobStatus.PROCESSING, JobStatus.COMPLETED, JobStatus.FAILED]
        for status in statuses:
            thread = threading.Thread(target=update_status, args=(status,))
            threads.append(thread)
            thread.start()
        
        # Wait for all threads
        for thread in threads:
            thread.join()
        
        # All updates should succeed (though final status is unpredictable)
        assert all(results)
        assert len(results) == 3
    
    def test_concurrent_callback_registration(self, manager):
        """Test concurrent callback registration."""
        job_id = "test_job"
        callbacks = []
        
        def register_callback(callback_id):
            def callback(job_id, event, job):
                pass
            callback.__name__ = f"callback_{callback_id}"
            callbacks.append(callback)
            manager.register_job_callback(job_id, callback)
        
        # Start multiple threads registering callbacks
        threads = []
        for i in range(10):
            thread = threading.Thread(target=register_callback, args=(i,))
            threads.append(thread)
            thread.start()
        
        # Wait for all threads
        for thread in threads:
            thread.join()
        
        # All callbacks should be registered
        assert len(manager._job_callbacks[job_id]) == 10
        assert len(callbacks) == 10
    
    def test_concurrent_cleanup_operations(self, manager, mock_redis):
        """Test concurrent cleanup operations."""
        # Mock some old jobs
        old_jobs = [
            Job(user_id=f"user_{i}", job_type=JobType.EMBEDDING, status=JobStatus.COMPLETED)
            for i in range(10)
        ]
        for job in old_jobs:
            job.completed_at = datetime.now() - timedelta(days=10)
        
        mock_redis.get_jobs_by_status.return_value = old_jobs
        
        results = []
        
        def cleanup_worker():
            try:
                with patch.object(manager, 'delete_job', return_value=True):
                    count = manager.cleanup_old_jobs()
                    results.append(count)
            except Exception as e:
                results.append(0)
        
        # Start multiple cleanup threads
        threads = []
        for _ in range(3):
            thread = threading.Thread(target=cleanup_worker)
            threads.append(thread)
            thread.start()
        
        # Wait for all threads
        for thread in threads:
            thread.join()
        
        # All cleanup operations should complete
        assert len(results) == 3
        assert all(isinstance(r, int) for r in results)


class TestJobManagerIntegration:
    """Integration tests for JobManager."""
    
    def test_job_lifecycle_integration(self):
        """Test complete job lifecycle integration."""
        manager = JobManager()
        
        with patch('app.shared.job_manager.redis_client') as mock_redis:
            mock_redis.set_job.return_value = True
            mock_redis.get_user_jobs.return_value = []
            
            # Create job
            job = manager.create_job("test_user", JobType.EMBEDDING, metadata={"file": "test.pdf"})
            assert job.status == JobStatus.PENDING
            
            # Mock job retrieval
            mock_redis.get_job.return_value = job
            
            # Update to processing
            assert manager.update_job_status(job.job_id, JobStatus.PROCESSING) is True
            assert job.status == JobStatus.PROCESSING
            assert job.started_at is not None
            
            # Update progress
            assert manager.update_job_progress(job.job_id, 0.5, "Half complete") is True
            assert job.progress == 0.5
            
            # Complete job
            result_data = {"chunks": 10, "embeddings": 1024}
            assert manager.update_job_status(job.job_id, JobStatus.COMPLETED, result=result_data) is True
            assert job.status == JobStatus.COMPLETED
            assert job.completed_at is not None
            assert job.result == result_data
            
            # Verify job is finished
            assert job.is_finished() is True
            
            # Delete job
            mock_redis.delete_job.return_value = True
            assert manager.delete_job(job.job_id) is True
    
    def test_job_context_integration(self):
        """Test job context manager integration."""
        manager = JobManager()
        
        with patch('app.shared.job_manager.redis_client') as mock_redis:
            mock_redis.set_job.return_value = True
            mock_redis.get_user_jobs.return_value = []
            
            # Create job
            job = manager.create_job("test_user", JobType.QUERY)
            mock_redis.get_job.return_value = job
            
            # Use context manager
            with manager.job_context(job.job_id) as context_job:
                assert context_job.status == JobStatus.PROCESSING
                # Simulate work
                manager.update_job_progress(job.job_id, 0.8)
            
            # Job should be completed
            assert job.status == JobStatus.COMPLETED
    
    def test_callback_integration(self):
        """Test callback system integration."""
        manager = JobManager()
        
        job_events = []
        status_events = []
        
        def job_callback(job_id, event, job):
            job_events.append((job_id, event, job.status))
        
        def status_callback(status, job):
            status_events.append((status, job.job_id))
        
        with patch('app.shared.job_manager.redis_client') as mock_redis:
            mock_redis.set_job.return_value = True
            mock_redis.get_user_jobs.return_value = []
            
            # Create job and register callbacks
            job = manager.create_job("test_user", JobType.EMBEDDING)
            manager.register_job_callback(job.job_id, job_callback)
            manager.register_status_callback(JobStatus.PROCESSING, status_callback)
            manager.register_status_callback(JobStatus.COMPLETED, status_callback)
            
            mock_redis.get_job.return_value = job
            
            # Update status - should trigger callbacks
            manager.update_job_status(job.job_id, JobStatus.PROCESSING)
            manager.update_job_status(job.job_id, JobStatus.COMPLETED)
            
            # Verify callbacks were called
            assert len(job_events) == 2
            assert job_events[0][1] == "status_changed"
            assert job_events[1][1] == "status_changed"
            
            assert len(status_events) == 2
            assert status_events[0][0] == JobStatus.PROCESSING
            assert status_events[1][0] == JobStatus.COMPLETED


if __name__ == "__main__":
    pytest.main([__file__])