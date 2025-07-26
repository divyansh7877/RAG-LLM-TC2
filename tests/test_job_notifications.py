"""
Tests for job notification system and WebSocket integration.
"""
import pytest
import asyncio
import json
from datetime import datetime, timedelta
from unittest.mock import Mock, AsyncMock, patch, MagicMock
from typing import Dict, Any

from app.shared.job_notifications import (
    JobNotificationService,
    job_notification_service,
    notify_job_progress,
    notify_custom_message
)
from app.shared.models import Job, JobStatus, JobType, UserSession
from app.shared.websocket_manager import WebSocketManager


@pytest.fixture
def test_job():
    """Create test job."""
    return Job(
        job_id="test-job-123",
        user_id="test-user",
        job_type=JobType.EMBEDDING,
        status=JobStatus.PROCESSING,
        progress=0.5,
        metadata={
            "group_id": "test-group",
            "file_count": 3,
            "filenames": ["doc1.pdf", "doc2.pdf", "doc3.pdf"]
        }
    )


@pytest.fixture
def test_user_session():
    """Create test user session."""
    return UserSession(
        session_id="test-session-123",
        user_id="test-user",
        groups=["test-group"],
        permissions=["upload", "query"],
        created_at=datetime.now(),
        last_activity=datetime.now(),
        is_active=True
    )


@pytest.fixture
def notification_service():
    """Create fresh notification service instance for testing."""
    return JobNotificationService()


class TestJobNotificationService:
    """Test JobNotificationService class."""
    
    def test_initialization(self, notification_service):
        """Test notification service initialization."""
        assert not notification_service._initialized
    
    @patch('app.shared.job_notifications.job_manager')
    def test_initialize_registers_callbacks(self, mock_job_manager, notification_service):
        """Test that initialize registers callbacks with job manager."""
        notification_service.initialize()
        
        assert notification_service._initialized
        
        # Check that callbacks were registered for all job statuses
        expected_calls = len(JobStatus)
        assert mock_job_manager.register_status_callback.call_count == expected_calls
        
        # Verify each status has a callback registered
        for status in JobStatus:
            mock_job_manager.register_status_callback.assert_any_call(
                status, notification_service._handle_status_change
            )
    
    @patch('app.shared.job_notifications.job_manager')
    def test_initialize_idempotent(self, mock_job_manager, notification_service):
        """Test that initialize can be called multiple times safely."""
        notification_service.initialize()
        notification_service.initialize()
        
        # Should only register callbacks once
        expected_calls = len(JobStatus)
        assert mock_job_manager.register_status_callback.call_count == expected_calls
    
    @pytest.mark.asyncio
    @patch('app.shared.job_notifications.websocket_manager')
    async def test_handle_status_change(self, mock_websocket_manager, notification_service, test_job):
        """Test handling job status changes."""
        mock_websocket_manager.send_to_user = AsyncMock()
        
        # Update job status to completed for the test
        test_job.status = JobStatus.COMPLETED
        
        # Handle status change
        notification_service._handle_status_change(JobStatus.COMPLETED, test_job)
        
        # Wait for async task to complete
        await asyncio.sleep(0.1)
        
        # Verify WebSocket message was sent
        mock_websocket_manager.send_to_user.assert_called_once()
        call_args = mock_websocket_manager.send_to_user.call_args
        
        assert call_args[0][0] == test_job.user_id  # First argument is user_id
        notification = call_args[0][1]  # Second argument is notification
        
        assert notification["type"] == "job_notification"
        assert notification["event"] == "status_changed"
        assert notification["job"]["job_id"] == test_job.job_id
        assert notification["job"]["status"] == JobStatus.COMPLETED.value
    
    def test_create_job_notification_processing(self, notification_service, test_job):
        """Test creating notification for processing job."""
        test_job.status = JobStatus.PROCESSING
        
        notification = notification_service._create_job_notification(test_job, "status_changed")
        
        assert notification["type"] == "job_notification"
        assert notification["event"] == "status_changed"
        assert notification["job"]["job_id"] == test_job.job_id
        assert notification["job"]["status"] == JobStatus.PROCESSING.value
        assert notification["job"]["progress"] == 0.5
        assert "estimated_completion" in notification["job"]
        assert "message" in notification
        assert "processing" in notification["message"].lower()
    
    def test_create_job_notification_completed(self, notification_service, test_job):
        """Test creating notification for completed job."""
        test_job.status = JobStatus.COMPLETED
        test_job.started_at = datetime.now() - timedelta(minutes=5)
        test_job.completed_at = datetime.now()
        test_job.result = {
            "documents_processed": 3,
            "chunks_created": 150,
            "processing_time": 300
        }
        
        notification = notification_service._create_job_notification(test_job, "status_changed")
        
        assert notification["type"] == "job_notification"
        assert notification["job"]["status"] == JobStatus.COMPLETED.value
        assert "completed successfully" in notification["message"]
        assert "duration" in notification["job"]
        assert "result_summary" in notification["job"]
        
        # Check result summary
        result_summary = notification["job"]["result_summary"]
        assert result_summary["documents_processed"] == 3
        assert result_summary["chunks_created"] == 150
    
    def test_create_job_notification_failed(self, notification_service, test_job):
        """Test creating notification for failed job."""
        test_job.status = JobStatus.FAILED
        test_job.error = "Processing failed due to invalid file format"
        
        notification = notification_service._create_job_notification(test_job, "status_changed")
        
        assert notification["type"] == "job_notification"
        assert notification["job"]["status"] == JobStatus.FAILED.value
        assert "failed" in notification["message"].lower()
        assert notification["job"]["error_details"] == test_job.error
    
    def test_create_job_notification_cancelled(self, notification_service, test_job):
        """Test creating notification for cancelled job."""
        test_job.status = JobStatus.CANCELLED
        test_job.metadata["cancellation_reason"] = "User requested cancellation"
        
        notification = notification_service._create_job_notification(test_job, "status_changed")
        
        assert notification["type"] == "job_notification"
        assert notification["job"]["status"] == JobStatus.CANCELLED.value
        assert "cancelled" in notification["message"].lower()
        assert notification["job"]["cancellation_reason"] == "User requested cancellation"
    
    def test_create_job_notification_pending(self, notification_service, test_job):
        """Test creating notification for pending job."""
        test_job.status = JobStatus.PENDING
        
        with patch.object(notification_service, '_get_queue_position', return_value=3):
            notification = notification_service._create_job_notification(test_job, "status_changed")
        
        assert notification["type"] == "job_notification"
        assert notification["job"]["status"] == JobStatus.PENDING.value
        assert "queued" in notification["message"].lower()
        assert notification["job"]["queue_position"] == 3
    
    def test_create_progress_notification(self, notification_service, test_job):
        """Test creating progress notification."""
        test_job.metadata["progress_message"] = "Processing document 2 of 3"
        
        notification = notification_service._create_progress_notification(test_job)
        
        assert notification["type"] == "job_progress"
        assert notification["job_id"] == test_job.job_id
        assert notification["progress"] == 0.5
        assert notification["message"] == "Processing document 2 of 3"
        assert "timestamp" in notification
    
    def test_estimate_completion_time(self, notification_service, test_job):
        """Test completion time estimation."""
        # Set up job with started time and progress
        test_job.started_at = datetime.now() - timedelta(minutes=2)  # Started 2 minutes ago
        test_job.progress = 0.4  # 40% complete
        
        estimated_completion = notification_service._estimate_completion_time(test_job)
        
        assert estimated_completion is not None
        # Should estimate about 3 more minutes (2 minutes for 40% = 5 minutes total - 2 elapsed = 3 remaining)
        completion_time = datetime.fromisoformat(estimated_completion)
        expected_time = datetime.now() + timedelta(minutes=3)
        
        # Allow 1 minute tolerance
        assert abs((completion_time - expected_time).total_seconds()) < 60
    
    def test_estimate_completion_time_no_progress(self, notification_service, test_job):
        """Test completion time estimation with no progress."""
        test_job.started_at = datetime.now() - timedelta(minutes=2)
        test_job.progress = 0.0
        
        estimated_completion = notification_service._estimate_completion_time(test_job)
        
        assert estimated_completion is None
    
    def test_estimate_completion_time_not_started(self, notification_service, test_job):
        """Test completion time estimation for job not started."""
        test_job.started_at = None
        test_job.progress = 0.5
        
        estimated_completion = notification_service._estimate_completion_time(test_job)
        
        assert estimated_completion is None
    
    def test_create_result_summary_embedding_job(self, notification_service, test_job):
        """Test creating result summary for embedding job."""
        test_job.job_type = JobType.EMBEDDING
        test_job.result = {
            "documents_processed": 5,
            "chunks_created": 250,
            "embeddings_generated": 250,
            "processing_time": 600,
            "files": [
                {"filename": "doc1.pdf", "pages": 10, "chunks": 50, "status": "completed"},
                {"filename": "doc2.pdf", "pages": 15, "chunks": 75, "status": "completed"}
            ]
        }
        
        summary = notification_service._create_result_summary(test_job)
        
        assert summary["documents_processed"] == 5
        assert summary["chunks_created"] == 250
        assert summary["embeddings_generated"] == 250
        assert summary["processing_time"] == 600
        assert len(summary["files"]) == 2
        assert summary["files"][0]["filename"] == "doc1.pdf"
        assert summary["files"][0]["chunks"] == 50
    
    def test_create_result_summary_query_job(self, notification_service, test_job):
        """Test creating result summary for query job."""
        test_job.job_type = JobType.QUERY
        test_job.result = {
            "results_found": 8,
            "search_time": 1.5,
            "query_text": "What is machine learning?",
            "response_generated": True
        }
        
        summary = notification_service._create_result_summary(test_job)
        
        assert summary["results_found"] == 8
        assert summary["search_time"] == 1.5
        assert summary["query_text"] == "What is machine learning?"
        assert summary["response_generated"] is True
    
    def test_create_result_summary_no_result(self, notification_service, test_job):
        """Test creating result summary with no result data."""
        test_job.result = None
        
        summary = notification_service._create_result_summary(test_job)
        
        assert summary == {}
    
    @patch('app.shared.job_notifications.job_manager')
    def test_get_queue_position(self, mock_job_manager, notification_service, test_job):
        """Test getting queue position for pending job."""
        # Mock pending jobs
        pending_job1 = Job(job_id="job1", user_id="user1", job_type=JobType.EMBEDDING, 
                          created_at=datetime.now() - timedelta(minutes=5))
        pending_job2 = Job(job_id="job2", user_id="user2", job_type=JobType.EMBEDDING,
                          created_at=datetime.now() - timedelta(minutes=3))
        test_job.created_at = datetime.now() - timedelta(minutes=1)
        
        mock_job_manager.get_jobs_by_status.return_value = [pending_job1, pending_job2, test_job]
        
        position = notification_service._get_queue_position(test_job)
        
        assert position == 3  # Third in queue (1-based)
    
    @patch('app.shared.job_notifications.job_manager')
    def test_get_queue_position_not_found(self, mock_job_manager, notification_service, test_job):
        """Test getting queue position when job not found."""
        mock_job_manager.get_jobs_by_status.return_value = []
        
        position = notification_service._get_queue_position(test_job)
        
        assert position is None
    
    @pytest.mark.asyncio
    @patch('app.shared.job_notifications.websocket_manager')
    async def test_broadcast_job_notification(self, mock_websocket_manager, notification_service):
        """Test broadcasting job notification."""
        mock_websocket_manager.send_to_user = AsyncMock()
        
        notification = {"type": "test", "message": "test message"}
        await notification_service._broadcast_job_notification("test-user", notification)
        
        mock_websocket_manager.send_to_user.assert_called_once_with("test-user", notification)
    
    @pytest.mark.asyncio
    @patch('app.shared.job_notifications.job_manager')
    @patch('app.shared.job_notifications.websocket_manager')
    async def test_broadcast_progress_update(self, mock_websocket_manager, mock_job_manager, 
                                           notification_service, test_job):
        """Test broadcasting progress update."""
        mock_job_manager.get_job.return_value = test_job
        mock_websocket_manager.send_to_user = AsyncMock()
        
        await notification_service.broadcast_progress_update(test_job.job_id)
        
        mock_job_manager.get_job.assert_called_once_with(test_job.job_id)
        mock_websocket_manager.send_to_user.assert_called_once()
        
        # Check notification content
        call_args = mock_websocket_manager.send_to_user.call_args
        assert call_args[0][0] == test_job.user_id
        notification = call_args[0][1]
        assert notification["type"] == "job_progress"
        assert notification["job_id"] == test_job.job_id
    
    @pytest.mark.asyncio
    @patch('app.shared.job_notifications.job_manager')
    @patch('app.shared.job_notifications.websocket_manager')
    async def test_broadcast_progress_update_job_not_found(self, mock_websocket_manager, 
                                                         mock_job_manager, notification_service):
        """Test broadcasting progress update for non-existent job."""
        mock_job_manager.get_job.return_value = None
        mock_websocket_manager.send_to_user = AsyncMock()
        
        await notification_service.broadcast_progress_update("non-existent-job")
        
        mock_job_manager.get_job.assert_called_once_with("non-existent-job")
        mock_websocket_manager.send_to_user.assert_not_called()
    
    @pytest.mark.asyncio
    @patch('app.shared.job_notifications.websocket_manager')
    async def test_broadcast_custom_notification(self, mock_websocket_manager, notification_service):
        """Test broadcasting custom notification."""
        mock_websocket_manager.send_to_user = AsyncMock()
        
        await notification_service.broadcast_custom_notification(
            "test-user", 
            "System maintenance scheduled", 
            "warning",
            {"maintenance_time": "2024-01-01T00:00:00Z"}
        )
        
        mock_websocket_manager.send_to_user.assert_called_once()
        call_args = mock_websocket_manager.send_to_user.call_args
        
        assert call_args[0][0] == "test-user"
        notification = call_args[0][1]
        assert notification["type"] == "custom_notification"
        assert notification["notification_type"] == "warning"
        assert notification["message"] == "System maintenance scheduled"
        assert notification["metadata"]["maintenance_time"] == "2024-01-01T00:00:00Z"
    
    @patch('app.shared.job_notifications.websocket_manager')
    @patch('app.shared.job_notifications.job_manager')
    def test_get_notification_stats(self, mock_job_manager, mock_websocket_manager, notification_service):
        """Test getting notification statistics."""
        notification_service._initialized = True
        
        mock_websocket_manager.get_connection_stats.return_value = {
            "total_connections": 5,
            "users_connected": 3,
            "active_topics": 8
        }
        
        mock_job_manager._status_callbacks = {status: [] for status in JobStatus}
        mock_job_manager._job_callbacks = {"job1": [], "job2": []}
        
        stats = notification_service.get_notification_stats()
        
        assert stats["service_initialized"] is True
        assert stats["websocket_connections"] == 5
        assert stats["users_connected"] == 3
        assert stats["active_topics"] == 8
        assert stats["callback_registrations"]["status_callbacks"] == len(JobStatus)
        assert stats["callback_registrations"]["job_callbacks"] == 2
    
    @patch('app.shared.job_notifications.websocket_manager')
    @patch('app.shared.job_notifications.job_manager')
    def test_health_check(self, mock_job_manager, mock_websocket_manager, notification_service):
        """Test health check."""
        notification_service._initialized = True
        
        mock_websocket_manager.health_check.return_value = {
            "websocket_manager": True,
            "errors": []
        }
        
        mock_job_manager.get_job_statistics.return_value = {"total_jobs": 10}
        
        health = notification_service.health_check()
        
        assert health["job_notification_service"] is True
        assert health["websocket_manager"] is True
        assert health["job_manager"] is True
        assert health["errors"] == []
    
    @patch('app.shared.job_notifications.websocket_manager')
    @patch('app.shared.job_notifications.job_manager')
    def test_health_check_with_errors(self, mock_job_manager, mock_websocket_manager, notification_service):
        """Test health check with errors."""
        notification_service._initialized = True
        
        mock_websocket_manager.health_check.return_value = {
            "websocket_manager": False,
            "errors": ["WebSocket connection failed"]
        }
        
        mock_job_manager.get_job_statistics.side_effect = Exception("Job manager error")
        
        health = notification_service.health_check()
        
        assert health["job_notification_service"] is True
        assert health["websocket_manager"] is False
        assert health["job_manager"] is False
        assert "WebSocket connection failed" in health["errors"]
        assert "Job manager health check failed" in str(health["errors"])


class TestConvenienceFunctions:
    """Test convenience functions for job notifications."""
    
    @pytest.mark.asyncio
    @patch('app.shared.job_notifications.job_manager')
    @patch('app.shared.job_notifications.job_notification_service')
    async def test_notify_job_progress(self, mock_notification_service, mock_job_manager, test_job):
        """Test notify_job_progress convenience function."""
        mock_job_manager.get_job.return_value = test_job
        mock_job_manager.update_job_progress.return_value = True
        mock_notification_service.broadcast_progress_update = AsyncMock()
        
        await notify_job_progress(test_job.job_id, 0.75, "Almost done")
        
        # Check job manager was called
        mock_job_manager.update_job_progress.assert_called_once_with(
            test_job.job_id, 0.75, "Almost done"
        )
        
        # Check notification service was called
        mock_notification_service.broadcast_progress_update.assert_called_once_with(test_job.job_id)
        
        # Check metadata was updated
        assert test_job.metadata["progress_message"] == "Almost done"
    
    @pytest.mark.asyncio
    @patch('app.shared.job_notifications.job_manager')
    @patch('app.shared.job_notifications.job_notification_service')
    async def test_notify_job_progress_no_message(self, mock_notification_service, mock_job_manager, test_job):
        """Test notify_job_progress without message."""
        mock_job_manager.get_job.return_value = test_job
        mock_job_manager.update_job_progress.return_value = True
        mock_notification_service.broadcast_progress_update = AsyncMock()
        
        await notify_job_progress(test_job.job_id, 0.75)
        
        mock_job_manager.update_job_progress.assert_called_once_with(
            test_job.job_id, 0.75, None
        )
        mock_notification_service.broadcast_progress_update.assert_called_once_with(test_job.job_id)
    
    @pytest.mark.asyncio
    @patch('app.shared.job_notifications.job_manager')
    @patch('app.shared.job_notifications.job_notification_service')
    async def test_notify_job_progress_job_not_found(self, mock_notification_service, mock_job_manager):
        """Test notify_job_progress with non-existent job."""
        mock_job_manager.get_job.return_value = None
        mock_job_manager.update_job_progress.return_value = True
        mock_notification_service.broadcast_progress_update = AsyncMock()
        
        await notify_job_progress("non-existent-job", 0.75, "Almost done")
        
        mock_job_manager.update_job_progress.assert_called_once()
        mock_notification_service.broadcast_progress_update.assert_called_once()
    
    @pytest.mark.asyncio
    @patch('app.shared.job_notifications.job_notification_service')
    async def test_notify_custom_message(self, mock_notification_service):
        """Test notify_custom_message convenience function."""
        mock_notification_service.broadcast_custom_notification = AsyncMock()
        
        await notify_custom_message("test-user", "Welcome to the system!", "success")
        
        mock_notification_service.broadcast_custom_notification.assert_called_once_with(
            "test-user", "Welcome to the system!", "success"
        )
    
    @pytest.mark.asyncio
    @patch('app.shared.job_notifications.job_notification_service')
    async def test_notify_custom_message_default_type(self, mock_notification_service):
        """Test notify_custom_message with default notification type."""
        mock_notification_service.broadcast_custom_notification = AsyncMock()
        
        await notify_custom_message("test-user", "System update available")
        
        mock_notification_service.broadcast_custom_notification.assert_called_once_with(
            "test-user", "System update available", "info"
        )


@pytest.mark.asyncio
async def test_integration_job_status_change_triggers_websocket():
    """Integration test: job status change triggers WebSocket notification."""
    # This test would require more complex setup with actual WebSocket connections
    # For now, we'll test that the callback system works correctly
    
    with patch('app.shared.job_notifications.websocket_manager') as mock_websocket_manager:
        mock_websocket_manager.send_to_user = AsyncMock()
        
        # Create notification service and initialize
        service = JobNotificationService()
        service.initialize()
        
        # Create test job
        test_job = Job(
            job_id="integration-test-job",
            user_id="integration-test-user",
            job_type=JobType.EMBEDDING,
            status=JobStatus.COMPLETED
        )
        
        # Simulate status change callback
        service._handle_status_change(JobStatus.COMPLETED, test_job)
        
        # Wait for async task
        await asyncio.sleep(0.1)
        
        # Verify WebSocket notification was sent
        mock_websocket_manager.send_to_user.assert_called_once()
        call_args = mock_websocket_manager.send_to_user.call_args
        
        assert call_args[0][0] == "integration-test-user"
        notification = call_args[0][1]
        assert notification["type"] == "job_notification"
        assert notification["job"]["job_id"] == "integration-test-job"