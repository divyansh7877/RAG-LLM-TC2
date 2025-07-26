"""
Job notification system for real-time WebSocket updates.
"""
import asyncio
import logging
from typing import Dict, Any, Optional
from datetime import datetime

from .models import Job, JobStatus, JobType
from .websocket_manager import websocket_manager
from .job_manager import job_manager

# Set up logging
logger = logging.getLogger(__name__)


class JobNotificationService:
    """
    Service for broadcasting job status updates via WebSocket.
    
    Integrates with the job manager's callback system to provide
    real-time notifications to users about their job progress.
    """
    
    def __init__(self):
        """Initialize the job notification service."""
        self._initialized = False
        logger.info("JobNotificationService initialized")
    
    def initialize(self):
        """Initialize the service by registering callbacks with job manager."""
        if self._initialized:
            return
        
        # Register callbacks for all job status changes
        for status in JobStatus:
            job_manager.register_status_callback(status, self._handle_status_change)
        
        self._initialized = True
        logger.info("JobNotificationService callbacks registered")
    
    def _handle_status_change(self, status: JobStatus, job: Job):
        """
        Handle job status changes and broadcast notifications.
        
        Args:
            status: New job status
            job: Job instance
        """
        try:
            # Create notification message
            notification = self._create_job_notification(job, "status_changed")
            
            # Broadcast to user
            asyncio.create_task(self._broadcast_job_notification(job.user_id, notification))
            
            logger.debug(f"Broadcasted status change notification for job {job.job_id}: {status.value}")
            
        except Exception as e:
            logger.error(f"Error handling status change for job {job.job_id}: {e}")
    
    def _create_job_notification(self, job: Job, event_type: str) -> Dict[str, Any]:
        """
        Create a job notification message.
        
        Args:
            job: Job instance
            event_type: Type of event (status_changed, progress_updated, etc.)
            
        Returns:
            Notification message dictionary
        """
        # Base notification structure
        notification = {
            "type": "job_notification",
            "event": event_type,
            "job": {
                "job_id": job.job_id,
                "job_type": job.job_type.value,
                "status": job.status.value,
                "progress": job.progress,
                "created_at": job.created_at.isoformat() if job.created_at else None,
                "started_at": job.started_at.isoformat() if job.started_at else None,
                "completed_at": job.completed_at.isoformat() if job.completed_at else None,
                "error": job.error,
                "metadata": job.metadata or {}
            },
            "timestamp": datetime.now().isoformat()
        }
        
        # Add status-specific information
        if job.status == JobStatus.PROCESSING:
            notification["message"] = f"Job {job.job_id} is now processing"
            notification["job"]["estimated_completion"] = self._estimate_completion_time(job)
        
        elif job.status == JobStatus.COMPLETED:
            notification["message"] = f"Job {job.job_id} completed successfully"
            notification["job"]["duration"] = job.get_duration()
            
            # Add result summary if available
            if job.result:
                notification["job"]["result_summary"] = self._create_result_summary(job)
        
        elif job.status == JobStatus.FAILED:
            notification["message"] = f"Job {job.job_id} failed"
            notification["job"]["error_details"] = job.error or "Unknown error"
        
        elif job.status == JobStatus.CANCELLED:
            notification["message"] = f"Job {job.job_id} was cancelled"
            cancellation_reason = job.metadata.get("cancellation_reason")
            if cancellation_reason:
                notification["job"]["cancellation_reason"] = cancellation_reason
        
        elif job.status == JobStatus.PENDING:
            notification["message"] = f"Job {job.job_id} is queued for processing"
            notification["job"]["queue_position"] = self._get_queue_position(job)
        
        return notification
    
    def _create_progress_notification(self, job: Job) -> Dict[str, Any]:
        """
        Create a progress update notification.
        
        Args:
            job: Job instance
            
        Returns:
            Progress notification message
        """
        notification = {
            "type": "job_progress",
            "job_id": job.job_id,
            "progress": job.progress,
            "message": job.metadata.get("progress_message", "Processing..."),
            "timestamp": datetime.now().isoformat()
        }
        
        # Add estimated completion time
        estimated_completion = self._estimate_completion_time(job)
        if estimated_completion:
            notification["estimated_completion"] = estimated_completion
        
        return notification
    
    def _estimate_completion_time(self, job: Job) -> Optional[str]:
        """
        Estimate job completion time based on progress and elapsed time.
        
        Args:
            job: Job instance
            
        Returns:
            Estimated completion time as ISO string, or None if cannot estimate
        """
        try:
            if not job.started_at or job.progress <= 0:
                return None
            
            # Calculate elapsed time
            elapsed = (datetime.now() - job.started_at).total_seconds()
            
            # Estimate total time based on current progress
            if job.progress > 0:
                estimated_total = elapsed / job.progress
                remaining = estimated_total - elapsed
                
                if remaining > 0:
                    completion_time = datetime.now().timestamp() + remaining
                    return datetime.fromtimestamp(completion_time).isoformat()
            
            return None
            
        except Exception as e:
            logger.debug(f"Error estimating completion time for job {job.job_id}: {e}")
            return None
    
    def _create_result_summary(self, job: Job) -> Dict[str, Any]:
        """
        Create a summary of job results.
        
        Args:
            job: Completed job instance
            
        Returns:
            Result summary dictionary
        """
        if not job.result:
            return {}
        
        summary = {}
        
        # Handle embedding job results
        if job.job_type == JobType.EMBEDDING:
            summary.update({
                "documents_processed": job.result.get("documents_processed", 0),
                "chunks_created": job.result.get("chunks_created", 0),
                "embeddings_generated": job.result.get("embeddings_generated", 0),
                "processing_time": job.result.get("processing_time", 0)
            })
            
            # Add file-specific information
            if "files" in job.result:
                summary["files"] = [
                    {
                        "filename": file_info.get("filename"),
                        "pages": file_info.get("pages", 0),
                        "chunks": file_info.get("chunks", 0),
                        "status": file_info.get("status", "unknown")
                    }
                    for file_info in job.result["files"]
                ]
        
        # Handle query job results
        elif job.job_type == JobType.QUERY:
            summary.update({
                "results_found": job.result.get("results_found", 0),
                "search_time": job.result.get("search_time", 0),
                "query_text": job.result.get("query_text", ""),
                "response_generated": job.result.get("response_generated", False)
            })
        
        return summary
    
    def _get_queue_position(self, job: Job) -> Optional[int]:
        """
        Get the position of a pending job in the queue.
        
        Args:
            job: Pending job instance
            
        Returns:
            Queue position (1-based) or None if cannot determine
        """
        try:
            # Get all pending jobs of the same type
            pending_jobs = job_manager.get_jobs_by_status(JobStatus.PENDING)
            
            # Filter by job type and sort by creation time
            same_type_jobs = [
                j for j in pending_jobs 
                if j.job_type == job.job_type and j.created_at
            ]
            same_type_jobs.sort(key=lambda x: x.created_at)
            
            # Find position
            for i, pending_job in enumerate(same_type_jobs):
                if pending_job.job_id == job.job_id:
                    return i + 1  # 1-based position
            
            return None
            
        except Exception as e:
            logger.debug(f"Error getting queue position for job {job.job_id}: {e}")
            return None
    
    async def _broadcast_job_notification(self, user_id: str, notification: Dict[str, Any]):
        """
        Broadcast job notification to user's WebSocket connections.
        
        Args:
            user_id: User identifier
            notification: Notification message
        """
        try:
            await websocket_manager.send_to_user(user_id, notification)
        except Exception as e:
            logger.error(f"Error broadcasting job notification to user {user_id}: {e}")
    
    async def broadcast_progress_update(self, job_id: str):
        """
        Broadcast a progress update for a specific job.
        
        Args:
            job_id: Job identifier
        """
        try:
            job = job_manager.get_job(job_id)
            if not job:
                logger.warning(f"Job {job_id} not found for progress broadcast")
                return
            
            # Create progress notification
            notification = self._create_progress_notification(job)
            
            # Broadcast to user
            await self._broadcast_job_notification(job.user_id, notification)
            
            logger.debug(f"Broadcasted progress update for job {job_id}: {job.progress:.1%}")
            
        except Exception as e:
            logger.error(f"Error broadcasting progress update for job {job_id}: {e}")
    
    async def broadcast_custom_notification(
        self, 
        user_id: str, 
        message: str, 
        notification_type: str = "info",
        metadata: Optional[Dict[str, Any]] = None
    ):
        """
        Broadcast a custom notification to a user.
        
        Args:
            user_id: User identifier
            message: Notification message
            notification_type: Type of notification (info, warning, error, success)
            metadata: Optional additional metadata
        """
        try:
            notification = {
                "type": "custom_notification",
                "notification_type": notification_type,
                "message": message,
                "metadata": metadata or {},
                "timestamp": datetime.now().isoformat()
            }
            
            await self._broadcast_job_notification(user_id, notification)
            
            logger.debug(f"Broadcasted custom notification to user {user_id}: {message}")
            
        except Exception as e:
            logger.error(f"Error broadcasting custom notification to user {user_id}: {e}")
    
    def get_notification_stats(self) -> Dict[str, Any]:
        """
        Get notification service statistics.
        
        Returns:
            Dictionary containing notification statistics
        """
        try:
            websocket_stats = websocket_manager.get_connection_stats()
            
            return {
                "service_initialized": self._initialized,
                "websocket_connections": websocket_stats["total_connections"],
                "users_connected": websocket_stats["users_connected"],
                "active_topics": websocket_stats["active_topics"],
                "callback_registrations": {
                    "status_callbacks": len(job_manager._status_callbacks),
                    "job_callbacks": len(job_manager._job_callbacks)
                }
            }
            
        except Exception as e:
            logger.error(f"Error getting notification stats: {e}")
            return {"error": str(e)}
    
    def health_check(self) -> Dict[str, Any]:
        """
        Perform health check on notification service.
        
        Returns:
            Health check results
        """
        health = {
            "job_notification_service": self._initialized,
            "websocket_manager": False,
            "job_manager": False,
            "errors": []
        }
        
        try:
            # Check WebSocket manager
            ws_health = websocket_manager.health_check()
            health["websocket_manager"] = ws_health.get("websocket_manager", False)
            if ws_health.get("errors"):
                health["errors"].extend(ws_health["errors"])
            
            # Check job manager (basic check)
            try:
                job_manager.get_job_statistics()
                health["job_manager"] = True
            except Exception as e:
                health["errors"].append(f"Job manager health check failed: {e}")
            
        except Exception as e:
            health["errors"].append(f"Notification service health check failed: {e}")
        
        return health


# Global job notification service instance
job_notification_service = JobNotificationService()


# Convenience functions for manual progress updates
async def notify_job_progress(job_id: str, progress: float, message: Optional[str] = None):
    """
    Manually notify job progress update.
    
    Args:
        job_id: Job identifier
        progress: Progress value (0.0 to 1.0)
        message: Optional progress message
    """
    # Update job progress in manager
    if message:
        job = job_manager.get_job(job_id)
        if job:
            job.metadata["progress_message"] = message
    
    job_manager.update_job_progress(job_id, progress, message)
    
    # Broadcast progress update
    await job_notification_service.broadcast_progress_update(job_id)


async def notify_custom_message(user_id: str, message: str, notification_type: str = "info"):
    """
    Send custom notification to user.
    
    Args:
        user_id: User identifier
        message: Notification message
        notification_type: Type of notification
    """
    await job_notification_service.broadcast_custom_notification(
        user_id, message, notification_type
    )