"""
Job management system for tracking and managing background tasks.
"""
import logging
import threading
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Callable, Set
from contextlib import contextmanager
from enum import Enum

from .models import Job, JobStatus, JobType
from .redis_client import redis_client
from .config import config

logger = logging.getLogger(__name__)


class JobPriority(Enum):
    """Job priority levels."""
    LOW = 1
    NORMAL = 5
    HIGH = 8
    URGENT = 10


class JobManagerError(Exception):
    """Exception raised by JobManager operations."""
    pass


class JobManager:
    """
    Comprehensive job management system that handles job lifecycle,
    progress tracking, cancellation, and cleanup.
    """
    
    def __init__(self):
        """Initialize the job manager."""
        self._lock = threading.RLock()
        self._job_callbacks: Dict[str, List[Callable]] = {}
        self._status_callbacks: Dict[JobStatus, List[Callable]] = {
            status: [] for status in JobStatus
        }
        self._cleanup_thread = None
        self._cleanup_active = False
        self._job_history_days = 7  # Keep job history for 7 days
        self._max_concurrent_jobs_per_user = 10
        
        logger.info("JobManager initialized")
    
    def create_job(
        self,
        user_id: str,
        job_type: JobType,
        metadata: Optional[Dict[str, Any]] = None,
        priority: JobPriority = JobPriority.NORMAL
    ) -> Job:
        """
        Create a new job.
        
        Args:
            user_id: User who owns the job
            job_type: Type of job to create
            metadata: Optional metadata for the job
            priority: Job priority level
            
        Returns:
            Created Job instance
            
        Raises:
            JobManagerError: If job creation fails
        """
        try:
            # Check user job limits
            user_jobs = self.get_user_jobs(user_id, active_only=True)
            if len(user_jobs) >= self._max_concurrent_jobs_per_user:
                raise JobManagerError(
                    f"User {user_id} has reached maximum concurrent jobs limit "
                    f"({self._max_concurrent_jobs_per_user})"
                )
            
            # Create job with metadata
            job_metadata = metadata or {}
            job_metadata.update({
                "priority": priority.value,
                "created_by": "job_manager",
                "version": "1.0"
            })
            
            job = Job(
                user_id=user_id,
                job_type=job_type,
                status=JobStatus.PENDING,
                metadata=job_metadata
            )
            
            # Store job in Redis
            if not redis_client.set_job(job):
                raise JobManagerError(f"Failed to store job {job.job_id}")
            
            logger.info(f"Created job {job.job_id} for user {user_id} (type: {job_type.value})")
            
            # Trigger callbacks
            self._trigger_job_callbacks(job.job_id, "created", job)
            
            return job
            
        except Exception as e:
            logger.error(f"Error creating job: {e}")
            if isinstance(e, JobManagerError):
                raise
            raise JobManagerError(f"Job creation failed: {e}")
    
    def get_job(self, job_id: str) -> Optional[Job]:
        """
        Get job by ID.
        
        Args:
            job_id: Job identifier
            
        Returns:
            Job instance or None if not found
        """
        try:
            return redis_client.get_job(job_id)
        except Exception as e:
            logger.error(f"Error getting job {job_id}: {e}")
            return None
    
    def update_job_status(
        self,
        job_id: str,
        status: JobStatus,
        error: Optional[str] = None,
        result: Optional[Dict[str, Any]] = None
    ) -> bool:
        """
        Update job status.
        
        Args:
            job_id: Job identifier
            status: New job status
            error: Optional error message
            result: Optional result data
            
        Returns:
            True if update successful, False otherwise
        """
        try:
            job = self.get_job(job_id)
            if not job:
                logger.warning(f"Job {job_id} not found for status update")
                return False
            
            # Update job status and timestamps
            old_status = job.status
            job.update_status(status, error)
            
            if result:
                job.result = result
            
            # Store updated job
            if not redis_client.set_job(job):
                logger.error(f"Failed to update job {job_id} status")
                return False
            
            logger.info(f"Updated job {job_id} status: {old_status.value} -> {status.value}")
            
            # Trigger callbacks
            self._trigger_job_callbacks(job_id, "status_changed", job)
            self._trigger_status_callbacks(status, job)
            
            return True
            
        except Exception as e:
            logger.error(f"Error updating job {job_id} status: {e}")
            return False
    
    def update_job_progress(self, job_id: str, progress: float, message: Optional[str] = None) -> bool:
        """
        Update job progress.
        
        Args:
            job_id: Job identifier
            progress: Progress value (0.0 to 1.0)
            message: Optional progress message
            
        Returns:
            True if update successful, False otherwise
        """
        try:
            job = self.get_job(job_id)
            if not job:
                logger.warning(f"Job {job_id} not found for progress update")
                return False
            
            # Update progress
            job.update_progress(progress)
            
            if message:
                job.metadata["progress_message"] = message
                job.metadata["last_progress_update"] = datetime.now().isoformat()
            
            # Store updated job
            if not redis_client.set_job(job):
                logger.error(f"Failed to update job {job_id} progress")
                return False
            
            logger.debug(f"Updated job {job_id} progress: {progress:.2%}")
            
            # Trigger callbacks
            self._trigger_job_callbacks(job_id, "progress_updated", job)
            
            return True
            
        except Exception as e:
            logger.error(f"Error updating job {job_id} progress: {e}")
            return False
    
    def cancel_job(self, job_id: str, reason: Optional[str] = None) -> bool:
        """
        Cancel a job.
        
        Args:
            job_id: Job identifier
            reason: Optional cancellation reason
            
        Returns:
            True if cancellation successful, False otherwise
        """
        try:
            job = self.get_job(job_id)
            if not job:
                logger.warning(f"Job {job_id} not found for cancellation")
                return False
            
            # Can only cancel pending or processing jobs
            if job.status not in [JobStatus.PENDING, JobStatus.PROCESSING]:
                logger.warning(f"Cannot cancel job {job_id} with status {job.status.value}")
                return False
            
            # Update job status
            error_msg = f"Job cancelled"
            if reason:
                error_msg += f": {reason}"
            
            job.update_status(JobStatus.CANCELLED, error_msg)
            job.metadata["cancelled_at"] = datetime.now().isoformat()
            if reason:
                job.metadata["cancellation_reason"] = reason
            
            # Store updated job
            if not redis_client.set_job(job):
                logger.error(f"Failed to cancel job {job_id}")
                return False
            
            logger.info(f"Cancelled job {job_id}: {reason or 'No reason provided'}")
            
            # Trigger callbacks
            self._trigger_job_callbacks(job_id, "cancelled", job)
            self._trigger_status_callbacks(JobStatus.CANCELLED, job)
            
            return True
            
        except Exception as e:
            logger.error(f"Error cancelling job {job_id}: {e}")
            return False
    
    def delete_job(self, job_id: str) -> bool:
        """
        Delete a job permanently.
        
        Args:
            job_id: Job identifier
            
        Returns:
            True if deletion successful, False otherwise
        """
        try:
            job = self.get_job(job_id)
            if not job:
                logger.warning(f"Job {job_id} not found for deletion")
                return False
            
            # Only delete finished jobs
            if not job.is_finished():
                logger.warning(f"Cannot delete active job {job_id} with status {job.status.value}")
                return False
            
            # Delete from Redis
            if not redis_client.delete_job(job_id):
                logger.error(f"Failed to delete job {job_id}")
                return False
            
            logger.info(f"Deleted job {job_id}")
            
            # Trigger callbacks
            self._trigger_job_callbacks(job_id, "deleted", job)
            
            return True
            
        except Exception as e:
            logger.error(f"Error deleting job {job_id}: {e}")
            return False
    
    def get_user_jobs(
        self,
        user_id: str,
        job_type: Optional[JobType] = None,
        status: Optional[JobStatus] = None,
        active_only: bool = False,
        limit: int = 100
    ) -> List[Job]:
        """
        Get jobs for a specific user.
        
        Args:
            user_id: User identifier
            job_type: Optional job type filter
            status: Optional status filter
            active_only: If True, only return non-finished jobs
            limit: Maximum number of jobs to return
            
        Returns:
            List of Job instances
        """
        try:
            all_jobs = redis_client.get_user_jobs(user_id)
            
            # Apply filters
            filtered_jobs = []
            for job in all_jobs:
                # Type filter
                if job_type and job.job_type != job_type:
                    continue
                
                # Status filter
                if status and job.status != status:
                    continue
                
                # Active only filter
                if active_only and job.is_finished():
                    continue
                
                filtered_jobs.append(job)
                
                # Limit check
                if len(filtered_jobs) >= limit:
                    break
            
            return filtered_jobs
            
        except Exception as e:
            logger.error(f"Error getting user jobs for {user_id}: {e}")
            return []
    
    def get_jobs_by_status(self, status: JobStatus, limit: int = 100) -> List[Job]:
        """
        Get jobs by status.
        
        Args:
            status: Job status to filter by
            limit: Maximum number of jobs to return
            
        Returns:
            List of Job instances
        """
        try:
            return redis_client.get_jobs_by_status(status.value)[:limit]
        except Exception as e:
            logger.error(f"Error getting jobs by status {status.value}: {e}")
            return []
    
    def get_job_statistics(self, user_id: Optional[str] = None) -> Dict[str, Any]:
        """
        Get job statistics.
        
        Args:
            user_id: Optional user ID to filter statistics
            
        Returns:
            Dictionary containing job statistics
        """
        try:
            if user_id:
                jobs = self.get_user_jobs(user_id, limit=1000)
            else:
                # Get all jobs (this could be expensive for large systems)
                jobs = []
                for status in JobStatus:
                    jobs.extend(self.get_jobs_by_status(status, limit=1000))
            
            # Calculate statistics
            stats = {
                "total_jobs": len(jobs),
                "by_status": {status.value: 0 for status in JobStatus},
                "by_type": {job_type.value: 0 for job_type in JobType},
                "active_jobs": 0,
                "completed_jobs": 0,
                "failed_jobs": 0,
                "average_duration": 0.0,
                "success_rate": 0.0
            }
            
            durations = []
            for job in jobs:
                # Status counts
                stats["by_status"][job.status.value] += 1
                
                # Type counts
                stats["by_type"][job.job_type.value] += 1
                
                # Active/completed/failed counts
                if not job.is_finished():
                    stats["active_jobs"] += 1
                elif job.status == JobStatus.COMPLETED:
                    stats["completed_jobs"] += 1
                elif job.status == JobStatus.FAILED:
                    stats["failed_jobs"] += 1
                
                # Duration calculation
                duration = job.get_duration()
                if duration is not None:
                    durations.append(duration)
            
            # Calculate averages
            if durations:
                stats["average_duration"] = sum(durations) / len(durations)
            
            # Calculate success rate
            finished_jobs = stats["completed_jobs"] + stats["failed_jobs"]
            if finished_jobs > 0:
                stats["success_rate"] = stats["completed_jobs"] / finished_jobs
            
            return stats
            
        except Exception as e:
            logger.error(f"Error getting job statistics: {e}")
            return {"error": str(e)}
    
    @contextmanager
    def job_context(self, job_id: str):
        """
        Context manager for job execution with automatic status updates.
        
        Args:
            job_id: Job identifier
        """
        job = self.get_job(job_id)
        if not job:
            raise JobManagerError(f"Job {job_id} not found")
        
        # Start job
        self.update_job_status(job_id, JobStatus.PROCESSING)
        
        try:
            yield job
            # Job completed successfully
            self.update_job_status(job_id, JobStatus.COMPLETED)
            
        except Exception as e:
            # Job failed
            error_msg = str(e)
            logger.error(f"Job {job_id} failed: {error_msg}")
            self.update_job_status(job_id, JobStatus.FAILED, error=error_msg)
            raise
    
    def register_job_callback(self, job_id: str, callback: Callable):
        """
        Register a callback for job events.
        
        Args:
            job_id: Job identifier
            callback: Callback function
        """
        with self._lock:
            if job_id not in self._job_callbacks:
                self._job_callbacks[job_id] = []
            self._job_callbacks[job_id].append(callback)
            logger.debug(f"Registered callback for job {job_id}")
    
    def unregister_job_callback(self, job_id: str, callback: Callable):
        """
        Unregister a callback for job events.
        
        Args:
            job_id: Job identifier
            callback: Callback function to remove
        """
        with self._lock:
            if job_id in self._job_callbacks and callback in self._job_callbacks[job_id]:
                self._job_callbacks[job_id].remove(callback)
                if not self._job_callbacks[job_id]:
                    del self._job_callbacks[job_id]
                logger.debug(f"Unregistered callback for job {job_id}")
    
    def register_status_callback(self, status: JobStatus, callback: Callable):
        """
        Register a callback for status changes.
        
        Args:
            status: Job status to monitor
            callback: Callback function
        """
        with self._lock:
            self._status_callbacks[status].append(callback)
            logger.debug(f"Registered callback for status {status.value}")
    
    def unregister_status_callback(self, status: JobStatus, callback: Callable):
        """
        Unregister a callback for status changes.
        
        Args:
            status: Job status to stop monitoring
            callback: Callback function to remove
        """
        with self._lock:
            if callback in self._status_callbacks[status]:
                self._status_callbacks[status].remove(callback)
                logger.debug(f"Unregistered callback for status {status.value}")
    
    def _trigger_job_callbacks(self, job_id: str, event: str, job: Job):
        """Trigger callbacks for job events."""
        try:
            with self._lock:
                callbacks = self._job_callbacks.get(job_id, [])
            
            for callback in callbacks:
                try:
                    callback(job_id, event, job)
                except Exception as e:
                    logger.error(f"Error in job callback: {e}")
                    
        except Exception as e:
            logger.error(f"Error triggering job callbacks: {e}")
    
    def _trigger_status_callbacks(self, status: JobStatus, job: Job):
        """Trigger callbacks for status changes."""
        try:
            with self._lock:
                callbacks = self._status_callbacks.get(status, [])
            
            for callback in callbacks:
                try:
                    callback(status, job)
                except Exception as e:
                    logger.error(f"Error in status callback: {e}")
                    
        except Exception as e:
            logger.error(f"Error triggering status callbacks: {e}")
    
    def start_cleanup_service(self, interval: float = 3600.0):
        """
        Start automatic cleanup service.
        
        Args:
            interval: Cleanup interval in seconds (default: 1 hour)
        """
        with self._lock:
            if self._cleanup_active:
                logger.warning("Cleanup service is already active")
                return
            
            self._cleanup_active = True
            self._cleanup_thread = threading.Thread(
                target=self._cleanup_loop,
                args=(interval,),
                daemon=True,
                name="JobCleanup"
            )
            self._cleanup_thread.start()
            logger.info(f"Job cleanup service started with {interval}s interval")
    
    def stop_cleanup_service(self):
        """Stop automatic cleanup service."""
        with self._lock:
            if not self._cleanup_active:
                return
            
            self._cleanup_active = False
            if self._cleanup_thread and self._cleanup_thread.is_alive():
                self._cleanup_thread.join(timeout=5.0)
            
            logger.info("Job cleanup service stopped")
    
    def _cleanup_loop(self, interval: float):
        """Main cleanup loop."""
        while self._cleanup_active:
            try:
                self.cleanup_old_jobs()
            except Exception as e:
                logger.error(f"Error in cleanup loop: {e}")
            
            time.sleep(interval)
    
    def cleanup_old_jobs(self, days: Optional[int] = None) -> int:
        """
        Clean up old completed jobs.
        
        Args:
            days: Number of days to keep jobs (default: configured value)
            
        Returns:
            Number of jobs cleaned up
        """
        try:
            days = days or self._job_history_days
            cutoff_date = datetime.now() - timedelta(days=days)
            
            cleaned_count = 0
            
            # Get all completed and failed jobs
            for status in [JobStatus.COMPLETED, JobStatus.FAILED, JobStatus.CANCELLED]:
                jobs = self.get_jobs_by_status(status, limit=1000)
                
                for job in jobs:
                    if job.completed_at and job.completed_at < cutoff_date:
                        if self.delete_job(job.job_id):
                            cleaned_count += 1
                            logger.debug(f"Cleaned up old job {job.job_id}")
            
            if cleaned_count > 0:
                logger.info(f"Cleaned up {cleaned_count} old jobs")
            
            return cleaned_count
            
        except Exception as e:
            logger.error(f"Error cleaning up old jobs: {e}")
            return 0
    
    def cancel_user_jobs(self, user_id: str, reason: Optional[str] = None) -> int:
        """
        Cancel all active jobs for a user.
        
        Args:
            user_id: User identifier
            reason: Optional cancellation reason
            
        Returns:
            Number of jobs cancelled
        """
        try:
            active_jobs = self.get_user_jobs(user_id, active_only=True)
            cancelled_count = 0
            
            for job in active_jobs:
                if self.cancel_job(job.job_id, reason):
                    cancelled_count += 1
            
            if cancelled_count > 0:
                logger.info(f"Cancelled {cancelled_count} jobs for user {user_id}")
            
            return cancelled_count
            
        except Exception as e:
            logger.error(f"Error cancelling user jobs for {user_id}: {e}")
            return 0
    
    def get_health_status(self) -> Dict[str, Any]:
        """
        Get health status of the job management system.
        
        Returns:
            Dictionary containing health status information
        """
        try:
            stats = self.get_job_statistics()
            
            health_info = {
                "status": "healthy",
                "timestamp": datetime.now().isoformat(),
                "cleanup_active": self._cleanup_active,
                "job_statistics": stats,
                "system_limits": {
                    "max_concurrent_jobs_per_user": self._max_concurrent_jobs_per_user,
                    "job_history_days": self._job_history_days
                },
                "active_callbacks": {
                    "job_callbacks": len(self._job_callbacks),
                    "status_callbacks": sum(len(callbacks) for callbacks in self._status_callbacks.values())
                }
            }
            
            # Check for potential issues
            warnings = []
            if stats.get("active_jobs", 0) > 100:
                warnings.append("High number of active jobs")
            
            if stats.get("failed_jobs", 0) > stats.get("completed_jobs", 0):
                warnings.append("High failure rate")
            
            if warnings:
                health_info["warnings"] = warnings
                health_info["status"] = "warning"
            
            return health_info
            
        except Exception as e:
            logger.error(f"Error getting job manager health status: {e}")
            return {
                "status": "error",
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }


# Global job manager instance
job_manager = JobManager()