"""
Celery worker for document embedding tasks with progress tracking and user isolation.
"""
import os
import shutil
import tempfile
import time
import logging
from typing import List, Tuple, Dict, Any
from celery import current_task
from celery.exceptions import Retry, WorkerLostError

from .celery_app import celery_app
from ..shared.redis_client import redis_client
from ..shared.models import JobStatus, Document
from ..shared.config import config
from ..shared.job_manager import job_manager

# Import embedding functionality
from ..shared.document_processor import document_processor, get_document_info, estimate_processing_time
import hashlib
import uuid

# Configure logging
logger = logging.getLogger(__name__)

# Constants
CHUNK_SIZE = 512
CHUNK_OVERLAP = 20
EMBED_MODEL_NAME = "./models/gte-large-en-v1.5"
DB_PATH = "./multi_user_db.lance"
TABLE_NAME = "document_embeddings"

# Performance monitoring constants
EMBEDDING_METRICS_KEY = "performance:embedding_metrics"
SLOW_EMBEDDING_THRESHOLD = 30.0  # seconds per file


def calculate_file_hash(file_path: str) -> str:
    """Calculate SHA-256 hash of file content for deduplication."""
    hash_sha256 = hashlib.sha256()
    try:
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hash_sha256.update(chunk)
        return hash_sha256.hexdigest()
    except Exception as e:
        logger.error(f"Failed to calculate hash for {file_path}: {e}")
        return str(uuid.uuid4())  # Fallback to UUID if hashing fails


def check_duplicate_document(user_id: str, group_id: str, file_hash: str) -> bool:
    """Check if document with same hash already exists for user/group."""
    try:
        # Check Redis for existing document metadata - only for this specific user/group
        pattern = f"document:{user_id}:{group_id}:*"
        keys = redis_client.redis_client.keys(pattern)
        
        for key in keys:
            doc_data = redis_client.get_json(key.decode('utf-8'))
            if doc_data and doc_data.get('file_hash') == file_hash:
                return True
        return False
    except Exception as e:
        logger.warning(f"Failed to check for duplicates: {e}")
        return False  # Proceed with processing if check fails


def store_document_metadata(user_id: str, group_id: str, file_path: str, 
                          page_count: int, chunk_count: int) -> str:
    """Store document metadata in Redis."""
    document_id = str(uuid.uuid4())
    filename = os.path.basename(file_path)
    file_size = os.path.getsize(file_path)
    file_hash = calculate_file_hash(file_path)
    
    document = Document(
        document_id=document_id,
        user_id=user_id,
        group_id=group_id,
        filename=filename,
        file_size=file_size,
        upload_date=time.time(),
        processing_status="completed",
        page_count=page_count,
        chunk_count=chunk_count,
        file_hash=file_hash,
        content_type="application/pdf"
    )
    
    # Store in Redis
    redis_key = f"document:{user_id}:{group_id}:{document_id}"
    redis_client.set_json(redis_key, document.to_dict(), expire_seconds=None)
    
    return document_id


def update_job_progress(job_id: str, progress: float, status_message: str = None):
    """Update job progress using job manager and trigger WebSocket notifications."""
    try:
        # Update job progress through job manager (this will trigger WebSocket notifications)
        success = job_manager.update_job_progress(job_id, progress, status_message)
        
        if not success:
            logger.warning(f"Failed to update job progress for {job_id}")
        
        # Update Celery task state for Celery monitoring
        if current_task:
            current_task.update_state(
                state="PROGRESS",
                meta={
                    "progress": progress,
                    "status": status_message or "Processing...",
                    "timestamp": time.time()
                }
            )
    except Exception as e:
        logger.error(f"Failed to update job progress: {e}")


def record_embedding_performance_metrics(job_id: str, user_id: str, file_count: int,
                                       total_processing_time: float, total_chunks: int,
                                       batch_size: int, success: bool, error: str = None):
    """
    Record embedding performance metrics for monitoring and optimization.
    
    Args:
        job_id: Job identifier
        user_id: User who submitted the job
        file_count: Number of files processed
        total_processing_time: Total time taken to process all files
        total_chunks: Total number of chunks created
        batch_size: Batch size used for processing
        success: Whether the job completed successfully
        error: Error message if job failed
    """
    try:
        timestamp = time.time()
        avg_time_per_file = total_processing_time / max(file_count, 1)
        
        # Create performance metrics entry
        metrics = {
            "job_id": job_id,
            "user_id": user_id,
            "file_count": file_count,
            "total_processing_time": total_processing_time,
            "avg_time_per_file": avg_time_per_file,
            "total_chunks": total_chunks,
            "avg_chunks_per_file": total_chunks / max(file_count, 1),
            "batch_size": batch_size,
            "timestamp": timestamp,
            "date": time.strftime("%Y-%m-%d", time.localtime(timestamp)),
            "hour": time.strftime("%H", time.localtime(timestamp)),
            "success": success,
            "error": error,
            "slow_processing": avg_time_per_file > SLOW_EMBEDDING_THRESHOLD
        }
        
        # Store individual metric
        metric_key = f"{EMBEDDING_METRICS_KEY}:{job_id}"
        redis_client.set_json(metric_key, metrics, expire_seconds=86400 * 7)  # Keep for 7 days
        
        # Update aggregated metrics
        _update_embedding_aggregated_metrics(metrics)
        
        # Log slow processing for investigation
        if avg_time_per_file > SLOW_EMBEDDING_THRESHOLD:
            logger.warning(f"Slow embedding detected: {job_id} took {avg_time_per_file:.2f}s per file")
        
    except Exception as e:
        logger.error(f"Failed to record embedding performance metrics for job {job_id}: {e}")


def _update_embedding_aggregated_metrics(metrics: Dict[str, Any]):
    """Update aggregated embedding performance metrics for dashboard and monitoring."""
    try:
        date_key = f"{EMBEDDING_METRICS_KEY}:daily:{metrics['date']}"
        hour_key = f"{EMBEDDING_METRICS_KEY}:hourly:{metrics['date']}:{metrics['hour']}"
        
        # Update daily aggregates
        daily_stats = redis_client.get_json(date_key) or {
            "date": metrics["date"],
            "total_jobs": 0,
            "successful_jobs": 0,
            "slow_jobs": 0,
            "total_files": 0,
            "total_chunks": 0,
            "total_processing_time": 0.0,
            "avg_processing_time": 0.0,
            "avg_time_per_file": 0.0,
            "avg_chunks_per_file": 0.0
        }
        
        daily_stats["total_jobs"] += 1
        if metrics["success"]:
            daily_stats["successful_jobs"] += 1
        if metrics["slow_processing"]:
            daily_stats["slow_jobs"] += 1
        
        daily_stats["total_files"] += metrics["file_count"]
        daily_stats["total_chunks"] += metrics["total_chunks"]
        daily_stats["total_processing_time"] += metrics["total_processing_time"]
        
        # Calculate averages
        daily_stats["avg_processing_time"] = daily_stats["total_processing_time"] / daily_stats["total_jobs"]
        daily_stats["avg_time_per_file"] = daily_stats["total_processing_time"] / max(daily_stats["total_files"], 1)
        daily_stats["avg_chunks_per_file"] = daily_stats["total_chunks"] / max(daily_stats["total_files"], 1)
        
        redis_client.set_json(date_key, daily_stats, expire_seconds=86400 * 30)  # Keep for 30 days
        
        # Update hourly aggregates (similar structure)
        hourly_stats = redis_client.get_json(hour_key) or {
            "date": metrics["date"],
            "hour": metrics["hour"],
            "total_jobs": 0,
            "successful_jobs": 0,
            "slow_jobs": 0,
            "total_files": 0,
            "avg_time_per_file": 0.0,
            "total_processing_time": 0.0
        }
        
        hourly_stats["total_jobs"] += 1
        if metrics["success"]:
            hourly_stats["successful_jobs"] += 1
        if metrics["slow_processing"]:
            hourly_stats["slow_jobs"] += 1
        
        hourly_stats["total_files"] += metrics["file_count"]
        hourly_stats["total_processing_time"] += metrics["total_processing_time"]
        hourly_stats["avg_time_per_file"] = hourly_stats["total_processing_time"] / max(hourly_stats["total_files"], 1)
        
        redis_client.set_json(hour_key, hourly_stats, expire_seconds=86400 * 7)  # Keep for 7 days
        
    except Exception as e:
        logger.error(f"Failed to update embedding aggregated metrics: {e}")


def get_embedding_performance_stats(days: int = 7) -> Dict[str, Any]:
    """
    Get embedding performance statistics for the specified number of days.
    
    Args:
        days: Number of days to retrieve statistics for
        
    Returns:
        Dictionary containing performance statistics
    """
    try:
        stats = {
            "period_days": days,
            "daily_stats": [],
            "summary": {
                "total_jobs": 0,
                "successful_jobs": 0,
                "slow_jobs": 0,
                "total_files": 0,
                "total_chunks": 0,
                "avg_processing_time": 0.0,
                "avg_time_per_file": 0.0,
                "avg_chunks_per_file": 0.0,
                "success_rate": 0.0,
                "slow_job_rate": 0.0
            }
        }
        
        total_processing_time = 0.0
        
        # Get daily stats for the specified period
        for i in range(days):
            date = time.strftime("%Y-%m-%d", time.localtime(time.time() - i * 86400))
            date_key = f"{EMBEDDING_METRICS_KEY}:daily:{date}"
            
            daily_data = redis_client.get_json(date_key)
            if daily_data:
                stats["daily_stats"].append(daily_data)
                
                # Update summary
                stats["summary"]["total_jobs"] += daily_data["total_jobs"]
                stats["summary"]["successful_jobs"] += daily_data["successful_jobs"]
                stats["summary"]["slow_jobs"] += daily_data["slow_jobs"]
                stats["summary"]["total_files"] += daily_data["total_files"]
                stats["summary"]["total_chunks"] += daily_data["total_chunks"]
                total_processing_time += daily_data["total_processing_time"]
        
        # Calculate summary rates
        total_jobs = stats["summary"]["total_jobs"]
        total_files = stats["summary"]["total_files"]
        
        if total_jobs > 0:
            stats["summary"]["avg_processing_time"] = total_processing_time / total_jobs
            stats["summary"]["success_rate"] = stats["summary"]["successful_jobs"] / total_jobs
            stats["summary"]["slow_job_rate"] = stats["summary"]["slow_jobs"] / total_jobs
        
        if total_files > 0:
            stats["summary"]["avg_time_per_file"] = total_processing_time / total_files
            stats["summary"]["avg_chunks_per_file"] = stats["summary"]["total_chunks"] / total_files
        
        return stats
        
    except Exception as e:
        logger.error(f"Failed to get embedding performance stats: {e}")
        return {"error": str(e)}


@celery_app.task(bind=True, name="process_document_embedding", 
                autoretry_for=(Exception,), retry_kwargs={'max_retries': 3, 'countdown': 60})
def process_document_embedding(self, job_id: str, user_id: str, group_id: str, file_paths: List[str]):
    """
    Process document embedding using the document processor service.
    
    Args:
        job_id: Unique job identifier
        user_id: User who submitted the job (for isolation)
        group_id: Group/destination for the documents (for isolation)
        file_paths: List of file paths to process
        
    Returns:
        Dict with processing results
        
    Raises:
        ValueError: For invalid input or processing errors
        Retry: For retryable errors
    """
    start_time = time.time()
    
    try:
        # Validate inputs
        if not job_id or not user_id or not group_id or not file_paths:
            raise ValueError("Missing required parameters")
        
        if not isinstance(file_paths, list) or len(file_paths) == 0:
            raise ValueError("file_paths must be a non-empty list")
        
        # Update job status to processing (this will trigger WebSocket notification)
        job_manager.update_job_status(job_id, JobStatus.PROCESSING)
        
        logger.info(f"Starting embedding job {job_id} for user {user_id}, group {group_id}, {len(file_paths)} files")
        
        # Update progress
        update_job_progress(job_id, 0.1, "Starting document processing...")
        
        # Use the document processor service
        result = document_processor.process_documents(
            file_paths=file_paths,
            user_id=user_id,
            group_id=group_id,
            job_id=job_id,
            chunk_size=CHUNK_SIZE,
            chunk_overlap=CHUNK_OVERLAP,
            db_path=DB_PATH,
            table_name=TABLE_NAME,
            embed_model_name=EMBED_MODEL_NAME,
            device="cpu"
        )
        
        if not result.success:
            raise ValueError(result.error or "Document processing failed")
        
        # Store document metadata for processed files
        processed_files = []
        failed_files = []
        
        for file_path in file_paths:
            try:
                filename = os.path.basename(file_path)
                doc_info = get_document_info(file_path)
                
                if "error" in doc_info:
                    failed_files.append({
                        "filename": filename,
                        "error": doc_info["error"]
                    })
                    continue
                
                # Store document metadata
                document_id = store_document_metadata(
                    user_id, group_id, file_path,
                    doc_info["page_count"], 
                    result.chunk_count // result.document_count  # Approximate chunks per file
                )
                
                processed_files.append({
                    "filename": filename,
                    "status": "processed",
                    "page_count": doc_info["page_count"],
                    "document_id": document_id,
                    "file_size": doc_info["file_size"]
                })
                
            except Exception as e:
                logger.error(f"Failed to store metadata for {os.path.basename(file_path)}: {e}")
                failed_files.append({
                    "filename": os.path.basename(file_path),
                    "error": f"Metadata storage failed: {e}"
                })
        
        # Final progress update
        update_job_progress(job_id, 1.0, "Embedding completed successfully")
        
        # Prepare final result
        final_result = {
            "total_files": len(file_paths),
            "processed_files": len(processed_files),
            "failed_files": len(failed_files),
            "total_chunks": result.chunk_count,
            "processed_details": processed_files,
            "failed_details": failed_files,
            "processing_time": result.processing_time,
            "message": f"Successfully processed {len(processed_files)} of {len(file_paths)} files"
        }
        
        # Record performance metrics
        record_embedding_performance_metrics(
            job_id, user_id, len(file_paths), result.processing_time,
            result.chunk_count, 5, True  # Default batch size of 5
        )
        
        # Update job status to completed (this will trigger WebSocket notification)
        job_manager.update_job_status(job_id, JobStatus.COMPLETED, result=final_result)
        
        logger.info(f"Completed embedding job {job_id}: {len(processed_files)} processed, {len(failed_files)} failed")
        return final_result
        
    except Exception as e:
        logger.error(f"Embedding job {job_id} failed: {e}")
        
        # Check if this is a retryable error
        if isinstance(e, (ConnectionError, TimeoutError)) and self.request.retries < 3:
            logger.info(f"Retrying job {job_id} due to {type(e).__name__}")
            raise self.retry(countdown=60, exc=e)
        
        # Record performance metrics for failed job
        total_processing_time = time.time() - start_time
        file_count = len(file_paths) if file_paths else 0
        record_embedding_performance_metrics(
            job_id, user_id, file_count, total_processing_time,
            0, 5, False, str(e)  # Default batch size of 5
        )
        
        # Mark job as failed (this will trigger WebSocket notification)
        error_message = str(e)
        job_manager.update_job_status(job_id, JobStatus.FAILED, error=error_message)
        
        # Re-raise for Celery error handling
        raise


@celery_app.task(name="cleanup_failed_embeddings")
def cleanup_failed_embeddings(user_id: str, job_id: str):
    """
    Cleanup task for failed embedding jobs.
    Removes any partially processed data.
    """
    try:
        logger.info(f"Cleaning up failed embedding job {job_id} for user {user_id}")
        
        # Remove job data from Redis
        redis_client.redis_client.delete(f"job:{job_id}")
        
        # Could add more cleanup logic here if needed
        # (e.g., remove partially indexed documents)
        
        logger.info(f"Cleanup completed for job {job_id}")
        
    except Exception as e:
        logger.error(f"Failed to cleanup job {job_id}: {e}")


@celery_app.task(name="get_embedding_job_status")
def get_embedding_job_status(job_id: str) -> Dict[str, Any]:
    """
    Get the current status of an embedding job.
    
    Args:
        job_id: Job identifier
        
    Returns:
        Dict with job status information
    """
    try:
        job_data = redis_client.get_json(f"job:{job_id}")
        if not job_data:
            return {"error": "Job not found"}
        
        return job_data
        
    except Exception as e:
        logger.error(f"Failed to get job status for {job_id}: {e}")
        return {"error": str(e)}