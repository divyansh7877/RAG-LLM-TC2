"""
Celery worker for document embedding tasks.
"""
from celery import current_task
from .celery_app import celery_app
from ..shared.redis_client import redis_client
from ..shared.models import JobStatus
import json
import time


@celery_app.task(bind=True, name="process_document_embedding")
def process_document_embedding(self, job_id: str, user_id: str, group_id: str, file_paths: list):
    """
    Process document embedding with progress tracking.
    
    Args:
        job_id: Unique job identifier
        user_id: User who submitted the job
        group_id: Group/destination for the documents
        file_paths: List of file paths to process
    """
    try:
        # Update job status to processing
        job_data = {
            "job_id": job_id,
            "user_id": user_id,
            "job_type": "embedding",
            "status": JobStatus.PROCESSING.value,
            "progress": 0.0,
            "started_at": time.time()
        }
        redis_client.set_json(f"job:{job_id}", job_data)
        
        # TODO: Implement actual embedding logic in subsequent tasks
        # For now, simulate processing
        total_files = len(file_paths)
        for i, file_path in enumerate(file_paths):
            # Simulate processing time
            time.sleep(1)
            
            # Update progress
            progress = (i + 1) / total_files
            job_data["progress"] = progress
            redis_client.set_json(f"job:{job_id}", job_data)
            
            # Update Celery task state
            self.update_state(
                state="PROGRESS",
                meta={"current": i + 1, "total": total_files, "status": f"Processing {file_path}"}
            )
        
        # Mark job as completed
        job_data.update({
            "status": JobStatus.COMPLETED.value,
            "progress": 1.0,
            "completed_at": time.time(),
            "result": {"files_processed": total_files, "message": "Embedding completed successfully"}
        })
        redis_client.set_json(f"job:{job_id}", job_data)
        
        return {"status": "completed", "files_processed": total_files}
        
    except Exception as e:
        # Mark job as failed
        job_data = {
            "job_id": job_id,
            "user_id": user_id,
            "job_type": "embedding",
            "status": JobStatus.FAILED.value,
            "progress": 0.0,
            "error": str(e),
            "completed_at": time.time()
        }
        redis_client.set_json(f"job:{job_id}", job_data)
        
        # Re-raise exception for Celery
        raise