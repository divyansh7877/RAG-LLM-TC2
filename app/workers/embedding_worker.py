#!/usr/bin/env python3
"""
Celery worker for document embedding tasks.

This worker is a thin wrapper around the DocumentProcessor service. Its main
responsibilities are to manage job state and call the processor.
"""
import logging
from typing import List
import hashlib
import os
import shutil
from datetime import datetime

from .celery_app import celery_app
from ..shared.models import JobStatus
from ..shared.config import config
from ..shared.job_manager import job_manager
from ..shared.document_processor import DocumentProcessor
from ..shared.redis_client import redis_client
from ..shared.models import Document as DocModel
from ..shared.job_notifications import job_notification_service

logger = logging.getLogger(__name__)

# --- Worker Task ---
@celery_app.task(bind=True, name="process_document_embedding", 
                autoretry_for=(Exception,), retry_kwargs={'max_retries': 2, 'countdown': 60})
def process_document_embedding(self, job_id: str, user_id: str, group_id: str, file_paths: List[str]):
    """
    Celery task to process and embed a list of documents.
    
    Args:
        job_id: The ID for the job being processed.
        user_id: The ID of the user who owns the documents.
        group_id: The group ID to associate with the documents.
        file_paths: A list of absolute paths to the document files.
    """
    logger.info(f"Embedding job {job_id} started for user '{user_id}'.")
    job_manager.update_job_status(job_id, JobStatus.PROCESSING)
    try:
        # best-effort notify over websockets that job started
        import asyncio
        asyncio.run(job_notification_service.broadcast_progress_update(job_id))
    except Exception:
        pass

    try:
        # Initialize the processor with configuration from the central config
        processor = DocumentProcessor(
            db_path=config.LANCEDB_PATH,
            table_name="document_embeddings",
            embed_model_name=config.EMBEDDING_MODEL_PATH,
            device="cuda" if config.HAS_CUDA else "cpu"
        )

        # Execute the processing task
        # Inform some initial progress
        job_manager.update_job_progress(job_id, 0.05, "Initializing embedding model and vector store")
        try:
            import asyncio
            asyncio.run(job_notification_service.broadcast_progress_update(job_id))
        except Exception:
            pass

        result = processor.process_documents(
            file_paths=file_paths,
            user_id=user_id,
            group_id=group_id
        )

        if result.success:
            logger.info(f"Embedding job {job_id} completed successfully.")
            job_manager.update_job_status(job_id, JobStatus.COMPLETED, result=result.to_dict())
            # Cleanup temp upload dir
            try:
                job = redis_client.get_job(job_id)
                temp_dir = job.metadata.get("temp_dir") if job and job.metadata else None
                if temp_dir and os.path.isdir(temp_dir):
                    shutil.rmtree(temp_dir, ignore_errors=True)
            except Exception:
                pass
            try:
                import asyncio
                asyncio.run(job_notification_service.broadcast_progress_update(job_id))
            except Exception:
                pass
            return result.to_dict()
        else:
            raise RuntimeError(result.error)

    except Exception as e:
        logger.error(f"Embedding job {job_id} failed: {e}", exc_info=True)
        job_manager.update_job_status(job_id, JobStatus.FAILED, error=str(e))
        # Cleanup temp upload dir on failure as well
        try:
            job = redis_client.get_job(job_id)
            temp_dir = job.metadata.get("temp_dir") if job and job.metadata else None
            if temp_dir and os.path.isdir(temp_dir):
                shutil.rmtree(temp_dir, ignore_errors=True)
        except Exception:
            pass
        try:
            import asyncio
            asyncio.run(job_notification_service.broadcast_progress_update(job_id))
        except Exception:
            pass
        # The task will be retried automatically by Celery based on the decorator config
        raise


# --- Lightweight utilities used by integration tests ---
def calculate_file_hash(file_path: str) -> str:
    sha256_hash = hashlib.sha256()
    with open(file_path, "rb") as f:
        for byte_block in iter(lambda: f.read(4096), b""):
            sha256_hash.update(byte_block)
    return sha256_hash.hexdigest()


def store_document_metadata(user_id: str, group_id: str, file_path: str, page_count: int, chunk_count: int) -> str:
    document = DocModel(
        user_id=user_id,
        group_id=group_id,
        filename=os.path.basename(file_path),
        file_size=os.path.getsize(file_path) if os.path.exists(file_path) else 0,
        processing_status="completed",
        page_count=page_count,
        chunk_count=chunk_count,
        file_hash=calculate_file_hash(file_path) if os.path.exists(file_path) else None,
    )
    # Key format compatible with API listing helpers
    key = f"document:{user_id}:{group_id}:{document.document_id}"
    redis_client.set_json(key, document.to_dict())
    return document.document_id


def get_embedding_job_status(job_id: str) -> dict:
    job = redis_client.get_job(job_id)
    if not job:
        return {"error": "job_not_found"}
    return job.to_dict()
