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
from ..shared.gpu_memory_manager import gpu_memory_manager
from ..shared.embedding_optimizer import clear_embedding_model
from ..shared.query_engine_factory import query_engine_factory

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
    logger.info(f"Embedding job {job_id} started for user '{user_id}'/'{group_id}'.")
    job_manager.update_job_status(job_id, JobStatus.PROCESSING)
    
    # Log initial GPU memory state
    initial_memory = gpu_memory_manager.get_memory_info()
    if initial_memory:
        logger.info(f"Initial GPU memory usage: {initial_memory.utilization_percent:.1f}%")
    
    try:
        # best-effort notify over websockets that job started
        import asyncio
        asyncio.run(job_notification_service.broadcast_progress_update(job_id))
    except Exception:
        pass

    try:
        # Clear any existing models and GPU cache to start fresh
        clear_embedding_model()
        gpu_memory_manager.clear_cache()
        
        # Initialize the processor with configuration from the central config
        table_name = os.getenv("LANCEDB_TABLE_NAME", "document_embeddings_v2")
        device = "cuda" if config.HAS_CUDA else "cpu"
        logger.info(
            f"Preparing DocumentProcessor with db_path={os.path.abspath(config.LANCEDB_PATH)}, "
            f"table={table_name}, model={config.EMBEDDING_MODEL_PATH}, device={device}"
        )
        
        # Check GPU memory before initializing processor
        if device == "cuda":
            memory_info = gpu_memory_manager.get_memory_info()
            if memory_info:
                logger.info(f"GPU memory before processor init: {memory_info.utilization_percent:.1f}%")
        
        processor = DocumentProcessor(
            db_path=os.path.abspath(config.LANCEDB_PATH),
            table_name=table_name,
            embed_model_name=config.EMBEDDING_MODEL_PATH,
            device=device
        )

        # Execute the processing task
        # Inform some initial progress
        job_manager.update_job_progress(job_id, 0.05, "Initializing embedding model and vector store")
        try:
            import asyncio
            asyncio.run(job_notification_service.broadcast_progress_update(job_id))
        except Exception:
            pass

        logger.info("Calling processor.process_documents with GPU memory management...")
        
        try:
            result = processor.process_documents(
                file_paths=file_paths,
                user_id=user_id,
                group_id=group_id
            )
            logger.info("processor.process_documents returned result")
            
        except Exception as process_error:
            # Check if it's a GPU memory issue
            error_str = str(process_error).lower()
            if "cuda out of memory" in error_str or "out of memory" in error_str:
                logger.warning(f"GPU OOM detected during processing: {process_error}")
                
                # Clear everything and try CPU fallback
                clear_embedding_model()
                gpu_memory_manager.clear_cache()
                
                logger.info("Attempting CPU fallback for entire processing pipeline")
                cpu_processor = DocumentProcessor(
                    db_path=os.path.abspath(config.LANCEDB_PATH),
                    table_name=table_name,
                    embed_model_name=config.EMBEDDING_MODEL_PATH,
                    device="cpu"
                )
                
                result = cpu_processor.process_documents(
                    file_paths=file_paths,
                    user_id=user_id,
                    group_id=group_id
                )
                logger.info("CPU fallback processing completed successfully")
            else:
                # Re-raise non-memory related errors
                raise

        if result.success:
            logger.info(f"Embedding job {job_id} completed successfully.")
            job_manager.update_job_progress(job_id, 1.0)
            job_manager.update_job_status(job_id, JobStatus.COMPLETED, result=result.to_dict())

            # Store metadata for each document
            for file_path in file_paths:
                try:
                    store_document_metadata(
                        user_id=user_id,
                        group_id=group_id,
                        file_path=file_path,
                        page_count=0,  # Placeholder, as DocumentProcessor doesn't return this yet
                        chunk_count=0  # Placeholder
                    )
                    logger.info(f"Stored metadata for {os.path.basename(file_path)}")
                except Exception as e:
                    logger.error(f"Failed to store metadata for {os.path.basename(file_path)}: {e}")
            # Log final GPU memory state
            final_memory = gpu_memory_manager.get_memory_info()
            if final_memory:
                logger.info(f"Final GPU memory usage: {final_memory.utilization_percent:.1f}%")
            
            # Cleanup temp upload dir
            try:
                job = redis_client.get_job(job_id)
                temp_dir = job.metadata.get("temp_dir") if job and job.metadata else None
                if temp_dir and os.path.isdir(temp_dir):
                    shutil.rmtree(temp_dir, ignore_errors=True)
            except Exception:
                pass
            
            # Invalidate query engine vector store to ensure fresh data retrieval
            # This fixes the bug where all queries were getting the same old documents
            try:
                query_engine_factory.invalidate_vector_store()
                logger.info(f"Invalidated query engine vector store after processing job {job_id}")
            except Exception as invalidation_error:
                logger.warning(f"Failed to invalidate vector store for job {job_id}: {invalidation_error}")
            
            # Final GPU cleanup
            clear_embedding_model()
            gpu_memory_manager.clear_cache()
            
            try:
                import asyncio
                asyncio.run(job_notification_service.broadcast_progress_update(job_id))
            except Exception:
                pass
            return result.to_dict()
        else:
            raise RuntimeError(result.error)

    except Exception as e:
        error_str = str(e).lower()
        is_gpu_oom = "cuda out of memory" in error_str or "out of memory" in error_str
        
        if is_gpu_oom:
            logger.error(f"Embedding job {job_id} failed with GPU OOM: {e}", exc_info=True)
        else:
            logger.error(f"Embedding job {job_id} failed: {e}", exc_info=True)
        
        # Always cleanup GPU memory on failure
        try:
            clear_embedding_model()
            gpu_memory_manager.clear_cache()
        except Exception as cleanup_error:
            logger.warning(f"Failed to cleanup GPU memory: {cleanup_error}")

        # Determine if Celery will retry this task
        will_retry = False
        try:
            current_retries = getattr(self.request, "retries", 0)
            max_retries = getattr(self, "max_retries", 0)
            will_retry = current_retries < max_retries
        except Exception:
            will_retry = False

        if will_retry:
            # Do NOT mark as FAILED yet; keep job in processing and retain temp files for retry
            if is_gpu_oom:
                retry_message = f"GPU OOM detected, retrying with CPU fallback: {str(e)}"
            else:
                retry_message = f"Retrying after error: {str(e)}"
            job_manager.update_job_progress(job_id, 0.05, retry_message)
        else:
            # Final failure: mark failed and cleanup temp upload directory
            job_manager.update_job_progress(job_id, 1.0)
            
            if is_gpu_oom:
                error_message = f"Processing failed due to insufficient GPU memory: {str(e)}"
            else:
                error_message = str(e)
                
            job_manager.update_job_status(job_id, JobStatus.FAILED, error=error_message)
            
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
        # Re-raise to allow Celery autoretry/final failure handling
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
