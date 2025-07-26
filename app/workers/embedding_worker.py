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
import fitz  # PyMuPDF
import lancedb
from llama_index.core import Settings, StorageContext, VectorStoreIndex
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.schema import Document as LlamaDocument
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.vector_stores.lancedb import LanceDBVectorStore
import re
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


def extract_pages_from_pdf(pdf_path: str) -> List[Tuple[str, int]]:
    """Extract text from PDF pages with error handling."""
    try:
        doc = fitz.open(pdf_path)
        pages: List[Tuple[str, int]] = []
        for i in range(len(doc)):
            try:
                page_text = doc.load_page(i).get_text()
                pages.append((page_text, i + 1))
            except Exception as e:
                logger.warning(f"Failed to extract text from page {i+1} of {pdf_path}: {e}")
                pages.append(("", i + 1))  # Add empty page to maintain page numbering
        doc.close()
        return pages
    except Exception as e:
        logger.error(f"Failed to open PDF {pdf_path}: {e}")
        raise ValueError(f"Cannot process PDF file: {e}")


def clean_text(raw_text: str) -> str:
    """Clean and normalize text content."""
    if not raw_text:
        return ""
    
    # Basic cleanup: trim spaces and collapse excess newlines/spaces
    txt = re.sub(r"\n{3,}", "\n\n", raw_text)
    txt = "\n".join(line.strip() for line in txt.split("\n"))
    txt = re.sub(r" {2,}", " ", txt)
    return txt.strip()


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


def create_nodes_from_pdf(pdf_path: str, user_id: str, group_id: str) -> List:
    """Create LlamaIndex nodes from a single PDF with user isolation metadata."""
    document_name = os.path.basename(pdf_path)
    file_size = os.path.getsize(pdf_path)
    file_hash = calculate_file_hash(pdf_path)
    
    # Extract pages
    pages = extract_pages_from_pdf(pdf_path)
    
    # Create LlamaIndex documents
    documents: List[LlamaDocument] = []
    for text, page_number in pages:
        cleaned_text = clean_text(text)
        if not cleaned_text:  # Skip empty pages
            continue
            
        metadata = {
            "document_name": document_name,
            "page_number": page_number,
            "user_id": user_id,
            "group_id": group_id,
            "file_size": file_size,
            "file_hash": file_hash,
            "upload_date": time.time(),
            "content_type": "application/pdf"
        }
        
        documents.append(
            LlamaDocument(
                text=cleaned_text,
                metadata=metadata,
                id_=f"{user_id}_{group_id}_{document_name}_p{page_number}",
            )
        )
    
    if not documents:
        raise ValueError(f"No readable content found in PDF: {document_name}")
    
    # Split documents into chunks
    splitter = SentenceSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        include_metadata=True,
    )
    
    nodes = splitter.get_nodes_from_documents(documents)
    logger.info(f"Created {len(nodes)} nodes from {document_name} ({len(pages)} pages)")
    
    return nodes


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


@celery_app.task(bind=True, name="process_document_embedding", 
                autoretry_for=(Exception,), retry_kwargs={'max_retries': 3, 'countdown': 60})
def process_document_embedding(self, job_id: str, user_id: str, group_id: str, file_paths: List[str]):
    """
    Process document embedding with comprehensive progress tracking and user isolation.
    
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
    temp_dir = None
    processed_files = []
    failed_files = []
    
    try:
        # Validate inputs
        if not job_id or not user_id or not group_id or not file_paths:
            raise ValueError("Missing required parameters")
        
        if not isinstance(file_paths, list) or len(file_paths) == 0:
            raise ValueError("file_paths must be a non-empty list")
        
        # Update job status to processing (this will trigger WebSocket notification)
        job_manager.update_job_status(job_id, JobStatus.PROCESSING)
        
        logger.info(f"Starting embedding job {job_id} for user {user_id}, group {group_id}, {len(file_paths)} files")
        
        # Initialize embedding model (singleton per worker)
        Settings.embed_model = HuggingFaceEmbedding(
            model_name=EMBED_MODEL_NAME,
            device="cpu",  # Use CPU for stability in worker environment
            trust_remote_code=True,
        )
        
        # Connect to LanceDB
        ldb = lancedb.connect(DB_PATH)
        vector_store = LanceDBVectorStore(uri=DB_PATH, table_name=TABLE_NAME)
        storage_context = StorageContext.from_defaults(vector_store=vector_store)
        
        # Process each file
        total_files = len(file_paths)
        all_nodes = []
        
        for i, file_path in enumerate(file_paths):
            try:
                # Validate file exists and is readable
                if not os.path.exists(file_path):
                    raise ValueError(f"File not found: {file_path}")
                
                if not os.access(file_path, os.R_OK):
                    raise ValueError(f"File not readable: {file_path}")
                
                filename = os.path.basename(file_path)
                update_job_progress(job_id, i / total_files, f"Processing {filename}...")
                
                # Check for duplicates
                file_hash = calculate_file_hash(file_path)
                if check_duplicate_document(user_id, group_id, file_hash):
                    logger.info(f"Skipping duplicate file: {filename}")
                    processed_files.append({
                        "filename": filename,
                        "status": "skipped",
                        "reason": "duplicate"
                    })
                    continue
                
                # Create nodes from PDF
                nodes = create_nodes_from_pdf(file_path, user_id, group_id)
                all_nodes.extend(nodes)
                
                # Store document metadata
                page_count = len(extract_pages_from_pdf(file_path))
                chunk_count = len(nodes)
                document_id = store_document_metadata(
                    user_id, group_id, file_path, page_count, chunk_count
                )
                
                processed_files.append({
                    "filename": filename,
                    "document_id": document_id,
                    "status": "processed",
                    "page_count": page_count,
                    "chunk_count": chunk_count
                })
                
                logger.info(f"Successfully processed {filename}: {page_count} pages, {chunk_count} chunks")
                
            except Exception as e:
                logger.error(f"Failed to process file {file_path}: {e}")
                failed_files.append({
                    "filename": os.path.basename(file_path),
                    "error": str(e)
                })
                
                # Continue processing other files
                continue
        
        # Update progress for embedding phase
        update_job_progress(job_id, 0.8, "Creating vector embeddings...")
        
        # Create or update vector index if we have nodes
        if all_nodes:
            try:
                index = VectorStoreIndex(
                    all_nodes, 
                    storage_context=storage_context, 
                    show_progress=False  # Disable progress bar in worker
                )
                
                # Persist index metadata
                persist_dir = os.path.join(DB_PATH, "li_storage")
                index.storage_context.persist(persist_dir)
                
                # Verify storage
                table = ldb.open_table(TABLE_NAME)
                total_vectors = table.count_rows()
                logger.info(f"Successfully stored {len(all_nodes)} new vectors. Total vectors in DB: {total_vectors}")
                
            except Exception as e:
                logger.error(f"Failed to create vector index: {e}")
                raise ValueError(f"Vector indexing failed: {e}")
        
        # Final progress update
        update_job_progress(job_id, 1.0, "Embedding completed successfully")
        
        # Mark job as completed
        result = {
            "total_files": total_files,
            "processed_files": len(processed_files),
            "failed_files": len(failed_files),
            "total_chunks": len(all_nodes),
            "processed_details": processed_files,
            "failed_details": failed_files,
            "message": f"Successfully processed {len(processed_files)} of {total_files} files"
        }
        
        # Update job status to completed (this will trigger WebSocket notification)
        job_manager.update_job_status(job_id, JobStatus.COMPLETED, result=result)
        
        logger.info(f"Completed embedding job {job_id}: {len(processed_files)} processed, {len(failed_files)} failed")
        return result
        
    except Exception as e:
        logger.error(f"Embedding job {job_id} failed: {e}")
        
        # Check if this is a retryable error
        if isinstance(e, (ConnectionError, TimeoutError)) and self.request.retries < 3:
            logger.info(f"Retrying job {job_id} due to {type(e).__name__}")
            raise self.retry(countdown=60, exc=e)
        
        # Mark job as failed (this will trigger WebSocket notification)
        error_message = str(e)
        job_manager.update_job_status(job_id, JobStatus.FAILED, error=error_message)
        
        # Re-raise for Celery error handling
        raise
    
    finally:
        # Cleanup temporary files
        if temp_dir and os.path.exists(temp_dir):
            try:
                shutil.rmtree(temp_dir)
            except Exception as e:
                logger.warning(f"Failed to cleanup temp directory {temp_dir}: {e}")


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