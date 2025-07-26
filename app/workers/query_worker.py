"""
Celery worker for query processing tasks with security isolation and caching.
"""
import os
import time
import logging
import hashlib
from typing import List, Dict, Any, Optional, Tuple
from celery import current_task
from celery.exceptions import Retry

from .celery_app import celery_app
from ..shared.redis_client import redis_client
from ..shared.models import JobStatus, Query
from ..shared.config import config
from ..shared.job_manager import job_manager
from ..shared.query_engine_factory import query_engine_factory

# Configure logging
logger = logging.getLogger(__name__)

# Constants
DB_PATH = config.LANCEDB_PATH
TABLE_NAME = "document_embeddings"
EMBED_MODEL_NAME = config.EMBEDDING_MODEL_PATH
LLM_MODEL_PATH = "./models/Llama-3.2-3B-Instruct-IQ3_M.gguf"
CACHE_EXPIRE_SECONDS = 3600  # 1 hour cache expiration
SIMILARITY_THRESHOLD = 0.7
MAX_RETRIEVED_NODES = 10


class QuerySecurityError(Exception):
    """Exception raised for query security violations."""
    pass


def validate_query_security(user_id: str, group_ids: List[str], query_text: str) -> None:
    """
    Validate query for security issues and potential data leakage attempts.
    
    Args:
        user_id: User identifier
        group_ids: List of group IDs
        query_text: Query text to validate
        
    Raises:
        QuerySecurityError: If security validation fails
    """
    if not user_id or not user_id.strip():
        raise QuerySecurityError("Invalid user ID")
    
    if not group_ids or not all(gid.strip() for gid in group_ids):
        raise QuerySecurityError("Invalid group IDs")
    
    if not query_text or not query_text.strip():
        raise QuerySecurityError("Query text cannot be empty")
    
    # Check for potential injection attempts or suspicious patterns
    suspicious_patterns = [
        "user_id:",
        "group_id:",
        "metadata:",
        "__",  # Double underscore might indicate internal field access
        "SELECT",
        "DROP",
        "DELETE",
        "UPDATE",
        "INSERT"
    ]
    
    query_lower = query_text.lower()
    for pattern in suspicious_patterns:
        if pattern.lower() in query_lower:
            logger.warning(f"Suspicious query pattern detected: {pattern} in query from user {user_id}")
            # Don't raise error for now, just log - could be legitimate query
    
    # Check query length to prevent abuse
    if len(query_text) > 2000:
        raise QuerySecurityError("Query text too long (max 2000 characters)")


def extract_source_info(response) -> List[str]:
    """
    Extract source document information from query response.
    
    Args:
        response: Query engine response object
        
    Returns:
        List of source document names
    """
    sources = []
    try:
        if hasattr(response, 'source_nodes') and response.source_nodes:
            for node in response.source_nodes:
                if hasattr(node, 'metadata') and node.metadata:
                    doc_name = node.metadata.get('document_name', 'Unknown Document')
                    page_num = node.metadata.get('page_number', '')
                    if page_num:
                        source_info = f"{doc_name} (Page {page_num})"
                    else:
                        source_info = doc_name
                    
                    if source_info not in sources:
                        sources.append(source_info)
                else:
                    # Handle nodes with missing or empty metadata
                    sources.append("Unknown Document")
        
        return sources[:5]  # Limit to top 5 sources
        
    except Exception as e:
        logger.warning(f"Failed to extract source info: {e}")
        return ["Source information unavailable"]


def update_query_progress(query_id: str, progress: float, status_message: str = None):
    """Update query progress using job manager and trigger WebSocket notifications."""
    try:
        # Update job progress through job manager (this will trigger WebSocket notifications)
        success = job_manager.update_job_progress(query_id, progress, status_message)
        
        if not success:
            logger.warning(f"Failed to update query progress for {query_id}")
        
        # Update Celery task state for Celery monitoring
        if current_task:
            current_task.update_state(
                state="PROGRESS",
                meta={
                    "progress": progress,
                    "status": status_message or "Processing query...",
                    "timestamp": time.time()
                }
            )
    except Exception as e:
        logger.error(f"Failed to update query progress: {e}")


@celery_app.task(bind=True, name="process_user_query", 
                autoretry_for=(ConnectionError, TimeoutError), 
                retry_kwargs={'max_retries': 2, 'countdown': 30})
def process_user_query(self, query_id: str, user_id: str, group_ids: List[str], query_text: str):
    """
    Process user query with comprehensive security isolation and performance optimization.
    
    Args:
        query_id: Unique query identifier
        user_id: User who submitted the query (for security isolation)
        group_ids: List of groups user has access to (for security filtering)
        query_text: The query text to process
        
    Returns:
        Dict with query results including answer, sources, and metadata
        
    Raises:
        QuerySecurityError: For security violations
        ValueError: For invalid input or processing errors
        Retry: For retryable errors
    """
    start_time = time.time()
    
    try:
        # Validate inputs and security
        validate_query_security(user_id, group_ids, query_text)
        
        # Update job status to processing (this will trigger WebSocket notification)
        job_manager.update_job_status(query_id, JobStatus.PROCESSING)
        
        logger.info(f"Starting query {query_id} for user {user_id} with groups {group_ids}")
        
        # Check cache first using the new factory cache
        update_query_progress(query_id, 0.1, "Checking cache...")
        cached_result = query_engine_factory.get_cached_query_result(user_id, group_ids, query_text)
        
        if cached_result:
            # Return cached result
            processing_time = time.time() - start_time
            cached_result["processing_time"] = processing_time
            cached_result["cached"] = True
            
            # Update job status to completed (this will trigger WebSocket notification)
            job_manager.update_job_status(query_id, JobStatus.COMPLETED, result=cached_result)
            
            logger.info(f"Query {query_id} completed from cache in {processing_time:.2f}s")
            return cached_result
        
        # Create query engine using the thread-safe factory
        update_query_progress(query_id, 0.3, "Initializing query engine...")
        query_engine = query_engine_factory.create_query_engine(user_id, group_ids)
        
        # Process query
        update_query_progress(query_id, 0.6, "Processing query...")
        response = query_engine.query(query_text)
        
        # Extract results
        update_query_progress(query_id, 0.9, "Extracting results...")
        answer = str(response.response) if response.response else "No answer found."
        sources = extract_source_info(response)
        processing_time = time.time() - start_time
        
        # Prepare result
        result = {
            "answer": answer,
            "sources": sources,
            "processing_time": processing_time,
            "cached": False,
            "result_count": len(sources),
            "query_metadata": {
                "similarity_threshold": SIMILARITY_THRESHOLD,
                "max_retrieved_nodes": MAX_RETRIEVED_NODES,
                "user_groups": group_ids
            }
        }
        
        # Cache the result for future queries using the factory cache
        query_engine_factory.cache_query_result(user_id, group_ids, query_text, result)
        
        # Update job status to completed (this will trigger WebSocket notification)
        job_manager.update_job_status(query_id, JobStatus.COMPLETED, result=result)
        
        logger.info(f"Query {query_id} completed successfully in {processing_time:.2f}s with {len(sources)} sources")
        return result
        
    except QuerySecurityError as e:
        logger.error(f"Security violation in query {query_id}: {e}")
        error_message = f"Security validation failed: {e}"
        
        # Update job status to failed (this will trigger WebSocket notification)
        job_manager.update_job_status(query_id, JobStatus.FAILED, error=error_message)
        
        # Don't retry security errors
        raise ValueError(error_message)
        
    except Exception as e:
        logger.error(f"Query {query_id} failed: {e}")
        error_message = str(e)
        
        # Check if this is a retryable error
        if isinstance(e, (ConnectionError, TimeoutError)) and self.request.retries < 2:
            logger.info(f"Retrying query {query_id} due to {type(e).__name__}")
            raise self.retry(countdown=30, exc=e)
        
        # Update job status to failed (this will trigger WebSocket notification)
        job_manager.update_job_status(query_id, JobStatus.FAILED, error=error_message)
        
        # Re-raise for Celery error handling
        raise


@celery_app.task(name="get_query_status")
def get_query_status(query_id: str) -> Dict[str, Any]:
    """
    Get the current status of a query.
    
    Args:
        query_id: Query identifier
        
    Returns:
        Dict with query status information
    """
    try:
        query_data = redis_client.get_json(f"query:{query_id}")
        if not query_data:
            return {"error": "Query not found"}
        
        return query_data
        
    except Exception as e:
        logger.error(f"Failed to get query status for {query_id}: {e}")
        return {"error": str(e)}


@celery_app.task(name="cleanup_query_cache")
def cleanup_query_cache():
    """
    Cleanup expired query cache entries and database connections.
    This task should be run periodically.
    """
    try:
        # Clean up the factory's cache and connections
        query_engine_factory.cleanup()
        
        # Get cache stats for reporting
        stats = query_engine_factory.get_factory_stats()
        cache_stats = stats.get("query_cache", {})
        
        logger.info(f"Query engine factory cleanup completed")
        return {
            "cache_stats": cache_stats,
            "connection_pool_stats": stats.get("connection_pool", {}),
            "cleanup_completed": True
        }
        
    except Exception as e:
        logger.error(f"Failed to cleanup query cache: {e}")
        return {"error": str(e)}