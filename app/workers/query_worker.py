"""
Celery worker for query processing tasks with security isolation and caching.
"""
import os
import time
import logging
import hashlib
from datetime import datetime
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
TABLE_NAME = "document_embeddings_v2"
EMBED_MODEL_NAME = config.EMBEDDING_MODEL_PATH
LLM_MODEL_PATH = "./models/Llama-3.2-3B-Instruct-IQ3_M.gguf"
CACHE_EXPIRE_SECONDS = 3600  # 1 hour cache expiration
SIMILARITY_THRESHOLD = 0.7
MAX_RETRIEVED_NODES = 10

# Performance monitoring constants
PERFORMANCE_METRICS_KEY = "performance:query_metrics"
SLOW_QUERY_THRESHOLD = 5.0  # seconds


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


def generate_cache_key(user_id: str, group_ids: List[str], query_text: str) -> str:
    """
    Generate a cache key for query results.
    
    Args:
        user_id: User identifier
        group_ids: List of group IDs
        query_text: Query text
        
    Returns:
        str: Cache key for the query
    """
    try:
        # Normalize inputs for consistent caching
        normalized_groups = ",".join(sorted(group_ids or []))
        normalized_query = query_text.strip().lower()
        
        # Create cache key payload
        payload = f"u:{user_id}|g:{normalized_groups}|q:{normalized_query}"
        
        # Generate hash
        cache_key = hashlib.sha256(payload.encode("utf-8")).hexdigest()
        return f"query_cache:{cache_key}"
        
    except Exception as e:
        logger.error(f"Failed to generate cache key: {e}")
        # Return a fallback key that won't match anything
        return f"query_cache:error_{time.time()}"


def create_user_security_filters(user_id: str, group_ids: List[str]):
    """
    Create security filters for user isolation.
    
    Args:
        user_id: User identifier
        group_ids: List of group IDs the user has access to
        
    Returns:
        MetadataFilters: Filters that ensure user can only access authorized documents
        
    Raises:
        ValueError: If inputs are invalid
    """
    from llama_index.core.vector_stores import MetadataFilters, ExactMatchFilter
    
    if not user_id or not user_id.strip():
        raise ValueError("User ID is required")
    
    if not group_ids or len(group_ids) == 0:
        raise ValueError("At least one group ID is required")
    
    try:
        # Create user filter
        user_filter = ExactMatchFilter(key="user_id", value=user_id)
        
        # Create group filters
        group_filters = [ExactMatchFilter(key="group_id", value=group_id) for group_id in group_ids]
        
        # Combine filters: user can access their personal docs OR docs from their groups
        all_filters = [user_filter] + group_filters
        
        return MetadataFilters(filters=all_filters, condition="or")
        
    except Exception as e:
        logger.error(f"Failed to create user security filters: {e}")
        raise ValueError(f"Failed to create security filters: {e}")


def extract_source_info(response) -> List[Dict[str, Any]]:
    """
    Extract source document information from query response.
    
    Args:
        response: Query engine response object
        
    Returns:
        List of dictionaries containing source information
    """
    sources = []
    try:
        if hasattr(response, 'source_nodes') and response.source_nodes:
            for node in response.source_nodes:
                if hasattr(node, 'node') and hasattr(node.node, 'metadata') and node.node.metadata:
                    metadata = node.node.metadata
                    doc_name = metadata.get('document_name', 'Unknown Document')
                    page_num = metadata.get('page_number', '')
                    
                    source_info = {
                        "document": doc_name,
                        "page": page_num if page_num else "Unknown"
                    }
                    
                    # Avoid duplicates
                    if source_info not in sources:
                        sources.append(source_info)
                elif hasattr(node, 'metadata') and node.metadata:
                    # Handle different node structure
                    metadata = node.metadata
                    doc_name = metadata.get('document_name', 'Unknown Document')
                    page_num = metadata.get('page_number', '')
                    
                    source_info = {
                        "document": doc_name,
                        "page": page_num if page_num else "Unknown"
                    }
                    
                    if source_info not in sources:
                        sources.append(source_info)
                else:
                    # Handle nodes with missing or empty metadata
                    sources.append({
                        "document": "Unknown Document",
                        "page": "Unknown"
                    })
        
        return sources[:5]  # Limit to top 5 sources
        
    except Exception as e:
        logger.warning(f"Failed to extract source info: {e}")
        return [{"document": "Source information unavailable", "page": "Unknown"}]


def update_query_progress(query_id: str, progress: float, status_message: str = None):
    """Update query progress using job manager and trigger WebSocket notifications."""
    try:
        # Get the job_id from the query data
        query_data = redis_client.get_json(f"query:{query_id}")
        if query_data and "job_id" in query_data:
            job_id = query_data["job_id"]
            # Update job progress through job manager (this will trigger WebSocket notifications)
            success = job_manager.update_job_progress(job_id, progress, status_message)
            
            if not success:
                logger.warning(f"Failed to update query progress for query {query_id}, job {job_id}")
        else:
            logger.warning(f"No job_id found for query {query_id}")
        
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


def record_query_performance_metrics(query_id: str, user_id: str, query_text: str, 
                                   processing_time: float, result_count: int, 
                                   cached: bool, error: Optional[str] = None):
    """
    Record query performance metrics for monitoring and optimization.
    
    Args:
        query_id: Query identifier
        user_id: User who submitted the query
        query_text: The query text
        processing_time: Time taken to process the query
        result_count: Number of results returned
        cached: Whether result was served from cache
        error: Error message if query failed
    """
    try:
        timestamp = time.time()
        
        # Create performance metrics entry
        metrics = {
            "query_id": query_id,
            "user_id": user_id,
            "query_length": len(query_text),
            "processing_time": processing_time,
            "result_count": result_count,
            "cached": cached,
            "timestamp": timestamp,
            "date": time.strftime("%Y-%m-%d", time.localtime(timestamp)),
            "hour": time.strftime("%H", time.localtime(timestamp)),
            "success": error is None,
            "error": error,
            "slow_query": processing_time > SLOW_QUERY_THRESHOLD
        }
        
        # Store individual metric
        metric_key = f"{PERFORMANCE_METRICS_KEY}:{query_id}"
        redis_client.set_json(metric_key, metrics, expire_seconds=86400 * 7)  # Keep for 7 days
        
        # Update aggregated metrics
        _update_aggregated_metrics(metrics)
        
        # Log slow queries for investigation
        if processing_time > SLOW_QUERY_THRESHOLD:
            logger.warning(f"Slow query detected: {query_id} took {processing_time:.2f}s")
        
    except Exception as e:
        logger.error(f"Failed to record performance metrics for query {query_id}: {e}")


def _update_aggregated_metrics(metrics: Dict[str, Any]):
    """Update aggregated performance metrics for dashboard and monitoring."""
    try:
        date_key = f"{PERFORMANCE_METRICS_KEY}:daily:{metrics['date']}"
        hour_key = f"{PERFORMANCE_METRICS_KEY}:hourly:{metrics['date']}:{metrics['hour']}"
        
        # Update daily aggregates
        daily_stats = redis_client.get_json(date_key) or {
            "date": metrics["date"],
            "total_queries": 0,
            "successful_queries": 0,
            "cached_queries": 0,
            "slow_queries": 0,
            "total_processing_time": 0.0,
            "avg_processing_time": 0.0,
            "avg_result_count": 0.0,
            "total_result_count": 0
        }
        
        daily_stats["total_queries"] += 1
        if metrics["success"]:
            daily_stats["successful_queries"] += 1
        if metrics["cached"]:
            daily_stats["cached_queries"] += 1
        if metrics["slow_query"]:
            daily_stats["slow_queries"] += 1
        
        daily_stats["total_processing_time"] += metrics["processing_time"]
        daily_stats["avg_processing_time"] = daily_stats["total_processing_time"] / daily_stats["total_queries"]
        
        daily_stats["total_result_count"] += metrics["result_count"]
        daily_stats["avg_result_count"] = daily_stats["total_result_count"] / daily_stats["total_queries"]
        
        redis_client.set_json(date_key, daily_stats, expire_seconds=86400 * 30)  # Keep for 30 days
        
        # Update hourly aggregates (similar structure)
        hourly_stats = redis_client.get_json(hour_key) or {
            "date": metrics["date"],
            "hour": metrics["hour"],
            "total_queries": 0,
            "successful_queries": 0,
            "cached_queries": 0,
            "slow_queries": 0,
            "avg_processing_time": 0.0,
            "total_processing_time": 0.0
        }
        
        hourly_stats["total_queries"] += 1
        if metrics["success"]:
            hourly_stats["successful_queries"] += 1
        if metrics["cached"]:
            hourly_stats["cached_queries"] += 1
        if metrics["slow_query"]:
            hourly_stats["slow_queries"] += 1
        
        hourly_stats["total_processing_time"] += metrics["processing_time"]
        hourly_stats["avg_processing_time"] = hourly_stats["total_processing_time"] / hourly_stats["total_queries"]
        
        redis_client.set_json(hour_key, hourly_stats, expire_seconds=86400 * 1)  # Keep for 7 days
        
    except Exception as e:
        logger.error(f"Failed to update aggregated metrics: {e}")


def get_query_performance_stats(days: int = 7) -> Dict[str, Any]:
    """
    Get query performance statistics for the specified number of days.
    
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
                "total_queries": 0,
                "successful_queries": 0,
                "cached_queries": 0,
                "slow_queries": 0,
                "avg_processing_time": 0.0,
                "cache_hit_rate": 0.0,
                "success_rate": 0.0,
                "slow_query_rate": 0.0
            }
        }
        
        total_processing_time = 0.0
        
        # Get daily stats for the specified period
        for i in range(days):
            date = time.strftime("%Y-%m-%d", time.localtime(time.time() - i * 86400))
            date_key = f"{PERFORMANCE_METRICS_KEY}:daily:{date}"
            
            daily_data = redis_client.get_json(date_key)
            if daily_data:
                stats["daily_stats"].append(daily_data)
                
                # Update summary
                stats["summary"]["total_queries"] += daily_data["total_queries"]
                stats["summary"]["successful_queries"] += daily_data["successful_queries"]
                stats["summary"]["cached_queries"] += daily_data["cached_queries"]
                stats["summary"]["slow_queries"] += daily_data["slow_queries"]
                total_processing_time += daily_data["total_processing_time"]
        
        # Calculate summary rates
        total_queries = stats["summary"]["total_queries"]
        if total_queries > 0:
            stats["summary"]["avg_processing_time"] = total_processing_time / total_queries
            stats["summary"]["cache_hit_rate"] = stats["summary"]["cached_queries"] / total_queries
            stats["summary"]["success_rate"] = stats["summary"]["successful_queries"] / total_queries
            stats["summary"]["slow_query_rate"] = stats["summary"]["slow_queries"] / total_queries
        
        return stats
        
    except Exception as e:
        logger.error(f"Failed to get query performance stats: {e}")
        return {"error": str(e)}


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
        
        # Get the job_id from the query data
        query_data = redis_client.get_json(f"query:{query_id}")
        job_id = query_data.get("job_id") if query_data else None
        
        if not job_id:
            logger.error(f"No job_id found for query {query_id}")
            raise ValueError(f"No job_id found for query {query_id}")
        
        # Update job status to processing (this will trigger WebSocket notification)
        job_manager.update_job_status(job_id, JobStatus.PROCESSING)
        
        logger.info(f"Starting query {query_id} (job {job_id}) for user {user_id} with groups {group_ids}")
        
        # Check cache first using the new factory cache
        update_query_progress(query_id, 0.1, "Checking cache...")
        cached_result = query_engine_factory.get_cached_query_result(user_id, group_ids, query_text)
        
        if cached_result:
            # Return cached result
            processing_time = time.time() - start_time
            cached_result["processing_time"] = processing_time
            cached_result["cached"] = True
            
            # Record performance metrics for cached result
            record_query_performance_metrics(
                query_id, user_id, query_text, processing_time,
                cached_result.get("result_count", 0), True
            )
            
            # Update job status to completed (this will trigger WebSocket notification)
            job_manager.update_job_status(job_id, JobStatus.COMPLETED, result=cached_result)
            
            # Update query record with cached result
            query_data = redis_client.get_json(f"query:{query_id}")
            if query_data:
                query_data["status"] = "completed"
                query_data["result"] = cached_result
                query_data["completed_at"] = datetime.now().isoformat()
                query_data["processing_time"] = processing_time
                redis_client.set_json(f"query:{query_id}", query_data, expire_seconds=3600)
            
            logger.info(f"Query {query_id} completed from cache in {processing_time:.2f}s")
            return cached_result
        
        # Create query engine using the thread-safe factory
        update_query_progress(query_id, 0.3, "Initializing query engine...")
        query_engine = query_engine_factory.create_query_engine(
            user_id=user_id,
            group_ids=group_ids,
        )
        
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
        
        # Record performance metrics for non-cached result
        record_query_performance_metrics(
            query_id, user_id, query_text, processing_time,
            len(sources), False
        )
        
        # Update job status to completed (this will trigger WebSocket notification)
        job_manager.update_job_status(job_id, JobStatus.COMPLETED, result=result)
        
        # Update query record with result
        query_data = redis_client.get_json(f"query:{query_id}")
        if query_data:
            query_data["status"] = "completed"
            query_data["result"] = result
            query_data["completed_at"] = datetime.now().isoformat()
            query_data["processing_time"] = processing_time
            redis_client.set_json(f"query:{query_id}", query_data, expire_seconds=3600)
        
        logger.info(f"Query {query_id} completed successfully in {processing_time:.2f}s with {len(sources)} sources")
        return result
        
    except QuerySecurityError as e:
        logger.error(f"Security violation in query {query_id}: {e}")
        error_message = f"Security validation failed: {e}"
        
        # Get job_id for error handling
        query_data = redis_client.get_json(f"query:{query_id}")
        job_id = query_data.get("job_id") if query_data else None
        
        if job_id:
            # Update job status to failed (this will trigger WebSocket notification)
            job_manager.update_job_status(job_id, JobStatus.FAILED, error=error_message)
        
        # Update query record with error
        if query_data:
            query_data["status"] = "failed"
            query_data["error"] = error_message
            query_data["error_type"] = "security_error"
            query_data["completed_at"] = datetime.now().isoformat()
            redis_client.set_json(f"query:{query_id}", query_data, expire_seconds=3600)
        
        # Don't retry security errors
        raise ValueError(error_message)
        
    except Exception as e:
        logger.error(f"Query {query_id} failed: {e}")
        error_message = str(e)
        
        # Check if this is a retryable error
        if isinstance(e, (ConnectionError, TimeoutError)) and self.request.retries < 2:
            logger.info(f"Retrying query {query_id} due to {type(e).__name__}")
            raise self.retry(countdown=30, exc=e)
        
        # Record performance metrics for failed query
        processing_time = time.time() - start_time
        record_query_performance_metrics(
            query_id, user_id, query_text, processing_time, 0, False, error_message
        )
        
        # Get job_id for error handling
        query_data = redis_client.get_json(f"query:{query_id}")
        job_id = query_data.get("job_id") if query_data else None
        
        if job_id:
            # Update job status to failed (this will trigger WebSocket notification)
            job_manager.update_job_status(job_id, JobStatus.FAILED, error=error_message)
        
        # Update query record with error
        if query_data:
            query_data["status"] = "failed"
            query_data["error"] = error_message
            query_data["error_type"] = "processing_error"
            query_data["completed_at"] = datetime.now().isoformat()
            redis_client.set_json(f"query:{query_id}", query_data, expire_seconds=3600)
        
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