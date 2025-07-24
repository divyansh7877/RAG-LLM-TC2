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

# Import query processing functionality
import lancedb
from llama_index.core import Settings, StorageContext, VectorStoreIndex
from llama_index.core.retrievers import VectorIndexRetriever
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core.postprocessor import SimilarityPostprocessor
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.vector_stores.lancedb import LanceDBVectorStore
from llama_index.core.vector_stores import MetadataFilter, MetadataFilters, FilterOperator
from llama_index.llms.llama_cpp import LlamaCPP

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


def create_user_security_filters(user_id: str, group_ids: List[str]) -> MetadataFilters:
    """
    Create security filters that ensure user can only access authorized documents.
    
    Args:
        user_id: User identifier
        group_ids: List of group IDs user has access to
        
    Returns:
        MetadataFilters object with proper user isolation
    """
    if not user_id:
        raise QuerySecurityError("User ID is required for security filtering")
    
    if not group_ids:
        raise QuerySecurityError("At least one group ID is required for security filtering")
    
    # Create filters for user's own documents and group documents
    filters = []
    
    # User can access their own documents
    user_filter = MetadataFilter(
        key="user_id",
        value=user_id,
        operator=FilterOperator.EQ
    )
    filters.append(user_filter)
    
    # User can access documents from their groups
    for group_id in group_ids:
        group_filter = MetadataFilter(
            key="group_id", 
            value=group_id,
            operator=FilterOperator.EQ
        )
        filters.append(group_filter)
    
    # Combine filters with OR condition (user can access own docs OR group docs)
    return MetadataFilters(filters=filters, condition="or")


def generate_cache_key(user_id: str, group_ids: List[str], query_text: str) -> str:
    """
    Generate a cache key for query results based on user context and query.
    
    Args:
        user_id: User identifier
        group_ids: List of group IDs
        query_text: Query text
        
    Returns:
        Cache key string
    """
    # Create a deterministic cache key that includes user context
    context_str = f"{user_id}:{':'.join(sorted(group_ids))}:{query_text.strip().lower()}"
    cache_hash = hashlib.sha256(context_str.encode()).hexdigest()[:16]
    return f"query_cache:{cache_hash}"


def get_cached_query_result(cache_key: str) -> Optional[Dict[str, Any]]:
    """
    Retrieve cached query result if available and not expired.
    
    Args:
        cache_key: Cache key to lookup
        
    Returns:
        Cached result dict or None if not found/expired
    """
    try:
        cached_data = redis_client.get_json(cache_key)
        if cached_data and cached_data.get("expires_at", 0) > time.time():
            logger.info(f"Cache hit for key: {cache_key}")
            return cached_data.get("result")
        elif cached_data:
            # Expired cache entry
            redis_client.redis_client.delete(cache_key)
            logger.info(f"Expired cache entry removed: {cache_key}")
    except Exception as e:
        logger.warning(f"Failed to retrieve cached result: {e}")
    
    return None


def cache_query_result(cache_key: str, result: Dict[str, Any]) -> None:
    """
    Cache query result with expiration.
    
    Args:
        cache_key: Cache key to store under
        result: Result data to cache
    """
    try:
        cache_data = {
            "result": result,
            "cached_at": time.time(),
            "expires_at": time.time() + CACHE_EXPIRE_SECONDS
        }
        redis_client.set_json(cache_key, cache_data, expire_seconds=CACHE_EXPIRE_SECONDS)
        logger.info(f"Cached query result with key: {cache_key}")
    except Exception as e:
        logger.warning(f"Failed to cache query result: {e}")


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


def initialize_query_engine(user_id: str, group_ids: List[str]) -> RetrieverQueryEngine:
    """
    Initialize a secure query engine with user-specific filtering.
    
    Args:
        user_id: User identifier
        group_ids: List of group IDs user has access to
        
    Returns:
        Configured RetrieverQueryEngine with security filters
    """
    try:
        # Initialize embedding model (reuse if already loaded)
        if not hasattr(Settings, 'embed_model') or Settings.embed_model is None:
            Settings.embed_model = HuggingFaceEmbedding(
                model_name=EMBED_MODEL_NAME,
                device="cpu",
                trust_remote_code=True,
            )
        
        # Initialize LLM (reuse if already loaded)
        if not hasattr(Settings, 'llm') or Settings.llm is None:
            Settings.llm = LlamaCPP(
                model_path=LLM_MODEL_PATH,
                temperature=0.1,
                max_new_tokens=512,
                context_window=2048,
                generate_kwargs={},
                model_kwargs={"n_gpu_layers": 0},  # CPU only for stability
                verbose=False,
            )
        
        # Connect to vector store with security filters
        vector_store = LanceDBVectorStore(
            uri=DB_PATH,
            table_name=TABLE_NAME
        )
        
        # Create storage context
        storage_context = StorageContext.from_defaults(vector_store=vector_store)
        
        # Load index from storage
        index = VectorStoreIndex.from_vector_store(
            vector_store=vector_store,
            storage_context=storage_context
        )
        
        # Create security filters
        security_filters = create_user_security_filters(user_id, group_ids)
        
        # Create retriever with security filters
        retriever = VectorIndexRetriever(
            index=index,
            similarity_top_k=MAX_RETRIEVED_NODES,
            filters=security_filters
        )
        
        # Create query engine with post-processing
        query_engine = RetrieverQueryEngine(
            retriever=retriever,
            node_postprocessors=[
                SimilarityPostprocessor(similarity_cutoff=SIMILARITY_THRESHOLD)
            ]
        )
        
        return query_engine
        
    except Exception as e:
        logger.error(f"Failed to initialize query engine: {e}")
        raise ValueError(f"Query engine initialization failed: {e}")


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
    """Update query progress in Redis and Celery state."""
    try:
        # Update Redis
        query_key = f"query:{query_id}"
        query_data = redis_client.get_json(query_key)
        if query_data:
            query_data["progress"] = progress
            query_data["last_updated"] = time.time()
            if status_message:
                query_data["status_message"] = status_message
            redis_client.set_json(query_key, query_data)
        
        # Update Celery task state
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
        
        # Initialize query status
        query_data = {
            "query_id": query_id,
            "user_id": user_id,
            "query_text": query_text,
            "status": "processing",
            "started_at": start_time,
            "progress": 0.0,
            "status_message": "Validating query..."
        }
        redis_client.set_json(f"query:{query_id}", query_data)
        
        logger.info(f"Starting query {query_id} for user {user_id} with groups {group_ids}")
        
        # Check cache first
        update_query_progress(query_id, 0.1, "Checking cache...")
        cache_key = generate_cache_key(user_id, group_ids, query_text)
        cached_result = get_cached_query_result(cache_key)
        
        if cached_result:
            # Return cached result
            processing_time = time.time() - start_time
            cached_result["processing_time"] = processing_time
            cached_result["cached"] = True
            
            query_data.update({
                "status": "completed",
                "completed_at": time.time(),
                "result": cached_result,
                "processing_time": processing_time,
                "progress": 1.0,
                "cached": True
            })
            redis_client.set_json(f"query:{query_id}", query_data)
            
            logger.info(f"Query {query_id} completed from cache in {processing_time:.2f}s")
            return cached_result
        
        # Initialize query engine with security filters
        update_query_progress(query_id, 0.3, "Initializing query engine...")
        query_engine = initialize_query_engine(user_id, group_ids)
        
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
        
        # Cache the result for future queries
        cache_query_result(cache_key, result)
        
        # Update query status
        query_data.update({
            "status": "completed",
            "completed_at": time.time(),
            "result": result,
            "processing_time": processing_time,
            "progress": 1.0,
            "result_count": len(sources)
        })
        redis_client.set_json(f"query:{query_id}", query_data)
        
        logger.info(f"Query {query_id} completed successfully in {processing_time:.2f}s with {len(sources)} sources")
        return result
        
    except QuerySecurityError as e:
        logger.error(f"Security violation in query {query_id}: {e}")
        error_message = f"Security validation failed: {e}"
        
        query_data = {
            "query_id": query_id,
            "user_id": user_id,
            "query_text": query_text,
            "status": "failed",
            "error": error_message,
            "error_type": "security",
            "completed_at": time.time(),
            "processing_time": time.time() - start_time
        }
        redis_client.set_json(f"query:{query_id}", query_data)
        
        # Don't retry security errors
        raise ValueError(error_message)
        
    except Exception as e:
        logger.error(f"Query {query_id} failed: {e}")
        processing_time = time.time() - start_time
        error_message = str(e)
        
        # Mark query as failed
        query_data = {
            "query_id": query_id,
            "user_id": user_id,
            "query_text": query_text,
            "status": "failed",
            "error": error_message,
            "error_type": "processing",
            "completed_at": time.time(),
            "processing_time": processing_time
        }
        redis_client.set_json(f"query:{query_id}", query_data)
        
        # Check if this is a retryable error
        if isinstance(e, (ConnectionError, TimeoutError)) and self.request.retries < 2:
            logger.info(f"Retrying query {query_id} due to {type(e).__name__}")
            raise self.retry(countdown=30, exc=e)
        
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
    Cleanup expired query cache entries.
    This task should be run periodically.
    """
    try:
        pattern = "query_cache:*"
        keys = redis_client.redis_client.keys(pattern)
        
        expired_count = 0
        for key in keys:
            try:
                cached_data = redis_client.get_json(key.decode('utf-8'))
                if cached_data and cached_data.get("expires_at", 0) <= time.time():
                    redis_client.redis_client.delete(key)
                    expired_count += 1
            except Exception as e:
                logger.warning(f"Failed to check cache entry {key}: {e}")
                # Delete problematic entries
                redis_client.redis_client.delete(key)
                expired_count += 1
        
        logger.info(f"Cleaned up {expired_count} expired query cache entries")
        return {"cleaned_entries": expired_count}
        
    except Exception as e:
        logger.error(f"Failed to cleanup query cache: {e}")
        return {"error": str(e)}