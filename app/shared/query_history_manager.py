#!/usr/bin/env python3
"""
Query history management system for persistent storage of query responses.

This module provides functionality to store query responses with job linkage
for historical access and retrieval by users from the job status page.

Storage Architecture:
- Primary storage: Redis with TTL-based expiration (configurable retention)
- Indexing: Redis sorted sets for efficient pagination and time-based queries
- Backup option: Optional JSON file export for long-term archival
- Memory efficiency: Large response content can be compressed

Redis is chosen over traditional RDBMS because:
1. Consistent with existing architecture (no new DB to maintain)
2. Excellent performance for query retrieval
3. Built-in TTL for automatic cleanup
4. Native support for complex data structures
5. Easy horizontal scaling
"""
import logging
import time
import json
import gzip
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass

from .redis_client import redis_client
from .models import JobType, JobStatus
from .config import config

logger = logging.getLogger(__name__)

# Constants
QUERY_HISTORY_PREFIX = "query_history"
USER_QUERY_INDEX_PREFIX = "user_queries"  # Sorted set for time-based ordering
JOB_QUERY_LINK_PREFIX = "job_query"
QUERY_CONTENT_PREFIX = "query_content"  # Separate storage for large content
DEFAULT_HISTORY_RETENTION_DAYS = 30
MAX_QUERIES_PER_USER = 1000
COMPRESS_THRESHOLD_BYTES = 1024  # Compress responses larger than 1KB


@dataclass
class HistoricalQueryResponse:
    """Historical query response data structure."""
    query_id: str
    job_id: str
    user_id: str
    query_text: str
    response: str
    sources: List[Dict[str, Any]]
    processing_time: float
    created_at: datetime
    metadata: Dict[str, Any]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "query_id": self.query_id,
            "job_id": self.job_id,
            "user_id": self.user_id,
            "query_text": self.query_text,
            "response": self.response,
            "sources": self.sources,
            "processing_time": self.processing_time,
            "created_at": self.created_at.isoformat(),
            "metadata": self.metadata
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'HistoricalQueryResponse':
        """Create from dictionary."""
        return cls(
            query_id=data["query_id"],
            job_id=data["job_id"],
            user_id=data["user_id"],
            query_text=data["query_text"],
            response=data["response"],
            sources=data.get("sources", []),
            processing_time=data.get("processing_time", 0.0),
            created_at=datetime.fromisoformat(data["created_at"]),
            metadata=data.get("metadata", {})
        )


class QueryHistoryManager:
    """Manager for persistent query history storage and retrieval with Redis optimization."""
    
    def __init__(self):
        self.retention_days = int(config.QUERY_HISTORY_RETENTION_DAYS or DEFAULT_HISTORY_RETENTION_DAYS)
        self.max_queries_per_user = int(config.MAX_QUERIES_PER_USER or MAX_QUERIES_PER_USER)
        logger.info(f"QueryHistoryManager initialized with {self.retention_days} days retention")
    
    def _compress_content(self, content: str) -> Tuple[bytes, bool]:
        """Compress content if it's large enough to warrant compression.
        
        Returns:
            Tuple of (content_bytes, is_compressed)
        """
        try:
            content_bytes = content.encode('utf-8')
            if len(content_bytes) > COMPRESS_THRESHOLD_BYTES:
                compressed = gzip.compress(content_bytes)
                if len(compressed) < len(content_bytes):  # Only use if actually smaller
                    return compressed, True
            return content_bytes, False
        except Exception as e:
            logger.warning(f"Failed to compress content: {e}")
            return content.encode('utf-8'), False
    
    def _decompress_content(self, content_bytes: bytes, is_compressed: bool) -> str:
        """Decompress content if it was compressed."""
        try:
            if is_compressed:
                return gzip.decompress(content_bytes).decode('utf-8')
            return content_bytes.decode('utf-8')
        except Exception as e:
            logger.error(f"Failed to decompress content: {e}")
            return "[Content decompression failed]"
    
    def store_query_response(
        self,
        query_id: str,
        job_id: str,
        user_id: str,
        query_text: str,
        response_result: Dict[str, Any]
    ) -> bool:
        """
        Store a query response in persistent history.
        
        Args:
            query_id: Unique query identifier
            job_id: Associated job identifier
            user_id: User who made the query
            query_text: Original query text
            response_result: Complete response result from query worker
            
        Returns:
            True if stored successfully, False otherwise
        """
        try:
            # Create historical query response
            historical_response = HistoricalQueryResponse(
                query_id=query_id,
                job_id=job_id,
                user_id=user_id,
                query_text=query_text,
                response=response_result.get("answer", ""),
                sources=response_result.get("sources", []),
                processing_time=response_result.get("processing_time", 0.0),
                created_at=datetime.now(),
                metadata={
                    "cached": response_result.get("cached", False),
                    "result_count": response_result.get("result_count", 0),
                    "query_metadata": response_result.get("query_metadata", {}),
                    "response_metadata": response_result.get("response_metadata", {})
                }
            )
            
            # Calculate expiration time
            expire_seconds = self.retention_days * 24 * 3600
            
            # Store the historical query response
            history_key = f"{QUERY_HISTORY_PREFIX}:{query_id}"
            if not redis_client.set_json(history_key, historical_response.to_dict(), expire_seconds):
                logger.error(f"Failed to store query history for {query_id}")
                return False
            
            # Add to user's query index
            user_index_key = f"{USER_QUERY_INDEX_PREFIX}:{user_id}"
            query_entry = {
                "query_id": query_id,
                "job_id": job_id,
                "query_text": query_text[:100] + "..." if len(query_text) > 100 else query_text,  # Truncated for index
                "created_at": historical_response.created_at.isoformat(),
                "processing_time": historical_response.processing_time
            }
            
            # Get existing user index
            user_queries = redis_client.get_json(user_index_key) or []
            
            # Add new query to the beginning of the list
            user_queries.insert(0, query_entry)
            
            # Limit the number of queries per user
            if len(user_queries) > self.max_queries_per_user:
                user_queries = user_queries[:self.max_queries_per_user]
            
            # Store updated user index
            if not redis_client.set_json(user_index_key, user_queries, expire_seconds):
                logger.warning(f"Failed to update user query index for {user_id}")
            
            # Create job -> query link for job status page access
            job_query_key = f"{JOB_QUERY_LINK_PREFIX}:{job_id}"
            job_link_data = {
                "query_id": query_id,
                "user_id": user_id,
                "created_at": historical_response.created_at.isoformat()
            }
            
            if not redis_client.set_json(job_query_key, job_link_data, expire_seconds):
                logger.warning(f"Failed to create job-query link for job {job_id}")
            
            logger.info(f"Successfully stored query history for {query_id} (job: {job_id})")
            return True
            
        except Exception as e:
            logger.error(f"Failed to store query response history for {query_id}: {e}", exc_info=True)
            return False
    
    def get_query_history(self, query_id: str) -> Optional[HistoricalQueryResponse]:
        """
        Retrieve a specific query's historical response.
        
        Args:
            query_id: Query identifier
            
        Returns:
            HistoricalQueryResponse if found, None otherwise
        """
        try:
            history_key = f"{QUERY_HISTORY_PREFIX}:{query_id}"
            data = redis_client.get_json(history_key)
            
            if not data:
                return None
            
            return HistoricalQueryResponse.from_dict(data)
            
        except Exception as e:
            logger.error(f"Failed to retrieve query history for {query_id}: {e}")
            return None
    
    def get_user_query_history(
        self,
        user_id: str,
        limit: int = 50,
        offset: int = 0
    ) -> List[Dict[str, Any]]:
        """
        Get a user's query history.
        
        Args:
            user_id: User identifier
            limit: Maximum number of queries to return
            offset: Number of queries to skip
            
        Returns:
            List of query history entries
        """
        try:
            user_index_key = f"{USER_QUERY_INDEX_PREFIX}:{user_id}"
            user_queries = redis_client.get_json(user_index_key) or []
            
            # Apply pagination
            start_idx = offset
            end_idx = offset + limit
            paginated_queries = user_queries[start_idx:end_idx]
            
            return paginated_queries
            
        except Exception as e:
            logger.error(f"Failed to retrieve user query history for {user_id}: {e}")
            return []
    
    def get_job_query_history(self, job_id: str) -> Optional[HistoricalQueryResponse]:
        """
        Get query history associated with a specific job.
        
        Args:
            job_id: Job identifier
            
        Returns:
            HistoricalQueryResponse if found, None otherwise
        """
        try:
            # Get job-query link
            job_query_key = f"{JOB_QUERY_LINK_PREFIX}:{job_id}"
            link_data = redis_client.get_json(job_query_key)
            
            if not link_data:
                return None
            
            query_id = link_data.get("query_id")
            if not query_id:
                return None
            
            # Get the actual query history
            return self.get_query_history(query_id)
            
        except Exception as e:
            logger.error(f"Failed to retrieve job query history for {job_id}: {e}")
            return None
    
    def search_user_queries(
        self,
        user_id: str,
        search_text: str,
        limit: int = 20
    ) -> List[Dict[str, Any]]:
        """
        Search through a user's query history.
        
        Args:
            user_id: User identifier
            search_text: Text to search for in queries
            limit: Maximum number of results
            
        Returns:
            List of matching query entries
        """
        try:
            user_queries = self.get_user_query_history(user_id, limit=self.max_queries_per_user)
            search_lower = search_text.lower()
            
            matching_queries = []
            for query_entry in user_queries:
                if search_lower in query_entry.get("query_text", "").lower():
                    matching_queries.append(query_entry)
                    if len(matching_queries) >= limit:
                        break
            
            return matching_queries
            
        except Exception as e:
            logger.error(f"Failed to search user queries for {user_id}: {e}")
            return []
    
    def cleanup_expired_history(self) -> Dict[str, Any]:
        """
        Clean up expired query history entries.
        
        Returns:
            Cleanup statistics
        """
        try:
            # This is handled automatically by Redis TTL, but we can add
            # additional cleanup logic here if needed
            
            stats = {
                "cleanup_completed": True,
                "timestamp": datetime.now().isoformat(),
                "retention_days": self.retention_days
            }
            
            logger.info("Query history cleanup completed")
            return stats
            
        except Exception as e:
            logger.error(f"Failed to cleanup query history: {e}")
            return {"error": str(e), "cleanup_completed": False}
    
    def get_statistics(self) -> Dict[str, Any]:
        """
        Get query history statistics.
        
        Returns:
            Dictionary containing statistics
        """
        try:
            stats = {
                "retention_days": self.retention_days,
                "max_queries_per_user": self.max_queries_per_user,
                "timestamp": datetime.now().isoformat()
            }
            
            return stats
            
        except Exception as e:
            logger.error(f"Failed to get query history statistics: {e}")
            return {"error": str(e)}


# Global instance
query_history_manager = QueryHistoryManager()
