"""
API endpoints for query history management.

These endpoints support the job status page integration for accessing
historical query responses.
"""
from fastapi import APIRouter, Depends, HTTPException, Query as QueryParam
from typing import List, Optional, Dict, Any
import logging

from ..shared.models import User
from ..shared.middleware import get_current_user
from ..shared.query_history_manager import query_history_manager, HistoricalQueryResponse
from ..shared.job_manager import job_manager

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/query-history", tags=["query-history"])


@router.get("/job/{job_id}")
async def get_job_query_history(
    job_id: str,
    current_user: User = Depends(get_current_user)
) -> Dict[str, Any]:
    """
    Get query history associated with a specific job.
    This is the primary endpoint for job status page integration.
    
    Args:
        job_id: Job identifier
        current_user: Authenticated user (for security)
    
    Returns:
        Query history data or error message
    """
    try:
        # Verify job belongs to the user (security check)
        job = job_manager.get_job(job_id)
        if not job:
            raise HTTPException(status_code=404, detail="Job not found")
        
        if job.user_id != current_user.user_id:
            raise HTTPException(status_code=403, detail="Access denied to this job")
        
        # Get the query history for this job
        query_history = query_history_manager.get_job_query_history(job_id)
        
        if not query_history:
            return {
                "job_id": job_id,
                "has_query": False,
                "message": "No query history found for this job"
            }
        
        return {
            "job_id": job_id,
            "has_query": True,
            "query_history": query_history.to_dict(),
            "summary": {
                "query_text_preview": query_history.query_text[:100] + "..." if len(query_history.query_text) > 100 else query_history.query_text,
                "response_length": len(query_history.response),
                "source_count": len(query_history.sources),
                "processing_time": query_history.processing_time,
                "created_at": query_history.created_at.isoformat()
            }
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get job query history for {job_id}: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")


@router.get("/user")
async def get_user_query_history(
    limit: int = QueryParam(20, ge=1, le=100),
    offset: int = QueryParam(0, ge=0),
    current_user: User = Depends(get_current_user)
) -> Dict[str, Any]:
    """
    Get user's complete query history with pagination.
    
    Args:
        limit: Maximum number of queries to return (1-100)
        offset: Number of queries to skip for pagination
        current_user: Authenticated user
    
    Returns:
        Paginated query history list
    """
    try:
        query_entries = query_history_manager.get_user_query_history(
            user_id=current_user.user_id,
            limit=limit,
            offset=offset
        )
        
        return {
            "user_id": current_user.user_id,
            "queries": query_entries,
            "pagination": {
                "limit": limit,
                "offset": offset,
                "count": len(query_entries),
                "has_more": len(query_entries) == limit
            }
        }
        
    except Exception as e:
        logger.error(f"Failed to get user query history for {current_user.user_id}: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")


@router.get("/search")
async def search_user_queries(
    q: str = QueryParam(..., min_length=1, max_length=200),
    limit: int = QueryParam(20, ge=1, le=50),
    current_user: User = Depends(get_current_user)
) -> Dict[str, Any]:
    """
    Search through user's query history.
    
    Args:
        q: Search text to look for in query history
        limit: Maximum number of results (1-50)
        current_user: Authenticated user
    
    Returns:
        Matching query entries
    """
    try:
        matching_queries = query_history_manager.search_user_queries(
            user_id=current_user.user_id,
            search_text=q,
            limit=limit
        )
        
        return {
            "user_id": current_user.user_id,
            "search_query": q,
            "matches": matching_queries,
            "count": len(matching_queries)
        }
        
    except Exception as e:
        logger.error(f"Failed to search queries for {current_user.user_id}: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")


@router.get("/{query_id}")
async def get_specific_query_history(
    query_id: str,
    current_user: User = Depends(get_current_user)
) -> Dict[str, Any]:
    """
    Get detailed history for a specific query.
    
    Args:
        query_id: Query identifier
        current_user: Authenticated user
    
    Returns:
        Complete query history details
    """
    try:
        query_history = query_history_manager.get_query_history(query_id)
        
        if not query_history:
            raise HTTPException(status_code=404, detail="Query history not found")
        
        # Security check - ensure user can access this query
        if query_history.user_id != current_user.user_id:
            raise HTTPException(status_code=403, detail="Access denied to this query")
        
        return {
            "query_id": query_id,
            "query_history": query_history.to_dict()
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get query history for {query_id}: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")


@router.get("/stats/summary")
async def get_query_history_stats(
    current_user: User = Depends(get_current_user)
) -> Dict[str, Any]:
    """
    Get query history statistics for the current user.
    
    Args:
        current_user: Authenticated user
    
    Returns:
        Statistics about user's query history
    """
    try:
        # Get recent queries to calculate stats
        recent_queries = query_history_manager.get_user_query_history(
            user_id=current_user.user_id,
            limit=100  # Sample for stats
        )
        
        if not recent_queries:
            return {
                "user_id": current_user.user_id,
                "total_queries": 0,
                "message": "No query history found"
            }
        
        total_queries = len(recent_queries)
        avg_processing_time = sum(q.get("processing_time", 0) for q in recent_queries) / total_queries if total_queries > 0 else 0
        
        return {
            "user_id": current_user.user_id,
            "total_queries": total_queries,
            "avg_processing_time": round(avg_processing_time, 3),
            "most_recent": recent_queries[0]["created_at"] if recent_queries else None,
            "retention_days": query_history_manager.retention_days
        }
        
    except Exception as e:
        logger.error(f"Failed to get query stats for {current_user.user_id}: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")


@router.delete("/{query_id}")
async def delete_query_history(
    query_id: str,
    current_user: User = Depends(get_current_user)
) -> Dict[str, Any]:
    """
    Delete a specific query from history (if user owns it).
    
    Args:
        query_id: Query identifier to delete
        current_user: Authenticated user
    
    Returns:
        Success/failure message
    """
    try:
        # First verify the query exists and user owns it
        query_history = query_history_manager.get_query_history(query_id)
        
        if not query_history:
            raise HTTPException(status_code=404, detail="Query history not found")
        
        if query_history.user_id != current_user.user_id:
            raise HTTPException(status_code=403, detail="Access denied to this query")
        
        # Delete the query history
        # Note: This would require implementing a delete method in QueryHistoryManager
        # For now, return a message about automatic expiration
        
        return {
            "query_id": query_id,
            "message": "Query history will automatically expire based on retention policy",
            "retention_days": query_history_manager.retention_days
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to delete query history for {query_id}: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")
