"""
Celery worker for query processing tasks.
"""
from .celery_app import celery_app
from ..shared.redis_client import redis_client
from ..shared.models import JobStatus
import time


@celery_app.task(bind=True, name="process_user_query")
def process_user_query(self, query_id: str, user_id: str, group_ids: list, query_text: str):
    """
    Process user query with security isolation.
    
    Args:
        query_id: Unique query identifier
        user_id: User who submitted the query
        group_ids: List of groups user has access to
        query_text: The query text to process
    """
    try:
        # Update query status to processing
        query_data = {
            "query_id": query_id,
            "user_id": user_id,
            "query_text": query_text,
            "status": "processing",
            "started_at": time.time()
        }
        redis_client.set_json(f"query:{query_id}", query_data)
        
        # TODO: Implement actual query processing logic in subsequent tasks
        # For now, simulate processing
        time.sleep(2)  # Simulate query processing time
        
        # Mock response
        result = {
            "answer": f"This is a mock response to: {query_text}",
            "sources": ["Document 1", "Document 2"],
            "processing_time": 2.0
        }
        
        # Update query with result
        query_data.update({
            "status": "completed",
            "completed_at": time.time(),
            "result": result,
            "processing_time": 2.0
        })
        redis_client.set_json(f"query:{query_id}", query_data)
        
        return result
        
    except Exception as e:
        # Mark query as failed
        query_data = {
            "query_id": query_id,
            "user_id": user_id,
            "query_text": query_text,
            "status": "failed",
            "error": str(e),
            "completed_at": time.time()
        }
        redis_client.set_json(f"query:{query_id}", query_data)
        
        # Re-raise exception for Celery
        raise