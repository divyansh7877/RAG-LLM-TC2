"""
Redis client utilities for session management and task queue.
"""
import json
import redis
import logging
from typing import Optional, Dict, Any, List, Set
from datetime import datetime, timedelta
from contextlib import contextmanager
from .config import config
from .models import UserSession, Job, Document, Query

# Set up logging
logger = logging.getLogger(__name__)


class RedisConnectionError(Exception):
    """Redis connection error."""
    pass


class RedisClient:
    """Redis client wrapper with connection pooling and enhanced session management."""
    
    def __init__(self):
        """Initialize Redis client with connection pools."""
        try:
            # Main Redis connection pool
            self.pool = redis.ConnectionPool.from_url(
                config.REDIS_URL,
                max_connections=20,
                retry_on_timeout=True,
                socket_keepalive=True,
                socket_keepalive_options={}
            )
            self.client = redis.Redis(connection_pool=self.pool, decode_responses=True)
            
            # Separate connection pool for sessions
            session_url = config.REDIS_URL.replace('/0', f'/{config.REDIS_SESSION_DB}')
            self.session_pool = redis.ConnectionPool.from_url(
                session_url,
                max_connections=10,
                retry_on_timeout=True,
                socket_keepalive=True,
                socket_keepalive_options={}
            )
            self.session_client = redis.Redis(connection_pool=self.session_pool, decode_responses=True)
            
            # Test connections
            self._test_connections()
            logger.info("Redis client initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize Redis client: {e}")
            raise RedisConnectionError(f"Redis initialization failed: {e}")
    
    def _test_connections(self):
        """Test Redis connections."""
        try:
            self.client.ping()
            self.session_client.ping()
        except Exception as e:
            raise RedisConnectionError(f"Redis connection test failed: {e}")
    
    @contextmanager
    def get_connection(self, use_session_db: bool = False):
        """Context manager for Redis connections with error handling."""
        client = self.session_client if use_session_db else self.client
        try:
            yield client
        except redis.ConnectionError as e:
            logger.error(f"Redis connection error: {e}")
            raise RedisConnectionError(f"Redis connection failed: {e}")
        except redis.TimeoutError as e:
            logger.error(f"Redis timeout error: {e}")
            raise RedisConnectionError(f"Redis timeout: {e}")
        except Exception as e:
            logger.error(f"Redis operation error: {e}")
            raise
    
    def get(self, key: str) -> Optional[str]:
        """Get value from Redis with proper error handling."""
        try:
            with self.get_connection() as client:
                return client.get(key)
        except RedisConnectionError:
            raise
        except Exception as e:
            logger.error(f"Redis GET error for key '{key}': {e}")
            return None
    
    def set(self, key: str, value: str, expire_seconds: Optional[int] = None) -> bool:
        """Set value in Redis with optional expiration and proper error handling."""
        try:
            with self.get_connection() as client:
                return bool(client.set(key, value, ex=expire_seconds))
        except RedisConnectionError:
            raise
        except Exception as e:
            logger.error(f"Redis SET error for key '{key}': {e}")
            return False
    
    def delete(self, key: str) -> bool:
        """Delete key from Redis with proper error handling."""
        try:
            with self.get_connection() as client:
                return bool(client.delete(key))
        except RedisConnectionError:
            raise
        except Exception as e:
            logger.error(f"Redis DELETE error for key '{key}': {e}")
            return False
    
    def exists(self, key: str) -> bool:
        """Check if key exists in Redis with proper error handling."""
        try:
            with self.get_connection() as client:
                return bool(client.exists(key))
        except RedisConnectionError:
            raise
        except Exception as e:
            logger.error(f"Redis EXISTS error for key '{key}': {e}")
            return False
    
    def get_json(self, key: str) -> Optional[Dict[str, Any]]:
        """Get JSON value from Redis with proper error handling."""
        try:
            value = self.get(key)
            if value:
                return json.loads(value)
            return None
        except json.JSONDecodeError as e:
            logger.error(f"JSON decode error for key '{key}': {e}")
            return None
        except Exception as e:
            logger.error(f"Get JSON error for key '{key}': {e}")
            return None
    
    def set_json(self, key: str, value: Dict[str, Any], expire_seconds: Optional[int] = None) -> bool:
        """Set JSON value in Redis with optional expiration and proper error handling."""
        try:
            json_str = json.dumps(value, default=str)
            return self.set(key, json_str, expire_seconds)
        except (TypeError, ValueError) as e:
            logger.error(f"JSON encode error for key '{key}': {e}")
            return False
        except Exception as e:
            logger.error(f"Set JSON error for key '{key}': {e}")
            return False
    
    # Enhanced session management methods
    def get_session(self, session_id: str) -> Optional[UserSession]:
        """Get session data from Redis using Pydantic model."""
        try:
            with self.get_connection(use_session_db=True) as client:
                value = client.get(f"session:{session_id}")
                if value:
                    return UserSession.from_redis(value)
                return None
        except RedisConnectionError:
            raise
        except Exception as e:
            logger.error(f"Session GET error for session '{session_id}': {e}")
            return None
    
    def set_session(self, session: UserSession) -> bool:
        """Set session data in Redis with expiration using Pydantic model."""
        try:
            with self.get_connection(use_session_db=True) as client:
                expire_seconds = config.SESSION_EXPIRE_HOURS * 3600
                return bool(client.set(
                    f"session:{session.session_id}", 
                    session.to_redis(), 
                    ex=expire_seconds
                ))
        except RedisConnectionError:
            raise
        except Exception as e:
            logger.error(f"Session SET error for session '{session.session_id}': {e}")
            return False
    
    def update_session_activity(self, session_id: str) -> bool:
        """Update session last activity timestamp."""
        try:
            session = self.get_session(session_id)
            if session:
                session.update_activity()
                return self.set_session(session)
            return False
        except Exception as e:
            logger.error(f"Session activity update error for session '{session_id}': {e}")
            return False
    
    def delete_session(self, session_id: str) -> bool:
        """Delete session from Redis with proper error handling."""
        try:
            with self.get_connection(use_session_db=True) as client:
                return bool(client.delete(f"session:{session_id}"))
        except RedisConnectionError:
            raise
        except Exception as e:
            logger.error(f"Session DELETE error for session '{session_id}': {e}")
            return False
    
    def get_all_active_sessions(self) -> List[UserSession]:
        """Get all active sessions from Redis."""
        try:
            with self.get_connection(use_session_db=True) as client:
                pattern = "session:*"
                keys = client.keys(pattern)
                sessions = []
                
                for key in keys:
                    try:
                        value = client.get(key)
                        if value:
                            session = UserSession.from_redis(value)
                            if session.is_active:
                                sessions.append(session)
                    except Exception as e:
                        logger.warning(f"Failed to parse session from key '{key}': {e}")
                        continue
                
                return sessions
        except RedisConnectionError:
            raise
        except Exception as e:
            logger.error(f"Get all sessions error: {e}")
            return []
    
    def get_user_sessions(self, user_id: str) -> List[UserSession]:
        """Get all active sessions for a specific user."""
        try:
            all_sessions = self.get_all_active_sessions()
            return [session for session in all_sessions if session.user_id == user_id]
        except Exception as e:
            logger.error(f"Get user sessions error for user '{user_id}': {e}")
            return []
    
    def cleanup_expired_sessions(self) -> int:
        """Clean up expired sessions and return count of cleaned sessions."""
        try:
            with self.get_connection(use_session_db=True) as client:
                pattern = "session:*"
                keys = client.keys(pattern)
                cleaned = 0
                current_time = datetime.now()
                
                for key in keys:
                    try:
                        value = client.get(key)
                        if value:
                            session = UserSession.from_redis(value)
                            # Check if session is expired based on last activity
                            expire_time = session.last_activity + timedelta(hours=config.SESSION_EXPIRE_HOURS)
                            if current_time > expire_time or not session.is_active:
                                client.delete(key)
                                cleaned += 1
                                logger.info(f"Cleaned expired session: {session.session_id}")
                    except Exception as e:
                        logger.warning(f"Error processing session key '{key}' during cleanup: {e}")
                        # Delete corrupted session data
                        client.delete(key)
                        cleaned += 1
                
                logger.info(f"Cleaned up {cleaned} expired sessions")
                return cleaned
        except RedisConnectionError:
            raise
        except Exception as e:
            logger.error(f"Session cleanup error: {e}")
            return 0
    
    def invalidate_user_sessions(self, user_id: str) -> int:
        """Invalidate all sessions for a specific user."""
        try:
            user_sessions = self.get_user_sessions(user_id)
            invalidated = 0
            
            for session in user_sessions:
                if self.delete_session(session.session_id):
                    invalidated += 1
                    logger.info(f"Invalidated session {session.session_id} for user {user_id}")
            
            return invalidated
        except Exception as e:
            logger.error(f"Error invalidating sessions for user '{user_id}': {e}")
            return 0
    
    # Job management methods
    def get_job(self, job_id: str) -> Optional[Job]:
        """Get job data from Redis using Pydantic model."""
        try:
            with self.get_connection() as client:
                value = client.get(f"job:{job_id}")
                if value:
                    return Job.from_redis(value)
                return None
        except RedisConnectionError:
            raise
        except Exception as e:
            logger.error(f"Job GET error for job '{job_id}': {e}")
            return None
    
    def set_job(self, job: Job) -> bool:
        """Set job data in Redis using Pydantic model."""
        try:
            with self.get_connection() as client:
                # Jobs don't expire automatically, they're cleaned up by maintenance tasks
                return bool(client.set(f"job:{job.job_id}", job.to_redis()))
        except RedisConnectionError:
            raise
        except Exception as e:
            logger.error(f"Job SET error for job '{job.job_id}': {e}")
            return False
    
    def delete_job(self, job_id: str) -> bool:
        """Delete job from Redis."""
        try:
            with self.get_connection() as client:
                return bool(client.delete(f"job:{job_id}"))
        except RedisConnectionError:
            raise
        except Exception as e:
            logger.error(f"Job DELETE error for job '{job_id}': {e}")
            return False
    
    def get_user_jobs(self, user_id: str) -> List[Job]:
        """Get all jobs for a specific user."""
        try:
            with self.get_connection() as client:
                pattern = "job:*"
                keys = client.keys(pattern)
                jobs = []
                
                for key in keys:
                    try:
                        value = client.get(key)
                        if value:
                            job = Job.from_redis(value)
                            if job.user_id == user_id:
                                jobs.append(job)
                    except Exception as e:
                        logger.warning(f"Failed to parse job from key '{key}': {e}")
                        continue
                
                # Sort by creation time, newest first
                jobs.sort(key=lambda x: x.created_at, reverse=True)
                return jobs
        except RedisConnectionError:
            raise
        except Exception as e:
            logger.error(f"Get user jobs error for user '{user_id}': {e}")
            return []
    
    def get_jobs_by_status(self, status: str) -> List[Job]:
        """Get all jobs with a specific status."""
        try:
            with self.get_connection() as client:
                pattern = "job:*"
                keys = client.keys(pattern)
                jobs = []
                
                for key in keys:
                    try:
                        value = client.get(key)
                        if value:
                            job = Job.from_redis(value)
                            if job.status == status:
                                jobs.append(job)
                    except Exception as e:
                        logger.warning(f"Failed to parse job from key '{key}': {e}")
                        continue
                
                return jobs
        except RedisConnectionError:
            raise
        except Exception as e:
            logger.error(f"Get jobs by status error for status '{status}': {e}")
            return []
    
    # Document management methods
    def get_document(self, document_id: str) -> Optional[Document]:
        """Get document data from Redis using Pydantic model."""
        try:
            with self.get_connection() as client:
                value = client.get(f"document:{document_id}")
                if value:
                    return Document.from_redis(value)
                return None
        except RedisConnectionError:
            raise
        except Exception as e:
            logger.error(f"Document GET error for document '{document_id}': {e}")
            return None
    
    def set_document(self, document: Document) -> bool:
        """Set document data in Redis using Pydantic model."""
        try:
            with self.get_connection() as client:
                return bool(client.set(f"document:{document.document_id}", document.to_redis()))
        except RedisConnectionError:
            raise
        except Exception as e:
            logger.error(f"Document SET error for document '{document.document_id}': {e}")
            return False
    
    def delete_document(self, document_id: str) -> bool:
        """Delete document from Redis."""
        try:
            with self.get_connection() as client:
                return bool(client.delete(f"document:{document_id}"))
        except RedisConnectionError:
            raise
        except Exception as e:
            logger.error(f"Document DELETE error for document '{document_id}': {e}")
            return False
    
    def get_user_documents(self, user_id: str) -> List[Document]:
        """Get all documents for a specific user."""
        try:
            with self.get_connection() as client:
                pattern = "document:*"
                keys = client.keys(pattern)
                documents = []
                
                for key in keys:
                    try:
                        value = client.get(key)
                        if value:
                            document = Document.from_redis(value)
                            if document.user_id == user_id:
                                documents.append(document)
                    except Exception as e:
                        logger.warning(f"Failed to parse document from key '{key}': {e}")
                        continue
                
                # Sort by upload date, newest first
                documents.sort(key=lambda x: x.upload_date, reverse=True)
                return documents
        except RedisConnectionError:
            raise
        except Exception as e:
            logger.error(f"Get user documents error for user '{user_id}': {e}")
            return []
    
    # Query management methods
    def get_query(self, query_id: str) -> Optional[Query]:
        """Get query data from Redis using Pydantic model."""
        try:
            with self.get_connection() as client:
                value = client.get(f"query:{query_id}")
                if value:
                    return Query.from_redis(value)
                return None
        except RedisConnectionError:
            raise
        except Exception as e:
            logger.error(f"Query GET error for query '{query_id}': {e}")
            return None
    
    def set_query(self, query: Query) -> bool:
        """Set query data in Redis using Pydantic model."""
        try:
            with self.get_connection() as client:
                # Queries expire after 24 hours to prevent unlimited growth
                expire_seconds = 24 * 3600
                return bool(client.set(f"query:{query.query_id}", query.to_redis(), ex=expire_seconds))
        except RedisConnectionError:
            raise
        except Exception as e:
            logger.error(f"Query SET error for query '{query.query_id}': {e}")
            return False
    
    def get_user_queries(self, user_id: str, limit: int = 50) -> List[Query]:
        """Get recent queries for a specific user."""
        try:
            with self.get_connection() as client:
                pattern = "query:*"
                keys = client.keys(pattern)
                queries = []
                
                for key in keys:
                    try:
                        value = client.get(key)
                        if value:
                            query = Query.from_redis(value)
                            if query.user_id == user_id:
                                queries.append(query)
                    except Exception as e:
                        logger.warning(f"Failed to parse query from key '{key}': {e}")
                        continue
                
                # Sort by creation time, newest first, and limit results
                queries.sort(key=lambda x: x.created_at, reverse=True)
                return queries[:limit]
        except RedisConnectionError:
            raise
        except Exception as e:
            logger.error(f"Get user queries error for user '{user_id}': {e}")
            return []
    
    # Health and maintenance methods
    def health_check(self) -> Dict[str, Any]:
        """Comprehensive Redis health check."""
        health_status = {
            "redis_main": False,
            "redis_sessions": False,
            "connection_pools": {},
            "memory_info": {},
            "errors": []
        }
        
        try:
            # Test main Redis connection
            with self.get_connection() as client:
                client.ping()
                health_status["redis_main"] = True
                
                # Get memory info
                info = client.info('memory')
                health_status["memory_info"] = {
                    "used_memory": info.get('used_memory', 0),
                    "used_memory_human": info.get('used_memory_human', 'N/A'),
                    "maxmemory": info.get('maxmemory', 0)
                }
        except Exception as e:
            health_status["errors"].append(f"Main Redis connection failed: {e}")
        
        try:
            # Test session Redis connection
            with self.get_connection(use_session_db=True) as client:
                client.ping()
                health_status["redis_sessions"] = True
        except Exception as e:
            health_status["errors"].append(f"Session Redis connection failed: {e}")
        
        # Check connection pool status
        try:
            health_status["connection_pools"] = {
                "main_pool": {
                    "created_connections": self.pool.created_connections,
                    "available_connections": len(self.pool._available_connections),
                    "in_use_connections": len(self.pool._in_use_connections)
                },
                "session_pool": {
                    "created_connections": self.session_pool.created_connections,
                    "available_connections": len(self.session_pool._available_connections),
                    "in_use_connections": len(self.session_pool._in_use_connections)
                }
            }
        except Exception as e:
            health_status["errors"].append(f"Connection pool status check failed: {e}")
        
        return health_status
    
    def get_stats(self) -> Dict[str, Any]:
        """Get Redis usage statistics."""
        try:
            stats = {
                "sessions": 0,
                "jobs": 0,
                "documents": 0,
                "queries": 0,
                "total_keys": 0
            }
            
            with self.get_connection() as client:
                # Count different types of keys
                for key_type, pattern in [
                    ("jobs", "job:*"),
                    ("documents", "document:*"),
                    ("queries", "query:*")
                ]:
                    keys = client.keys(pattern)
                    stats[key_type] = len(keys)
                    stats["total_keys"] += len(keys)
            
            with self.get_connection(use_session_db=True) as client:
                session_keys = client.keys("session:*")
                stats["sessions"] = len(session_keys)
                stats["total_keys"] += len(session_keys)
            
            return stats
        except Exception as e:
            logger.error(f"Get stats error: {e}")
            return {"error": str(e)}


# Global Redis client instance
redis_client = RedisClient()