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
from .models import Job, Document, Query, UserSession

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

    # ------------------------------
    # Session management methods
    # ------------------------------
    def _session_key(self, session_id: str) -> str:
        """Build the Redis key for a session object."""
        return f"session:{session_id}"

    def _user_sessions_key(self, user_id: str) -> str:
        """Build the Redis key for the user's session ID set."""
        return f"user_sessions:{user_id}"

    def set_session(self, session: UserSession) -> bool:
        """Create or update a session in the session DB with TTL and index by user."""
        try:
            expire_seconds = int(config.SESSION_EXPIRE_HOURS) * 3600
            with self.get_connection(use_session_db=True) as client:
                # Store the session JSON with TTL
                ok = bool(client.set(self._session_key(session.session_id), session.to_redis(), ex=expire_seconds))
                if ok:
                    # Maintain a set of session IDs per user for efficient lookup
                    client.sadd(self._user_sessions_key(session.user_id), session.session_id)
                    # Ensure the index set also expires eventually (same window)
                    client.expire(self._user_sessions_key(session.user_id), expire_seconds)
                return ok
        except RedisConnectionError:
            raise
        except Exception as e:
            logger.error(f"Session SET error for session '{getattr(session, 'session_id', 'unknown')}': {e}")
            return False

    def get_session(self, session_id: str) -> Optional[UserSession]:
        """Retrieve a session by ID from the session DB."""
        try:
            with self.get_connection(use_session_db=True) as client:
                raw = client.get(self._session_key(session_id))
                if not raw:
                    return None
                return UserSession.from_redis(raw)
        except RedisConnectionError:
            raise
        except Exception as e:
            logger.error(f"Session GET error for session '{session_id}': {e}")
            return None

    def delete_session(self, session_id: str) -> bool:
        """Delete a session by ID and remove from the user index set if possible."""
        try:
            with self.get_connection(use_session_db=True) as client:
                # Attempt to fetch to know the user_id for index cleanup
                raw = client.get(self._session_key(session_id))
                user_id: Optional[str] = None
                try:
                    if raw:
                        user_id = UserSession.from_redis(raw).user_id
                except Exception:
                    user_id = None

                deleted = bool(client.delete(self._session_key(session_id)))
                if deleted and user_id:
                    client.srem(self._user_sessions_key(user_id), session_id)
                return deleted
        except RedisConnectionError:
            raise
        except Exception as e:
            logger.error(f"Session DELETE error for session '{session_id}': {e}")
            return False

    def get_user_sessions(self, user_id: str) -> List[UserSession]:
        """Return all sessions for a given user from the session DB."""
        try:
            with self.get_connection(use_session_db=True) as client:
                sessions: List[UserSession] = []
                session_ids = list(client.smembers(self._user_sessions_key(user_id)) or [])

                if session_ids:
                    pipeline = client.pipeline()
                    for session_id in session_ids:
                        pipeline.get(self._session_key(session_id))
                    results = pipeline.execute()
                    for raw in results:
                        if not raw:
                            continue
                        try:
                            session = UserSession.from_redis(raw)
                            if session.user_id == user_id:
                                sessions.append(session)
                        except Exception as parse_error:
                            logger.warning(f"Failed to parse session for user '{user_id}': {parse_error}")
                else:
                    # Fallback: scan all sessions
                    for key in client.scan_iter(match="session:*"):
                        try:
                            raw = client.get(key)
                            if not raw:
                                continue
                            session = UserSession.from_redis(raw)
                            if session.user_id == user_id:
                                sessions.append(session)
                        except Exception as parse_error:
                            logger.warning(f"Failed to parse session key '{key}': {parse_error}")
                return sessions
        except RedisConnectionError:
            raise
        except Exception as e:
            logger.error(f"Get user sessions error for user '{user_id}': {e}")
            return []

    def get_all_active_sessions(self) -> List[UserSession]:
        """Return all active, non-expired sessions from the session DB."""
        try:
            with self.get_connection(use_session_db=True) as client:
                sessions: List[UserSession] = []
                expire_hours = int(config.SESSION_EXPIRE_HOURS)
                for key in client.scan_iter(match="session:*"):
                    try:
                        raw = client.get(key)
                        if not raw:
                            continue
                        session = UserSession.from_redis(raw)
                        # Check active flag and expiration window
                        if not session.is_active:
                            continue
                        expire_time = session.last_activity + timedelta(hours=expire_hours)
                        if datetime.now() <= expire_time:
                            sessions.append(session)
                    except Exception as parse_error:
                        logger.warning(f"Failed to parse session key '{key}': {parse_error}")
                        continue
                return sessions
        except RedisConnectionError:
            raise
        except Exception as e:
            logger.error(f"Get all active sessions error: {e}")
            return []

    def cleanup_expired_sessions(self) -> int:
        """Remove sessions that are inactive or past the expiration window. Returns count removed."""
        try:
            cleaned = 0
            expire_hours = int(config.SESSION_EXPIRE_HOURS)
            with self.get_connection(use_session_db=True) as client:
                for key in client.scan_iter(match="session:*"):
                    try:
                        raw = client.get(key)
                        if not raw:
                            # Already gone
                            continue
                        session = UserSession.from_redis(raw)
                        should_delete = (not session.is_active)
                        if not should_delete:
                            expire_time = session.last_activity + timedelta(hours=expire_hours)
                            if datetime.now() > expire_time:
                                should_delete = True
                        if should_delete:
                            if client.delete(key):
                                cleaned += 1
                                # Clean user index set
                                client.srem(self._user_sessions_key(session.user_id), session.session_id)
                    except Exception as parse_error:
                        logger.warning(f"Error cleaning session key '{key}': {parse_error}")
                        continue
            return cleaned
        except RedisConnectionError:
            raise
        except Exception as e:
            logger.error(f"Cleanup expired sessions error: {e}")
            return 0
    
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
    
    def reconnect(self):
        """Attempt to reconnect to Redis servers."""
        try:
            # Close existing connections
            self.client.connection_pool.disconnect()
            self.session_client.connection_pool.disconnect()
            
            # Recreate connection pools
            self.pool = redis.ConnectionPool.from_url(
                config.REDIS_URL,
                max_connections=20,
                retry_on_timeout=True,
                socket_keepalive=True,
                socket_keepalive_options={}
            )
            self.client = redis.Redis(connection_pool=self.pool, decode_responses=True)
            
            session_url = config.REDIS_URL.replace('/0', f'/{config.REDIS_SESSION_DB}')
            self.session_pool = redis.ConnectionPool.from_url(
                session_url,
                max_connections=10,
                retry_on_timeout=True,
                socket_keepalive=True,
                socket_keepalive_options={}
            )
            self.session_client = redis.Redis(connection_pool=self.session_pool, decode_responses=True)
            
            # Test new connections
            self._test_connections()
            logger.info("Redis reconnection successful")
            
        except Exception as e:
            logger.error(f"Redis reconnection failed: {e}")
            raise RedisConnectionError(f"Reconnection failed: {e}")

    # Health and maintenance methods
    def health_check(self) -> bool:
        """Simple Redis health check that returns boolean status."""
        try:
            # Test main Redis connection
            with self.get_connection() as client:
                client.ping()
            
            # Test session Redis connection
            with self.get_connection(use_session_db=True) as client:
                client.ping()
            
            return True
        except Exception as e:
            logger.error(f"Redis health check failed: {e}")
            return False
    
    def detailed_health_check(self) -> Dict[str, Any]:
        """Comprehensive Redis health check with detailed information."""
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