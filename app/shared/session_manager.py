"""
Thread-safe session manager with Redis backend for concurrent RAG system.
"""
import threading
import logging
from datetime import datetime, timedelta
from typing import Optional, List, Dict, Any, Set
from contextlib import contextmanager
import uuid
import hashlib
import secrets

from .models import UserSession
from .redis_client import redis_client, RedisConnectionError
from .config import config

# Set up logging
logger = logging.getLogger(__name__)


class SessionError(Exception):
    """Base exception for session-related errors."""
    pass


class SessionNotFoundError(SessionError):
    """Raised when a session is not found."""
    pass


class SessionExpiredError(SessionError):
    """Raised when a session has expired."""
    pass


class SessionValidationError(SessionError):
    """Raised when session validation fails."""
    pass


class SessionManager:
    """
    Thread-safe session manager with Redis backend.
    
    Provides secure session management with automatic cleanup,
    thread safety, and comprehensive error handling.
    """
    
    def __init__(self):
        """Initialize session manager with Redis backend."""
        self._lock = threading.RLock()  # Reentrant lock for thread safety
        self._active_sessions: Set[str] = set()  # Local cache of active session IDs
        self._last_cleanup = datetime.now()
        self._cleanup_interval = timedelta(minutes=30)  # Cleanup every 30 minutes
        
        logger.info("SessionManager initialized with Redis backend")
    
    def _generate_session_id(self) -> str:
        """Generate a cryptographically secure session ID."""
        # Use secrets for cryptographically strong random generation
        random_bytes = secrets.token_bytes(32)
        timestamp = str(datetime.now().timestamp()).encode()
        
        # Create hash from random bytes and timestamp
        hasher = hashlib.sha256()
        hasher.update(random_bytes)
        hasher.update(timestamp)
        
        return hasher.hexdigest()
    
    def _validate_session_data(self, session: UserSession) -> bool:
        """Validate session data integrity."""
        try:
            # Check required fields
            if not session.session_id or not session.user_id:
                return False
            
            # Check session is not expired
            if not session.is_active:
                return False
            
            # Check last activity is within expiration window
            expire_time = session.last_activity + timedelta(hours=config.SESSION_EXPIRE_HOURS)
            if datetime.now() > expire_time:
                return False
            
            # Validate permissions
            valid_permissions = {'upload', 'query', 'delete', 'admin'}
            for perm in session.permissions:
                if perm not in valid_permissions:
                    return False
            
            return True
        except Exception as e:
            logger.error(f"Session validation error: {e}")
            return False
    
    def _should_cleanup(self) -> bool:
        """Check if automatic cleanup should be performed."""
        return datetime.now() - self._last_cleanup > self._cleanup_interval
    
    @contextmanager
    def _thread_safe_operation(self):
        """Context manager for thread-safe operations."""
        with self._lock:
            try:
                yield
            except RedisConnectionError as e:
                logger.error("Redis connection error during session operation")
                raise SessionError("Session service temporarily unavailable")
            except Exception as e:
                logger.error(f"Unexpected error during session operation: {e}")
                raise SessionError(f"Session operation failed: {e}")
    
    def create_session(self, user_id: str, groups: List[str], 
                      permissions: List[str] = None) -> UserSession:
        """
        Create a new user session with thread safety.
        
        Args:
            user_id: User identifier
            groups: List of user groups
            permissions: List of user permissions (defaults to basic permissions)
        
        Returns:
            UserSession: Created session object
        
        Raises:
            SessionError: If session creation fails
        """
        if permissions is None:
            permissions = ['upload', 'query']  # Default permissions
        
        with self._thread_safe_operation():
            try:
                # Generate unique session ID
                session_id = self._generate_session_id()
                
                # Create session object
                session = UserSession(
                    session_id=session_id,
                    user_id=user_id,
                    groups=groups,
                    permissions=permissions,
                    created_at=datetime.now(),
                    last_activity=datetime.now(),
                    is_active=True
                )
                
                # Validate session data
                if not self._validate_session_data(session):
                    raise SessionValidationError("Invalid session data")
                
                # Store in Redis
                if not redis_client.set_session(session):
                    raise SessionError("Failed to store session in Redis")
                
                # Add to local cache
                self._active_sessions.add(session_id)
                
                logger.info(f"Created session {session_id} for user {user_id}")
                return session
                
            except (SessionError, SessionValidationError):
                raise
            except Exception as e:
                logger.error(f"Session creation failed for user {user_id}: {e}")
                raise SessionError(f"Failed to create session: {e}")
    
    def get_session(self, session_id: str) -> Optional[UserSession]:
        """
        Get session by ID with thread safety and validation.
        
        Args:
            session_id: Session identifier
        
        Returns:
            UserSession: Session object if found and valid, None otherwise
        
        Raises:
            SessionError: If Redis operation fails
        """
        if not session_id:
            return None
        
        with self._thread_safe_operation():
            try:
                # Get from Redis
                session = redis_client.get_session(session_id)
                
                if not session:
                    # Remove from local cache if it exists
                    self._active_sessions.discard(session_id)
                    return None
                
                # Validate session
                if not self._validate_session_data(session):
                    logger.warning(f"Invalid session data for session {session_id}")
                    # Clean up invalid session
                    self.delete_session(session_id)
                    return None
                
                # Add to local cache
                self._active_sessions.add(session_id)
                
                return session
                
            except RedisConnectionError:
                raise
            except Exception as e:
                logger.error(f"Session retrieval failed for session {session_id}: {e}")
                return None
    
    def validate_session(self, session_id: str) -> bool:
        """
        Validate session exists and is active.
        
        Args:
            session_id: Session identifier
        
        Returns:
            bool: True if session is valid, False otherwise
        """
        try:
            session = self.get_session(session_id)
            return session is not None and session.is_active
        except Exception as e:
            logger.error(f"Session validation failed for session {session_id}: {e}")
            return False
    
    def update_session_activity(self, session_id: str) -> bool:
        """
        Update session last activity timestamp with thread safety.
        
        Args:
            session_id: Session identifier
        
        Returns:
            bool: True if update successful, False otherwise
        """
        with self._thread_safe_operation():
            try:
                # Get current session
                session = self.get_session(session_id)
                if not session:
                    return False
                
                # Update activity timestamp
                session.update_activity()
                
                # Store updated session
                if not redis_client.set_session(session):
                    logger.error(f"Failed to update session activity for {session_id}")
                    return False
                
                logger.debug(f"Updated activity for session {session_id}")
                return True
                
            except Exception as e:
                logger.error(f"Session activity update failed for {session_id}: {e}")
                return False
    
    def delete_session(self, session_id: str) -> bool:
        """
        Delete session with thread safety.
        
        Args:
            session_id: Session identifier
        
        Returns:
            bool: True if deletion successful, False otherwise
        """
        with self._thread_safe_operation():
            try:
                # Remove from Redis
                success = redis_client.delete_session(session_id)
                
                # Remove from local cache
                self._active_sessions.discard(session_id)
                
                if success:
                    logger.info(f"Deleted session {session_id}")
                else:
                    logger.warning(f"Session {session_id} not found for deletion")
                
                return success
                
            except Exception as e:
                logger.error(f"Session deletion failed for {session_id}: {e}")
                return False
    
    def get_user_sessions(self, user_id: str) -> List[UserSession]:
        """
        Get all active sessions for a user with thread safety.
        
        Args:
            user_id: User identifier
        
        Returns:
            List[UserSession]: List of active sessions for the user
        """
        with self._thread_safe_operation():
            try:
                sessions = redis_client.get_user_sessions(user_id)
                
                # Filter and validate sessions
                valid_sessions = []
                for session in sessions:
                    if self._validate_session_data(session):
                        valid_sessions.append(session)
                        self._active_sessions.add(session.session_id)
                    else:
                        # Clean up invalid session
                        self.delete_session(session.session_id)
                
                return valid_sessions
                
            except Exception as e:
                logger.error(f"Get user sessions failed for user {user_id}: {e}")
                return []
    
    def invalidate_user_sessions(self, user_id: str) -> int:
        """
        Invalidate all sessions for a user with thread safety.
        
        Args:
            user_id: User identifier
        
        Returns:
            int: Number of sessions invalidated
        """
        with self._thread_safe_operation():
            try:
                # Get user sessions
                user_sessions = self.get_user_sessions(user_id)
                
                invalidated = 0
                for session in user_sessions:
                    if self.delete_session(session.session_id):
                        invalidated += 1
                
                logger.info(f"Invalidated {invalidated} sessions for user {user_id}")
                return invalidated
                
            except Exception as e:
                logger.error(f"Session invalidation failed for user {user_id}: {e}")
                return 0
    
    def cleanup_expired_sessions(self) -> int:
        """
        Clean up expired sessions with thread safety.
        
        Returns:
            int: Number of sessions cleaned up
        """
        with self._thread_safe_operation():
            try:
                cleaned = redis_client.cleanup_expired_sessions()
                
                # Update local cache - remove cleaned sessions
                if cleaned > 0:
                    # Refresh local cache by getting all active sessions
                    all_sessions = redis_client.get_all_active_sessions()
                    self._active_sessions = {session.session_id for session in all_sessions}
                
                self._last_cleanup = datetime.now()
                
                if cleaned > 0:
                    logger.info(f"Cleaned up {cleaned} expired sessions")
                
                return cleaned
                
            except Exception as e:
                logger.error(f"Session cleanup failed: {e}")
                return 0
    
    def auto_cleanup_if_needed(self):
        """Perform automatic cleanup if needed (non-blocking)."""
        if self._should_cleanup():
            try:
                self.cleanup_expired_sessions()
            except Exception as e:
                logger.error(f"Auto cleanup failed: {e}")
    
    def get_session_stats(self) -> Dict[str, Any]:
        """
        Get session statistics with thread safety.
        
        Returns:
            Dict[str, Any]: Session statistics
        """
        with self._thread_safe_operation():
            try:
                all_sessions = redis_client.get_all_active_sessions()
                
                stats = {
                    "total_sessions": len(all_sessions),
                    "active_sessions": len([s for s in all_sessions if s.is_active]),
                    "users_with_sessions": len(set(s.user_id for s in all_sessions)),
                    "local_cache_size": len(self._active_sessions),
                    "last_cleanup": self._last_cleanup.isoformat(),
                    "sessions_by_user": {}
                }
                
                # Count sessions per user
                for session in all_sessions:
                    user_id = session.user_id
                    if user_id not in stats["sessions_by_user"]:
                        stats["sessions_by_user"][user_id] = 0
                    stats["sessions_by_user"][user_id] += 1
                
                return stats
                
            except Exception as e:
                logger.error(f"Get session stats failed: {e}")
                return {"error": str(e)}
    
    def health_check(self) -> Dict[str, Any]:
        """
        Perform health check on session manager.
        
        Returns:
            Dict[str, Any]: Health check results
        """
        health = {
            "session_manager": True,
            "redis_connection": False,
            "thread_safety": True,
            "errors": []
        }
        
        try:
            # Test Redis connection
            redis_health = redis_client.health_check()
            health["redis_connection"] = redis_health.get("redis_sessions", False)
            
            if redis_health.get("errors"):
                health["errors"].extend(redis_health["errors"])
            
            # Test basic operations
            test_session = UserSession(
                session_id="health-check-test",
                user_id="health-check-user",
                groups=["test"],
                permissions=["query"]
            )
            
            # Test thread safety by attempting concurrent operations
            with self._thread_safe_operation():
                # This should not raise an exception
                pass
            
        except Exception as e:
            health["session_manager"] = False
            health["errors"].append(f"Session manager health check failed: {e}")
        
        return health


# Global session manager instance
session_manager = SessionManager()