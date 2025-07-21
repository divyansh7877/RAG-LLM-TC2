"""
Redis client utilities for session management and task queue.
"""
import json
import redis
from typing import Optional, Dict, Any
from datetime import datetime, timedelta
from .config import config


class RedisClient:
    """Redis client wrapper with connection pooling."""
    
    def __init__(self):
        self.pool = redis.ConnectionPool.from_url(config.REDIS_URL)
        self.client = redis.Redis(connection_pool=self.pool)
        
        # Separate client for sessions
        session_url = config.REDIS_URL.replace('/0', f'/{config.REDIS_SESSION_DB}')
        self.session_pool = redis.ConnectionPool.from_url(session_url)
        self.session_client = redis.Redis(connection_pool=self.session_pool)
    
    def get(self, key: str) -> Optional[str]:
        """Get value from Redis."""
        try:
            value = self.client.get(key)
            return value.decode('utf-8') if value else None
        except Exception as e:
            print(f"Redis GET error: {e}")
            return None
    
    def set(self, key: str, value: str, expire_seconds: Optional[int] = None) -> bool:
        """Set value in Redis with optional expiration."""
        try:
            return self.client.set(key, value, ex=expire_seconds)
        except Exception as e:
            print(f"Redis SET error: {e}")
            return False
    
    def delete(self, key: str) -> bool:
        """Delete key from Redis."""
        try:
            return bool(self.client.delete(key))
        except Exception as e:
            print(f"Redis DELETE error: {e}")
            return False
    
    def exists(self, key: str) -> bool:
        """Check if key exists in Redis."""
        try:
            return bool(self.client.exists(key))
        except Exception as e:
            print(f"Redis EXISTS error: {e}")
            return False
    
    def get_json(self, key: str) -> Optional[Dict[str, Any]]:
        """Get JSON value from Redis."""
        value = self.get(key)
        if value:
            try:
                return json.loads(value)
            except json.JSONDecodeError as e:
                print(f"JSON decode error: {e}")
        return None
    
    def set_json(self, key: str, value: Dict[str, Any], expire_seconds: Optional[int] = None) -> bool:
        """Set JSON value in Redis with optional expiration."""
        try:
            json_str = json.dumps(value, default=str)
            return self.set(key, json_str, expire_seconds)
        except Exception as e:
            print(f"JSON encode error: {e}")
            return False
    
    # Session-specific methods
    def get_session(self, session_id: str) -> Optional[Dict[str, Any]]:
        """Get session data from Redis."""
        try:
            value = self.session_client.get(f"session:{session_id}")
            if value:
                return json.loads(value.decode('utf-8'))
        except Exception as e:
            print(f"Session GET error: {e}")
        return None
    
    def set_session(self, session_id: str, session_data: Dict[str, Any]) -> bool:
        """Set session data in Redis with expiration."""
        try:
            json_str = json.dumps(session_data, default=str)
            expire_seconds = config.SESSION_EXPIRE_HOURS * 3600
            return self.session_client.set(f"session:{session_id}", json_str, ex=expire_seconds)
        except Exception as e:
            print(f"Session SET error: {e}")
            return False
    
    def delete_session(self, session_id: str) -> bool:
        """Delete session from Redis."""
        try:
            return bool(self.session_client.delete(f"session:{session_id}"))
        except Exception as e:
            print(f"Session DELETE error: {e}")
            return False
    
    def cleanup_expired_sessions(self) -> int:
        """Clean up expired sessions and return count of cleaned sessions."""
        try:
            # Redis automatically handles expiration, but we can scan for any orphaned data
            pattern = "session:*"
            keys = self.session_client.keys(pattern)
            cleaned = 0
            
            for key in keys:
                # Check if key still exists (not expired)
                if not self.session_client.exists(key):
                    cleaned += 1
            
            return cleaned
        except Exception as e:
            print(f"Session cleanup error: {e}")
            return 0
    
    def health_check(self) -> bool:
        """Check Redis connection health."""
        try:
            self.client.ping()
            self.session_client.ping()
            return True
        except Exception as e:
            print(f"Redis health check failed: {e}")
            return False


# Global Redis client instance
redis_client = RedisClient()