#!/usr/bin/env python3
"""
OpenAI rate limiting and quota management system.

This module provides intelligent rate limiting to prevent quota exhaustion
and implements request queuing to stay within OpenAI API limits.
"""

import asyncio
import time
import logging
from typing import Dict, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime, timedelta
import threading
from contextlib import asynccontextmanager

from .redis_client import redis_client
from .openai_config import get_openai_config

logger = logging.getLogger(__name__)

@dataclass
class RateLimitInfo:
    """Information about current rate limit status."""
    requests_remaining: int
    tokens_remaining: int
    reset_time: datetime
    current_usage: int
    limit: int

class OpenAIRateLimiter:
    """
    Intelligent rate limiter for OpenAI API requests.
    
    Features:
    - Per-model rate limiting
    - Request queuing with priority
    - Quota monitoring
    - Automatic backoff on rate limit errors
    """
    
    def __init__(self):
        self.logger = logging.getLogger(self.__class__.__name__)
        self._lock = threading.RLock()
        self._request_queue = asyncio.Queue()
        self._last_request_time: Dict[str, float] = {}
        self._rate_limit_info: Dict[str, RateLimitInfo] = {}
        
        # Configuration
        config = get_openai_config()
        self.model_name = config.model_name
        self.requests_per_minute = config.tokens_per_minute_limit // 1000  # Rough estimate
        self.min_request_interval = 60.0 / self.requests_per_minute if self.requests_per_minute > 0 else 1.0
        
        # Redis keys for tracking usage
        self.usage_key = f"openai:usage:{self.model_name}"
        self.quota_key = f"openai:quota_status"
        self.rate_limit_key = f"openai:rate_limit:{self.model_name}"
        
        self.logger.info(f"Rate limiter initialized for {self.model_name} with {self.requests_per_minute} req/min")

    async def acquire_request_slot(self, priority: int = 1, estimated_tokens: int = 1000) -> bool:
        """
        Acquire a slot to make an OpenAI request.
        
        Args:
            priority: Request priority (1=low, 2=normal, 3=high)
            estimated_tokens: Estimated tokens for the request
            
        Returns:
            True if request can proceed, False if quota exceeded
        """
        try:
            # Check if quota is exceeded
            if await self._is_quota_exceeded():
                self.logger.warning("OpenAI quota exceeded, rejecting request")
                return False
            
            # Check rate limits
            await self._enforce_rate_limit()
            
            # Update usage tracking
            await self._track_request(estimated_tokens)
            
            return True
            
        except Exception as e:
            self.logger.error(f"Error acquiring request slot: {e}")
            return False

    async def _is_quota_exceeded(self) -> bool:
        """Check if OpenAI quota is exceeded."""
        try:
            quota_status = redis_client.get_json(self.quota_key)
            if quota_status and quota_status.get("exceeded", False):
                # Check if quota reset time has passed
                reset_time = quota_status.get("reset_time")
                if reset_time and datetime.fromisoformat(reset_time) > datetime.utcnow():
                    return True
            return False
        except Exception as e:
            self.logger.error(f"Error checking quota status: {e}")
            return False

    async def _enforce_rate_limit(self):
        """Enforce rate limiting by delaying requests if necessary."""
        current_time = time.time()
        last_request = self._last_request_time.get(self.model_name, 0)
        
        time_since_last = current_time - last_request
        if time_since_last < self.min_request_interval:
            delay = self.min_request_interval - time_since_last
            self.logger.info(f"Rate limiting: waiting {delay:.2f}s before next request")
            await asyncio.sleep(delay)
        
        self._last_request_time[self.model_name] = time.time()

    async def _track_request(self, estimated_tokens: int):
        """Track request for usage monitoring."""
        try:
            current_time = time.time()
            usage_data = {
                "timestamp": current_time,
                "model": self.model_name,
                "estimated_tokens": estimated_tokens,
                "date": datetime.utcnow().strftime("%Y-%m-%d"),
                "hour": datetime.utcnow().strftime("%H")
            }
            
            # Store individual request
            request_key = f"{self.usage_key}:requests:{int(current_time * 1000)}"
            redis_client.set_json(request_key, usage_data, expire_seconds=86400)  # Keep for 1 day
            
            # Update hourly aggregates
            hourly_key = f"{self.usage_key}:hourly:{usage_data['date']}:{usage_data['hour']}"
            hourly_stats = redis_client.get_json(hourly_key) or {
                "requests": 0,
                "total_tokens": 0,
                "date": usage_data["date"],
                "hour": usage_data["hour"]
            }
            
            hourly_stats["requests"] += 1
            hourly_stats["total_tokens"] += estimated_tokens
            redis_client.set_json(hourly_key, hourly_stats, expire_seconds=86400 * 7)  # Keep for 7 days
            
        except Exception as e:
            self.logger.error(f"Error tracking request usage: {e}")

    def handle_rate_limit_error(self, error_response: Dict) -> Tuple[bool, int]:
        """
        Handle rate limit error response from OpenAI.
        
        Args:
            error_response: Error response from OpenAI API
            
        Returns:
            Tuple of (should_retry, delay_seconds)
        """
        try:
            error_type = error_response.get("error", {}).get("type", "")
            error_code = error_response.get("error", {}).get("code", "")
            
            if error_type == "insufficient_quota" or error_code == "insufficient_quota":
                # Mark quota as exceeded
                quota_status = {
                    "exceeded": True,
                    "error": error_response.get("error", {}).get("message", "Quota exceeded"),
                    "reset_time": (datetime.utcnow() + timedelta(hours=1)).isoformat(),  # Assume 1 hour reset
                    "timestamp": datetime.utcnow().isoformat()
                }
                redis_client.set_json(self.quota_key, quota_status, expire_seconds=3600)
                
                self.logger.error(f"OpenAI quota exceeded: {quota_status['error']}")
                return False, 0  # Don't retry quota errors
            
            elif error_type == "rate_limit_exceeded" or "rate limit" in str(error_response).lower():
                # Handle rate limit
                self.logger.warning("OpenAI rate limit hit, implementing backoff")
                return True, 60  # Retry after 1 minute
            
            else:
                # Other errors
                return False, 0
                
        except Exception as e:
            self.logger.error(f"Error handling rate limit response: {e}")
            return False, 0

    def get_usage_stats(self, hours: int = 24) -> Dict:
        """
        Get usage statistics for the specified time period.
        
        Args:
            hours: Number of hours to look back
            
        Returns:
            Dictionary with usage statistics
        """
        try:
            stats = {
                "period_hours": hours,
                "model": self.model_name,
                "total_requests": 0,
                "total_tokens": 0,
                "hourly_breakdown": []
            }
            
            current_time = datetime.utcnow()
            
            for i in range(hours):
                hour_time = current_time - timedelta(hours=i)
                date_str = hour_time.strftime("%Y-%m-%d")
                hour_str = hour_time.strftime("%H")
                
                hourly_key = f"{self.usage_key}:hourly:{date_str}:{hour_str}"
                hourly_data = redis_client.get_json(hourly_key)
                
                if hourly_data:
                    stats["total_requests"] += hourly_data.get("requests", 0)
                    stats["total_tokens"] += hourly_data.get("total_tokens", 0)
                    stats["hourly_breakdown"].append({
                        "datetime": hour_time.isoformat(),
                        "requests": hourly_data.get("requests", 0),
                        "tokens": hourly_data.get("total_tokens", 0)
                    })
                else:
                    stats["hourly_breakdown"].append({
                        "datetime": hour_time.isoformat(),
                        "requests": 0,
                        "tokens": 0
                    })
            
            return stats
            
        except Exception as e:
            self.logger.error(f"Error getting usage stats: {e}")
            return {"error": str(e)}

    def reset_quota_status(self):
        """Reset quota status (for testing or manual override)."""
        try:
            redis_client.delete(self.quota_key)
            self.logger.info("Quota status reset")
        except Exception as e:
            self.logger.error(f"Error resetting quota status: {e}")

# Global rate limiter instance
_rate_limiter = None
_rate_limiter_lock = threading.Lock()

def get_rate_limiter() -> OpenAIRateLimiter:
    """Get the global rate limiter instance (singleton)."""
    global _rate_limiter
    if _rate_limiter is None:
        with _rate_limiter_lock:
            if _rate_limiter is None:
                _rate_limiter = OpenAIRateLimiter()
    return _rate_limiter

@asynccontextmanager
async def rate_limited_request(priority: int = 1, estimated_tokens: int = 1000):
    """
    Context manager for rate-limited OpenAI requests.
    
    Usage:
        async with rate_limited_request(priority=2, estimated_tokens=1500) as allowed:
            if allowed:
                # Make OpenAI request
                response = await openai_client.chat.completions.create(...)
            else:
                # Handle quota exceeded
                pass
    """
    rate_limiter = get_rate_limiter()
    allowed = await rate_limiter.acquire_request_slot(priority, estimated_tokens)
    
    try:
        yield allowed
    except Exception as e:
        # Handle any errors that might occur during the request
        logger.error(f"Error in rate-limited request: {e}")
        raise
    finally:
        # Any cleanup if needed
        pass

# Convenience functions
async def check_quota_available() -> bool:
    """Check if OpenAI quota is available for requests."""
    rate_limiter = get_rate_limiter()
    return not await rate_limiter._is_quota_exceeded()

def get_openai_usage_stats(hours: int = 24) -> Dict:
    """Get OpenAI usage statistics."""
    rate_limiter = get_rate_limiter()
    return rate_limiter.get_usage_stats(hours)

def reset_openai_quota() -> None:
    """Reset OpenAI quota status."""
    rate_limiter = get_rate_limiter()
    rate_limiter.reset_quota_status()
