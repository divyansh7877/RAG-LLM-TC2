"""
Authentication and authorization middleware for FastAPI using Keycloak.
"""
import logging
from typing import Optional, List
from fastapi import HTTPException, status, Depends, Request
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials

from .auth import auth_manager, TokenExpiredError, TokenInvalidError, AuthenticationError
from .models import User
from .error_handling import error_handler, set_log_context

# Set up logging
logger = logging.getLogger(__name__)

# HTTP Bearer token scheme
security = HTTPBearer()

async def get_current_user(credentials: Optional[HTTPAuthorizationCredentials] = Depends(security)) -> User:
    """Dependency to get current user from Keycloak token."""
    if credentials is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Bearer token required",
            headers={"WWW-Authenticate": "Bearer"},
        )
    try:
        token = credentials.credentials
        payload = auth_manager.decode_token(token)
        # Normalize groups from either 'groups' claim or single 'group'
        groups_claim = payload.get('groups') or payload.get('group') or []
        
        # Extract user info from token claims
        user = User(
            id=payload.get('sub'),
            username=payload.get('preferred_username'),
            email=payload.get('email'),
            groups=groups_claim,
            roles=payload.get('realm_access', {}).get('roles', []),
        )
        set_log_context(user_id=user.id)
        return user
    except TokenExpiredError:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Token has expired",
            headers={"WWW-Authenticate": "Bearer"},
        )
    except TokenInvalidError as e:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=f"Invalid token: {e}",
            headers={"WWW-Authenticate": "Bearer"},
        )
    except AuthenticationError as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Authentication service error: {e}"
        )

def require_roles(required_roles: List[str]):
    """
    Create a dependency that requires specific roles from the Keycloak token.
    """
    async def role_checker(current_user: User = Depends(get_current_user)) -> User:
        user_roles = set(current_user.roles)
        required = set(required_roles)
        if not required.issubset(user_roles):
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"Insufficient permissions. Required roles: {', '.join(required)}"
            )
        return current_user
    return role_checker

# Rate limiting middleware (basic implementation)
class RateLimitMiddleware:
    """Basic rate limiting middleware."""
    
    def __init__(self):
        """Initialize rate limiting middleware."""
        self._request_counts = {}  # In production, use Redis
        self._last_reset = {}
        logger.info("RateLimitMiddleware initialized")
    
    def _get_client_id(self, request: Request) -> str:
        """Get client identifier for rate limiting."""
        # Use IP address as client ID (in production, consider user ID)
        return request.client.host if request.client else "unknown"
    
    def _reset_if_needed(self, client_id: str, window_seconds: int):
        """Reset rate limit counter if window has passed."""
        import time
        current_time = time.time()
        
        if client_id not in self._last_reset:
            self._last_reset[client_id] = current_time
            self._request_counts[client_id] = 0
        elif current_time - self._last_reset[client_id] >= window_seconds:
            self._last_reset[client_id] = current_time
            self._request_counts[client_id] = 0
    
    def check_rate_limit(
        self, 
        request: Request, 
        max_requests: int, 
        window_seconds: int
    ) -> bool:
        """
        Check if request is within rate limit.
        
        Args:
            request: FastAPI request object
            max_requests: Maximum requests allowed
            window_seconds: Time window in seconds
        
        Returns:
            bool: True if within limit, False otherwise
        """
        client_id = self._get_client_id(request)
        
        # Reset counter if needed
        self._reset_if_needed(client_id, window_seconds)
        
        # Check current count
        current_count = self._request_counts.get(client_id, 0)
        
        if current_count >= max_requests:
            return False
        
        # Increment counter
        self._request_counts[client_id] = current_count + 1
        return True
    
    def create_rate_limiter(self, max_requests: int, window_seconds: int):
        """
        Create a rate limiting dependency.
        
        Args:
            max_requests: Maximum requests allowed
            window_seconds: Time window in seconds
        
        Returns:
            Callable: FastAPI dependency function
        """
        async def rate_limit_checker(request: Request):
            """Check rate limit for request."""
            if not self.check_rate_limit(request, max_requests, window_seconds):
                raise HTTPException(
                    status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                    detail=f"Rate limit exceeded. Max {max_requests} requests per {window_seconds} seconds."
                )
        
        return rate_limit_checker


# Global rate limiting instance
rate_limiter = RateLimitMiddleware()