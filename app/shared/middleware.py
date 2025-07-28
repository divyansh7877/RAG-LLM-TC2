"""
Authentication and authorization middleware for FastAPI.
"""
import logging
from typing import Optional, List, Callable, Any
from fastapi import HTTPException, status, Depends, Request
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from functools import wraps

from .auth import auth_manager, AuthenticationError, TokenExpiredError, TokenInvalidError
from .models import UserSession
from .error_handling import error_handler, set_log_context, clear_log_context, with_error_handling, StructuredLogger
from .monitoring import metric_collector, alert_manager

# Set up structured logging
logger = StructuredLogger(__name__)

# HTTP Bearer token scheme
security = HTTPBearer(auto_error=False)


class AuthenticationMiddleware:
    """Authentication middleware for FastAPI applications."""
    
    def __init__(self):
        """Initialize authentication middleware."""
        self.auth_manager = auth_manager
        logger.info("AuthenticationMiddleware initialized")
    
    @with_error_handling()
    async def get_current_user_optional(
        self, 
        credentials: Optional[HTTPAuthorizationCredentials] = Depends(security)
    ) -> Optional[UserSession]:
        """
        Get current user from token (optional - doesn't raise if no token).
        
        Args:
            credentials: HTTP Bearer credentials
        
        Returns:
            UserSession: Current user session if authenticated, None otherwise
        """
        if not credentials:
            return None
        
        try:
            user_session = self.auth_manager.get_current_user(credentials.credentials)
            if user_session:
                # Set log context for this user
                set_log_context(
                    user_id=user_session.user_id,
                    session_id=user_session.session_id
                )
            return user_session
        except Exception as e:
            # Handle error through centralized system
            error_handler.handle_error(e, {
                'operation': 'optional_authentication',
                'has_credentials': bool(credentials)
            })
            return None
    
    @with_error_handling()
    async def get_current_user(
        self, 
        credentials: Optional[HTTPAuthorizationCredentials] = Depends(security)
    ) -> UserSession:
        """
        Get current user from token (required - raises if no valid token).
        
        Args:
            credentials: HTTP Bearer credentials
        
        Returns:
            UserSession: Current user session
        
        Raises:
            HTTPException: If authentication fails
        """
        if not credentials:
            error_context = {
                'operation': 'authentication',
                'error_type': 'missing_credentials'
            }
            error_handler.handle_error(
                AuthenticationError("No credentials provided"), 
                error_context
            )
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Authentication required",
                headers={"WWW-Authenticate": "Bearer"},
            )
        
        try:
            user_session = self.auth_manager.get_current_user(credentials.credentials)
            if not user_session:
                error_context = {
                    'operation': 'authentication',
                    'error_type': 'invalid_credentials'
                }
                error_handler.handle_error(
                    AuthenticationError("Invalid credentials"), 
                    error_context
                )
                raise HTTPException(
                    status_code=status.HTTP_401_UNAUTHORIZED,
                    detail="Invalid authentication credentials",
                    headers={"WWW-Authenticate": "Bearer"},
                )
            
            # Set log context for authenticated user
            set_log_context(
                user_id=user_session.user_id,
                session_id=user_session.session_id
            )
            
            return user_session
        
        except TokenExpiredError as e:
            error_handler.handle_error(e, {'operation': 'authentication'})
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Token has expired",
                headers={"WWW-Authenticate": "Bearer"},
            )
        
        except TokenInvalidError as e:
            error_handler.handle_error(e, {'operation': 'authentication'})
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail=f"Invalid token: {e}",
                headers={"WWW-Authenticate": "Bearer"},
            )
        
        except AuthenticationError as e:
            error_handler.handle_error(e, {'operation': 'authentication'})
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail=f"Authentication failed: {e}",
                headers={"WWW-Authenticate": "Bearer"},
            )
        
        except Exception as e:
            error_handler.handle_error(e, {'operation': 'authentication'})
            logger.error(f"Authentication error: {e}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Authentication service error"
            )
    
    def require_permissions(self, required_permissions: List[str]):
        """
        Create a dependency that requires specific permissions.
        
        Args:
            required_permissions: List of required permissions
        
        Returns:
            Callable: FastAPI dependency function
        """
        async def permission_checker(
            current_user: UserSession = Depends(self.get_current_user)
        ) -> UserSession:
            """Check if user has required permissions."""
            user_permissions = set(current_user.permissions)
            required_perms = set(required_permissions)
            
            if not required_perms.issubset(user_permissions):
                missing_perms = required_perms - user_permissions
                raise HTTPException(
                    status_code=status.HTTP_403_FORBIDDEN,
                    detail=f"Insufficient permissions. Missing: {', '.join(missing_perms)}"
                )
            
            return current_user
        
        return permission_checker
    
    def require_groups(self, required_groups: List[str]):
        """
        Create a dependency that requires membership in specific groups.
        
        Args:
            required_groups: List of required groups
        
        Returns:
            Callable: FastAPI dependency function
        """
        async def group_checker(
            current_user: UserSession = Depends(self.get_current_user)
        ) -> UserSession:
            """Check if user belongs to required groups."""
            user_groups = set(current_user.groups)
            required_group_set = set(required_groups)
            
            # User must belong to at least one of the required groups
            if not user_groups.intersection(required_group_set):
                raise HTTPException(
                    status_code=status.HTTP_403_FORBIDDEN,
                    detail=f"Access denied. Required group membership: {', '.join(required_groups)}"
                )
            
            return current_user
        
        return group_checker
    
    async def validate_token_endpoint(
        self, 
        credentials: Optional[HTTPAuthorizationCredentials] = Depends(security)
    ) -> dict:
        """
        Endpoint dependency for token validation.
        
        Args:
            credentials: HTTP Bearer credentials
        
        Returns:
            dict: Token validation result
        
        Raises:
            HTTPException: If token validation fails
        """
        if not credentials:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Token required",
                headers={"WWW-Authenticate": "Bearer"},
            )
        
        try:
            user_info = self.auth_manager.validate_token(credentials.credentials)
            return {
                "valid": True,
                "user_info": user_info
            }
        
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
        
        except Exception as e:
            logger.error(f"Token validation error: {e}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Token validation service error"
            )


# Global middleware instance
auth_middleware = AuthenticationMiddleware()

# Convenience dependency functions
get_current_user = auth_middleware.get_current_user
get_current_user_optional = auth_middleware.get_current_user_optional
require_permissions = auth_middleware.require_permissions
require_groups = auth_middleware.require_groups
validate_token = auth_middleware.validate_token_endpoint


# Decorator for function-based permission checking
def requires_permissions(permissions: List[str]):
    """
    Decorator to require specific permissions for a function.
    
    Args:
        permissions: List of required permissions
    
    Returns:
        Callable: Decorated function
    """
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        async def wrapper(*args, **kwargs):
            # Extract current_user from kwargs if present
            current_user = kwargs.get('current_user')
            if not current_user or not isinstance(current_user, UserSession):
                raise HTTPException(
                    status_code=status.HTTP_401_UNAUTHORIZED,
                    detail="Authentication required"
                )
            
            # Check permissions
            user_permissions = set(current_user.permissions)
            required_perms = set(permissions)
            
            if not required_perms.issubset(user_permissions):
                missing_perms = required_perms - user_permissions
                raise HTTPException(
                    status_code=status.HTTP_403_FORBIDDEN,
                    detail=f"Insufficient permissions. Missing: {', '.join(missing_perms)}"
                )
            
            return await func(*args, **kwargs)
        
        return wrapper
    return decorator


def requires_groups(groups: List[str]):
    """
    Decorator to require specific group membership for a function.
    
    Args:
        groups: List of required groups
    
    Returns:
        Callable: Decorated function
    """
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        async def wrapper(*args, **kwargs):
            # Extract current_user from kwargs if present
            current_user = kwargs.get('current_user')
            if not current_user or not isinstance(current_user, UserSession):
                raise HTTPException(
                    status_code=status.HTTP_401_UNAUTHORIZED,
                    detail="Authentication required"
                )
            
            # Check group membership
            user_groups = set(current_user.groups)
            required_group_set = set(groups)
            
            if not user_groups.intersection(required_group_set):
                raise HTTPException(
                    status_code=status.HTTP_403_FORBIDDEN,
                    detail=f"Access denied. Required group membership: {', '.join(groups)}"
                )
            
            return await func(*args, **kwargs)
        
        return wrapper
    return decorator


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