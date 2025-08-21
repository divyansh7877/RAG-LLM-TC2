"""
JWT-based authentication system for concurrent RAG system.
"""
import logging
from datetime import datetime, timedelta
from typing import Optional, Dict, Any, List
import jwt
from passlib.context import CryptContext
from passlib.hash import bcrypt
import secrets

from .config import config
from .models import UserSession
from .session_manager import session_manager, SessionError

# Set up logging
logger = logging.getLogger(__name__)

# Password hashing context
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")


class AuthenticationError(Exception):
    """Base exception for authentication errors."""
    pass


class InvalidCredentialsError(AuthenticationError):
    """Raised when credentials are invalid."""
    pass


class TokenExpiredError(AuthenticationError):
    """Raised when JWT token has expired."""
    pass


class TokenInvalidError(AuthenticationError):
    """Raised when JWT token is invalid."""
    pass


class AuthenticationManager:
    """
    JWT-based authentication manager with session integration.
    
    Provides secure authentication with JWT tokens, password hashing,
    and integration with the session management system.
    """
    
    def __init__(self):
        """Initialize authentication manager."""
        self.secret_key = config.SECRET_KEY
        self.algorithm = "HS256"
        self.access_token_expire_minutes = config.ACCESS_TOKEN_EXPIRE_MINUTES
        
        # Ensure secret key is secure
        if self.secret_key == "your-secret-key-change-in-production":
            logger.warning("Using default secret key - change in production!")
        
        # Token blacklist for revoked tokens (in production, use Redis)
        self._revoked_tokens = set()
        
        logger.info("AuthenticationManager initialized")
    
    def _hash_password(self, password: str) -> str:
        """Hash a password using bcrypt."""
        return pwd_context.hash(password)
    
    def _verify_password(self, plain_password: str, hashed_password: str) -> bool:
        """Verify a password against its hash."""
        return pwd_context.verify(plain_password, hashed_password)
    
    def _get_user_data(self, username: str) -> Optional[Dict[str, Any]]:
        """
        Get user data from configuration.
        
        In production, this should query a proper user database.
        """
        if username == "guest":
            return {
                "username": "guest",
                "password_hash": self._hash_password("guest"),
                "groups": ["personal"],
                "permissions": ["query", "upload"]
            }

        user_data = config.USERS.get(username)
        if user_data:
            return {
                "username": username,
                "password_hash": self._hash_password(user_data["password"]),
                "groups": user_data["groups"],
                "permissions": self._get_user_permissions(user_data["groups"])
            }
        return None
    
    def _get_user_permissions(self, groups: List[str]) -> List[str]:
        """
        Get user permissions based on groups.
        
        In production, this should be configurable and stored in database.
        """
        permissions = set()
        
        # Basic permissions for all users
        permissions.add("query")
        
        # Group-based permissions
        if "assistance" in groups:
            permissions.update(["upload", "delete"])
        
        if "admin" in groups:
            permissions.update(["upload", "delete", "admin"])
        
        return list(permissions)
    
    def _create_access_token(self, data: Dict[str, Any], expires_delta: Optional[timedelta] = None) -> str:
        """Create a JWT access token."""
        to_encode = data.copy()
        
        if expires_delta:
            expire = datetime.utcnow() + expires_delta
        else:
            expire = datetime.utcnow() + timedelta(minutes=self.access_token_expire_minutes)
        
        to_encode.update({
            "exp": expire,
            "iat": datetime.utcnow(),
            "jti": secrets.token_hex(16)  # JWT ID for token revocation
        })
        
        encoded_jwt = jwt.encode(to_encode, self.secret_key, algorithm=self.algorithm)
        return encoded_jwt
    
    def _decode_access_token(self, token: str) -> Dict[str, Any]:
        """Decode and validate a JWT access token."""
        try:
            payload = jwt.decode(token, self.secret_key, algorithms=[self.algorithm])
            
            # Check if token is revoked
            jti = payload.get("jti")
            if jti and jti in self._revoked_tokens:
                raise TokenInvalidError("Token has been revoked")
            
            return payload
        
        except jwt.ExpiredSignatureError:
            raise TokenExpiredError("Token has expired")
        except jwt.DecodeError as e:
            raise TokenInvalidError(f"Invalid token: {e}")
        except jwt.InvalidTokenError as e:
            raise TokenInvalidError(f"Invalid token: {e}")
    
    def authenticate_user(self, username: str, password: str) -> Dict[str, Any]:
        """
        Authenticate user with username and password.
        
        Args:
            username: User's username
            password: User's plain text password
        
        Returns:
            Dict containing user info and access token
        
        Raises:
            InvalidCredentialsError: If credentials are invalid
            AuthenticationError: If authentication fails
        """
        try:
            # Get user data
            user_data = self._get_user_data(username)
            if not user_data:
                raise InvalidCredentialsError("Invalid username or password")
            
            # Verify password
            if not self._verify_password(password, user_data["password_hash"]):
                raise InvalidCredentialsError("Invalid username or password")
            
            # Create session
            session = session_manager.create_session(
                user_id=username,
                groups=user_data["groups"],
                permissions=user_data["permissions"]
            )
            
            # Create JWT token
            token_data = {
                "sub": username,  # Subject (user ID)
                "session_id": session.session_id,
                "groups": user_data["groups"],
                "permissions": user_data["permissions"]
            }
            
            access_token = self._create_access_token(token_data)
            
            logger.info(f"User {username} authenticated successfully")
            
            return {
                "access_token": access_token,
                "token_type": "bearer",
                "user_id": username,
                "groups": user_data["groups"],
                "permissions": user_data["permissions"],
                "session_id": session.session_id,
                "expires_in": self.access_token_expire_minutes * 60
            }
        
        except (InvalidCredentialsError, SessionError):
            raise
        except Exception as e:
            logger.error(f"Authentication failed for user {username}: {e}")
            raise AuthenticationError(f"Authentication failed: {e}")
    
    def validate_token(self, token: str) -> Dict[str, Any]:
        """
        Validate JWT token and return user information.
        
        Args:
            token: JWT access token
        
        Returns:
            Dict containing user information from token
        
        Raises:
            TokenExpiredError: If token has expired
            TokenInvalidError: If token is invalid
            AuthenticationError: If validation fails
        """
        try:
            # Decode token
            payload = self._decode_access_token(token)
            
            # Extract user information
            username = payload.get("sub")
            session_id = payload.get("session_id")
            
            if not username or not session_id:
                raise TokenInvalidError("Token missing required claims")
            
            # Validate session is still active
            if not session_manager.validate_session(session_id):
                raise TokenInvalidError("Session is no longer valid")
            
            # Update session activity
            session_manager.update_session_activity(session_id)
            
            return {
                "username": username,
                "session_id": session_id,
                "groups": payload.get("groups", []),
                "permissions": payload.get("permissions", [])
            }
        
        except (TokenExpiredError, TokenInvalidError):
            raise
        except Exception as e:
            logger.error(f"Token validation failed: {e}")
            raise AuthenticationError(f"Token validation failed: {e}")
    
    def refresh_token(self, token: str) -> Dict[str, Any]:
        """
        Refresh an access token.
        
        Args:
            token: Current JWT access token
        
        Returns:
            Dict containing new access token
        
        Raises:
            TokenExpiredError: If token has expired
            TokenInvalidError: If token is invalid
            AuthenticationError: If refresh fails
        """
        try:
            # Validate current token (this will raise if invalid)
            user_info = self.validate_token(token)
            
            # Create new token with same data
            token_data = {
                "sub": user_info["username"],
                "session_id": user_info["session_id"],
                "groups": user_info["groups"],
                "permissions": user_info["permissions"]
            }
            
            new_token = self._create_access_token(token_data)
            
            logger.info(f"Token refreshed for user {user_info['username']}")
            
            return {
                "access_token": new_token,
                "token_type": "bearer",
                "user_id": user_info["username"],
                "groups": user_info["groups"],
                "permissions": user_info["permissions"],
                "session_id": user_info["session_id"],
                "expires_in": self.access_token_expire_minutes * 60
            }
        
        except (TokenExpiredError, TokenInvalidError, AuthenticationError):
            raise
        except Exception as e:
            logger.error(f"Token refresh failed: {e}")
            raise AuthenticationError(f"Token refresh failed: {e}")
    
    def revoke_token(self, token: str) -> bool:
        """
        Revoke a JWT token.
        
        Args:
            token: JWT access token to revoke
        
        Returns:
            bool: True if token was revoked successfully
        """
        try:
            # Decode token to get JTI
            payload = self._decode_access_token(token)
            jti = payload.get("jti")
            
            if jti:
                self._revoked_tokens.add(jti)
                logger.info(f"Token revoked for user {payload.get('sub')}")
                return True
            
            return False
        
        except Exception as e:
            logger.error(f"Token revocation failed: {e}")
            return False
    
    def logout_user(self, token: str) -> bool:
        """
        Logout user by revoking token and invalidating session.
        
        Args:
            token: JWT access token
        
        Returns:
            bool: True if logout was successful
        """
        try:
            # Validate token to get session info
            user_info = self.validate_token(token)
            
            # Revoke token
            self.revoke_token(token)
            
            # Delete session
            session_manager.delete_session(user_info["session_id"])
            
            logger.info(f"User {user_info['username']} logged out successfully")
            return True
        
        except Exception as e:
            logger.error(f"Logout failed: {e}")
            return False
    
    def logout_all_user_sessions(self, username: str) -> int:
        """
        Logout all sessions for a user.
        
        Args:
            username: Username to logout
        
        Returns:
            int: Number of sessions logged out
        """
        try:
            # Invalidate all user sessions
            invalidated = session_manager.invalidate_user_sessions(username)
            
            # Note: In production, we should also revoke all tokens for this user
            # This would require storing token JTIs by user in Redis
            
            logger.info(f"Logged out {invalidated} sessions for user {username}")
            return invalidated
        
        except Exception as e:
            logger.error(f"Logout all sessions failed for user {username}: {e}")
            return 0
    
    def get_current_user(self, token: str) -> Optional[UserSession]:
        """
        Get current user session from token.
        
        Args:
            token: JWT access token
        
        Returns:
            UserSession: Current user session if valid, None otherwise
        """
        try:
            user_info = self.validate_token(token)
            return session_manager.get_session(user_info["session_id"])
        except Exception as e:
            logger.debug(f"Get current user failed: {e}")
            return None
    
    def has_permission(self, token: str, permission: str) -> bool:
        """
        Check if user has specific permission.
        
        Args:
            token: JWT access token
            permission: Permission to check
        
        Returns:
            bool: True if user has permission, False otherwise
        """
        try:
            user_info = self.validate_token(token)
            return permission in user_info.get("permissions", [])
        except Exception:
            return False
    
    def health_check(self) -> Dict[str, Any]:
        """
        Perform health check on authentication system.
        
        Returns:
            Dict[str, Any]: Health check results
        """
        health = {
            "authentication_manager": True,
            "jwt_algorithm": self.algorithm,
            "token_expiry_minutes": self.access_token_expire_minutes,
            "revoked_tokens_count": len(self._revoked_tokens),
            "session_manager": True,
            "errors": []
        }
        
        try:
            # Test JWT encoding/decoding
            test_data = {"test": "data", "exp": datetime.utcnow() + timedelta(minutes=1)}
            test_token = jwt.encode(test_data, self.secret_key, algorithm=self.algorithm)
            decoded = jwt.decode(test_token, self.secret_key, algorithms=[self.algorithm])
            
            if decoded.get("test") != "data":
                health["errors"].append("JWT encoding/decoding test failed")
                health["authentication_manager"] = False
        
        except Exception as e:
            health["errors"].append(f"JWT test failed: {e}")
            health["authentication_manager"] = False
        
        try:
            # Test session manager integration
            session_health = session_manager.health_check()
            if not session_health.get("session_manager", False):
                health["session_manager"] = False
                health["errors"].extend(session_health.get("errors", []))
        
        except Exception as e:
            health["session_manager"] = False
            health["errors"].append(f"Session manager health check failed: {e}")
        
        return health


# Global authentication manager instance
auth_manager = AuthenticationManager()