"""
Unit tests for JWT-based authentication system.
"""
import pytest
import jwt
import time
from datetime import datetime, timedelta
from unittest.mock import Mock, patch, MagicMock

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.shared.auth import (
    AuthenticationManager, AuthenticationError, InvalidCredentialsError,
    TokenExpiredError, TokenInvalidError, auth_manager
)
from app.shared.models import UserSession
from app.shared.session_manager import SessionError


class TestAuthenticationManager:
    """Test suite for AuthenticationManager."""
    
    @pytest.fixture
    def auth_manager_instance(self):
        """Create a fresh AuthenticationManager instance for testing."""
        return AuthenticationManager()
    
    @pytest.fixture
    def mock_session_manager(self):
        """Mock session manager for testing."""
        with patch('app.shared.auth.session_manager') as mock:
            # Configure mock to return successful operations
            mock_session = UserSession(
                session_id="test-session-id",
                user_id="test_user",
                groups=["test_group"],
                permissions=["upload", "query"]
            )
            mock.create_session.return_value = mock_session
            mock.validate_session.return_value = True
            mock.update_session_activity.return_value = True
            mock.get_session.return_value = mock_session
            mock.delete_session.return_value = True
            mock.invalidate_user_sessions.return_value = 1
            yield mock
    
    @pytest.fixture
    def mock_config(self):
        """Mock configuration for testing."""
        with patch('app.shared.auth.config') as mock:
            mock.SECRET_KEY = "test-secret-key-for-testing"
            mock.ACCESS_TOKEN_EXPIRE_MINUTES = 30
            mock.USERS = {
                "test_user": {
                    "password": "test_password",
                    "groups": ["test_group"]
                },
                "admin_user": {
                    "password": "admin_password",
                    "groups": ["admin"]
                }
            }
            yield mock
    
    def test_authentication_manager_initialization(self, auth_manager_instance):
        """Test AuthenticationManager initialization."""
        assert auth_manager_instance is not None
        assert auth_manager_instance.algorithm == "HS256"
        assert hasattr(auth_manager_instance, '_revoked_tokens')
        assert isinstance(auth_manager_instance._revoked_tokens, set)
    
    def test_hash_and_verify_password(self, auth_manager_instance):
        """Test password hashing and verification."""
        password = "test_password_123"
        
        # Hash password
        hashed = auth_manager_instance._hash_password(password)
        
        # Verify correct password
        assert auth_manager_instance._verify_password(password, hashed) is True
        
        # Verify incorrect password
        assert auth_manager_instance._verify_password("wrong_password", hashed) is False
        
        # Verify hashes are different for same password
        hashed2 = auth_manager_instance._hash_password(password)
        assert hashed != hashed2  # Salt makes them different
        assert auth_manager_instance._verify_password(password, hashed2) is True
    
    def test_get_user_data_valid_user(self, auth_manager_instance, mock_config):
        """Test getting user data for valid user."""
        user_data = auth_manager_instance._get_user_data("test_user")
        
        assert user_data is not None
        assert user_data["username"] == "test_user"
        assert user_data["groups"] == ["test_group"]
        assert "password_hash" in user_data
        assert "permissions" in user_data
    
    def test_get_user_data_invalid_user(self, auth_manager_instance, mock_config):
        """Test getting user data for invalid user."""
        user_data = auth_manager_instance._get_user_data("nonexistent_user")
        assert user_data is None
    
    def test_get_user_permissions(self, auth_manager_instance):
        """Test user permission assignment based on groups."""
        # Basic user permissions
        permissions = auth_manager_instance._get_user_permissions(["common_rules"])
        assert "query" in permissions
        assert "upload" not in permissions
        
        # Assistance group permissions
        permissions = auth_manager_instance._get_user_permissions(["assistance"])
        assert "query" in permissions
        assert "upload" in permissions
        assert "delete" in permissions
        
        # Admin permissions
        permissions = auth_manager_instance._get_user_permissions(["admin"])
        assert "query" in permissions
        assert "upload" in permissions
        assert "delete" in permissions
        assert "admin" in permissions
    
    def test_create_and_decode_access_token(self, auth_manager_instance):
        """Test JWT token creation and decoding."""
        test_data = {
            "sub": "test_user",
            "session_id": "test-session",
            "groups": ["test_group"],
            "permissions": ["upload", "query"]
        }
        
        # Create token
        token = auth_manager_instance._create_access_token(test_data)
        assert isinstance(token, str)
        assert len(token) > 0
        
        # Decode token
        decoded = auth_manager_instance._decode_access_token(token)
        assert decoded["sub"] == "test_user"
        assert decoded["session_id"] == "test-session"
        assert decoded["groups"] == ["test_group"]
        assert decoded["permissions"] == ["upload", "query"]
        assert "exp" in decoded
        assert "iat" in decoded
        assert "jti" in decoded
    
    def test_decode_expired_token(self, auth_manager_instance):
        """Test decoding expired token."""
        test_data = {"sub": "test_user"}
        
        # Create token with very short expiration
        token = auth_manager_instance._create_access_token(
            test_data, 
            expires_delta=timedelta(seconds=-1)  # Already expired
        )
        
        # Should raise TokenExpiredError
        with pytest.raises(TokenExpiredError):
            auth_manager_instance._decode_access_token(token)
    
    def test_decode_invalid_token(self, auth_manager_instance):
        """Test decoding invalid token."""
        invalid_token = "invalid.jwt.token"
        
        with pytest.raises(TokenInvalidError):
            auth_manager_instance._decode_access_token(invalid_token)
    
    def test_authenticate_user_success(self, auth_manager_instance, mock_config, mock_session_manager):
        """Test successful user authentication."""
        result = auth_manager_instance.authenticate_user("test_user", "test_password")
        
        assert "access_token" in result
        assert result["token_type"] == "bearer"
        assert result["user_id"] == "test_user"
        assert result["groups"] == ["test_group"]
        assert "permissions" in result
        assert "session_id" in result
        assert "expires_in" in result
        
        # Verify session was created
        mock_session_manager.create_session.assert_called_once()
    
    def test_authenticate_user_invalid_username(self, auth_manager_instance, mock_config, mock_session_manager):
        """Test authentication with invalid username."""
        with pytest.raises(InvalidCredentialsError):
            auth_manager_instance.authenticate_user("nonexistent_user", "password")
    
    def test_authenticate_user_invalid_password(self, auth_manager_instance, mock_config, mock_session_manager):
        """Test authentication with invalid password."""
        with pytest.raises(InvalidCredentialsError):
            auth_manager_instance.authenticate_user("test_user", "wrong_password")
    
    def test_authenticate_user_session_creation_failure(self, auth_manager_instance, mock_config, mock_session_manager):
        """Test authentication when session creation fails."""
        mock_session_manager.create_session.side_effect = SessionError("Session creation failed")
        
        with pytest.raises(SessionError):
            auth_manager_instance.authenticate_user("test_user", "test_password")
    
    def test_validate_token_success(self, auth_manager_instance, mock_config, mock_session_manager):
        """Test successful token validation."""
        # First authenticate to get a token
        auth_result = auth_manager_instance.authenticate_user("test_user", "test_password")
        token = auth_result["access_token"]
        
        # Validate the token
        user_info = auth_manager_instance.validate_token(token)
        
        assert user_info["username"] == "test_user"
        assert "session_id" in user_info
        assert user_info["groups"] == ["test_group"]
        assert "permissions" in user_info
        
        # Verify session validation was called
        mock_session_manager.validate_session.assert_called()
        mock_session_manager.update_session_activity.assert_called()
    
    def test_validate_token_invalid_session(self, auth_manager_instance, mock_config, mock_session_manager):
        """Test token validation with invalid session."""
        # First authenticate to get a token
        auth_result = auth_manager_instance.authenticate_user("test_user", "test_password")
        token = auth_result["access_token"]
        
        # Mock session as invalid
        mock_session_manager.validate_session.return_value = False
        
        with pytest.raises(TokenInvalidError, match="Session is no longer valid"):
            auth_manager_instance.validate_token(token)
    
    def test_validate_token_missing_claims(self, auth_manager_instance):
        """Test token validation with missing required claims."""
        # Create token with missing claims
        incomplete_data = {"sub": "test_user"}  # Missing session_id
        token = auth_manager_instance._create_access_token(incomplete_data)
        
        with pytest.raises(TokenInvalidError, match="Token missing required claims"):
            auth_manager_instance.validate_token(token)
    
    def test_refresh_token_success(self, auth_manager_instance, mock_config, mock_session_manager):
        """Test successful token refresh."""
        # First authenticate to get a token
        auth_result = auth_manager_instance.authenticate_user("test_user", "test_password")
        original_token = auth_result["access_token"]
        
        # Refresh the token
        refresh_result = auth_manager_instance.refresh_token(original_token)
        
        assert "access_token" in refresh_result
        assert refresh_result["token_type"] == "bearer"
        assert refresh_result["user_id"] == "test_user"
        assert refresh_result["access_token"] != original_token  # Should be different
    
    def test_revoke_token_success(self, auth_manager_instance, mock_config, mock_session_manager):
        """Test successful token revocation."""
        # First authenticate to get a token
        auth_result = auth_manager_instance.authenticate_user("test_user", "test_password")
        token = auth_result["access_token"]
        
        # Revoke the token
        success = auth_manager_instance.revoke_token(token)
        assert success is True
        
        # Token should now be invalid
        with pytest.raises(TokenInvalidError, match="Token has been revoked"):
            auth_manager_instance.validate_token(token)
    
    def test_logout_user_success(self, auth_manager_instance, mock_config, mock_session_manager):
        """Test successful user logout."""
        # First authenticate to get a token
        auth_result = auth_manager_instance.authenticate_user("test_user", "test_password")
        token = auth_result["access_token"]
        
        # Logout user
        success = auth_manager_instance.logout_user(token)
        assert success is True
        
        # Verify session was deleted
        mock_session_manager.delete_session.assert_called()
    
    def test_logout_all_user_sessions(self, auth_manager_instance, mock_session_manager):
        """Test logging out all user sessions."""
        mock_session_manager.invalidate_user_sessions.return_value = 3
        
        invalidated = auth_manager_instance.logout_all_user_sessions("test_user")
        assert invalidated == 3
        
        mock_session_manager.invalidate_user_sessions.assert_called_once_with("test_user")
    
    def test_get_current_user_success(self, auth_manager_instance, mock_config, mock_session_manager):
        """Test getting current user from token."""
        # First authenticate to get a token
        auth_result = auth_manager_instance.authenticate_user("test_user", "test_password")
        token = auth_result["access_token"]
        
        # Get current user
        user_session = auth_manager_instance.get_current_user(token)
        
        assert user_session is not None
        assert user_session.user_id == "test_user"
        
        mock_session_manager.get_session.assert_called()
    
    def test_get_current_user_invalid_token(self, auth_manager_instance):
        """Test getting current user with invalid token."""
        user_session = auth_manager_instance.get_current_user("invalid_token")
        assert user_session is None
    
    def test_has_permission_success(self, auth_manager_instance, mock_config, mock_session_manager):
        """Test permission checking with valid permission."""
        # First authenticate to get a token
        auth_result = auth_manager_instance.authenticate_user("test_user", "test_password")
        token = auth_result["access_token"]
        
        # Check permission that user has
        has_perm = auth_manager_instance.has_permission(token, "query")
        assert has_perm is True
        
        # Check permission that user doesn't have
        has_perm = auth_manager_instance.has_permission(token, "admin")
        assert has_perm is False
    
    def test_has_permission_invalid_token(self, auth_manager_instance):
        """Test permission checking with invalid token."""
        has_perm = auth_manager_instance.has_permission("invalid_token", "query")
        assert has_perm is False
    
    def test_health_check_success(self, auth_manager_instance, mock_session_manager):
        """Test health check with healthy system."""
        mock_session_manager.health_check.return_value = {
            "session_manager": True,
            "errors": []
        }
        
        health = auth_manager_instance.health_check()
        
        assert health["authentication_manager"] is True
        assert health["session_manager"] is True
        assert health["jwt_algorithm"] == "HS256"
        assert len(health["errors"]) == 0
    
    def test_health_check_jwt_failure(self, auth_manager_instance, mock_session_manager):
        """Test health check with JWT failure."""
        # Mock JWT to fail
        with patch('app.shared.auth.jwt') as mock_jwt:
            mock_jwt.encode.side_effect = Exception("JWT encoding failed")
            
            mock_session_manager.health_check.return_value = {
                "session_manager": True,
                "errors": []
            }
            
            health = auth_manager_instance.health_check()
            
            assert health["authentication_manager"] is False
            assert "JWT test failed" in str(health["errors"])
    
    def test_health_check_session_manager_failure(self, auth_manager_instance, mock_session_manager):
        """Test health check with session manager failure."""
        mock_session_manager.health_check.return_value = {
            "session_manager": False,
            "errors": ["Session manager error"]
        }
        
        health = auth_manager_instance.health_check()
        
        assert health["session_manager"] is False
        assert "Session manager error" in health["errors"]


class TestAuthenticationIntegration:
    """Integration tests for authentication system."""
    
    @pytest.fixture
    def auth_manager_instance(self):
        """Create a fresh AuthenticationManager instance for testing."""
        return AuthenticationManager()
    
    @pytest.fixture
    def mock_config(self):
        """Mock configuration for testing."""
        with patch('app.shared.auth.config') as mock:
            mock.SECRET_KEY = "test-secret-key-for-testing"
            mock.ACCESS_TOKEN_EXPIRE_MINUTES = 30
            mock.USERS = {
                "user1": {"password": "password1", "groups": ["assistance"]},
                "user2": {"password": "password2", "groups": ["common_rules"]},
                "admin": {"password": "admin_pass", "groups": ["admin"]}
            }
            yield mock
    
    def test_full_authentication_flow(self, auth_manager_instance, mock_config):
        """Test complete authentication flow."""
        with patch('app.shared.auth.session_manager') as mock_session_manager:
            # Configure mock session manager
            mock_session = UserSession(
                session_id="test-session",
                user_id="user1",
                groups=["assistance"],
                permissions=["upload", "query", "delete"]
            )
            mock_session_manager.create_session.return_value = mock_session
            mock_session_manager.validate_session.return_value = True
            mock_session_manager.update_session_activity.return_value = True
            mock_session_manager.get_session.return_value = mock_session
            mock_session_manager.delete_session.return_value = True
            
            # 1. Authenticate user
            auth_result = auth_manager_instance.authenticate_user("user1", "password1")
            token = auth_result["access_token"]
            
            # 2. Validate token
            user_info = auth_manager_instance.validate_token(token)
            assert user_info["username"] == "user1"
            
            # 3. Check permissions
            assert auth_manager_instance.has_permission(token, "upload") is True
            assert auth_manager_instance.has_permission(token, "admin") is False
            
            # 4. Get current user
            current_user = auth_manager_instance.get_current_user(token)
            assert current_user.user_id == "user1"
            
            # 5. Refresh token
            refresh_result = auth_manager_instance.refresh_token(token)
            new_token = refresh_result["access_token"]
            assert new_token != token
            
            # 6. Logout user
            success = auth_manager_instance.logout_user(new_token)
            assert success is True
    
    def test_concurrent_authentication(self, auth_manager_instance, mock_config):
        """Test concurrent authentication operations."""
        import threading
        import time
        
        results = []
        errors = []
        
        with patch('app.shared.auth.session_manager') as mock_session_manager:
            # Configure mock to return different sessions for different users
            def create_session_side_effect(user_id, groups, permissions):
                return UserSession(
                    session_id=f"session-{user_id}",
                    user_id=user_id,
                    groups=groups,
                    permissions=permissions
                )
            
            mock_session_manager.create_session.side_effect = create_session_side_effect
            mock_session_manager.validate_session.return_value = True
            mock_session_manager.update_session_activity.return_value = True
            
            def auth_worker(user_id, password):
                try:
                    # Authenticate
                    auth_result = auth_manager_instance.authenticate_user(user_id, password)
                    token = auth_result["access_token"]
                    
                    # Validate token
                    user_info = auth_manager_instance.validate_token(token)
                    
                    results.append({
                        "user_id": user_id,
                        "token": token,
                        "user_info": user_info
                    })
                except Exception as e:
                    errors.append(f"User {user_id}: {e}")
            
            # Create threads for concurrent authentication
            threads = []
            users = [("user1", "password1"), ("user2", "password2"), ("admin", "admin_pass")]
            
            for user_id, password in users:
                thread = threading.Thread(target=auth_worker, args=(user_id, password))
                threads.append(thread)
                thread.start()
            
            # Wait for all threads to complete
            for thread in threads:
                thread.join()
            
            # Verify results
            assert len(errors) == 0, f"Errors occurred: {errors}"
            assert len(results) == 3
            
            # Verify each user got their own token and info
            user_ids = [r["user_info"]["username"] for r in results]
            assert set(user_ids) == {"user1", "user2", "admin"}
            
            # Verify tokens are unique
            tokens = [r["token"] for r in results]
            assert len(set(tokens)) == 3
    
    def test_token_expiration_and_refresh_cycle(self, auth_manager_instance, mock_config):
        """Test token expiration and refresh cycle."""
        with patch('app.shared.auth.session_manager') as mock_session_manager:
            mock_session = UserSession(
                session_id="test-session",
                user_id="user1",
                groups=["assistance"],
                permissions=["upload", "query"]
            )
            mock_session_manager.create_session.return_value = mock_session
            mock_session_manager.validate_session.return_value = True
            mock_session_manager.update_session_activity.return_value = True
            mock_session_manager.get_session.return_value = mock_session
            
            # Test with manually created expired token
            expired_token_data = {
                "sub": "user1",
                "session_id": "test-session",
                "groups": ["assistance"],
                "permissions": ["upload", "query"],
                "exp": datetime.utcnow() - timedelta(seconds=1),  # Already expired
                "iat": datetime.utcnow() - timedelta(seconds=2),
                "jti": "test-jti"
            }
            
            expired_token = jwt.encode(
                expired_token_data, 
                auth_manager_instance.secret_key, 
                algorithm=auth_manager_instance.algorithm
            )
            
            # Expired token should raise TokenExpiredError
            with pytest.raises(TokenExpiredError):
                auth_manager_instance.validate_token(expired_token)
            
            # Test normal authentication flow
            auth_result = auth_manager_instance.authenticate_user("user1", "password1")
            valid_token = auth_result["access_token"]
            
            # Valid token should work
            user_info = auth_manager_instance.validate_token(valid_token)
            assert user_info["username"] == "user1"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])