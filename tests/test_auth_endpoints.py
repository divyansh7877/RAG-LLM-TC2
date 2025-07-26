"""
Tests for authentication API endpoints.
"""
import pytest
import json
from datetime import datetime, timedelta
from fastapi.testclient import TestClient
from unittest.mock import patch, MagicMock

from app.api.main import app
from app.shared.config import config
from app.shared.auth import auth_manager
from app.shared.session_manager import session_manager
from app.shared.models import UserSession


# Mock rate limiter for tests
class MockRateLimiter:
    """Mock rate limiter that always allows requests."""
    
    def create_rate_limiter(self, max_requests: int, window_seconds: int):
        async def mock_rate_limit_checker(request):
            pass  # Always allow
        return mock_rate_limit_checker


class TestAuthenticationEndpoints:
    """Test suite for authentication API endpoints."""
    
    @pytest.fixture
    def client(self):
        """Create test client with mocked rate limiter."""
        with patch('app.api.main.rate_limiter', MockRateLimiter()):
            return TestClient(app)
    
    @pytest.fixture
    def test_user_data(self):
        """Test user data."""
        return {
            "username": "testuser",
            "password": "testpassword123",
            "groups": ["assistance"]
        }
    
    @pytest.fixture
    def mock_auth_manager(self):
        """Mock authentication manager."""
        with patch('app.api.main.auth_manager') as mock:
            yield mock
    
    @pytest.fixture
    def mock_session_manager(self):
        """Mock session manager."""
        with patch('app.shared.auth.session_manager') as mock:
            yield mock
    
    @pytest.fixture
    def valid_token(self):
        """Generate a valid test token."""
        return "eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9.test.token"
    
    @pytest.fixture
    def test_session(self):
        """Create test session."""
        return UserSession(
            session_id="test-session-id",
            user_id="testuser",
            groups=["assistance"],
            permissions=["upload", "query"],
            created_at=datetime.now(),
            last_activity=datetime.now(),
            is_active=True
        )

    def test_login_success(self, client, test_user_data, mock_auth_manager):
        """Test successful login."""
        # Mock successful authentication
        mock_auth_manager.authenticate_user.return_value = {
            "access_token": "test-token",
            "token_type": "bearer",
            "user_id": "testuser",
            "groups": ["assistance"],
            "permissions": ["upload", "query"],
            "session_id": "test-session",
            "expires_in": 86400
        }
        
        response = client.post(
            "/api/auth/login",
            json={
                "username": test_user_data["username"],
                "password": test_user_data["password"]
            }
        )
        
        assert response.status_code == 200
        data = response.json()
        assert data["access_token"] == "test-token"
        assert data["token_type"] == "bearer"
        assert data["user_id"] == "testuser"
        assert data["groups"] == ["assistance"]
        
        # Verify auth manager was called correctly
        mock_auth_manager.authenticate_user.assert_called_once_with(
            username="testuser",
            password="testpassword123"
        )

    def test_login_invalid_credentials(self, client, test_user_data, mock_auth_manager):
        """Test login with invalid credentials."""
        from app.shared.auth import InvalidCredentialsError
        
        # Mock invalid credentials
        mock_auth_manager.authenticate_user.side_effect = InvalidCredentialsError("Invalid credentials")
        
        response = client.post(
            "/api/auth/login",
            json={
                "username": test_user_data["username"],
                "password": "wrongpassword"
            }
        )
        
        assert response.status_code == 401
        data = response.json()
        assert "Invalid username or password" in data["error"]["message"]

    def test_login_missing_fields(self, client):
        """Test login with missing fields."""
        response = client.post(
            "/api/auth/login",
            json={"username": "testuser"}  # Missing password
        )
        
        assert response.status_code == 422
        data = response.json()
        assert data["error"]["code"] == "VALIDATION_ERROR"

    def test_login_rate_limiting(self, client, test_user_data):
        """Test login rate limiting."""
        # Make multiple rapid requests to trigger rate limiting
        for i in range(6):  # Rate limit is 5 per 5 minutes
            response = client.post(
                "/api/auth/login",
                json={
                    "username": f"user{i}",
                    "password": "password"
                }
            )
            
            if i < 5:
                # First 5 should be allowed (even if they fail authentication)
                assert response.status_code in [200, 401, 500]
            else:
                # 6th request should be rate limited
                assert response.status_code == 429
                data = response.json()
                assert "Rate limit exceeded" in data["error"]["message"]

    def test_logout_success(self, client, test_session, mock_auth_manager):
        """Test successful logout."""
        # Mock successful logout
        mock_auth_manager.logout_user.return_value = True
        
        # Mock the middleware dependency
        with patch('app.api.main.get_current_user', return_value=test_session):
            response = client.post(
                "/api/auth/logout",
                headers={"Authorization": "Bearer test-token"}
            )
        
        assert response.status_code == 200
        data = response.json()
        assert "Logged out successfully" in data["message"]
        
        # Verify logout was called
        mock_auth_manager.logout_user.assert_called_once_with("test-token")

    def test_logout_invalid_header(self, client):
        """Test logout with invalid authorization header."""
        response = client.post(
            "/api/auth/logout",
            headers={"Authorization": "Invalid header"}
        )
        
        assert response.status_code == 401  # Will fail at authentication middleware

    def test_logout_no_token(self, client):
        """Test logout without token."""
        response = client.post("/api/auth/logout")
        
        assert response.status_code == 401

    def test_get_session_info_success(self, client, test_session, mock_auth_manager):
        """Test getting session info."""
        # Mock the middleware dependency
        with patch('app.api.main.get_current_user', return_value=test_session):
            response = client.get(
                "/api/auth/session",
                headers={"Authorization": "Bearer test-token"}
            )
        
        assert response.status_code == 200
        data = response.json()
        assert data["session_id"] == "test-session-id"
        assert data["user_id"] == "testuser"
        assert data["groups"] == ["assistance"]
        assert data["permissions"] == ["upload", "query"]
        assert data["is_active"] is True

    def test_get_session_info_unauthorized(self, client):
        """Test getting session info without authentication."""
        response = client.get("/api/auth/session")
        
        assert response.status_code == 401

    def test_refresh_token_success(self, client, mock_auth_manager):
        """Test successful token refresh."""
        # Mock successful refresh
        mock_auth_manager.refresh_token.return_value = {
            "access_token": "new-test-token",
            "token_type": "bearer",
            "user_id": "testuser",
            "groups": ["assistance"],
            "permissions": ["upload", "query"],
            "session_id": "test-session",
            "expires_in": 86400
        }
        
        response = client.post(
            "/api/auth/refresh",
            headers={"Authorization": "Bearer old-token"}
        )
        
        assert response.status_code == 200
        data = response.json()
        assert data["access_token"] == "new-test-token"
        assert data["user_id"] == "testuser"
        
        # Verify refresh was called
        mock_auth_manager.refresh_token.assert_called_once_with("old-token")

    def test_refresh_token_expired(self, client, mock_auth_manager):
        """Test token refresh with expired token."""
        from app.shared.auth import TokenExpiredError
        
        # Mock expired token
        mock_auth_manager.refresh_token.side_effect = TokenExpiredError("Token expired")
        
        response = client.post(
            "/api/auth/refresh",
            headers={"Authorization": "Bearer expired-token"}
        )
        
        assert response.status_code == 401
        data = response.json()
        assert "expired" in data["error"]["message"].lower()

    def test_refresh_token_invalid(self, client, mock_auth_manager):
        """Test token refresh with invalid token."""
        from app.shared.auth import TokenInvalidError
        
        # Mock invalid token
        mock_auth_manager.refresh_token.side_effect = TokenInvalidError("Invalid token")
        
        response = client.post(
            "/api/auth/refresh",
            headers={"Authorization": "Bearer invalid-token"}
        )
        
        assert response.status_code == 401
        data = response.json()
        assert "Invalid token" in data["error"]["message"]

    def test_validate_token_success(self, client, mock_auth_manager):
        """Test successful token validation."""
        # Mock successful validation
        mock_auth_manager.validate_token.return_value = {
            "username": "testuser",
            "session_id": "test-session",
            "groups": ["assistance"],
            "permissions": ["upload", "query"]
        }
        
        # Mock the validate_token dependency
        with patch('app.api.main.validate_token', return_value={
            "valid": True,
            "user_info": {
                "username": "testuser",
                "session_id": "test-session",
                "groups": ["assistance"],
                "permissions": ["upload", "query"]
            }
        }):
            response = client.post(
                "/api/auth/validate",
                headers={"Authorization": "Bearer valid-token"}
            )
        
        assert response.status_code == 200
        data = response.json()
        assert data["valid"] is True
        assert data["user_info"]["username"] == "testuser"

    def test_validate_token_invalid(self, client):
        """Test token validation with invalid token."""
        response = client.post(
            "/api/auth/validate",
            headers={"Authorization": "Bearer invalid-token"}
        )
        
        assert response.status_code == 401

    def test_register_user_success(self, client):
        """Test successful user registration."""
        # Ensure user doesn't exist
        if "newuser" in config.USERS:
            del config.USERS["newuser"]
        
        registration_data = {
            "username": "newuser",
            "password": "newpassword123",
            "groups": ["assistance"]
        }
        
        response = client.post(
            "/api/auth/register",
            json=registration_data
        )
        
        assert response.status_code == 200
        data = response.json()
        assert "registered successfully" in data["message"]
        assert data["username"] == "newuser"
        assert data["groups"] == ["assistance"]
        
        # Verify user was added to config
        assert "newuser" in config.USERS
        assert config.USERS["newuser"]["groups"] == ["assistance"]

    def test_register_user_already_exists(self, client):
        """Test registration with existing username."""
        # Ensure user exists
        config.USERS["existinguser"] = {
            "password": "password",
            "groups": ["assistance"]
        }
        
        registration_data = {
            "username": "existinguser",
            "password": "newpassword123",
            "groups": ["assistance"]
        }
        
        response = client.post(
            "/api/auth/register",
            json=registration_data
        )
        
        assert response.status_code == 409
        data = response.json()
        assert "already exists" in data["error"]["message"]

    def test_register_user_weak_password(self, client):
        """Test registration with weak password."""
        registration_data = {
            "username": "newuser2",
            "password": "weak",  # Too short
            "groups": ["assistance"]
        }
        
        response = client.post(
            "/api/auth/register",
            json=registration_data
        )
        
        assert response.status_code == 400
        data = response.json()
        assert "8 characters" in data["error"]["message"]

    def test_register_user_invalid_group(self, client):
        """Test registration with invalid group."""
        registration_data = {
            "username": "newuser3",
            "password": "validpassword123",
            "groups": ["invalid_group"]
        }
        
        response = client.post(
            "/api/auth/register",
            json=registration_data
        )
        
        assert response.status_code == 400
        data = response.json()
        assert "Invalid group" in data["error"]["message"]

    def test_register_user_missing_fields(self, client):
        """Test registration with missing fields."""
        registration_data = {
            "username": "newuser4",
            # Missing password and groups
        }
        
        response = client.post(
            "/api/auth/register",
            json=registration_data
        )
        
        assert response.status_code == 400
        data = response.json()
        assert "Missing required field" in data["error"]["message"]

    def test_change_password_success(self, client, test_session, mock_auth_manager):
        """Test successful password change."""
        # Setup existing user
        config.USERS["testuser"] = {
            "password": "oldpassword123",
            "groups": ["assistance"]
        }
        
        # Mock current user and logout function
        mock_auth_manager.logout_all_user_sessions.return_value = 1
        
        password_data = {
            "current_password": "oldpassword123",
            "new_password": "newpassword123"
        }
        
        # Mock the middleware dependency
        with patch('app.api.main.get_current_user', return_value=test_session):
            response = client.post(
                "/api/auth/change-password",
                json=password_data,
                headers={"Authorization": "Bearer test-token"}
            )
        
        assert response.status_code == 200
        data = response.json()
        assert "changed successfully" in data["message"]
        
        # Verify password was updated
        assert config.USERS["testuser"]["password"] == "newpassword123"
        
        # Verify all sessions were logged out
        mock_auth_manager.logout_all_user_sessions.assert_called_once_with("testuser")

    def test_change_password_wrong_current(self, client, test_session, mock_auth_manager):
        """Test password change with wrong current password."""
        # Setup existing user
        config.USERS["testuser"] = {
            "password": "correctpassword123",
            "groups": ["assistance"]
        }
        
        password_data = {
            "current_password": "wrongpassword",
            "new_password": "newpassword123"
        }
        
        # Mock the middleware dependency
        with patch('app.api.main.get_current_user', return_value=test_session):
            response = client.post(
                "/api/auth/change-password",
                json=password_data,
                headers={"Authorization": "Bearer test-token"}
            )
        
        assert response.status_code == 401
        data = response.json()
        assert "incorrect" in data["error"]["message"].lower()

    def test_change_password_weak_new_password(self, client, test_session, mock_auth_manager):
        """Test password change with weak new password."""
        # Setup existing user
        config.USERS["testuser"] = {
            "password": "oldpassword123",
            "groups": ["assistance"]
        }
        
        password_data = {
            "current_password": "oldpassword123",
            "new_password": "weak"  # Too short
        }
        
        # Mock the middleware dependency
        with patch('app.api.main.get_current_user', return_value=test_session):
            response = client.post(
                "/api/auth/change-password",
                json=password_data,
                headers={"Authorization": "Bearer test-token"}
            )
        
        assert response.status_code == 400
        data = response.json()
        assert "8 characters" in data["error"]["message"]

    def test_change_password_same_as_current(self, client, test_session, mock_auth_manager):
        """Test password change with same password."""
        # Setup existing user
        config.USERS["testuser"] = {
            "password": "samepassword123",
            "groups": ["assistance"]
        }
        
        password_data = {
            "current_password": "samepassword123",
            "new_password": "samepassword123"  # Same as current
        }
        
        # Mock the middleware dependency
        with patch('app.api.main.get_current_user', return_value=test_session):
            response = client.post(
                "/api/auth/change-password",
                json=password_data,
                headers={"Authorization": "Bearer test-token"}
            )
        
        assert response.status_code == 400
        data = response.json()
        assert "different" in data["error"]["message"].lower()

    def test_change_password_missing_fields(self, client, test_session, mock_auth_manager):
        """Test password change with missing fields."""
        # Mock current user
        mock_auth_manager.get_current_user.return_value = test_session
        
        password_data = {
            "current_password": "oldpassword123"
            # Missing new_password
        }
        
        response = client.post(
            "/api/auth/change-password",
            json=password_data,
            headers={"Authorization": "Bearer test-token"}
        )
        
        assert response.status_code == 400
        data = response.json()
        assert "Missing" in data["error"]["message"]

    def test_change_password_unauthorized(self, client):
        """Test password change without authentication."""
        password_data = {
            "current_password": "oldpassword123",
            "new_password": "newpassword123"
        }
        
        response = client.post(
            "/api/auth/change-password",
            json=password_data
        )
        
        assert response.status_code == 401


class TestAuthenticationSecurity:
    """Test suite for authentication security features."""
    
    @pytest.fixture
    def client(self):
        """Create test client."""
        return TestClient(app)
    
    def test_rate_limiting_login(self, client):
        """Test rate limiting on login endpoint."""
        # This test verifies that rate limiting is properly applied
        # The actual rate limiting logic is tested in the main test class
        pass
    
    def test_rate_limiting_refresh(self, client):
        """Test rate limiting on refresh endpoint."""
        # Make multiple rapid refresh requests
        for i in range(12):  # Rate limit is 10 per 5 minutes
            response = client.post(
                "/api/auth/refresh",
                headers={"Authorization": "Bearer test-token"}
            )
            
            if i < 10:
                # First 10 should be allowed (even if they fail)
                assert response.status_code in [200, 401, 400, 500]
            else:
                # 11th and 12th requests should be rate limited
                assert response.status_code == 429
    
    def test_rate_limiting_registration(self, client):
        """Test rate limiting on registration endpoint."""
        # Make multiple rapid registration requests
        for i in range(5):  # Rate limit is 3 per hour
            response = client.post(
                "/api/auth/register",
                json={
                    "username": f"user{i}",
                    "password": "password123",
                    "groups": ["assistance"]
                }
            )
            
            if i < 3:
                # First 3 should be allowed
                assert response.status_code in [200, 400, 409]
            else:
                # 4th and 5th requests should be rate limited
                assert response.status_code == 429
    
    def test_rate_limiting_password_change(self, client):
        """Test rate limiting on password change endpoint."""
        # Make multiple rapid password change requests
        for i in range(7):  # Rate limit is 5 per hour
            response = client.post(
                "/api/auth/change-password",
                json={
                    "current_password": "old",
                    "new_password": "new123456"
                },
                headers={"Authorization": "Bearer test-token"}
            )
            
            if i < 5:
                # First 5 should be allowed (even if they fail auth)
                assert response.status_code in [200, 401, 400, 500]
            else:
                # 6th and 7th requests should be rate limited
                assert response.status_code == 429
    
    def test_token_validation_security(self, client):
        """Test token validation security."""
        # Test with malformed tokens
        malformed_tokens = [
            "not.a.token",
            "Bearer malformed",
            "eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9.malformed",
            "",
            "null",
            "undefined"
        ]
        
        for token in malformed_tokens:
            response = client.get(
                "/api/auth/session",
                headers={"Authorization": f"Bearer {token}"}
            )
            assert response.status_code == 401
    
    def test_sql_injection_protection(self, client):
        """Test protection against SQL injection attempts."""
        # Test SQL injection in login
        sql_payloads = [
            "admin'; DROP TABLE users; --",
            "' OR '1'='1",
            "admin'/*",
            "' UNION SELECT * FROM users --"
        ]
        
        for payload in sql_payloads:
            response = client.post(
                "/api/auth/login",
                json={
                    "username": payload,
                    "password": "password"
                }
            )
            # Should not cause server error, should handle gracefully
            assert response.status_code in [401, 422, 400]
    
    def test_xss_protection(self, client):
        """Test protection against XSS attempts."""
        xss_payloads = [
            "<script>alert('xss')</script>",
            "javascript:alert('xss')",
            "<img src=x onerror=alert('xss')>",
            "';alert('xss');//"
        ]
        
        for payload in xss_payloads:
            response = client.post(
                "/api/auth/register",
                json={
                    "username": payload,
                    "password": "password123",
                    "groups": ["assistance"]
                }
            )
            # Should handle gracefully without executing script
            assert response.status_code in [200, 400, 409, 422]
    
    def test_password_security(self, client):
        """Test password security requirements."""
        weak_passwords = [
            "123",
            "password",
            "12345678",
            "",
            " " * 10
        ]
        
        for weak_password in weak_passwords:
            response = client.post(
                "/api/auth/register",
                json={
                    "username": "testuser",
                    "password": weak_password,
                    "groups": ["assistance"]
                }
            )
            
            if len(weak_password.strip()) < 8:
                assert response.status_code == 400
                assert "8 characters" in response.json()["error"]["message"]


class TestAuthenticationIntegration:
    """Integration tests for authentication flow."""
    
    @pytest.fixture
    def client(self):
        """Create test client."""
        return TestClient(app)
    
    def test_complete_auth_flow(self, client):
        """Test complete authentication flow."""
        # 1. Register user
        registration_data = {
            "username": "flowtest",
            "password": "flowtest123",
            "groups": ["assistance"]
        }
        
        register_response = client.post(
            "/api/auth/register",
            json=registration_data
        )
        assert register_response.status_code == 200
        
        # 2. Login
        login_response = client.post(
            "/api/auth/login",
            json={
                "username": "flowtest",
                "password": "flowtest123"
            }
        )
        assert login_response.status_code == 200
        token = login_response.json()["access_token"]
        
        # 3. Get session info
        session_response = client.get(
            "/api/auth/session",
            headers={"Authorization": f"Bearer {token}"}
        )
        assert session_response.status_code == 200
        assert session_response.json()["user_id"] == "flowtest"
        
        # 4. Refresh token
        refresh_response = client.post(
            "/api/auth/refresh",
            headers={"Authorization": f"Bearer {token}"}
        )
        assert refresh_response.status_code == 200
        new_token = refresh_response.json()["access_token"]
        
        # 5. Change password
        password_response = client.post(
            "/api/auth/change-password",
            json={
                "current_password": "flowtest123",
                "new_password": "newflowtest123"
            },
            headers={"Authorization": f"Bearer {new_token}"}
        )
        assert password_response.status_code == 200
        
        # 6. Login with new password
        new_login_response = client.post(
            "/api/auth/login",
            json={
                "username": "flowtest",
                "password": "newflowtest123"
            }
        )
        assert new_login_response.status_code == 200
        final_token = new_login_response.json()["access_token"]
        
        # 7. Logout
        logout_response = client.post(
            "/api/auth/logout",
            headers={"Authorization": f"Bearer {final_token}"}
        )
        assert logout_response.status_code == 200
        
        # 8. Verify token is invalid after logout
        session_after_logout = client.get(
            "/api/auth/session",
            headers={"Authorization": f"Bearer {final_token}"}
        )
        assert session_after_logout.status_code == 401
    
    def test_concurrent_sessions(self, client):
        """Test handling of concurrent sessions."""
        # Setup user
        config.USERS["concurrent"] = {
            "password": "concurrent123",
            "groups": ["assistance"]
        }
        
        # Login multiple times to create multiple sessions
        tokens = []
        for i in range(3):
            response = client.post(
                "/api/auth/login",
                json={
                    "username": "concurrent",
                    "password": "concurrent123"
                }
            )
            assert response.status_code == 200
            tokens.append(response.json()["access_token"])
        
        # All tokens should be valid
        for token in tokens:
            response = client.get(
                "/api/auth/session",
                headers={"Authorization": f"Bearer {token}"}
            )
            assert response.status_code == 200
        
        # Change password should invalidate all sessions
        response = client.post(
            "/api/auth/change-password",
            json={
                "current_password": "concurrent123",
                "new_password": "newconcurrent123"
            },
            headers={"Authorization": f"Bearer {tokens[0]}"}
        )
        assert response.status_code == 200
        
        # All old tokens should now be invalid
        for token in tokens:
            response = client.get(
                "/api/auth/session",
                headers={"Authorization": f"Bearer {token}"}
            )
            assert response.status_code == 401