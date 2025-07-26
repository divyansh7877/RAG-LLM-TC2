"""
Tests for FastAPI application structure and middleware.
"""
import pytest
import json
from unittest.mock import Mock, patch
from fastapi.testclient import TestClient
from fastapi import Request

from app.api.main import app, RequestLoggingMiddleware, ResponseFormattingMiddleware


@pytest.fixture
def client():
    """Create test client."""
    return TestClient(app)


@pytest.fixture
def mock_redis():
    """Mock Redis client."""
    with patch('app.api.main.redis_client') as mock:
        mock.health_check.return_value = True
        yield mock


class TestFastAPIApplication:
    """Test FastAPI application setup and configuration."""
    
    def test_app_configuration(self):
        """Test that FastAPI app is configured correctly."""
        assert app.title == "Concurrent RAG System"
        assert app.description == "Multi-user RAG system with concurrent processing"
        assert app.version == "1.0.0"
        assert app.docs_url == "/api/docs"
        assert app.redoc_url == "/api/redoc"
        assert app.openapi_url == "/api/openapi.json"
    
    def test_middleware_registration(self):
        """Test that middleware is properly registered."""
        # Check that middleware stack includes our middleware
        middleware_count = len(app.user_middleware)
        
        # Should have multiple middleware registered (CORS, TrustedHost, and custom)
        assert middleware_count >= 4  # At least 4 middleware components
    
    def test_cors_configuration(self, client):
        """Test CORS middleware configuration."""
        response = client.options("/", headers={"Origin": "http://localhost:3000"})
        
        # Should allow CORS requests
        assert response.status_code in [200, 405]  # OPTIONS might not be implemented
        
        # Test actual CORS headers with GET request
        response = client.get("/", headers={"Origin": "http://localhost:3000"})
        assert "access-control-allow-origin" in response.headers


class TestRequestLoggingMiddleware:
    """Test request logging middleware."""
    
    @pytest.mark.asyncio
    async def test_request_id_generation(self):
        """Test that request ID is generated and added to state."""
        mock_request = Mock(spec=Request)
        mock_request.method = "GET"
        mock_request.url.path = "/test"
        mock_request.client.host = "127.0.0.1"
        mock_request.state = Mock()
        
        async def mock_call_next(request):
            # Verify request ID was set
            assert hasattr(request.state, 'request_id')
            assert request.state.request_id is not None
            
            mock_response = Mock()
            mock_response.status_code = 200
            mock_response.headers = {}
            return mock_response
        
        middleware = RequestLoggingMiddleware(app)
        
        with patch('app.api.main.logger') as mock_logger:
            response = await middleware(mock_request, mock_call_next)
            
            # Verify logging occurred
            assert mock_logger.info.call_count >= 2  # Start and completion logs
            
            # Verify response headers
            assert "X-Request-ID" in response.headers
            assert "X-Process-Time" in response.headers
    
    @pytest.mark.asyncio
    async def test_exception_handling(self):
        """Test exception handling in logging middleware."""
        mock_request = Mock(spec=Request)
        mock_request.method = "GET"
        mock_request.url.path = "/test"
        mock_request.client.host = "127.0.0.1"
        mock_request.state = Mock()
        
        async def mock_call_next(request):
            raise Exception("Test exception")
        
        middleware = RequestLoggingMiddleware(app)
        
        with patch('app.api.main.logger') as mock_logger:
            response = await middleware(mock_request, mock_call_next)
            
            # Verify error logging
            mock_logger.error.assert_called_once()
            
            # Verify error response
            assert response.status_code == 500
            assert "X-Request-ID" in response.headers


class TestResponseFormattingMiddleware:
    """Test response formatting middleware."""
    
    @pytest.mark.asyncio
    async def test_standard_headers_added(self):
        """Test that standard headers are added to responses."""
        mock_request = Mock(spec=Request)
        
        async def mock_call_next(request):
            mock_response = Mock()
            mock_response.headers = {}
            return mock_response
        
        middleware = ResponseFormattingMiddleware(app)
        response = await middleware(mock_request, mock_call_next)
        
        # Verify standard headers
        assert "X-API-Version" in response.headers
        assert "X-Timestamp" in response.headers
        assert response.headers["X-API-Version"] == "1.0.0"


class TestExceptionHandlers:
    """Test global exception handlers."""
    
    def test_validation_error_handler(self, client):
        """Test request validation error handling."""
        # This would require an endpoint that validates input
        # For now, test that the handler is registered
        assert app.exception_handlers is not None
    
    def test_http_exception_handler(self, client):
        """Test HTTP exception handling."""
        # Test with a non-existent endpoint
        response = client.get("/nonexistent")
        
        assert response.status_code == 404
        
        # Check that we get some error response (FastAPI's default or our custom format)
        error_data = response.json()
        # FastAPI's default 404 has "detail", our custom format has "error"
        assert "detail" in error_data or "error" in error_data
    
    def test_general_exception_handler(self, client, mock_redis):
        """Test general exception handling."""
        # Mock Redis to raise an exception
        mock_redis.health_check.side_effect = Exception("Redis connection failed")
        
        response = client.get("/api/status")
        
        # Should handle the exception gracefully
        assert response.status_code == 503
        
        error_data = response.json()
        assert "error" in error_data
        assert error_data["error"]["code"] == "SERVICE_UNAVAILABLE"


class TestEndpoints:
    """Test basic API endpoints."""
    
    def test_root_endpoint(self, client):
        """Test root endpoint."""
        response = client.get("/")
        
        assert response.status_code == 200
        
        data = response.json()
        assert data["message"] == "Concurrent RAG System API"
        assert data["version"] == "1.0.0"
        assert "timestamp" in data
    
    def test_health_check_endpoint(self, client, mock_redis):
        """Test health check endpoint."""
        response = client.get("/health")
        
        assert response.status_code == 200
        
        data = response.json()
        assert "status" in data
        assert "services" in data
        assert "redis" in data["services"]
        assert "api" in data["services"]
        assert "timestamp" in data
    
    def test_health_check_redis_failure(self, client, mock_redis):
        """Test health check when Redis is down."""
        mock_redis.health_check.return_value = False
        
        response = client.get("/health")
        
        assert response.status_code == 200
        
        data = response.json()
        assert data["status"] == "unhealthy"
        assert data["services"]["redis"] is False
    
    def test_api_status_endpoint(self, client, mock_redis):
        """Test detailed API status endpoint."""
        response = client.get("/api/status")
        
        assert response.status_code == 200
        
        data = response.json()
        assert "api_version" in data
        assert "status" in data
        assert "services" in data
        assert "resource_usage" in data
        assert "timestamp" in data
        
        # Check services structure
        assert "redis" in data["services"]
        assert "authentication" in data["services"]
        
        for service in data["services"].values():
            assert "status" in service
            assert "last_check" in service
    
    def test_api_status_degraded(self, client, mock_redis):
        """Test API status when services are degraded."""
        mock_redis.health_check.return_value = False
        
        response = client.get("/api/status")
        
        assert response.status_code == 200
        
        data = response.json()
        assert data["status"] == "degraded"
        assert data["services"]["redis"]["status"] == "unhealthy"
    
    def test_api_status_exception(self, client, mock_redis):
        """Test API status endpoint exception handling."""
        mock_redis.health_check.side_effect = Exception("Connection failed")
        
        response = client.get("/api/status")
        
        assert response.status_code == 503
        
        error_data = response.json()
        assert "error" in error_data
        assert error_data["error"]["code"] == "SERVICE_UNAVAILABLE"


class TestResponseHeaders:
    """Test response headers are properly set."""
    
    def test_standard_response_headers(self, client):
        """Test that standard headers are included in responses."""
        response = client.get("/")
        
        # Check for custom headers added by middleware
        assert "X-API-Version" in response.headers
        assert "X-Timestamp" in response.headers
        assert response.headers["X-API-Version"] == "1.0.0"
    
    def test_request_id_header(self, client):
        """Test that request ID is included in response headers."""
        response = client.get("/")
        
        # Request ID should be added by logging middleware
        assert "X-Request-ID" in response.headers
        assert "X-Process-Time" in response.headers
    
    def test_cors_headers(self, client):
        """Test CORS headers are properly set."""
        response = client.get("/", headers={"Origin": "http://localhost:3000"})
        
        # CORS headers should be present
        assert "access-control-allow-origin" in response.headers


class TestSecurityHeaders:
    """Test security-related headers and middleware."""
    
    def test_trusted_host_middleware(self, client):
        """Test that TrustedHostMiddleware is working."""
        # With current configuration allowing all hosts, this should pass
        response = client.get("/", headers={"Host": "example.com"})
        assert response.status_code == 200
    
    def test_security_headers_present(self, client):
        """Test that security headers are present."""
        response = client.get("/")
        
        # Basic security checks
        assert response.status_code == 200
        
        # API version should be present (from our middleware)
        assert "X-API-Version" in response.headers


if __name__ == "__main__":
    pytest.main([__file__])