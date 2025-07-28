"""
Tests for centralized error handling and logging system.
"""
import pytest
import time
import threading
from unittest.mock import Mock, patch, MagicMock
from datetime import datetime

from app.shared.error_handling import (
    ErrorHandler, ErrorCategory, ErrorSeverity, ErrorContext,
    CircuitBreaker, CircuitBreakerConfig, CircuitBreakerState,
    StructuredLogger, with_error_handling, set_log_context, 
    clear_log_context, get_log_context, error_handler
)
from app.shared.auth import AuthenticationError, InvalidCredentialsError


class TestErrorHandler:
    """Test cases for ErrorHandler class."""
    
    def setup_method(self):
        """Setup test environment."""
        self.error_handler = ErrorHandler()
    
    def test_categorize_error_authentication(self):
        """Test error categorization for authentication errors."""
        auth_error = AuthenticationError("Invalid credentials")
        category = self.error_handler.categorize_error(auth_error)
        assert category == ErrorCategory.AUTHENTICATION
    
    def test_categorize_error_validation(self):
        """Test error categorization for validation errors."""
        validation_error = ValueError("Invalid input")
        category = self.error_handler.categorize_error(validation_error)
        assert category == ErrorCategory.VALIDATION
    
    def test_categorize_error_resource(self):
        """Test error categorization for resource errors."""
        memory_error = MemoryError("Out of memory")
        category = self.error_handler.categorize_error(memory_error)
        assert category == ErrorCategory.RESOURCE
    
    def test_categorize_error_unknown(self):
        """Test error categorization for unknown errors."""
        unknown_error = RuntimeError("Unknown error")
        category = self.error_handler.categorize_error(unknown_error)
        assert category == ErrorCategory.UNKNOWN
    
    def test_get_severity_critical_keywords(self):
        """Test severity determination for critical keywords."""
        critical_error = Exception("Critical system failure")
        category = ErrorCategory.SYSTEM
        severity = self.error_handler.get_severity(category, critical_error)
        assert severity == ErrorSeverity.CRITICAL
    
    def test_get_severity_security_keywords(self):
        """Test severity determination for security keywords."""
        security_error = Exception("Security breach detected")
        category = ErrorCategory.SYSTEM
        severity = self.error_handler.get_severity(category, security_error)
        assert severity == ErrorSeverity.CRITICAL
    
    def test_get_severity_base_mapping(self):
        """Test severity determination using base mapping."""
        auth_error = AuthenticationError("Invalid token")
        category = ErrorCategory.AUTHENTICATION
        severity = self.error_handler.get_severity(category, auth_error)
        assert severity == ErrorSeverity.MEDIUM
    
    @patch('app.shared.error_handling.redis_client')
    def test_handle_error_creates_context(self, mock_redis):
        """Test that handle_error creates proper error context."""
        # Setup
        mock_redis.set_json.return_value = True
        mock_redis.get_json.return_value = None
        
        # Set thread context
        set_log_context(user_id="test_user", session_id="test_session")
        
        # Test
        test_error = ValueError("Test error")
        context = self.error_handler.handle_error(test_error)
        
        # Assertions
        assert isinstance(context, ErrorContext)
        assert context.category == ErrorCategory.VALIDATION
        assert context.severity == ErrorSeverity.LOW
        assert context.message == "Test error"
        assert context.exception_type == "ValueError"
        assert context.user_id == "test_user"
        assert context.session_id == "test_session"
        assert context.error_id is not None
        
        # Cleanup
        clear_log_context()
    
    @patch('app.shared.error_handling.redis_client')
    def test_handle_error_stores_in_redis(self, mock_redis):
        """Test that handle_error stores error in Redis."""
        # Setup
        mock_redis.set_json.return_value = True
        mock_redis.get_json.return_value = None
        
        # Test
        test_error = ValueError("Test error")
        context = self.error_handler.handle_error(test_error)
        
        # Verify Redis calls
        assert mock_redis.set_json.call_count >= 1
        
        # Check that error was stored
        error_call = None
        for call in mock_redis.set_json.call_args_list:
            if call[0][0].startswith("error:"):
                error_call = call
                break
        
        assert error_call is not None
        assert error_call[0][1]["error_id"] == context.error_id
    
    def test_get_error_statistics_structure(self):
        """Test error statistics structure."""
        stats = self.error_handler.get_error_statistics(days=7)
        
        assert "period_days" in stats
        assert "daily_stats" in stats
        assert "summary" in stats
        assert "total_errors" in stats["summary"]
        assert "by_category" in stats["summary"]
        assert "by_severity" in stats["summary"]
    
    def test_get_recent_errors_limit(self):
        """Test recent errors limit functionality."""
        # Add some errors to the handler
        for i in range(10):
            error = ValueError(f"Test error {i}")
            self.error_handler.handle_error(error)
        
        # Get recent errors with limit
        recent = self.error_handler.get_recent_errors(limit=5)
        
        assert len(recent) <= 5
        assert all(isinstance(error, dict) for error in recent)


class TestCircuitBreaker:
    """Test cases for CircuitBreaker class."""
    
    def test_circuit_breaker_closed_state(self):
        """Test circuit breaker in closed state."""
        config = CircuitBreakerConfig(failure_threshold=3, recovery_timeout=60)
        breaker = CircuitBreaker(config)
        
        assert breaker.state == CircuitBreakerState.CLOSED
        
        # Successful call should keep it closed
        def success_func():
            return "success"
        
        result = breaker.call(success_func)
        assert result == "success"
        assert breaker.state == CircuitBreakerState.CLOSED
    
    def test_circuit_breaker_opens_on_failures(self):
        """Test circuit breaker opens after threshold failures."""
        config = CircuitBreakerConfig(failure_threshold=2, recovery_timeout=60)
        breaker = CircuitBreaker(config)
        
        def failing_func():
            raise Exception("Test failure")
        
        # First failure
        with pytest.raises(Exception):
            breaker.call(failing_func)
        assert breaker.state == CircuitBreakerState.CLOSED
        
        # Second failure should open the circuit
        with pytest.raises(Exception):
            breaker.call(failing_func)
        assert breaker.state == CircuitBreakerState.OPEN
    
    def test_circuit_breaker_blocks_when_open(self):
        """Test circuit breaker blocks calls when open."""
        config = CircuitBreakerConfig(failure_threshold=1, recovery_timeout=60)
        breaker = CircuitBreaker(config)
        
        # Cause failure to open circuit
        def failing_func():
            raise Exception("Test failure")
        
        with pytest.raises(Exception):
            breaker.call(failing_func)
        
        assert breaker.state == CircuitBreakerState.OPEN
        
        # Should block subsequent calls
        def success_func():
            return "success"
        
        with pytest.raises(Exception, match="Circuit breaker.*is OPEN"):
            breaker.call(success_func)
    
    def test_circuit_breaker_half_open_recovery(self):
        """Test circuit breaker recovery through half-open state."""
        config = CircuitBreakerConfig(failure_threshold=1, recovery_timeout=0.1)
        breaker = CircuitBreaker(config)
        
        # Open the circuit
        def failing_func():
            raise Exception("Test failure")
        
        with pytest.raises(Exception):
            breaker.call(failing_func)
        
        assert breaker.state == CircuitBreakerState.OPEN
        
        # Wait for recovery timeout
        time.sleep(0.2)
        
        # Successful call should close the circuit
        def success_func():
            return "success"
        
        result = breaker.call(success_func)
        assert result == "success"
        assert breaker.state == CircuitBreakerState.CLOSED


class TestStructuredLogger:
    """Test cases for StructuredLogger class."""
    
    def setup_method(self):
        """Setup test environment."""
        self.logger = StructuredLogger("test_logger")
    
    def test_logger_initialization(self):
        """Test logger initialization."""
        assert self.logger.logger.name == "test_logger"
        assert len(self.logger.logger.handlers) > 0
    
    @patch('app.shared.error_handling.threading.current_thread')
    def test_get_context_with_thread_context(self, mock_thread):
        """Test context extraction from thread-local storage."""
        # Setup mock thread with context
        mock_thread.return_value.log_context = {
            'request_id': 'test_request',
            'user_id': 'test_user',
            'session_id': 'test_session'
        }
        
        context = self.logger._get_context()
        
        assert context['request_id'] == 'test_request'
        assert context['user_id'] == 'test_user'
        assert context['session_id'] == 'test_session'
    
    @patch('app.shared.error_handling.threading.current_thread')
    def test_get_context_without_thread_context(self, mock_thread):
        """Test context extraction without thread-local storage."""
        # Setup mock thread without context
        mock_thread.return_value.log_context = {}
        
        context = self.logger._get_context()
        
        assert context['request_id'] == 'N/A'
        assert context['user_id'] == 'N/A'
        assert context['session_id'] == 'N/A'


class TestLogContext:
    """Test cases for log context management."""
    
    def test_set_log_context(self):
        """Test setting log context."""
        set_log_context(user_id="test_user", request_id="test_request")
        
        context = get_log_context()
        assert context['user_id'] == "test_user"
        assert context['request_id'] == "test_request"
        
        clear_log_context()
    
    def test_clear_log_context(self):
        """Test clearing log context."""
        set_log_context(user_id="test_user")
        assert get_log_context()['user_id'] == "test_user"
        
        clear_log_context()
        context = get_log_context()
        assert len(context) == 0
    
    def test_get_log_context_empty(self):
        """Test getting empty log context."""
        clear_log_context()
        context = get_log_context()
        assert isinstance(context, dict)
        assert len(context) == 0


class TestWithErrorHandlingDecorator:
    """Test cases for with_error_handling decorator."""
    
    @patch('app.shared.error_handling.error_handler')
    def test_decorator_handles_exceptions(self, mock_error_handler):
        """Test that decorator handles exceptions properly."""
        mock_error_handler.handle_error.return_value = Mock()
        
        @with_error_handling()
        def failing_function():
            raise ValueError("Test error")
        
        with pytest.raises(ValueError):
            failing_function()
        
        # Verify error handler was called
        mock_error_handler.handle_error.assert_called_once()
        call_args = mock_error_handler.handle_error.call_args
        assert isinstance(call_args[0][0], ValueError)
        assert call_args[0][0].args[0] == "Test error"
    
    @patch('app.shared.error_handling.error_handler')
    def test_decorator_passes_context(self, mock_error_handler):
        """Test that decorator passes function context to error handler."""
        mock_error_handler.handle_error.return_value = Mock()
        
        @with_error_handling()
        def failing_function(arg1, arg2, kwarg1=None):
            raise ValueError("Test error")
        
        with pytest.raises(ValueError):
            failing_function("test1", "test2", kwarg1="test3")
        
        # Verify context was passed
        call_args = mock_error_handler.handle_error.call_args
        context = call_args[0][1]
        assert context['function'] == 'failing_function'
        assert context['args_count'] == 2
        assert 'kwarg1' in context['kwargs_keys']
    
    def test_decorator_preserves_return_value(self):
        """Test that decorator preserves return value for successful calls."""
        @with_error_handling()
        def successful_function():
            return "success"
        
        result = successful_function()
        assert result == "success"


class TestErrorHandlerIntegration:
    """Integration tests for error handling system."""
    
    @patch('app.shared.error_handling.redis_client')
    def test_error_recovery_database(self, mock_redis):
        """Test database error recovery."""
        # Setup
        mock_redis.health_check.side_effect = [False, True]  # Fail then succeed
        mock_redis.reconnect.return_value = None
        mock_redis.set_json.return_value = True
        mock_redis.get_json.return_value = None
        
        handler = ErrorHandler()
        
        # Simulate database error
        db_error = ConnectionError("Database connection failed")
        context = handler.handle_error(db_error)
        
        # Verify recovery was attempted
        assert context.recovery_attempted == True
        mock_redis.reconnect.assert_called_once()
    
    def test_error_statistics_aggregation(self):
        """Test error statistics aggregation."""
        handler = ErrorHandler()
        
        # Generate various types of errors
        errors = [
            ValueError("Validation error 1"),
            ValueError("Validation error 2"),
            AuthenticationError("Auth error 1"),
            MemoryError("Resource error 1")
        ]
        
        for error in errors:
            handler.handle_error(error)
        
        # Check statistics
        with handler._lock:
            assert handler.error_counts[ErrorCategory.VALIDATION] == 2
            assert handler.error_counts[ErrorCategory.AUTHENTICATION] == 1
            assert handler.error_counts[ErrorCategory.RESOURCE] == 1
            assert len(handler.recent_errors) == 4
    
    def test_concurrent_error_handling(self):
        """Test error handling under concurrent access."""
        handler = ErrorHandler()
        errors_handled = []
        
        def handle_error_thread(error_msg):
            try:
                error = ValueError(error_msg)
                context = handler.handle_error(error)
                errors_handled.append(context.error_id)
            except Exception as e:
                pytest.fail(f"Error handling failed: {e}")
        
        # Create multiple threads handling errors concurrently
        threads = []
        for i in range(10):
            thread = threading.Thread(
                target=handle_error_thread, 
                args=(f"Concurrent error {i}",)
            )
            threads.append(thread)
            thread.start()
        
        # Wait for all threads to complete
        for thread in threads:
            thread.join()
        
        # Verify all errors were handled
        assert len(errors_handled) == 10
        assert len(set(errors_handled)) == 10  # All unique error IDs


if __name__ == "__main__":
    pytest.main([__file__])