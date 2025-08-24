"""
Centralized error handling and logging system for the concurrent RAG application.
"""
import logging
import time
import traceback
import uuid
from datetime import datetime
from enum import Enum
from typing import Dict, Any, Optional, List, Callable
from dataclasses import dataclass, asdict
from functools import wraps
import threading
from collections import defaultdict, deque

from .redis_client import redis_client
from .config import config


class ErrorCategory(Enum):
    """Error categories for classification and handling."""
    AUTHENTICATION = "authentication"
    AUTHORIZATION = "authorization"
    VALIDATION = "validation"
    RESOURCE = "resource"
    SYSTEM = "system"
    NETWORK = "network"
    DATABASE = "database"
    EXTERNAL_SERVICE = "external_service"
    BUSINESS_LOGIC = "business_logic"
    UNKNOWN = "unknown"


class ErrorSeverity(Enum):
    """Error severity levels."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass
class ErrorContext:
    """Context information for error tracking."""
    error_id: str
    timestamp: float
    category: ErrorCategory
    severity: ErrorSeverity
    message: str
    exception_type: str
    stack_trace: str
    user_id: Optional[str] = None
    request_id: Optional[str] = None
    endpoint: Optional[str] = None
    user_agent: Optional[str] = None
    ip_address: Optional[str] = None
    additional_context: Optional[Dict[str, Any]] = None
    recovery_attempted: bool = False
    recovery_successful: bool = False
    retry_count: int = 0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        result = asdict(self)
        result['category'] = self.category.value
        result['severity'] = self.severity.value
        result['datetime'] = datetime.fromtimestamp(self.timestamp).isoformat()
        return result


class CircuitBreakerState(Enum):
    """Circuit breaker states."""
    CLOSED = "closed"
    OPEN = "open"
    HALF_OPEN = "half_open"


@dataclass
class CircuitBreakerConfig:
    """Configuration for circuit breaker."""
    failure_threshold: int = 5
    recovery_timeout: int = 60
    expected_exception: type = Exception
    name: str = "default"


class CircuitBreaker:
    """Circuit breaker implementation for error recovery."""
    
    def __init__(self, config: CircuitBreakerConfig):
        self.config = config
        self.failure_count = 0
        self.last_failure_time = None
        self.state = CircuitBreakerState.CLOSED
        self._lock = threading.Lock()
    
    def call(self, func: Callable, *args, **kwargs):
        """Execute function with circuit breaker protection."""
        with self._lock:
            if self.state == CircuitBreakerState.OPEN:
                if self._should_attempt_reset():
                    self.state = CircuitBreakerState.HALF_OPEN
                else:
                    raise Exception(f"Circuit breaker {self.config.name} is OPEN")
            
            try:
                result = func(*args, **kwargs)
                self._on_success()
                return result
            except self.config.expected_exception as e:
                self._on_failure()
                raise e
    
    def _should_attempt_reset(self) -> bool:
        """Check if circuit breaker should attempt reset."""
        return (
            self.last_failure_time and
            time.time() - self.last_failure_time >= self.config.recovery_timeout
        )
    
    def _on_success(self):
        """Handle successful execution."""
        self.failure_count = 0
        self.state = CircuitBreakerState.CLOSED
    
    def _on_failure(self):
        """Handle failed execution."""
        self.failure_count += 1
        self.last_failure_time = time.time()
        
        if self.failure_count >= self.config.failure_threshold:
            self.state = CircuitBreakerState.OPEN


class StructuredLogger:
    """Structured logger with user context and request tracking."""
    
    def __init__(self, name: str):
        self.logger = logging.getLogger(name)
        self._setup_logger()
    
    def _setup_logger(self):
        """Setup logger with structured formatting."""
        if not self.logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - '
                '[%(request_id)s] [%(user_id)s] - %(message)s',
                defaults={
                    'request_id': 'N/A',
                    'user_id': 'N/A'
                }
            )
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)
            self.logger.setLevel(logging.INFO)
    
    def _get_context(self) -> Dict[str, str]:
        """Get current context from thread-local storage."""
        context = getattr(threading.current_thread(), 'log_context', {})
        return {
            'request_id': context.get('request_id', 'N/A'),
            'user_id': context.get('user_id', 'N/A'),
        }
    
    def info(self, message: str, **kwargs):
        """Log info message with context."""
        context = self._get_context()
        self.logger.info(message, extra=context, **kwargs)
    
    def warning(self, message: str, **kwargs):
        """Log warning message with context."""
        context = self._get_context()
        self.logger.warning(message, extra=context, **kwargs)
    
    def error(self, message: str, exc_info=None, **kwargs):
        """Log error message with context."""
        context = self._get_context()
        self.logger.error(message, extra=context, exc_info=exc_info, **kwargs)
    
    def critical(self, message: str, exc_info=None, **kwargs):
        """Log critical message with context."""
        context = self._get_context()
        self.logger.critical(message, extra=context, exc_info=exc_info, **kwargs)


class ErrorHandler:
    """Centralized error handler with categorization and recovery strategies."""
    
    def __init__(self):
        self.logger = StructuredLogger(__name__)
        self.circuit_breakers: Dict[str, CircuitBreaker] = {}
        self.error_counts = defaultdict(int)
        self.recent_errors = deque(maxlen=1000)  # Keep last 1000 errors
        self._lock = threading.Lock()
        
        # Error categorization rules
        self.categorization_rules = {
            'AuthenticationError': ErrorCategory.AUTHENTICATION,
            'TokenExpiredError': ErrorCategory.AUTHENTICATION,
            'TokenInvalidError': ErrorCategory.AUTHENTICATION,
            'PermissionError': ErrorCategory.AUTHORIZATION,
            'ValidationError': ErrorCategory.VALIDATION,
            'RequestValidationError': ErrorCategory.VALIDATION,
            'ValueError': ErrorCategory.VALIDATION,
            'MemoryError': ErrorCategory.RESOURCE,
            'TimeoutError': ErrorCategory.RESOURCE,
            'ConnectionError': ErrorCategory.DATABASE,  # Database connection errors
            'DatabaseError': ErrorCategory.DATABASE,
            'RedisError': ErrorCategory.DATABASE,
            'RedisConnectionError': ErrorCategory.DATABASE,
            'HTTPException': ErrorCategory.SYSTEM,
            'FileNotFoundError': ErrorCategory.SYSTEM,
            'OSError': ErrorCategory.SYSTEM,
        }
        
        # Severity mapping
        self.severity_mapping = {
            ErrorCategory.AUTHENTICATION: ErrorSeverity.MEDIUM,
            ErrorCategory.AUTHORIZATION: ErrorSeverity.MEDIUM,
            ErrorCategory.VALIDATION: ErrorSeverity.LOW,
            ErrorCategory.RESOURCE: ErrorSeverity.HIGH,
            ErrorCategory.SYSTEM: ErrorSeverity.HIGH,
            ErrorCategory.NETWORK: ErrorSeverity.MEDIUM,
            ErrorCategory.DATABASE: ErrorSeverity.CRITICAL,
            ErrorCategory.EXTERNAL_SERVICE: ErrorSeverity.MEDIUM,
            ErrorCategory.BUSINESS_LOGIC: ErrorSeverity.MEDIUM,
            ErrorCategory.UNKNOWN: ErrorSeverity.MEDIUM,
        }
    
    def categorize_error(self, exception: Exception) -> ErrorCategory:
        """Categorize error based on exception type and message."""
        exception_name = type(exception).__name__
        
        # Check direct mapping first
        if exception_name in self.categorization_rules:
            return self.categorization_rules[exception_name]
        
        # Check message patterns for more specific categorization
        error_message = str(exception).lower()
        
        if any(keyword in error_message for keyword in ['permission', 'forbidden', 'unauthorized']):
            return ErrorCategory.AUTHORIZATION
        elif any(keyword in error_message for keyword in ['validation', 'invalid', 'required']):
            return ErrorCategory.VALIDATION
        elif any(keyword in error_message for keyword in ['memory', 'resource', 'limit']):
            return ErrorCategory.RESOURCE
        elif any(keyword in error_message for keyword in ['connection', 'network', 'timeout']):
            return ErrorCategory.NETWORK
        elif any(keyword in error_message for keyword in ['database', 'sql', 'redis']):
            return ErrorCategory.DATABASE
        
        return ErrorCategory.UNKNOWN
    
    def get_severity(self, category: ErrorCategory, exception: Exception) -> ErrorSeverity:
        """Determine error severity based on category and context."""
        base_severity = self.severity_mapping.get(category, ErrorSeverity.MEDIUM)
        
        # Upgrade severity for certain conditions
        error_message = str(exception).lower()
        if any(keyword in error_message for keyword in ['critical', 'fatal', 'crash']):
            return ErrorSeverity.CRITICAL
        elif any(keyword in error_message for keyword in ['security', 'breach', 'attack']):
            return ErrorSeverity.CRITICAL
        
        return base_severity
    
    def handle_error(self, exception: Exception, context: Optional[Dict[str, Any]] = None) -> ErrorContext:
        """Handle error with categorization, logging, and recovery attempts."""
        error_id = str(uuid.uuid4())
        timestamp = time.time()
        
        # Categorize error
        category = self.categorize_error(exception)
        severity = self.get_severity(category, exception)
        
        # Extract context information
        thread_context = getattr(threading.current_thread(), 'log_context', {})
        
        # Create error context
        error_context = ErrorContext(
            error_id=error_id,
            timestamp=timestamp,
            category=category,
            severity=severity,
            message=str(exception),
            exception_type=type(exception).__name__,
            stack_trace=traceback.format_exc(),
            user_id=thread_context.get('user_id'),
            request_id=thread_context.get('request_id'),
            endpoint=thread_context.get('endpoint'),
            user_agent=thread_context.get('user_agent'),
            ip_address=thread_context.get('ip_address'),
            additional_context=context
        )
        
        # Log error with appropriate level
        self._log_error(error_context)
        
        # Store error for monitoring
        self._store_error(error_context)
        
        # Update error statistics
        with self._lock:
            self.error_counts[category] += 1
            self.recent_errors.append(error_context)
        
        # Attempt recovery if applicable
        self._attempt_recovery(error_context, exception)
        
        return error_context
    
    def _log_error(self, error_context: ErrorContext):
        """Log error with appropriate level based on severity."""
        message = f"[{error_context.category.value.upper()}] {error_context.message}"
        
        if error_context.severity == ErrorSeverity.CRITICAL:
            self.logger.critical(message, exc_info=True)
        elif error_context.severity == ErrorSeverity.HIGH:
            self.logger.error(message, exc_info=True)
        elif error_context.severity == ErrorSeverity.MEDIUM:
            self.logger.warning(message)
        else:
            self.logger.info(message)
    
    def _store_error(self, error_context: ErrorContext):
        """Store error in Redis for monitoring and analysis."""
        try:
            # Store individual error
            error_key = f"error:{error_context.error_id}"
            redis_client.set_json(error_key, error_context.to_dict(), expire_seconds=86400 * 7)
            
            # Update error statistics
            date_key = f"error_stats:daily:{time.strftime('%Y-%m-%d', time.localtime(error_context.timestamp))}"
            stats = redis_client.get_json(date_key) or {
                "date": time.strftime('%Y-%m-%d', time.localtime(error_context.timestamp)),
                "total_errors": 0,
                "by_category": {},
                "by_severity": {},
                "by_user": {}
            }
            
            stats["total_errors"] += 1
            stats["by_category"][error_context.category.value] = stats["by_category"].get(error_context.category.value, 0) + 1
            stats["by_severity"][error_context.severity.value] = stats["by_severity"].get(error_context.severity.value, 0) + 1
            
            if error_context.user_id:
                stats["by_user"][error_context.user_id] = stats["by_user"].get(error_context.user_id, 0) + 1
            
            redis_client.set_json(date_key, stats, expire_seconds=86400 * 30)
            
        except Exception as e:
            # Don't let error storage failure affect the main application
            self.logger.warning(f"Failed to store error context: {e}")
    
    def _attempt_recovery(self, error_context: ErrorContext, exception: Exception):
        """Attempt error recovery based on error category."""
        recovery_attempted = False
        recovery_successful = False
        
        try:
            if error_context.category == ErrorCategory.NETWORK:
                # For network errors, we might want to retry with exponential backoff
                recovery_attempted = True
                # Implementation would depend on specific use case
                
            elif error_context.category == ErrorCategory.RESOURCE:
                # For resource errors, attempt cleanup
                recovery_attempted = True
                self._attempt_resource_cleanup()
                
            elif error_context.category == ErrorCategory.DATABASE:
                # For database errors, attempt reconnection
                recovery_attempted = True
                recovery_successful = self._attempt_database_recovery()
            
            # Update error context with recovery information
            error_context.recovery_attempted = recovery_attempted
            error_context.recovery_successful = recovery_successful
            
        except Exception as recovery_error:
            self.logger.error(f"Recovery attempt failed for error {error_context.error_id}: {recovery_error}")
    
    def _attempt_resource_cleanup(self):
        """Attempt to free up system resources."""
        try:
            import gc
            gc.collect()
            # Additional cleanup logic could be added here
        except Exception as e:
            self.logger.warning(f"Resource cleanup failed: {e}")
    
    def _attempt_database_recovery(self) -> bool:
        """Attempt to recover database connections."""
        try:
            # Test Redis connection
            if redis_client.health_check():
                return True
            else:
                # Attempt to reconnect
                redis_client.reconnect()
                return redis_client.health_check()
        except Exception as e:
            self.logger.error(f"Database recovery failed: {e}")
            return False
    
    def get_circuit_breaker(self, name: str, config: Optional[CircuitBreakerConfig] = None) -> CircuitBreaker:
        """Get or create circuit breaker for a service."""
        if name not in self.circuit_breakers:
            if config is None:
                config = CircuitBreakerConfig(name=name)
            self.circuit_breakers[name] = CircuitBreaker(config)
        return self.circuit_breakers[name]
    
    def get_error_statistics(self, days: int = 7) -> Dict[str, Any]:
        """Get error statistics for monitoring dashboard."""
        try:
            stats = {
                "period_days": days,
                "daily_stats": [],
                "summary": {
                    "total_errors": 0,
                    "by_category": defaultdict(int),
                    "by_severity": defaultdict(int),
                    "top_users": {},
                    "error_rate_trend": []
                }
            }
            
            # Get daily stats for the specified period
            for i in range(days):
                date = time.strftime("%Y-%m-%d", time.localtime(time.time() - i * 86400))
                date_key = f"error_stats:daily:{date}"
                
                daily_data = redis_client.get_json(date_key)
                if daily_data:
                    stats["daily_stats"].append(daily_data)
                    
                    # Update summary
                    stats["summary"]["total_errors"] += daily_data["total_errors"]
                    
                    for category, count in daily_data.get("by_category", {}).items():
                        stats["summary"]["by_category"][category] += count
                    
                    for severity, count in daily_data.get("by_severity", {}).items():
                        stats["summary"]["by_severity"][severity] += count
                    
                    # Track error rate trend
                    stats["summary"]["error_rate_trend"].append({
                        "date": date,
                        "error_count": daily_data["total_errors"]
                    })
            
            # Convert defaultdicts to regular dicts for JSON serialization
            stats["summary"]["by_category"] = dict(stats["summary"]["by_category"])
            stats["summary"]["by_severity"] = dict(stats["summary"]["by_severity"])
            
            return stats
            
        except Exception as e:
            self.logger.error(f"Failed to get error statistics: {e}")
            return {"error": str(e)}
    
    def get_recent_errors(self, limit: int = 50) -> List[Dict[str, Any]]:
        """Get recent errors for monitoring."""
        with self._lock:
            recent = list(self.recent_errors)[-limit:]
            return [error.to_dict() for error in recent]


# Global error handler instance
error_handler = ErrorHandler()


def with_error_handling(category: Optional[ErrorCategory] = None):
    """Decorator for automatic error handling."""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                context = {
                    'function': func.__name__,
                    'module': func.__module__,
                    'args_count': len(args),
                    'kwargs_keys': list(kwargs.keys())
                }
                
                error_context = error_handler.handle_error(e, context)
                
                # Re-raise the exception after handling
                raise e
        return wrapper
    return decorator


def set_log_context(**context):
    """Set logging context for current thread."""
    if not hasattr(threading.current_thread(), 'log_context'):
        threading.current_thread().log_context = {}
    
    threading.current_thread().log_context.update(context)


def clear_log_context():
    """Clear logging context for current thread."""
    if hasattr(threading.current_thread(), 'log_context'):
        threading.current_thread().log_context.clear()


def get_log_context() -> Dict[str, Any]:
    """Get current logging context."""
    return getattr(threading.current_thread(), 'log_context', {})