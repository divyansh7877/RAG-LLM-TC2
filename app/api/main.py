"""
FastAPI application main module.
"""
import logging
import time
import uuid
from datetime import datetime
from typing import Dict, Any, Optional

from fastapi import FastAPI, HTTPException, Depends, status, Request, Response
import os
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from fastapi.responses import JSONResponse
from fastapi.exceptions import RequestValidationError
from fastapi.staticfiles import StaticFiles
from contextlib import asynccontextmanager
import uvicorn

from ..shared.config import config
from ..shared.redis_client import redis_client
from ..shared.middleware import rate_limiter, get_current_user, require_roles, require_any_role
from ..shared.error_handling import error_handler, set_log_context, clear_log_context, StructuredLogger
from ..shared.monitoring import metric_collector, alert_manager, health_checker, start_monitoring_thread
from ..shared.models import Document, User
from ..shared.auth import auth_manager, AuthenticationError, TokenExpiredError, TokenInvalidError

# Set up structured logging
logger = StructuredLogger(__name__)

# Security
security = HTTPBearer()




class RequestLoggingMiddleware:
    """Enhanced middleware for request logging and monitoring with error handling."""
    
    def __init__(self, app: FastAPI):
        self.app = app
    
    async def __call__(self, request: Request, call_next):
        """Process request with enhanced logging, monitoring, and error handling."""
        # Generate request ID
        request_id = str(uuid.uuid4())
        request.state.request_id = request_id
        
        # Extract client information
        client_ip = request.client.host if request.client else "unknown"
        user_agent = request.headers.get("user-agent", "unknown")
        
        # Set log context for this request
        set_log_context(
            request_id=request_id,
            endpoint=request.url.path,
            ip_address=client_ip,
            user_agent=user_agent
        )
        
        # Log request start
        start_time = time.time()
        logger.info(
            f"Request started - Method: {request.method}, "
            f"Path: {request.url.path}, Client: {client_ip}"
        )
        
        try:
            # Process request
            response = await call_next(request)
            
            # Log successful response
            process_time = time.time() - start_time
            logger.info(
                f"Request completed - Status: {response.status_code}, "
                f"Time: {process_time:.3f}s"
            )
            
            # Add request ID and timing to response headers
            response.headers["X-Request-ID"] = request_id
            response.headers["X-Process-Time"] = f"{process_time:.3f}"
            
            return response
            
        except Exception as e:
            # Handle error through centralized system
            process_time = time.time() - start_time
            
            error_context = error_handler.handle_error(e, {
                'request_method': request.method,
                'request_path': request.url.path,
                'process_time': process_time,
                'client_ip': client_ip,
                'user_agent': user_agent
            })
            
            # Return structured error response
            return JSONResponse(
                status_code=500,
                content={
                    "error": {
                        "code": "INTERNAL_SERVER_ERROR",
                        "message": "An internal server error occurred",
                        "request_id": request_id,
                        "error_id": error_context.error_id,
                        "timestamp": datetime.utcnow().isoformat() + "Z"
                    }
                },
                headers={"X-Request-ID": request_id, "X-Error-ID": error_context.error_id}
            )
        
        finally:
            # Clear log context
            clear_log_context()


class ResponseFormattingMiddleware:
    """Middleware for consistent response formatting."""
    
    def __init__(self, app: FastAPI):
        self.app = app
    
    async def __call__(self, request: Request, call_next):
        """Process response with consistent formatting."""
        response = await call_next(request)
        
        # Add standard headers
        response.headers["X-API-Version"] = "1.0.0"
        response.headers["X-Timestamp"] = datetime.utcnow().isoformat() + "Z"
        
        return response


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan events with enhanced monitoring and error handling."""
    # Startup
    logger.info("Starting FastAPI application...")
    
    try:
        # Load Keycloak public keys
        await auth_manager.load_jwks()
        logger.info("Authentication manager initialized with Keycloak.")

        # Check Redis connection
        if not redis_client.health_check():
            logger.warning("Redis connection failed")
        else:
            logger.info("Redis connection successful")
        
        # Initialize middleware components
        logger.info("Initializing middleware components...")
        
        # Initialize job notification service
        from ..shared.job_notifications import job_notification_service
        job_notification_service.initialize()
        logger.info("Job notification service initialized")
        
        # Start monitoring thread
        logger.info("Starting system monitoring...")
        monitoring_thread = start_monitoring_thread(interval=60)  # Monitor every minute
        logger.info("System monitoring started")
        
        # Setup alert callbacks for critical alerts
        def critical_alert_callback(alert):
            """Handle critical alerts."""
            if alert.level.value == "critical":
                logger.critical(f"CRITICAL ALERT: {alert.title} - {alert.message}")
                # In production, you might want to send notifications here
        
        alert_manager.add_alert_callback(critical_alert_callback)
        logger.info("Alert system initialized")
        
    except Exception as e:
        error_handler.handle_error(e, {'operation': 'application_startup'})
        logger.error(f"Error during application startup: {e}")
        # Continue startup even if monitoring fails
    
    yield
    
    # Shutdown
    logger.info("Shutting down FastAPI application...")
    try:
        # Cleanup monitoring resources if needed
        logger.info("Monitoring system shutdown complete")
    except Exception as e:
        logger.error(f"Error during shutdown: {e}")


# Create FastAPI app
app = FastAPI(
    title="Concurrent RAG System",
    description="Multi-user RAG system with concurrent processing",
    version="1.0.0",
    lifespan=lifespan,
    docs_url="/api/docs",
    redoc_url="/api/redoc",
    openapi_url="/api/openapi.json"
)

# Add security middleware
app.add_middleware(TrustedHostMiddleware, allowed_hosts=["*"])  # Configure for production

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
    allow_headers=["*"],
)

# Add custom middleware
app.middleware("http")(RequestLoggingMiddleware(app))
app.middleware("http")(ResponseFormattingMiddleware(app))

# Mount static files
from pathlib import Path
from fastapi.responses import FileResponse
static_dir = Path(__file__).parent.parent / "static"
app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")


# Global exception handlers
@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    """Handle request validation errors."""
    request_id = getattr(request.state, 'request_id', 'unknown')
    
    logger.warning(f"Validation error - Request ID: {request_id}, Errors: {exc.errors()}")
    
    return JSONResponse(
        status_code=422,
        content={
            "error": {
                "code": "VALIDATION_ERROR",
                "message": "Request validation failed",
                "details": exc.errors(),
                "request_id": request_id,
                "timestamp": datetime.utcnow().isoformat() + "Z"
            }
        },
        headers={"X-Request-ID": request_id}
    )


@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    """Handle HTTP exceptions with consistent formatting."""
    request_id = getattr(request.state, 'request_id', 'unknown')
    
    logger.warning(f"HTTP exception - Request ID: {request_id}, Status: {exc.status_code}, Detail: {exc.detail}")
    
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "error": {
                "code": "HTTP_ERROR",
                "message": exc.detail,
                "request_id": request_id,
                "timestamp": datetime.utcnow().isoformat() + "Z"
            }
        },
        headers={"X-Request-ID": request_id}
    )


@app.exception_handler(Exception)
async def general_exception_handler(request: Request, exc: Exception):
    """Handle general exceptions."""
    request_id = getattr(request.state, 'request_id', 'unknown')
    
    logger.error(f"Unhandled exception - Request ID: {request_id}, Error: {str(exc)}", exc_info=True)
    
    return JSONResponse(
        status_code=500,
        content={
            "error": {
                "code": "INTERNAL_SERVER_ERROR",
                "message": "An internal server error occurred",
                "request_id": request_id,
                "timestamp": datetime.utcnow().isoformat() + "Z"
            }
        },
        headers={"X-Request-ID": request_id}
    )


@app.get("/")
async def serve_frontend():
    """Serve the main frontend application."""
    static_dir = Path(__file__).parent.parent / "static"
    return FileResponse(str(static_dir / "index.html"))

@app.get("/app")
async def serve_app():
    """Alternative route to serve the frontend application."""
    static_dir = Path(__file__).parent.parent / "static"
    return FileResponse(str(static_dir / "index.html"))

@app.get("/api")
async def api_root():
    """API root endpoint."""
    return {
        "message": "Concurrent RAG System API",
        "version": "1.0.0",
        "timestamp": datetime.utcnow().isoformat() + "Z"
    }


@app.get("/health")
async def health_check():
    """Enhanced health check endpoint with comprehensive system monitoring."""
    try:
        # Run all registered health checks
        health_results = health_checker.run_health_checks()
        
        return {
            "status": "healthy" if health_results["overall_healthy"] else "unhealthy",
            "checks": health_results["checks"],
            "timestamp": datetime.utcnow().isoformat() + "Z"
        }
    except Exception as e:
        error_handler.handle_error(e, {'operation': 'health_check'})
        return JSONResponse(
            status_code=503,
            content={
                "status": "unhealthy",
                "error": "Health check system failure",
                "timestamp": datetime.utcnow().isoformat() + "Z"
            }
        )


@app.get("/api/status")
async def api_status():
    """Detailed API status endpoint."""
    try:
        # Check various system components
        redis_healthy = redis_client.health_check()
        
        # TODO: Add checks for other components (Celery workers, database, etc.)
        
        status = {
            "api_version": "1.0.0",
            "status": "operational",
            "services": {
                "redis": {
                    "status": "healthy" if redis_healthy else "unhealthy",
                    "last_check": datetime.utcnow().isoformat() + "Z"
                },
                "authentication": {
                    "status": "healthy",
                    "last_check": datetime.utcnow().isoformat() + "Z"
                }
            },
            "resource_usage": {
                # TODO: Add actual resource monitoring
                "memory_usage": "unknown",
                "cpu_usage": "unknown",
                "active_connections": "unknown"
            },
            "timestamp": datetime.utcnow().isoformat() + "Z"
        }
        
        overall_healthy = all(
            service.get("status") == "healthy" 
            for service in status["services"].values()
        )
        
        if not overall_healthy:
            status["status"] = "degraded"
        
        return status
        
    except Exception as e:
        logger.error(f"Status check failed: {e}")
        return JSONResponse(
            status_code=503,
            content={
                "error": {
                    "code": "SERVICE_UNAVAILABLE",
                    "message": "Unable to determine system status",
                    "timestamp": datetime.utcnow().isoformat() + "Z"
                }
            }
        )


# Enhanced monitoring and error handling endpoints
@app.get("/api/monitoring/metrics", tags=["Monitoring"])
async def get_system_metrics(
    current_user: User = Depends(get_current_user)
):
    """
    Get current system metrics.
    
    Args:
        current_user: Current authenticated user
    
    Returns:
        dict: Current system metrics
    """
    try:
        # Collect current metrics
        metrics = metric_collector.collect_system_metrics()
        
        return {
            "metrics": metrics.to_dict(),
            "requested_by": current_user.id,
            "timestamp": datetime.utcnow().isoformat() + "Z"
        }
    
    except Exception as e:
        error_handler.handle_error(e, {'operation': 'get_system_metrics'})
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Metrics collection service error"
        )


@app.get("/api/monitoring/metrics/history", tags=["Monitoring"])
async def get_metrics_history(
    hours: int = 24,
    current_user: User = Depends(get_current_user)
):
    """
    Get historical system metrics.
    
    Args:
        hours: Number of hours of history to retrieve (default: 24)
        current_user: Current authenticated user
    
    Returns:
        dict: Historical system metrics
    """
    try:
        # Validate hours parameter
        if hours < 1 or hours > 168:  # Max 1 week
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Hours parameter must be between 1 and 168"
            )
        
        history = metric_collector.get_metrics_history(hours=hours)
        
        return {
            "metrics_history": history,
            "period_hours": hours,
            "requested_by": current_user.id,
            "timestamp": datetime.utcnow().isoformat() + "Z"
        }
    
    except HTTPException:
        raise
    except Exception as e:
        error_handler.handle_error(e, {'operation': 'get_metrics_history'})
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Metrics history service error"
        )


@app.get("/api/monitoring/alerts", tags=["Monitoring"])
async def get_active_alerts(
    current_user: User = Depends(get_current_user)
):
    """
    Get all active system alerts.
    
    Args:
        current_user: Current authenticated user
    
    Returns:
        dict: Active system alerts
    """
    try:
        active_alerts = alert_manager.get_active_alerts()
        
        return {
            "active_alerts": active_alerts,
            "alert_count": len(active_alerts),
            "requested_by": current_user.id,
            "timestamp": datetime.utcnow().isoformat() + "Z"
        }
    
    except Exception as e:
        error_handler.handle_error(e, {'operation': 'get_active_alerts'})
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Alert service error"
        )


@app.get("/api/monitoring/alerts/history", tags=["Monitoring"])
async def get_alert_history(
    hours: int = 24,
    current_user: User = Depends(get_current_user)
):
    """
    Get alert history.
    
    Args:
        hours: Number of hours of history to retrieve (default: 24)
        current_user: Current authenticated user
    
    Returns:
        dict: Alert history
    """
    try:
        # Validate hours parameter
        if hours < 1 or hours > 168:  # Max 1 week
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Hours parameter must be between 1 and 168"
            )
        
        history = alert_manager.get_alert_history(hours=hours)
        
        return {
            "alert_history": history,
            "period_hours": hours,
            "requested_by": current_user.id,
            "timestamp": datetime.utcnow().isoformat() + "Z"
        }
    
    except HTTPException:
        raise
    except Exception as e:
        error_handler.handle_error(e, {'operation': 'get_alert_history'})
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Alert history service error"
        )


@app.get("/api/monitoring/errors", tags=["Monitoring"])
async def get_error_statistics(
    days: int = 7,
    current_user: User = Depends(get_current_user)
):
    """
    Get error statistics and recent errors.
    
    Args:
        days: Number of days of statistics to retrieve (default: 7)
        current_user: Current authenticated user
    
    Returns:
        dict: Error statistics and recent errors
    """
    try:
        # Validate days parameter
        if days < 1 or days > 30:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Days parameter must be between 1 and 30"
            )
        
        error_stats = error_handler.get_error_statistics(days=days)
        recent_errors = error_handler.get_recent_errors(limit=50)
        
        return {
            "error_statistics": error_stats,
            "recent_errors": recent_errors,
            "requested_by": current_user.id,
            "timestamp": datetime.utcnow().isoformat() + "Z"
        }
    
    except HTTPException:
        raise
    except Exception as e:
        error_handler.handle_error(e, {'operation': 'get_error_statistics'})
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Error statistics service error"
        )


@app.get("/api/monitoring/health/detailed", tags=["Monitoring"])
async def get_detailed_health_status(
    current_user: User = Depends(get_current_user)
):
    """
    Get detailed health status of all system components.
    
    Args:
        current_user: Current authenticated user
    
    Returns:
        dict: Detailed health status
    """
    try:
        # Run comprehensive health checks
        health_results = health_checker.run_health_checks()
        
        # Get current metrics for additional context
        current_metrics = metric_collector.collect_system_metrics()
        
        # Get active alerts that might affect health
        active_alerts = alert_manager.get_active_alerts()
        critical_alerts = [alert for alert in active_alerts if alert.get('level') == 'critical']
        
        return {
            "overall_health": health_results,
            "current_metrics": current_metrics.to_dict(),
            "critical_alerts": critical_alerts,
            "health_summary": {
                "overall_healthy": health_results["overall_healthy"],
                "total_checks": len(health_results["checks"]),
                "failed_checks": len([
                    check for check in health_results["checks"].values() 
                    if not check.get("healthy", False)
                ]),
                "critical_alert_count": len(critical_alerts)
            },
            "requested_by": current_user.id,
            "timestamp": datetime.utcnow().isoformat() + "Z"
        }
    
    except Exception as e:
        error_handler.handle_error(e, {'operation': 'get_detailed_health_status'})
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Health status service error"
        )


# Performance monitoring endpoints
@app.get("/api/performance/query", tags=["Performance"])
async def get_query_performance_metrics(
    days: int = 7,
    current_user: User = Depends(get_current_user)
):
    """
    Get query performance metrics and statistics.
    
    Args:
        days: Number of days to retrieve statistics for (default: 7)
        current_user: Current authenticated user
    
    Returns:
        dict: Query performance statistics
    """
    try:
        from ..workers.query_worker import get_query_performance_stats
        
        # Validate days parameter
        if days < 1 or days > 30:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Days parameter must be between 1 and 30"
            )
        
        stats = get_query_performance_stats(days=days)
        
        return {
            "query_performance": stats,
            "requested_by": current_user.id,
            "timestamp": datetime.utcnow().isoformat() + "Z"
        }
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting query performance metrics: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Performance metrics service error"
        )


@app.get("/api/performance/embedding", tags=["Performance"])
async def get_embedding_performance_metrics(
    days: int = 7,
    current_user: User = Depends(get_current_user)
):
    """
    Get embedding performance metrics and statistics.
    
    Args:
        days: Number of days to retrieve statistics for (default: 7)
        current_user: Current authenticated user
    
    Returns:
        dict: Embedding performance statistics
    """
    try:
        from ..workers.embedding_worker import get_embedding_performance_stats
        
        # Validate days parameter
        if days < 1 or days > 30:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Days parameter must be between 1 and 30"
            )
        
        stats = get_embedding_performance_stats(days=days)
        
        return {
            "embedding_performance": stats,
            "requested_by": current_user.id,
            "timestamp": datetime.utcnow().isoformat() + "Z"
        }
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting embedding performance metrics: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Performance metrics service error"
        )


@app.get("/api/performance/system", tags=["Performance"])
async def get_system_performance_metrics(
    current_user: User = Depends(get_current_user)
):
    """
    Get overall system performance metrics.
    
    Args:
        current_user: Current authenticated user
    
    Returns:
        dict: System performance metrics
    """
    try:
        from ..shared.resource_manager import resource_manager
        from ..shared.query_engine_factory import query_engine_factory
        
        # Get resource manager health status
        resource_health = resource_manager.get_health_status()
        
        # Get query engine factory stats
        factory_stats = query_engine_factory.get_factory_stats()
        
        # Get current metrics
        current_metrics = resource_manager.get_current_metrics()
        
        system_performance = {
            "resource_health": resource_health,
            "query_engine_stats": factory_stats,
            "current_metrics": {
                "memory_usage": f"{current_metrics.memory_usage:.1f}%" if current_metrics else "unknown",
                "cpu_usage": f"{current_metrics.cpu_usage:.1f}%" if current_metrics else "unknown",
                "disk_usage": f"{current_metrics.disk_usage:.1f}%" if current_metrics else "unknown",
                "active_workers": current_metrics.active_workers if current_metrics else 0,
                "active_tasks": current_metrics.active_tasks if current_metrics else 0,
                "queue_lengths": current_metrics.queue_lengths if current_metrics else {}
            } if current_metrics else {
                "memory_usage": "unknown",
                "cpu_usage": "unknown", 
                "disk_usage": "unknown",
                "active_workers": 0,
                "active_tasks": 0,
                "queue_lengths": {}
            }
        }
        
        return {
            "system_performance": system_performance,
            "requested_by": current_user.id,
            "timestamp": datetime.utcnow().isoformat() + "Z"
        }
    
    except Exception as e:
        logger.error(f"Error getting system performance metrics: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="System performance metrics service error"
        )


@app.get("/api/performance/cache", tags=["Performance"])
async def get_cache_performance_metrics(
    current_user: User = Depends(get_current_user)
):
    """
    Get cache performance metrics and statistics.
    
    Args:
        current_user: Current authenticated user
    
    Returns:
        dict: Cache performance metrics
    """
    try:
        from ..shared.query_engine_factory import query_engine_factory
        
        # Get factory stats which include cache information
        factory_stats = query_engine_factory.get_factory_stats()
        
        cache_performance = {
            "query_cache": factory_stats.get("query_cache", {}),
            "connection_pool": factory_stats.get("connection_pool", {}),
            "factory_health": query_engine_factory.health_check()
        }
        
        return {
            "cache_performance": cache_performance,
            "requested_by": current_user.id,
            "timestamp": datetime.utcnow().isoformat() + "Z"
        }
    
    except Exception as e:
        logger.error(f"Error getting cache performance metrics: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Cache performance metrics service error"
        )





# Document management endpoints
import tempfile
import shutil
from pathlib import Path
from fastapi import UploadFile, File, Form
from typing import List
from ..shared.job_manager import job_manager, JobType, JobStatus
from ..workers.celery_app import celery_app

@app.get("/api/documents", tags=["Documents"])
async def list_documents(
    group_id: Optional[str] = None,
    status: Optional[str] = None,
    limit: int = 50,
    current_user: User = Depends(get_current_user)
):
    """
    List user's documents with optional filtering.
    """
    try:
        if limit < 1 or limit > 100:
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Limit must be between 1 and 100")

        documents = redis_client.get_user_documents(current_user.id)

        if group_id:
            documents = [d for d in documents if d.group_id == group_id]

        if status:
            documents = [d for d in documents if d.processing_status == status]

        documents = documents[:limit]

        return {
            "documents": [d.to_dict() for d in documents],
            "total_count": len(documents)
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error listing documents for user {current_user.id}: {e}")
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Document listing service error")

@app.delete("/api/documents/{document_id}", tags=["Documents"])
async def delete_document(
    document_id: str,
    current_user: User = Depends(get_current_user)
):
    """Delete a user's document by id (searching user/group-prefixed keys)."""
    try:
        # Try user/group composite key space first (preferred)
        for group_id in current_user.groups + [current_user.id]:
            key = f"document:{current_user.id}:{group_id}:{document_id}"
            if redis_client.delete(key):
                return {"message": "Document deleted", "document_id": document_id}

        # Fallback to legacy simple key
        if redis_client.delete(f"document:{document_id}"):
            return {"message": "Document deleted", "document_id": document_id}

        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Document not found")
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error deleting document {document_id} for user {current_user.id}: {e}")
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Document deletion service error")

@app.post("/api/documents/upload", tags=["Documents"])
async def upload_documents(
    request: Request,
    files: List[UploadFile] = File(...),
    group_id: str = Form(...),
    # Accept users with upload, standard, or admin role
    current_user: User = Depends(require_any_role(["upload", "standard", "admin"])),
    rate_limit: None = Depends(rate_limiter.create_rate_limiter(10, 300))  # 10 uploads per 5 minutes
):
    """
    Upload documents for processing.
    
    Args:
        request: FastAPI request object
        files: List of uploaded files
        group_id: Group ID for document organization
        current_user: Current authenticated user
        rate_limit: Rate limiting dependency
    
    Returns:
        dict: Upload result with job information
    
    Raises:
        HTTPException: If upload fails
    """
    temp_dir = None
    temp_files = []
    
    try:
        # Validate group access
        if group_id not in current_user.groups and group_id != current_user.id:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"User does not have access to group: {group_id}"
            )
        
        # Validate files
        if not files or len(files) == 0:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="No files provided"
            )
        
        if len(files) > 10:  # Limit number of files per upload
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Maximum 10 files allowed per upload"
            )
        
        # Create temporary directory for file processing
        temp_dir = tempfile.mkdtemp(prefix="rag_upload_")
        
        # Process and validate each file
        for file in files:
            # Validate file type using the document processor's supported formats
            from ..shared.pdf_utils import is_supported_format, get_supported_extensions
            
            if not file.filename or not is_supported_format(file.filename):
                supported_formats = ', '.join(get_supported_extensions())
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail=f"Unsupported file format: {file.filename}. Supported formats: {supported_formats}"
                )
            
            # Save file to temporary location and validate size against config
            temp_file_path = Path(temp_dir) / file.filename
            try:
                with open(temp_file_path, "wb") as buffer:
                    shutil.copyfileobj(file.file, buffer)
                # Validate file size (max from config)
                file_size = os.path.getsize(temp_file_path)
                if file_size > config.MAX_FILE_SIZE:
                    # Cleanup oversized file immediately
                    try:
                        os.remove(temp_file_path)
                    except Exception:
                        pass
                    raise HTTPException(
                        status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
                        detail=f"File too large: {file.filename}. Maximum size is {int(config.MAX_FILE_SIZE/(1024*1024))}MB"
                    )
                temp_files.append(str(temp_file_path))
            except HTTPException:
                raise
            except Exception as e:
                logger.error(f"Failed to save uploaded file {file.filename}: {e}")
                raise HTTPException(
                    status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                    detail=f"Failed to save file: {file.filename}"
                )
        
        # Create embedding job
        try:
            job = job_manager.create_job(
                user_id=current_user.id,
                job_type=JobType.EMBEDDING,
                metadata={
                    "group_id": group_id,
                    "file_count": len(temp_files),
                    "filenames": [Path(f).name for f in temp_files],
                    "temp_dir": temp_dir,
                    "upload_source": "api"
                }
            )
        except Exception as job_error:
            # Handle job creation failures with better error messages
            error_msg = str(job_error)
            logger.error(f"Failed to create job for user {current_user.id}: {error_msg}")
            
            # Check for specific error conditions and provide helpful messages
            if "maximum concurrent jobs limit" in error_msg.lower():
                # Get current active job count for user
                active_jobs = job_manager.get_user_jobs(current_user.id, active_only=True)
                raise HTTPException(
                    status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                    detail={
                        "error": "Too many concurrent jobs",
                        "message": f"You have {len(active_jobs)} active jobs running. Please wait for some jobs to complete before uploading more files.",
                        "active_job_count": len(active_jobs),
                        "max_allowed": 10,
                        "suggestion": "You can check job status at /api/jobs or cancel stuck jobs if any."
                    }
                )
            else:
                # Generic job creation error
                raise HTTPException(
                    status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                    detail=f"Failed to create processing job: {error_msg}"
                )
        
        # Queue embedding task
        task = celery_app.send_task(
            "process_document_embedding",
            args=[
                job.job_id,
                current_user.id,
                group_id,
                temp_files
            ],
            queue="embedding"
        )
        
        # Update job with task ID
        job.metadata["celery_task_id"] = task.id
        redis_client.set_job(job)
        
        logger.info(f"Created embedding job {job.job_id} for user {current_user.id} with {len(temp_files)} files")
        
        return {
            "job_id": job.job_id,
            "message": f"Upload successful. Processing {len(temp_files)} files.",
            "files_count": len(temp_files),
            "filenames": [Path(f).name for f in temp_files],
            "status": "queued",
            "estimated_processing_time": f"{len(temp_files) * 30} seconds"
        }
    
    except HTTPException:
        # Clean up temp files on HTTP errors
        if temp_dir and Path(temp_dir).exists():
            try:
                shutil.rmtree(temp_dir)
            except Exception as e:
                logger.warning(f"Failed to cleanup temp directory after error: {e}")
        raise
    
    except Exception as e:
        # Clean up temp files on unexpected errors
        if temp_dir and Path(temp_dir).exists():
            try:
                shutil.rmtree(temp_dir)
            except Exception as e:
                logger.warning(f"Failed to cleanup temp directory after error: {e}")
        
        logger.error(f"Document upload error for user {current_user.id}: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Document upload service error"
        )


@app.get("/api/documents/{document_id}", tags=["Documents"])
async def get_document(
    document_id: str,
    current_user: User = Depends(get_current_user)
):
    """
    Get specific document metadata.
    
    Args:
        document_id: Document identifier
        current_user: Current authenticated user
    
    Returns:
        dict: Document metadata
    
    Raises:
        HTTPException: If document not found or access denied
    """
    try:
        # Search for document across user's groups
        document = None
        for group_id in current_user.groups:
            doc_key = f"document:{current_user.id}:{group_id}:{document_id}"
            doc_data = redis_client.get_json(doc_key)
            if doc_data:
                document = Document.from_dict(doc_data)
                break
        
        if not document:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Document not found"
            )
        
        return document.to_dict()
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting document {document_id} for user {current_user.id}: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Document retrieval service error"
        )





@app.get("/api/documents/{document_id}/status", tags=["Documents"])
async def get_document_status(
    document_id: str,
    current_user: User = Depends(get_current_user)
):
    """
    Get document processing status.
    
    Args:
        document_id: Document identifier
        current_user: Current authenticated user
    
    Returns:
        dict: Document processing status
    
    Raises:
        HTTPException: If document not found or access denied
    """
    try:
        # Search for document across user's groups
        document = None
        for group_id in current_user.groups:
            doc_key = f"document:{current_user.id}:{group_id}:{document_id}"
            doc_data = redis_client.get_json(doc_key)
            if doc_data:
                document = Document.from_dict(doc_data)
                break
        
        if not document:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Document not found"
            )
        
        return {
            "document_id": document_id,
            "filename": document.filename,
            "processing_status": document.processing_status,
            "upload_date": document.upload_date.isoformat() + "Z" if document.upload_date else None,
            "file_size": document.file_size,
            "page_count": document.page_count,
            "chunk_count": document.chunk_count
        }
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting document status {document_id} for user {current_user.id}: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Document status service error"
        )



# Job management endpoints
from ..shared.job_manager import job_manager, JobType, JobStatus


@app.get("/api/jobs", tags=["Jobs"])
async def list_jobs(
    request: Request,
    job_type: Optional[str] = None,
    status: Optional[str] = None,
    active_only: bool = False,
    limit: int = 50,
    current_user: User = Depends(get_current_user)
):
    """
    List user's jobs with filtering.
    
    Args:
        request: FastAPI request object
        job_type: Optional job type filter (embedding, query)
        status: Optional status filter
        active_only: If True, only return non-finished jobs
        limit: Maximum number of jobs to return
        current_user: Current authenticated user
    
    Returns:
        dict: List of jobs with metadata
    """
    try:
        # Parse job type filter
        job_type_filter = None
        if job_type:
            try:
                job_type_filter = JobType(job_type.lower())
            except ValueError:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail=f"Invalid job type: {job_type}. Valid types: {[t.value for t in JobType]}"
                )
        
        # Parse status filter
        status_filter = None
        if status:
            try:
                status_filter = JobStatus(status.lower())
            except ValueError:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail=f"Invalid status: {status}. Valid statuses: {[s.value for s in JobStatus]}"
                )
        
        # Get user jobs
        jobs = job_manager.get_user_jobs(
            user_id=current_user.id,
            job_type=job_type_filter,
            status=status_filter,
            active_only=active_only,
            limit=limit
        )
        
        # Convert to dict format
        job_list = []
        for job in jobs:
            job_dict = job.to_dict()
            # Add estimated completion time for processing jobs
            if job.status == JobStatus.PROCESSING and job.progress > 0:
                try:
                    elapsed = (datetime.now() - job.started_at).total_seconds()
                    estimated_total = elapsed / job.progress
                    remaining = estimated_total - elapsed
                    if remaining > 0:
                        completion_time = datetime.now().timestamp() + remaining
                        job_dict["estimated_completion"] = datetime.fromtimestamp(completion_time).isoformat()
                except Exception:
                    pass  # Skip if calculation fails
            
            job_list.append(job_dict)
        
        return {
            "jobs": job_list,
            "total_count": len(job_list),
            "filters": {
                "job_type": job_type,
                "status": status,
                "active_only": active_only
            }
        }
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error listing jobs for user {current_user.id}: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Job listing service error"
        )


@app.get("/api/jobs/{job_id}", tags=["Jobs"])
async def get_job(
    job_id: str,
    current_user: User = Depends(get_current_user)
):
    """
    Get specific job details.
    
    Args:
        job_id: Job identifier
        current_user: Current authenticated user
    
    Returns:
        dict: Job details
    
    Raises:
        HTTPException: If job not found or access denied
    """
    try:
        job = job_manager.get_job(job_id)
        
        if not job:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Job not found"
            )
        
        # Verify job ownership
        if job.user_id != current_user.id:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Access denied to job"
            )
        
        job_dict = job.to_dict()
        
        # Add additional computed fields
        if job.status == JobStatus.PROCESSING and job.progress > 0:
            try:
                elapsed = (datetime.now() - job.started_at).total_seconds()
                estimated_total = elapsed / job.progress
                remaining = estimated_total - elapsed
                if remaining > 0:
                    completion_time = datetime.now().timestamp() + remaining
                    job_dict["estimated_completion"] = datetime.fromtimestamp(completion_time).isoformat()
            except Exception:
                pass
        
        # Add duration for completed jobs
        duration = job.get_duration()
        if duration is not None:
            job_dict["duration_seconds"] = duration
        
        return job_dict
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting job {job_id} for user {current_user.id}: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Job retrieval service error"
        )


@app.post("/api/jobs/{job_id}/cancel", tags=["Jobs"])
async def cancel_job(
    job_id: str,
    current_user: User = Depends(get_current_user),
    rate_limit: None = Depends(rate_limiter.create_rate_limiter(10, 300))  # 10 cancellations per 5 minutes
):
    """
    Cancel a job.
    
    Args:
        job_id: Job identifier
        current_user: Current authenticated user
        rate_limit: Rate limiting dependency
    
    Returns:
        dict: Cancellation confirmation
    
    Raises:
        HTTPException: If job not found, access denied, or cancellation fails
    """
    try:
        job = job_manager.get_job(job_id)
        
        if not job:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Job not found"
            )
        
        # Verify job ownership
        if job.user_id != current_user.id:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Access denied to job"
            )
        
        # Check if job can be cancelled
        if job.status not in [JobStatus.PENDING, JobStatus.PROCESSING]:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Cannot cancel job with status: {job.status.value}"
            )
        
        # Cancel the job
        success = job_manager.cancel_job(job_id, reason="Cancelled by user")
        
        if not success:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Failed to cancel job"
            )
        
        # Try to cancel the Celery task if it exists
        celery_task_id = job.metadata.get("celery_task_id")
        if celery_task_id:
            try:
                celery_app.control.revoke(celery_task_id, terminate=True)
                logger.info(f"Revoked Celery task {celery_task_id} for job {job_id}")
            except Exception as e:
                logger.warning(f"Failed to revoke Celery task {celery_task_id}: {e}")
        
        logger.info(f"Cancelled job {job_id} for user {current_user.id}")
        
        return {
            "message": "Job cancelled successfully",
            "job_id": job_id,
            "timestamp": datetime.utcnow().isoformat() + "Z"
        }
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error cancelling job {job_id} for user {current_user.id}: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Job cancellation service error"
        )


@app.get("/api/jobs/status/summary", tags=["Jobs"])
async def get_job_status_summary(
    current_user: User = Depends(get_current_user)
):
    """
    Get a summary of user's job status including limits and recommendations.
    
    Args:
        current_user: Current authenticated user
    
    Returns:
        dict: Job status summary with recommendations
    """
    try:
        # Get user statistics
        stats = job_manager.get_job_statistics(current_user.id)
        active_jobs = job_manager.get_user_jobs(current_user.id, active_only=True)
        
        # Check for stuck jobs
        from datetime import timedelta
        now = datetime.now()
        stuck_jobs = []
        old_pending = []
        
        for job in active_jobs:
            if job.status == JobStatus.PROCESSING and job.started_at:
                duration = now - job.started_at
                if duration > timedelta(hours=2):
                    stuck_jobs.append({
                        "job_id": job.job_id,
                        "type": job.job_type.value,
                        "duration_hours": duration.total_seconds() / 3600,
                        "started_at": job.started_at.isoformat()
                    })
            elif job.status == JobStatus.PENDING:
                age = now - job.created_at
                if age > timedelta(hours=1):
                    old_pending.append({
                        "job_id": job.job_id,
                        "type": job.job_type.value,
                        "age_hours": age.total_seconds() / 3600,
                        "created_at": job.created_at.isoformat()
                    })
        
        # Generate recommendations
        recommendations = []
        if len(stuck_jobs) > 0:
            recommendations.append({
                "type": "warning",
                "message": f"You have {len(stuck_jobs)} stuck processing jobs that may need to be cancelled.",
                "action": "Consider cancelling stuck jobs using POST /api/jobs/{job_id}/cancel"
            })
        
        if len(old_pending) > 0:
            recommendations.append({
                "type": "info", 
                "message": f"You have {len(old_pending)} old pending jobs that may be stuck in queue.",
                "action": "These jobs might start processing soon, or you can cancel them if not needed."
            })
        
        if len(active_jobs) >= 8:  # Approaching limit
            recommendations.append({
                "type": "warning",
                "message": f"You are approaching the maximum concurrent jobs limit ({len(active_jobs)}/10).",
                "action": "Consider waiting for some jobs to complete before uploading more files."
            })
        
        return {
            "user_id": current_user.id,
            "job_limits": {
                "max_concurrent_jobs": 10,
                "current_active_jobs": len(active_jobs),
                "remaining_slots": max(0, 10 - len(active_jobs))
            },
            "job_statistics": stats,
            "stuck_jobs": stuck_jobs,
            "old_pending_jobs": old_pending,
            "recommendations": recommendations,
            "can_upload": len(active_jobs) < 10,
            "timestamp": datetime.utcnow().isoformat() + "Z"
        }
    
    except Exception as e:
        logger.error(f"Error getting job status summary for user {current_user.id}: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Job status summary service error"
        )


@app.post("/api/jobs/cleanup/stuck", tags=["Jobs"])
async def cleanup_stuck_jobs_for_user(
    current_user: User = Depends(get_current_user),
    rate_limit: None = Depends(rate_limiter.create_rate_limiter(3, 3600))  # 3 cleanups per hour
):
    """
    Clean up stuck jobs for the current user.
    
    Args:
        current_user: Current authenticated user
        rate_limit: Rate limiting dependency
    
    Returns:
        dict: Cleanup results
    """
    try:
        from datetime import timedelta
        
        active_jobs = job_manager.get_user_jobs(current_user.id, active_only=True)
        now = datetime.now()
        cleaned_jobs = []
        
        # Clean up stuck processing jobs (> 2 hours)
        for job in active_jobs:
            if job.status == JobStatus.PROCESSING and job.started_at:
                duration = now - job.started_at
                if duration > timedelta(hours=2):
                    success = job_manager.cancel_job(
                        job.job_id,
                        f"Cancelled by user cleanup - stuck for {duration}"
                    )
                    if success:
                        cleaned_jobs.append({
                            "job_id": job.job_id,
                            "type": job.job_type.value,
                            "reason": "stuck_processing",
                            "duration_hours": duration.total_seconds() / 3600
                        })
        
        # Clean up old pending jobs (> 1 hour)
        for job in active_jobs:
            if job.status == JobStatus.PENDING:
                age = now - job.created_at
                if age > timedelta(hours=1):
                    success = job_manager.cancel_job(
                        job.job_id,
                        f"Cancelled by user cleanup - pending for {age}"
                    )
                    if success:
                        cleaned_jobs.append({
                            "job_id": job.job_id,
                            "type": job.job_type.value,
                            "reason": "old_pending",
                            "age_hours": age.total_seconds() / 3600
                        })
        
        message = f"Cleaned up {len(cleaned_jobs)} stuck jobs" if cleaned_jobs else "No stuck jobs found"
        
        logger.info(f"User {current_user.id} cleaned up {len(cleaned_jobs)} stuck jobs")
        
        return {
            "message": message,
            "cleaned_jobs_count": len(cleaned_jobs),
            "cleaned_jobs": cleaned_jobs,
            "timestamp": datetime.utcnow().isoformat() + "Z"
        }
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error cleaning up stuck jobs for user {current_user.id}: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Job cleanup service error"
        )




@app.get("/api/jobs/{job_id}", tags=["Jobs"])
async def get_job(
    job_id: str,
    current_user: User = Depends(get_current_user)
):
    """
    Get specific job details.
    
    Args:
        job_id: Job identifier
        current_user: Current authenticated user
    
    Returns:
        dict: Job details
    
    Raises:
        HTTPException: If job not found or access denied
    """
    try:
        job = job_manager.get_job(job_id)
        
        if not job:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Job not found"
            )
        
        # Verify job ownership
        if job.user_id != current_user.id:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Access denied to job"
            )
        
        job_dict = job.to_dict()
        
        # Add additional computed fields
        if job.status == JobStatus.PROCESSING and job.progress > 0:
            try:
                elapsed = (datetime.now() - job.started_at).total_seconds()
                estimated_total = elapsed / job.progress
                remaining = estimated_total - elapsed
                if remaining > 0:
                    completion_time = datetime.now().timestamp() + remaining
                    job_dict["estimated_completion"] = datetime.fromtimestamp(completion_time).isoformat()
            except Exception:
                pass
        
        # Add duration for completed jobs
        duration = job.get_duration()
        if duration is not None:
            job_dict["duration_seconds"] = duration
        
        return job_dict
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting job {job_id} for user {current_user.id}: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Job retrieval service error"
        )


@app.post("/api/jobs/{job_id}/cancel", tags=["Jobs"])
async def cancel_job(
    job_id: str,
    current_user: User = Depends(get_current_user),
    rate_limit: None = Depends(rate_limiter.create_rate_limiter(10, 300))  # 10 cancellations per 5 minutes
):
    """
    Cancel a job.
    
    Args:
        job_id: Job identifier
        current_user: Current authenticated user
        rate_limit: Rate limiting dependency
    
    Returns:
        dict: Cancellation confirmation
    
    Raises:
        HTTPException: If job not found, access denied, or cancellation fails
    """
    try:
        job = job_manager.get_job(job_id)
        
        if not job:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Job not found"
            )
        
        # Verify job ownership
        if job.user_id != current_user.id:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Access denied to job"
            )
        
        # Check if job can be cancelled
        if job.status not in [JobStatus.PENDING, JobStatus.PROCESSING]:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Cannot cancel job with status: {job.status.value}"
            )
        
        # Cancel the job
        success = job_manager.cancel_job(job_id, reason="Cancelled by user")
        
        if not success:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Failed to cancel job"
            )
        
        # Try to cancel the Celery task if it exists
        celery_task_id = job.metadata.get("celery_task_id")
        if celery_task_id:
            try:
                celery_app.control.revoke(celery_task_id, terminate=True)
                logger.info(f"Revoked Celery task {celery_task_id} for job {job_id}")
            except Exception as e:
                logger.warning(f"Failed to revoke Celery task {celery_task_id}: {e}")
        
        logger.info(f"Cancelled job {job_id} for user {current_user.id}")
        
        return {
            "message": "Job cancelled successfully",
            "job_id": job_id,
            "timestamp": datetime.utcnow().isoformat() + "Z"
        }
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error cancelling job {job_id} for user {current_user.id}: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Job cancellation service error"
        )


@app.delete("/api/jobs/{job_id}", tags=["Jobs"])
async def delete_job(
    job_id: str,
    current_user: User = Depends(get_current_user),
    rate_limit: None = Depends(rate_limiter.create_rate_limiter(20, 300))  # 20 deletions per 5 minutes
):
    """
    Delete a completed job.
    
    Args:
        job_id: Job identifier
        current_user: Current authenticated user
        rate_limit: Rate limiting dependency
    
    Returns:
        dict: Deletion confirmation
    
    Raises:
        HTTPException: If job not found, access denied, or deletion fails
    """
    try:
        job = job_manager.get_job(job_id)
        
        if not job:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Job not found"
            )
        
        # Verify job ownership
        if job.user_id != current_user.id:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Access denied to job"
            )
        
        # Check if job can be deleted (only finished jobs)
        if not job.is_finished():
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Cannot delete active job with status: {job.status.value}"
            )
        
        # Delete the job
        success = job_manager.delete_job(job_id)
        
        if not success:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Failed to delete job"
            )
        
        logger.info(f"Deleted job {job_id} for user {current_user.id}")
        
        return {
            "message": "Job deleted successfully",
            "job_id": job_id,
            "timestamp": datetime.utcnow().isoformat() + "Z"
        }
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error deleting job {job_id} for user {current_user.id}: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Job deletion service error"
        )


@app.get("/api/jobs/stats", tags=["Jobs"])
async def get_job_statistics(
    current_user: User = Depends(get_current_user)
):
    """
    Get job statistics for the current user.
    
    Args:
        current_user: Current authenticated user
    
    Returns:
        dict: Job statistics
    """
    try:
        stats = job_manager.get_job_statistics(user_id=current_user.id)
        
        return {
            "user_id": current_user.id,
            "statistics": stats,
            "timestamp": datetime.utcnow().isoformat() + "Z"
        }
    
    except Exception as e:
        logger.error(f"Error getting job statistics for user {current_user.id}: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Job statistics service error"
        )


# WebSocket endpoints
from fastapi import WebSocket, WebSocketDisconnect, Query
from ..shared.websocket_manager import websocket_manager, WebSocketAuthenticationError, WebSocketConnectionError

@app.websocket("/ws/updates")
async def websocket_endpoint(
    websocket: WebSocket,
    token: str = Query(..., description="JWT authentication token")
):
    """
    WebSocket endpoint for real-time updates.
    
    Args:
        websocket: WebSocket connection
        token: JWT authentication token
    
    This endpoint provides real-time updates for:
    - Job status changes
    - Document processing progress
    - System notifications
    - User-specific messages
    """
    connection = None
    
    try:
        # Authenticate and establish connection
        connection = await websocket_manager.connect(websocket, token)
        
        logger.info(f"WebSocket connection established for user {connection.user.id}")
        
        # Handle incoming messages
        while True:
            try:
                # Receive message from client
                message = await websocket.receive_text()
                
                # Handle the message
                await websocket_manager.handle_message(connection.connection_id, message)
                
            except WebSocketDisconnect:
                logger.info(f"WebSocket client disconnected: {connection.connection_id}")
                break
            
            except Exception as e:
                logger.error(f"Error handling WebSocket message: {e}")
                # Send error to client but continue connection
                if connection:
                    await connection.send_error("MESSAGE_ERROR", "Error processing message")
    
    except WebSocketAuthenticationError as e:
        logger.warning(f"WebSocket authentication failed: {e}")
        # Connection will be closed by the manager
    
    except WebSocketConnectionError as e:
        logger.error(f"WebSocket connection error: {e}")
    
    except Exception as e:
        logger.error(f"Unexpected WebSocket error: {e}")
    
    finally:
        # Clean up connection
        if connection:
            await websocket_manager.disconnect(connection.connection_id)


@app.get("/api/websocket/stats", tags=["WebSocket"])
async def get_websocket_stats(
    current_user: User = Depends(require_roles(["admin"]))
):
    """
    Get WebSocket connection statistics (admin only).
    
    Args:
        current_user: Current authenticated user (must have admin permission)
    """
    try:
        stats = websocket_manager.get_connection_stats()
        return {
            "websocket_stats": stats,
            "timestamp": datetime.utcnow().isoformat() + "Z"
        }
    
    except Exception as e:
        logger.error(f"Error getting WebSocket stats: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="WebSocket stats service error"
        )


@app.get("/api/websocket/health", tags=["WebSocket"])
async def websocket_health_check():
    """
    WebSocket health check endpoint.
    
    Returns:
        dict: WebSocket health status
    """
    try:
        health = websocket_manager.health_check()
        
        status_code = 200 if health.get("websocket_manager", False) else 503
        
        return JSONResponse(
            status_code=status_code,
            content={
                "websocket_health": health,
                "timestamp": datetime.utcnow().isoformat() + "Z"
            }
        )
    
    except Exception as e:
        logger.error(f"WebSocket health check failed: {e}")
        return JSONResponse(
            status_code=503,
            content={
                "error": {
                    "code": "HEALTH_CHECK_FAILED",
                    "message": "WebSocket health check failed",
                    "details": str(e)
                },
                "timestamp": datetime.utcnow().isoformat() + "Z"
            }
        )


# User management endpoints (basic implementation)
@app.post("/api/auth/register", tags=["Authentication"])
async def register_user(
    request: Request,
    registration_data: dict,
    rate_limit: None = Depends(rate_limiter.create_rate_limiter(3, 3600))  # 3 registrations per hour
):
    """
    Register new user (basic implementation).
    
    Note: This is a basic implementation for development.
    In production, implement proper user registration with email verification,
    password strength requirements, and database storage.
    
    Args:
        request: FastAPI request object
        registration_data: User registration data
        rate_limit: Rate limiting dependency
    
    Returns:
        dict: Registration result
    """
    try:
        # Basic validation
        required_fields = ["username", "password", "groups"]
        for field in required_fields:
            if field not in registration_data:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail=f"Missing required field: {field}"
                )
        
        username = registration_data["username"]
        password = registration_data["password"]
        groups = registration_data["groups"]
        
        # Check if user already exists
        if username in config.USERS:
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail="Username already exists"
            )
        
        # Basic password validation
        if len(password) < 8:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Password must be at least 8 characters long"
            )
        
        # Validate groups
        valid_groups = {"assistance", "common_rules", "admin"}
        for group in groups:
            if group not in valid_groups:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail=f"Invalid group: {group}"
                )
        
        # Add user to config (in production, save to database)
        config.USERS[username] = {
            "password": password,
            "groups": groups
        }
        
        logger.info(f"User {username} registered successfully")
        
        return {
            "message": "User registered successfully",
            "username": username,
            "groups": groups,
            "timestamp": datetime.utcnow().isoformat() + "Z"
        }
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Registration error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Registration service error"
        )


@app.post("/api/auth/change-password", tags=["Authentication"])
async def change_password(
    request: Request,
    password_data: dict,
    current_user: User = Depends(get_current_user),
    rate_limit: None = Depends(rate_limiter.create_rate_limiter(5, 3600))  # 5 changes per hour
):
    """
    Change user password.
    
    Args:
        request: FastAPI request object
        password_data: Password change data
        current_user: Current authenticated user
        rate_limit: Rate limiting dependency
    
    Returns:
        dict: Password change result
    """
    try:
        # Validate input
        if "current_password" not in password_data or "new_password" not in password_data:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Missing current_password or new_password"
            )
        
        current_password = password_data["current_password"]
        new_password = password_data["new_password"]
        
        # Verify current password
        user_data = config.USERS.get(current_user.id)
        if not user_data or user_data["password"] != current_password:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Current password is incorrect"
            )
        
        # Validate new password
        if len(new_password) < 8:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="New password must be at least 8 characters long"
            )
        
        if new_password == current_password:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="New password must be different from current password"
            )
        
        # Update password (in production, hash and save to database)
        config.USERS[current_user.id]["password"] = new_password
        
        # In a Keycloak setup, password changes should be handled via Keycloak APIs.
        logger.info(f"Password changed for user {current_user.id}")
        
        return {
            "message": "Password changed successfully. Please log in again.",
            "timestamp": datetime.utcnow().isoformat() + "Z"
        }
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Password change error for user {current_user.id}: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Password change service error"
        )


# Query processing endpoints
from ..shared.models import QueryRequest, QueryResponse, Query, JobStatus
from ..workers.query_worker import process_user_query

# Import and include query history endpoints
from .query_history_endpoints import router as query_history_router


@app.post("/api/query", tags=["Query"])
async def submit_query(
    request: Request,
    query_data: QueryRequest,
    current_user: User = Depends(get_current_user),
    rate_limit: None = Depends(rate_limiter.create_rate_limiter(30, 300))  # 30 queries per 5 minutes
):
    """
    Submit a query for processing with user context validation.
    
    Args:
        request: FastAPI request object
        query_data: Query request data
        current_user: Current authenticated user
        rate_limit: Rate limiting dependency
    
    Returns:
        dict: Query submission result with query ID and status
    
    Raises:
        HTTPException: If query submission fails
    """
    try:
        # Validate permissions
        if not current_user.has_permission("query"):
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="User does not have query permission"
            )
        
        # Validate query text
        if not query_data.query_text or not query_data.query_text.strip():
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Query text cannot be empty"
            )
        
        # Check query length
        if len(query_data.query_text) > 2000:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Query text too long (maximum 2000 characters)"
            )
        
        # Create query record
        query = Query(
            user_id=current_user.id,
            query_text=query_data.query_text.strip(),
            status="pending"
        )
        
        # Store query in Redis
        query_key = f"query:{query.query_id}"
        redis_client.set_json(query_key, query.to_dict(), expire_seconds=3600)  # 1 hour expiration
        
        # Create job for tracking
        job = job_manager.create_job(
            user_id=current_user.id,
            job_type=JobType.QUERY,
            metadata={
                "query_id": query.query_id,
                "query_text": query_data.query_text.strip(),
                "groups": current_user.groups,
                "query_source": "api"
            }
        )
        
        # Queue query processing task
        task = process_user_query.delay(
            query.query_id,
            current_user.id,
            current_user.groups,
            query_data.query_text.strip()
        )
        
        # Update job with task ID
        job.metadata["celery_task_id"] = task.id
        redis_client.set_job(job)
        
        # Update query with job ID
        query_dict = query.to_dict()
        query_dict["job_id"] = job.job_id
        query_dict["task_id"] = task.id
        redis_client.set_json(query_key, query_dict, expire_seconds=3600)
        
        logger.info(f"Created query {query.query_id} for user {current_user.id}")
        
        return {
            "query_id": query.query_id,
            "job_id": job.job_id,
            "message": "Query submitted successfully",
            "status": "pending",
            "estimated_processing_time": "5-30 seconds"
        }
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Query submission error for user {current_user.id}: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Query submission service error"
        )


@app.get("/api/query/{query_id}", tags=["Query"])
async def get_query_result(
    query_id: str,
    current_user: User = Depends(get_current_user)
):
    """
    Get query result and status.
    
    Args:
        query_id: Query identifier
        current_user: Current authenticated user
    
    Returns:
        dict: Query result with answer, sources, and metadata
    
    Raises:
        HTTPException: If query not found or access denied
    """
    try:
        # Get query data from Redis
        query_key = f"query:{query_id}"
        query_data = redis_client.get_json(query_key)
        
        if not query_data:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Query not found"
            )
        
        # Validate user ownership
        if query_data.get("user_id") != current_user.id:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Access denied to this query"
            )
        
        # Get associated job information if available
        job_info = None
        if "job_id" in query_data:
            try:
                job = job_manager.get_job(query_data["job_id"])
                if job:
                    job_info = {
                        "job_id": job.job_id,
                        "status": job.status.value,
                        "progress": job.progress,
                        "created_at": job.created_at.isoformat() + "Z",
                        "started_at": job.started_at.isoformat() + "Z" if job.started_at else None,
                        "completed_at": job.completed_at.isoformat() + "Z" if job.completed_at else None,
                        "error": job.error
                    }
            except Exception as e:
                logger.warning(f"Failed to get job info for query {query_id}: {e}")
        
        # Prepare response
        response_data = {
            "query_id": query_id,
            "query_text": query_data.get("query_text"),
            "status": query_data.get("status", "pending"),
            "created_at": query_data.get("created_at"),
            "processing_time": query_data.get("processing_time"),
            "job_info": job_info
        }
        
        # Add result data if query is completed
        if query_data.get("status") == "completed" and "result" in query_data:
            result = query_data["result"]
            response_data.update({
                "answer": result.get("answer"),
                "sources": result.get("sources", []),
                "result_count": result.get("result_count", 0),
                "cached": result.get("cached", False),
                "query_metadata": result.get("query_metadata", {})
            })
        
        # Add error information if query failed
        if query_data.get("status") == "failed":
            response_data["error"] = query_data.get("error")
            response_data["error_type"] = query_data.get("error_type")
        
        return response_data
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting query result {query_id} for user {current_user.id}: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Query retrieval service error"
        )


@app.get("/api/query/{query_id}/status", tags=["Query"])
async def get_query_status(
    query_id: str,
    current_user: User = Depends(get_current_user)
):
    """
    Get query processing status and progress.
    
    Args:
        query_id: Query identifier
        current_user: Current authenticated user
    
    Returns:
        dict: Query status and progress information
    
    Raises:
        HTTPException: If query not found or access denied
    """
    try:
        # Get query data from Redis
        query_key = f"query:{query_id}"
        query_data = redis_client.get_json(query_key)
        
        if not query_data:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Query not found"
            )
        
        # Validate user ownership
        if query_data.get("user_id") != current_user.id:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Access denied to this query"
            )
        
        # Get Celery task status if available
        task_status = None
        if "task_id" in query_data:
            try:
                from celery.result import AsyncResult
                task_result = AsyncResult(query_data["task_id"], app=celery_app)
                task_status = {
                    "task_id": query_data["task_id"],
                    "state": task_result.state,
                    "info": task_result.info if task_result.info else {}
                }
            except Exception as e:
                logger.warning(f"Failed to get task status for query {query_id}: {e}")
        
        return {
            "query_id": query_id,
            "status": query_data.get("status", "pending"),
            "progress": query_data.get("progress", 0.0),
            "status_message": query_data.get("status_message"),
            "created_at": query_data.get("created_at"),
            "started_at": query_data.get("started_at"),
            "completed_at": query_data.get("completed_at"),
            "processing_time": query_data.get("processing_time"),
            "error": query_data.get("error"),
            "task_status": task_status,
            "last_updated": query_data.get("last_updated")
        }
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting query status {query_id} for user {current_user.id}: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Query status service error"
        )


@app.get("/api/queries", tags=["Query"])
async def list_user_queries(
    request: Request,
    status: Optional[str] = None,
    limit: int = 50,
    offset: int = 0,
    current_user: User = Depends(get_current_user)
):
    """
    List user's query history with filtering and pagination.
    
    Args:
        request: FastAPI request object
        status: Optional status filter (pending, processing, completed, failed)
        limit: Maximum number of queries to return
        offset: Number of queries to skip
        current_user: Current authenticated user
    
    Returns:
        dict: List of queries with metadata
    """
    try:
        # Validate status filter
        if status and status not in ["pending", "processing", "completed", "failed"]:
            raise HTTPException(
                status_code=400,
                detail="Invalid status filter. Must be one of: pending, processing, completed, failed"
            )
        
        # Validate pagination parameters
        if limit < 1 or limit > 100:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Limit must be between 1 and 100"
            )
        
        if offset < 0:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Offset must be non-negative"
            )
        
        # Get user's queries from Redis
        queries = []
        pattern = f"query:*"
        
        with redis_client.get_connection() as client:
            keys = client.keys(pattern)
        
        for key in keys:
            try:
                query_data = redis_client.get_json(key.decode('utf-8'))
                if query_data and query_data.get("user_id") == current_user.id:
                    # Apply status filter if specified
                    if status and query_data.get("status") != status:
                        continue
                    
                    # Add query to results
                    query_summary = {
                        "query_id": query_data.get("query_id"),
                        "query_text": query_data.get("query_text", "")[:100] + "..." if len(query_data.get("query_text", "")) > 100 else query_data.get("query_text", ""),
                        "status": query_data.get("status", "pending"),
                        "created_at": query_data.get("created_at"),
                        "completed_at": query_data.get("completed_at"),
                        "processing_time": query_data.get("processing_time"),
                        "result_count": query_data.get("result", {}).get("result_count", 0) if query_data.get("result") else 0,
                        "cached": query_data.get("result", {}).get("cached", False) if query_data.get("result") else False,
                        "error": query_data.get("error")
                    }
                    queries.append(query_summary)
                    
            except Exception as e:
                logger.warning(f"Failed to parse query data from key {key}: {e}")
                continue
        
        # Sort by creation time (newest first)
        queries.sort(key=lambda x: x.get("created_at", ""), reverse=True)
        
        # Apply pagination
        total_count = len(queries)
        paginated_queries = queries[offset:offset + limit]
        
        return {
            "queries": paginated_queries,
            "total_count": total_count,
            "limit": limit,
            "offset": offset,
            "has_more": offset + limit < total_count,
            "status_filter": status
        }
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error listing queries for user {current_user.id}: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Query listing service error"
        )


@app.delete("/api/query/{query_id}", tags=["Query"])
async def delete_query(
    query_id: str,
    current_user: User = Depends(get_current_user),
    rate_limit: None = Depends(rate_limiter.create_rate_limiter(50, 300))  # 50 deletions per 5 minutes
):
    """
    Delete a query and its associated data.
    
    Args:
        query_id: Query identifier
        current_user: Current authenticated user
        rate_limit: Rate limiting dependency
    
    Returns:
        dict: Deletion confirmation
    
    Raises:
        HTTPException: If query not found or deletion fails
    """
    try:
        # Get query data from Redis
        query_key = f"query:{query_id}"
        query_data = redis_client.get_json(query_key)
        
        if not query_data:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Query not found"
            )
        
        # Validate user ownership
        if query_data.get("user_id") != current_user.id:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Access denied to this query"
            )
        
        # Cancel Celery task if still running
        if "task_id" in query_data and query_data.get("status") in ["pending", "processing"]:
            try:
                from celery.result import AsyncResult
                task_result = AsyncResult(query_data["task_id"], app=celery_app)
                task_result.revoke(terminate=True)
                logger.info(f"Cancelled Celery task {query_data['task_id']} for query {query_id}")
            except Exception as e:
                logger.warning(f"Failed to cancel Celery task for query {query_id}: {e}")
        
        # Delete associated job if exists
        if "job_id" in query_data:
            try:
                job = job_manager.get_job(query_data["job_id"])
                if job:
                    job.update_status(JobStatus.CANCELLED)
                    redis_client.set_job(job)
            except Exception as e:
                logger.warning(f"Failed to cancel job for query {query_id}: {e}")
        
        # Delete query from Redis
        if not redis_client.delete(query_key):
            logger.warning(f"Failed to delete query data for {query_id}")
        
        logger.info(f"Deleted query {query_id} for user {current_user.id}")
        
        return {
            "message": "Query deleted successfully",
            "query_id": query_id,
            "query_text": query_data.get("query_text", "")[:50] + "..." if len(query_data.get("query_text", "")) > 50 else query_data.get("query_text", ""),
            "timestamp": datetime.utcnow().isoformat() + "Z"
        }
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error deleting query {query_id} for user {current_user.id}: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Query deletion service error"
        )


@app.get("/api/query/{query_id}/cache", tags=["Query"])
async def get_query_cache_info(
    query_id: str,
    current_user: User = Depends(get_current_user)
):
    """
    Get query cache information and statistics.
    
    Args:
        query_id: Query identifier
        current_user: Current authenticated user
    
    Returns:
        dict: Cache information and statistics
    """
    try:
        # Get query data from Redis
        query_key = f"query:{query_id}"
        query_data = redis_client.get_json(query_key)
        
        if not query_data:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Query not found"
            )
        
        # Validate user ownership
        if query_data.get("user_id") != current_user.id:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Access denied to this query"
            )
        
        # Generate cache key for this query
        from ..workers.query_worker import generate_cache_key
        cache_key = generate_cache_key(
            current_user.id,
            current_user.groups,
            query_data.get("query_text", "")
        )
        
        # Check if result was cached
        cached_result = query_data.get("result", {}).get("cached", False)
        
        # Get cache statistics
        cache_info = {
            "query_id": query_id,
            "cache_key": cache_key,
            "was_cached": cached_result,
            "cache_available": False,
            "cache_expires_at": None,
            "cache_created_at": None
        }
        
        # Check current cache status
        try:
            cached_data = redis_client.get_json(cache_key)
            if cached_data:
                cache_info.update({
                    "cache_available": True,
                    "cache_expires_at": datetime.fromtimestamp(cached_data.get("expires_at", 0)).isoformat() + "Z" if cached_data.get("expires_at") else None,
                    "cache_created_at": datetime.fromtimestamp(cached_data.get("cached_at", 0)).isoformat() + "Z" if cached_data.get("cached_at") else None
                })
        except Exception as e:
            logger.warning(f"Failed to check cache status for query {query_id}: {e}")
        
        return cache_info
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting cache info for query {query_id}: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Cache info service error"
        )


# Include query history router
app.include_router(query_history_router)

# WebSocket endpoint for real-time updates
from fastapi import WebSocket, WebSocketDisconnect
from ..shared.websocket_manager import websocket_manager



if __name__ == "__main__":
    uvicorn.run(
        "app.api.main:app",
        host="0.0.0.0",
        port=8000
    )