"""
FastAPI application main module.
"""
from fastapi import FastAPI, HTTPException, Depends, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import HTTPBearer
from contextlib import asynccontextmanager
import uvicorn

from ..shared.config import config
from ..shared.redis_client import redis_client


# Security
security = HTTPBearer()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan events."""
    # Startup
    print("Starting FastAPI application...")
    
    # Check Redis connection
    if not redis_client.health_check():
        print("Warning: Redis connection failed")
    else:
        print("Redis connection successful")
    
    yield
    
    # Shutdown
    print("Shutting down FastAPI application...")


# Create FastAPI app
app = FastAPI(
    title="Concurrent RAG System",
    description="Multi-user RAG system with concurrent processing",
    version="1.0.0",
    lifespan=lifespan
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
async def root():
    """Root endpoint."""
    return {"message": "Concurrent RAG System API", "version": "1.0.0"}


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    redis_healthy = redis_client.health_check()
    
    return {
        "status": "healthy" if redis_healthy else "unhealthy",
        "redis": redis_healthy,
        "timestamp": "2025-01-21T10:30:00Z"  # TODO: Use actual timestamp
    }


# TODO: Add authentication, document, query, and WebSocket endpoints in subsequent tasks


if __name__ == "__main__":
    uvicorn.run(
        "app.api.main:app",
        host="0.0.0.0",
        port=8000,
        reload=True
    )