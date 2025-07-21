"""
Shared data models for the concurrent RAG system.
"""
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import Dict, List, Optional, Any
from pydantic import BaseModel


class JobStatus(str, Enum):
    """Job status enumeration."""
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass
class UserSession:
    """User session data model."""
    session_id: str
    user_id: str
    groups: List[str]
    created_at: datetime
    last_activity: datetime
    permissions: List[str]
    is_active: bool = True


@dataclass
class Job:
    """Job data model for tracking background tasks."""
    job_id: str
    user_id: str
    job_type: str  # "embedding" or "query"
    status: JobStatus
    created_at: datetime
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    progress: float = 0.0  # 0.0 to 1.0
    result: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    metadata: Dict[str, Any] = None

    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


@dataclass
class Document:
    """Document data model."""
    document_id: str
    user_id: str
    group_id: str
    filename: str
    file_size: int
    upload_date: datetime
    processing_status: str
    page_count: Optional[int] = None
    chunk_count: Optional[int] = None


@dataclass
class Query:
    """Query data model."""
    query_id: str
    user_id: str
    query_text: str
    created_at: datetime
    processing_time: Optional[float] = None
    result_count: Optional[int] = None
    status: str = "pending"


# Pydantic models for API requests/responses
class LoginRequest(BaseModel):
    """Login request model."""
    username: str
    password: str


class LoginResponse(BaseModel):
    """Login response model."""
    access_token: str
    token_type: str = "bearer"
    user_id: str
    groups: List[str]


class DocumentUploadResponse(BaseModel):
    """Document upload response model."""
    job_id: str
    message: str
    files_count: int


class QueryRequest(BaseModel):
    """Query request model."""
    query_text: str


class QueryResponse(BaseModel):
    """Query response model."""
    query_id: str
    answer: str
    sources: List[str]
    processing_time: float


class JobResponse(BaseModel):
    """Job response model."""
    job_id: str
    job_type: str
    status: JobStatus
    progress: float
    created_at: datetime
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    result: Optional[Dict[str, Any]] = None
    error: Optional[str] = None


class ErrorResponse(BaseModel):
    """Error response model."""
    error: Dict[str, Any]