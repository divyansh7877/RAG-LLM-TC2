"""
Shared data models for the concurrent RAG system.
"""
import json
from datetime import datetime
from enum import Enum
from typing import Dict, List, Optional, Any, Union
from pydantic import BaseModel, Field, validator
import uuid


class JobStatus(str, Enum):
    """Job status enumeration."""
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class JobType(str, Enum):
    """Job type enumeration."""
    EMBEDDING = "embedding"
    QUERY = "query"


class RedisSerializable(BaseModel):
    """Base class for models that can be serialized to/from Redis."""
    
    def to_redis(self) -> str:
        """Serialize model to JSON string for Redis storage."""
        return self.json()
    
    @classmethod
    def from_redis(cls, data: Union[str, bytes]) -> 'RedisSerializable':
        """Deserialize model from Redis JSON string."""
        if isinstance(data, bytes):
            data = data.decode('utf-8')
        return cls.parse_raw(data)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert model to dictionary."""
        return self.dict()
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'RedisSerializable':
        """Create model from dictionary."""
        return cls.parse_obj(data)


class UserSession(RedisSerializable):
    """User session data model with Redis serialization support."""
    session_id: str = Field(default_factory=lambda: str(uuid.uuid4()), description="Unique session identifier")
    user_id: str = Field(..., description="User identifier")
    groups: List[str] = Field(default_factory=list, description="User groups")
    created_at: datetime = Field(default_factory=datetime.now, description="Session creation time")
    last_activity: datetime = Field(default_factory=datetime.now, description="Last activity timestamp")
    permissions: List[str] = Field(default_factory=list, description="User permissions")
    is_active: bool = Field(default=True, description="Session active status")
    
    @validator('permissions')
    def validate_permissions(cls, v):
        """Validate permissions list."""
        valid_permissions = {'upload', 'query', 'delete', 'admin'}
        for perm in v:
            if perm not in valid_permissions:
                raise ValueError(f"Invalid permission: {perm}")
        return v
    
    def update_activity(self):
        """Update last activity timestamp."""
        self.last_activity = datetime.now()
    
    def has_permission(self, permission: str) -> bool:
        """Check if session has specific permission."""
        return permission in self.permissions
    
    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }


class Job(RedisSerializable):
    """Job data model for tracking background tasks with Redis serialization."""
    job_id: str = Field(default_factory=lambda: str(uuid.uuid4()), description="Unique job identifier")
    user_id: str = Field(..., description="User who created the job")
    job_type: JobType = Field(..., description="Type of job")
    status: JobStatus = Field(default=JobStatus.PENDING, description="Current job status")
    created_at: datetime = Field(default_factory=datetime.now, description="Job creation time")
    started_at: Optional[datetime] = Field(None, description="Job start time")
    completed_at: Optional[datetime] = Field(None, description="Job completion time")
    progress: float = Field(default=0.0, ge=0.0, le=1.0, description="Job progress (0.0 to 1.0)")
    result: Optional[Dict[str, Any]] = Field(None, description="Job result data")
    error: Optional[str] = Field(None, description="Error message if job failed")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional job metadata")
    
    @validator('progress')
    def validate_progress(cls, v):
        """Validate progress is between 0.0 and 1.0."""
        if not 0.0 <= v <= 1.0:
            raise ValueError("Progress must be between 0.0 and 1.0")
        return v
    
    def update_status(self, status: JobStatus, error: Optional[str] = None):
        """Update job status and set timestamps."""
        self.status = status
        if status == JobStatus.PROCESSING and not self.started_at:
            self.started_at = datetime.now()
        elif status in [JobStatus.COMPLETED, JobStatus.FAILED, JobStatus.CANCELLED]:
            self.completed_at = datetime.now()
        if error:
            self.error = error
    
    def update_progress(self, progress: float):
        """Update job progress."""
        if not 0.0 <= progress <= 1.0:
            raise ValueError("Progress must be between 0.0 and 1.0")
        self.progress = progress
    
    def is_finished(self) -> bool:
        """Check if job is in a finished state."""
        return self.status in [JobStatus.COMPLETED, JobStatus.FAILED, JobStatus.CANCELLED]
    
    def get_duration(self) -> Optional[float]:
        """Get job duration in seconds if completed."""
        if self.started_at and self.completed_at:
            return (self.completed_at - self.started_at).total_seconds()
        return None
    
    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }


class Document(RedisSerializable):
    """Document data model with Redis serialization support."""
    document_id: str = Field(default_factory=lambda: str(uuid.uuid4()), description="Unique document identifier")
    user_id: str = Field(..., description="Owner user ID")
    group_id: str = Field(..., description="Document group ID")
    filename: str = Field(..., description="Original filename")
    file_size: int = Field(..., ge=0, description="File size in bytes")
    upload_date: datetime = Field(default_factory=datetime.now, description="Upload timestamp")
    processing_status: str = Field(default="pending", description="Processing status")
    page_count: Optional[int] = Field(None, ge=0, description="Number of pages")
    chunk_count: Optional[int] = Field(None, ge=0, description="Number of chunks")
    file_hash: Optional[str] = Field(None, description="File content hash")
    content_type: Optional[str] = Field(None, description="MIME content type")
    
    @validator('processing_status')
    def validate_processing_status(cls, v):
        """Validate processing status."""
        valid_statuses = {'pending', 'processing', 'completed', 'failed'}
        if v not in valid_statuses:
            raise ValueError(f"Invalid processing status: {v}")
        return v
    
    @validator('filename')
    def validate_filename(cls, v):
        """Validate filename is not empty."""
        if not v or not v.strip():
            raise ValueError("Filename cannot be empty")
        return v.strip()
    
    def update_processing_status(self, status: str):
        """Update processing status with validation."""
        valid_statuses = {'pending', 'processing', 'completed', 'failed'}
        if status not in valid_statuses:
            raise ValueError(f"Invalid processing status: {status}")
        self.processing_status = status
    
    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }


class Query(RedisSerializable):
    """Query data model with Redis serialization support."""
    query_id: str = Field(default_factory=lambda: str(uuid.uuid4()), description="Unique query identifier")
    user_id: str = Field(..., description="User who made the query")
    query_text: str = Field(..., description="Query text")
    created_at: datetime = Field(default_factory=datetime.now, description="Query creation time")
    processing_time: Optional[float] = Field(None, ge=0, description="Processing time in seconds")
    result_count: Optional[int] = Field(None, ge=0, description="Number of results returned")
    status: str = Field(default="pending", description="Query status")
    response: Optional[str] = Field(None, description="Query response")
    sources: List[str] = Field(default_factory=list, description="Source documents")
    
    @validator('query_text')
    def validate_query_text(cls, v):
        """Validate query text is not empty."""
        if not v or not v.strip():
            raise ValueError("Query text cannot be empty")
        return v.strip()
    
    @validator('status')
    def validate_status(cls, v):
        """Validate query status."""
        valid_statuses = {'pending', 'processing', 'completed', 'failed'}
        if v not in valid_statuses:
            raise ValueError(f"Invalid query status: {v}")
        return v
    
    def update_status(self, status: str):
        """Update query status with validation."""
        valid_statuses = {'pending', 'processing', 'completed', 'failed'}
        if status not in valid_statuses:
            raise ValueError(f"Invalid query status: {status}")
        self.status = status
    
    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }


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