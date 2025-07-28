# Project Structure

## Directory Organization

```
├── app/                          # Main application code
│   ├── api/                      # FastAPI web application
│   │   └── main.py              # API endpoints and middleware
│   ├── workers/                  # Celery background workers
│   │   ├── celery_app.py        # Celery configuration
│   │   ├── embedding_worker.py  # Document embedding tasks
│   │   ├── query_worker.py      # Query processing tasks
│   │   └── maintenance_worker.py # System maintenance tasks
│   ├── shared/                   # Shared components and utilities
│   │   ├── auth.py              # Authentication management
│   │   ├── config.py            # Configuration settings
│   │   ├── error_handling.py    # Centralized error handling
│   │   ├── middleware.py        # FastAPI middleware
│   │   ├── models.py            # Data models (Pydantic)
│   │   ├── monitoring.py        # System monitoring and alerts
│   │   ├── redis_client.py      # Redis connection utilities
│   │   └── resource_manager.py  # Resource management
│   ├── static/                   # Frontend assets
│   │   ├── css/styles.css       # Stylesheets
│   │   ├── js/app.js           # JavaScript application
│   │   └── index.html          # Main HTML template
│   ├── main.py                  # Gradio interface (legacy)
│   ├── new_embedder.py          # Document embedding service
│   └── new_rag_ui.py           # RAG query engine
├── tests/                       # Test suite
│   ├── test_*.py               # Unit and integration tests
│   └── __init__.py
├── models/                      # AI model files
│   ├── Llama-3.2-3B-Instruct-IQ3_M.gguf  # LLM model
│   └── gte-large-en-v1.5/      # Embedding model
├── pdfs/                        # Sample PDF documents
├── lancedb_*/                   # Vector database storage
├── logs/                        # Application logs
└── temp_uploads/               # Temporary file storage
```

## Architecture Patterns

### Modular Design
- **Separation of Concerns**: Clear boundaries between API, workers, and shared utilities
- **Dependency Injection**: Configuration and clients passed as dependencies
- **Factory Pattern**: Query engine factory for thread-safe model management

### Data Flow
1. **Upload**: `API → Worker → Embedder → LanceDB`
2. **Query**: `API → Worker → RAG Engine → LLM → Response`
3. **Session**: `API → Redis → Session Manager`

### Security Model
- **Authentication**: JWT tokens with Redis session storage
- **Authorization**: User/group-based document access control
- **Data Isolation**: Metadata filters ensure user data separation

## Code Organization Principles

### Naming Conventions
- **Files**: snake_case (e.g., `error_handling.py`)
- **Classes**: PascalCase (e.g., `ErrorHandler`)
- **Functions/Variables**: snake_case (e.g., `process_query`)
- **Constants**: UPPER_SNAKE_CASE (e.g., `MAX_FILE_SIZE`)

### Import Structure
```python
# Standard library imports
import os
import time

# Third-party imports
from fastapi import FastAPI
from celery import Celery

# Local imports
from app.shared.config import config
from app.shared.models import UserSession
```

### Error Handling
- **Centralized**: All errors flow through `app.shared.error_handling`
- **Categorized**: Errors classified by type (auth, validation, resource, etc.)
- **Logged**: Structured logging with context information
- **Monitored**: Integration with alerting system

### Configuration Management
- **Environment Variables**: External configuration via env vars
- **Defaults**: Sensible defaults in `app.shared.config`
- **Validation**: Pydantic models for configuration validation

### Testing Structure
- **Unit Tests**: Individual component testing
- **Integration Tests**: Cross-component interaction testing
- **Infrastructure Tests**: System dependency verification
- **Performance Tests**: Load and benchmark testing

## Key Design Decisions

### Thread Safety
- **Singleton Pattern**: Shared resources (LLM, embeddings) loaded once
- **Factory Pattern**: New instances for user-specific operations
- **Locking**: Thread locks for critical sections

### Resource Management
- **Connection Pooling**: Redis and database connections
- **Memory Management**: Configurable worker concurrency
- **Cleanup**: Automatic resource cleanup and monitoring

### Scalability Considerations
- **Horizontal Scaling**: Celery workers can run on multiple machines
- **Queue Separation**: Different queues for different task types
- **Caching**: Redis caching for frequently accessed data