# WARP.md

This file provides guidance to WARP (warp.dev) when working with code in this repository.

## System Overview

This is a production-ready, concurrent multi-user RAG (Retrieval-Augmented Generation) system built with FastAPI, Celery workers, and Redis. The system enables multiple users to simultaneously upload private documents and query them using a local LLM with complete data isolation and security.

## Core Architecture

### Frontend Layer
- **FastAPI API** (`app/api/main.py`): REST endpoints with WebSocket support for real-time updates
- **Static Web UI** (`app/static/`): Responsive HTML/CSS/JavaScript frontend with Keycloak integration

### Application Layer
- **Session Manager** (`app/shared/session_manager.py`): Redis-backed distributed session management
- **Job Manager** (`app/shared/job_manager.py`): Comprehensive job lifecycle management and tracking
- **Resource Manager** (`app/shared/resource_manager.py`): Intelligent resource allocation and monitoring
- **Query Engine Factory** (`app/shared/query_engine_factory.py`): Thread-safe query engine with connection pooling

### Task Processing Layer
- **Celery Workers** with specialized queues:
  - `embedding`: Document processing (limited concurrency for GPU resources)
  - `query`: User queries (higher concurrency)
  - `maintenance`: System cleanup and health checks

### Data Layer
- **LanceDB**: Vector storage with user isolation metadata
- **Redis**: Sessions, job tracking, caching, and message brokering
- **Docling**: Advanced document processing (PDF, DOCX, PPTX, XLSX, HTML, MD, CSV)

## Common Development Tasks

### Environment Setup
```bash
# Create conda environment
conda env create -f environment.yml
conda activate expert-rag

# Or use pip
pip install -r requirements.txt
pip install -r test-requirements.txt  # For testing
```

### Starting Services

#### Development Mode (All-in-one)
```bash
./start_dev.sh
```
This sets up dependencies and shows individual service start commands.

#### Production Mode
```bash
# 1. Setup and start Redis
./setup_redis.sh

# 2. Start FastAPI application
python -m app.api.main

# 3. Start all Celery workers
./start_workers.sh

# 4. Monitor workers (optional)
./monitor_workers.py
```

#### Individual Worker Control
```bash
# Embedding worker (GPU-intensive, low concurrency)
celery -A app.workers.celery_app worker --queues=embedding --concurrency=1 --pool=threads

# Query worker (CPU-bound, higher concurrency)
celery -A app.workers.celery_app worker --queues=query --concurrency=4 --pool=solo

# Maintenance worker (cleanup tasks)
celery -A app.workers.celery_app worker --queues=maintenance --concurrency=1

# Monitor with Flower
celery -A app.workers.celery_app flower
```

### Testing

#### Run All Tests
```bash
# Using custom test runner (recommended)
python run_tests.py

# Using pytest directly
python -m pytest tests/ -v

# With coverage
python -m pytest tests/ --cov=app --cov-report=html
```

#### Test Categories
```bash
# Unit tests (security, performance, API)
python -m pytest -m unit tests/

# Integration tests (end-to-end workflows)
python -m pytest -m integration tests/

# Performance tests (load testing, 50+ concurrent users)
python -m pytest -m performance tests/

# Security tests (authentication, data isolation)
python -m pytest -m security tests/
```

### Model Management

#### Download Models
```bash
python download_model.py  # Downloads embedding model to ./models/
```

#### GPU Configuration
- CUDA auto-detection in `app/shared/config.py`
- Docling uses GPU when available for faster OCR
- Embedding model runs on GPU with CPU fallback

## Key Development Patterns

### User Data Isolation
All data operations use user/group-scoped keys:
```python
# Document storage pattern
key = f"document:{user_id}:{group_id}:{document_id}"

# LanceDB filtering pattern
filter_conditions = f"user_id = '{user_id}' AND group_id IN {user_groups}"
```

### Job Management Pattern
```python
from app.shared.job_manager import job_manager, JobType

# Create job
job = job_manager.create_job(
    user_id=user_id,
    job_type=JobType.EMBEDDING,
    metadata={"group_id": group_id}
)

# Update progress
job_manager.update_job_progress(job.job_id, 0.5)

# Complete job
job_manager.complete_job(job.job_id, result_data)
```

### Resource Management
```python
from app.shared.resource_manager import resource_manager

# Check resource availability
if resource_manager.can_allocate_resources("embedding"):
    # Proceed with resource-intensive task
    pass
```

### Authentication Pattern
```python
from app.shared.middleware import get_current_user, require_roles

# Route with authentication
@app.get("/api/secure-endpoint")
async def secure_endpoint(
    current_user: User = Depends(get_current_user)
):
    # User is authenticated via Keycloak JWT
    pass

# Route with role requirement
@app.post("/api/admin-only")
async def admin_only(
    current_user: User = Depends(require_roles(["admin"]))
):
    pass
```

## Monitoring and Debugging

### Health Checks
```bash
# Basic health check
curl http://localhost:8000/health

# Detailed API status
curl http://localhost:8000/api/status

# System metrics (requires auth)
curl -H "Authorization: Bearer $TOKEN" http://localhost:8000/api/monitoring/metrics
```

### Log Locations
- Application logs: `logs/`
- Worker logs: `logs/workers/`
- Redis logs: System logs via systemd

### Monitoring Workers
```bash
# Check worker status
celery -A app.workers.celery_app inspect active

# Monitor worker performance
./monitor_workers.py

# Check queue lengths
celery -A app.workers.celery_app inspect reserved
```

## Performance Optimization

### GPU Memory Management
- Embedding workers use `--pool=threads` for GPU safety
- Single embedding worker per GPU to avoid VRAM conflicts
- Model singleton pattern prevents repeated loading

### Concurrency Configuration
- Embedding queue: 1-2 workers (GPU-bound)
- Query queue: 4-8 workers (CPU/memory-bound)
- Maintenance queue: 1 worker (cleanup tasks)

### Database Optimization
- LanceDB auto-creates tables on first insert
- Connection pooling via query engine factory
- Redis-based caching for frequent queries

## Security Considerations

### User Isolation
- All database operations include user/group filtering
- JWT token validation on every request
- Session management with configurable timeouts

### File Upload Security
- File type validation using supported extensions
- Size limits enforced (configurable in `app/shared/config.py`)
- Temporary files cleaned up after processing

### Rate Limiting
- Configurable per-endpoint rate limits
- Redis-backed rate limiting with user-specific keys

## Configuration

Key configuration in `app/shared/config.py`:
- `REDIS_URL`: Redis connection string
- `LANCEDB_PATH`: Vector database location
- `MAX_FILE_SIZE`: Upload size limit
- `RESOURCE_LIMITS`: Worker concurrency and memory limits
- `KEYCLOAK_*`: Authentication provider settings

Environment variables override defaults for deployment flexibility.

## Troubleshooting

### Common Issues

#### Redis Connection Errors
```bash
# Check Redis status
redis-cli ping

# Restart Redis
sudo systemctl restart redis-server
```

#### Worker Issues
```bash
# Check worker logs
tail -f logs/workers/embedding_worker.log

# Restart workers
./stop_workers.sh && ./start_workers.sh
```

#### GPU/CUDA Issues
- Verify CUDA availability: Check `app/shared/config.py` for `HAS_CUDA`
- Memory errors: Reduce embedding batch size
- Driver issues: Check CUDA_VISIBLE_DEVICES environment variable

#### Database Issues
- LanceDB corruption: Remove `multi_user_db.lance/` directory (data loss)
- Performance issues: Check disk space and I/O capacity

### Development Debugging

#### Interactive Testing
```python
# Test Redis connection
from app.shared.redis_client import redis_client
redis_client.health_check()

# Test authentication
from app.shared.auth import auth_manager
await auth_manager.load_jwks()
```

#### Performance Profiling
```bash
# Memory profiling
python -m memory_profiler script.py

# Performance benchmarking
python -m pytest tests/test_performance_benchmarks.py -v
```

## Data Management

### Backup Procedures
- Vector data: Back up `multi_user_db.lance/` directory
- Redis data: Use Redis persistence (RDB/AOF)
- Models: Keep in version control or artifact storage

### Data Migration
- LanceDB schema changes require manual migration
- User data keyed by user_id and group_id for isolation
- Session data expires automatically

This system prioritizes security, performance, and concurrent access while maintaining data isolation between users. The architecture supports horizontal scaling through additional Celery workers and Redis clustering.
