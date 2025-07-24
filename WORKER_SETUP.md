# Celery Worker Infrastructure Setup

This document describes the Celery worker infrastructure for the concurrent RAG optimization system.

## Overview

The system uses Celery with Redis as the message broker to handle background tasks with proper resource management, monitoring, and automatic restart capabilities.

## Architecture

### Worker Types

1. **Embedding Workers** (`embedding` queue)
   - Process document uploads and generate embeddings
   - Limited to 2 concurrent workers to manage memory usage
   - Handle large file processing with progress tracking

2. **Query Workers** (`query` queue)
   - Process user queries with security isolation
   - Support up to 4 concurrent workers for better throughput
   - Implement query result caching and optimization

3. **Maintenance Workers** (`maintenance` queue)
   - Handle system cleanup and health monitoring
   - Single worker for scheduled maintenance tasks
   - Perform session cleanup and system health checks

### Key Features

- **Resource Management**: Automatic memory and CPU monitoring with configurable limits
- **Health Monitoring**: Real-time worker health reporting and system status
- **Automatic Restart**: Workers restart after processing 100 tasks to prevent memory leaks
- **Progress Tracking**: Real-time progress updates for long-running tasks
- **Error Handling**: Comprehensive error handling with retry logic
- **Security Isolation**: Proper user data isolation across all operations

## Prerequisites

1. **Redis Server**: Must be running on localhost:6379 (or configured URL)
2. **Python Dependencies**: All requirements from `requirements.txt` installed
3. **System Resources**: Minimum 4GB RAM recommended for full worker setup

## Quick Start

### 1. Start Redis (if not running)
```bash
# Option 1: Use setup script
./setup_redis.sh

# Option 2: Start manually
redis-server
```

### 2. Start All Workers
```bash
./start_workers.sh
```

This starts:
- 2 embedding workers
- 4 query workers  
- 1 maintenance worker
- 1 beat scheduler (for periodic tasks)

### 3. Monitor Workers
```bash
# Check system health
python monitor_workers.py

# Watch real-time status
python monitor_workers.py watch

# Check specific components
python monitor_workers.py workers
python monitor_workers.py queues
```

### 4. Stop Workers
```bash
./stop_workers.sh
```

## Configuration

### Environment Variables

```bash
# Redis Configuration
REDIS_URL=redis://localhost:6379/0

# Resource Limits
MAX_CONCURRENT_EMBEDDINGS=2
MAX_CONCURRENT_QUERIES=10
MAX_MEMORY_PER_WORKER=2GB
WORKER_TIMEOUT=300

# Rate Limiting
QUERY_MAX_REQUESTS=20
QUERY_WINDOW_SECONDS=60
```

### Worker Configuration

Edit `app/shared/config.py` to modify:

```python
RESOURCE_LIMITS = {
    "max_concurrent_embeddings": 2,
    "max_concurrent_queries": 10,
    "max_memory_per_worker": "2GB",
    "max_queue_size": 100,
    "worker_timeout": 300
}
```

## Monitoring and Management

### Health Monitoring

The system provides comprehensive health monitoring:

```bash
# System overview
python monitor_workers.py health

# Detailed worker status
python monitor_workers.py workers

# Queue statistics
python monitor_workers.py queues

# Clean up stale data
python monitor_workers.py cleanup
```

### Log Files

Worker logs are stored in `logs/workers/`:
- `embedding_worker.log` - Embedding worker logs
- `query_worker.log` - Query worker logs
- `maintenance_worker.log` - Maintenance worker logs
- `beat.log` - Scheduler logs

### Manual Worker Management

```bash
# Start specific worker type
celery -A app.workers.celery_app worker --queues=embedding --concurrency=2

# Check active tasks
celery -A app.workers.celery_app inspect active

# Check worker statistics
celery -A app.workers.celery_app inspect stats

# Purge all queues (CAUTION: Deletes all pending tasks)
celery -A app.workers.celery_app purge
```

## Troubleshooting

### Common Issues

1. **Redis Connection Error**
   ```
   Error: Redis is not running
   ```
   **Solution**: Start Redis server with `redis-server` or `./setup_redis.sh`

2. **Worker Memory Issues**
   ```
   Memory usage high: 1800MB / 2048MB
   ```
   **Solution**: Workers automatically restart when approaching limits. Increase `MAX_MEMORY_PER_WORKER` if needed.

3. **Workers Not Starting**
   ```
   Failed to start embedding_worker
   ```
   **Solution**: Check logs in `logs/workers/` and ensure all dependencies are installed.

4. **High Queue Length**
   ```
   Queue length: 50+ pending tasks
   ```
   **Solution**: Scale up workers or check for stuck tasks with `celery inspect active`

### Performance Tuning

1. **Increase Worker Concurrency** (if you have more resources):
   ```bash
   # Edit start_workers.sh
   start_worker "embedding_worker" "embedding" 4  # Increase from 2
   ```

2. **Adjust Memory Limits**:
   ```python
   # In app/shared/config.py
   "max_memory_per_worker": "4GB"  # Increase from 2GB
   ```

3. **Optimize Queue Processing**:
   ```python
   # In celery_app.py
   worker_prefetch_multiplier=2  # Increase from 1 for better throughput
   ```

## Testing

### Unit Tests
```bash
python -m pytest tests/test_celery_config.py -v
```

### Integration Tests
```bash
python test_worker_integration.py
```

### Load Testing
```bash
# Start workers first
./start_workers.sh

# Run load test (if available)
python test_load.py
```

## Security Considerations

1. **User Isolation**: All tasks include user context and security filters
2. **Resource Limits**: Prevent resource exhaustion attacks
3. **Task Validation**: All task inputs are validated before processing
4. **Session Security**: Secure session management across worker processes

## Production Deployment

For production deployment:

1. **Use Process Manager**: Deploy with systemd, supervisor, or Docker
2. **Configure Monitoring**: Set up proper logging and alerting
3. **Scale Resources**: Adjust worker counts based on load
4. **Backup Strategy**: Implement Redis persistence and backup
5. **Security Hardening**: Configure Redis authentication and network security

## API Integration

The worker infrastructure integrates with the FastAPI application:

```python
# Submit embedding task
from app.workers.embedding_worker import process_document_embedding

job_id = str(uuid.uuid4())
task = process_document_embedding.delay(job_id, user_id, group_id, file_paths)

# Submit query task  
from app.workers.query_worker import process_user_query

query_id = str(uuid.uuid4())
task = process_user_query.delay(query_id, user_id, group_ids, query_text)
```

## Support

For issues or questions:
1. Check logs in `logs/workers/`
2. Run `python monitor_workers.py health` for system status
3. Use `python test_worker_integration.py` to verify setup
4. Review this documentation and configuration files