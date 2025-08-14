# Application Startup Guide

## Overview

This guide will help you run the Concurrent Multi-User Private RAG System. The application has been transformed from a single-threaded Gradio app into a modern, concurrent web application with multiple components.

## Prerequisites

1. **Python Environment**: Python 3.11+ with conda or venv
2. **Redis Server**: Required for task queuing and session management
3. **Models**: AI models need to be downloaded
4. **Dependencies**: All Python packages installed

## Quick Start (Recommended)

### 1. Environment Setup

```bash
# Activate your conda environment
conda activate llm  # or your environment name

# Install dependencies
pip install -r requirements.txt
pip install -r test-requirements.txt  # For testing
```

### 2. Download Models

```bash
# Download required AI models
python download_model.py
```

### 3. Setup Redis

```bash
# Install and start Redis
./setup_redis.sh

# Verify Redis is running
redis-cli ping
# Should return: PONG
```

### 4. Start the Application

You have two options: **Modern Web Application** (recommended) or **Legacy Gradio Interface**.

## Option A: Modern Web Application (Recommended)

This is the new concurrent, multi-user system with FastAPI backend and modern web interface.

### Start All Services

**Terminal 1: Start Redis (if not already running)**
```bash
redis-server
```

**Terminal 2: Start FastAPI Web Server**
```bash
python -m app.api.main
```

**Terminal 3: Start Celery Workers**
```bash
# Start all workers at once
./start_workers.sh

# Or start workers individually:
# celery -A app.workers.celery_app worker --queues=embedding --concurrency=2
# celery -A app.workers.celery_app worker --queues=query --concurrency=4
# celery -A app.workers.celery_app worker --queues=maintenance --concurrency=1
```

**Terminal 4: Monitor Workers (Optional)**
```bash
# Monitor worker status
celery -A app.workers.celery_app inspect active

# Or start Flower for web-based monitoring
celery -A app.workers.celery_app flower
# Access at: http://localhost:5555
```

### Access the Application

1. **Web Interface**: Open http://localhost:8000 in your browser
2. **API Documentation**: Visit http://localhost:8000/docs for interactive API docs
3. **Health Check**: Visit http://localhost:8000/health to verify system status

### Default Login Credentials

The system includes several test users:

- **Username**: `assistant1`, **Password**: `password1` (Groups: assistance, common_rules)
- **Username**: `assistant2`, **Password**: `password2` (Groups: assistance)
- **Username**: `guest`, **Password**: `password` (Groups: common_rules)

## Option B: Legacy Gradio Interface

This runs the original single-user Gradio interface (for comparison/fallback).

```bash
python app/main.py
```

Access at: http://localhost:7860

## Application Features

### Modern Web Application Features

1. **Multi-User Support**: Multiple users can use the system simultaneously
2. **Real-Time Updates**: WebSocket connections provide live job status updates
3. **Background Processing**: Document uploads and queries are processed asynchronously
4. **Job Management**: Track and monitor all your processing jobs
5. **Security**: JWT-based authentication with session management
6. **Resource Management**: Intelligent resource allocation and monitoring

### How to Use

1. **Login**: Use one of the default credentials or create new users
2. **Upload Documents**: 
   - Navigate to the upload section
   - Select PDF files
   - Choose a destination group
   - Files are processed in the background with progress updates
3. **Query Documents**: 
   - Submit questions about your uploaded documents
   - Get real-time status updates
   - Receive answers with source citations
4. **Monitor Jobs**: 
   - View all your active and completed jobs
   - Track progress and view results
   - Cancel jobs if needed

## Troubleshooting

### Common Issues

1. **Redis Connection Error**
   ```bash
   # Check if Redis is running
   redis-cli ping
   
   # If not running, start it
   redis-server
   # or
   ./setup_redis.sh
   ```

2. **Port Already in Use**
   ```bash
   # Check what's using port 8000
   lsof -i :8000
   
   # Kill the process if needed
   kill -9 <PID>
   ```

3. **Models Not Found**
   ```bash
   # Download models
   python download_model.py
   
   # Verify models exist
   ls -la models/
   ```

4. **Import Errors**
   ```bash
   # Ensure you're in the project root
   pwd
   
   # Set PYTHONPATH if needed
   export PYTHONPATH=$PWD:$PYTHONPATH
   ```

5. **Worker Connection Issues**
   ```bash
   # Check worker status
   celery -A app.workers.celery_app inspect ping
   
   # Restart workers
   ./stop_workers.sh
   ./start_workers.sh
   ```

### Performance Issues

1. **Slow Document Processing**
   - Increase embedding worker concurrency (but watch memory usage)
   - Ensure models are properly loaded

2. **Slow Queries**
   - Increase query worker concurrency
   - Check if vector database is properly indexed

3. **Memory Issues**
   - Reduce worker concurrency
   - Monitor system resources: `htop` or `top`

### Logs and Debugging

1. **Application Logs**
   ```bash
   # FastAPI logs (in terminal where you started the server)
   # Worker logs
   ls logs/workers/
   tail -f logs/workers/embedding_worker.log
   ```

2. **Redis Logs**
   ```bash
   # Check Redis logs
   redis-cli monitor
   ```

3. **System Resources**
   ```bash
   # Monitor system resources
   htop
   
   # Check disk space
   df -h
   
   # Check memory usage
   free -h
   ```

## Development Mode

For development, you can use the development startup script:

```bash
./start_dev.sh
```

This will:
- Check Redis connection
- Install dependencies
- Create necessary directories
- Provide instructions for starting services

## Testing

Run the comprehensive test suite:

```bash
# Run all tests
python run_tests.py

# Run specific test suites
python run_tests.py --suite unit
python run_tests.py --suite integration
python run_tests.py --suite performance
python run_tests.py --suite security

# Run with coverage
python run_tests.py --coverage
```

## Production Deployment

For production deployment:

1. **Use a process manager** (systemd, supervisor, or Docker)
2. **Configure proper logging** (structured logging to files)
3. **Set up monitoring** (health checks, metrics collection)
4. **Use a reverse proxy** (nginx, Apache)
5. **Configure SSL/TLS** for secure connections
6. **Set up backup strategies** for data persistence

## Architecture Overview

The application consists of:

- **FastAPI Web Server**: REST API and WebSocket endpoints
- **Celery Workers**: Background task processing
- **Redis**: Message broker and session storage
- **LanceDB**: Vector database for document embeddings
- **Local LLM**: Llama-3.2-3B for text generation
- **Embedding Model**: gte-large-en-v1.5 for document vectorization

## Next Steps

1. **Customize Authentication**: Integrate with your identity provider
2. **Add More Document Types**: Extend beyond PDF support
3. **Scale Horizontally**: Add more worker instances
4. **Implement Monitoring**: Add comprehensive observability
5. **Optimize Performance**: Tune for your specific use case

For more detailed information, see:
- `SETUP.md` - Infrastructure setup details
- `TEST_SUITE_SUMMARY.md` - Testing documentation
- `TESTING_GUIDE.md` - How to run tests
- `Readme.md` - Complete project overview