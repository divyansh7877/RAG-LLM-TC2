#!/bin/bash

# Development startup script for Concurrent RAG System

echo "Starting Concurrent RAG System in development mode..."

# Check if Redis is running
if ! redis-cli ping &> /dev/null; then
    echo "Redis is not running. Please run ./setup_redis.sh first or start Redis manually."
    exit 1
fi

# Create necessary directories
mkdir -p temp_uploads
mkdir -p logs

# Install Python dependencies
echo "Installing Python dependencies..."
pip install -r requirements.txt

echo ""
echo "Starting services..."
echo "You can now start the following services in separate terminals:"
echo ""
echo "1. FastAPI Web Server:"
echo "   python -m app.api.main"
echo ""
echo "2. Celery Worker (Embedding):"
echo "   celery -A app.workers.celery_app worker --loglevel=info --queues=embedding --concurrency=2"
echo ""
echo "3. Celery Worker (Query):"
echo "   celery -A app.workers.celery_app worker --loglevel=info --queues=query --concurrency=4"
echo ""
echo "4. Celery Worker (Maintenance):"
echo "   celery -A app.workers.celery_app worker --loglevel=info --queues=maintenance --concurrency=1"
echo ""
echo "5. Celery Flower (Optional - for monitoring):"
echo "   celery -A app.workers.celery_app flower"
echo ""
echo "Or run all workers together:"
echo "   celery -A app.workers.celery_app worker --loglevel=info --concurrency=4"