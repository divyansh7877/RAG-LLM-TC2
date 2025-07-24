#!/bin/bash

# Start Celery workers for concurrent RAG optimization
# This script starts different worker types with appropriate resource limits

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${GREEN}Starting Celery workers for concurrent RAG system...${NC}"

# Check if Redis is running
if ! redis-cli ping > /dev/null 2>&1; then
    echo -e "${RED}Error: Redis is not running. Please start Redis first.${NC}"
    echo "Run: redis-server or ./setup_redis.sh"
    exit 1
fi

# Create log directory
mkdir -p logs/workers

# Function to start a worker
start_worker() {
    local worker_name=$1
    local queue=$2
    local concurrency=$3
    local log_file="logs/workers/${worker_name}.log"
    
    echo -e "${YELLOW}Starting ${worker_name} worker (queue: ${queue}, concurrency: ${concurrency})...${NC}"
    
    celery -A app.workers.celery_app worker \
        --hostname="${worker_name}@%h" \
        --queues="${queue}" \
        --concurrency="${concurrency}" \
        --loglevel=info \
        --logfile="${log_file}" \
        --pidfile="logs/workers/${worker_name}.pid" \
        --detach
    
    if [ $? -eq 0 ]; then
        echo -e "${GREEN}✓ ${worker_name} worker started successfully${NC}"
    else
        echo -e "${RED}✗ Failed to start ${worker_name} worker${NC}"
        exit 1
    fi
}

# Start embedding workers (limited concurrency due to memory usage)
start_worker "embedding_worker" "embedding" 2

# Start query workers (higher concurrency for better throughput)
start_worker "query_worker" "query" 4

# Start maintenance worker (single worker for cleanup tasks)
start_worker "maintenance_worker" "maintenance" 1

# Start Celery Beat for scheduled tasks
echo -e "${YELLOW}Starting Celery Beat scheduler...${NC}"
celery -A app.workers.celery_app beat \
    --loglevel=info \
    --logfile=logs/workers/beat.log \
    --pidfile=logs/workers/beat.pid \
    --detach

if [ $? -eq 0 ]; then
    echo -e "${GREEN}✓ Celery Beat scheduler started successfully${NC}"
else
    echo -e "${RED}✗ Failed to start Celery Beat scheduler${NC}"
    exit 1
fi

echo -e "${GREEN}All workers started successfully!${NC}"
echo ""
echo "Worker status:"
echo "- Embedding workers: 2 processes (queue: embedding)"
echo "- Query workers: 4 processes (queue: query)" 
echo "- Maintenance worker: 1 process (queue: maintenance)"
echo "- Beat scheduler: 1 process (scheduled tasks)"
echo ""
echo "Log files are in: logs/workers/"
echo "To monitor workers: celery -A app.workers.celery_app inspect active"
echo "To stop workers: ./stop_workers.sh"