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

# Ensure we run from repo root and Python can import the app
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"
export PYTHONPATH="$SCRIPT_DIR:${PYTHONPATH}"
export CUDA_VISIBLE_DEVICES=0
export LD_LIBRARY_PATH="/usr/local/cuda/lib64:${LD_LIBRARY_PATH}"

# Verify celery is available
if ! command -v celery >/dev/null 2>&1; then
    echo -e "${RED}Error: 'celery' not found on PATH. Activate your conda env (e.g., 'conda activate llm').${NC}"
    exit 1
fi

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
    local pid_file="logs/workers/${worker_name}.pid"
    
    echo -e "${YELLOW}Starting ${worker_name} worker (queue: ${queue}, concurrency: ${concurrency})...${NC}"
    
    # Use solo pool for embedding and query workers to avoid fork with
    # fork-unsafe libs (PyTorch/CUDA, LanceDB/Arrow). This prevents hangs.
    local pool_arg=""
    if [ "${worker_name}" = "embedding_worker" ]; then
        pool_arg="--pool=solo"
    elif [ "${worker_name}" = "query_worker" ]; then
        pool_arg="--pool=solo"
    fi
    # Start in background using nohup and capture stdout/stderr into log file
    # We manage the PID ourselves for reliability
    nohup celery -A app.workers.celery_app worker \
        --hostname="${worker_name}@%h" \
        --queues="${queue}" \
        --concurrency="${concurrency}" \
        ${pool_arg} \
        --loglevel=info \
        --logfile="${log_file}" \
        >> "${log_file}" 2>&1 &
    echo $! > "${pid_file}"
    
    # Verify the worker actually started
    sleep 1
    if [ -f "${pid_file}" ] && kill -0 "$(cat "${pid_file}")" 2>/dev/null; then
        echo -e "${GREEN}✓ ${worker_name} worker started successfully (PID: $(cat "${pid_file}"))${NC}"
    else
        echo -e "${RED}✗ Failed to start ${worker_name} worker${NC}"
        echo -e "${YELLOW}Debug: Checking ${log_file} for errors...${NC}"
        if [ -f "${log_file}" ]; then
            tail -n 50 "${log_file}" || true
        fi
        exit 1
    fi
}

# Start embedding workers (limited concurrency due to memory usage)
start_worker "embedding_worker" "embedding" 1

# Start query workers
start_worker "query_worker" "query" 1

# Start maintenance worker (single worker for cleanup tasks)
start_worker "maintenance_worker" "maintenance" 1

# Start Celery Beat for scheduled tasks
echo -e "${YELLOW}Starting Celery Beat scheduler...${NC}"
nohup celery -A app.workers.celery_app beat \
    --loglevel=info \
    --logfile=logs/workers/beat.log \
    >> logs/workers/beat.log 2>&1 &
echo $! > logs/workers/beat.pid

if [ $? -eq 0 ]; then
    echo -e "${GREEN}✓ Celery Beat scheduler started successfully${NC}"
else
    echo -e "${RED}✗ Failed to start Celery Beat scheduler${NC}"
    exit 1
fi

echo -e "${GREEN}All workers started successfully!${NC}"


cat << EOF

Worker status:
- Embedding workers: 1 process (queue: embedding)
- Query workers: 4 processes (queue: query)
- Maintenance worker: 1 process (queue: maintenance)
- Beat scheduler: 1 process (scheduled tasks)

Log files are in: logs/workers/
To monitor workers: tail -f logs/workers/*.log
To inspect workers: celery -A app.workers.celery_app inspect active
To stop workers: ./stop_workers.sh
EOF