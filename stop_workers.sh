#!/bin/bash

# Stop all Celery workers and beat scheduler

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${YELLOW}Stopping Celery workers and beat scheduler...${NC}"

# Function to stop a worker by PID file
stop_worker() {
    local worker_name=$1
    local pid_file="logs/workers/${worker_name}.pid"
    
    if [ -f "$pid_file" ]; then
        local pid=$(cat "$pid_file")
        if kill -0 "$pid" 2>/dev/null; then
            echo -e "${YELLOW}Stopping ${worker_name} (PID: ${pid})...${NC}"
            kill -TERM "$pid"
            
            # Wait for graceful shutdown
            local count=0
            while kill -0 "$pid" 2>/dev/null && [ $count -lt 10 ]; do
                sleep 1
                count=$((count + 1))
            done
            
            # Force kill if still running
            if kill -0 "$pid" 2>/dev/null; then
                echo -e "${RED}Force killing ${worker_name}...${NC}"
                kill -KILL "$pid"
            fi
            
            echo -e "${GREEN}✓ ${worker_name} stopped${NC}"
        else
            echo -e "${YELLOW}${worker_name} is not running${NC}"
        fi
        rm -f "$pid_file"
    else
        echo -e "${YELLOW}No PID file found for ${worker_name}${NC}"
    fi
}

# Stop all workers
stop_worker "embedding_worker"
stop_worker "query_worker" 
stop_worker "maintenance_worker"
stop_worker "beat"

# Also try to stop any remaining celery processes
echo -e "${YELLOW}Checking for remaining Celery processes...${NC}"
pkill -f "celery.*app.workers.celery_app" || true

echo -e "${GREEN}All workers stopped successfully!${NC}"