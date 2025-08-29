#!/bin/bash
set -euo pipefail

# Colors
RED=$'\033[0;31m'; GREEN=$'\033[0;32m'; YELLOW=$'\033[1;33m'; NC=$'\033[0m'

APP="app.workers.celery_app"   # ← your Celery app path
PID_DIR="logs/workers"         # ← where you write pidfiles

echo "${YELLOW}Gracefully shutting down Celery (broadcast)…${NC}"
# Try graceful broadcast first (workers/beat will exit themselves)
# NOTE: requires broker reachable and workers alive.
timeout 5s celery -A "$APP" control shutdown 2>/dev/null || true

# Optional: stop beat specifically if you run it
# timeout 5s celery -A "$APP" beat -S redbeat stop 2>/dev/null || true

stop_by_pidfile () {
  local name="$1"
  local pidfile="${PID_DIR}/${name}.pid"
  if [[ -f "$pidfile" ]]; then
    local pid
    pid="$(cat "$pidfile" || true)"
    if [[ -n "${pid}" ]] && kill -0 "$pid" 2>/dev/null; then
      echo "${YELLOW}Stopping ${name} (PID ${pid})…${NC}"
      # Send INT for Celery so it can cleanup pools nicely; fall back to TERM
      kill -INT "$pid" 2>/dev/null || kill -TERM "$pid" 2>/dev/null || true

      # Wait up to 12s for graceful shutdown
      for _ in {1..12}; do
        sleep 1
        kill -0 "$pid" 2>/dev/null || break
      done

      # Kill the whole process group if still alive (reaps child workers)
      if kill -0 "$pid" 2>/dev/null; then
        echo "${YELLOW}Killing process group for ${name}…${NC}"
        PGID=$(ps -o pgid= -p "$pid" | tr -d ' ')
        [[ -n "$PGID" ]] && kill -TERM -"$PGID" 2>/dev/null || true
        sleep 2
      fi

      # Last resort: SIGKILL
      if kill -0 "$pid" 2>/dev/null; then
        echo "${RED}Force killing ${name}…${NC}"
        kill -KILL "$pid" 2>/dev/null || true
      fi
      echo "${GREEN}✓ ${name} stopped${NC}"
    else
      echo "${YELLOW}${name} not running${NC}"
    fi
    rm -f "$pidfile"
  else
    echo "${YELLOW}No PID file for ${name}${NC}"
  fi
}

echo "${YELLOW}Stopping workers/beat by pidfile…${NC}"
stop_by_pidfile embedding_worker
stop_by_pidfile query_worker
stop_by_pidfile maintenance_worker
stop_by_pidfile beat

echo "${YELLOW}Sweeping any stray Celery children…${NC}"
# Reap leftover Celery forked workers (prefork/spawn) and task pools
pkill -TERM -f "celery.*${APP}" 2>/dev/null || true
sleep 2
pkill -KILL -f "celery.*${APP}" 2>/dev/null || true

# Optional: if you run llama.cpp server or other model servers separately, reap them too
pkill -TERM -f "llama(-cpp)?(.*server)?" 2>/dev/null || true
sleep 2
pkill -KILL -f "llama(-cpp)?(.*server)?" 2>/dev/null || true

# Optional: free PyTorch allocators by killing Python processes known to hold CUDA (be specific!)
pkill -TERM -f "python.*(sentence_transformers|your_embedder_entrypoint)" 2>/dev/null || true

echo "${YELLOW}GPU usage after stop (nvidia-smi)…${NC}"
command -v nvidia-smi >/dev/null && nvidia-smi || true

echo "${GREEN}All workers stopped.${NC}"
