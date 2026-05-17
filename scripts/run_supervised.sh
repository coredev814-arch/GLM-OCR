#!/bin/bash
# Minimal supervisor for environments without systemd/docker.
# Restarts uvicorn on any exit so the engine's CUDA-poison self-heal
# (os._exit on fatal CUDA errors) actually leads to a recovery.
set -u
LOG=${LOG:-/workspace/uvicorn.log}
HOST=${HOST:-0.0.0.0}
PORT=${PORT:-8889}
BACKOFF=${BACKOFF:-5}

cd /workspace
while true; do
  echo "$(date -Iseconds) supervisor: starting uvicorn on $HOST:$PORT" >> "$LOG"
  uvicorn app.main:app --host "$HOST" --port "$PORT" >> "$LOG" 2>&1
  rc=$?
  echo "$(date -Iseconds) supervisor: uvicorn exited rc=$rc, restarting in ${BACKOFF}s" >> "$LOG"
  sleep "$BACKOFF"
done
