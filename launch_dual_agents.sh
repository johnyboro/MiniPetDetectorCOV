#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOG_DIR="$ROOT_DIR/logs"
TIMESTAMP="$(date +%Y%m%d_%H%M%S)"

mkdir -p "$LOG_DIR"

source .venv/bin/activate
CMD=(python3 "$ROOT_DIR/train.py" "$@")

nohup env CUDA_VISIBLE_DEVICES=0 "${CMD[@]}" >"$LOG_DIR/agent_gpu0_${TIMESTAMP}.log" 2>&1 &
PID0=$!

nohup env CUDA_VISIBLE_DEVICES=1 "${CMD[@]}" >"$LOG_DIR/agent_gpu1_${TIMESTAMP}.log" 2>&1 &
PID1=$!

echo "Started dual agents in background"
echo "- GPU 0 PID: $PID0 (log: $LOG_DIR/agent_gpu0_${TIMESTAMP}.log)"
echo "- GPU 1 PID: $PID1 (log: $LOG_DIR/agent_gpu1_${TIMESTAMP}.log)"
echo "Use 'ps -fp $PID0,$PID1' to check status."
