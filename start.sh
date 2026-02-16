#!/bin/bash
# Start the WebSocket server

cd "$(dirname "$0")"

# Export environment
export WS_HOST="0.0.0.0"
export WS_PORT="${WS_PORT:-8765}"
export MODEL_DIR="${MODEL_DIR:-lingbot-world-base-cam}"
export NUM_GPUS="${NUM_GPUS:-8}"
export TARGET_FPS="${TARGET_FPS:-16}"
export FRAME_WIDTH="${FRAME_WIDTH:-1280}"
export FRAME_HEIGHT="${FRAME_HEIGHT:-720}"

echo "Starting World Model WebSocket Server..."
echo "  Host: $WS_HOST:$WS_PORT"
echo "  Model: $MODEL_DIR"
echo "  GPUs: $NUM_GPUS"
echo "  FPS: $TARGET_FPS"

# Run in background with nohup
nohup python ws_server.py > ws_server.log 2>&1 &
echo $! > ws_server.pid

echo "Server started. PID: $(cat ws_server.pid)"
echo "Logs: tail -f ws_server.log"
echo "WebSocket URL: ws://$(hostname -I | awk '{print $1}'):$WS_PORT"
