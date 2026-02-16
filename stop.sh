#!/bin/bash
# Stop the WebSocket server gracefully

cd "$(dirname "$0")"

if [ -f ws_server.pid ]; then
    PID=$(cat ws_server.pid)
    if kill -0 $PID 2>/dev/null; then
        echo "Stopping server (PID: $PID)..."
        kill $PID
        sleep 2
        if kill -0 $PID 2>/dev/null; then
            echo "Force killing..."
            kill -9 $PID
        fi
        rm ws_server.pid
        echo "Server stopped."
    else
        echo "Server not running."
        rm ws_server.pid
    fi
else
    echo "No PID file found. Server may not be running."
    # Try to find and kill anyway
    pkill -f "ws_server.py" && echo "Killed ws_server.py processes"
fi
