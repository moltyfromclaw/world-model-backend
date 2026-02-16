#!/bin/bash
set -euxo pipefail

echo "=== LingBot-World Setup ==="
echo "Started: $(date)"

# Check GPU
nvidia-smi || { echo "No GPU detected!"; exit 1; }

# Install system deps
apt-get update
apt-get install -y git curl wget jq htop tmux

# Clone LingBot-World
if [ ! -d "lingbot-world" ]; then
    git clone https://github.com/robbyant/lingbot-world.git
fi
cd lingbot-world

# Install Python deps
pip install -r requirements.txt
pip install flash-attn --no-build-isolation

# Install WebSocket server deps
pip install websockets aiohttp pillow

# Download model (if not exists)
MODEL_DIR="lingbot-world-base-cam"
if [ ! -d "$MODEL_DIR" ]; then
    echo "Downloading model (~30GB)..."
    pip install "huggingface_hub[cli]"
    huggingface-cli download robbyant/lingbot-world-base-cam --local-dir ./$MODEL_DIR
else
    echo "Model already downloaded"
fi

# Copy WebSocket server
cp ../ws_server.py .
cp ../start.sh .
cp ../stop.sh .

echo "=== Setup Complete ==="
echo "Run ./start.sh to start the WebSocket server"
echo "Finished: $(date)"
