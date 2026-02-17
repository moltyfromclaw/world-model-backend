#!/bin/bash
set -euo pipefail

echo "=== LingBot-World Setup ==="
echo "Started: $(date)"

# Check GPU
nvidia-smi || { echo "No GPU detected!"; exit 1; }

# Detect GPU architecture
COMPUTE_CAP=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader | head -1 | tr -d '.')
echo "Detected GPU compute capability: $COMPUTE_CAP"

# Check if Blackwell (compute 12.0+)
NEEDS_NIGHTLY=false
if [ "$COMPUTE_CAP" -ge "120" ]; then
    echo "⚠️  Blackwell GPU detected (sm_120+) - need PyTorch nightly"
    NEEDS_NIGHTLY=true
fi

# Install system deps
apt-get update && apt-get install -y git curl wget jq htop tmux

# Handle PyTorch for Blackwell GPUs
if [ "$NEEDS_NIGHTLY" = true ]; then
    echo "Installing PyTorch nightly with CUDA 12.9 (Blackwell support)..."
    pip uninstall torch torchvision torchaudio -y 2>/dev/null || true
    pip install --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cu129
    
    # Verify sm_120 support
    python -c "import torch; archs = torch.cuda.get_arch_list(); assert 'sm_120' in archs or 'compute_120' in archs, f'sm_120 not in {archs}'" || {
        echo "ERROR: PyTorch nightly doesn't have sm_120 support"
        echo "Try: pip install torch==2.10.0a0+sm120 --extra-index-url https://pytorch-sm120.github.io/whl/"
        exit 1
    }
    echo "✓ PyTorch with sm_120 support installed"
fi

# Clone LingBot-World
cd /workspace
if [ ! -d "lingbot-world" ]; then
    git clone https://github.com/robbyant/lingbot-world.git
fi
cd lingbot-world

# Install Python deps
pip install -r requirements.txt

# Install/rebuild flash-attn (must be after PyTorch)
echo "Installing flash-attn (this takes a few minutes)..."
pip uninstall flash-attn -y 2>/dev/null || true
pip install flash-attn --no-build-isolation

# Install WebSocket server deps  
pip install websockets aiohttp pillow

# Download model (if not exists) - ~85GB for full model
MODEL_DIR="/workspace/lingbot-world-base-cam"
if [ ! -d "$MODEL_DIR" ] || [ ! -f "$MODEL_DIR/models_t5_umt5-xxl-enc-bf16.pth" ]; then
    echo "Downloading model (~85GB for full, ~30GB for quantized)..."
    pip install "huggingface_hub[cli]"
    
    # Use quantized version if low on space (check available disk)
    AVAIL_GB=$(df /workspace --output=avail -BG | tail -1 | tr -d 'G ')
    if [ "$AVAIL_GB" -lt 100 ]; then
        echo "Low disk space ($AVAIL_GB GB) - downloading quantized model..."
        huggingface-cli download cahlen/lingbot-world-base-cam-nf4 \
            --local-dir $MODEL_DIR \
            --local-dir-use-symlinks False
    else
        huggingface-cli download robbyant/lingbot-world-base-cam \
            --local-dir $MODEL_DIR \
            --local-dir-use-symlinks False
    fi
else
    echo "Model already downloaded"
fi

# Create symlink if model is in different location
if [ -d "$MODEL_DIR" ] && [ ! -d "/workspace/lingbot-world/lingbot-world-base-cam" ]; then
    ln -sf $MODEL_DIR /workspace/lingbot-world/lingbot-world-base-cam
fi

# Pull latest world-model-backend code
cd /workspace
if [ -d "world-model-backend" ]; then
    cd world-model-backend
    git pull || true
    cd ..
else
    git clone https://github.com/moltyfromclaw/world-model-backend.git
fi

echo ""
echo "=== Setup Complete ==="
echo "Finished: $(date)"
echo ""
echo "Quick test (single GPU):"
echo "  cd /workspace/lingbot-world"
echo "  python generate.py --task i2v-A14B --size 832*480 --ckpt_dir lingbot-world-base-cam --image examples/02/image.jpg --frame_num 17 --offload_model True --prompt 'A forest scene'"
echo ""
echo "Starting WebSocket server..."
cd /workspace/world-model-backend
exec python ws_server.py
