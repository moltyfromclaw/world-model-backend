# World Model Backend - LingBot-World on RunPod

RunPod deployment for LingBot-World world model inference with WebSocket frame streaming.

## Quick Start

```bash
# SSH into RunPod pod
cd /workspace
git clone https://github.com/moltyfromclaw/world-model-backend.git
cd world-model-backend
chmod +x setup.sh && ./setup.sh
```

The setup script auto-detects your GPU and handles:
- ✅ Blackwell GPUs (RTX 6000, sm_120) - installs PyTorch nightly
- ✅ Ampere/Hopper (A100, H100, sm_80/90) - uses stable PyTorch
- ✅ Model download (~85GB full, ~30GB quantized)
- ✅ Flash attention compilation

## GPU Compatibility

| GPU | Compute | Status | Notes |
|-----|---------|--------|-------|
| A100 80GB | sm_80 | ✅ Best | Recommended, stable |
| H100 80GB | sm_90 | ✅ Works | Faster, pricier |
| L40S | sm_89 | ✅ Works | Good budget option |
| RTX 6000 Blackwell | sm_120 | ⚠️ Needs nightly | Script handles this |
| RTX 4090 | sm_89 | ✅ Works | 24GB VRAM limit |

## Requirements

- **Full model:** 8x 80GB GPUs, ~100GB disk
- **Quantized:** 1x 48GB+ GPU, ~50GB disk

## API

### HTTP Health Endpoints (port 8080)

Test the server without WebSocket:

```bash
# Basic health - is server running?
curl http://localhost:8080/health

# Readiness - is model loaded?
curl http://localhost:8080/ready

# Full status with GPU info
curl http://localhost:8080/status

# Generate a test frame (returns JPEG)
curl http://localhost:8080/test-frame -o test.jpg
curl "http://localhost:8080/test-frame?prompt=A%20forest" -o forest.jpg
```

### WebSocket Messages (port 8765)

**Client → Server:**
```json
// Initialize world
{ "type": "init", "prompt": "A medieval castle..." }

// Control input (WASD)
{ "type": "control", "key": "w", "action": "down" }
{ "type": "control", "key": "w", "action": "up" }
```

**Server → Client:**
```json
// Frame data (binary JPEG)
<binary frame data>
```

## Manual Testing

```bash
cd /workspace/lingbot-world

# Single GPU test (slower, uses model offloading)
python generate.py \
  --task i2v-A14B \
  --size 832*480 \
  --ckpt_dir lingbot-world-base-cam \
  --image examples/02/image.jpg \
  --frame_num 17 \
  --offload_model True \
  --prompt "A peaceful forest scene"

# Multi-GPU (8x, full speed)
torchrun --nproc_per_node=8 generate.py \
  --task i2v-A14B \
  --size 1280*720 \
  --ckpt_dir lingbot-world-base-cam \
  --image examples/02/image.jpg \
  --dit_fsdp --t5_fsdp \
  --ulysses_size 8 \
  --frame_num 17 \
  --prompt "A peaceful forest scene"
```

### Valid Sizes
`1280*720`, `720*1280`, `832*480`, `480*832`, `1280*704`, `704*1280`, `704*1024`, `1024*704`

## Troubleshooting

### "no kernel image is available for execution"
**Cause:** GPU architecture not supported by PyTorch build.
**Fix:** Run `./setup.sh` - it auto-detects Blackwell and installs nightly.

```bash
# Manual fix:
pip uninstall torch torchvision torchaudio -y
pip install --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cu129
pip install flash-attn --no-build-isolation --force-reinstall
```

### "undefined symbol" in flash_attn
**Cause:** Flash attention compiled against different PyTorch version.
**Fix:**
```bash
pip uninstall flash-attn -y
pip install flash-attn --no-build-isolation
```

### "FileNotFoundError: models_t5_umt5-xxl-enc-bf16.pth"
**Cause:** Model download incomplete.
**Fix:**
```bash
rm -rf /workspace/lingbot-world-base-cam
huggingface-cli download robbyant/lingbot-world-base-cam \
  --local-dir /workspace/lingbot-world-base-cam
```

### "invalid choice" for --size
**Cause:** Using wrong resolution format.
**Fix:** Use one of: `1280*720`, `832*480`, `720*1280`, etc. (not `720*480`)

### Multi-GPU "monitoredBarrier" timeout
**Cause:** NCCL communication issue between GPUs.
**Fix:**
```bash
export NCCL_IB_DISABLE=1
export NCCL_P2P_DISABLE=1
# Then retry torchrun command
```

## Cost Estimate

| Config | Hourly Cost | Notes |
|--------|-------------|-------|
| 8x A100 SXM | ~$10-13/hr | Best value for full model |
| 8x H100 | ~$20-28/hr | Fastest |
| 8x RTX 6000 Blackwell | ~$8-12/hr | Needs PyTorch nightly |
| 1x A100 + quantized | ~$1.64/hr | Budget option |

## Scripts

- `setup.sh` - Full setup (GPU detection, deps, model download)
- `start.sh` - Start WebSocket server
- `stop.sh` - Stop server gracefully
- `ws_server.py` - WebSocket/HTTP server

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│  Frontend (world-model-frontend)                            │
│  https://world-model-frontend.holly-3f6.workers.dev         │
└─────────────────────────────────────────────────────────────┘
                            │ WebSocket (port 8765)
                            ▼
┌─────────────────────────────────────────────────────────────┐
│  This Server (ws_server.py)                                 │
│  - HTTP health endpoints (port 8080)                        │
│  - Receives WASD controls                                   │
│  - Streams frames at 16fps                                  │
└─────────────────────────────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────┐
│  LingBot-World Inference                                    │
│  - torchrun --nproc_per_node=8 (multi-GPU)                  │
│  - FSDP + DeepSpeed Ulysses                                 │
└─────────────────────────────────────────────────────────────┘
```

## License

Apache 2.0 (same as LingBot-World)
