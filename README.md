# World Model Backend - LingBot-World on RunPod

RunPod deployment for LingBot-World world model inference with WebSocket frame streaming.

## Requirements

- **8x A100 80GB** (full model) OR
- **1x A100/L40S** (quantized model)
- ~100GB storage for model weights

## Quick Start

### Option 1: RunPod Template (Recommended)

1. Go to RunPod Console → Templates
2. Create from this Docker image: `moltyfromclaw/world-model-backend:latest`
3. Deploy as GPU Pod (8x A100 SXM)

### Option 2: Manual Setup

```bash
# SSH into RunPod instance
git clone https://github.com/moltyfromclaw/world-model-backend.git
cd world-model-backend
./setup.sh
```

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│  Frontend (Cloudflare Workers)                              │
│  world-model-frontend                                       │
└─────────────────────────────────────────────────────────────┘
                            │ WebSocket
                            ▼
┌─────────────────────────────────────────────────────────────┐
│  WebSocket Server (this repo)                               │
│  - Receives WASD controls                                   │
│  - Streams frames back at 16fps                             │
└─────────────────────────────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────┐
│  LingBot-World Inference                                    │
│  - torchrun --nproc_per_node=8                              │
│  - FSDP + DeepSpeed Ulysses                                 │
└─────────────────────────────────────────────────────────────┘
```

## API

### WebSocket Messages

**Client → Server:**
```json
// Initialize world
{ "type": "init", "prompt": "A medieval castle..." }

// Control input
{ "type": "control", "key": "w", "action": "down" }
{ "type": "control", "key": "w", "action": "up" }
```

**Server → Client:**
```json
// Frame data (binary JPEG/PNG)
<binary frame data>

// Or JSON with base64
{ "type": "frame", "image": "data:image/jpeg;base64,..." }
```

## Cost Estimate

| Config | Hourly Cost | Notes |
|--------|-------------|-------|
| 8x A100 SXM | ~$13-16/hr | Full model, best quality |
| 4x A100 | ~$6-8/hr | Reduced frames/resolution |
| 1x A100 + quantized | ~$1.64/hr | 4-bit model, some quality loss |

## Scripts

- `setup.sh` - Install dependencies and download model
- `start.sh` - Start WebSocket server
- `stop.sh` - Stop server gracefully

## License

Apache 2.0 (same as LingBot-World)
