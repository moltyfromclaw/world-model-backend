#!/usr/bin/env python3
"""
WebSocket server for LingBot-World frame streaming.

Receives control inputs (WASD), runs inference, streams frames back.
Also provides HTTP health endpoints for testing without WebSocket.
"""

import asyncio
import json
import logging
import os
import subprocess
import sys
import tempfile
import time
import threading
import queue
from pathlib import Path
from typing import Optional, List
from http import HTTPStatus
from aiohttp import web
from concurrent.futures import ThreadPoolExecutor

import websockets
from PIL import Image
import io
import numpy as np

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global state for health checks
server_state = {
    "started_at": None,
    "model_loaded": False,
    "model_dir": None,
    "active_connections": 0,
    "total_frames_generated": 0,
}

# Configuration
HOST = os.getenv("WS_HOST", "0.0.0.0")
PORT = int(os.getenv("WS_PORT", "8765"))
MODEL_DIR = os.getenv("MODEL_DIR", "/workspace/lingbot-world-base-cam")
FRAME_WIDTH = int(os.getenv("FRAME_WIDTH", "1280"))
FRAME_HEIGHT = int(os.getenv("FRAME_HEIGHT", "720"))
TARGET_FPS = int(os.getenv("TARGET_FPS", "16"))
NUM_GPUS = int(os.getenv("NUM_GPUS", "8"))
FRAMES_PER_BATCH = int(os.getenv("FRAMES_PER_BATCH", "17"))  # Frames to generate at once

# Thread pool for blocking inference
executor = ThreadPoolExecutor(max_workers=2)

# Control state
class ControlState:
    def __init__(self):
        self.w = False
        self.a = False
        self.s = False
        self.d = False
        self.prompt = ""
        
    def update(self, key: str, action: str):
        if hasattr(self, key):
            setattr(self, key, action == "down")
            
    def to_action_string(self) -> str:
        """Convert controls to action string for model."""
        actions = []
        if self.w: actions.append("forward")
        if self.s: actions.append("backward")
        if self.a: actions.append("left")
        if self.d: actions.append("right")
        return ",".join(actions) if actions else "none"


class WorldModelInference:
    """Wrapper for LingBot-World inference using the actual model."""
    
    def __init__(self, model_dir: str, num_gpus: int = 8):
        self.model_dir = model_dir
        self.num_gpus = num_gpus
        self.initialized = False
        self.wan_pipeline = None
        self.current_image = None
        self.prompt = ""
        self.frame_buffer: queue.Queue = queue.Queue(maxsize=100)
        self.generating = False
        self.generation_thread = None
        
    def _load_model(self):
        """Load the WanI2V model (blocking, run in thread)."""
        import torch
        import torch.distributed as dist
        
        # Add lingbot-world to path
        lingbot_path = "/workspace/lingbot-world"
        if lingbot_path not in sys.path:
            sys.path.insert(0, lingbot_path)
        
        import wan
        from wan.configs import WAN_CONFIGS, MAX_AREA_CONFIGS
        
        logger.info("Loading WanI2V model...")
        
        rank = int(os.getenv("RANK", 0))
        world_size = int(os.getenv("WORLD_SIZE", 1))
        local_rank = int(os.getenv("LOCAL_RANK", 0))
        
        cfg = WAN_CONFIGS["i2v-A14B"]
        
        self.wan_pipeline = wan.WanI2V(
            config=cfg,
            checkpoint_dir=self.model_dir,
            device_id=local_rank,
            rank=rank,
            t5_fsdp=(world_size > 1),
            dit_fsdp=(world_size > 1),
            use_sp=(self.num_gpus > 1),
            t5_cpu=False,
        )
        
        self.cfg = cfg
        self.max_area = MAX_AREA_CONFIGS[f"{FRAME_WIDTH}*{FRAME_HEIGHT}"]
        
        logger.info("Model loaded successfully!")
        server_state["model_loaded"] = True
        
    def _generate_batch(self, prompt: str, image: Image.Image, frame_num: int = 17) -> List[Image.Image]:
        """Generate a batch of frames (blocking, run in thread)."""
        import torch
        from wan.utils.utils import save_video
        
        logger.info(f"Generating {frame_num} frames...")
        
        video_tensor = self.wan_pipeline.generate(
            prompt,
            image,
            max_area=self.max_area,
            frame_num=frame_num,
            shift=self.cfg.sample_shift,
            sample_solver='unipc',
            sampling_steps=self.cfg.sample_steps,
            guide_scale=self.cfg.sample_guide_scale,
            seed=int(time.time()) % 10000,
            offload_model=False,
        )
        
        # Convert tensor to list of PIL images
        # video_tensor shape: [frames, channels, height, width]
        frames = []
        video_np = video_tensor.cpu().numpy()
        
        # Normalize from [-1, 1] to [0, 255]
        video_np = ((video_np + 1) / 2 * 255).clip(0, 255).astype(np.uint8)
        
        for i in range(video_np.shape[0]):
            frame = video_np[i].transpose(1, 2, 0)  # CHW -> HWC
            frames.append(Image.fromarray(frame))
            
        logger.info(f"Generated {len(frames)} frames")
        return frames
        
    async def initialize(self, prompt: str, image_path: Optional[str] = None):
        """Initialize the world with a prompt and optional starting image."""
        logger.info(f"Initializing world with prompt: {prompt[:50]}...")
        
        self.prompt = prompt
        
        # Load starting image
        if image_path and os.path.exists(image_path):
            self.current_image = Image.open(image_path).convert("RGB")
        else:
            # Use default example image
            default_image = "/workspace/lingbot-world/examples/02/image.jpg"
            if os.path.exists(default_image):
                self.current_image = Image.open(default_image).convert("RGB")
            else:
                # Create a placeholder image
                self.current_image = Image.new("RGB", (FRAME_WIDTH, FRAME_HEIGHT), (50, 50, 80))
        
        # Resize to target dimensions
        self.current_image = self.current_image.resize((FRAME_WIDTH, FRAME_HEIGHT), Image.LANCZOS)
        
        # Load model if not loaded
        if not server_state["model_loaded"]:
            loop = asyncio.get_event_loop()
            await loop.run_in_executor(executor, self._load_model)
        
        self.initialized = True
        
        # Start background generation
        self.generating = True
        self.generation_thread = threading.Thread(target=self._generation_loop, daemon=True)
        self.generation_thread.start()
        
        return True
    
    def _generation_loop(self):
        """Background thread that continuously generates frames."""
        import torch
        
        while self.generating:
            try:
                # Generate a batch of frames
                frames = self._generate_batch(
                    self.prompt,
                    self.current_image,
                    frame_num=FRAMES_PER_BATCH
                )
                
                # Add frames to buffer
                for frame in frames:
                    if not self.generating:
                        break
                    try:
                        self.frame_buffer.put(frame, timeout=1.0)
                    except queue.Full:
                        # Drop oldest frame if buffer full
                        try:
                            self.frame_buffer.get_nowait()
                            self.frame_buffer.put(frame)
                        except:
                            pass
                
                # Use last frame as input for next batch (for continuity)
                if frames:
                    self.current_image = frames[-1]
                    
            except Exception as e:
                logger.error(f"Generation error: {e}")
                import traceback
                traceback.print_exc()
                time.sleep(1)  # Back off on error
        
    async def get_next_frame(self) -> bytes:
        """Get the next frame from the buffer."""
        loop = asyncio.get_event_loop()
        
        try:
            # Try to get frame from buffer
            frame = await asyncio.wait_for(
                loop.run_in_executor(executor, lambda: self.frame_buffer.get(timeout=2.0)),
                timeout=3.0
            )
            
            # Encode as JPEG
            buffer = io.BytesIO()
            frame.save(buffer, format='JPEG', quality=85)
            server_state["total_frames_generated"] += 1
            return buffer.getvalue()
            
        except (asyncio.TimeoutError, queue.Empty):
            # Buffer empty, return placeholder
            return self._create_loading_frame()
    
    def _create_loading_frame(self) -> bytes:
        """Create a loading placeholder frame."""
        from PIL import ImageDraw, ImageFont
        
        img = Image.new('RGB', (FRAME_WIDTH, FRAME_HEIGHT), (20, 20, 40))
        draw = ImageDraw.Draw(img)
        
        try:
            font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 36)
            small_font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 24)
        except:
            font = ImageFont.load_default()
            small_font = font
        
        draw.text((FRAME_WIDTH//2 - 150, FRAME_HEIGHT//2 - 40), "Generating...", fill=(255, 255, 255), font=font)
        draw.text((FRAME_WIDTH//2 - 200, FRAME_HEIGHT//2 + 20), f"Prompt: {self.prompt[:40]}...", fill=(180, 180, 180), font=small_font)
        draw.text((FRAME_WIDTH//2 - 150, FRAME_HEIGHT//2 + 60), f"Buffer: {self.frame_buffer.qsize()} frames", fill=(150, 150, 150), font=small_font)
        
        buffer = io.BytesIO()
        img.save(buffer, format='JPEG', quality=85)
        return buffer.getvalue()
        
    def cleanup(self):
        """Clean up resources."""
        self.generating = False
        if self.generation_thread:
            self.generation_thread.join(timeout=5.0)
        # Clear buffer
        while not self.frame_buffer.empty():
            try:
                self.frame_buffer.get_nowait()
            except:
                break


async def handle_client(websocket):
    """Handle a single WebSocket client connection."""
    logger.info(f"Client connected: {websocket.remote_address}")
    server_state["active_connections"] += 1
    
    controls = ControlState()
    inference = WorldModelInference(MODEL_DIR, NUM_GPUS)
    frame_interval = 1.0 / TARGET_FPS
    streaming_task = None
    
    try:
        async for message in websocket:
            try:
                data = json.loads(message)
                msg_type = data.get("type")
                
                if msg_type == "init":
                    prompt = data.get("prompt", "A beautiful landscape")
                    image_path = data.get("image_path")
                    controls.prompt = prompt
                    await inference.initialize(prompt, image_path)
                    
                    # Start frame streaming
                    streaming_task = asyncio.create_task(stream_frames(websocket, inference, controls, frame_interval))
                    
                elif msg_type == "control":
                    key = data.get("key", "").lower()
                    action = data.get("action", "up")
                    controls.update(key, action)
                    
            except json.JSONDecodeError:
                logger.warning(f"Invalid JSON: {message}")
                
    except websockets.exceptions.ConnectionClosed:
        logger.info(f"Client disconnected: {websocket.remote_address}")
    finally:
        server_state["active_connections"] -= 1
        if streaming_task:
            streaming_task.cancel()
        inference.cleanup()


async def stream_frames(websocket, inference: WorldModelInference, controls: ControlState, interval: float):
    """Stream frames to client at target FPS."""
    logger.info("Starting frame streaming")
    
    try:
        while True:
            start_time = time.time()
            
            # Get next frame from buffer
            frame_data = await inference.get_next_frame()
            
            # Send as binary
            await websocket.send(frame_data)
            
            # Maintain target FPS
            elapsed = time.time() - start_time
            sleep_time = max(0, interval - elapsed)
            await asyncio.sleep(sleep_time)
            
    except websockets.exceptions.ConnectionClosed:
        logger.info("Frame streaming stopped - client disconnected")
    except asyncio.CancelledError:
        logger.info("Frame streaming cancelled")
    except Exception as e:
        logger.error(f"Frame streaming error: {e}")
        import traceback
        traceback.print_exc()


### HTTP Health Endpoints ###

HTTP_PORT = int(os.getenv("HTTP_PORT", "8080"))

async def health_handler(request):
    """Basic health check - is the server running?"""
    return web.json_response({
        "status": "ok",
        "service": "world-model-backend",
        "timestamp": time.time(),
    })

async def ready_handler(request):
    """Readiness check - is the model loaded and ready?"""
    ready = server_state["model_loaded"]
    status = HTTPStatus.OK if ready else HTTPStatus.SERVICE_UNAVAILABLE
    return web.json_response({
        "ready": ready,
        "model_dir": server_state["model_dir"],
        "uptime_seconds": time.time() - server_state["started_at"] if server_state["started_at"] else 0,
        "active_connections": server_state["active_connections"],
        "total_frames_generated": server_state["total_frames_generated"],
    }, status=status)

async def status_handler(request):
    """Full status with all details."""
    # Check if model files exist
    model_path = Path(MODEL_DIR)
    model_exists = model_path.exists()
    
    # Check GPU availability
    gpu_info = "unknown"
    try:
        result = subprocess.run(["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader"], 
                                capture_output=True, text=True, timeout=5)
        if result.returncode == 0:
            gpu_info = result.stdout.strip().split("\n")
    except:
        gpu_info = "nvidia-smi not available"
    
    return web.json_response({
        "status": "ok",
        "service": "world-model-backend",
        "version": "2.0.0",
        "config": {
            "ws_port": PORT,
            "http_port": HTTP_PORT,
            "model_dir": MODEL_DIR,
            "num_gpus": NUM_GPUS,
            "frame_size": f"{FRAME_WIDTH}x{FRAME_HEIGHT}",
            "target_fps": TARGET_FPS,
            "frames_per_batch": FRAMES_PER_BATCH,
        },
        "state": {
            "started_at": server_state["started_at"],
            "uptime_seconds": time.time() - server_state["started_at"] if server_state["started_at"] else 0,
            "model_loaded": server_state["model_loaded"],
            "model_exists": model_exists,
            "active_connections": server_state["active_connections"],
            "total_frames_generated": server_state["total_frames_generated"],
        },
        "gpu_info": gpu_info,
    })

async def test_frame_handler(request):
    """Generate a single test frame - loads model if needed."""
    inference = WorldModelInference(MODEL_DIR, NUM_GPUS)
    
    prompt = request.query.get("prompt", "A test scene")
    
    try:
        await inference.initialize(prompt)
        
        # Wait for first frame
        frame_data = await inference.get_next_frame()
        inference.cleanup()
        
        return web.Response(
            body=frame_data,
            content_type="image/jpeg",
            headers={"X-Prompt": prompt[:50]}
        )
    except Exception as e:
        logger.error(f"Test frame error: {e}")
        import traceback
        traceback.print_exc()
        return web.json_response({
            "error": str(e),
            "status": "inference_failed"
        }, status=HTTPStatus.INTERNAL_SERVER_ERROR)
    finally:
        inference.cleanup()

def create_http_app():
    """Create the HTTP app with health endpoints."""
    app = web.Application()
    app.router.add_get("/health", health_handler)
    app.router.add_get("/ready", ready_handler)
    app.router.add_get("/status", status_handler)
    app.router.add_get("/test-frame", test_frame_handler)
    return app


async def main():
    """Start both WebSocket and HTTP servers."""
    server_state["started_at"] = time.time()
    server_state["model_dir"] = MODEL_DIR
    
    logger.info(f"Starting WebSocket server on ws://{HOST}:{PORT}")
    logger.info(f"Starting HTTP server on http://{HOST}:{HTTP_PORT}")
    logger.info(f"Model: {MODEL_DIR}, GPUs: {NUM_GPUS}, Target FPS: {TARGET_FPS}")
    logger.info(f"Health endpoints: /health, /ready, /status, /test-frame")
    
    # Start HTTP server
    http_app = create_http_app()
    http_runner = web.AppRunner(http_app)
    await http_runner.setup()
    http_site = web.TCPSite(http_runner, HOST, HTTP_PORT)
    await http_site.start()
    
    # Start WebSocket server
    async with websockets.serve(handle_client, HOST, PORT):
        logger.info("Servers running. Press Ctrl+C to stop.")
        await asyncio.Future()  # Run forever


if __name__ == "__main__":
    asyncio.run(main())
