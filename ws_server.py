#!/usr/bin/env python3
"""
WebSocket server for LingBot-World frame streaming.

Uses subprocess to call generate.py exactly as it works from CLI,
then extracts frames from the output video and streams them.
"""

import asyncio
import json
import logging
import os
import subprocess
import shutil
import sys
import tempfile
import time
import threading
import queue
import glob
from pathlib import Path
from typing import Optional, List
from http import HTTPStatus
from aiohttp import web
from concurrent.futures import ThreadPoolExecutor

import websockets
from websockets.server import serve
from PIL import Image
import io

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Global state
server_state = {
    "started_at": None,
    "model_loaded": False,
    "active_connections": 0,
    "total_frames_generated": 0,
    "last_error": None,
}

# Configuration - use environment variables with sensible defaults
HOST = os.getenv("WS_HOST", "0.0.0.0")
WS_PORT = int(os.getenv("WS_PORT", "8765"))
HTTP_PORT = int(os.getenv("HTTP_PORT", "8080"))

# Paths - these should match the RunPod setup
LINGBOT_DIR = os.getenv("LINGBOT_DIR", "/workspace/lingbot-world")
MODEL_DIR = os.getenv("MODEL_DIR", "lingbot-world-base-cam")  # Relative to LINGBOT_DIR
DEFAULT_IMAGE = os.getenv("DEFAULT_IMAGE", "examples/02/image.jpg")  # Relative to LINGBOT_DIR

# Generation settings
FRAME_SIZE = os.getenv("FRAME_SIZE", "832*480")  # Use * not x, as generate.py expects
FRAMES_PER_BATCH = int(os.getenv("FRAMES_PER_BATCH", "17"))
TARGET_FPS = int(os.getenv("TARGET_FPS", "16"))

# Thread pool
executor = ThreadPoolExecutor(max_workers=4)


class ControlState:
    """Track WASD control state."""
    def __init__(self):
        self.w = False
        self.a = False
        self.s = False
        self.d = False
        self.prompt = ""
        
    def update(self, key: str, action: str):
        if key in ('w', 'a', 's', 'd'):
            setattr(self, key, action == "down")


class WorldModelInference:
    """
    Manages LingBot-World inference using subprocess calls to generate.py.
    
    Flow:
    1. Client sends init with prompt
    2. Start background thread that calls generate.py in a loop
    3. Each call generates a short video (17 frames)
    4. Extract frames with ffmpeg and add to buffer
    5. Stream frames to client while next batch generates
    """
    
    def __init__(self):
        self.initialized = False
        self.prompt = ""
        self.current_image = None  # Path to current starting image
        self.frame_buffer: queue.Queue = queue.Queue(maxsize=200)
        self.generating = False
        self.generation_thread = None
        self.temp_dir = None
        self.batch_count = 0
        self.error_count = 0
        
    def _setup_temp_dir(self):
        """Create temp directory for this session."""
        self.temp_dir = tempfile.mkdtemp(prefix="lingbot_session_")
        logger.info(f"Created temp directory: {self.temp_dir}")
        
    def _cleanup_temp_dir(self):
        """Remove temp directory."""
        if self.temp_dir and os.path.exists(self.temp_dir):
            try:
                shutil.rmtree(self.temp_dir)
                logger.info(f"Cleaned up temp directory: {self.temp_dir}")
            except Exception as e:
                logger.warning(f"Failed to cleanup temp dir: {e}")
                
    def _run_generate(self, output_path: str) -> bool:
        """
        Run generate.py to create a video.
        Returns True on success, False on failure.
        """
        # Determine input image
        if self.current_image and os.path.exists(self.current_image):
            image_path = self.current_image
        else:
            image_path = os.path.join(LINGBOT_DIR, DEFAULT_IMAGE)
            
        if not os.path.exists(image_path):
            logger.error(f"Input image not found: {image_path}")
            return False
            
        # Build command - exactly like the working CLI command
        cmd = [
            sys.executable,  # Use same Python interpreter
            "generate.py",
            "--task", "i2v-A14B",
            "--size", FRAME_SIZE,
            "--ckpt_dir", MODEL_DIR,
            "--image", image_path,
            "--frame_num", str(FRAMES_PER_BATCH),
            "--offload_model", "True",
            "--prompt", self.prompt,
            "--save_file", output_path,
        ]
        
        logger.info(f"Running generate.py (batch #{self.batch_count})...")
        logger.debug(f"Command: {' '.join(cmd)}")
        
        try:
            result = subprocess.run(
                cmd,
                cwd=LINGBOT_DIR,
                capture_output=True,
                text=True,
                timeout=600,  # 10 minute timeout
                env={**os.environ, "PYTHONUNBUFFERED": "1"}
            )
            
            if result.returncode != 0:
                logger.error(f"generate.py failed (exit code {result.returncode})")
                logger.error(f"STDERR: {result.stderr[-1000:]}")
                server_state["last_error"] = result.stderr[-500:]
                return False
                
            if not os.path.exists(output_path):
                # Check for auto-generated filename
                pattern = os.path.join(LINGBOT_DIR, "i2v-A14B_*.mp4")
                matches = sorted(glob.glob(pattern), key=os.path.getmtime, reverse=True)
                if matches:
                    # Move to expected location
                    shutil.move(matches[0], output_path)
                else:
                    logger.error("No output video file found")
                    return False
                    
            logger.info(f"Generated video: {output_path}")
            server_state["model_loaded"] = True
            return True
            
        except subprocess.TimeoutExpired:
            logger.error("generate.py timed out after 10 minutes")
            server_state["last_error"] = "Generation timeout"
            return False
        except Exception as e:
            logger.error(f"Error running generate.py: {e}")
            server_state["last_error"] = str(e)
            return False
            
    def _extract_frames(self, video_path: str) -> List[Image.Image]:
        """Extract frames from video using ffmpeg."""
        frames = []
        frame_dir = os.path.join(self.temp_dir, f"frames_{self.batch_count}")
        os.makedirs(frame_dir, exist_ok=True)
        
        cmd = [
            "ffmpeg", "-y",
            "-i", video_path,
            "-vf", f"fps={TARGET_FPS}",
            "-q:v", "2",  # High quality JPEG
            os.path.join(frame_dir, "frame_%04d.jpg")
        ]
        
        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=60
            )
            
            if result.returncode != 0:
                logger.error(f"ffmpeg failed: {result.stderr[-500:]}")
                return frames
                
            # Load frames
            frame_files = sorted(glob.glob(os.path.join(frame_dir, "frame_*.jpg")))
            for f in frame_files:
                try:
                    img = Image.open(f)
                    frames.append(img.copy())
                    img.close()
                except Exception as e:
                    logger.warning(f"Failed to load frame {f}: {e}")
                    
            logger.info(f"Extracted {len(frames)} frames")
            
            # Cleanup frame files
            shutil.rmtree(frame_dir, ignore_errors=True)
            
        except Exception as e:
            logger.error(f"Frame extraction error: {e}")
            
        return frames
        
    def _generation_loop(self):
        """Background thread that continuously generates and buffers frames."""
        logger.info("Generation loop started")
        
        while self.generating:
            self.batch_count += 1
            output_path = os.path.join(self.temp_dir, f"video_{self.batch_count}.mp4")
            
            try:
                # Generate video
                success = self._run_generate(output_path)
                
                if success:
                    self.error_count = 0
                    
                    # Extract frames
                    frames = self._extract_frames(output_path)
                    
                    if frames:
                        # Add to buffer
                        for frame in frames:
                            if not self.generating:
                                break
                            try:
                                self.frame_buffer.put(frame, timeout=2.0)
                            except queue.Full:
                                # Drop oldest
                                try:
                                    old = self.frame_buffer.get_nowait()
                                    old.close()
                                except:
                                    pass
                                self.frame_buffer.put(frame)
                                
                        # Use last frame as next starting point
                        last_frame_path = os.path.join(self.temp_dir, "last_frame.jpg")
                        frames[-1].save(last_frame_path, "JPEG", quality=95)
                        self.current_image = last_frame_path
                        
                    # Cleanup video
                    try:
                        os.remove(output_path)
                    except:
                        pass
                else:
                    self.error_count += 1
                    if self.error_count >= 3:
                        logger.error("Too many consecutive errors, stopping generation")
                        break
                    time.sleep(5)  # Back off on error
                    
            except Exception as e:
                logger.error(f"Generation loop error: {e}")
                import traceback
                traceback.print_exc()
                self.error_count += 1
                time.sleep(5)
                
        logger.info("Generation loop stopped")
        
    async def initialize(self, prompt: str, image_path: Optional[str] = None):
        """Start generation with given prompt."""
        logger.info(f"Initializing with prompt: {prompt[:80]}...")
        
        self.prompt = prompt
        self.current_image = image_path
        self._setup_temp_dir()
        self.initialized = True
        self.generating = True
        self.error_count = 0
        self.batch_count = 0
        
        # Start background generation
        self.generation_thread = threading.Thread(
            target=self._generation_loop,
            daemon=True,
            name="GenerationLoop"
        )
        self.generation_thread.start()
        
        return True
        
    async def get_next_frame(self) -> bytes:
        """Get next frame from buffer, or loading placeholder."""
        loop = asyncio.get_event_loop()
        
        try:
            frame = await asyncio.wait_for(
                loop.run_in_executor(
                    executor,
                    lambda: self.frame_buffer.get(timeout=2.0)
                ),
                timeout=3.0
            )
            
            buffer = io.BytesIO()
            frame.save(buffer, format='JPEG', quality=85)
            frame.close()
            server_state["total_frames_generated"] += 1
            return buffer.getvalue()
            
        except (asyncio.TimeoutError, queue.Empty):
            return self._create_placeholder()
            
    def _create_placeholder(self) -> bytes:
        """Create loading/generating placeholder frame."""
        # Parse size
        try:
            w, h = map(int, FRAME_SIZE.split('*'))
        except:
            w, h = 832, 480
            
        img = Image.new('RGB', (w, h), (15, 15, 25))
        
        try:
            from PIL import ImageDraw, ImageFont
            draw = ImageDraw.Draw(img)
            
            try:
                font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 28)
                small = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 18)
            except:
                font = small = ImageFont.load_default()
                
            cx, cy = w // 2, h // 2
            
            if self.error_count > 0:
                draw.text((cx - 80, cy - 20), "Error - Retrying...", fill=(255, 100, 100), font=font)
            else:
                draw.text((cx - 80, cy - 20), "Generating...", fill=(255, 255, 255), font=font)
                
            draw.text((cx - 100, cy + 20), f"Batch #{self.batch_count} | Buffer: {self.frame_buffer.qsize()}", 
                     fill=(150, 150, 150), font=small)
                     
        except Exception as e:
            logger.warning(f"Failed to draw placeholder text: {e}")
            
        buffer = io.BytesIO()
        img.save(buffer, format='JPEG', quality=80)
        return buffer.getvalue()
        
    def cleanup(self):
        """Stop generation and cleanup."""
        logger.info("Cleaning up inference...")
        self.generating = False
        
        if self.generation_thread and self.generation_thread.is_alive():
            self.generation_thread.join(timeout=10.0)
            
        # Clear buffer
        while not self.frame_buffer.empty():
            try:
                frame = self.frame_buffer.get_nowait()
                frame.close()
            except:
                break
                
        self._cleanup_temp_dir()


async def handle_client(websocket):
    """Handle WebSocket client connection."""
    client_addr = websocket.remote_address
    logger.info(f"Client connected: {client_addr}")
    server_state["active_connections"] += 1
    
    inference = WorldModelInference()
    streaming_task = None
    
    try:
        async for message in websocket:
            try:
                data = json.loads(message)
                msg_type = data.get("type")
                
                if msg_type == "init":
                    prompt = data.get("prompt", "A beautiful landscape")
                    image_path = data.get("image_path")
                    
                    logger.info(f"Init request - prompt: {prompt[:50]}...")
                    await inference.initialize(prompt, image_path)
                    
                    # Start streaming
                    frame_interval = 1.0 / TARGET_FPS
                    streaming_task = asyncio.create_task(
                        stream_frames(websocket, inference, frame_interval)
                    )
                    
                elif msg_type == "control":
                    # For now, just log controls (future: influence generation)
                    key = data.get("key", "").lower()
                    action = data.get("action", "up")
                    logger.debug(f"Control: {key} {action}")
                    
            except json.JSONDecodeError:
                logger.warning(f"Invalid JSON received")
                
    except websockets.exceptions.ConnectionClosed:
        logger.info(f"Client disconnected: {client_addr}")
    except Exception as e:
        logger.error(f"Client handler error: {e}")
    finally:
        server_state["active_connections"] -= 1
        if streaming_task:
            streaming_task.cancel()
            try:
                await streaming_task
            except asyncio.CancelledError:
                pass
        inference.cleanup()


async def stream_frames(websocket, inference: WorldModelInference, interval: float):
    """Stream frames to client at target FPS."""
    logger.info("Starting frame stream")
    
    try:
        while True:
            start = time.time()
            
            frame_data = await inference.get_next_frame()
            await websocket.send(frame_data)
            
            elapsed = time.time() - start
            await asyncio.sleep(max(0, interval - elapsed))
            
    except websockets.exceptions.ConnectionClosed:
        logger.info("Stream ended - client disconnected")
    except asyncio.CancelledError:
        logger.info("Stream cancelled")
    except Exception as e:
        logger.error(f"Stream error: {e}")


# ============ HTTP Endpoints ============

async def health_handler(request):
    return web.json_response({
        "status": "ok",
        "service": "world-model-backend",
        "version": "2.2.0",
        "timestamp": time.time(),
    })

async def ready_handler(request):
    ready = server_state["model_loaded"]
    return web.json_response({
        "ready": ready,
        "uptime": time.time() - server_state["started_at"] if server_state["started_at"] else 0,
        "connections": server_state["active_connections"],
        "frames_generated": server_state["total_frames_generated"],
        "last_error": server_state["last_error"],
    }, status=HTTPStatus.OK if ready else HTTPStatus.SERVICE_UNAVAILABLE)

async def status_handler(request):
    # Check paths
    lingbot_exists = os.path.isdir(LINGBOT_DIR)
    model_path = os.path.join(LINGBOT_DIR, MODEL_DIR)
    model_exists = os.path.isdir(model_path)
    generate_exists = os.path.isfile(os.path.join(LINGBOT_DIR, "generate.py"))
    
    # GPU info
    gpu_info = []
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,memory.total,memory.free", "--format=csv,noheader"],
            capture_output=True, text=True, timeout=5
        )
        if result.returncode == 0:
            gpu_info = result.stdout.strip().split("\n")
    except:
        gpu_info = ["nvidia-smi not available"]
    
    return web.json_response({
        "status": "ok",
        "service": "world-model-backend",
        "version": "2.2.0",
        "config": {
            "ws_port": WS_PORT,
            "http_port": HTTP_PORT,
            "lingbot_dir": LINGBOT_DIR,
            "model_dir": MODEL_DIR,
            "frame_size": FRAME_SIZE,
            "target_fps": TARGET_FPS,
            "frames_per_batch": FRAMES_PER_BATCH,
        },
        "paths": {
            "lingbot_exists": lingbot_exists,
            "model_exists": model_exists,
            "generate_exists": generate_exists,
        },
        "state": {
            "started_at": server_state["started_at"],
            "uptime": time.time() - server_state["started_at"] if server_state["started_at"] else 0,
            "model_loaded": server_state["model_loaded"],
            "active_connections": server_state["active_connections"],
            "total_frames_generated": server_state["total_frames_generated"],
            "last_error": server_state["last_error"],
        },
        "gpu_info": gpu_info,
    })


def create_http_app():
    app = web.Application()
    app.router.add_get("/health", health_handler)
    app.router.add_get("/ready", ready_handler)
    app.router.add_get("/status", status_handler)
    return app


async def main():
    """Start WebSocket and HTTP servers."""
    server_state["started_at"] = time.time()
    
    # Validate setup
    logger.info("=" * 50)
    logger.info("LingBot-World WebSocket Server v2.2.0")
    logger.info("=" * 50)
    logger.info(f"LINGBOT_DIR: {LINGBOT_DIR}")
    logger.info(f"MODEL_DIR: {MODEL_DIR}")
    logger.info(f"FRAME_SIZE: {FRAME_SIZE}")
    logger.info(f"TARGET_FPS: {TARGET_FPS}")
    logger.info(f"FRAMES_PER_BATCH: {FRAMES_PER_BATCH}")
    
    # Check paths
    if not os.path.isdir(LINGBOT_DIR):
        logger.error(f"LINGBOT_DIR not found: {LINGBOT_DIR}")
        sys.exit(1)
        
    generate_py = os.path.join(LINGBOT_DIR, "generate.py")
    if not os.path.isfile(generate_py):
        logger.error(f"generate.py not found: {generate_py}")
        sys.exit(1)
        
    model_path = os.path.join(LINGBOT_DIR, MODEL_DIR)
    if not os.path.isdir(model_path):
        logger.warning(f"Model directory not found: {model_path}")
        logger.warning("Generation may fail - check MODEL_DIR setting")
    
    logger.info("=" * 50)
    
    # Start HTTP server
    http_app = create_http_app()
    http_runner = web.AppRunner(http_app)
    await http_runner.setup()
    http_site = web.TCPSite(http_runner, HOST, HTTP_PORT)
    await http_site.start()
    logger.info(f"HTTP server: http://{HOST}:{HTTP_PORT}")
    
    # Start WebSocket server with permissive settings for proxied connections
    async with websockets.serve(
        handle_client, 
        HOST, 
        WS_PORT,
        # Allow connections from any origin (for RunPod proxy)
        origins=None,
        # Increase timeouts for slow connections
        ping_interval=30,
        ping_timeout=30,
        close_timeout=10,
    ):
        logger.info(f"WebSocket server: ws://{HOST}:{WS_PORT}")
        logger.info("Ready for connections!")
        await asyncio.Future()


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("Shutting down...")
