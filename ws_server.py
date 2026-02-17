#!/usr/bin/env python3
"""
WebSocket server for LingBot-World frame streaming.

Uses subprocess to call generate.py (which handles distributed setup correctly),
then streams the output video frames to clients.
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
import glob
from pathlib import Path
from typing import Optional, List
from http import HTTPStatus
from aiohttp import web
from concurrent.futures import ThreadPoolExecutor

import websockets
from PIL import Image
import io

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
LINGBOT_DIR = os.getenv("LINGBOT_DIR", "/workspace/lingbot-world")
FRAME_WIDTH = int(os.getenv("FRAME_WIDTH", "832"))  # Use smaller size for faster generation
FRAME_HEIGHT = int(os.getenv("FRAME_HEIGHT", "480"))
TARGET_FPS = int(os.getenv("TARGET_FPS", "16"))
FRAMES_PER_BATCH = int(os.getenv("FRAMES_PER_BATCH", "17"))  # Frames to generate at once

# Thread pool for blocking operations
executor = ThreadPoolExecutor(max_workers=4)


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


class WorldModelInference:
    """Wrapper for LingBot-World inference using subprocess."""
    
    def __init__(self):
        self.initialized = False
        self.prompt = ""
        self.current_image_path = None
        self.frame_buffer: queue.Queue = queue.Queue(maxsize=200)
        self.generating = False
        self.generation_thread = None
        self.output_dir = tempfile.mkdtemp(prefix="lingbot_frames_")
        self.batch_count = 0
        
    def _extract_frames_from_video(self, video_path: str) -> List[Image.Image]:
        """Extract frames from generated video using ffmpeg."""
        frames = []
        frame_dir = os.path.join(self.output_dir, f"batch_{self.batch_count}")
        os.makedirs(frame_dir, exist_ok=True)
        
        try:
            # Use ffmpeg to extract frames
            cmd = [
                "ffmpeg", "-y",
                "-i", video_path,
                "-vf", f"fps={TARGET_FPS}",
                os.path.join(frame_dir, "frame_%04d.jpg")
            ]
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
            
            if result.returncode != 0:
                logger.error(f"ffmpeg error: {result.stderr}")
                return frames
            
            # Load extracted frames
            frame_files = sorted(glob.glob(os.path.join(frame_dir, "frame_*.jpg")))
            for frame_file in frame_files:
                try:
                    img = Image.open(frame_file)
                    frames.append(img.copy())
                    img.close()
                except Exception as e:
                    logger.error(f"Error loading frame {frame_file}: {e}")
                    
            logger.info(f"Extracted {len(frames)} frames from video")
            
        except subprocess.TimeoutExpired:
            logger.error("ffmpeg timed out")
        except Exception as e:
            logger.error(f"Error extracting frames: {e}")
            
        return frames
        
    def _generate_video_batch(self) -> Optional[str]:
        """Generate a video using generate.py subprocess."""
        self.batch_count += 1
        output_file = os.path.join(self.output_dir, f"output_{self.batch_count}.mp4")
        
        # Determine image to use
        image_path = self.current_image_path or os.path.join(LINGBOT_DIR, "examples/02/image.jpg")
        
        cmd = [
            "python", "generate.py",
            "--task", "i2v-A14B",
            "--size", f"{FRAME_WIDTH}*{FRAME_HEIGHT}",
            "--ckpt_dir", MODEL_DIR,
            "--image", image_path,
            "--frame_num", str(FRAMES_PER_BATCH),
            "--offload_model", "True",
            "--prompt", self.prompt,
            "--save_file", output_file,
        ]
        
        logger.info(f"Running generate.py: {' '.join(cmd[:10])}...")
        
        try:
            result = subprocess.run(
                cmd,
                cwd=LINGBOT_DIR,
                capture_output=True,
                text=True,
                timeout=300,  # 5 minute timeout
            )
            
            if result.returncode != 0:
                logger.error(f"generate.py failed: {result.stderr[-500:]}")
                return None
                
            if os.path.exists(output_file):
                logger.info(f"Generated video: {output_file}")
                server_state["model_loaded"] = True
                return output_file
            else:
                # Check for default output filename pattern
                pattern = os.path.join(LINGBOT_DIR, "i2v-A14B_*.mp4")
                matches = sorted(glob.glob(pattern), key=os.path.getmtime, reverse=True)
                if matches:
                    latest = matches[0]
                    logger.info(f"Found generated video: {latest}")
                    return latest
                    
                logger.error("No output video found")
                return None
                
        except subprocess.TimeoutExpired:
            logger.error("generate.py timed out after 5 minutes")
            return None
        except Exception as e:
            logger.error(f"Error running generate.py: {e}")
            return None
    
    async def initialize(self, prompt: str, image_path: Optional[str] = None):
        """Initialize the world with a prompt."""
        logger.info(f"Initializing world with prompt: {prompt[:50]}...")
        
        self.prompt = prompt
        self.current_image_path = image_path
        self.initialized = True
        
        # Start background generation
        self.generating = True
        self.generation_thread = threading.Thread(target=self._generation_loop, daemon=True)
        self.generation_thread.start()
        
        return True
    
    def _generation_loop(self):
        """Background thread that continuously generates video batches."""
        while self.generating:
            try:
                # Generate a video
                video_path = self._generate_video_batch()
                
                if video_path:
                    # Extract frames
                    frames = self._extract_frames_from_video(video_path)
                    
                    # Add frames to buffer
                    for frame in frames:
                        if not self.generating:
                            break
                        try:
                            self.frame_buffer.put(frame, timeout=1.0)
                        except queue.Full:
                            # Drop oldest frame if buffer full
                            try:
                                old = self.frame_buffer.get_nowait()
                                old.close()
                                self.frame_buffer.put(frame)
                            except:
                                pass
                    
                    # Save last frame as starting point for next batch
                    if frames:
                        last_frame_path = os.path.join(self.output_dir, "last_frame.jpg")
                        frames[-1].save(last_frame_path, "JPEG", quality=95)
                        self.current_image_path = last_frame_path
                        
                    # Clean up video file
                    try:
                        os.remove(video_path)
                    except:
                        pass
                else:
                    # Generation failed, wait before retry
                    time.sleep(5)
                    
            except Exception as e:
                logger.error(f"Generation loop error: {e}")
                import traceback
                traceback.print_exc()
                time.sleep(5)
        
    async def get_next_frame(self) -> bytes:
        """Get the next frame from the buffer."""
        loop = asyncio.get_event_loop()
        
        try:
            frame = await asyncio.wait_for(
                loop.run_in_executor(executor, lambda: self.frame_buffer.get(timeout=2.0)),
                timeout=3.0
            )
            
            # Encode as JPEG
            buffer = io.BytesIO()
            frame.save(buffer, format='JPEG', quality=85)
            frame.close()
            server_state["total_frames_generated"] += 1
            return buffer.getvalue()
            
        except (asyncio.TimeoutError, queue.Empty):
            return self._create_loading_frame()
    
    def _create_loading_frame(self) -> bytes:
        """Create a loading placeholder frame."""
        from PIL import ImageDraw, ImageFont
        
        img = Image.new('RGB', (FRAME_WIDTH, FRAME_HEIGHT), (20, 20, 40))
        draw = ImageDraw.Draw(img)
        
        try:
            font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 24)
            small_font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 16)
        except:
            font = ImageFont.load_default()
            small_font = font
        
        # Center text
        cx, cy = FRAME_WIDTH // 2, FRAME_HEIGHT // 2
        draw.text((cx - 100, cy - 30), "Generating...", fill=(255, 255, 255), font=font)
        draw.text((cx - 150, cy + 10), f"Batch #{self.batch_count}", fill=(180, 180, 180), font=small_font)
        draw.text((cx - 150, cy + 35), f"Buffer: {self.frame_buffer.qsize()} frames", fill=(150, 150, 150), font=small_font)
        
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
                frame = self.frame_buffer.get_nowait()
                frame.close()
            except:
                break
        # Clean up temp directory
        import shutil
        try:
            shutil.rmtree(self.output_dir)
        except:
            pass


async def handle_client(websocket):
    """Handle a single WebSocket client connection."""
    logger.info(f"Client connected: {websocket.remote_address}")
    server_state["active_connections"] += 1
    
    controls = ControlState()
    inference = WorldModelInference()
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
            
            frame_data = await inference.get_next_frame()
            await websocket.send(frame_data)
            
            elapsed = time.time() - start_time
            sleep_time = max(0, interval - elapsed)
            await asyncio.sleep(sleep_time)
            
    except websockets.exceptions.ConnectionClosed:
        logger.info("Frame streaming stopped - client disconnected")
    except asyncio.CancelledError:
        logger.info("Frame streaming cancelled")
    except Exception as e:
        logger.error(f"Frame streaming error: {e}")


### HTTP Health Endpoints ###

HTTP_PORT = int(os.getenv("HTTP_PORT", "8080"))

async def health_handler(request):
    return web.json_response({
        "status": "ok",
        "service": "world-model-backend",
        "timestamp": time.time(),
    })

async def ready_handler(request):
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
    model_path = Path(MODEL_DIR)
    model_exists = model_path.exists()
    
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
        "version": "2.1.0",
        "config": {
            "ws_port": PORT,
            "http_port": HTTP_PORT,
            "model_dir": MODEL_DIR,
            "lingbot_dir": LINGBOT_DIR,
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

def create_http_app():
    app = web.Application()
    app.router.add_get("/health", health_handler)
    app.router.add_get("/ready", ready_handler)
    app.router.add_get("/status", status_handler)
    return app


async def main():
    server_state["started_at"] = time.time()
    server_state["model_dir"] = MODEL_DIR
    
    logger.info(f"Starting WebSocket server on ws://{HOST}:{PORT}")
    logger.info(f"Starting HTTP server on http://{HOST}:{HTTP_PORT}")
    logger.info(f"Model: {MODEL_DIR}")
    logger.info(f"LingBot dir: {LINGBOT_DIR}")
    logger.info(f"Frame size: {FRAME_WIDTH}x{FRAME_HEIGHT} @ {TARGET_FPS}fps")
    logger.info(f"Frames per batch: {FRAMES_PER_BATCH}")
    
    http_app = create_http_app()
    http_runner = web.AppRunner(http_app)
    await http_runner.setup()
    http_site = web.TCPSite(http_runner, HOST, HTTP_PORT)
    await http_site.start()
    
    async with websockets.serve(handle_client, HOST, PORT):
        logger.info("Servers running. Press Ctrl+C to stop.")
        await asyncio.Future()


if __name__ == "__main__":
    asyncio.run(main())
