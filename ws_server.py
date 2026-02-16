#!/usr/bin/env python3
"""
WebSocket server for LingBot-World frame streaming.

Receives control inputs (WASD), runs inference, streams frames back.
"""

import asyncio
import json
import logging
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Optional

import websockets
from PIL import Image
import io
import base64

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configuration
HOST = os.getenv("WS_HOST", "0.0.0.0")
PORT = int(os.getenv("WS_PORT", "8765"))
MODEL_DIR = os.getenv("MODEL_DIR", "lingbot-world-base-cam")
FRAME_WIDTH = int(os.getenv("FRAME_WIDTH", "1280"))
FRAME_HEIGHT = int(os.getenv("FRAME_HEIGHT", "720"))
TARGET_FPS = int(os.getenv("TARGET_FPS", "16"))
NUM_GPUS = int(os.getenv("NUM_GPUS", "8"))

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
            
    def to_camera_delta(self) -> tuple[float, float, float]:
        """Convert controls to camera movement delta."""
        dx, dy, dz = 0.0, 0.0, 0.0
        if self.w: dz -= 0.1  # Forward
        if self.s: dz += 0.1  # Backward
        if self.a: dx -= 0.05  # Turn left
        if self.d: dx += 0.05  # Turn right
        return dx, dy, dz


class WorldModelInference:
    """Wrapper for LingBot-World inference."""
    
    def __init__(self, model_dir: str, num_gpus: int = 8):
        self.model_dir = model_dir
        self.num_gpus = num_gpus
        self.process: Optional[subprocess.Popen] = None
        self.output_dir = tempfile.mkdtemp(prefix="lingbot_")
        self.frame_index = 0
        self.initialized = False
        
    async def initialize(self, prompt: str, image_path: Optional[str] = None):
        """Initialize the world with a prompt and optional starting image."""
        logger.info(f"Initializing world with prompt: {prompt[:50]}...")
        
        # For now, we'll use a mock implementation
        # Real implementation would call torchrun with the model
        self.prompt = prompt
        self.initialized = True
        self.frame_index = 0
        
        # TODO: Real implementation
        # cmd = [
        #     "torchrun", f"--nproc_per_node={self.num_gpus}",
        #     "generate.py",
        #     "--task", "i2v-A14B",
        #     "--size", f"{FRAME_HEIGHT}*{FRAME_WIDTH}",
        #     "--ckpt_dir", self.model_dir,
        #     "--dit_fsdp", "--t5_fsdp",
        #     f"--ulysses_size", str(self.num_gpus),
        #     "--frame_num", "961",  # 1 minute at 16fps
        #     "--prompt", prompt,
        #     "--output_dir", self.output_dir,
        #     "--streaming"  # hypothetical streaming mode
        # ]
        
        return True
        
    async def generate_frame(self, controls: ControlState) -> bytes:
        """Generate next frame based on current controls."""
        if not self.initialized:
            raise RuntimeError("World not initialized")
            
        # Mock frame generation - creates a gradient based on controls
        # Real implementation would get frame from inference process
        
        img = Image.new('RGB', (FRAME_WIDTH, FRAME_HEIGHT))
        pixels = img.load()
        
        t = time.time()
        dx, dy, dz = controls.to_camera_delta()
        
        for y in range(FRAME_HEIGHT):
            for x in range(FRAME_WIDTH):
                # Create animated gradient
                r = int(128 + 127 * ((x + t * 50 * (1 if controls.d else -1 if controls.a else 0)) % 256) / 256)
                g = int(128 + 127 * ((y + t * 50 * (1 if controls.s else -1 if controls.w else 0)) % 256) / 256)
                b = int(128 + 127 * ((x + y + t * 30) % 256) / 256)
                pixels[x, y] = (r % 256, g % 256, b % 256)
        
        # Add text overlay
        from PIL import ImageDraw, ImageFont
        draw = ImageDraw.Draw(img)
        try:
            font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 24)
        except:
            font = ImageFont.load_default()
        
        draw.text((20, 20), "MOCK INFERENCE", fill=(255, 255, 255), font=font)
        draw.text((20, 50), f"Prompt: {self.prompt[:40]}...", fill=(200, 200, 200), font=font)
        draw.text((20, 80), f"Frame: {self.frame_index}", fill=(200, 200, 200), font=font)
        draw.text((20, FRAME_HEIGHT - 40), "Connect real model for actual output", fill=(255, 200, 100), font=font)
        
        self.frame_index += 1
        
        # Encode as JPEG
        buffer = io.BytesIO()
        img.save(buffer, format='JPEG', quality=85)
        return buffer.getvalue()
        
    def cleanup(self):
        """Clean up resources."""
        if self.process:
            self.process.terminate()
            self.process = None


async def handle_client(websocket):
    """Handle a single WebSocket client connection."""
    logger.info(f"Client connected: {websocket.remote_address}")
    
    controls = ControlState()
    inference = WorldModelInference(MODEL_DIR, NUM_GPUS)
    frame_interval = 1.0 / TARGET_FPS
    
    try:
        async for message in websocket:
            try:
                data = json.loads(message)
                msg_type = data.get("type")
                
                if msg_type == "init":
                    prompt = data.get("prompt", "A beautiful landscape")
                    controls.prompt = prompt
                    await inference.initialize(prompt)
                    
                    # Start frame streaming
                    asyncio.create_task(stream_frames(websocket, inference, controls, frame_interval))
                    
                elif msg_type == "control":
                    key = data.get("key", "").lower()
                    action = data.get("action", "up")
                    controls.update(key, action)
                    
            except json.JSONDecodeError:
                logger.warning(f"Invalid JSON: {message}")
                
    except websockets.exceptions.ConnectionClosed:
        logger.info(f"Client disconnected: {websocket.remote_address}")
    finally:
        inference.cleanup()


async def stream_frames(websocket, inference: WorldModelInference, controls: ControlState, interval: float):
    """Stream frames to client at target FPS."""
    logger.info("Starting frame streaming")
    
    try:
        while True:
            start_time = time.time()
            
            # Generate frame
            frame_data = await inference.generate_frame(controls)
            
            # Send as binary
            await websocket.send(frame_data)
            
            # Maintain target FPS
            elapsed = time.time() - start_time
            sleep_time = max(0, interval - elapsed)
            await asyncio.sleep(sleep_time)
            
    except websockets.exceptions.ConnectionClosed:
        logger.info("Frame streaming stopped - client disconnected")
    except Exception as e:
        logger.error(f"Frame streaming error: {e}")


async def main():
    """Start the WebSocket server."""
    logger.info(f"Starting WebSocket server on ws://{HOST}:{PORT}")
    logger.info(f"Model: {MODEL_DIR}, GPUs: {NUM_GPUS}, Target FPS: {TARGET_FPS}")
    
    async with websockets.serve(handle_client, HOST, PORT):
        logger.info("Server running. Press Ctrl+C to stop.")
        await asyncio.Future()  # Run forever


if __name__ == "__main__":
    asyncio.run(main())
