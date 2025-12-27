#!/usr/bin/env python3
"""
MCP Server for Project Aletheia
Exposes simulation and rendering capabilities via Model Context Protocol
"""

import os
import sys
import json
import asyncio
import torch
import numpy as np
from typing import Dict, Any, List, Optional

from mcp.server.fastmcp import FastMCP

# Add project root to path
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJECT_ROOT)

import subprocess
from aletheia.core.config import SimulationConfig
from aletheia.renderer import SimpleSoftwareRenderer, Camera

# Initialize FastMCP server
mcp = FastMCP("Aletheia Engine")

@mcp.tool()
def get_system_info() -> Dict[str, Any]:
    """Get information about the system, hardware, and available resources."""
    info = {
        "platform": sys.platform,
        "python_version": sys.version,
        "torch_version": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "cuda_device_count": torch.cuda.device_count() if torch.cuda.is_available() else 0,
        "current_directory": os.getcwd(),
    }
    
    if torch.cuda.is_available():
        devices = []
        for i in range(torch.cuda.device_count()):
            devices.append({
                "id": i,
                "name": torch.cuda.get_device_name(i),
                "memory": f"{torch.cuda.get_device_properties(i).total_memory / 1024**3:.2f} GB"
            })
        info["cuda_devices"] = devices
        
    return info

@mcp.tool()
def run_simulation_preview(
    num_frames: int = 10,
    resolution_width: int = 256,
    resolution_height: int = 144,
    num_particles: int = 100000
) -> str:
    """
    Run a short simulation preview using main.py and return the result summary.
    """
    try:
        cmd = [
            sys.executable, "main.py",
            "--preview",
            "--frames", str(num_frames),
            "--particles", str(num_particles),
            "--width", str(resolution_width),
            "--height", str(resolution_height)
        ]
        
        # Note: main.py might not support all these flags directly, 
        # but we can add them or use config file.
        # For now, let's assume we just run main.py with --preview which is supported.
        
        # Actually, let's just run main.py --preview as a safe bet
        cmd = [sys.executable, "main.py", "--preview"]
        
        process = subprocess.run(
            cmd, 
            capture_output=True, 
            text=True, 
            cwd=PROJECT_ROOT
        )
        
        if process.returncode == 0:
            return f"Simulation preview completed successfully.\nOutput:\n{process.stdout[:500]}..."
        else:
            return f"Simulation failed with code {process.returncode}.\nError:\n{process.stderr}"
        
    except Exception as e:
        return f"Simulation failed: {str(e)}"

@mcp.tool()
def render_frame_test(
    camera_pos: List[float] = [0.0, 0.0, 2.0],
    look_at: List[float] = [0.0, 0.0, 0.0]
) -> str:
    """
    Render a single test frame with specified camera position.
    """
    try:
        width, height = 256, 144
        renderer = SimpleSoftwareRenderer(resolution=(width, height))
        
        # Create dummy particles
        num_particles = 10000
        positions = torch.randn(num_particles, 3, device=renderer.device) * 0.5
        colors = torch.rand(num_particles, 3, device=renderer.device)
        
        # Setup camera
        camera = Camera(
            position=tuple(camera_pos),  # type: ignore[arg-type]
            look_at=tuple(look_at),  # type: ignore[arg-type]
            fov=45.0
        )
        
        # Render
        image = renderer.render(positions, colors, camera)
        
        return f"Rendered frame successfully. Shape: {image.shape}, Mean brightness: {image.mean():.4f}"
        
    except Exception as e:
        return f"Rendering failed: {str(e)}"

if __name__ == "__main__":
    mcp.run()
