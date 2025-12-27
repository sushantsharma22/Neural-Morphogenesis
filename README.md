# Project Aletheia - Neural Morphogenesis Engine

**A PyTorch DDP + Taichi Lang framework for growing organic 3D structures from noise using Neural Fields**

## Overview

Project Aletheia uses a Neural Field to "grow" organic, bioluminescent 3D structures from noise. The AI acts as the "Universal Law" (PDE) that organizes 10 million particles into complex forms such as neural networks, coral-like lattices, or stellar nurseries.

## System Requirements

- 4x NVIDIA A16 GPUs (64GB total VRAM)
- CUDA 11.8+
- Python 3.10+
- Headless SSH environment (no X11/GUI required)

## Installation

```bash
# Create virtual environment (no sudo required)
python -m venv --system-site-packages ~/.venvs/aletheia
source ~/.venvs/aletheia/bin/activate

# Install dependencies
pip install --user -r requirements.txt

# Or run the setup script
bash scripts/setup.sh
```

## Quick Start

```bash
# Activate environment
source ~/.venvs/aletheia/bin/activate

# Run with default configuration (4 GPUs)
python -m torch.distributed.launch --nproc_per_node=4 main.py

# Or use the runner script
bash scripts/run.sh --config configs/stellar_nursery.yaml

# Single GPU test
python main.py --single-gpu --test
```

## Project Structure

```
aletheia/
├── configs/                    # YAML configuration files
├── aletheia/
│   ├── core/
│   │   ├── config.py          # Configuration management
│   │   ├── distributed.py     # DDP utilities
│   │   └── memory.py          # GPU memory management
│   ├── models/
│   │   ├── fno.py             # Fourier Neural Operator
│   │   ├── unet_mpm.py        # U-Net Neural MPM
│   │   └── latent_dna.py      # Latent DNA system
│   ├── physics/
│   │   ├── mpm.py             # Material Point Method (Taichi)
│   │   ├── fields.py          # Neural field simulation
│   │   └── pde.py             # PDE perturbation system
│   ├── renderer/
│   │   ├── raytracer.py       # Headless ray-tracing
│   │   ├── effects.py         # Bloom, DoF, motion blur
│   │   └── camera.py          # Camera system
│   └── output/
│       ├── exporter.py        # Frame/video export
│       └── metadata.py        # Latent DNA recording
├── scripts/
│   ├── setup.sh               # Installation script
│   └── run.sh                  # Launch script
└── main.py                     # Entry point
```

## Output

- **Frames**: `/data/frames/` - 4K PNG sequences
- **Video**: `output/aletheia_render.mp4` - Cinematic MP4
- **Metadata**: `output/latent_dna.json` - Physics DNA recording

## Configuration

Edit `configs/default.yaml` for simulation parameters:

```yaml
simulation:
  num_particles: 10_000_000
  frames: 1000
  dt: 1e-4

neural:
  model: "fno"  # or "unet"
  latent_dim: 512

render:
  resolution: [3840, 2160]  # 4K
  bloom_intensity: 2.5
  dof_aperture: 0.02
```

## GPU Memory Management

Optimized for A16 with 16GB per GPU:
- Per-GPU memory limit: 15GB (safe headroom)
- Particle distribution: 2.5M per GPU
- Gradient checkpointing enabled
- Mixed precision (FP16) training

## License

MIT License - See LICENSE file
