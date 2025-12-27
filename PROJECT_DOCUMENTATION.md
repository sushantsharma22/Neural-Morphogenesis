# Project Aletheia: Neural Morphogenesis Engine

## Complete Technical Documentation

> **Version:** 1.0.0  
> **Last Updated:** December 26, 2025  
> **Author:** Neural Morphogenesis Team  
> **Hardware:** 4x NVIDIA A16 GPUs (14.5GB each)

---

## Table of Contents

1. [Project Overview](#1-project-overview)
2. [Architecture](#2-architecture)
3. [Directory Structure](#3-directory-structure)
4. [Core Modules](#4-core-modules)
5. [Neural Network Models](#5-neural-network-models)
6. [Physics Simulation](#6-physics-simulation)
7. [Rendering System](#7-rendering-system)
8. [Output System](#8-output-system)
9. [Configuration System](#9-configuration-system)
10. [MCP Server](#10-mcp-server)
11. [Running the Simulation](#11-running-the-simulation)
12. [Tweaking Guide](#12-tweaking-guide)
13. [Troubleshooting](#13-troubleshooting)
14. [Performance Optimization](#14-performance-optimization)

---

## 1. Project Overview

### What is Neural Morphogenesis?

Neural Morphogenesis is a GPU-accelerated particle simulation system that combines:
- **Neural Networks (FNO/U-Net)**: Learn velocity field predictions
- **Physics Simulation (MPM)**: Material Point Method for realistic particle dynamics
- **PDE Solvers**: Navier-Stokes + Reaction-Diffusion for fluid/pattern effects
- **Real-time Rendering**: GPU-accelerated particle visualization

### The "Latent DNA" Concept

The system uses a "Latent DNA" vector - a high-dimensional seed (default 512D) that:
- Perturbs the physics simulation
- Creates unique, reproducible morphogenesis patterns
- Evolves through phases: Chaos → Emergence → Refinement → Stabilization

### Key Features

- **10M+ particles** at 60 FPS (with 4x A16 GPUs)
- **Multi-GPU support** via PyTorch DDP
- **4K rendering** with bloom, DoF, motion blur
- **Checkpoint/resume** for long simulations
- **MCP server** for AI integration

---

## 2. Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                     ALETHEIA ENGINE                              │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│   ┌──────────────┐    ┌──────────────┐    ┌──────────────┐     │
│   │  Latent DNA  │───▶│ Neural Model │───▶│ Velocity     │     │
│   │  (512D seed) │    │ (FNO/U-Net)  │    │ Field        │     │
│   └──────────────┘    └──────────────┘    └──────┬───────┘     │
│                                                   │              │
│   ┌──────────────┐    ┌──────────────┐           │              │
│   │ PDE Solver   │───▶│ Perturbation │───────────┤              │
│   │ (NS + R-D)   │    │ Field        │           │              │
│   └──────────────┘    └──────────────┘           ▼              │
│                                           ┌──────────────┐      │
│                                           │ MPM Physics  │      │
│                                           │ (Taichi GPU) │      │
│                                           └──────┬───────┘      │
│                                                  │               │
│                                                  ▼               │
│   ┌──────────────┐    ┌──────────────┐    ┌──────────────┐     │
│   │ Frame Export │◀───│ Post-Process │◀───│ Renderer     │     │
│   │ (PNG/MP4)    │    │ (Bloom/DoF)  │    │ (Particles)  │     │
│   └──────────────┘    └──────────────┘    └──────────────┘     │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

---

## 3. Directory Structure

```
Neural-Morphogenesis/
│
├── main.py                      # Main orchestrator - starts everything
├── mcp_server.py                # MCP server for AI integration
├── test_system.py               # System verification tests
├── run_simulation.sh            # One-click run script (on server)
├── download_to_mac.sh           # Download results to Mac
│
├── aletheia/                    # Core library
│   ├── __init__.py
│   │
│   ├── core/                    # Core utilities
│   │   ├── __init__.py
│   │   ├── config.py            # All configuration dataclasses
│   │   ├── distributed.py       # Multi-GPU DDP management
│   │   └── memory.py            # GPU memory management
│   │
│   ├── models/                  # Neural network models
│   │   ├── __init__.py
│   │   ├── fno.py               # Fourier Neural Operator (main model)
│   │   ├── unet_mpm.py          # U-Net alternative model
│   │   └── latent_dna.py        # Latent DNA system + scheduler
│   │
│   ├── physics/                 # Physics simulation
│   │   ├── __init__.py
│   │   ├── mpm.py               # Material Point Method (Taichi)
│   │   ├── pde.py               # PDE solvers (Navier-Stokes, R-D)
│   │   └── fields.py            # Neural field solver interface
│   │
│   ├── renderer/                # Rendering system
│   │   ├── __init__.py
│   │   ├── raytracer.py         # GPU particle renderer
│   │   ├── effects.py           # Post-processing (bloom, DoF, etc.)
│   │   └── camera.py            # Camera and animation
│   │
│   └── output/                  # Output management
│       ├── __init__.py
│       ├── frame_exporter.py    # PNG export, metadata
│       └── video_encoder.py     # FFmpeg video encoding
│
├── configs/                     # Configuration files
│   ├── default.yaml             # Default settings
│   └── preview.yaml             # Quick preview settings
│
├── output/                      # Simulation outputs (generated)
│   ├── frame_000000.png
│   ├── ...
│   ├── morphogenesis.mp4
│   └── metadata/
│
├── scripts/                     # Utility scripts
│   ├── run.sh                   # Multi-GPU launch script
│   └── benchmark.sh             # Performance benchmarks
│
└── venv/                        # Python virtual environment
```

---

## 4. Core Modules

### 4.1 Configuration System (`aletheia/core/config.py`)

The configuration is organized into dataclasses for type safety:

```python
@dataclass
class SimulationConfig:
    """Physics simulation parameters"""
    num_particles: int = 10_000_000     # Total particles
    particles_per_gpu: int = 2_500_000  # Particles per GPU
    total_frames: int = 1000            # Animation length
    substeps_per_frame: int = 20        # Physics substeps
    dt: float = 5.0e-5                  # Time step
    grid_resolution: int = 256          # Simulation grid size
    
    # Domain bounds
    domain_min: Tuple[float, float, float] = (-2.0, -2.0, -2.0)
    domain_max: Tuple[float, float, float] = (2.0, 2.0, 2.0)
    
    # Physics
    gravity: Tuple[float, float, float] = (0.0, -0.5, 0.0)
    damping: float = 0.999
    
    # Material (Neo-Hookean)
    material_type: str = "neo_hookean"
    density: float = 1000.0             # kg/m³
    young_modulus: float = 5000.0       # Stiffness
    poisson_ratio: float = 0.35         # Compressibility
    viscosity: float = 0.1

@dataclass
class NeuralConfig:
    """Neural network settings"""
    model_type: str = "fno"             # "fno" or "unet"
    latent_dim: int = 512               # DNA vector dimension
    
    # FNO settings
    fno_modes: Tuple[int, int, int] = (16, 16, 16)  # Fourier modes
    fno_width: int = 64                 # Channel width
    fno_num_layers: int = 4             # Number of FNO blocks
    
    # U-Net settings
    unet_base_channels: int = 64
    unet_channel_multipliers: Tuple[int, ...] = (1, 2, 4, 8)

@dataclass
class RenderConfig:
    """Rendering settings"""
    resolution: Tuple[int, int] = (3840, 2160)  # 4K
    samples_per_pixel: int = 64
    camera_fov: float = 45.0
    
    # Effects
    bloom_enabled: bool = True
    bloom_intensity: float = 2.5
    bloom_threshold: float = 0.8
    motion_blur_enabled: bool = True
    motion_blur_samples: int = 8
    
    # Colors
    particle_color_palette: str = "bioluminescent"
```

**Key Configuration File Locations:**
- Default config: `configs/default.yaml`
- Command-line override: `--config path/to/config.yaml`
- Code override: Modify in `main.py` setup()

### 4.2 Distributed Computing (`aletheia/core/distributed.py`)

Manages multi-GPU training via PyTorch's DistributedDataParallel:

```python
class DistributedManager:
    """Handles multi-GPU distribution"""
    
    def initialize(self):
        """Initialize distributed environment"""
        # Auto-detects SLURM, torchrun, or single-GPU
        
    def wrap_model(self, model: nn.Module) -> nn.Module:
        """Wrap model in DDP"""
        return DistributedDataParallel(model, device_ids=[self.local_rank])
    
    def barrier(self):
        """Synchronization point across GPUs"""
        
    def all_reduce(self, tensor: torch.Tensor) -> torch.Tensor:
        """Sum tensor across all GPUs"""
```

**Usage:**
```bash
# Single GPU
python main.py --preview

# Multi-GPU with torchrun
torchrun --nproc_per_node=4 main.py

# SLURM cluster
sbatch scripts/run.sh
```

### 4.3 Memory Management (`aletheia/core/memory.py`)

Handles GPU memory efficiently for large simulations:

```python
class MemoryManager:
    def __init__(self, max_memory_gb: float = 15.0, device_id: int = 0):
        self.max_allocation = max_memory_gb * 1024**3
        
    def get_memory_stats(self) -> Dict[str, float]:
        """Returns allocated, reserved, and free memory"""
        
    def check_allocation(self, size_bytes: int) -> bool:
        """Check if allocation is safe"""
        
    def optimize_batch_size(self, base_batch: int) -> int:
        """Auto-adjust batch size based on memory"""
```

---

## 5. Neural Network Models

### 5.1 Fourier Neural Operator (`aletheia/models/fno.py`)

The FNO is the primary model for velocity field prediction.

**Architecture:**
```
Input: (B, 4, D, H, W)  # [velocity_x, velocity_y, velocity_z, density]
       ↓
    Lift Conv3D (4 → width)
       ↓
    ┌─────────────────────────┐
    │   FNO Block × 4         │
    │  ┌───────────────────┐  │
    │  │ Spectral Conv     │  │  ← FFT → multiply weights → iFFT
    │  │ + Linear Conv     │  │
    │  │ + Time Modulation │  │  ← Time embedding injection
    │  │ + Latent FiLM     │  │  ← DNA conditioning (scale + shift)
    │  │ + Activation      │  │
    │  └───────────────────┘  │
    └─────────────────────────┘
       ↓
    Project Conv3D (width → 3)
       ↓
Output: (B, 3, D, H, W)  # Predicted velocity field
```

**Key Classes:**

```python
class SpectralConv3d(nn.Module):
    """3D Fourier convolution in spectral domain"""
    def __init__(self, in_channels, out_channels, modes1, modes2, modes3):
        # Learnable Fourier mode weights
        self.weights = nn.Parameter(...)  # Complex weights
        
    def forward(self, x):
        x_ft = torch.fft.rfftn(x, dim=[-3, -2, -1])
        # Multiply selected modes by weights
        out_ft = self.compl_mul3d(x_ft, self.weights)
        return torch.fft.irfftn(out_ft, s=x.shape[-3:])

class TimeConditionedFNO3D(FourierNeuralOperator3D):
    """FNO with time and DNA conditioning"""
    def forward(self, x, latent, time_embedding=None):
        # Time embedding via sinusoidal encoding
        t_emb = self.time_embed(time_embedding)
        
        # DNA conditioning via FiLM
        conditions = self.conditioner(latent)  # scale, shift pairs
        
        for i, block in enumerate(self.blocks):
            x = block(x)
            x = self.time_norm[i](x + t_emb)
            scale, shift = conditions[i]
            x = x * (1 + scale) + shift  # FiLM modulation
            
        return self.project(x)
```

**Tweaking FNO:**
| Parameter | Effect | Recommended Range |
|-----------|--------|-------------------|
| `fno_modes` | Frequency resolution | (8,8,8) to (32,32,32) |
| `fno_width` | Model capacity | 32 to 128 |
| `fno_num_layers` | Depth | 2 to 8 |
| `latent_dim` | DNA expressiveness | 256 to 1024 |

### 5.2 U-Net Model (`aletheia/models/unet_mpm.py`)

Alternative lightweight model:

```python
class LightweightUNetMPM(nn.Module):
    """Efficient U-Net for MPM coupling"""
    # Encoder: progressively downsample
    # Bottleneck: latent processing
    # Decoder: upsample with skip connections
```

**When to use U-Net vs FNO:**
- **FNO**: Better for smooth, global patterns. Higher memory.
- **U-Net**: Better for local details, faster, less memory.

### 5.3 Latent DNA System (`aletheia/models/latent_dna.py`)

The DNA system creates unique simulation "seeds":

```python
class LatentDNA:
    def __init__(self, dim=512, seed_type="gaussian", seed=None):
        self.latent_vector = self._generate_latent()
    
    def _generate_latent(self):
        if self.seed_type == "gaussian":
            return torch.randn(1, self.dim)
        elif self.seed_type == "structured":
            # Low freq (global) + mid freq (local) + high freq (detail)
            latent = torch.zeros(1, self.dim)
            latent[:, :dim//4] = torch.randn(...) * 2.0      # Global
            latent[:, dim//4:dim//2] = torch.randn(...) * 1.0  # Local
            latent[:, dim//2:] = torch.randn(...) * 0.5        # Detail
            return latent
    
    def mutate(self, strength=0.1):
        """Create variation of this DNA"""
        noise = torch.randn_like(self.latent_vector) * strength
        return self.latent_vector + noise
    
    def interpolate(self, other, alpha):
        """Spherical interpolation between two DNAs"""
        # SLERP for smooth morphing between forms
```

**Morphogenesis Scheduler:**

```python
class MorphogenesisScheduler:
    """Controls simulation phases"""
    
    def get_phase(self, frame) -> str:
        if frame < self.chaos_end:       # 0-100
            return "chaos"               # Random exploration
        elif frame < self.emergence_end:  # 100-400
            return "emergence"           # Patterns form
        elif frame < self.refinement_end: # 400-700
            return "refinement"          # Details sharpen
        else:
            return "stabilization"       # Final form
    
    def get_dna_influence(self, frame) -> float:
        # DNA influence ramps up over time
        # chaos: 0→0.2, emergence: 0.2→1.0, refinement: ~1.0
```

---

## 6. Physics Simulation

### 6.1 Material Point Method (`aletheia/physics/mpm.py`)

MPM is the core physics engine using Taichi for GPU acceleration:

```python
@ti.data_oriented
class MPMSimulator:
    def __init__(
        self,
        num_particles: int,
        grid_resolution: int = 256,
        domain_min: Tuple = (-2, -2, -2),
        domain_max: Tuple = (2, 2, 2),
        material: MaterialProperties = None,
        dt: float = 5e-5,
        gravity: Tuple = (0, -0.5, 0),
        damping: float = 0.999,
    ):
        # Taichi fields for particles
        self.x = ti.Vector.field(3, dtype=ti.f32, shape=num_particles)  # Position
        self.v = ti.Vector.field(3, dtype=ti.f32, shape=num_particles)  # Velocity
        self.C = ti.Matrix.field(3, 3, dtype=ti.f32, shape=num_particles)  # APIC
        self.F = ti.Matrix.field(3, 3, dtype=ti.f32, shape=num_particles)  # Deformation
        
        # Grid fields
        self.grid_v = ti.Vector.field(3, shape=(n, n, n))
        self.grid_m = ti.field(dtype=ti.f32, shape=(n, n, n))
    
    @ti.kernel
    def particle_to_grid(self):
        """P2G: Transfer particle momentum to grid"""
        for p in range(self.num_particles):
            # Quadratic B-spline interpolation
            # Neo-Hookean stress computation
            # Scatter to grid nodes
    
    @ti.kernel
    def grid_operations(self):
        """Apply gravity, boundary conditions"""
        for i, j, k in self.grid_v:
            if self.grid_m[i, j, k] > 0:
                self.grid_v[i, j, k] /= self.grid_m[i, j, k]
                self.grid_v[i, j, k] += self.dt * gravity
                # Boundary collision
    
    @ti.kernel
    def grid_to_particle(self):
        """G2P: Transfer grid velocity back to particles"""
        for p in range(self.num_particles):
            # Gather from grid nodes
            # Update position, velocity, deformation gradient
```

**Material Properties:**
```python
@dataclass
class MaterialProperties:
    density: float = 1000.0        # kg/m³
    young_modulus: float = 5000.0  # Stiffness (Pa)
    poisson_ratio: float = 0.35    # 0=compressible, 0.5=incompressible
    viscosity: float = 0.1         # Damping
    
    @property
    def lame_mu(self):  # Shear modulus
        return E / (2 * (1 + nu))
    
    @property
    def lame_lambda(self):  # Bulk modulus related
        return E * nu / ((1 + nu) * (1 - 2*nu))
```

**Tweaking Physics:**
| Parameter | Effect | Range |
|-----------|--------|-------|
| `young_modulus` | Stiffness | 100 (jelly) to 100000 (stiff) |
| `poisson_ratio` | Volume preservation | 0.2 to 0.49 |
| `gravity` | Downward force | (0, -0.1, 0) to (0, -2.0, 0) |
| `damping` | Energy loss | 0.9 (heavy) to 0.999 (light) |
| `dt` | Time step | 1e-5 to 1e-4 |

### 6.2 PDE Solvers (`aletheia/physics/pde.py`)

Two coupled PDE systems:

**Navier-Stokes (Fluid Dynamics):**
```python
class NavierStokesSolver:
    """Incompressible fluid simulation"""
    # Advection: move fluid
    # Diffusion: viscosity spreading
    # Pressure projection: enforce incompressibility
    
    def step(self, dt):
        self.advect(dt)
        self.diffuse(dt)
        self.project()  # Pressure solve
```

**Reaction-Diffusion (Pattern Formation):**
```python
class ReactionDiffusionSolver:
    """Gray-Scott reaction-diffusion"""
    # A + 2B → 3B (autocatalysis)
    # B → P (decay)
    
    # Creates Turing patterns: spots, stripes, mazes
    
    def step(self, dt):
        # Diffusion
        laplacian_A = self._laplacian(self.A)
        laplacian_B = self._laplacian(self.B)
        
        # Reaction
        reaction = self.A * self.B * self.B
        
        # Update
        self.A += (self.Da * laplacian_A - reaction + self.f * (1 - self.A)) * dt
        self.B += (self.Db * laplacian_B + reaction - (self.k + self.f) * self.B) * dt
```

**Coupled Solver:**
```python
class CoupledPDESolver:
    """Combines NS + RD for fluid patterns"""
    # Fluid velocity advects chemical concentrations
    # Chemical gradients create forces in fluid
```

**Tweaking PDE:**
| Parameter | Effect | Range |
|-----------|--------|-------|
| `viscosity` | Fluid thickness | 0.001 (water) to 0.1 (honey) |
| `diffusion_a/b` | Pattern spread | 0.1 to 1.0 |
| `feed_rate` | Pattern density | 0.01 to 0.1 |
| `kill_rate` | Pattern type | 0.045 (spots) to 0.065 (stripes) |

### 6.3 Neural Field Solver (`aletheia/physics/fields.py`)

Bridges neural network output to physics:

```python
class NeuralFieldSolver:
    """Converts NN output to physics inputs"""
    
    def __init__(self, model, grid_resolution, domain_min, domain_max):
        self.model = model
        self._create_coordinate_grid()
    
    def compute_density_field(self, positions):
        """Particles → density grid"""
        # Histogram + Gaussian smoothing
        
    def compute_velocity_field(self, positions, velocities, latent):
        """Get NN-predicted velocity field"""
        input_field = self.prepare_input(positions, velocities)
        return self.model(input_field, latent)
```

---

## 7. Rendering System

### 7.1 Particle Renderer (`aletheia/renderer/raytracer.py`)

**Taichi-based Renderer (High Quality):**
```python
@ti.data_oriented
class ParticleRenderer:
    """GPU ray-traced particle rendering"""
    
    @ti.kernel
    def render_kernel(self):
        for i, j in self.image:
            # Cast ray from camera through pixel
            ray = self.get_ray(i, j)
            
            # March through volume
            color = ti.Vector([0.0, 0.0, 0.0])
            for step in range(max_steps):
                # Check particle intersections
                # Accumulate emission color
                # Handle occlusion
            
            self.image[i, j] = color
```

**Simple Software Renderer (Fast Preview):**
```python
class SimpleSoftwareRenderer:
    """PyTorch-based point rendering"""
    
    def render(self, positions, colors, camera=None):
        # Project particles to screen space
        # Z-sort for painter's algorithm
        # Splat colored points
        return image  # (H, W, 3)
```

### 7.2 Post-Processing Effects (`aletheia/renderer/effects.py`)

**Bloom Effect:**
```python
class BloomEffect:
    """Bioluminescent glow"""
    
    def apply(self, image, intensity=2.5, threshold=0.8):
        # Extract bright regions
        bright = image * (luminance > threshold)
        
        # Multi-scale Gaussian blur
        for scale in range(5):
            bloom += gaussian_blur(bright, sigma=2**scale)
        
        # Color shift (cyan/magenta tint)
        bloom *= color_shift
        
        return image + bloom * intensity
```

**Depth of Field:**
```python
class DepthOfField:
    """Cinematic focus blur"""
    
    def apply(self, image, depth_map, focus_distance, aperture):
        # Compute circle of confusion per pixel
        coc = abs(focus_distance - depth) * aperture
        
        # Variable-radius blur
        for radius in range(max_blur):
            if coc[pixel] >= radius:
                apply_bokeh_kernel(pixel, radius)
```

**Motion Blur:**
```python
class MotionBlur:
    """Velocity-based blur"""
    
    def apply(self, image, velocity_field, shutter_time=0.5):
        # Sample along velocity vectors
        for t in linspace(-0.5, 0.5, samples):
            sample_pos = pixel + velocity * t * shutter_time
            result += sample(image, sample_pos)
        return result / samples
```

### 7.3 Camera System (`aletheia/renderer/camera.py`)

```python
class Camera:
    def __init__(self, position, look_at, fov=45, aspect_ratio=16/9):
        self.position = np.array(position)
        self.look_at = np.array(look_at)
        self._update_matrices()
    
    def get_ray(self, u, v):
        """Get ray direction for pixel (u, v)"""

class CameraAnimator:
    """Animates camera over time"""
    
    def set_orbit_animation(self, center, radius, speed):
        """Circular orbit around subject"""
        
    def set_dolly_animation(self, start, end, ease_type):
        """Linear camera move"""
        
    def add_keyframe(self, frame, position, look_at, fov):
        """Custom keyframe animation"""
    
    def update(self, frame):
        """Update camera for current frame"""
```

**Tweaking Rendering:**
| Parameter | Effect | Range |
|-----------|--------|-------|
| `samples_per_pixel` | Quality vs speed | 1 (fast) to 256 (cinema) |
| `bloom_intensity` | Glow strength | 0.0 to 5.0 |
| `bloom_threshold` | What glows | 0.5 to 0.95 |
| `camera_fov` | Zoom | 20 (telephoto) to 90 (wide) |
| `particle_color_palette` | Color scheme | "bioluminescent", "stellar", "ocean", "synaptic" |

---

## 8. Output System

### 8.1 Frame Exporter (`aletheia/output/frame_exporter.py`)

```python
class FrameExporter:
    def __init__(self, output_dir, resolution, format="png"):
        self.output_dir = Path(output_dir)
        
    def export_frame(self, frame, frame_idx, metadata=None):
        """Save frame as PNG with optional metadata"""
        # Convert to uint8
        # Write with PIL
        # Save EXIF metadata
        
class MetadataRecorder:
    def record_frame(self, frame_idx, positions, velocities, colors):
        """Record per-frame statistics"""
        return {
            "frame": frame_idx,
            "particle_count": len(positions),
            "velocity_stats": {
                "mean": velocities.mean(),
                "max": velocities.max(),
            },
            "bounding_box": compute_bounds(positions),
        }
```

### 8.2 Video Encoder (`aletheia/output/video_encoder.py`)

```python
class VideoEncoder:
    def encode_video(self, input_dir, output_name, fps=30, quality="high"):
        """Encode frames to MP4 using FFmpeg"""
        
        presets = {
            "draft": {"crf": 28, "preset": "ultrafast"},
            "medium": {"crf": 23, "preset": "medium"},
            "high": {"crf": 18, "preset": "slow"},
            "ultra": {"crf": 15, "preset": "veryslow"},
        }
        
        cmd = [
            "ffmpeg", "-y",
            "-framerate", str(fps),
            "-i", f"{input_dir}/frame_%06d.png",
            "-c:v", "libx264",
            "-crf", str(preset["crf"]),
            "-preset", preset["preset"],
            "-pix_fmt", "yuv420p",
            output_path,
        ]
```

---

## 9. Configuration System

### 9.1 Config File Format (YAML)

```yaml
# configs/custom.yaml

simulation:
  num_particles: 1000000
  total_frames: 500
  grid_resolution: 128
  dt: 0.00005
  
  gravity: [0.0, -0.5, 0.0]
  damping: 0.999
  
  # Material
  density: 1000.0
  young_modulus: 5000.0
  poisson_ratio: 0.35

neural:
  model_type: "fno"
  latent_dim: 512
  fno_modes: [16, 16, 16]
  fno_width: 64
  fno_num_layers: 4

render:
  resolution: [1920, 1080]
  samples_per_pixel: 16
  bloom_intensity: 2.5
  particle_color_palette: "bioluminescent"

morphogenesis:
  chaos_frames: [0, 50]
  emergence_frames: [50, 200]
  refinement_frames: [200, 400]

output:
  frame_format: "png"
  video_fps: 30
  video_codec: "libx264"
```

### 9.2 Loading Configuration

```python
# In code
from aletheia.core.config import load_config, AletheiaConfig

config = load_config("configs/custom.yaml")

# Override specific values
config.simulation.num_particles = 500000
config.render.resolution = (1280, 720)

# Command line
python main.py --config configs/custom.yaml --preview
```

---

## 10. MCP Server

The Model Context Protocol server enables AI integration:

```python
# mcp_server.py

from mcp.server import Server
from mcp.types import Tool

server = Server("neural-morphogenesis")

@server.tool()
async def get_system_info() -> dict:
    """Get GPU and system information"""
    return {
        "cuda_available": torch.cuda.is_available(),
        "gpu_count": torch.cuda.device_count(),
        "devices": [...],
    }

@server.tool()
async def render_frame_test(seed: int = 42) -> str:
    """Render a test frame"""
    # Quick render for testing
    
@server.tool()
async def run_simulation(
    frames: int = 100,
    particles: int = 100000,
    output_dir: str = "./output"
) -> dict:
    """Run a simulation"""

# Start server
if __name__ == "__main__":
    server.run()
```

**Using MCP:**
```bash
# Start MCP server
python mcp_server.py

# In Claude/AI assistant
"Use neural-morphogenesis to render a test frame"
```

---

## 11. Running the Simulation

### Quick Start

```bash
# 1. Activate environment
source venv/bin/activate

# 2. Run preview (fast, low quality)
python main.py --preview --output ./output

# 3. Run full simulation
python main.py --config configs/default.yaml --output ./output
```

### Command Line Options

```bash
python main.py [OPTIONS]

Options:
  --config PATH       Path to config YAML file
  --output PATH       Output directory (default: ./output)
  --preview          Quick preview mode (100 frames, 100k particles)
  --seed INT         Random seed for reproducibility
  --resume PATH      Resume from checkpoint
```

### Multi-GPU

```bash
# Using torchrun
torchrun --nproc_per_node=4 main.py --config configs/default.yaml

# Using the provided script
./scripts/run.sh --gpus 4 --config configs/default.yaml
```

### One-Click Scripts

```bash
# On server: Run simulation
./run_simulation.sh

# On Mac: Download results
./download_to_mac.sh
```

---

## 12. Tweaking Guide

### Creating Different Effects

**Jelly/Soft Body:**
```yaml
simulation:
  young_modulus: 500        # Very soft
  poisson_ratio: 0.45       # Nearly incompressible
  gravity: [0, -1.0, 0]     # Noticeable sag
  damping: 0.95             # Bouncy
```

**Fluid-like:**
```yaml
simulation:
  young_modulus: 100        # Extremely soft
  poisson_ratio: 0.3
  viscosity: 0.5
  damping: 0.99
```

**Stiff Structure:**
```yaml
simulation:
  young_modulus: 50000      # Very stiff
  poisson_ratio: 0.25
  gravity: [0, -0.2, 0]     # Minimal sag
  damping: 0.999
```

**Fast Chaos:**
```yaml
morphogenesis:
  chaos_frames: [0, 200]    # Longer chaos phase
  emergence_frames: [200, 600]
```

**Quick Stabilization:**
```yaml
morphogenesis:
  chaos_frames: [0, 20]
  emergence_frames: [20, 100]
  refinement_frames: [100, 200]
```

### Color Palettes

```python
# In aletheia/renderer/effects.py

PALETTES = {
    "bioluminescent": [
        (0.1, 0.8, 1.0),   # Cyan
        (0.4, 0.9, 0.8),   # Teal
        (0.2, 0.6, 1.0),   # Blue
        (0.8, 0.4, 1.0),   # Purple
        (0.3, 1.0, 0.6),   # Green
    ],
    "stellar": [           # Stars/galaxies
        (1.0, 0.9, 0.8),   # White-yellow
        (0.8, 0.9, 1.0),   # Blue-white
        (1.0, 0.6, 0.4),   # Orange
    ],
    "ocean": [             # Deep sea
        (0.0, 0.5, 0.8),   # Deep blue
        (0.2, 0.8, 0.9),   # Cyan
        (0.0, 0.9, 0.7),   # Teal
    ],
    "synaptic": [          # Neural/brain
        (0.8, 0.3, 1.0),   # Violet
        (0.4, 0.5, 1.0),   # Blue-violet
        (1.0, 0.4, 0.8),   # Pink
    ],
}
```

### Performance vs Quality Tradeoffs

| Setting | Low (Fast) | Medium | High (Quality) |
|---------|------------|--------|----------------|
| `num_particles` | 100,000 | 1,000,000 | 10,000,000 |
| `grid_resolution` | 32 | 128 | 256 |
| `substeps_per_frame` | 5 | 20 | 50 |
| `samples_per_pixel` | 1 | 16 | 64 |
| `fno_width` | 32 | 64 | 128 |
| `resolution` | 960×540 | 1920×1080 | 3840×2160 |

---

## 13. Troubleshooting

### Common Errors

**CUDA Out of Memory:**
```
torch.cuda.OutOfMemoryError: CUDA out of memory
```
**Solution:** Reduce `num_particles`, `grid_resolution`, or `fno_width`

**Taichi Field Error:**
```
TaichiRuntimeError: Cannot create field, maybe you forgot to call ti.init()
```
**Solution:** Ensure `init_taichi()` is called before creating MPMSimulator

**ComplexHalf Not Supported:**
```
NotImplementedError: "baddbmm_cuda" not implemented for 'ComplexHalf'
```
**Solution:** Disable autocast for FNO forward pass (already fixed in code)

**FFmpeg Not Found:**
```
Warning: FFmpeg not found. Video encoding will be disabled.
```
**Solution:** Install FFmpeg: `module load ffmpeg` or use provided frames

### Debug Mode

```python
# In main.py, enable verbose logging
logging.basicConfig(level=logging.DEBUG)

# In physics, enable Taichi profiler
ti.init(kernel_profiler=True)
```

### Memory Debugging

```python
# Check GPU memory
from aletheia.core.memory import MemoryManager
mm = MemoryManager(device_id=0)
print(mm.get_memory_stats())

# Force garbage collection
import gc
gc.collect()
torch.cuda.empty_cache()
```

---

## 14. Performance Optimization

### GPU Memory Optimization

1. **Gradient Checkpointing:**
```python
# In config
neural:
  gradient_checkpointing: true
```

2. **Mixed Precision (for non-FFT parts):**
```python
with torch.amp.autocast('cuda'):
    # Forward pass
```

3. **Chunk Processing:**
```python
# Process particles in batches
for chunk in particle_chunks:
    process(chunk)
    torch.cuda.empty_cache()
```

### Multi-GPU Scaling

| GPUs | Particles | Expected FPS |
|------|-----------|--------------|
| 1 | 2.5M | 15-20 |
| 2 | 5M | 25-30 |
| 4 | 10M | 40-50 |

### Profiling

```bash
# PyTorch profiler
python -m torch.utils.bottleneck main.py --preview

# Taichi profiler
# Enabled via ti.init(kernel_profiler=True)
# Results printed after simulation
```

---

## Appendix A: File Reference

| File | Lines | Purpose |
|------|-------|---------|
| `main.py` | ~780 | Main orchestrator, simulation loop |
| `mcp_server.py` | ~150 | MCP server for AI integration |
| `test_system.py` | ~200 | System verification tests |
| `aletheia/core/config.py` | ~580 | Configuration dataclasses |
| `aletheia/core/distributed.py` | ~200 | Multi-GPU DDP management |
| `aletheia/core/memory.py` | ~250 | GPU memory management |
| `aletheia/models/fno.py` | ~500 | Fourier Neural Operator |
| `aletheia/models/unet_mpm.py` | ~400 | U-Net model |
| `aletheia/models/latent_dna.py` | ~510 | DNA system + scheduler |
| `aletheia/physics/mpm.py` | ~535 | MPM physics (Taichi) |
| `aletheia/physics/pde.py` | ~500 | PDE solvers |
| `aletheia/physics/fields.py` | ~460 | Neural field interface |
| `aletheia/renderer/raytracer.py` | ~565 | GPU particle renderer |
| `aletheia/renderer/effects.py` | ~555 | Post-processing effects |
| `aletheia/renderer/camera.py` | ~485 | Camera system |
| `aletheia/output/frame_exporter.py` | ~590 | Frame/metadata export |
| `aletheia/output/video_encoder.py` | ~420 | FFmpeg video encoding |

---

## Appendix B: Quick Reference

### Environment Variables

```bash
export CUDA_VISIBLE_DEVICES=0,1,2,3  # Select GPUs
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True  # Memory optimization
```

### Common Commands

```bash
# Run preview
python main.py --preview

# Run with custom config
python main.py --config configs/custom.yaml

# Multi-GPU
torchrun --nproc_per_node=4 main.py

# Test system
python test_system.py

# MCP server
python mcp_server.py
```

### Download Commands (Run on Mac)

```bash
# Download latest output
scp -r user@server:/path/to/output ~/Downloads/NeuralMorphogenesis/

# Download specific video
scp user@server:/path/to/output/morphogenesis.mp4 ~/Downloads/
```

---

*Documentation generated for Project Aletheia v1.0.0*
