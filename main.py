#!/usr/bin/env python3
"""
Project Aletheia: Neural Morphogenesis Engine
Main Orchestrator

Coordinates the full simulation pipeline:
1. Initialize distributed environment (4x A16 GPUs)
2. Load configuration and create models
3. Initialize particle system with noise
4. Run simulation loop (physics + neural + render)
5. Export frames and metadata
"""

from __future__ import annotations

import os
import sys
import time
import argparse
import logging
from pathlib import Path
from typing import Optional, Dict, Any, TYPE_CHECKING, cast
import datetime
import json
from dataclasses import asdict

import torch
import torch.distributed as dist
import numpy as np

# Add project root to path
PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))

# Type checking imports
if TYPE_CHECKING:
    from aletheia.core.config import AletheiaConfig as MorphogenesisConfig
    from aletheia.core import DistributedManager, MemoryManager
    from aletheia.models import TimeConditionedFNO3D, LightweightUNetMPM, LatentDNA, MorphogenesisScheduler
    from aletheia.physics import MPMSimulator, NeuralFieldSolver, CoupledPDESolver
    from aletheia.renderer import BloomEffect, DepthOfField, MotionBlur, Camera, CameraAnimator
    from aletheia.renderer.raytracer import SimpleSoftwareRenderer
    from aletheia.output import FrameExporter, MetadataRecorder, VideoEncoder


def setup_logging(rank: int, log_dir: str = "logs") -> logging.Logger:
    """Setup logging for distributed training"""
    log_path = Path(log_dir)
    log_path.mkdir(parents=True, exist_ok=True)
    
    logger = logging.getLogger(f"aletheia_rank{rank}")
    logger.setLevel(logging.DEBUG if rank == 0 else logging.WARNING)
    
    # File handler (all ranks)
    fh = logging.FileHandler(log_path / f"rank_{rank}.log")
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    ))
    logger.addHandler(fh)
    
    # Console handler (rank 0 only)
    if rank == 0:
        ch = logging.StreamHandler()
        ch.setLevel(logging.INFO)
        ch.setFormatter(logging.Formatter(
            '%(asctime)s - %(levelname)s - %(message)s'
        ))
        logger.addHandler(ch)
    
    return logger


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description="Project Aletheia: Neural Morphogenesis Engine"
    )
    
    parser.add_argument(
        "--config", "-c",
        type=str,
        default="configs/default.yaml",
        help="Path to configuration file"
    )
    parser.add_argument(
        "--preset",
        type=str,
        choices=["stellar_nursery", "coral_lattice", "neural_network"],
        help="Use a preset configuration"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed for reproducibility"
    )
    parser.add_argument(
        "--frames",
        type=int,
        default=None,
        help="Override number of frames"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="/data/frames",
        help="Output directory for frames"
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Resume from checkpoint"
    )
    parser.add_argument(
        "--preview",
        action="store_true",
        help="Generate quick preview (fewer frames, lower quality)"
    )
    parser.add_argument(
        "--local-rank",
        type=int,
        default=-1,
        help="Local rank for distributed training (set by torchrun)"
    )
    
    return parser.parse_args()


class AletheiaEngine:
    """
    Main simulation engine orchestrating all components
    """
    
    def __init__(
        self,
        config_path: str,
        output_dir: str = "/data/frames",
        seed: Optional[int] = None,
        preview_mode: bool = False,
    ):
        self.config_path = config_path
        self.output_dir = Path(output_dir)
        self.seed = seed or int(time.time())
        self.preview_mode = preview_mode
        
        # Will be initialized in setup() - use Optional types
        self.config: Optional[MorphogenesisConfig] = None
        self.distributed: Optional[DistributedManager] = None
        self.memory_manager: Optional[MemoryManager] = None
        self.logger: Optional[logging.Logger] = None
        self.device: Optional[torch.device] = None
        self.rank: int = 0
        self.world_size: int = 1
        self.local_rank: int = 0
        
        # Components - use Union types for models
        self.neural_model: Optional[torch.nn.Module] = None
        self.dna_system: Optional[LatentDNA] = None
        self.mpm_simulator: Optional[MPMSimulator] = None
        self.field_solver: Optional[NeuralFieldSolver] = None
        self.pde_solver: Optional[CoupledPDESolver] = None
        self.renderer: Optional[Any] = None  # Can be SimpleSoftwareRenderer or EnhancedSoftwareRenderer
        self.frame_exporter: Optional[FrameExporter] = None
        self.metadata_recorder: Optional[MetadataRecorder] = None
        self.video_encoder: Optional[VideoEncoder] = None
        self.scheduler: Optional[MorphogenesisScheduler] = None
        self.bloom: Optional[BloomEffect] = None
        self.dof: Optional[DepthOfField] = None
        self.motion_blur: Optional[MotionBlur] = None
        self.camera: Optional[Camera] = None
        self.camera_animator: Optional[CameraAnimator] = None
        self.palette_name: str = "bioluminescent"
        self.latent_dna: Optional[torch.Tensor] = None
        
        # State
        self.particle_positions: Optional[torch.Tensor] = None
        self.particle_velocities: Optional[torch.Tensor] = None
        self.particle_colors: Optional[torch.Tensor] = None
        self.grid_velocity: Optional[torch.Tensor] = None
        self.current_frame: int = 0
        self.total_frames: int = 1000
    
    def setup(self):
        """Initialize all components"""
        # Import modules
        from aletheia.core import load_config, DistributedManager, MemoryManager
        
        # Setup distributed training
        self.distributed = DistributedManager()
        self.distributed.initialize()
        self.rank = self.distributed.rank
        self.world_size = self.distributed.world_size
        self.local_rank = self.distributed.local_rank
        self.device = self.distributed.device
        
        # Setup logging
        self.logger = setup_logging(self.rank, str(self.output_dir / "logs"))
        
        # Assert logger is not None for type narrowing
        assert self.logger is not None, "Logger failed to initialize"
        self.logger.info(f"Initializing Aletheia Engine (rank {self.rank}/{self.world_size})")
        
        # Load configuration
        self.config = load_config(self.config_path)
        
        # Assert config and device are not None for type narrowing
        assert self.config is not None, "Config failed to load"
        assert self.device is not None, "Device failed to initialize"
        
        # Override for preview mode
        if self.preview_mode:
            self.config.simulation.total_frames = min(60, self.config.simulation.total_frames)  # More frames for preview
            self.config.simulation.num_particles = min(50000, self.config.simulation.num_particles)  # Fewer but still visible
            self.config.simulation.grid_resolution = 32  # Smaller grid
            self.config.render.resolution = (1280, 720)  # Better resolution for preview
            self.logger.info("Preview mode: Using optimized settings for quick visualization")
        
        self.total_frames = self.config.simulation.total_frames
        
        # Setup memory management
        self.memory_manager = MemoryManager(
            max_memory_gb=self.config.memory.max_allocation_gb,
            device_id=self.local_rank,
            enable_monitoring=True,
        )
        
        # Set random seed
        self._set_seed(self.seed)
        
        # Initialize components
        self._init_neural_models()
        self._init_physics()
        self._init_renderer()
        self._init_output()
        self._init_particles()
        
        self.logger.info("Aletheia Engine initialized successfully")
    
    def _assert_initialized(self) -> None:
        """Assert that all required components are initialized. Called by methods that require setup()."""
        assert self.config is not None, "Config not initialized. Call setup() first."
        assert self.logger is not None, "Logger not initialized. Call setup() first."
        assert self.device is not None, "Device not initialized. Call setup() first."
    
    def _set_seed(self, seed: int):
        """Set random seeds for reproducibility"""
        assert self.logger is not None
        torch.manual_seed(seed + self.rank)
        np.random.seed(seed + self.rank)
        
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed + self.rank)
        
        self.logger.info(f"Random seed set to {seed + self.rank}")
    
    def _init_neural_models(self):
        """Initialize neural network models"""
        assert self.config is not None
        assert self.logger is not None
        assert self.device is not None
        
        self.logger.info("Initializing neural models...")
        
        from aletheia.models import (
            TimeConditionedFNO3D,
            LightweightUNetMPM,
            LatentDNA,
            MorphogenesisScheduler,
        )
        
        # Choose model based on config
        model_type = self.config.neural.model_type
        
        if model_type == "fno":
            self.neural_model = TimeConditionedFNO3D(
                in_channels=4,  # velocity (3) + density (1)
                out_channels=3,  # velocity
                modes=self.config.neural.fno_modes,
                width=self.config.neural.fno_width,
                num_layers=self.config.neural.fno_num_layers,
                latent_dim=self.config.neural.latent_dim,
            )
        else:  # unet
            self.neural_model = LightweightUNetMPM(
                in_channels=4,
                out_channels=3,
                base_channels=self.config.neural.unet_base_channels,
                latent_dim=self.config.neural.latent_dim,
            )
        
        self.neural_model = self.neural_model.to(self.device)
        
        # Wrap with DDP if distributed
        if self.world_size > 1:
            assert self.distributed is not None
            self.neural_model = self.distributed.wrap_model(self.neural_model)
        
        # DNA system
        self.dna_system = LatentDNA(
            dim=self.config.neural.latent_dim,
            device=self.device,
        )
        # Generate initial latent
        torch.manual_seed(self.seed)
        self.latent_dna = self.dna_system.vector
        
        # Morphogenesis scheduler
        self.scheduler = MorphogenesisScheduler(
            total_frames=self.total_frames,
            chaos_end=self.config.morphogenesis.chaos_frames[1],
            emergence_end=self.config.morphogenesis.emergence_frames[1],
            refinement_end=self.config.morphogenesis.refinement_frames[1],
        )
        
        self.logger.info(f"Neural model: {model_type} with {sum(p.numel() for p in self.neural_model.parameters()):,} parameters")
    
    def _init_physics(self):
        """Initialize physics simulation components"""
        assert self.config is not None
        assert self.logger is not None
        assert self.device is not None
        assert self.neural_model is not None
        
        self.logger.info("Initializing physics engine...")
        
        from aletheia.physics import (
            MPMSimulator,
            NeuralFieldSolver,
            CoupledPDESolver,
        )
        from aletheia.physics.mpm import MaterialProperties, init_taichi
        
        # Initialize Taichi for GPU computation
        init_taichi(device_id=0, memory_gb=12.0)
        
        # Create material properties from config
        material = MaterialProperties(
            density=self.config.simulation.density,
            young_modulus=self.config.simulation.young_modulus,
            poisson_ratio=self.config.simulation.poisson_ratio,
            viscosity=self.config.simulation.viscosity,
        )
        
        # MPM simulator - cast gravity to proper tuple type
        gravity_tuple: tuple[float, float, float] = (
            float(self.config.simulation.gravity[0]),
            float(self.config.simulation.gravity[1]),
            float(self.config.simulation.gravity[2]),
        )
        
        self.mpm_simulator = MPMSimulator(
            num_particles=self.config.simulation.num_particles,
            grid_resolution=self.config.simulation.grid_resolution,
            domain_min=self.config.simulation.domain_min,
            domain_max=self.config.simulation.domain_max,
            material=material,
            dt=self.config.simulation.dt,
            gravity=gravity_tuple,
            damping=self.config.simulation.damping,
        )
        
        # Neural field solver
        self.field_solver = NeuralFieldSolver(
            model=self.neural_model,
            grid_resolution=self.config.simulation.grid_resolution,
            domain_min=self.config.simulation.domain_min,
            domain_max=self.config.simulation.domain_max,
            device=self.device,
        )
        
        # PDE solver
        self.pde_solver = CoupledPDESolver(
            resolution=self.config.simulation.grid_resolution,
            device=self.device,
        )
        
        self.logger.info("Physics engine initialized")
    
    def _init_renderer(self):
        """Initialize rendering system"""
        assert self.config is not None
        assert self.logger is not None
        assert self.device is not None
        
        self.logger.info("Initializing renderer...")
        
        from aletheia.renderer import (
            BloomEffect,
            DepthOfField,
            MotionBlur,
            BioluminescentPalette,
            Camera,
            CameraAnimator,
        )
        from aletheia.renderer.raytracer import EnhancedSoftwareRenderer
        
        # Create resolution tuple explicitly with proper type
        render_res: tuple[int, int] = (self.config.render.resolution[0], self.config.render.resolution[1])
        
        # Use enhanced software renderer for better visuals
        self.renderer = EnhancedSoftwareRenderer(
            resolution=render_res,
            device=self.device,
            particle_base_size=5.0,
            glow_intensity=1.8,
            background_color=(0.01, 0.01, 0.03),
        )
        
        # Post-processing effects
        self.bloom = BloomEffect(
            resolution=render_res,
            device=self.device,
        )
        
        self.dof = DepthOfField(
            resolution=render_res,
            device=self.device,
        )
        
        self.motion_blur = MotionBlur(
            resolution=render_res,
            samples=self.config.render.motion_blur_samples,
            device=self.device,
        )
        
        # Color palette (use as class with classmethods)
        self.palette_name = self.config.render.particle_color_palette
        
        # Camera
        self.camera = Camera(
            position=(0, 0, 8),
            look_at=(0, 0, 0),
            fov=self.config.render.camera_fov,
            aspect_ratio=self.config.render.resolution[0] / self.config.render.resolution[1],
        )
        
        # Camera animator
        self.camera_animator = CameraAnimator(
            camera=self.camera,
            total_frames=self.total_frames,
        )
        self.camera_animator.set_orbit_animation(
            center=(0, 0, 0),
            radius=8.0,
            height=1.0,
            speed=0.5,
            vertical_oscillation=0.5,
        )
        
        self.logger.info("Renderer initialized")
    
    def _init_output(self):
        """Initialize output systems"""
        assert self.config is not None
        assert self.logger is not None
        
        self.logger.info("Initializing output systems...")
        
        from aletheia.output import (
            FrameExporter,
            MetadataRecorder,
            VideoEncoder,
        )
        
        # Frame exporter
        self.frame_exporter = FrameExporter(
            output_dir=str(self.output_dir),
            resolution=self.config.render.resolution,
            format=self.config.output.frame_format,
            prefix="frame",
            rank=self.rank,
        )
        
        # Metadata recorder
        self.metadata_recorder = MetadataRecorder(
            output_dir=str(self.output_dir / "metadata"),
            rank=self.rank,
        )
        
        # Video encoder
        self.video_encoder = VideoEncoder(
            output_dir=str(self.output_dir.parent / "output"),
            rank=self.rank,
        )
        
        self.logger.info("Output systems initialized")
    
    def _init_particles(self):
        """Initialize particle system with noise"""
        assert self.config is not None
        assert self.logger is not None
        assert self.device is not None
        assert self.mpm_simulator is not None
        
        self.logger.info("Initializing particle system...")
        
        from aletheia.renderer.effects import BioluminescentPalette
        
        num_particles = self.config.simulation.num_particles
        
        # Initialize from noise (sphere distribution)
        theta = torch.rand(num_particles, device=self.device) * 2 * np.pi
        phi = torch.acos(2 * torch.rand(num_particles, device=self.device) - 1)
        r = torch.pow(torch.rand(num_particles, device=self.device), 1/3) * 2  # Radius 2
        
        # Spherical to Cartesian
        self.particle_positions = torch.stack([
            r * torch.sin(phi) * torch.cos(theta),
            r * torch.sin(phi) * torch.sin(theta),
            r * torch.cos(phi),
        ], dim=1)
        
        # Add some noise
        noise = torch.randn_like(self.particle_positions) * 0.1
        self.particle_positions += noise
        
        # Zero initial velocity
        self.particle_velocities = torch.zeros_like(self.particle_positions)
        
        # Initial colors from palette based on radial distance
        radial_dist = torch.norm(self.particle_positions, dim=1)
        radial_normalized = radial_dist / radial_dist.max()  # Normalize to [0, 1]
        self.particle_colors = BioluminescentPalette.colorize_by_value(
            radial_normalized,
            palette_name=self.palette_name,
        )
        
        # Initialize grid
        grid_res = self.config.simulation.grid_resolution
        self.grid_velocity = torch.zeros(
            (1, 3, grid_res, grid_res, grid_res),
            device=self.device,
        )
        
        # Initialize MPM simulator with particle data
        pos_np = self.particle_positions.cpu().numpy().astype(np.float32)
        vel_np = self.particle_velocities.cpu().numpy().astype(np.float32)
        color_np = self.particle_colors.cpu().numpy().astype(np.float32)
        
        self.mpm_simulator.x.from_numpy(pos_np)
        self.mpm_simulator.v.from_numpy(vel_np)
        self.mpm_simulator.color.from_numpy(color_np)
        
        # Initialize deformation gradients to identity
        F_np = np.tile(np.eye(3, dtype=np.float32), (num_particles, 1, 1))
        self.mpm_simulator.F.from_numpy(F_np)
        
        # Initialize APIC to zero
        C_np = np.zeros((num_particles, 3, 3), dtype=np.float32)
        self.mpm_simulator.C.from_numpy(C_np)
        
        # Initialize Jp
        self.mpm_simulator.Jp.fill(1.0)
        
        self.logger.info(f"Initialized {num_particles:,} particles")
    
    def run(self):
        """Run the full simulation"""
        assert self.logger is not None
        assert self.config is not None
        assert self.metadata_recorder is not None
        assert self.distributed is not None
        
        self.logger.info(f"Starting simulation ({self.total_frames} frames)")
        
        # Record simulation start
        try:
            config_dict = asdict(self.config)
        except Exception:
            config_dict = {}
        self.metadata_recorder.record_simulation_start(
            config=config_dict,
            dna_seed=self.seed,
        )
        
        start_time = time.time()
        
        try:
            for frame in range(self.current_frame, self.total_frames):
                frame_start = time.time()
                
                # Step simulation
                self._step_simulation(frame)
                
                # Render frame
                rendered = self._render_frame(frame)
                
                # Export frame
                if self.rank == 0:
                    self._export_frame(frame, rendered)
                
                # Synchronize distributed
                if self.world_size > 1:
                    self.distributed.barrier()
                
                # Logging
                frame_time = time.time() - frame_start
                if frame % 10 == 0 or frame == self.total_frames - 1:
                    elapsed = time.time() - start_time
                    eta = (elapsed / (frame + 1)) * (self.total_frames - frame - 1)
                    self.logger.info(
                        f"Frame {frame + 1}/{self.total_frames} | "
                        f"Time: {frame_time:.2f}s | "
                        f"ETA: {eta/60:.1f}min"
                    )
                
                self.current_frame = frame + 1
            
            # Finalize
            self._finalize()
            
        except Exception as e:
            self.logger.error(f"Simulation failed: {e}")
            self.metadata_recorder.record_simulation_end(
                success=False,
                error_message=str(e),
            )
            raise
    
    def _step_simulation(self, frame: int):
        """Step physics simulation forward"""
        assert self.scheduler is not None
        assert self.dna_system is not None
        assert self.field_solver is not None
        assert self.neural_model is not None
        assert self.pde_solver is not None
        assert self.config is not None
        assert self.device is not None
        assert self.mpm_simulator is not None
        assert self.particle_positions is not None
        assert self.grid_velocity is not None
        
        # Get current phase and parameters from scheduler
        phase = self.scheduler.get_phase(frame)
        phase_params = self.scheduler.get_schedule_params(frame)
        
        # Get DNA latent vector
        dna_latent = self.dna_system.vector
        
        # Compute density field from particles
        density = self.field_solver.compute_density_field(self.particle_positions)
        
        # Combine velocity and density for neural input (need to match dimensions)
        # grid_velocity is (1, 3, D, H, W), density is (D, H, W)
        density_4d = density.unsqueeze(0).unsqueeze(0)  # (1, 1, D, H, W)
        input_field = torch.cat([self.grid_velocity, density_4d], dim=1)  # (1, 4, D, H, W)
        
        # Neural prediction (disable autocast for complex FFT operations in FNO)
        t_embed = torch.tensor([frame / self.total_frames], device=self.device, dtype=torch.float32)
        with torch.no_grad():
            predicted_velocity = self.neural_model(
                input_field.float(),  # Ensure float32
                latent=dna_latent.float(),
                time_embedding=t_embed,
            )
        
        # PDE step (just step the PDE solver, it maintains its own state)
        self.pde_solver.step(self.config.simulation.dt)
        pde_velocity = self.pde_solver.get_combined_field()
        
        # Blend neural and PDE based on phase
        dna_influence = phase_params.get("dna_influence", 0.7)
        
        # Ensure pde_velocity matches grid_velocity shape
        if pde_velocity.dim() == 4:  # (3, D, H, W)
            pde_velocity = pde_velocity.unsqueeze(0)  # (1, 3, D, H, W)
        
        self.grid_velocity = (
            dna_influence * predicted_velocity +
            (1 - dna_influence) * pde_velocity
        )
        
        # Convert grid velocity to numpy for MPM
        assert self.grid_velocity is not None  # Reassert after reassignment
        grid_vel_np = self.grid_velocity[0].permute(1, 2, 3, 0).cpu().numpy()  # (D, H, W, 3)
        
        # MPM substep with external force from grid velocity
        self.mpm_simulator.substep(n_substeps=self.config.simulation.substeps_per_frame)
        
        # Update particle positions/velocities from MPM
        self.particle_positions = torch.from_numpy(
            self.mpm_simulator.get_positions()
        ).to(self.device)
        self.particle_velocities = torch.from_numpy(
            self.mpm_simulator.get_velocities()
        ).to(self.device)
        
        # Update colors based on velocity/phase
        self._update_particle_colors(frame, phase)
    
    def _update_particle_colors(self, frame: int, phase: str):
        """Update particle colors based on dynamics"""
        assert self.particle_velocities is not None
        
        from aletheia.renderer.effects import BioluminescentPalette
        
        # Velocity-based coloring
        speeds = torch.norm(self.particle_velocities, dim=1)
        max_speed = speeds.max() + 1e-6
        
        # Normalize to [0, 1]
        normalized_speed = speeds / max_speed
        
        # Use palette to colorize by velocity
        self.particle_colors = BioluminescentPalette.colorize_by_value(
            normalized_speed,
            palette_name=self.palette_name,
        )
    
    def _render_frame(self, frame: int) -> np.ndarray:
        """Render current frame"""
        assert self.camera_animator is not None
        assert self.camera is not None
        assert self.renderer is not None
        assert self.bloom is not None
        assert self.config is not None
        assert self.device is not None
        assert self.particle_positions is not None
        assert self.particle_colors is not None
        
        # Update camera
        self.camera_animator.update(frame)
        
        # Set camera on renderer
        self.renderer.set_camera(
            position=tuple(self.camera.position),
            look_at=tuple(self.camera.look_at),
            fov=self.camera.fov,
        )
        
        # Render particles using SimpleSoftwareRenderer.render
        rendered = self.renderer.render(
            positions=self.particle_positions,
            colors=self.particle_colors,
        )
        
        # Apply bloom effect
        rendered_tensor = torch.from_numpy(rendered).to(self.device).permute(2, 0, 1).unsqueeze(0)  # (1, 3, H, W)
        rendered_tensor = self.bloom.apply(
            rendered_tensor,
            intensity=self.config.render.bloom_intensity,
            threshold=self.config.render.bloom_threshold,
        )
        rendered = rendered_tensor.squeeze(0).permute(1, 2, 0).cpu().numpy()
        
        return np.clip(rendered, 0, 1)
    
    def _export_frame(self, frame: int, rendered: np.ndarray):
        """Export rendered frame and metadata"""
        assert self.frame_exporter is not None
        assert self.metadata_recorder is not None
        assert self.particle_positions is not None
        assert self.particle_velocities is not None
        assert self.particle_colors is not None
        
        # Export frame
        self.frame_exporter.export_frame(
            frame=rendered,
            frame_idx=frame,
        )
        
        # Record metadata
        metadata = self.metadata_recorder.record_frame(
            frame_idx=frame,
            particle_positions=self.particle_positions.cpu().numpy(),
            particle_velocities=self.particle_velocities.cpu().numpy(),
            particle_colors=self.particle_colors.cpu().numpy(),
        )
        
        # Save periodic checkpoints (every 100 frames)
        checkpoint_interval = 100
        if frame % checkpoint_interval == 0 and frame > 0:
            self._save_checkpoint(frame)
    
    def _save_checkpoint(self, frame: int):
        """Save simulation checkpoint"""
        assert self.particle_positions is not None
        assert self.particle_velocities is not None
        assert self.particle_colors is not None
        assert self.logger is not None
        
        from aletheia.output.frame_exporter import ParticleStateExporter
        
        exporter = ParticleStateExporter(
            output_dir=str(self.output_dir / "checkpoints"),
            rank=self.rank,
        )
        
        exporter.export_checkpoint(
            frame_idx=frame,
            particle_state={
                "positions": self.particle_positions.cpu().numpy(),
                "velocities": self.particle_velocities.cpu().numpy(),
                "colors": self.particle_colors.cpu().numpy(),
            },
            model_state=self.neural_model.state_dict() if self.neural_model is not None and hasattr(self.neural_model, 'state_dict') else None,
        )
        
        self.logger.info(f"Checkpoint saved at frame {frame}")
    
    def _finalize(self):
        """Finalize simulation and encode video"""
        assert self.logger is not None
        assert self.metadata_recorder is not None
        assert self.frame_exporter is not None
        assert self.video_encoder is not None
        assert self.config is not None
        
        self.logger.info("Finalizing simulation...")
        
        # Record simulation end
        self.metadata_recorder.record_simulation_end(success=True)
        self.metadata_recorder.save_all_metadata()
        
        # Save export log
        self.frame_exporter.save_export_log()
        
        # Encode video
        if self.rank == 0:
            self.logger.info("Encoding video...")
            
            video_path = self.video_encoder.encode_video(
                input_dir=str(self.output_dir),
                output_name="morphogenesis.mp4",
                fps=self.config.output.video_fps,
                quality_preset="high",
            )
            
            if video_path:
                self.logger.info(f"Video saved to: {video_path}")
            
            # Print FFmpeg command for manual use
            self.video_encoder.print_ffmpeg_command(
                input_dir=str(self.output_dir),
                output_path=str(self.output_dir.parent / "output" / "morphogenesis.mp4"),
                fps=self.config.output.video_fps,
            )
        
        self.logger.info("Simulation complete!")
    
    def cleanup(self):
        """Cleanup resources"""
        if self.distributed:
            self.distributed.cleanup()
        
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def main():
    """Main entry point"""
    args = parse_args()
    
    # Determine config path
    if args.preset:
        config_path = f"configs/{args.preset}.yaml"
    else:
        config_path = args.config
    
    # Override frames if specified
    if args.frames:
        # Will be handled in engine setup
        pass
    
    # Create engine
    engine = AletheiaEngine(
        config_path=config_path,
        output_dir=args.output_dir,
        seed=args.seed,
        preview_mode=args.preview,
    )
    
    try:
        # Setup and run
        engine.setup()
        engine.run()
        
    except KeyboardInterrupt:
        print("\nSimulation interrupted by user")
        
    except Exception as e:
        print(f"Simulation failed: {e}")
        import traceback
        traceback.print_exc()
        
    finally:
        engine.cleanup()


if __name__ == "__main__":
    main()
