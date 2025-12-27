"""
Configuration Management for Project Aletheia
Handles YAML config loading, validation, and runtime configuration
"""

import os
import yaml
from dataclasses import dataclass, field
from typing import List, Tuple, Optional, Dict, Any, Union
from pathlib import Path
import torch


@dataclass
class SystemConfig:
    """System-level configuration"""
    seed: int = 42
    deterministic: bool = True
    benchmark: bool = True
    
    # GPU settings
    gpu_count: int = 4
    memory_per_gpu_gb: float = 15.0
    total_memory_gb: float = 60.0
    
    # CUDA settings
    device_memory_fraction: float = 0.94
    allow_growth: bool = False
    matmul_precision: str = "high"
    cudnn_benchmark: bool = True
    
    # Paths
    output_path: str = "/data/frames"
    checkpoint_path: str = "./checkpoints"
    log_path: str = "./logs"
    metadata_path: str = "./output"


@dataclass
class SimulationConfig:
    """Simulation parameters"""
    num_particles: int = 10_000_000
    particles_per_gpu: int = 2_500_000
    
    total_frames: int = 1000
    substeps_per_frame: int = 20
    dt: float = 5.0e-5
    
    # Domain
    domain_min: Tuple[float, float, float] = (-2.0, -2.0, -2.0)
    domain_max: Tuple[float, float, float] = (2.0, 2.0, 2.0)
    grid_resolution: int = 256
    
    # Physics
    physics_mode: str = "mpm"
    gravity: Tuple[float, float, float] = (0.0, -0.5, 0.0)
    damping: float = 0.999
    
    # Material
    material_type: str = "neo_hookean"
    density: float = 1000.0
    young_modulus: float = 5000.0
    poisson_ratio: float = 0.35
    viscosity: float = 0.1


@dataclass
class NeuralConfig:
    """Neural network configuration"""
    model_type: str = "fno"
    
    # Latent DNA
    latent_dim: int = 512
    seed_type: str = "gaussian"
    
    # FNO settings
    fno_modes: Tuple[int, int, int] = (16, 16, 16)
    fno_width: int = 64
    fno_num_layers: int = 4
    fno_activation: str = "gelu"
    
    # U-Net settings
    unet_base_channels: int = 64
    unet_channel_multipliers: Tuple[int, ...] = (1, 2, 4, 8)
    unet_num_res_blocks: int = 2
    unet_attention_resolutions: Tuple[int, ...] = (16, 8)
    unet_dropout: float = 0.1
    
    # Training
    learning_rate: float = 1.0e-4
    weight_decay: float = 1.0e-5
    batch_size: int = 4
    gradient_accumulation: int = 2
    mixed_precision: bool = True
    gradient_checkpointing: bool = True
    
    # DDP
    distributed_backend: str = "nccl"
    find_unused_parameters: bool = False
    broadcast_buffers: bool = True


@dataclass
class PDEConfig:
    """PDE and physics perturbation configuration"""
    base_type: str = "reaction_diffusion"
    
    # Navier-Stokes
    ns_viscosity: float = 0.01
    ns_pressure_iterations: int = 50
    ns_vorticity_confinement: float = 0.5
    
    # Reaction-Diffusion
    rd_diffusion_a: float = 1.0
    rd_diffusion_b: float = 0.5
    rd_feed_rate: float = 0.055
    rd_kill_rate: float = 0.062
    
    # Neural perturbation
    perturbation_strength: float = 0.3
    perturbation_frequency_bands: int = 8
    perturbation_spatial_scales: Tuple[float, ...] = (0.1, 0.5, 1.0, 2.0)


@dataclass
class MorphogenesisConfig:
    """Self-organization dynamics configuration"""
    # Phases (frame ranges)
    chaos_frames: Tuple[int, int] = (0, 100)
    emergence_frames: Tuple[int, int] = (100, 400)
    refinement_frames: Tuple[int, int] = (400, 700)
    stabilization_frames: Tuple[int, int] = (700, 1000)
    
    # Attractors
    attractor_type: str = "neural"
    attractor_strength_schedule: str = "cosine"
    attractor_min_strength: float = 0.1
    attractor_max_strength: float = 1.0
    
    # Templates
    templates: List[Dict[str, Any]] = field(default_factory=lambda: [
        {"name": "neural_network", "weight": 0.3},
        {"name": "coral_lattice", "weight": 0.3},
        {"name": "stellar_nursery", "weight": 0.4},
    ])


@dataclass
class RenderConfig:
    """Rendering configuration"""
    resolution: Tuple[int, int] = (3840, 2160)
    
    # Ray tracing
    raytracing_enabled: bool = True
    samples_per_pixel: int = 64
    max_bounces: int = 8
    russian_roulette_depth: int = 4
    
    # Camera
    camera_type: str = "perspective"
    camera_fov: float = 45.0
    camera_position: Tuple[float, float, float] = (0.0, 0.0, 8.0)
    camera_look_at: Tuple[float, float, float] = (0.0, 0.0, 0.0)
    camera_up: Tuple[float, float, float] = (0.0, 1.0, 0.0)
    
    # Camera animation
    camera_animation_enabled: bool = True
    camera_animation_type: str = "orbit"
    camera_orbit_radius: float = 8.0
    camera_orbit_speed: float = 0.5
    
    # Depth of Field
    dof_enabled: bool = True
    dof_aperture: float = 0.02
    dof_focus_distance: float = 8.0
    
    # Motion Blur
    motion_blur_enabled: bool = True
    motion_blur_shutter_time: float = 0.5
    motion_blur_samples: int = 8
    
    # Bloom
    bloom_enabled: bool = True
    bloom_intensity: float = 2.5
    bloom_threshold: float = 0.8
    bloom_radius: float = 0.01
    bloom_color_shift: Tuple[float, float, float] = (1.0, 0.9, 1.2)
    
    # Color grading
    exposure: float = 1.2
    gamma: float = 2.2
    saturation: float = 1.1
    contrast: float = 1.05
    
    # Particles
    particle_base_radius: float = 0.002
    particle_emission_strength: float = 5.0
    particle_color_palette: str = "bioluminescent"
    particle_color_by: str = "velocity"


@dataclass
class OutputConfig:
    """Output configuration"""
    frame_format: str = "png"
    frame_quality: int = 100
    frame_naming: str = "frame_{:06d}.png"
    
    video_enabled: bool = True
    video_format: str = "mp4"
    video_codec: str = "libx264"
    video_crf: int = 18
    video_preset: str = "slow"
    video_fps: int = 30
    
    save_latent_dna: bool = True
    save_particle_stats: bool = True
    save_render_params: bool = True
    metadata_format: str = "json"


@dataclass
class MemoryConfig:
    """Memory management configuration"""
    max_allocation_gb: float = 15.0
    chunk_size: int = 500_000
    prefetch_chunks: int = 2
    gradient_checkpointing: bool = True
    empty_cache_frequency: int = 10
    oom_retry: bool = True
    oom_chunk_reduction: float = 0.8


@dataclass
class LoggingConfig:
    """Logging configuration"""
    level: str = "INFO"
    console: bool = True
    file: bool = True
    
    progress_update_frequency: int = 10
    progress_show_eta: bool = True
    progress_show_memory: bool = True
    
    tensorboard_enabled: bool = True
    tensorboard_log_frequency: int = 50
    
    wandb_enabled: bool = False
    wandb_project: str = "aletheia"
    wandb_entity: Optional[str] = None


@dataclass
class AletheiaConfig:
    """Master configuration for Project Aletheia"""
    system: SystemConfig = field(default_factory=SystemConfig)
    simulation: SimulationConfig = field(default_factory=SimulationConfig)
    neural: NeuralConfig = field(default_factory=NeuralConfig)
    pde: PDEConfig = field(default_factory=PDEConfig)
    morphogenesis: MorphogenesisConfig = field(default_factory=MorphogenesisConfig)
    render: RenderConfig = field(default_factory=RenderConfig)
    output: OutputConfig = field(default_factory=OutputConfig)
    memory: MemoryConfig = field(default_factory=MemoryConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)
    
    def validate(self) -> bool:
        """Validate configuration parameters"""
        # Check GPU memory constraints
        particles_per_gpu = self.simulation.num_particles // self.system.gpu_count
        estimated_memory_gb = particles_per_gpu * 32 / (1024**3)  # ~32 bytes per particle
        
        if estimated_memory_gb > self.memory.max_allocation_gb * 0.5:
            print(f"Warning: Estimated particle memory ({estimated_memory_gb:.2f}GB) "
                  f"may exceed safe limits")
        
        # Validate frame ranges
        total_frames = self.simulation.total_frames
        for phase_name in ['chaos_frames', 'emergence_frames', 'refinement_frames', 'stabilization_frames']:
            phase = getattr(self.morphogenesis, phase_name)
            if phase[1] > total_frames:
                print(f"Warning: {phase_name} exceeds total_frames")
        
        return True
    
    def setup_cuda(self):
        """Configure CUDA settings for A16 GPUs"""
        if torch.cuda.is_available():
            # Set matmul precision for Ampere (TF32)
            torch.set_float32_matmul_precision(self.system.matmul_precision)
            
            # Enable cudnn benchmark
            torch.backends.cudnn.benchmark = self.system.cudnn_benchmark
            
            # Set deterministic mode if requested
            if self.system.deterministic:
                torch.backends.cudnn.deterministic = True
                torch.use_deterministic_algorithms(True, warn_only=True)
    
    def get_device(self, local_rank: int = 0) -> torch.device:
        """Get the appropriate device for a given rank"""
        if torch.cuda.is_available():
            return torch.device(f"cuda:{local_rank}")
        return torch.device("cpu")


def load_config(config_path: Union[str, Path]) -> AletheiaConfig:
    """Load configuration from YAML file"""
    config_path = Path(config_path)
    
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    
    with open(config_path, 'r') as f:
        yaml_config = yaml.safe_load(f)
    
    # Handle defaults inheritance
    if '_defaults_' in yaml_config:
        base_configs = yaml_config.pop('_defaults_')
        base_path = config_path.parent
        
        merged_config = {}
        for base_name in base_configs:
            base_file = base_path / f"{base_name}.yaml"
            if base_file.exists():
                with open(base_file, 'r') as f:
                    base_config = yaml.safe_load(f)
                    merged_config = deep_merge(merged_config, base_config)
        
        yaml_config = deep_merge(merged_config, yaml_config)
    
    return parse_yaml_to_config(yaml_config)


def deep_merge(base: Dict, override: Dict) -> Dict:
    """Deep merge two dictionaries"""
    result = base.copy()
    
    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = deep_merge(result[key], value)
        else:
            result[key] = value
    
    return result


def parse_yaml_to_config(yaml_config: Dict[str, Any]) -> AletheiaConfig:
    """Parse YAML dictionary into AletheiaConfig dataclass"""
    config = AletheiaConfig()
    
    # Parse system config
    if 'system' in yaml_config:
        sys_cfg = yaml_config['system']
        config.system = SystemConfig(
            seed=sys_cfg.get('seed', 42),
            deterministic=sys_cfg.get('deterministic', True),
            benchmark=sys_cfg.get('benchmark', True),
            gpu_count=sys_cfg.get('gpus', {}).get('count', 4),
            memory_per_gpu_gb=sys_cfg.get('gpus', {}).get('memory_per_gpu_gb', 15),
            total_memory_gb=sys_cfg.get('gpus', {}).get('total_memory_gb', 60),
            device_memory_fraction=sys_cfg.get('cuda', {}).get('device_memory_fraction', 0.94),
            matmul_precision=sys_cfg.get('cuda', {}).get('matmul_precision', 'high'),
            cudnn_benchmark=sys_cfg.get('cuda', {}).get('cudnn_benchmark', True),
            output_path=sys_cfg.get('paths', {}).get('output', '/data/frames'),
            checkpoint_path=sys_cfg.get('paths', {}).get('checkpoints', './checkpoints'),
            log_path=sys_cfg.get('paths', {}).get('logs', './logs'),
            metadata_path=sys_cfg.get('paths', {}).get('metadata', './output'),
        )
    
    # Parse simulation config
    if 'simulation' in yaml_config:
        sim_cfg = yaml_config['simulation']
        domain = sim_cfg.get('domain', {})
        physics = sim_cfg.get('physics', {})
        material = sim_cfg.get('material', {})
        
        config.simulation = SimulationConfig(
            num_particles=sim_cfg.get('num_particles', 10_000_000),
            particles_per_gpu=sim_cfg.get('particles_per_gpu', 2_500_000),
            total_frames=sim_cfg.get('total_frames', 1000),
            substeps_per_frame=sim_cfg.get('substeps_per_frame', 20),
            dt=sim_cfg.get('dt', 5.0e-5),
            domain_min=tuple(domain.get('min', [-2.0, -2.0, -2.0])),
            domain_max=tuple(domain.get('max', [2.0, 2.0, 2.0])),
            grid_resolution=domain.get('grid_resolution', 256),
            physics_mode=physics.get('mode', 'mpm'),
            gravity=tuple(physics.get('gravity', [0.0, -0.5, 0.0])),
            damping=physics.get('damping', 0.999),
            material_type=material.get('type', 'neo_hookean'),
            density=material.get('density', 1000.0),
            young_modulus=material.get('young_modulus', 5000.0),
            poisson_ratio=material.get('poisson_ratio', 0.35),
            viscosity=material.get('viscosity', 0.1),
        )
    
    # Parse neural config
    if 'neural' in yaml_config:
        nn_cfg = yaml_config['neural']
        latent = nn_cfg.get('latent', {})
        fno = nn_cfg.get('fno', {})
        unet = nn_cfg.get('unet', {})
        training = nn_cfg.get('training', {})
        distributed = nn_cfg.get('distributed', {})
        
        config.neural = NeuralConfig(
            model_type=nn_cfg.get('model_type', 'fno'),
            latent_dim=latent.get('dim', 512),
            seed_type=latent.get('seed_type', 'gaussian'),
            fno_modes=tuple(fno.get('modes', [16, 16, 16])),
            fno_width=fno.get('width', 64),
            fno_num_layers=fno.get('num_layers', 4),
            fno_activation=fno.get('activation', 'gelu'),
            unet_base_channels=unet.get('base_channels', 64),
            unet_channel_multipliers=tuple(unet.get('channel_multipliers', [1, 2, 4, 8])),
            unet_num_res_blocks=unet.get('num_res_blocks', 2),
            unet_attention_resolutions=tuple(unet.get('attention_resolutions', [16, 8])),
            unet_dropout=unet.get('dropout', 0.1),
            learning_rate=training.get('learning_rate', 1.0e-4),
            weight_decay=training.get('weight_decay', 1.0e-5),
            batch_size=training.get('batch_size', 4),
            gradient_accumulation=training.get('gradient_accumulation', 2),
            mixed_precision=training.get('mixed_precision', True),
            gradient_checkpointing=training.get('gradient_checkpointing', True),
            distributed_backend=distributed.get('backend', 'nccl'),
            find_unused_parameters=distributed.get('find_unused_parameters', False),
            broadcast_buffers=distributed.get('broadcast_buffers', True),
        )
    
    # Parse PDE config
    if 'pde' in yaml_config:
        pde_cfg = yaml_config['pde']
        ns = pde_cfg.get('navier_stokes', {})
        rd = pde_cfg.get('reaction_diffusion', {})
        pert = pde_cfg.get('perturbation', {})
        
        config.pde = PDEConfig(
            base_type=pde_cfg.get('base_type', 'reaction_diffusion'),
            ns_viscosity=ns.get('viscosity', 0.01),
            ns_pressure_iterations=ns.get('pressure_iterations', 50),
            ns_vorticity_confinement=ns.get('vorticity_confinement', 0.5),
            rd_diffusion_a=rd.get('diffusion_a', 1.0),
            rd_diffusion_b=rd.get('diffusion_b', 0.5),
            rd_feed_rate=rd.get('feed_rate', 0.055),
            rd_kill_rate=rd.get('kill_rate', 0.062),
            perturbation_strength=pert.get('strength', 0.3),
            perturbation_frequency_bands=pert.get('frequency_bands', 8),
            perturbation_spatial_scales=tuple(pert.get('spatial_scale', [0.1, 0.5, 1.0, 2.0])),
        )
    
    # Parse morphogenesis config
    if 'morphogenesis' in yaml_config:
        morph_cfg = yaml_config['morphogenesis']
        phases = morph_cfg.get('phases', {})
        attractors = morph_cfg.get('attractors', {})
        
        config.morphogenesis = MorphogenesisConfig(
            chaos_frames=tuple(phases.get('chaos_frames', [0, 100])),
            emergence_frames=tuple(phases.get('emergence_frames', [100, 400])),
            refinement_frames=tuple(phases.get('refinement_frames', [400, 700])),
            stabilization_frames=tuple(phases.get('stabilization_frames', [700, 1000])),
            attractor_type=attractors.get('type', 'neural'),
            attractor_strength_schedule=attractors.get('strength_schedule', 'cosine'),
            attractor_min_strength=attractors.get('min_strength', 0.1),
            attractor_max_strength=attractors.get('max_strength', 1.0),
            templates=morph_cfg.get('templates', []),
        )
    
    # Parse render config
    if 'render' in yaml_config:
        render_cfg = yaml_config['render']
        rt = render_cfg.get('raytracing', {})
        cam = render_cfg.get('camera', {})
        cam_anim = cam.get('animation', {})
        dof = render_cfg.get('dof', {})
        mb = render_cfg.get('motion_blur', {})
        bloom = render_cfg.get('bloom', {})
        color = render_cfg.get('color', {})
        particles = render_cfg.get('particles', {})
        
        config.render = RenderConfig(
            resolution=tuple(render_cfg.get('resolution', [3840, 2160])),
            raytracing_enabled=rt.get('enabled', True),
            samples_per_pixel=rt.get('samples_per_pixel', 64),
            max_bounces=rt.get('max_bounces', 8),
            russian_roulette_depth=rt.get('russian_roulette_depth', 4),
            camera_type=cam.get('type', 'perspective'),
            camera_fov=cam.get('fov', 45.0),
            camera_position=tuple(cam.get('position', [0.0, 0.0, 8.0])),
            camera_look_at=tuple(cam.get('look_at', [0.0, 0.0, 0.0])),
            camera_up=tuple(cam.get('up', [0.0, 1.0, 0.0])),
            camera_animation_enabled=cam_anim.get('enabled', True),
            camera_animation_type=cam_anim.get('type', 'orbit'),
            camera_orbit_radius=cam_anim.get('orbit_radius', 8.0),
            camera_orbit_speed=cam_anim.get('orbit_speed', 0.5),
            dof_enabled=dof.get('enabled', True),
            dof_aperture=dof.get('aperture', 0.02),
            dof_focus_distance=dof.get('focus_distance', 8.0),
            motion_blur_enabled=mb.get('enabled', True),
            motion_blur_shutter_time=mb.get('shutter_time', 0.5),
            motion_blur_samples=mb.get('samples', 8),
            bloom_enabled=bloom.get('enabled', True),
            bloom_intensity=bloom.get('intensity', 2.5),
            bloom_threshold=bloom.get('threshold', 0.8),
            bloom_radius=bloom.get('radius', 0.01),
            bloom_color_shift=tuple(bloom.get('color_shift', [1.0, 0.9, 1.2])),
            exposure=color.get('exposure', 1.2),
            gamma=color.get('gamma', 2.2),
            saturation=color.get('saturation', 1.1),
            contrast=color.get('contrast', 1.05),
            particle_base_radius=particles.get('base_radius', 0.002),
            particle_emission_strength=particles.get('emission_strength', 5.0),
            particle_color_palette=particles.get('color_palette', 'bioluminescent'),
            particle_color_by=particles.get('color_by', 'velocity'),
        )
    
    # Parse output config
    if 'output' in yaml_config:
        out_cfg = yaml_config['output']
        frames = out_cfg.get('frames', {})
        video = out_cfg.get('video', {})
        meta = out_cfg.get('metadata', {})
        
        config.output = OutputConfig(
            frame_format=frames.get('format', 'png'),
            frame_quality=frames.get('quality', 100),
            frame_naming=frames.get('naming', 'frame_{:06d}.png'),
            video_enabled=video.get('enabled', True),
            video_format=video.get('format', 'mp4'),
            video_codec=video.get('codec', 'libx264'),
            video_crf=video.get('crf', 18),
            video_preset=video.get('preset', 'slow'),
            video_fps=video.get('fps', 30),
            save_latent_dna=out_cfg.get('save_latent_dna', True),
            save_particle_stats=out_cfg.get('save_particle_stats', True),
            save_render_params=out_cfg.get('save_render_params', True),
            metadata_format=meta.get('format', 'json'),
        )
    
    # Parse memory config
    if 'memory' in yaml_config:
        mem_cfg = yaml_config['memory']
        config.memory = MemoryConfig(
            max_allocation_gb=mem_cfg.get('max_allocation_gb', 15.0),
            chunk_size=mem_cfg.get('chunk_size', 500_000),
            prefetch_chunks=mem_cfg.get('prefetch_chunks', 2),
            gradient_checkpointing=mem_cfg.get('gradient_checkpointing', True),
            empty_cache_frequency=mem_cfg.get('empty_cache_frequency', 10),
            oom_retry=mem_cfg.get('oom_retry', True),
            oom_chunk_reduction=mem_cfg.get('oom_chunk_reduction', 0.8),
        )
    
    # Parse logging config
    if 'logging' in yaml_config:
        log_cfg = yaml_config['logging']
        progress = log_cfg.get('progress', {})
        tb = log_cfg.get('tensorboard', {})
        wandb = log_cfg.get('wandb', {})
        
        config.logging = LoggingConfig(
            level=log_cfg.get('level', 'INFO'),
            console=log_cfg.get('console', True),
            file=log_cfg.get('file', True),
            progress_update_frequency=progress.get('update_frequency', 10),
            progress_show_eta=progress.get('show_eta', True),
            progress_show_memory=progress.get('show_memory', True),
            tensorboard_enabled=tb.get('enabled', True),
            tensorboard_log_frequency=tb.get('log_frequency', 50),
            wandb_enabled=wandb.get('enabled', False),
            wandb_project=wandb.get('project', 'aletheia'),
            wandb_entity=wandb.get('entity', None),
        )
    
    return config


def create_default_config() -> AletheiaConfig:
    """Create a default configuration"""
    return AletheiaConfig()
