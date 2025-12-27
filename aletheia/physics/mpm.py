"""
Material Point Method (MPM) Simulator using Taichi Lang
Implements "squishy/organic" physics for neural morphogenesis
Optimized for 4x NVIDIA A16 GPUs with 15GB per-GPU allocation

Note: This file uses Taichi-specific constructs that may show type errors
in static analysis tools but work correctly at runtime.
"""
# pyright: reportGeneralTypeIssues=false
# type: ignore

import taichi as ti
import numpy as np
import torch
from typing import Tuple, Optional, Dict, Any
from dataclasses import dataclass
import math


@dataclass
class MaterialProperties:
    """Material properties for MPM simulation"""
    density: float = 1000.0
    young_modulus: float = 5000.0
    poisson_ratio: float = 0.35
    viscosity: float = 0.1
    
    @property
    def lame_mu(self) -> float:
        """Shear modulus (Lamé's second parameter)"""
        return self.young_modulus / (2 * (1 + self.poisson_ratio))
    
    @property
    def lame_lambda(self) -> float:
        """Lamé's first parameter"""
        E = self.young_modulus
        nu = self.poisson_ratio
        return E * nu / ((1 + nu) * (1 - 2 * nu))


class ParticleSystem:
    """
    Manages particle data for distributed MPM simulation
    Handles particle partitioning across GPUs
    """
    
    def __init__(
        self,
        num_particles: int,
        dim: int = 3,
        device: torch.device = torch.device("cuda"),
    ):
        self.num_particles = num_particles
        self.dim = dim
        self.device = device
        
        # Particle data (PyTorch tensors for DDP compatibility)
        self.positions = torch.zeros(num_particles, dim, device=device)
        self.velocities = torch.zeros(num_particles, dim, device=device)
        self.colors = torch.zeros(num_particles, 3, device=device)
        self.masses = torch.ones(num_particles, device=device)
        
        # Deformation gradient (3x3 per particle for 3D)
        self.F = torch.eye(dim, device=device).unsqueeze(0).repeat(num_particles, 1, 1)
        
        # Affine velocity field (APIC)
        self.C = torch.zeros(num_particles, dim, dim, device=device)
        
        # Per-particle properties
        self.volumes = torch.zeros(num_particles, device=device)
        self.temperatures = torch.ones(num_particles, device=device)
    
    def initialize_cloud(
        self,
        center: Tuple[float, float, float] = (0.0, 0.0, 0.0),
        radius: float = 1.0,
        distribution: str = "gaussian",
    ):
        """Initialize particles as a chaotic cloud"""
        if distribution == "gaussian":
            # Gaussian distribution
            positions = torch.randn(self.num_particles, self.dim, device=self.device)
            positions = positions * radius / 3.0  # 3-sigma within radius
            
        elif distribution == "uniform_sphere":
            # Uniform distribution within sphere
            u = torch.rand(self.num_particles, device=self.device)
            v = torch.rand(self.num_particles, device=self.device)
            w = torch.rand(self.num_particles, device=self.device)
            
            r = radius * (u ** (1/3))
            theta = 2 * math.pi * v
            phi = torch.acos(2 * w - 1)
            
            positions = torch.stack([
                r * torch.sin(phi) * torch.cos(theta),
                r * torch.sin(phi) * torch.sin(theta),
                r * torch.cos(phi),
            ], dim=-1)
            
        elif distribution == "uniform_cube":
            positions = (torch.rand(self.num_particles, self.dim, device=self.device) - 0.5) * 2 * radius
            
        else:
            positions = torch.randn(self.num_particles, self.dim, device=self.device) * radius
        
        # Apply center offset
        center_tensor = torch.tensor(center, device=self.device)
        self.positions = positions + center_tensor
        
        # Initialize velocities with small random values
        self.velocities = torch.randn(self.num_particles, self.dim, device=self.device) * 0.1
        
        # Initialize colors based on position
        self._update_colors()
        
        # Reset deformation gradients
        self.F = torch.eye(self.dim, device=self.device).unsqueeze(0).repeat(self.num_particles, 1, 1)
        self.C = torch.zeros(self.num_particles, self.dim, self.dim, device=self.device)
    
    def _update_colors(self):
        """Update particle colors based on velocity or position"""
        # Normalize positions to [0, 1] for color mapping
        pos_min = self.positions.min(dim=0)[0]
        pos_max = self.positions.max(dim=0)[0]
        pos_range = pos_max - pos_min + 1e-6
        
        normalized = (self.positions - pos_min) / pos_range
        
        # Map to bioluminescent color palette (cyan to magenta)
        self.colors[:, 0] = 0.2 + 0.6 * normalized[:, 0]  # R
        self.colors[:, 1] = 0.8 - 0.3 * normalized[:, 1]  # G
        self.colors[:, 2] = 0.6 + 0.4 * normalized[:, 2]  # B
    
    def get_partition(
        self,
        rank: int,
        world_size: int,
    ) -> 'ParticleSystem':
        """Get particle partition for a specific GPU rank"""
        particles_per_rank = self.num_particles // world_size
        start = rank * particles_per_rank
        end = start + particles_per_rank if rank < world_size - 1 else self.num_particles
        
        partition = ParticleSystem(end - start, self.dim, self.device)
        partition.positions = self.positions[start:end].clone()
        partition.velocities = self.velocities[start:end].clone()
        partition.colors = self.colors[start:end].clone()
        partition.masses = self.masses[start:end].clone()
        partition.F = self.F[start:end].clone()
        partition.C = self.C[start:end].clone()
        partition.volumes = self.volumes[start:end].clone()
        
        return partition
    
    def to_numpy(self) -> Dict[str, np.ndarray]:
        """Export particle data to numpy arrays"""
        return {
            "positions": self.positions.cpu().numpy(),
            "velocities": self.velocities.cpu().numpy(),
            "colors": self.colors.cpu().numpy(),
            "masses": self.masses.cpu().numpy(),
        }


# Initialize Taichi for GPU computation
def init_taichi(device_id: int = 0, memory_gb: float = 15.0):
    """
    Initialize Taichi for A16 GPU
    Uses strict memory limits for shared environment
    """
    ti.init(
        arch=ti.cuda,
        device_memory_GB=memory_gb,
        kernel_profiler=False,
        default_fp=ti.f32,
        default_ip=ti.i32,
        fast_math=True,
    )


@ti.data_oriented
class MPMSimulator:
    """
    Taichi-based MPM Simulator for Neural Morphogenesis
    Implements neo-Hookean material model for organic/squishy behavior
    """
    
    def __init__(
        self,
        num_particles: int,
        grid_resolution: int = 256,
        domain_min: Tuple[float, float, float] = (-2.0, -2.0, -2.0),
        domain_max: Tuple[float, float, float] = (2.0, 2.0, 2.0),
        material: Optional[MaterialProperties] = None,
        dt: float = 5e-5,
        gravity: Tuple[float, float, float] = (0.0, -0.5, 0.0),
        damping: float = 0.999,
    ):
        self.num_particles = num_particles
        self.grid_res = grid_resolution
        self.domain_min = domain_min
        self.domain_max = domain_max
        self.dt = dt
        self.gravity = gravity
        self.damping = damping
        
        # Material properties
        self.material = material or MaterialProperties()
        
        # Grid spacing
        domain_size = tuple(
            domain_max[i] - domain_min[i] for i in range(3)
        )
        self.dx = max(domain_size) / grid_resolution
        self.inv_dx = 1.0 / self.dx
        
        # Particle volume
        self.p_vol = (self.dx * 0.5) ** 3
        self.p_mass = self.material.density * self.p_vol
        
        # Taichi fields
        self._init_fields()
    
    def _init_fields(self):
        """Initialize Taichi fields for particles and grid"""
        n_particles = self.num_particles
        n_grid = self.grid_res
        
        # Particle fields
        self.x = ti.Vector.field(3, dtype=ti.f32, shape=n_particles)  # Position
        self.v = ti.Vector.field(3, dtype=ti.f32, shape=n_particles)  # Velocity
        self.C = ti.Matrix.field(3, 3, dtype=ti.f32, shape=n_particles)  # APIC affine velocity
        self.F = ti.Matrix.field(3, 3, dtype=ti.f32, shape=n_particles)  # Deformation gradient
        self.Jp = ti.field(dtype=ti.f32, shape=n_particles)  # Plastic deformation
        self.color = ti.Vector.field(3, dtype=ti.f32, shape=n_particles)  # RGB color
        
        # Grid fields
        self.grid_v = ti.Vector.field(3, dtype=ti.f32, shape=(n_grid, n_grid, n_grid))
        self.grid_m = ti.field(dtype=ti.f32, shape=(n_grid, n_grid, n_grid))
        
        # External force field (from neural network)
        self.external_force = ti.Vector.field(3, dtype=ti.f32, shape=(n_grid, n_grid, n_grid))
    
    @ti.kernel
    def reset_grid(self):
        """Reset grid fields to zero"""
        for i, j, k in self.grid_v:
            self.grid_v[i, j, k] = ti.Vector([0.0, 0.0, 0.0])
            self.grid_m[i, j, k] = 0.0
    
    @ti.kernel
    def particle_to_grid(self):
        """Transfer particle data to grid (P2G)"""
        for p in range(self.num_particles):
            # Get grid base index
            base = (self.x[p] * self.inv_dx - 0.5).cast(int)  # type: ignore[union-attr]
            fx = self.x[p] * self.inv_dx - base.cast(float)  # type: ignore[union-attr]
            
            # Quadratic B-spline weights
            w = [
                0.5 * (1.5 - fx) ** 2,
                0.75 - (fx - 1) ** 2,
                0.5 * (fx - 0.5) ** 2
            ]
            
            # Neo-Hookean stress
            F = self.F[p]
            J = F.determinant()
            
            # Clamp J to prevent numerical issues
            J = ti.max(J, 0.1)
            
            # Compute Cauchy stress
            mu = self.material.lame_mu
            la = self.material.lame_lambda
            
            # Neo-Hookean: σ = μ(F F^T - I) / J + λ log(J) I / J
            F_T = F.transpose()
            stress = mu * (F @ F_T - ti.Matrix.identity(float, 3)) / J + \
                     la * ti.log(J) * ti.Matrix.identity(float, 3) / J
            
            # Affine momentum
            affine = -self.dt * self.p_vol * 4 * self.inv_dx ** 2 * stress + \
                     self.p_mass * self.C[p]
            
            # Scatter to grid
            for i, j, k in ti.static(ti.ndrange(3, 3, 3)):
                offset = ti.Vector([i, j, k])
                dpos = (offset.cast(float) - fx) * self.dx
                weight = w[i][0] * w[j][1] * w[k][2]  # type: ignore[index]
                
                grid_idx = base + offset
                
                # Bounds check - type ignore needed for Taichi Expr indexing
                if (0 <= grid_idx[0] < self.grid_res and  # type: ignore[index]
                   0 <= grid_idx[1] < self.grid_res and  # type: ignore[index]
                   0 <= grid_idx[2] < self.grid_res):  # type: ignore[index]
                    
                    self.grid_v[grid_idx] += weight * (
                        self.p_mass * self.v[p] + affine @ dpos  # type: ignore[operator]
                    )
                    self.grid_m[grid_idx] += weight * self.p_mass
    
    @ti.kernel
    def grid_operations(self):
        """Grid velocity update with boundary conditions"""
        for i, j, k in self.grid_v:
            if self.grid_m[i, j, k] > 1e-10:
                # Normalize by mass
                self.grid_v[i, j, k] /= self.grid_m[i, j, k]
                
                # Apply gravity
                self.grid_v[i, j, k] += self.dt * ti.Vector([
                    self.gravity[0],
                    self.gravity[1],
                    self.gravity[2]
                ])
                
                # Apply external neural force
                self.grid_v[i, j, k] += self.dt * self.external_force[i, j, k]
                
                # Apply damping
                self.grid_v[i, j, k] *= self.damping
                
                # Boundary conditions (sticky boundaries)
                boundary = 3
                if i < boundary or i >= self.grid_res - boundary:
                    self.grid_v[i, j, k][0] = 0.0
                if j < boundary or j >= self.grid_res - boundary:
                    self.grid_v[i, j, k][1] = 0.0
                if k < boundary or k >= self.grid_res - boundary:
                    self.grid_v[i, j, k][2] = 0.0
    
    @ti.kernel
    def grid_to_particle(self):
        """Transfer grid data back to particles (G2P)"""
        for p in range(self.num_particles):
            base = (self.x[p] * self.inv_dx - 0.5).cast(int)  # type: ignore[union-attr]
            fx = self.x[p] * self.inv_dx - base.cast(float)  # type: ignore[union-attr]
            
            # Quadratic B-spline weights
            w = [
                0.5 * (1.5 - fx) ** 2,
                0.75 - (fx - 1) ** 2,
                0.5 * (fx - 0.5) ** 2
            ]
            
            new_v = ti.Vector.zero(float, 3)
            new_C = ti.Matrix.zero(float, 3, 3)
            
            # Gather from grid
            for i, j, k in ti.static(ti.ndrange(3, 3, 3)):
                offset = ti.Vector([i, j, k])
                dpos = (offset.cast(float) - fx) * self.dx
                weight = w[i][0] * w[j][1] * w[k][2]  # type: ignore[index]
                
                grid_idx = base + offset
                
                # Bounds check - type ignore needed for Taichi Expr indexing
                if (0 <= grid_idx[0] < self.grid_res and  # type: ignore[index]
                   0 <= grid_idx[1] < self.grid_res and  # type: ignore[index]
                   0 <= grid_idx[2] < self.grid_res):  # type: ignore[index]
                    
                    g_v = self.grid_v[grid_idx]
                    new_v += weight * g_v
                    new_C += 4 * self.inv_dx ** 2 * weight * g_v.outer_product(dpos)
            
            # Update particle velocity and APIC
            self.v[p] = new_v
            self.C[p] = new_C
            
            # Update position
            self.x[p] += self.dt * self.v[p]
            
            # Update deformation gradient
            self.F[p] = (ti.Matrix.identity(float, 3) + self.dt * self.C[p]) @ self.F[p]  # type: ignore[operator]
            
            # Clamp position to domain
            for d in ti.static(range(3)):
                self.x[p][d] = ti.max(self.x[p][d], self.domain_min[d] + 0.05)
                self.x[p][d] = ti.min(self.x[p][d], self.domain_max[d] - 0.05)
    
    def step(self, external_force: Optional[np.ndarray] = None):
        """Perform one MPM simulation step"""
        # Apply external force field if provided
        if external_force is not None:
            self.external_force.from_numpy(external_force.astype(np.float32))
        else:
            # Reset external force
            self.external_force.fill(0)
        
        # MPM substep
        self.reset_grid()
        self.particle_to_grid()
        self.grid_operations()
        self.grid_to_particle()
    
    def substep(self, n_substeps: int = 20, external_force: Optional[np.ndarray] = None):
        """Perform multiple substeps"""
        for _ in range(n_substeps):
            self.step(external_force)
    
    def from_particle_system(self, particles: ParticleSystem):
        """Load particle data from ParticleSystem"""
        pos_np = particles.positions.cpu().numpy().astype(np.float32)
        vel_np = particles.velocities.cpu().numpy().astype(np.float32)
        color_np = particles.colors.cpu().numpy().astype(np.float32)
        
        self.x.from_numpy(pos_np)
        self.v.from_numpy(vel_np)
        self.color.from_numpy(color_np)
        
        # Initialize deformation gradients to identity
        F_np = np.tile(np.eye(3, dtype=np.float32), (self.num_particles, 1, 1))
        self.F.from_numpy(F_np)
        
        # Initialize APIC to zero
        C_np = np.zeros((self.num_particles, 3, 3), dtype=np.float32)
        self.C.from_numpy(C_np)
        
        # Initialize Jp
        self.Jp.fill(1.0)
    
    def to_particle_system(self, particles: ParticleSystem):
        """Export particle data to ParticleSystem"""
        particles.positions = torch.from_numpy(self.x.to_numpy()).to(particles.device)
        particles.velocities = torch.from_numpy(self.v.to_numpy()).to(particles.device)
        particles.colors = torch.from_numpy(self.color.to_numpy()).to(particles.device)
    
    def get_positions(self) -> np.ndarray:
        """Get particle positions as numpy array"""
        return self.x.to_numpy()
    
    def get_velocities(self) -> np.ndarray:
        """Get particle velocities as numpy array"""
        return self.v.to_numpy()
    
    def get_colors(self) -> np.ndarray:
        """Get particle colors as numpy array"""
        return self.color.to_numpy()


class TorchMPMSimulator:
    """
    Pure PyTorch MPM implementation for DDP compatibility
    Less efficient than Taichi but works with distributed training
    """
    
    def __init__(
        self,
        particles: ParticleSystem,
        grid_resolution: int = 128,
        domain_min: Tuple[float, float, float] = (-2.0, -2.0, -2.0),
        domain_max: Tuple[float, float, float] = (2.0, 2.0, 2.0),
        material: Optional[MaterialProperties] = None,
        dt: float = 1e-4,
        gravity: Tuple[float, float, float] = (0.0, -0.5, 0.0),
    ):
        self.particles = particles
        self.grid_res = grid_resolution
        self.domain_min = torch.tensor(domain_min, device=particles.device)
        self.domain_max = torch.tensor(domain_max, device=particles.device)
        self.dt = dt
        self.gravity = torch.tensor(gravity, device=particles.device)
        
        self.material = material or MaterialProperties()
        
        domain_size = self.domain_max - self.domain_min
        self.dx = domain_size.max() / grid_resolution
        self.inv_dx = 1.0 / self.dx
        
        self.p_vol = (self.dx * 0.5) ** 3
        self.p_mass = self.material.density * self.p_vol
    
    def step(self, external_force: Optional[torch.Tensor] = None):
        """
        Simplified MPM step in PyTorch
        For full accuracy, use Taichi version
        """
        # Apply gravity
        self.particles.velocities += self.dt * self.gravity
        
        # Apply external force if provided
        if external_force is not None:
            # Interpolate force at particle positions
            force = self._interpolate_force(external_force)
            self.particles.velocities += self.dt * force / self.p_mass
        
        # Update positions
        self.particles.positions += self.dt * self.particles.velocities
        
        # Apply boundary conditions
        self._apply_boundaries()
    
    def _interpolate_force(self, force_field: torch.Tensor) -> torch.Tensor:
        """Interpolate force field at particle positions"""
        # Normalize positions to grid coordinates
        pos = self.particles.positions
        grid_pos = (pos - self.domain_min) / (self.domain_max - self.domain_min)
        grid_pos = grid_pos * 2 - 1  # [-1, 1] for grid_sample
        
        # Reshape for grid_sample
        grid_pos = grid_pos.view(1, 1, 1, -1, 3)
        force_field = force_field.unsqueeze(0)  # (1, 3, D, H, W)
        
        # Sample force at particle positions
        force = torch.nn.functional.grid_sample(
            force_field, grid_pos,
            mode='bilinear', padding_mode='zeros', align_corners=True
        )
        
        return force.view(3, -1).T  # (N, 3)
    
    def _apply_boundaries(self):
        """Apply boundary conditions"""
        pos = self.particles.positions
        vel = self.particles.velocities
        
        # Clamp positions
        margin = 0.05
        pos_min = self.domain_min + margin
        pos_max = self.domain_max - margin
        
        # Apply boundary reflection
        for d in range(3):
            below_min = pos[:, d] < pos_min[d]
            above_max = pos[:, d] > pos_max[d]
            
            pos[:, d] = torch.where(below_min, pos_min[d], pos[:, d])
            pos[:, d] = torch.where(above_max, pos_max[d], pos[:, d])
            
            vel[:, d] = torch.where(below_min, torch.abs(vel[:, d]) * 0.5, vel[:, d])
            vel[:, d] = torch.where(above_max, -torch.abs(vel[:, d]) * 0.5, vel[:, d])
