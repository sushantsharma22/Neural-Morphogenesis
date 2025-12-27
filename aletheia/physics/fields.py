"""
Neural Field Solver for Morphogenesis
Bridges neural network predictions with particle simulation
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Tuple, Optional, Dict
from dataclasses import dataclass


@dataclass
class GridField:
    """3D grid field data structure"""
    data: torch.Tensor  # (C, D, H, W) or (B, C, D, H, W)
    resolution: Tuple[int, int, int]
    domain_min: Tuple[float, float, float]
    domain_max: Tuple[float, float, float]
    
    @property
    def dx(self) -> float:
        """Grid spacing"""
        domain_size = max(
            self.domain_max[i] - self.domain_min[i]
            for i in range(3)
        )
        return domain_size / max(self.resolution)
    
    def sample_at(
        self,
        positions: torch.Tensor,  # (N, 3)
    ) -> torch.Tensor:
        """Sample field values at arbitrary positions"""
        # Normalize positions to [-1, 1]
        domain_min = torch.tensor(self.domain_min, device=positions.device)
        domain_max = torch.tensor(self.domain_max, device=positions.device)
        
        normalized = (positions - domain_min) / (domain_max - domain_min)
        normalized = normalized * 2 - 1  # [-1, 1]
        
        # Reshape for grid_sample: (1, 1, 1, N, 3)
        grid_coords = normalized.view(1, 1, 1, -1, 3)
        
        # Sample
        data = self.data
        if data.dim() == 4:
            data = data.unsqueeze(0)
        
        sampled = F.grid_sample(
            data, grid_coords,
            mode='bilinear', padding_mode='border', align_corners=True
        )
        
        return sampled.view(data.shape[1], -1).T  # (N, C)
    
    def to_numpy(self) -> np.ndarray:
        """Export to numpy"""
        return self.data.cpu().numpy()


class NeuralFieldSolver:
    """
    Solves for velocity field using neural network
    Manages the interface between NN and physics simulation
    """
    
    def __init__(
        self,
        model: nn.Module,
        grid_resolution: int = 64,
        domain_min: Tuple[float, float, float] = (-2.0, -2.0, -2.0),
        domain_max: Tuple[float, float, float] = (2.0, 2.0, 2.0),
        device: torch.device = torch.device("cuda"),
    ):
        self.model = model
        self.grid_resolution = grid_resolution
        self.domain_min = domain_min
        self.domain_max = domain_max
        self.device = device
        
        # Create coordinate grid
        self._create_coordinate_grid()
    
    def _create_coordinate_grid(self):
        """Create 3D coordinate grid for neural network input"""
        res = self.grid_resolution
        
        # Create normalized coordinates
        x = torch.linspace(self.domain_min[0], self.domain_max[0], res, device=self.device)
        y = torch.linspace(self.domain_min[1], self.domain_max[1], res, device=self.device)
        z = torch.linspace(self.domain_min[2], self.domain_max[2], res, device=self.device)
        
        # Create meshgrid
        xx, yy, zz = torch.meshgrid(x, y, z, indexing='ij')
        
        # Stack coordinates: (3, D, H, W)
        self.coord_grid = torch.stack([xx, yy, zz], dim=0)
    
    def compute_density_field(
        self,
        positions: torch.Tensor,  # (N, 3)
        kernel_size: float = 0.1,
    ) -> torch.Tensor:
        """Compute density field from particle positions using SPH kernel"""
        res = self.grid_resolution
        density = torch.zeros(res, res, res, device=self.device)
        
        # Normalize positions to grid indices
        domain_min = torch.tensor(self.domain_min, device=self.device)
        domain_max = torch.tensor(self.domain_max, device=self.device)
        domain_size = domain_max - domain_min
        
        grid_pos = (positions - domain_min) / domain_size * res
        grid_pos = grid_pos.long().clamp(0, res - 1)
        
        # Simple histogram-based density
        for i in range(positions.shape[0]):
            xi, yi, zi = grid_pos[i]
            density[xi, yi, zi] += 1.0
        
        # Smooth with Gaussian kernel
        density = density.unsqueeze(0).unsqueeze(0)
        density = F.avg_pool3d(
            F.pad(density, (2, 2, 2, 2, 2, 2), mode='replicate'),
            kernel_size=5, stride=1
        )
        
        return density.squeeze()
    
    def prepare_input(
        self,
        positions: torch.Tensor,
        velocities: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Prepare input tensor for neural network
        
        Args:
            positions: Particle positions (N, 3)
            velocities: Particle velocities (N, 3), optional
            
        Returns:
            Input tensor (1, C, D, H, W) where C=4 (x,y,z,density)
        """
        # Compute density field
        density = self.compute_density_field(positions)
        
        # Normalize coordinates
        coords = self.coord_grid  # (3, D, H, W)
        
        # Stack with density: (4, D, H, W)
        input_field = torch.cat([
            coords,
            density.unsqueeze(0)
        ], dim=0)
        
        return input_field.unsqueeze(0)  # (1, 4, D, H, W)
    
    @torch.no_grad()
    def solve(
        self,
        positions: torch.Tensor,
        latent: torch.Tensor,
        time_step: Optional[torch.Tensor] = None,
    ) -> GridField:
        """
        Solve for velocity field using neural network
        
        Args:
            positions: Particle positions (N, 3)
            latent: Latent DNA vector (1, latent_dim)
            time_step: Normalized time [0, 1]
            
        Returns:
            GridField containing velocity field
        """
        # Prepare input
        input_field = self.prepare_input(positions)
        
        # Run neural network
        self.model.eval()
        
        if time_step is not None:
            velocity_field = self.model(input_field, latent, time_step)
        else:
            velocity_field = self.model(input_field, latent)
        
        # Create GridField
        res_tuple: Tuple[int, int, int] = (self.grid_resolution, self.grid_resolution, self.grid_resolution)
        return GridField(
            data=velocity_field.squeeze(0),  # (3, D, H, W)
            resolution=res_tuple,
            domain_min=self.domain_min,
            domain_max=self.domain_max,
        )
    
    def apply_to_particles(
        self,
        velocity_field: GridField,
        positions: torch.Tensor,
        velocities: torch.Tensor,
        dt: float,
        blend_factor: float = 0.5,
    ) -> torch.Tensor:
        """
        Apply velocity field to particles
        
        Args:
            velocity_field: Predicted velocity field
            positions: Particle positions (N, 3)
            velocities: Current particle velocities (N, 3)
            dt: Time step
            blend_factor: Blend between current and predicted velocity
            
        Returns:
            Updated velocities (N, 3)
        """
        # Sample velocity at particle positions
        predicted_vel = velocity_field.sample_at(positions)
        
        # Blend current and predicted velocities
        new_velocities = (1 - blend_factor) * velocities + blend_factor * predicted_vel
        
        return new_velocities


class ParticleToGridConverter:
    """
    Converts particle data to grid representation for neural network
    Supports multiple interpolation schemes
    """
    
    def __init__(
        self,
        grid_resolution: int = 64,
        domain_min: Tuple[float, float, float] = (-2.0, -2.0, -2.0),
        domain_max: Tuple[float, float, float] = (2.0, 2.0, 2.0),
        kernel_type: str = "gaussian",
        kernel_radius: float = 0.1,
        device: torch.device = torch.device("cuda"),
    ):
        self.grid_resolution = grid_resolution
        self.domain_min = torch.tensor(domain_min, device=device)
        self.domain_max = torch.tensor(domain_max, device=device)
        self.kernel_type = kernel_type
        self.kernel_radius = kernel_radius
        self.device = device
        
        self.dx = (self.domain_max - self.domain_min).max() / grid_resolution
    
    def scatter_particles(
        self,
        positions: torch.Tensor,  # (N, 3)
        values: torch.Tensor,     # (N, C)
        method: str = "nearest",
    ) -> torch.Tensor:
        """
        Scatter particle values to grid
        
        Args:
            positions: Particle positions (N, 3)
            values: Particle values to scatter (N, C)
            method: Interpolation method ('nearest', 'trilinear', 'sph')
            
        Returns:
            Grid tensor (C, D, H, W)
        """
        res = self.grid_resolution
        C = values.shape[1] if values.dim() > 1 else 1
        
        if method == "nearest":
            return self._scatter_nearest(positions, values, C, res)
        elif method == "trilinear":
            return self._scatter_trilinear(positions, values, C, res)
        elif method == "sph":
            return self._scatter_sph(positions, values, C, res)
        else:
            raise ValueError(f"Unknown scatter method: {method}")
    
    def _scatter_nearest(
        self,
        positions: torch.Tensor,
        values: torch.Tensor,
        C: int,
        res: int,
    ) -> torch.Tensor:
        """Nearest neighbor scattering"""
        grid = torch.zeros(C, res, res, res, device=self.device)
        counts = torch.zeros(res, res, res, device=self.device)
        
        # Normalize to grid indices
        normalized = (positions - self.domain_min) / (self.domain_max - self.domain_min)
        indices = (normalized * res).long().clamp(0, res - 1)
        
        # Scatter
        if values.dim() == 1:
            values = values.unsqueeze(1)
        
        for i in range(positions.shape[0]):
            xi, yi, zi = indices[i]
            grid[:, xi, yi, zi] += values[i]
            counts[xi, yi, zi] += 1
        
        # Normalize by counts
        mask = counts > 0
        for c in range(C):
            grid[c][mask] /= counts[mask]
        
        return grid
    
    def _scatter_trilinear(
        self,
        positions: torch.Tensor,
        values: torch.Tensor,
        C: int,
        res: int,
    ) -> torch.Tensor:
        """Trilinear interpolation scattering"""
        grid = torch.zeros(C, res, res, res, device=self.device)
        weights = torch.zeros(res, res, res, device=self.device)
        
        # Normalize to continuous grid coordinates
        normalized = (positions - self.domain_min) / (self.domain_max - self.domain_min)
        continuous = normalized * (res - 1)
        
        # Get base indices and fractional parts
        base = continuous.floor().long().clamp(0, res - 2)
        frac = continuous - base.float()
        
        if values.dim() == 1:
            values = values.unsqueeze(1)
        
        # Trilinear weights
        for di in range(2):
            for dj in range(2):
                for dk in range(2):
                    w = ((1 - di) * (1 - frac[:, 0]) + di * frac[:, 0]) * \
                        ((1 - dj) * (1 - frac[:, 1]) + dj * frac[:, 1]) * \
                        ((1 - dk) * (1 - frac[:, 2]) + dk * frac[:, 2])
                    
                    idx = base + torch.tensor([di, dj, dk], device=self.device)
                    idx = idx.clamp(0, res - 1)
                    
                    for i in range(positions.shape[0]):
                        xi, yi, zi = idx[i]
                        grid[:, xi, yi, zi] += w[i] * values[i]
                        weights[xi, yi, zi] += w[i]
        
        # Normalize
        mask = weights > 0
        for c in range(C):
            grid[c][mask] /= weights[mask]
        
        return grid
    
    def _scatter_sph(
        self,
        positions: torch.Tensor,
        values: torch.Tensor,
        C: int,
        res: int,
    ) -> torch.Tensor:
        """SPH kernel scattering"""
        # For efficiency, fall back to nearest + smoothing
        grid = self._scatter_nearest(positions, values, C, res)
        
        # Apply Gaussian smoothing
        kernel_size = max(3, int(self.kernel_radius / self.dx.item()) * 2 + 1)
        if kernel_size % 2 == 0:
            kernel_size += 1
        
        padding = kernel_size // 2
        grid = grid.unsqueeze(0)
        grid = F.avg_pool3d(
            F.pad(grid, (padding,) * 6, mode='replicate'),
            kernel_size=kernel_size, stride=1
        )
        
        return grid.squeeze(0)


class VelocityFieldIntegrator:
    """
    Integrates velocity field to update particle positions
    Supports multiple integration schemes
    """
    
    def __init__(
        self,
        method: str = "rk4",
    ):
        self.method = method
    
    def integrate(
        self,
        positions: torch.Tensor,
        velocities: torch.Tensor,
        velocity_field: GridField,
        dt: float,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Integrate velocity field to update particles
        
        Returns:
            Updated (positions, velocities)
        """
        if self.method == "euler":
            return self._euler(positions, velocities, velocity_field, dt)
        elif self.method == "rk2":
            return self._rk2(positions, velocities, velocity_field, dt)
        elif self.method == "rk4":
            return self._rk4(positions, velocities, velocity_field, dt)
        else:
            return self._euler(positions, velocities, velocity_field, dt)
    
    def _euler(
        self,
        positions: torch.Tensor,
        velocities: torch.Tensor,
        velocity_field: GridField,
        dt: float,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward Euler integration"""
        v = velocity_field.sample_at(positions)
        new_positions = positions + dt * v
        new_velocities = v
        return new_positions, new_velocities
    
    def _rk2(
        self,
        positions: torch.Tensor,
        velocities: torch.Tensor,
        velocity_field: GridField,
        dt: float,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """2nd order Runge-Kutta integration"""
        k1 = velocity_field.sample_at(positions)
        k2 = velocity_field.sample_at(positions + 0.5 * dt * k1)
        
        new_positions = positions + dt * k2
        new_velocities = k2
        return new_positions, new_velocities
    
    def _rk4(
        self,
        positions: torch.Tensor,
        velocities: torch.Tensor,
        velocity_field: GridField,
        dt: float,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """4th order Runge-Kutta integration"""
        k1 = velocity_field.sample_at(positions)
        k2 = velocity_field.sample_at(positions + 0.5 * dt * k1)
        k3 = velocity_field.sample_at(positions + 0.5 * dt * k2)
        k4 = velocity_field.sample_at(positions + dt * k3)
        
        v_avg = (k1 + 2*k2 + 2*k3 + k4) / 6
        new_positions = positions + dt * v_avg
        new_velocities = v_avg
        return new_positions, new_velocities
