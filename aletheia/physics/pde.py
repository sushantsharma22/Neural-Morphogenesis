"""
PDE Perturbation System for Neural Morphogenesis
Implements Navier-Stokes and Reaction-Diffusion equations
with neural perturbations from Latent DNA
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Tuple, Optional, Dict, List
import math


class PDEPerturbation:
    """
    Base class for PDE perturbation from latent DNA
    Maps latent vectors to PDE coefficient modifications
    """
    
    def __init__(
        self,
        latent_dim: int = 512,
        num_params: int = 32,
        device: torch.device = torch.device("cuda"),
    ):
        self.latent_dim = latent_dim
        self.num_params = num_params
        self.device = device
        
        # Linear mapping from latent to perturbation parameters
        self.projection = nn.Linear(latent_dim, num_params).to(device)
        nn.init.xavier_uniform_(self.projection.weight)
        nn.init.zeros_(self.projection.bias)
    
    def get_perturbation(
        self,
        latent: torch.Tensor,
        strength: float = 1.0,
    ) -> torch.Tensor:
        """
        Get perturbation parameters from latent
        
        Args:
            latent: Latent DNA vector (batch, latent_dim)
            strength: Perturbation strength multiplier
            
        Returns:
            Perturbation parameters (batch, num_params)
        """
        params = self.projection(latent)
        params = torch.tanh(params) * strength  # Bounded perturbations
        return params


class NavierStokesSolver:
    """
    3D Navier-Stokes solver with neural perturbation
    Implements incompressible fluid dynamics for particle guidance
    """
    
    def __init__(
        self,
        resolution: int = 64,
        domain_size: float = 4.0,
        viscosity: float = 0.01,
        pressure_iterations: int = 50,
        vorticity_confinement: float = 0.5,
        device: torch.device = torch.device("cuda"),
    ):
        self.resolution = resolution
        self.domain_size = domain_size
        self.dx = domain_size / resolution
        self.viscosity = viscosity
        self.pressure_iterations = pressure_iterations
        self.vorticity_confinement = vorticity_confinement
        self.device = device
        
        # Initialize velocity and pressure fields
        self._init_fields()
    
    def _init_fields(self):
        """Initialize fluid fields"""
        res = self.resolution
        
        # Velocity field (3, D, H, W)
        self.velocity = torch.zeros(3, res, res, res, device=self.device)
        
        # Pressure field (D, H, W)
        self.pressure = torch.zeros(res, res, res, device=self.device)
        
        # Divergence (D, H, W)
        self.divergence = torch.zeros(res, res, res, device=self.device)
        
        # External force (3, D, H, W)
        self.external_force = torch.zeros(3, res, res, res, device=self.device)
    
    def set_perturbation(
        self,
        perturbation: torch.Tensor,
        perturbation_type: str = "force",
    ):
        """
        Apply perturbation from latent DNA
        
        Args:
            perturbation: Perturbation parameters
            perturbation_type: Type of perturbation ('force', 'viscosity', 'vorticity')
        """
        if perturbation_type == "force":
            # Interpret as Fourier coefficients for force field
            self._apply_force_perturbation(perturbation)
        elif perturbation_type == "viscosity":
            # Modulate viscosity
            self.viscosity = 0.01 * (1 + perturbation[0].item() * 0.5)
        elif perturbation_type == "vorticity":
            self.vorticity_confinement = 0.5 * (1 + perturbation[0].item())
    
    def _apply_force_perturbation(self, perturbation: torch.Tensor):
        """Apply force field from perturbation parameters"""
        res = self.resolution
        
        # Create coordinate grids
        x = torch.linspace(-1, 1, res, device=self.device)
        xx, yy, zz = torch.meshgrid(x, x, x, indexing='ij')
        
        # Interpret perturbation as Fourier coefficients
        num_modes = min(8, perturbation.shape[-1] // 6)
        
        force = torch.zeros(3, res, res, res, device=self.device)
        
        for i in range(num_modes):
            freq = (i + 1) * math.pi
            idx = i * 6
            
            # Amplitudes and phases for each direction
            ax, ay, az = perturbation[..., idx:idx+3].squeeze()
            px, py, pz = perturbation[..., idx+3:idx+6].squeeze()
            
            force[0] += ax * torch.sin(freq * xx + px)
            force[1] += ay * torch.sin(freq * yy + py)
            force[2] += az * torch.sin(freq * zz + pz)
        
        self.external_force = force
    
    def advect(self, field: torch.Tensor, dt: float) -> torch.Tensor:
        """Semi-Lagrangian advection"""
        # Create coordinate grid
        res = self.resolution
        coords = torch.linspace(-1, 1, res, device=self.device)
        grid_z, grid_y, grid_x = torch.meshgrid(coords, coords, coords, indexing='ij')
        grid = torch.stack([grid_x, grid_y, grid_z], dim=-1)  # (D, H, W, 3)
        
        # Backtrace
        velocity_normalized = self.velocity * dt / self.domain_size * 2
        backtrace = grid - velocity_normalized.permute(1, 2, 3, 0)
        
        # Clamp to domain
        backtrace = backtrace.clamp(-1, 1)
        
        # Sample field at backtraced positions
        field_batch = field.unsqueeze(0)  # (1, C, D, H, W)
        backtrace_batch = backtrace.unsqueeze(0)  # (1, D, H, W, 3)
        
        advected = F.grid_sample(
            field_batch, backtrace_batch,
            mode='bilinear', padding_mode='border', align_corners=True
        )
        
        return advected.squeeze(0)
    
    def diffuse(self, field: torch.Tensor, dt: float) -> torch.Tensor:
        """Diffusion using Jacobi iteration"""
        alpha = self.dx ** 2 / (self.viscosity * dt)
        beta = 6 + alpha
        
        result = field.clone()
        
        for _ in range(20):  # Jacobi iterations
            # 3D Laplacian stencil
            padded = F.pad(result.unsqueeze(0), (1, 1, 1, 1, 1, 1), mode='replicate')
            
            laplacian = (
                padded[..., 2:, 1:-1, 1:-1] + padded[..., :-2, 1:-1, 1:-1] +  # x neighbors
                padded[..., 1:-1, 2:, 1:-1] + padded[..., 1:-1, :-2, 1:-1] +  # y neighbors
                padded[..., 1:-1, 1:-1, 2:] + padded[..., 1:-1, 1:-1, :-2]    # z neighbors
            ).squeeze(0)
            
            result = (field * alpha + laplacian) / beta
        
        return result
    
    def compute_divergence(self):
        """Compute velocity divergence"""
        # Central differences
        padded = F.pad(self.velocity.unsqueeze(0), (1, 1, 1, 1, 1, 1), mode='replicate')
        
        du_dx = (padded[0, 0, 2:, 1:-1, 1:-1] - padded[0, 0, :-2, 1:-1, 1:-1]) / (2 * self.dx)
        dv_dy = (padded[0, 1, 1:-1, 2:, 1:-1] - padded[0, 1, 1:-1, :-2, 1:-1]) / (2 * self.dx)
        dw_dz = (padded[0, 2, 1:-1, 1:-1, 2:] - padded[0, 2, 1:-1, 1:-1, :-2]) / (2 * self.dx)
        
        self.divergence = du_dx + dv_dy + dw_dz
    
    def solve_pressure(self):
        """Solve pressure Poisson equation"""
        alpha = -self.dx ** 2
        beta = 6
        
        self.pressure.zero_()
        
        for _ in range(self.pressure_iterations):
            padded = F.pad(self.pressure.unsqueeze(0).unsqueeze(0), 
                          (1, 1, 1, 1, 1, 1), mode='replicate')
            
            neighbors = (
                padded[..., 2:, 1:-1, 1:-1] + padded[..., :-2, 1:-1, 1:-1] +
                padded[..., 1:-1, 2:, 1:-1] + padded[..., 1:-1, :-2, 1:-1] +
                padded[..., 1:-1, 1:-1, 2:] + padded[..., 1:-1, 1:-1, :-2]
            ).squeeze()
            
            self.pressure = (neighbors + alpha * self.divergence) / beta
    
    def project(self):
        """Project velocity to be divergence-free"""
        self.compute_divergence()
        self.solve_pressure()
        
        # Subtract pressure gradient
        padded = F.pad(self.pressure.unsqueeze(0).unsqueeze(0),
                      (1, 1, 1, 1, 1, 1), mode='replicate')
        
        dp_dx = (padded[..., 2:, 1:-1, 1:-1] - padded[..., :-2, 1:-1, 1:-1]) / (2 * self.dx)
        dp_dy = (padded[..., 1:-1, 2:, 1:-1] - padded[..., 1:-1, :-2, 1:-1]) / (2 * self.dx)
        dp_dz = (padded[..., 1:-1, 1:-1, 2:] - padded[..., 1:-1, 1:-1, :-2]) / (2 * self.dx)
        
        self.velocity[0] -= dp_dx.squeeze()
        self.velocity[1] -= dp_dy.squeeze()
        self.velocity[2] -= dp_dz.squeeze()
    
    def add_vorticity_confinement(self):
        """Add vorticity confinement for visual turbulence"""
        if self.vorticity_confinement <= 0:
            return
        
        # Compute vorticity (curl of velocity)
        padded = F.pad(self.velocity.unsqueeze(0), (1, 1, 1, 1, 1, 1), mode='replicate')
        
        # ω = ∇ × v
        dw_dy = (padded[0, 2, 1:-1, 2:, 1:-1] - padded[0, 2, 1:-1, :-2, 1:-1]) / (2 * self.dx)
        dv_dz = (padded[0, 1, 1:-1, 1:-1, 2:] - padded[0, 1, 1:-1, 1:-1, :-2]) / (2 * self.dx)
        du_dz = (padded[0, 0, 1:-1, 1:-1, 2:] - padded[0, 0, 1:-1, 1:-1, :-2]) / (2 * self.dx)
        dw_dx = (padded[0, 2, 2:, 1:-1, 1:-1] - padded[0, 2, :-2, 1:-1, 1:-1]) / (2 * self.dx)
        dv_dx = (padded[0, 1, 2:, 1:-1, 1:-1] - padded[0, 1, :-2, 1:-1, 1:-1]) / (2 * self.dx)
        du_dy = (padded[0, 0, 1:-1, 2:, 1:-1] - padded[0, 0, 1:-1, :-2, 1:-1]) / (2 * self.dx)
        
        omega_x = dw_dy - dv_dz
        omega_y = du_dz - dw_dx
        omega_z = dv_dx - du_dy
        
        omega_mag = torch.sqrt(omega_x**2 + omega_y**2 + omega_z**2 + 1e-6)
        
        # Compute gradient of vorticity magnitude
        omega_padded = F.pad(omega_mag.unsqueeze(0).unsqueeze(0), 
                            (1, 1, 1, 1, 1, 1), mode='replicate')
        
        eta_x = (omega_padded[..., 2:, 1:-1, 1:-1] - omega_padded[..., :-2, 1:-1, 1:-1]).squeeze()
        eta_y = (omega_padded[..., 1:-1, 2:, 1:-1] - omega_padded[..., 1:-1, :-2, 1:-1]).squeeze()
        eta_z = (omega_padded[..., 1:-1, 1:-1, 2:] - omega_padded[..., 1:-1, 1:-1, :-2]).squeeze()
        
        eta_mag = torch.sqrt(eta_x**2 + eta_y**2 + eta_z**2 + 1e-6)
        eta_x /= eta_mag
        eta_y /= eta_mag
        eta_z /= eta_mag
        
        # Vorticity confinement force
        force_x = self.vorticity_confinement * self.dx * (eta_y * omega_z - eta_z * omega_y)
        force_y = self.vorticity_confinement * self.dx * (eta_z * omega_x - eta_x * omega_z)
        force_z = self.vorticity_confinement * self.dx * (eta_x * omega_y - eta_y * omega_x)
        
        self.velocity[0] += force_x
        self.velocity[1] += force_y
        self.velocity[2] += force_z
    
    def step(self, dt: float):
        """Perform one Navier-Stokes step"""
        # Add external forces
        self.velocity += dt * self.external_force
        
        # Advection
        self.velocity = self.advect(self.velocity, dt)
        
        # Diffusion
        self.velocity = self.diffuse(self.velocity, dt)
        
        # Vorticity confinement
        self.add_vorticity_confinement()
        
        # Pressure projection
        self.project()
    
    def get_velocity_field(self) -> torch.Tensor:
        """Get current velocity field"""
        return self.velocity.clone()


class ReactionDiffusionSolver:
    """
    3D Reaction-Diffusion solver for Turing patterns
    Creates organic patterns that guide morphogenesis
    """
    
    def __init__(
        self,
        resolution: int = 64,
        diffusion_a: float = 1.0,
        diffusion_b: float = 0.5,
        feed_rate: float = 0.055,
        kill_rate: float = 0.062,
        device: torch.device = torch.device("cuda"),
    ):
        self.resolution = resolution
        self.Da = diffusion_a
        self.Db = diffusion_b
        self.f = feed_rate
        self.k = kill_rate
        self.device = device
        
        # Initialize concentration fields
        self._init_fields()
    
    def _init_fields(self):
        """Initialize concentration fields with random seed"""
        res = self.resolution
        
        # Chemical A (high concentration background)
        self.A = torch.ones(res, res, res, device=self.device)
        
        # Chemical B (localized seeds)
        self.B = torch.zeros(res, res, res, device=self.device)
        
        # Add random seeds for B
        center = res // 2
        seed_size = res // 8
        
        self.B[
            center - seed_size:center + seed_size,
            center - seed_size:center + seed_size,
            center - seed_size:center + seed_size
        ] = 1.0
        
        # Add noise
        self.B += torch.rand_like(self.B) * 0.1
    
    def set_parameters_from_latent(
        self,
        perturbation: torch.Tensor,
    ):
        """Set reaction-diffusion parameters from latent perturbation"""
        if perturbation.numel() >= 4:
            self.f = 0.055 + perturbation[0].item() * 0.02
            self.k = 0.062 + perturbation[1].item() * 0.02
            self.Da = 1.0 + perturbation[2].item() * 0.3
            self.Db = 0.5 + perturbation[3].item() * 0.2
    
    def laplacian_3d(self, field: torch.Tensor) -> torch.Tensor:
        """Compute 3D Laplacian using convolution"""
        # 3D Laplacian kernel (7-point stencil)
        kernel = torch.zeros(1, 1, 3, 3, 3, device=self.device)
        kernel[0, 0, 1, 1, 0] = 1.0  # -z
        kernel[0, 0, 1, 1, 2] = 1.0  # +z
        kernel[0, 0, 1, 0, 1] = 1.0  # -y
        kernel[0, 0, 1, 2, 1] = 1.0  # +y
        kernel[0, 0, 0, 1, 1] = 1.0  # -x
        kernel[0, 0, 2, 1, 1] = 1.0  # +x
        kernel[0, 0, 1, 1, 1] = -6.0  # center
        
        field_padded = F.pad(
            field.unsqueeze(0).unsqueeze(0),
            (1, 1, 1, 1, 1, 1),
            mode='circular'  # Periodic boundary
        )
        
        laplacian = F.conv3d(field_padded, kernel, padding=0)
        return laplacian.squeeze()
    
    def step(self, dt: float = 1.0):
        """Perform one reaction-diffusion step"""
        # Compute Laplacians
        lap_A = self.laplacian_3d(self.A)
        lap_B = self.laplacian_3d(self.B)
        
        # Gray-Scott reaction-diffusion
        AB2 = self.A * self.B * self.B
        
        dA = self.Da * lap_A - AB2 + self.f * (1 - self.A)
        dB = self.Db * lap_B + AB2 - (self.k + self.f) * self.B
        
        # Update
        self.A = self.A + dt * dA
        self.B = self.B + dt * dB
        
        # Clamp to [0, 1]
        self.A = self.A.clamp(0, 1)
        self.B = self.B.clamp(0, 1)
    
    def get_pattern_field(self) -> torch.Tensor:
        """Get pattern as 3D field"""
        # Combine A and B into gradient field
        # High B concentration creates attracting regions
        
        pattern = self.B - self.A * 0.5
        
        # Compute gradient
        padded = F.pad(pattern.unsqueeze(0).unsqueeze(0), 
                      (1, 1, 1, 1, 1, 1), mode='replicate')
        
        grad_x = (padded[..., 2:, 1:-1, 1:-1] - padded[..., :-2, 1:-1, 1:-1]).squeeze()
        grad_y = (padded[..., 1:-1, 2:, 1:-1] - padded[..., 1:-1, :-2, 1:-1]).squeeze()
        grad_z = (padded[..., 1:-1, 1:-1, 2:] - padded[..., 1:-1, 1:-1, :-2]).squeeze()
        
        return torch.stack([grad_x, grad_y, grad_z], dim=0)
    
    def get_concentration_B(self) -> torch.Tensor:
        """Get concentration of chemical B"""
        return self.B.clone()


class CoupledPDESolver:
    """
    Couples Navier-Stokes with Reaction-Diffusion
    Creates fluid-driven pattern formation
    """
    
    def __init__(
        self,
        resolution: int = 64,
        device: torch.device = torch.device("cuda"),
    ):
        self.resolution = resolution
        self.device = device
        
        # Initialize solvers
        self.ns = NavierStokesSolver(resolution=resolution, device=device)
        self.rd = ReactionDiffusionSolver(resolution=resolution, device=device)
        
        # Coupling strength
        self.coupling_ns_to_rd = 0.1  # Flow advects chemicals
        self.coupling_rd_to_ns = 0.5  # Patterns create forces
    
    def set_perturbation(
        self,
        perturbation: torch.Tensor,
    ):
        """Apply perturbation to both solvers"""
        # Split perturbation
        half = perturbation.shape[-1] // 2
        
        self.ns.set_perturbation(perturbation[..., :half], "force")
        self.rd.set_parameters_from_latent(perturbation[..., half:])
    
    def step(self, dt: float):
        """Coupled simulation step"""
        # Step reaction-diffusion with advection by fluid
        self.rd.step(dt)
        
        # Get pattern gradient as force for fluid
        pattern_force = self.rd.get_pattern_field()
        self.ns.external_force = pattern_force * self.coupling_rd_to_ns
        
        # Step fluid
        self.ns.step(dt)
        
        # Advect concentration by fluid velocity
        self._advect_concentration(dt)
    
    def _advect_concentration(self, dt: float):
        """Advect R-D concentrations by fluid velocity"""
        velocity = self.ns.get_velocity_field()
        
        # Simple semi-Lagrangian advection
        A_field = self.rd.A.unsqueeze(0)
        B_field = self.rd.B.unsqueeze(0)
        
        self.rd.A = self.ns.advect(A_field, dt * self.coupling_ns_to_rd).squeeze()
        self.rd.B = self.ns.advect(B_field, dt * self.coupling_ns_to_rd).squeeze()
    
    def get_combined_field(self) -> torch.Tensor:
        """Get combined velocity + pattern field"""
        velocity = self.ns.get_velocity_field()
        pattern = self.rd.get_pattern_field()
        
        # Combine (velocity dominates near patterns)
        pattern_strength = self.rd.get_concentration_B()
        pattern_strength = pattern_strength.unsqueeze(0).expand(3, -1, -1, -1)
        
        combined = velocity + pattern * pattern_strength
        return combined
