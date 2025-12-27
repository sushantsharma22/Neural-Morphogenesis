"""
Latent DNA System for Neural Morphogenesis
Maps latent seed vectors to physics perturbations
The DNA defines the "genetic code" that guides structure formation
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional, Dict, List, Any
import numpy as np
import json
from pathlib import Path


class LatentDNA:
    """
    Manages the latent DNA vector that defines morphogenesis behavior
    Acts as the "Universal Law" seed that perturbs PDEs
    """
    
    def __init__(
        self,
        dim: int = 512,
        seed_type: str = "gaussian",
        seed: Optional[int] = None,
        device: torch.device = torch.device("cuda"),
    ):
        self.dim = dim
        self.seed_type = seed_type
        self.device = device
        
        if seed is not None:
            torch.manual_seed(seed)
            np.random.seed(seed)
        
        self.latent_vector = self._generate_latent()
        self.generation_seed = seed
        self.metadata: Dict[str, Any] = {}
    
    def _generate_latent(self) -> torch.Tensor:
        """Generate latent vector based on seed type"""
        if self.seed_type == "gaussian":
            return torch.randn(1, self.dim, device=self.device)
        
        elif self.seed_type == "uniform":
            return torch.rand(1, self.dim, device=self.device) * 2 - 1
        
        elif self.seed_type == "structured":
            # Structured latent with frequency bands
            latent = torch.zeros(1, self.dim, device=self.device)
            
            # Low frequency components (global structure)
            latent[:, :self.dim // 4] = torch.randn(1, self.dim // 4, device=self.device) * 2.0
            
            # Mid frequency (local structure)
            latent[:, self.dim // 4:self.dim // 2] = torch.randn(1, self.dim // 4, device=self.device) * 1.0
            
            # High frequency (fine details)
            latent[:, self.dim // 2:] = torch.randn(1, self.dim // 2, device=self.device) * 0.5
            
            return latent
        
        else:
            return torch.randn(1, self.dim, device=self.device)
    
    @property
    def vector(self) -> torch.Tensor:
        """Get the latent vector"""
        return self.latent_vector
    
    def interpolate(
        self,
        other: 'LatentDNA',
        alpha: float,
    ) -> torch.Tensor:
        """Spherical linear interpolation between two DNA vectors"""
        v1 = F.normalize(self.latent_vector, dim=-1)
        v2 = F.normalize(other.latent_vector, dim=-1)
        
        dot = torch.sum(v1 * v2, dim=-1, keepdim=True)
        omega = torch.acos(torch.clamp(dot, -1, 1))
        
        sin_omega = torch.sin(omega)
        
        # Handle small angles
        if sin_omega.abs() < 1e-6:
            return (1 - alpha) * self.latent_vector + alpha * other.latent_vector
        
        s1 = torch.sin((1 - alpha) * omega) / sin_omega
        s2 = torch.sin(alpha * omega) / sin_omega
        
        return s1 * self.latent_vector + s2 * other.latent_vector
    
    def mutate(
        self,
        mutation_strength: float = 0.1,
        preserve_norm: bool = True,
    ) -> 'LatentDNA':
        """Create mutated version of DNA"""
        noise = torch.randn_like(self.latent_vector) * mutation_strength
        new_latent = self.latent_vector + noise
        
        if preserve_norm:
            original_norm = torch.norm(self.latent_vector, dim=-1, keepdim=True)
            new_latent = F.normalize(new_latent, dim=-1) * original_norm
        
        new_dna = LatentDNA(self.dim, self.seed_type, device=self.device)
        new_dna.latent_vector = new_latent
        return new_dna
    
    def crossover(
        self,
        other: 'LatentDNA',
        crossover_point: Optional[int] = None,
    ) -> 'LatentDNA':
        """Genetic crossover between two DNA vectors"""
        if crossover_point is None:
            crossover_point = self.dim // 2
        
        new_latent = torch.cat([
            self.latent_vector[:, :crossover_point],
            other.latent_vector[:, crossover_point:],
        ], dim=-1)
        
        new_dna = LatentDNA(self.dim, self.seed_type, device=self.device)
        new_dna.latent_vector = new_latent
        return new_dna
    
    def to_dict(self) -> Dict[str, Any]:
        """Serialize DNA to dictionary"""
        return {
            "dim": self.dim,
            "seed_type": self.seed_type,
            "generation_seed": self.generation_seed,
            "vector": self.latent_vector.cpu().numpy().tolist(),
            "metadata": self.metadata,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any], device: torch.device) -> 'LatentDNA':
        """Deserialize DNA from dictionary"""
        dna = cls(
            dim=data["dim"],
            seed_type=data["seed_type"],
            seed=data.get("generation_seed"),
            device=device,
        )
        dna.latent_vector = torch.tensor(data["vector"], device=device)
        dna.metadata = data.get("metadata", {})
        return dna
    
    def save(self, path: Path):
        """Save DNA to JSON file"""
        with open(path, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)
    
    @classmethod
    def load(cls, path: Path, device: torch.device) -> 'LatentDNA':
        """Load DNA from JSON file"""
        with open(path, 'r') as f:
            data = json.load(f)
        return cls.from_dict(data, device)


class DNAEncoder(nn.Module):
    """
    Encodes structure properties into latent DNA space
    Used for learning from observed structures
    """
    
    def __init__(
        self,
        input_dim: int = 256,  # e.g., from point cloud encoder
        latent_dim: int = 512,
        hidden_dims: Tuple[int, ...] = (512, 512, 512),
    ):
        super().__init__()
        
        layers = []
        in_dim = input_dim
        
        for h_dim in hidden_dims:
            layers.extend([
                nn.Linear(in_dim, h_dim),
                nn.LayerNorm(h_dim),
                nn.GELU(),
                nn.Dropout(0.1),
            ])
            in_dim = h_dim
        
        self.encoder = nn.Sequential(*layers)
        
        # Output mean and log variance for VAE-style encoding
        self.fc_mu = nn.Linear(hidden_dims[-1], latent_dim)
        self.fc_logvar = nn.Linear(hidden_dims[-1], latent_dim)
    
    def forward(
        self,
        x: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Encode input to latent DNA parameters
        Returns mean and log variance for reparameterization
        """
        h = self.encoder(x)
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        return mu, logvar
    
    def reparameterize(
        self,
        mu: torch.Tensor,
        logvar: torch.Tensor,
    ) -> torch.Tensor:
        """Reparameterization trick for VAE training"""
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std


class DNADecoder(nn.Module):
    """
    Decodes latent DNA to physics parameters and structure templates
    Maps DNA to concrete perturbation parameters
    """
    
    def __init__(
        self,
        latent_dim: int = 512,
        physics_params: int = 64,    # Number of physics parameters
        spatial_freqs: int = 32,     # Spatial frequency coefficients
        hidden_dims: Tuple[int, ...] = (512, 512),
    ):
        super().__init__()
        
        self.latent_dim = latent_dim
        self.physics_params = physics_params
        self.spatial_freqs = spatial_freqs
        
        layers = []
        in_dim = latent_dim
        
        for h_dim in hidden_dims:
            layers.extend([
                nn.Linear(in_dim, h_dim),
                nn.LayerNorm(h_dim),
                nn.GELU(),
            ])
            in_dim = h_dim
        
        self.decoder = nn.Sequential(*layers)
        
        # Physics parameter heads
        self.physics_head = nn.Sequential(
            nn.Linear(hidden_dims[-1], physics_params),
            nn.Tanh(),  # Bounded physics perturbations
        )
        
        # Spatial frequency head for Fourier-based perturbations
        self.freq_head = nn.Sequential(
            nn.Linear(hidden_dims[-1], spatial_freqs * 3),  # x, y, z frequencies
        )
        
        # Attractor strength head
        self.attractor_head = nn.Sequential(
            nn.Linear(hidden_dims[-1], 8),  # 8 attractor points
            nn.Sigmoid(),
        )
    
    def forward(
        self,
        latent: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """
        Decode latent DNA to physics perturbation parameters
        
        Returns dict with:
            - physics_params: (batch, physics_params) - PDE coefficients
            - spatial_freqs: (batch, spatial_freqs, 3) - Fourier frequencies
            - attractor_strengths: (batch, 8) - Attractor point strengths
        """
        h = self.decoder(latent)
        
        physics = self.physics_head(h)
        freqs = self.freq_head(h).view(-1, self.spatial_freqs, 3)
        attractors = self.attractor_head(h)
        
        return {
            "physics_params": physics,
            "spatial_freqs": freqs,
            "attractor_strengths": attractors,
        }


class DNAPerturbationField(nn.Module):
    """
    Generates spatial perturbation field from DNA
    Creates the force field that guides particle organization
    """
    
    def __init__(
        self,
        latent_dim: int = 512,
        field_resolution: int = 64,
        num_frequencies: int = 8,
    ):
        super().__init__()
        
        self.latent_dim = latent_dim
        self.field_resolution = field_resolution
        self.num_frequencies = num_frequencies
        
        # DNA to frequency coefficients
        self.freq_net = nn.Sequential(
            nn.Linear(latent_dim, 256),
            nn.GELU(),
            nn.Linear(256, num_frequencies * 6),  # amplitude + phase for x, y, z
        )
        
        # DNA to attractor positions
        self.attractor_net = nn.Sequential(
            nn.Linear(latent_dim, 256),
            nn.GELU(),
            nn.Linear(256, 8 * 4),  # 8 attractors with (x, y, z, strength)
        )
        
        # Create coordinate grid
        coords = torch.linspace(-1, 1, field_resolution)
        self.register_buffer(
            'grid',
            torch.stack(torch.meshgrid(coords, coords, coords, indexing='ij'), dim=-1)
        )
    
    def forward(
        self,
        latent: torch.Tensor,
    ) -> torch.Tensor:
        """
        Generate 3D perturbation field from DNA
        
        Args:
            latent: (batch, latent_dim)
            
        Returns:
            field: (batch, 3, D, H, W) - 3D vector field
        """
        batch_size = latent.shape[0]
        
        # Get frequency components
        freq_params = self.freq_net(latent)
        freq_params = freq_params.view(batch_size, self.num_frequencies, 6)
        
        # Get attractor parameters
        attractor_params = self.attractor_net(latent)
        attractor_params = attractor_params.view(batch_size, 8, 4)
        
        # Build field from Fourier components
        field = self._build_fourier_field(freq_params)
        
        # Add attractor forces
        attractor_field = self._build_attractor_field(attractor_params)
        
        field = field + attractor_field
        
        return field
    
    def _build_fourier_field(
        self,
        freq_params: torch.Tensor,
    ) -> torch.Tensor:
        """Build field from Fourier components"""
        batch_size = freq_params.shape[0]
        res = self.field_resolution
        
        field = torch.zeros(batch_size, 3, res, res, res, device=freq_params.device)
        
        grid = self.grid.unsqueeze(0).expand(batch_size, -1, -1, -1, -1)  # type: ignore[attr-defined]
        
        for i in range(self.num_frequencies):
            freq = (i + 1) * math.pi
            
            amp_x = freq_params[:, i, 0:1].unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
            amp_y = freq_params[:, i, 1:2].unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
            amp_z = freq_params[:, i, 2:3].unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
            
            phase_x = freq_params[:, i, 3:4].unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
            phase_y = freq_params[:, i, 4:5].unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
            phase_z = freq_params[:, i, 5:6].unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
            
            x = grid[..., 0]
            y = grid[..., 1]
            z = grid[..., 2]
            
            field[:, 0] += (amp_x * torch.sin(freq * x + phase_x)).squeeze(-1)
            field[:, 1] += (amp_y * torch.sin(freq * y + phase_y)).squeeze(-1)
            field[:, 2] += (amp_z * torch.sin(freq * z + phase_z)).squeeze(-1)
        
        return field
    
    def _build_attractor_field(
        self,
        attractor_params: torch.Tensor,
    ) -> torch.Tensor:
        """Build field from point attractors"""
        batch_size = attractor_params.shape[0]
        res = self.field_resolution
        
        field = torch.zeros(batch_size, 3, res, res, res, device=attractor_params.device)
        
        grid = self.grid.unsqueeze(0).expand(batch_size, -1, -1, -1, -1)  # type: ignore[attr-defined]  # (B, D, H, W, 3)
        
        for i in range(8):
            pos = attractor_params[:, i, :3]  # (B, 3)
            strength = attractor_params[:, i, 3:4]  # (B, 1)
            
            # Compute direction to attractor
            pos = pos.view(batch_size, 1, 1, 1, 3)
            diff = pos - grid  # (B, D, H, W, 3)
            
            dist = torch.norm(diff, dim=-1, keepdim=True) + 1e-6
            direction = diff / dist
            
            # Inverse square law with cutoff
            force = strength.view(batch_size, 1, 1, 1, 1) / (dist ** 2 + 0.1)
            
            field += (force * direction).permute(0, 4, 1, 2, 3)
        
        return field


class MorphogenesisScheduler:
    """
    Schedules DNA influence over simulation time
    Controls the phases of morphogenesis (chaos -> emergence -> stabilization)
    """
    
    def __init__(
        self,
        total_frames: int = 1000,
        chaos_end: int = 100,
        emergence_end: int = 400,
        refinement_end: int = 700,
    ):
        self.total_frames = total_frames
        self.chaos_end = chaos_end
        self.emergence_end = emergence_end
        self.refinement_end = refinement_end
    
    def get_phase(self, frame: int) -> str:
        """Get current morphogenesis phase"""
        if frame < self.chaos_end:
            return "chaos"
        elif frame < self.emergence_end:
            return "emergence"
        elif frame < self.refinement_end:
            return "refinement"
        else:
            return "stabilization"
    
    def get_dna_influence(self, frame: int) -> float:
        """
        Get DNA influence strength for current frame
        Low during chaos, increases through emergence, stable in refinement
        """
        if frame < self.chaos_end:
            # Chaos phase: DNA influence grows from 0 to 0.2
            t = frame / self.chaos_end
            return 0.2 * t
        
        elif frame < self.emergence_end:
            # Emergence phase: DNA influence grows from 0.2 to 1.0
            t = (frame - self.chaos_end) / (self.emergence_end - self.chaos_end)
            return 0.2 + 0.8 * self._smooth_step(t)
        
        elif frame < self.refinement_end:
            # Refinement phase: Full DNA influence with slight variations
            t = (frame - self.emergence_end) / (self.refinement_end - self.emergence_end)
            return 1.0 - 0.1 * math.sin(t * math.pi)
        
        else:
            # Stabilization phase: DNA influence decreases slightly
            t = (frame - self.refinement_end) / (self.total_frames - self.refinement_end)
            return 0.9 + 0.1 * math.cos(t * math.pi)
    
    def get_noise_strength(self, frame: int) -> float:
        """
        Get noise strength for current frame
        High during chaos, decreases over time
        """
        if frame < self.chaos_end:
            return 1.0
        elif frame < self.emergence_end:
            t = (frame - self.chaos_end) / (self.emergence_end - self.chaos_end)
            return 1.0 - 0.8 * self._smooth_step(t)
        else:
            return 0.2
    
    def _smooth_step(self, t: float) -> float:
        """Smooth interpolation function"""
        return t * t * (3 - 2 * t)
    
    def get_schedule_params(self, frame: int) -> Dict[str, Any]:
        """Get all schedule parameters for current frame"""
        return {
            "phase": self.get_phase(frame),
            "dna_influence": self.get_dna_influence(frame),
            "noise_strength": self.get_noise_strength(frame),
            "progress": frame / self.total_frames,
        }
