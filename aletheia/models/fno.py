"""
Fourier Neural Operator (FNO) for 3D Neural Morphogenesis
Predicts high-fidelity particle velocity fields from latent DNA
Optimized for A16 Ampere GPUs with DDP support
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint
from typing import Tuple, Optional, List
from einops import rearrange


class SpectralConv3d(nn.Module):
    """
    3D Spectral Convolution Layer (Fourier Layer)
    Performs convolution in Fourier space for global receptive field
    """
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        modes1: int,
        modes2: int,
        modes3: int,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes1 = modes1  # Number of Fourier modes in x
        self.modes2 = modes2  # Number of Fourier modes in y
        self.modes3 = modes3  # Number of Fourier modes in z
        
        self.scale = 1 / (in_channels * out_channels)
        
        # Complex weights for 8 octants of Fourier space
        self.weights1 = nn.Parameter(
            self.scale * torch.rand(in_channels, out_channels, modes1, modes2, modes3, 2)
        )
        self.weights2 = nn.Parameter(
            self.scale * torch.rand(in_channels, out_channels, modes1, modes2, modes3, 2)
        )
        self.weights3 = nn.Parameter(
            self.scale * torch.rand(in_channels, out_channels, modes1, modes2, modes3, 2)
        )
        self.weights4 = nn.Parameter(
            self.scale * torch.rand(in_channels, out_channels, modes1, modes2, modes3, 2)
        )
    
    def compl_mul3d(
        self,
        input: torch.Tensor,
        weights: torch.Tensor,
    ) -> torch.Tensor:
        """Complex multiplication in Fourier space"""
        # input: (batch, in_channel, x, y, z, 2)
        # weights: (in_channel, out_channel, x, y, z, 2)
        # output: (batch, out_channel, x, y, z, 2)
        
        return torch.einsum("bixyz,ioxyz->boxyz", 
                          torch.view_as_complex(input),
                          torch.view_as_complex(weights))
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through spectral convolution
        Args:
            x: Input tensor of shape (batch, channels, x, y, z)
        Returns:
            Output tensor of shape (batch, out_channels, x, y, z)
        """
        batch_size = x.shape[0]
        
        # Compute 3D FFT
        x_ft = torch.fft.rfftn(x, dim=[-3, -2, -1])
        
        # Allocate output tensor in Fourier space
        out_ft = torch.zeros(
            batch_size,
            self.out_channels,
            x.size(-3),
            x.size(-2),
            x.size(-1) // 2 + 1,
            dtype=torch.cfloat,
            device=x.device,
        )
        
        # Multiply relevant Fourier modes
        # Upper half of first octant
        out_ft[:, :, :self.modes1, :self.modes2, :self.modes3] = \
            self.compl_mul3d(
                torch.view_as_real(x_ft[:, :, :self.modes1, :self.modes2, :self.modes3]),
                self.weights1
            )
        
        # Lower half of first octant
        out_ft[:, :, -self.modes1:, :self.modes2, :self.modes3] = \
            self.compl_mul3d(
                torch.view_as_real(x_ft[:, :, -self.modes1:, :self.modes2, :self.modes3]),
                self.weights2
            )
        
        # Upper half of second octant
        out_ft[:, :, :self.modes1, -self.modes2:, :self.modes3] = \
            self.compl_mul3d(
                torch.view_as_real(x_ft[:, :, :self.modes1, -self.modes2:, :self.modes3]),
                self.weights3
            )
        
        # Lower half of second octant
        out_ft[:, :, -self.modes1:, -self.modes2:, :self.modes3] = \
            self.compl_mul3d(
                torch.view_as_real(x_ft[:, :, -self.modes1:, -self.modes2:, :self.modes3]),
                self.weights4
            )
        
        # Inverse FFT
        x = torch.fft.irfftn(out_ft, s=(x.size(-3), x.size(-2), x.size(-1)))
        
        return x


class FNOBlock3D(nn.Module):
    """
    Single FNO Block: Spectral Conv + Skip Connection + Activation
    """
    
    def __init__(
        self,
        width: int,
        modes: Tuple[int, int, int],
        activation: str = "gelu",
    ):
        super().__init__()
        
        self.spectral_conv = SpectralConv3d(
            width, width, modes[0], modes[1], modes[2]
        )
        self.conv = nn.Conv3d(width, width, 1)
        
        if activation == "gelu":
            self.activation = nn.GELU()
        elif activation == "relu":
            self.activation = nn.ReLU()
        elif activation == "silu":
            self.activation = nn.SiLU()
        else:
            self.activation = nn.GELU()
        
        self.norm = nn.GroupNorm(8, width)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through FNO block"""
        # Spectral pathway
        x1 = self.spectral_conv(x)
        
        # Skip pathway with 1x1 conv
        x2 = self.conv(x)
        
        # Combine and activate
        x = self.norm(x1 + x2)
        x = self.activation(x)
        
        return x


class LatentConditioner(nn.Module):
    """
    Conditions FNO computation on latent DNA vector
    Implements FiLM (Feature-wise Linear Modulation)
    """
    
    def __init__(
        self,
        latent_dim: int,
        feature_dim: int,
        num_layers: int = 2,
    ):
        super().__init__()
        
        self.mlp = nn.Sequential(
            nn.Linear(latent_dim, latent_dim * 2),
            nn.GELU(),
            nn.Linear(latent_dim * 2, feature_dim * 2 * num_layers),
        )
        
        self.num_layers = num_layers
        self.feature_dim = feature_dim
    
    def forward(self, latent: torch.Tensor) -> List[Tuple[torch.Tensor, torch.Tensor]]:
        """
        Generate scale and shift parameters for each layer
        Args:
            latent: Latent DNA vector (batch, latent_dim)
        Returns:
            List of (scale, shift) tuples for each layer
        """
        params = self.mlp(latent)
        params = params.view(-1, self.num_layers, self.feature_dim, 2)
        
        conditions = []
        for i in range(self.num_layers):
            scale = params[:, i, :, 0].unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
            shift = params[:, i, :, 1].unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
            conditions.append((scale, shift))
        
        return conditions


class FourierNeuralOperator3D(nn.Module):
    """
    3D Fourier Neural Operator for Neural Morphogenesis
    Predicts velocity field from particle positions and latent DNA
    
    Architecture:
        1. Lift input to higher dimension
        2. Apply N FNO blocks (spectral conv + skip)
        3. Project to output dimension
        4. Condition on latent DNA via FiLM
    """
    
    def __init__(
        self,
        in_channels: int = 4,        # (x, y, z, density)
        out_channels: int = 3,       # (vx, vy, vz)
        width: int = 64,
        modes: Tuple[int, int, int] = (16, 16, 16),
        num_layers: int = 4,
        latent_dim: int = 512,
        activation: str = "gelu",
        use_checkpoint: bool = True,
    ):
        super().__init__()
        
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.width = width
        self.modes = modes
        self.num_layers = num_layers
        self.latent_dim = latent_dim
        self.use_checkpoint = use_checkpoint
        
        # Lift layer
        self.lift = nn.Conv3d(in_channels, width, 1)
        
        # FNO blocks
        self.blocks = nn.ModuleList([
            FNOBlock3D(width, modes, activation)
            for _ in range(num_layers)
        ])
        
        # Projection layers
        self.project = nn.Sequential(
            nn.Conv3d(width, width * 2, 1),
            nn.GELU(),
            nn.Conv3d(width * 2, out_channels, 1),
        )
        
        # Latent conditioning
        self.conditioner = LatentConditioner(latent_dim, width, num_layers)
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize network weights"""
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
    
    def forward(
        self,
        x: torch.Tensor,
        latent: torch.Tensor,
        time_embedding: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Forward pass through FNO
        
        Args:
            x: Input field tensor (batch, in_channels, D, H, W)
            latent: Latent DNA vector (batch, latent_dim)
            time_embedding: Optional time embedding for temporal coherence
            
        Returns:
            Velocity field tensor (batch, out_channels, D, H, W)
        """
        # Get conditioning parameters
        conditions = self.conditioner(latent)
        
        # Lift to higher dimension
        x = self.lift(x)
        
        # Apply FNO blocks with conditioning
        for i, block in enumerate(self.blocks):
            if self.use_checkpoint and self.training:
                x = checkpoint(block, x, use_reentrant=False)  # type: ignore
            else:
                x = block(x)
            
            # Apply FiLM conditioning
            scale, shift = conditions[i]
            x = x * (1 + scale) + shift
        
        # Project to velocity field
        velocity = self.project(x)
        
        return velocity
    
    def get_memory_estimate(self, grid_size: int) -> float:
        """Estimate memory usage in GB for given grid size"""
        # Rough estimate: 4 bytes per float32
        params = sum(p.numel() for p in self.parameters())
        activations = self.width * (grid_size ** 3) * self.num_layers * 2
        
        total_bytes = (params + activations) * 4
        return total_bytes / 1024**3


class MultiscaleFNO3D(nn.Module):
    """
    Multi-scale FNO for capturing both fine and coarse dynamics
    Uses pyramid of FNOs at different resolutions
    """
    
    def __init__(
        self,
        in_channels: int = 4,
        out_channels: int = 3,
        base_width: int = 32,
        modes: Tuple[int, int, int] = (8, 8, 8),
        num_scales: int = 3,
        latent_dim: int = 512,
    ):
        super().__init__()
        
        self.num_scales = num_scales
        
        # FNO at each scale
        self.scale_fnos = nn.ModuleList([
            FourierNeuralOperator3D(
                in_channels=in_channels if i == 0 else out_channels,
                out_channels=out_channels,
                width=base_width * (2 ** i),
                modes=tuple(m // (2 ** i) for m in modes),
                num_layers=2,
                latent_dim=latent_dim,
            )
            for i in range(num_scales)
        ])
        
        # Upsampling layers
        self.upsamplers = nn.ModuleList([
            nn.Upsample(scale_factor=2, mode='trilinear', align_corners=False)
            for _ in range(num_scales - 1)
        ])
        
        # Fusion layer
        self.fusion = nn.Conv3d(out_channels * num_scales, out_channels, 1)
    
    def forward(
        self,
        x: torch.Tensor,
        latent: torch.Tensor,
    ) -> torch.Tensor:
        """Multi-scale forward pass"""
        outputs = []
        current = x
        
        for i, fno in enumerate(self.scale_fnos):
            # Process at current scale
            out = fno(current, latent)
            outputs.append(out)
            
            # Downsample for next scale
            if i < self.num_scales - 1:
                current = F.avg_pool3d(out, 2)
        
        # Upsample and combine
        for i in range(self.num_scales - 2, -1, -1):
            outputs[i + 1] = self.upsamplers[i](outputs[i + 1])
            outputs[i + 1] = F.interpolate(
                outputs[i + 1], 
                size=outputs[i].shape[-3:],
                mode='trilinear',
                align_corners=False
            )
        
        # Concatenate and fuse
        combined = torch.cat(outputs, dim=1)
        velocity = self.fusion(combined)
        
        return velocity


class TimeConditionedFNO3D(FourierNeuralOperator3D):
    """
    FNO with explicit time conditioning for morphogenesis phases
    Allows different behaviors during chaos, emergence, and stabilization
    """
    
    def __init__(
        self,
        in_channels: int = 4,
        out_channels: int = 3,
        width: int = 64,
        modes: Tuple[int, int, int] = (16, 16, 16),
        num_layers: int = 4,
        latent_dim: int = 512,
        max_time_steps: int = 1000,
    ):
        super().__init__(
            in_channels, out_channels, width, modes, num_layers, latent_dim
        )
        
        # Time embedding
        self.time_embed = nn.Sequential(
            SinusoidalPositionalEmbedding(width),
            nn.Linear(width, width * 4),
            nn.GELU(),
            nn.Linear(width * 4, width),
        )
        
        # Time-conditioned normalization
        self.time_norm = nn.ModuleList([
            nn.GroupNorm(8, width)
            for _ in range(num_layers)
        ])
    
    def forward(
        self,
        x: torch.Tensor,
        latent: torch.Tensor,
        time_embedding: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Forward pass with time conditioning"""
        time_step = time_embedding  # Alias for compatibility
        if time_step is None:
            time_step = torch.zeros(x.shape[0], device=x.device)
        
        # Get time embedding
        t_emb = self.time_embed(time_step)  # (batch, width)
        t_emb = t_emb.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)  # (batch, width, 1, 1, 1)
        
        # Get latent conditioning
        conditions = self.conditioner(latent)
        
        # Lift
        x = self.lift(x)
        
        # Apply blocks with time and latent conditioning
        for i, block in enumerate(self.blocks):
            x = block(x)
            
            # Time modulation
            x = self.time_norm[i](x + t_emb)
            
            # Latent FiLM conditioning
            scale, shift = conditions[i]
            x = x * (1 + scale) + shift
        
        velocity = self.project(x)
        return velocity


class SinusoidalPositionalEmbedding(nn.Module):
    """Sinusoidal positional embedding for time steps"""
    
    def __init__(self, dim: int, max_period: float = 10000):
        super().__init__()
        self.dim = dim
        self.max_period = max_period
    
    def forward(self, time: torch.Tensor) -> torch.Tensor:
        """
        Args:
            time: Tensor of shape (batch,) with time values [0, 1]
        Returns:
            Embedding of shape (batch, dim)
        """
        half_dim = self.dim // 2
        freqs = torch.exp(
            -math.log(self.max_period) 
            * torch.arange(half_dim, device=time.device) 
            / half_dim
        )
        
        args = time.unsqueeze(-1) * freqs.unsqueeze(0)
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        
        if self.dim % 2:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
        
        return embedding
