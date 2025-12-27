"""
U-Net Neural MPM (Material Point Method)
Predicts particle velocities using a 3D U-Net architecture
Optimized for organic/squishy physics simulation
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint
from typing import Tuple, Optional, List
from einops import rearrange


class ResBlock3D(nn.Module):
    """
    3D Residual Block with optional time and latent conditioning
    """
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        time_emb_dim: Optional[int] = None,
        dropout: float = 0.1,
        use_conv_shortcut: bool = False,
        groups: int = 8,
    ):
        super().__init__()
        
        self.in_channels = in_channels
        self.out_channels = out_channels
        
        # First convolution block
        self.norm1 = nn.GroupNorm(groups, in_channels)
        self.conv1 = nn.Conv3d(in_channels, out_channels, 3, padding=1)
        
        # Time embedding projection
        self.time_emb_proj = None
        if time_emb_dim is not None:
            self.time_emb_proj = nn.Sequential(
                nn.SiLU(),
                nn.Linear(time_emb_dim, out_channels),
            )
        
        # Second convolution block
        self.norm2 = nn.GroupNorm(groups, out_channels)
        self.dropout = nn.Dropout(dropout)
        self.conv2 = nn.Conv3d(out_channels, out_channels, 3, padding=1)
        
        # Skip connection
        if in_channels != out_channels:
            if use_conv_shortcut:
                self.shortcut = nn.Conv3d(in_channels, out_channels, 3, padding=1)
            else:
                self.shortcut = nn.Conv3d(in_channels, out_channels, 1)
        else:
            self.shortcut = nn.Identity()
        
        self.act = nn.SiLU()
    
    def forward(
        self,
        x: torch.Tensor,
        time_emb: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Forward pass with optional time conditioning"""
        h = x
        
        # First block
        h = self.norm1(h)
        h = self.act(h)
        h = self.conv1(h)
        
        # Add time embedding
        if time_emb is not None and self.time_emb_proj is not None:
            time_emb_proj = self.time_emb_proj(time_emb)
            h = h + time_emb_proj[:, :, None, None, None]
        
        # Second block
        h = self.norm2(h)
        h = self.act(h)
        h = self.dropout(h)
        h = self.conv2(h)
        
        return h + self.shortcut(x)


class AttentionBlock3D(nn.Module):
    """
    3D Self-Attention Block for capturing long-range dependencies
    Memory-efficient implementation using chunked attention
    """
    
    def __init__(
        self,
        channels: int,
        num_heads: int = 8,
        head_dim: int = 64,
    ):
        super().__init__()
        
        self.channels = channels
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.inner_dim = num_heads * head_dim
        self.scale = head_dim ** -0.5
        
        self.norm = nn.GroupNorm(8, channels)
        
        self.to_qkv = nn.Conv3d(channels, self.inner_dim * 3, 1)
        self.to_out = nn.Sequential(
            nn.Conv3d(self.inner_dim, channels, 1),
            nn.Dropout(0.1),
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with memory-efficient attention"""
        b, c, d, h, w = x.shape
        residual = x
        
        x = self.norm(x)
        
        # Compute Q, K, V
        qkv = self.to_qkv(x)
        qkv = rearrange(qkv, 'b (three heads dim) d h w -> three b heads (d h w) dim',
                       three=3, heads=self.num_heads, dim=self.head_dim)
        q, k, v = qkv[0], qkv[1], qkv[2]
        
        # Scaled dot-product attention
        attn = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        attn = F.softmax(attn, dim=-1)
        
        # Apply attention to values
        out = torch.matmul(attn, v)
        out = rearrange(out, 'b heads (d h w) dim -> b (heads dim) d h w',
                       d=d, h=h, w=w)
        
        out = self.to_out(out)
        return out + residual


class DownBlock3D(nn.Module):
    """Downsampling block with ResBlocks and optional attention"""
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        time_emb_dim: int,
        num_res_blocks: int = 2,
        use_attention: bool = False,
        dropout: float = 0.1,
    ):
        super().__init__()
        
        self.res_blocks = nn.ModuleList([
            ResBlock3D(
                in_channels if i == 0 else out_channels,
                out_channels,
                time_emb_dim,
                dropout,
            )
            for i in range(num_res_blocks)
        ])
        
        self.attention = None
        if use_attention:
            self.attention = AttentionBlock3D(out_channels)
        
        self.downsample = nn.Conv3d(out_channels, out_channels, 3, stride=2, padding=1)
    
    def forward(
        self,
        x: torch.Tensor,
        time_emb: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Returns downsampled output and skip connection"""
        for res_block in self.res_blocks:
            x = res_block(x, time_emb)
        
        if self.attention is not None:
            x = self.attention(x)
        
        skip = x
        x = self.downsample(x)
        
        return x, skip


class UpBlock3D(nn.Module):
    """Upsampling block with ResBlocks and skip connections"""
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        skip_channels: int,
        time_emb_dim: int,
        num_res_blocks: int = 2,
        use_attention: bool = False,
        dropout: float = 0.1,
    ):
        super().__init__()
        
        self.upsample = nn.ConvTranspose3d(in_channels, in_channels, 4, stride=2, padding=1)
        
        self.res_blocks = nn.ModuleList([
            ResBlock3D(
                in_channels + skip_channels if i == 0 else out_channels,
                out_channels,
                time_emb_dim,
                dropout,
            )
            for i in range(num_res_blocks)
        ])
        
        self.attention = None
        if use_attention:
            self.attention = AttentionBlock3D(out_channels)
    
    def forward(
        self,
        x: torch.Tensor,
        skip: torch.Tensor,
        time_emb: torch.Tensor,
    ) -> torch.Tensor:
        """Forward with skip connection from encoder"""
        x = self.upsample(x)
        
        # Handle size mismatch
        if x.shape[-3:] != skip.shape[-3:]:
            x = F.interpolate(x, size=skip.shape[-3:], mode='trilinear', align_corners=False)
        
        x = torch.cat([x, skip], dim=1)
        
        for res_block in self.res_blocks:
            x = res_block(x, time_emb)
        
        if self.attention is not None:
            x = self.attention(x)
        
        return x


class MiddleBlock3D(nn.Module):
    """Middle block at bottleneck with attention"""
    
    def __init__(
        self,
        channels: int,
        time_emb_dim: int,
        dropout: float = 0.1,
    ):
        super().__init__()
        
        self.res1 = ResBlock3D(channels, channels, time_emb_dim, dropout)
        self.attention = AttentionBlock3D(channels)
        self.res2 = ResBlock3D(channels, channels, time_emb_dim, dropout)
    
    def forward(self, x: torch.Tensor, time_emb: torch.Tensor) -> torch.Tensor:
        x = self.res1(x, time_emb)
        x = self.attention(x)
        x = self.res2(x, time_emb)
        return x


class LatentFiLM(nn.Module):
    """Feature-wise Linear Modulation from latent DNA"""
    
    def __init__(
        self,
        latent_dim: int,
        feature_channels: List[int],
    ):
        super().__init__()
        
        self.projections = nn.ModuleList([
            nn.Sequential(
                nn.Linear(latent_dim, latent_dim),
                nn.GELU(),
                nn.Linear(latent_dim, ch * 2),
            )
            for ch in feature_channels
        ])
    
    def forward(self, latent: torch.Tensor) -> List[Tuple[torch.Tensor, torch.Tensor]]:
        """Generate scale and shift for each feature level"""
        conditions = []
        
        for proj in self.projections:
            params = proj(latent)
            scale, shift = params.chunk(2, dim=-1)
            scale = scale.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
            shift = shift.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
            conditions.append((scale, shift))
        
        return conditions


class TimeEmbedding(nn.Module):
    """Sinusoidal time embedding"""
    
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim
        
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.SiLU(),
            nn.Linear(dim * 4, dim),
        )
    
    def forward(self, time: torch.Tensor) -> torch.Tensor:
        """
        Args:
            time: Tensor of shape (batch,) with values in [0, 1]
        """
        half_dim = self.dim // 2
        freqs = torch.exp(
            -math.log(10000) * torch.arange(half_dim, device=time.device) / half_dim
        )
        
        args = time.unsqueeze(-1) * freqs.unsqueeze(0) * 1000  # Scale up
        emb = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)
        
        return self.mlp(emb)


class UNetNeuralMPM(nn.Module):
    """
    3D U-Net for Neural Material Point Method
    Predicts particle velocity fields conditioned on latent DNA
    
    Architecture:
        - Encoder: Progressive downsampling with residual blocks
        - Middle: Bottleneck with attention
        - Decoder: Progressive upsampling with skip connections
        - Conditioning: Time + Latent DNA via FiLM
    """
    
    def __init__(
        self,
        in_channels: int = 4,           # (x, y, z, density)
        out_channels: int = 3,          # (vx, vy, vz)
        base_channels: int = 64,
        channel_multipliers: Tuple[int, ...] = (1, 2, 4, 8),
        num_res_blocks: int = 2,
        attention_resolutions: Tuple[int, ...] = (16, 8),
        dropout: float = 0.1,
        latent_dim: int = 512,
        use_checkpoint: bool = True,
    ):
        super().__init__()
        
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.base_channels = base_channels
        self.num_levels = len(channel_multipliers)
        self.use_checkpoint = use_checkpoint
        
        # Channel dimensions at each level
        channels = [base_channels * m for m in channel_multipliers]
        
        # Time embedding
        time_emb_dim = base_channels * 4
        self.time_embed = TimeEmbedding(time_emb_dim)
        
        # Latent conditioning
        self.latent_film = LatentFiLM(latent_dim, channels)
        
        # Input projection
        self.input_conv = nn.Conv3d(in_channels, base_channels, 3, padding=1)
        
        # Encoder (Downsampling path)
        self.down_blocks = nn.ModuleList()
        in_ch = base_channels
        current_res = 64  # Assuming input resolution
        
        for i, mult in enumerate(channel_multipliers):
            out_ch = base_channels * mult
            use_attn = current_res in attention_resolutions
            
            self.down_blocks.append(
                DownBlock3D(
                    in_ch, out_ch, time_emb_dim,
                    num_res_blocks, use_attn, dropout
                )
            )
            
            in_ch = out_ch
            current_res //= 2
        
        # Middle (Bottleneck)
        self.middle = MiddleBlock3D(channels[-1], time_emb_dim, dropout)
        
        # Decoder (Upsampling path)
        self.up_blocks = nn.ModuleList()
        
        for i, mult in enumerate(reversed(channel_multipliers)):
            out_ch = base_channels * mult
            skip_ch = channels[-(i + 1)]
            use_attn = current_res in attention_resolutions
            
            self.up_blocks.append(
                UpBlock3D(
                    in_ch, out_ch, skip_ch, time_emb_dim,
                    num_res_blocks, use_attn, dropout
                )
            )
            
            in_ch = out_ch
            current_res *= 2
        
        # Output projection
        self.output_norm = nn.GroupNorm(8, base_channels)
        self.output_conv = nn.Sequential(
            nn.SiLU(),
            nn.Conv3d(base_channels, out_channels, 3, padding=1),
        )
        
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights with improved scheme"""
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.GroupNorm):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
        
        # Zero-init last layer for stable training
        last_conv = self.output_conv[-1]
        if hasattr(last_conv, 'weight'):
            nn.init.zeros_(last_conv.weight)  # type: ignore
        if hasattr(last_conv, 'bias') and last_conv.bias is not None:
            nn.init.zeros_(last_conv.bias)  # type: ignore
    
    def forward(
        self,
        x: torch.Tensor,
        latent: torch.Tensor,
        time_step: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Forward pass through U-Net
        
        Args:
            x: Input field (batch, in_channels, D, H, W)
            latent: Latent DNA vector (batch, latent_dim)
            time_step: Normalized time [0, 1] (batch,)
            
        Returns:
            Velocity field (batch, out_channels, D, H, W)
        """
        # Time embedding
        if time_step is None:
            time_step = torch.zeros(x.shape[0], device=x.device)
        time_emb = self.time_embed(time_step)
        
        # Latent conditioning parameters
        latent_conditions = self.latent_film(latent)
        
        # Input projection
        h = self.input_conv(x)
        
        # Encoder with skip connections
        skips = []
        for i, down_block in enumerate(self.down_blocks):
            if self.use_checkpoint and self.training:
                result = checkpoint(
                    down_block, h, time_emb,
                    use_reentrant=False
                )
                h, skip = result  # type: ignore
            else:
                h, skip = down_block(h, time_emb)
            
            # Apply latent FiLM conditioning
            scale, shift = latent_conditions[i]
            skip = skip * (1 + scale) + shift
            
            skips.append(skip)
        
        # Middle
        if self.use_checkpoint and self.training:
            h = checkpoint(self.middle, h, time_emb, use_reentrant=False)
        else:
            h = self.middle(h, time_emb)
        
        # Decoder with skip connections
        for i, up_block in enumerate(self.up_blocks):
            skip = skips[-(i + 1)]
            
            if self.use_checkpoint and self.training:
                h = checkpoint(
                    up_block, h, skip, time_emb,
                    use_reentrant=False
                )
            else:
                h = up_block(h, skip, time_emb)
        
        # Output projection
        h = self.output_norm(h)
        velocity = self.output_conv(h)
        
        return velocity
    
    def get_memory_estimate(self, grid_size: int, batch_size: int = 1) -> float:
        """Estimate memory usage in GB"""
        params = sum(p.numel() for p in self.parameters())
        
        # Estimate activations (very rough)
        total_activations = 0
        size = grid_size
        for mult in (1, 2, 4, 8):
            total_activations += self.base_channels * mult * (size ** 3)
            size //= 2
        
        total_activations *= batch_size * 2  # Forward + backward
        
        total_bytes = (params + total_activations) * 4  # float32
        return total_bytes / 1024**3


class LightweightUNetMPM(nn.Module):
    """
    Lightweight U-Net variant for memory-constrained environments
    Uses depthwise separable convolutions and reduced attention
    """
    
    def __init__(
        self,
        in_channels: int = 4,
        out_channels: int = 3,
        base_channels: int = 32,
        latent_dim: int = 256,
    ):
        super().__init__()
        
        # Depthwise separable encoder
        self.encoder = nn.Sequential(
            DepthwiseSeparableConv3d(in_channels, base_channels, 3),
            nn.SiLU(),
            DepthwiseSeparableConv3d(base_channels, base_channels * 2, 3, stride=2),
            nn.SiLU(),
            DepthwiseSeparableConv3d(base_channels * 2, base_channels * 4, 3, stride=2),
            nn.SiLU(),
        )
        
        # Latent conditioning
        self.latent_proj = nn.Linear(latent_dim, base_channels * 4)
        
        # Decoder
        self.decoder = nn.Sequential(
            nn.ConvTranspose3d(base_channels * 4, base_channels * 2, 4, stride=2, padding=1),
            nn.SiLU(),
            nn.ConvTranspose3d(base_channels * 2, base_channels, 4, stride=2, padding=1),
            nn.SiLU(),
            nn.Conv3d(base_channels, out_channels, 3, padding=1),
        )
    
    def forward(
        self,
        x: torch.Tensor,
        latent: torch.Tensor,
        time_step: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        # Encode
        h = self.encoder(x)
        
        # Add latent conditioning
        latent_emb = self.latent_proj(latent)
        latent_emb = latent_emb.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
        h = h + latent_emb
        
        # Decode
        velocity = self.decoder(h)
        
        # Resize to input resolution
        if velocity.shape[-3:] != x.shape[-3:]:
            velocity = F.interpolate(velocity, size=x.shape[-3:], mode='trilinear', align_corners=False)
        
        return velocity


class DepthwiseSeparableConv3d(nn.Module):
    """Memory-efficient depthwise separable 3D convolution"""
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        padding: int = 1,
    ):
        super().__init__()
        
        self.depthwise = nn.Conv3d(
            in_channels, in_channels, kernel_size,
            stride=stride, padding=padding, groups=in_channels
        )
        self.pointwise = nn.Conv3d(in_channels, out_channels, 1)
        self.norm = nn.GroupNorm(min(8, in_channels), in_channels)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.depthwise(x)
        x = self.norm(x)
        x = self.pointwise(x)
        return x
