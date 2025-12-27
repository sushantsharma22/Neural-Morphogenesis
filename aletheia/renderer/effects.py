"""
Post-Processing Effects for Neural Morphogenesis Renderer
Implements Bioluminescent Bloom, Depth-of-Field, and Motion Blur
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Tuple, Optional
import math


class BloomEffect:
    """
    Bioluminescent bloom effect using multi-scale Gaussian blur
    Creates ethereal glow around bright particles
    """
    
    def __init__(
        self,
        resolution: Tuple[int, int] = (1920, 1080),
        num_passes: int = 5,
        device: torch.device = torch.device("cuda"),
    ):
        self.width, self.height = resolution
        self.num_passes = num_passes
        self.device = device
        
        # Pre-compute Gaussian kernels at different scales
        self.kernels = self._create_kernels()
    
    def _create_kernels(self) -> list:
        """Create Gaussian kernels for multi-scale blur"""
        kernels = []
        
        for i in range(self.num_passes):
            size = 2 ** (i + 2) + 1  # 5, 9, 17, 33, 65
            sigma = size / 6.0
            
            kernel = self._gaussian_kernel(size, sigma)
            kernels.append(kernel.to(self.device))
        
        return kernels
    
    def _gaussian_kernel(self, size: int, sigma: float) -> torch.Tensor:
        """Create 2D Gaussian kernel"""
        coords = torch.arange(size, dtype=torch.float32) - size // 2
        grid = torch.stack(torch.meshgrid(coords, coords, indexing='ij'), dim=-1)
        
        kernel = torch.exp(-torch.sum(grid ** 2, dim=-1) / (2 * sigma ** 2))
        kernel = kernel / kernel.sum()
        
        return kernel.unsqueeze(0).unsqueeze(0)
    
    def _separable_blur(
        self,
        image: torch.Tensor,
        kernel: torch.Tensor,
    ) -> torch.Tensor:
        """Apply separable Gaussian blur"""
        size = kernel.shape[-1]
        padding = size // 2
        
        # Horizontal pass
        kernel_h = kernel.view(1, 1, 1, size).expand(3, 1, -1, -1)
        blurred = F.conv2d(
            F.pad(image, (padding, padding, 0, 0), mode='reflect'),
            kernel_h, groups=3
        )
        
        # Vertical pass
        kernel_v = kernel.view(1, 1, size, 1).expand(3, 1, -1, -1)
        blurred = F.conv2d(
            F.pad(blurred, (0, 0, padding, padding), mode='reflect'),
            kernel_v, groups=3
        )
        
        return blurred
    
    def apply(
        self,
        image: torch.Tensor,  # (1, 3, H, W)
        intensity: float = 2.5,
        threshold: float = 0.8,
        color_shift: Tuple[float, float, float] = (1.0, 0.9, 1.2),
    ) -> torch.Tensor:
        """
        Apply bioluminescent bloom effect
        
        Args:
            image: Input image tensor (1, 3, H, W)
            intensity: Bloom intensity multiplier
            threshold: Brightness threshold for bloom
            color_shift: RGB multipliers for bloom tint
            
        Returns:
            Image with bloom applied
        """
        # Extract bright regions
        luminance = 0.299 * image[:, 0] + 0.587 * image[:, 1] + 0.114 * image[:, 2]
        mask = (luminance > threshold).float().unsqueeze(1)
        bright = image * mask
        
        # Multi-scale blur
        bloom = torch.zeros_like(image)
        
        for i, kernel in enumerate(self.kernels):
            # Downsample for larger kernels
            scale = 2 ** i
            if scale > 1:
                h, w = image.shape[2] // scale, image.shape[3] // scale
                downsampled = F.interpolate(bright, size=(h, w), mode='bilinear', align_corners=False)
            else:
                downsampled = bright
            
            # Apply blur
            kernel_1d = kernel.squeeze().sum(dim=0, keepdim=True)
            kernel_1d = kernel_1d / kernel_1d.sum()
            
            blurred = self._separable_blur(downsampled, kernel_1d.unsqueeze(0).unsqueeze(0))
            
            # Upsample back
            if scale > 1:
                blurred = F.interpolate(blurred, size=image.shape[2:], mode='bilinear', align_corners=False)
            
            # Accumulate with decreasing weight
            weight = 1.0 / (i + 1)
            bloom += blurred * weight
        
        # Apply color shift
        color_tensor = torch.tensor(color_shift, device=self.device).view(1, 3, 1, 1)
        bloom = bloom * color_tensor
        
        # Combine with original
        result = image + bloom * intensity
        
        return result


class DepthOfField:
    """
    Depth-of-field effect using circle of confusion
    Creates cinematic focus with bokeh blur
    """
    
    def __init__(
        self,
        resolution: Tuple[int, int] = (1920, 1080),
        max_blur_radius: int = 20,
        device: torch.device = torch.device("cuda"),
    ):
        self.width, self.height = resolution
        self.max_blur_radius = max_blur_radius
        self.device = device
        
        # Pre-compute bokeh kernels
        self.bokeh_kernels = self._create_bokeh_kernels()
    
    def _create_bokeh_kernels(self) -> list:
        """Create circular bokeh kernels at different sizes"""
        kernels = []
        
        for radius in range(1, self.max_blur_radius + 1):
            size = radius * 2 + 1
            center = radius
            
            y, x = torch.meshgrid(
                torch.arange(size, dtype=torch.float32),
                torch.arange(size, dtype=torch.float32),
                indexing='ij'
            )
            
            dist = torch.sqrt((x - center) ** 2 + (y - center) ** 2)
            
            # Circular kernel with soft edge
            kernel = torch.clamp(1.0 - (dist - radius + 1), 0, 1)
            kernel = kernel / kernel.sum()
            
            kernels.append(kernel.to(self.device))
        
        return kernels
    
    def compute_coc(
        self,
        depth: torch.Tensor,  # (H, W)
        focus_distance: float,
        aperture: float,
        focal_length: float = 50.0,
    ) -> torch.Tensor:
        """
        Compute circle of confusion from depth buffer
        
        Args:
            depth: Depth buffer (H, W)
            focus_distance: Distance to focus plane
            aperture: Lens aperture (f-stop)
            focal_length: Lens focal length in mm
            
        Returns:
            CoC radius for each pixel
        """
        # Simplified CoC formula
        # CoC = |aperture * focal_length * (focus_distance - depth) / (depth * (focus_distance - focal_length))|
        
        f = focal_length / 1000.0  # Convert to meters
        
        coc = torch.abs(
            aperture * f * (focus_distance - depth) / 
            (depth * (focus_distance - f) + 1e-6)
        )
        
        # Normalize and scale to blur radius
        coc = coc * self.max_blur_radius * 1000
        coc = torch.clamp(coc, 0, self.max_blur_radius)
        
        return coc
    
    def apply(
        self,
        image: torch.Tensor,  # (1, 3, H, W)
        depth: torch.Tensor,  # (H, W) or (1, 1, H, W)
        focus_distance: float = 5.0,
        aperture: float = 0.02,
    ) -> torch.Tensor:
        """
        Apply depth-of-field effect
        
        Args:
            image: Input image (1, 3, H, W)
            depth: Depth buffer
            focus_distance: Distance to focus plane
            aperture: Lens aperture
            
        Returns:
            Image with DOF applied
        """
        if depth.dim() == 4:
            depth = depth.squeeze()
        
        # Compute CoC
        coc = self.compute_coc(depth, focus_distance, aperture)
        
        # Quantize CoC to discrete blur levels
        coc_levels = (coc * (len(self.bokeh_kernels) - 1) / self.max_blur_radius).long()
        coc_levels = torch.clamp(coc_levels, 0, len(self.bokeh_kernels) - 1)
        
        # Apply variable blur (simplified - uses max blur per region)
        # For better quality, would use scatter-gather approach
        
        result = image.clone()
        
        # Create blur pyramid
        blur_pyramid = [image]
        for i, kernel in enumerate(self.bokeh_kernels):
            size = kernel.shape[0]
            padding = size // 2
            
            kernel_4d = kernel.unsqueeze(0).unsqueeze(0).expand(3, 1, -1, -1)
            
            blurred = F.conv2d(
                F.pad(image, (padding, padding, padding, padding), mode='reflect'),
                kernel_4d, groups=3
            )
            blur_pyramid.append(blurred)
        
        # Blend based on CoC
        for i in range(len(self.bokeh_kernels)):
            mask = (coc_levels == i).float().unsqueeze(0).unsqueeze(0)
            result = result * (1 - mask) + blur_pyramid[i + 1] * mask
        
        return result


class MotionBlur:
    """
    Motion blur effect using velocity buffer
    Creates cinematic motion trails
    """
    
    def __init__(
        self,
        resolution: Tuple[int, int] = (1920, 1080),
        samples: int = 8,
        device: torch.device = torch.device("cuda"),
    ):
        self.width, self.height = resolution
        self.samples = samples
        self.device = device
    
    def apply(
        self,
        image: torch.Tensor,  # (1, 3, H, W)
        velocity: torch.Tensor,  # (2, H, W) - pixel velocity in x, y
        shutter_time: float = 0.5,
    ) -> torch.Tensor:
        """
        Apply motion blur using velocity buffer
        
        Args:
            image: Input image
            velocity: Per-pixel velocity in screen space
            shutter_time: Exposure time (0-1)
            
        Returns:
            Motion-blurred image
        """
        _, _, h, w = image.shape
        
        # Create sample coordinates
        result = torch.zeros_like(image)
        
        for i in range(self.samples):
            # Sample along velocity vector
            t = (i / (self.samples - 1) - 0.5) * shutter_time
            
            # Offset coordinates
            y_coords, x_coords = torch.meshgrid(
                torch.arange(h, device=self.device, dtype=torch.float32),
                torch.arange(w, device=self.device, dtype=torch.float32),
                indexing='ij'
            )
            
            # Apply velocity offset
            sample_x = x_coords + velocity[0] * t
            sample_y = y_coords + velocity[1] * t
            
            # Normalize to [-1, 1] for grid_sample
            sample_x = sample_x / (w - 1) * 2 - 1
            sample_y = sample_y / (h - 1) * 2 - 1
            
            grid = torch.stack([sample_x, sample_y], dim=-1).unsqueeze(0)
            
            # Sample image
            sampled = F.grid_sample(
                image, grid,
                mode='bilinear', padding_mode='border', align_corners=True
            )
            
            result += sampled
        
        result /= self.samples
        return result
    
    def apply_temporal(
        self,
        current_frame: torch.Tensor,
        prev_frame: torch.Tensor,
        blend_factor: float = 0.3,
    ) -> torch.Tensor:
        """
        Simple temporal motion blur using frame blending
        
        Args:
            current_frame: Current frame
            prev_frame: Previous frame
            blend_factor: Blend weight for previous frame
            
        Returns:
            Temporally blurred frame
        """
        return (1 - blend_factor) * current_frame + blend_factor * prev_frame


class ColorGrading:
    """
    Color grading and tone mapping for cinematic look
    """
    
    def __init__(
        self,
        device: torch.device = torch.device("cuda"),
    ):
        self.device = device
    
    def apply(
        self,
        image: torch.Tensor,  # (1, 3, H, W)
        exposure: float = 1.0,
        gamma: float = 2.2,
        saturation: float = 1.0,
        contrast: float = 1.0,
        shadows: float = 0.0,
        highlights: float = 0.0,
    ) -> torch.Tensor:
        """
        Apply color grading
        
        Args:
            image: Input image (1, 3, H, W) in linear space
            exposure: Exposure adjustment
            gamma: Gamma correction value
            saturation: Color saturation
            contrast: Contrast adjustment
            shadows: Shadow lift
            highlights: Highlight compression
            
        Returns:
            Graded image
        """
        result = image.clone()
        
        # Exposure
        result = result * exposure
        
        # Shadows and highlights
        luminance = 0.299 * result[:, 0:1] + 0.587 * result[:, 1:2] + 0.114 * result[:, 2:3]
        
        if shadows != 0:
            shadow_mask = torch.clamp(1.0 - luminance * 2, 0, 1)
            result = result + shadow_mask * shadows
        
        if highlights != 0:
            highlight_mask = torch.clamp(luminance * 2 - 1, 0, 1)
            result = result - highlight_mask * highlights
        
        # Contrast
        if contrast != 1.0:
            result = (result - 0.5) * contrast + 0.5
        
        # Saturation
        if saturation != 1.0:
            gray = 0.299 * result[:, 0:1] + 0.587 * result[:, 1:2] + 0.114 * result[:, 2:3]
            result = gray + saturation * (result - gray)
        
        # Tone mapping (simple Reinhard)
        result = result / (1 + result)
        
        # Gamma correction
        result = torch.pow(torch.clamp(result, 1e-6, None), 1.0 / gamma)
        
        return torch.clamp(result, 0, 1)
    
    def apply_lut(
        self,
        image: torch.Tensor,
        lut: torch.Tensor,  # (32, 32, 32, 3) or similar
    ) -> torch.Tensor:
        """Apply 3D LUT for advanced color grading"""
        # Simplified - would use trilinear interpolation
        return image  # Passthrough for now
    
    def filmic_tonemap(
        self,
        image: torch.Tensor,
    ) -> torch.Tensor:
        """ACES-like filmic tone mapping"""
        # ACES approximation
        a = 2.51
        b = 0.03
        c = 2.43
        d = 0.59
        e = 0.14
        
        result = (image * (a * image + b)) / (image * (c * image + d) + e)
        return torch.clamp(result, 0, 1)


class BioluminescentPalette:
    """
    Color palette generator for bioluminescent effects
    Creates organic, glowing color schemes
    """
    
    PALETTES = {
        "bioluminescent": [
            (0.1, 0.8, 1.0),   # Cyan
            (0.4, 0.9, 0.8),   # Teal
            (0.2, 0.6, 1.0),   # Blue
            (0.8, 0.4, 1.0),   # Purple
            (0.3, 1.0, 0.6),   # Green
        ],
        "stellar": [
            (1.0, 0.9, 0.8),   # White-yellow
            (0.8, 0.9, 1.0),   # Blue-white
            (1.0, 0.6, 0.4),   # Orange
            (0.6, 0.8, 1.0),   # Light blue
            (1.0, 0.8, 0.9),   # Pink-white
        ],
        "ocean": [
            (0.0, 0.5, 0.8),   # Deep blue
            (0.2, 0.8, 0.9),   # Cyan
            (0.0, 0.9, 0.7),   # Teal
            (0.4, 0.6, 1.0),   # Periwinkle
            (0.1, 0.7, 0.6),   # Sea green
        ],
        "synaptic": [
            (0.8, 0.3, 1.0),   # Violet
            (0.4, 0.5, 1.0),   # Blue-violet
            (1.0, 0.4, 0.8),   # Pink
            (0.6, 0.8, 1.0),   # Light blue
            (0.9, 0.6, 1.0),   # Lavender
        ],
    }
    
    @classmethod
    def get_color(
        cls,
        palette_name: str,
        t: float,  # [0, 1]
        device: torch.device = torch.device("cuda"),
    ) -> torch.Tensor:
        """
        Get interpolated color from palette
        
        Args:
            palette_name: Name of color palette
            t: Interpolation parameter [0, 1]
            device: Torch device
            
        Returns:
            RGB color tensor (3,)
        """
        palette = cls.PALETTES.get(palette_name, cls.PALETTES["bioluminescent"])
        n = len(palette)
        
        t_scaled = t * (n - 1)
        idx = int(t_scaled)
        frac = t_scaled - idx
        
        idx = min(idx, n - 2)
        
        c1 = torch.tensor(palette[idx], device=device)
        c2 = torch.tensor(palette[idx + 1], device=device)
        
        return c1 * (1 - frac) + c2 * frac
    
    @classmethod
    def colorize_by_value(
        cls,
        values: torch.Tensor,  # (N,) values in [0, 1]
        palette_name: str = "bioluminescent",
    ) -> torch.Tensor:
        """
        Map values to colors using palette
        
        Args:
            values: Normalized values (N,)
            palette_name: Color palette name
            
        Returns:
            RGB colors (N, 3)
        """
        palette = cls.PALETTES.get(palette_name, cls.PALETTES["bioluminescent"])
        palette_tensor = torch.tensor(palette, device=values.device)
        n = len(palette)
        
        t_scaled = values * (n - 1)
        idx = t_scaled.long().clamp(0, n - 2)
        frac = (t_scaled - idx.float()).unsqueeze(1)
        
        c1 = palette_tensor[idx]
        c2 = palette_tensor[idx + 1]
        
        return c1 * (1 - frac) + c2 * frac
