"""
Headless Ray-Tracing Renderer for Neural Morphogenesis
Renders bioluminescent particle systems without X11/GUI
Uses Taichi for GPU-accelerated ray tracing

Note: Taichi-specific constructs may show type errors in static analysis.
"""
# pyright: reportGeneralTypeIssues=false

import taichi as ti
import numpy as np
import torch
from typing import Tuple, Optional, Dict, List, Union, TYPE_CHECKING
from dataclasses import dataclass
import math

if TYPE_CHECKING:
    from aletheia.renderer.camera import Camera


@dataclass
class RenderSettings:
    """Rendering settings"""
    resolution: Tuple[int, int] = (3840, 2160)  # 4K
    samples_per_pixel: int = 64
    max_bounces: int = 8
    particle_radius: float = 0.002
    emission_strength: float = 5.0
    background_color: Tuple[float, float, float] = (0.0, 0.0, 0.02)


@ti.data_oriented
class ParticleRenderer:
    """
    GPU-accelerated particle renderer using Taichi
    Implements volumetric rendering with emission
    """
    
    def __init__(
        self,
        resolution: Tuple[int, int] = (1920, 1080),
        max_particles: int = 10_000_000,
        samples_per_pixel: int = 16,
        device_id: int = 0,
    ):
        self.width, self.height = resolution
        self.max_particles = max_particles
        self.spp = samples_per_pixel
        self.device_id = device_id
        
        # Particle data
        self.particle_pos = ti.Vector.field(3, dtype=ti.f32, shape=max_particles)
        self.particle_color = ti.Vector.field(3, dtype=ti.f32, shape=max_particles)
        self.particle_radius = ti.field(dtype=ti.f32, shape=max_particles)
        self.particle_emission = ti.field(dtype=ti.f32, shape=max_particles)
        self.num_particles = ti.field(dtype=ti.i32, shape=())
        
        # Image buffer
        self.image = ti.Vector.field(3, dtype=ti.f32, shape=(self.width, self.height))
        self.accumulator = ti.Vector.field(3, dtype=ti.f32, shape=(self.width, self.height))
        self.sample_count = ti.field(dtype=ti.i32, shape=())
        
        # Camera parameters
        self.camera_pos = ti.Vector.field(3, dtype=ti.f32, shape=())
        self.camera_look_at = ti.Vector.field(3, dtype=ti.f32, shape=())
        self.camera_up = ti.Vector.field(3, dtype=ti.f32, shape=())
        self.camera_fov = ti.field(dtype=ti.f32, shape=())
        
        # Depth of field parameters
        self.dof_aperture = ti.field(dtype=ti.f32, shape=())
        self.dof_focus_dist = ti.field(dtype=ti.f32, shape=())
        
        # Initialize defaults
        self.camera_pos[None] = ti.Vector([0.0, 0.0, 8.0])
        self.camera_look_at[None] = ti.Vector([0.0, 0.0, 0.0])
        self.camera_up[None] = ti.Vector([0.0, 1.0, 0.0])
        self.camera_fov[None] = 45.0
        self.dof_aperture[None] = 0.0
        self.dof_focus_dist[None] = 8.0
    
    def set_particles(
        self,
        positions: np.ndarray,
        colors: np.ndarray,
        radii: Optional[np.ndarray] = None,
        emissions: Optional[np.ndarray] = None,
    ):
        """Load particle data into renderer"""
        n = positions.shape[0]
        self.num_particles[None] = n
        
        self.particle_pos.from_numpy(positions.astype(np.float32))
        self.particle_color.from_numpy(colors.astype(np.float32))
        
        if radii is not None:
            self.particle_radius.from_numpy(radii.astype(np.float32))
        else:
            radii_np = np.full(n, 0.002, dtype=np.float32)
            self.particle_radius.from_numpy(radii_np)
        
        if emissions is not None:
            self.particle_emission.from_numpy(emissions.astype(np.float32))
        else:
            emission_np = np.full(n, 5.0, dtype=np.float32)
            self.particle_emission.from_numpy(emission_np)
    
    def set_camera(
        self,
        position: Tuple[float, float, float],
        look_at: Tuple[float, float, float],
        up: Tuple[float, float, float] = (0.0, 1.0, 0.0),
        fov: float = 45.0,
    ):
        """Set camera parameters"""
        self.camera_pos[None] = ti.Vector(position)
        self.camera_look_at[None] = ti.Vector(look_at)
        self.camera_up[None] = ti.Vector(up)
        self.camera_fov[None] = fov
    
    def set_dof(
        self,
        aperture: float,
        focus_distance: float,
    ):
        """Set depth of field parameters"""
        self.dof_aperture[None] = aperture
        self.dof_focus_dist[None] = focus_distance
    
    @ti.func
    def ray_sphere_intersect(  # type: ignore[no-untyped-def]
        self,
        ray_origin,
        ray_dir,
        sphere_center,
        sphere_radius,
    ):
        """Ray-sphere intersection test"""
        oc = ray_origin - sphere_center
        a = ray_dir.dot(ray_dir)
        b = 2.0 * oc.dot(ray_dir)
        c = oc.dot(oc) - sphere_radius * sphere_radius
        
        discriminant = b * b - 4 * a * c
        t = -1.0
        
        if discriminant >= 0:
            t = (-b - ti.sqrt(discriminant)) / (2.0 * a)
            if t < 0:
                t = (-b + ti.sqrt(discriminant)) / (2.0 * a)
        
        return t
    
    @ti.func
    def random_in_unit_disk(self):  # type: ignore[no-untyped-def]
        """Generate random point in unit disk for DOF"""
        theta = 2.0 * math.pi * ti.random()
        r = ti.sqrt(ti.random())
        return ti.Vector([r * ti.cos(theta), r * ti.sin(theta)])
    
    @ti.kernel
    def render_kernel(self):
        """Main rendering kernel"""
        # Camera setup
        cam_pos = self.camera_pos[None]
        cam_target = self.camera_look_at[None]
        cam_up = self.camera_up[None]
        fov = self.camera_fov[None]
        
        # Compute camera basis
        diff = cam_target - cam_pos
        forward = diff.normalized()  # type: ignore[union-attr]
        right = forward.cross(cam_up).normalized()  # type: ignore[union-attr]
        up = right.cross(forward).normalized()  # type: ignore[union-attr]
        
        # Compute viewport
        aspect = self.width / self.height
        fov_rad = fov * math.pi / 180.0
        viewport_height = 2.0 * ti.tan(fov_rad / 2.0)
        viewport_width = aspect * viewport_height
        
        for i, j in self.image:
            color = ti.Vector([0.0, 0.0, 0.0])
            
            for _ in range(self.spp):
                # Pixel coordinates with jitter
                u = (i + ti.random()) / self.width
                v = (j + ti.random()) / self.height
                
                # Ray direction
                ray_dir = forward + \
                          (u - 0.5) * viewport_width * right + \
                          (v - 0.5) * viewport_height * up
                ray_dir = ray_dir.normalized()
                
                # DOF offset
                ray_origin = cam_pos
                if self.dof_aperture[None] > 0:
                    disk_sample = self.random_in_unit_disk()
                    offset = (disk_sample[0] * right + disk_sample[1] * up) * self.dof_aperture[None]
                    ray_origin = cam_pos + offset
                    
                    # Adjust ray direction to focus plane
                    focus_point = cam_pos + ray_dir * self.dof_focus_dist[None]
                    ray_dir = (focus_point - ray_origin).normalized()
                
                # Trace ray
                sample_color = self.trace_ray(ray_origin, ray_dir)
                color += sample_color
            
            color /= self.spp
            self.image[i, j] = color
    
    @ti.func
    def trace_ray(  # type: ignore[no-untyped-def]
        self,
        ray_origin,
        ray_dir,
    ):
        """Trace a single ray through the particle field"""
        color = ti.Vector([0.0, 0.0, 0.02])  # Background color
        
        closest_t = 1e10
        hit_particle = -1
        
        # Find closest intersection
        num_p = self.num_particles[None]
        for p in range(num_p):  # type: ignore[arg-type]
            pos = self.particle_pos[p]
            radius = self.particle_radius[p]
            
            t = self.ray_sphere_intersect(ray_origin, ray_dir, pos, radius)
            
            if t > 0.001 and t < closest_t:
                closest_t = t
                hit_particle = p
        
        if hit_particle >= 0:
            # Get particle properties
            hit_color = self.particle_color[hit_particle]
            emission = self.particle_emission[hit_particle]
            
            # Emissive particles (bioluminescence)
            color = hit_color * emission
            
            # Simple volumetric accumulation along ray
            accumulated = ti.Vector([0.0, 0.0, 0.0])
            transmittance = 1.0
            
            step_size = 0.01
            t = 0.01
            
            while t < closest_t and transmittance > 0.01:
                sample_pos = ray_origin + ray_dir * t
                
                # Sample nearby particles for volumetric effect
                local_density = 0.0
                local_color = ti.Vector([0.0, 0.0, 0.0])
                
                for p in range(self.num_particles[None]):  # type: ignore[arg-type]
                    pos = self.particle_pos[p]
                    dist = (sample_pos - pos).norm()
                    radius = self.particle_radius[p] * 5.0  # Influence radius
                    
                    if dist < radius:
                        weight = 1.0 - dist / radius
                        weight = weight * weight
                        local_density += weight
                        local_color += self.particle_color[p] * self.particle_emission[p] * weight
                
                if local_density > 0:
                    local_color /= local_density
                    absorption = 0.5 * local_density
                    accumulated += local_color * transmittance * absorption * step_size
                    transmittance *= ti.exp(-absorption * step_size)
                
                t += step_size
            
            color = color * transmittance + accumulated
        
        return color
    
    def render(self) -> np.ndarray:
        """Render and return image"""
        self.render_kernel()
        return self.image.to_numpy()
    
    def clear(self):
        """Clear image buffer"""
        self.image.fill(0)
        self.accumulator.fill(0)
        self.sample_count[None] = 0


class HeadlessRenderer:
    """
    High-level headless rendering interface
    Combines particle rendering with post-processing
    """
    
    def __init__(
        self,
        settings: RenderSettings,
        device: torch.device = torch.device("cuda"),
    ):
        self.settings = settings
        self.device = device
        
        # Initialize Taichi renderer
        self.particle_renderer = ParticleRenderer(
            resolution=settings.resolution,
            samples_per_pixel=settings.samples_per_pixel,
        )
        
        # Import effects
        from aletheia.renderer.effects import BloomEffect, DepthOfField, MotionBlur, ColorGrading
        
        # Post-processing effects
        self.bloom = BloomEffect(
            resolution=settings.resolution,
            device=device,
        )
        self.color_grading = ColorGrading(device=device)
        
        # Motion blur buffer
        self.prev_frame = None
    
    def set_camera(
        self,
        position: Tuple[float, float, float],
        look_at: Tuple[float, float, float],
        fov: float = 45.0,
    ):
        """Set camera parameters"""
        self.particle_renderer.set_camera(position, look_at, fov=fov)
    
    def set_dof(
        self,
        aperture: float,
        focus_distance: float,
    ):
        """Set depth of field parameters"""
        self.particle_renderer.set_dof(aperture, focus_distance)
    
    def render_frame(
        self,
        positions: np.ndarray,
        colors: np.ndarray,
        velocities: Optional[np.ndarray] = None,
        apply_bloom: bool = True,
        bloom_intensity: float = 2.5,
        bloom_threshold: float = 0.8,
        apply_motion_blur: bool = False,
        motion_blur_strength: float = 0.5,
    ) -> np.ndarray:
        """
        Render a single frame
        
        Args:
            positions: Particle positions (N, 3)
            colors: Particle colors (N, 3)
            velocities: Particle velocities for motion blur (N, 3)
            apply_bloom: Whether to apply bloom effect
            bloom_intensity: Bloom intensity multiplier
            bloom_threshold: Bloom brightness threshold
            apply_motion_blur: Whether to apply motion blur
            motion_blur_strength: Motion blur strength
            
        Returns:
            Rendered image as numpy array (H, W, 3) in [0, 1]
        """
        # Compute per-particle emission from velocity magnitude
        if velocities is not None:
            vel_mag = np.linalg.norm(velocities, axis=1)
            vel_normalized = vel_mag / (vel_mag.max() + 1e-6)
            emissions = self.settings.emission_strength * (0.5 + vel_normalized)
        else:
            emissions = np.full(positions.shape[0], self.settings.emission_strength)
        
        # Set particle data
        self.particle_renderer.set_particles(
            positions, colors,
            radii=np.full(positions.shape[0], self.settings.particle_radius),
            emissions=emissions,
        )
        
        # Render
        image = self.particle_renderer.render()
        
        # Convert to torch for post-processing
        image_tensor = torch.from_numpy(image).to(self.device)
        image_tensor = image_tensor.permute(2, 0, 1).unsqueeze(0)  # (1, 3, H, W)
        
        # Apply bloom
        if apply_bloom:
            image_tensor = self.bloom.apply(
                image_tensor,
                intensity=bloom_intensity,
                threshold=bloom_threshold,
            )
        
        # Apply motion blur
        if apply_motion_blur and self.prev_frame is not None:
            alpha = motion_blur_strength
            image_tensor = (1 - alpha) * image_tensor + alpha * self.prev_frame
        
        self.prev_frame = image_tensor.clone()
        
        # Color grading
        image_tensor = self.color_grading.apply(
            image_tensor,
            exposure=1.2,
            gamma=2.2,
            saturation=1.1,
        )
        
        # Convert back to numpy
        image = image_tensor.squeeze(0).permute(1, 2, 0).cpu().numpy()
        image = np.clip(image, 0, 1)
        
        return image
    
    def render_sequence(
        self,
        positions_sequence: List[np.ndarray],
        colors_sequence: List[np.ndarray],
        velocities_sequence: Optional[List[np.ndarray]] = None,
        camera_positions: Optional[List[Tuple[float, float, float]]] = None,
        **kwargs,
    ) -> List[np.ndarray]:
        """Render a sequence of frames"""
        frames = []
        n_frames = len(positions_sequence)
        
        for i in range(n_frames):
            # Update camera if animated
            if camera_positions is not None:
                look_at = (0.0, 0.0, 0.0)
                self.set_camera(camera_positions[i], look_at)
            
            # Get velocities if available
            velocities = None
            if velocities_sequence is not None:
                velocities = velocities_sequence[i]
            
            # Render frame
            frame = self.render_frame(
                positions_sequence[i],
                colors_sequence[i],
                velocities=velocities,
                **kwargs,
            )
            frames.append(frame)
        
        return frames


class SimpleSoftwareRenderer:
    """
    Fallback software renderer using pure PyTorch
    Used when Taichi is unavailable or for debugging
    """
    
    def __init__(
        self,
        resolution: Tuple[int, int] = (1920, 1080),
        device: torch.device = torch.device("cuda"),
    ):
        self.width, self.height = resolution
        self.device = device
        
        # Camera
        self.camera_pos = torch.tensor([0.0, 0.0, 8.0], device=device)
        self.camera_look_at = torch.tensor([0.0, 0.0, 0.0], device=device)
        self.camera_fov = 45.0
    
    def set_camera(
        self,
        position: Tuple[float, float, float],
        look_at: Tuple[float, float, float],
        fov: float = 45.0,
    ):
        self.camera_pos = torch.tensor(position, device=self.device)
        self.camera_look_at = torch.tensor(look_at, device=self.device)
        self.camera_fov = fov
    
    @torch.no_grad()
    def render(
        self,
        positions: Union[torch.Tensor, np.ndarray],
        colors: Union[torch.Tensor, np.ndarray],
        camera: Optional['Camera'] = None,
        point_size: float = 3.0,
    ) -> np.ndarray:
        """Simple point-based rendering using projection"""
        # Convert inputs to tensor
        if isinstance(positions, np.ndarray):
            positions = torch.from_numpy(positions).to(self.device)
        if isinstance(colors, np.ndarray):
            colors = torch.from_numpy(colors).to(self.device)
            
        # Update camera if provided
        if camera is not None:
            self.set_camera(
                position=tuple(camera.position),
                look_at=tuple(camera.look_at),
                fov=camera.fov
            )

        # Compute view matrix
        forward = (self.camera_look_at - self.camera_pos)
        forward = forward / forward.norm()
        
        up = torch.tensor([0.0, 1.0, 0.0], device=self.device)
        right = torch.linalg.cross(forward, up)
        right = right / right.norm()
        up = torch.linalg.cross(right, forward)
        
        # Project particles
        rel_pos = positions - self.camera_pos
        
        # Camera space coordinates
        x = torch.sum(rel_pos * right, dim=1)
        y = torch.sum(rel_pos * up, dim=1)
        z = torch.sum(rel_pos * forward, dim=1)
        
        # Perspective projection
        fov_rad = self.camera_fov * math.pi / 180.0
        aspect = self.width / self.height
        
        mask = z > 0.1  # Only render particles in front of camera
        
        proj_x = (x / z) / math.tan(fov_rad / 2.0)
        proj_y = (y / z) / math.tan(fov_rad / 2.0) * aspect
        
        # Convert to pixel coordinates
        px = ((proj_x + 1) / 2 * self.width).long()
        py = ((1 - proj_y) / 2 * self.height).long()
        
        # Create image
        image = torch.zeros(3, self.height, self.width, device=self.device)
        depth = torch.full((self.height, self.width), 1e10, device=self.device)
        
        # Simple point splatting
        valid = mask & (px >= 0) & (px < self.width) & (py >= 0) & (py < self.height)
        
        # Vectorized splatting (faster than loop)
        indices = torch.where(valid)[0]
        if len(indices) > 0:
            # Just render center points for speed in this simple renderer
            valid_px = px[indices]
            valid_py = py[indices]
            valid_colors = colors[indices]
            
            # Sort by depth (painter's algorithm)
            valid_z = z[indices]
            sorted_idx = torch.argsort(valid_z, descending=True)
            
            valid_px = valid_px[sorted_idx]
            valid_py = valid_py[sorted_idx]
            valid_colors = valid_colors[sorted_idx]
            
            # Splat
            image[:, valid_py, valid_px] = valid_colors.permute(1, 0)
        
        return image.permute(1, 2, 0).cpu().numpy()  # (H, W, 3)
    
    def _draw_particle_splat(
        self,
        image: torch.Tensor,
        px: int,
        py: int,
        color: torch.Tensor,
        size: int,
        intensity: float = 1.0,
    ):
        """Draw a soft particle splat with glow effect"""
        h, w = image.shape[1], image.shape[2]
        
        for dy in range(-size, size + 1):
            for dx in range(-size, size + 1):
                nx, ny = px + dx, py + dy
                if 0 <= nx < w and 0 <= ny < h:
                    dist = math.sqrt(dx * dx + dy * dy)
                    if dist <= size:
                        # Gaussian falloff for soft edges
                        falloff = math.exp(-dist * dist / (size * size * 0.5))
                        image[:, ny, nx] = torch.clamp(
                            image[:, ny, nx] + color * falloff * intensity,
                            0, 1
                        )


class EnhancedSoftwareRenderer:
    """
    Enhanced software renderer with proper particle visualization
    Renders particles as glowing spheres with depth-based effects
    """
    
    def __init__(
        self,
        resolution: Tuple[int, int] = (1920, 1080),
        device: torch.device = torch.device("cuda"),
        particle_base_size: float = 4.0,
        glow_intensity: float = 1.5,
        background_color: Tuple[float, float, float] = (0.02, 0.02, 0.05),
    ):
        self.width, self.height = resolution
        self.device = device
        self.particle_base_size = particle_base_size
        self.glow_intensity = glow_intensity
        self.background_color = background_color
        
        # Camera
        self.camera_pos = torch.tensor([0.0, 0.0, 8.0], device=device)
        self.camera_look_at = torch.tensor([0.0, 0.0, 0.0], device=device)
        self.camera_fov = 45.0
    
    def set_camera(
        self,
        position: Tuple[float, float, float],
        look_at: Tuple[float, float, float],
        fov: float = 45.0,
    ):
        self.camera_pos = torch.tensor(position, device=self.device)
        self.camera_look_at = torch.tensor(look_at, device=self.device)
        self.camera_fov = fov
    
    @torch.no_grad()
    def render(
        self,
        positions: Union[torch.Tensor, np.ndarray],
        colors: Union[torch.Tensor, np.ndarray],
        camera: Optional['Camera'] = None,
        point_size: float = 4.0,
    ) -> np.ndarray:
        """Render particles with glow effect"""
        # Convert inputs to tensor
        if isinstance(positions, np.ndarray):
            positions = torch.from_numpy(positions).to(self.device)
        if isinstance(colors, np.ndarray):
            colors = torch.from_numpy(colors).to(self.device)
            
        # Update camera if provided
        if camera is not None:
            self.set_camera(
                position=tuple(camera.position),
                look_at=tuple(camera.look_at),
                fov=camera.fov
            )

        # Compute view matrix
        forward = (self.camera_look_at - self.camera_pos)
        forward = forward / (forward.norm() + 1e-8)
        
        up = torch.tensor([0.0, 1.0, 0.0], device=self.device)
        right = torch.linalg.cross(forward, up)
        right = right / (right.norm() + 1e-8)
        up = torch.linalg.cross(right, forward)
        
        # Project particles
        rel_pos = positions - self.camera_pos
        
        # Camera space coordinates
        x = torch.sum(rel_pos * right, dim=1)
        y = torch.sum(rel_pos * up, dim=1)
        z = torch.sum(rel_pos * forward, dim=1)
        
        # Perspective projection
        fov_rad = self.camera_fov * math.pi / 180.0
        aspect = self.width / self.height
        
        mask = z > 0.1  # Only render particles in front of camera
        
        proj_x = (x / (z + 1e-8)) / math.tan(fov_rad / 2.0)
        proj_y = (y / (z + 1e-8)) / math.tan(fov_rad / 2.0) * aspect
        
        # Convert to pixel coordinates
        px = ((proj_x + 1) / 2 * self.width)
        py = ((1 - proj_y) / 2 * self.height)
        
        # Create image with dark background
        bg = torch.tensor(self.background_color, device=self.device).view(3, 1, 1)
        image = bg.expand(3, self.height, self.width).clone()
        
        # Valid particles
        valid = mask & (px >= -50) & (px < self.width + 50) & (py >= -50) & (py < self.height + 50)
        
        # Get valid indices sorted by depth (back to front)
        indices = torch.where(valid)[0]
        if len(indices) > 0:
            valid_z = z[indices]
            sorted_idx = torch.argsort(valid_z, descending=True)
            indices = indices[sorted_idx]
            
            # Limit particles rendered for speed (render closest ones)
            max_particles = min(len(indices), 50000)
            indices = indices[-max_particles:]  # Closest particles
            
            # Render particles with size based on depth
            for idx in indices:
                pxi = int(px[idx].item())
                pyi = int(py[idx].item())
                zi = z[idx].item()
                
                # Size decreases with distance
                size = max(1, int(self.particle_base_size * 5.0 / (zi + 0.5)))
                
                # Intensity based on depth (closer = brighter)
                intensity = min(1.0, 3.0 / (zi + 0.5)) * self.glow_intensity
                
                # Draw particle with glow
                color = colors[idx]
                self._draw_glow(image, pxi, pyi, color, size, intensity)
        
        # Apply simple tone mapping
        image = torch.clamp(image, 0, 1)
        
        return image.permute(1, 2, 0).cpu().numpy()  # (H, W, 3)
    
    def _draw_glow(
        self,
        image: torch.Tensor,
        px: int,
        py: int,
        color: torch.Tensor,
        size: int,
        intensity: float,
    ):
        """Draw a glowing particle"""
        h, w = image.shape[1], image.shape[2]
        
        # Draw larger glow area
        glow_size = size * 2
        for dy in range(-glow_size, glow_size + 1):
            for dx in range(-glow_size, glow_size + 1):
                nx, ny = px + dx, py + dy
                if 0 <= nx < w and 0 <= ny < h:
                    dist_sq = dx * dx + dy * dy
                    if dist_sq <= glow_size * glow_size:
                        # Gaussian falloff
                        sigma_sq = (size * 0.7) ** 2
                        falloff = math.exp(-dist_sq / (2 * sigma_sq))
                        contribution = color * falloff * intensity
                        image[:, ny, nx] = torch.clamp(
                            image[:, ny, nx] + contribution,
                            0, 1.5  # Allow some HDR
                        )
