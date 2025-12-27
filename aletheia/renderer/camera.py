"""
Camera System for Neural Morphogenesis Renderer
Handles camera positioning, animation, and projection
"""

import torch
import numpy as np
from typing import Tuple, Optional, List, Callable
import math


class Camera:
    """
    3D Camera for rendering
    Supports perspective and orthographic projections
    """
    
    def __init__(
        self,
        position: Tuple[float, float, float] = (0.0, 0.0, 8.0),
        look_at: Tuple[float, float, float] = (0.0, 0.0, 0.0),
        up: Tuple[float, float, float] = (0.0, 1.0, 0.0),
        fov: float = 45.0,
        aspect_ratio: float = 16/9,
        near: float = 0.1,
        far: float = 100.0,
        projection_type: str = "perspective",
    ):
        self.position = np.array(position, dtype=np.float32)
        self.look_at = np.array(look_at, dtype=np.float32)
        self.up = np.array(up, dtype=np.float32)
        self.fov = fov
        self.aspect_ratio = aspect_ratio
        self.near = near
        self.far = far
        self.projection_type = projection_type
        
        self._update_matrices()
    
    def _update_matrices(self):
        """Update view and projection matrices"""
        self.view_matrix = self._compute_view_matrix()
        self.projection_matrix = self._compute_projection_matrix()
    
    def _compute_view_matrix(self) -> np.ndarray:
        """Compute view matrix (world to camera)"""
        forward = self.look_at - self.position
        forward = forward / np.linalg.norm(forward)
        
        right = np.cross(forward, self.up)
        right = right / np.linalg.norm(right)
        
        up = np.cross(right, forward)
        
        # View matrix (rotation + translation)
        view = np.eye(4, dtype=np.float32)
        view[0, :3] = right
        view[1, :3] = up
        view[2, :3] = -forward
        view[0, 3] = -np.dot(right, self.position)
        view[1, 3] = -np.dot(up, self.position)
        view[2, 3] = np.dot(forward, self.position)
        
        return view
    
    def _compute_projection_matrix(self) -> np.ndarray:
        """Compute projection matrix"""
        if self.projection_type == "perspective":
            return self._perspective_matrix()
        else:
            return self._orthographic_matrix()
    
    def _perspective_matrix(self) -> np.ndarray:
        """Perspective projection matrix"""
        fov_rad = np.radians(self.fov)
        f = 1.0 / np.tan(fov_rad / 2)
        
        proj = np.zeros((4, 4), dtype=np.float32)
        proj[0, 0] = f / self.aspect_ratio
        proj[1, 1] = f
        proj[2, 2] = (self.far + self.near) / (self.near - self.far)
        proj[2, 3] = (2 * self.far * self.near) / (self.near - self.far)
        proj[3, 2] = -1
        
        return proj
    
    def _orthographic_matrix(self) -> np.ndarray:
        """Orthographic projection matrix"""
        size = 5.0  # Half-size of orthographic view
        
        proj = np.zeros((4, 4), dtype=np.float32)
        proj[0, 0] = 1 / (size * self.aspect_ratio)
        proj[1, 1] = 1 / size
        proj[2, 2] = -2 / (self.far - self.near)
        proj[2, 3] = -(self.far + self.near) / (self.far - self.near)
        proj[3, 3] = 1
        
        return proj
    
    def set_position(self, position: Tuple[float, float, float]):
        """Set camera position"""
        self.position = np.array(position, dtype=np.float32)
        self._update_matrices()
    
    def set_look_at(self, look_at: Tuple[float, float, float]):
        """Set camera look-at point"""
        self.look_at = np.array(look_at, dtype=np.float32)
        self._update_matrices()
    
    def set_fov(self, fov: float):
        """Set field of view"""
        self.fov = fov
        self._update_matrices()
    
    def get_ray_direction(
        self,
        x: float,
        y: float,
    ) -> np.ndarray:
        """
        Get ray direction for pixel coordinates
        
        Args:
            x: Normalized x coordinate [-1, 1]
            y: Normalized y coordinate [-1, 1]
            
        Returns:
            Ray direction vector (3,)
        """
        # Compute camera basis
        forward = self.look_at - self.position
        forward = forward / np.linalg.norm(forward)
        
        right = np.cross(forward, self.up)
        right = right / np.linalg.norm(right)
        
        up = np.cross(right, forward)
        
        # Compute ray direction
        fov_rad = np.radians(self.fov)
        h = np.tan(fov_rad / 2)
        w = h * self.aspect_ratio
        
        direction = forward + x * w * right + y * h * up
        direction = direction / np.linalg.norm(direction)
        
        return direction
    
    def project_point(
        self,
        point: np.ndarray,
    ) -> np.ndarray:
        """
        Project 3D point to screen coordinates
        
        Args:
            point: 3D point (3,) or (N, 3)
            
        Returns:
            Screen coordinates (2,) or (N, 2) in [-1, 1]
        """
        if point.ndim == 1:
            point = point.reshape(1, 3)
        
        # Convert to homogeneous coordinates
        ones = np.ones((point.shape[0], 1), dtype=np.float32)
        point_h = np.concatenate([point, ones], axis=1)
        
        # Apply view and projection
        mvp = self.projection_matrix @ self.view_matrix
        projected = point_h @ mvp.T
        
        # Perspective divide
        projected = projected[:, :2] / projected[:, 3:4]
        
        if projected.shape[0] == 1:
            return projected[0]
        return projected
    
    def to_dict(self) -> dict:
        """Export camera parameters to dictionary"""
        return {
            "position": self.position.tolist(),
            "look_at": self.look_at.tolist(),
            "up": self.up.tolist(),
            "fov": self.fov,
            "aspect_ratio": self.aspect_ratio,
            "near": self.near,
            "far": self.far,
            "projection_type": self.projection_type,
        }
    
    @classmethod
    def from_dict(cls, data: dict) -> 'Camera':
        """Create camera from dictionary"""
        return cls(
            position=tuple(data["position"]),
            look_at=tuple(data["look_at"]),
            up=tuple(data.get("up", [0, 1, 0])),
            fov=data.get("fov", 45.0),
            aspect_ratio=data.get("aspect_ratio", 16/9),
            near=data.get("near", 0.1),
            far=data.get("far", 100.0),
            projection_type=data.get("projection_type", "perspective"),
        )


class CameraAnimator:
    """
    Animates camera along predefined paths
    Supports orbit, dolly, and custom keyframe animations
    """
    
    def __init__(
        self,
        camera: Camera,
        total_frames: int = 1000,
    ):
        self.camera = camera
        self.total_frames = total_frames
        
        # Animation state
        self.current_frame = 0
        self.animation_type = "orbit"
        self.keyframes: List[dict] = []
    
    def set_orbit_animation(
        self,
        center: Tuple[float, float, float] = (0.0, 0.0, 0.0),
        radius: float = 8.0,
        height: float = 0.0,
        speed: float = 1.0,
        vertical_oscillation: float = 0.0,
    ):
        """
        Set up orbital camera animation
        
        Args:
            center: Orbit center point
            radius: Orbit radius
            height: Camera height above center
            speed: Rotation speed (revolutions per total_frames)
            vertical_oscillation: Vertical bob amount
        """
        self.animation_type = "orbit"
        self.orbit_params = {
            "center": np.array(center, dtype=np.float32),
            "radius": radius,
            "height": height,
            "speed": speed,
            "vertical_oscillation": vertical_oscillation,
        }
    
    def set_dolly_animation(
        self,
        start_position: Tuple[float, float, float],
        end_position: Tuple[float, float, float],
        look_at: Tuple[float, float, float] = (0.0, 0.0, 0.0),
        ease_type: str = "smooth",
    ):
        """
        Set up dolly (linear movement) animation
        
        Args:
            start_position: Starting camera position
            end_position: Ending camera position
            look_at: Point to look at throughout
            ease_type: Easing function type
        """
        self.animation_type = "dolly"
        self.dolly_params = {
            "start": np.array(start_position, dtype=np.float32),
            "end": np.array(end_position, dtype=np.float32),
            "look_at": np.array(look_at, dtype=np.float32),
            "ease_type": ease_type,
        }
    
    def add_keyframe(
        self,
        frame: int,
        position: Tuple[float, float, float],
        look_at: Tuple[float, float, float],
        fov: Optional[float] = None,
    ):
        """Add keyframe for custom animation"""
        self.animation_type = "keyframe"
        self.keyframes.append({
            "frame": frame,
            "position": np.array(position, dtype=np.float32),
            "look_at": np.array(look_at, dtype=np.float32),
            "fov": fov,
        })
        self.keyframes.sort(key=lambda k: k["frame"])
    
    def update(self, frame: int) -> Camera:
        """
        Update camera for given frame
        
        Args:
            frame: Current frame number
            
        Returns:
            Updated camera
        """
        self.current_frame = frame
        t = frame / self.total_frames  # Normalized time [0, 1]
        
        if self.animation_type == "orbit":
            self._update_orbit(t)
        elif self.animation_type == "dolly":
            self._update_dolly(t)
        elif self.animation_type == "keyframe":
            self._update_keyframe(frame)
        
        return self.camera
    
    def _update_orbit(self, t: float):
        """Update camera position for orbital animation"""
        params = self.orbit_params
        
        angle = t * 2 * math.pi * params["speed"]
        
        x = params["center"][0] + params["radius"] * math.cos(angle)
        z = params["center"][2] + params["radius"] * math.sin(angle)
        y = params["center"][1] + params["height"] + \
            params["vertical_oscillation"] * math.sin(t * 4 * math.pi)
        
        self.camera.set_position((x, y, z))
        self.camera.set_look_at(tuple(params["center"]))
    
    def _update_dolly(self, t: float):
        """Update camera position for dolly animation"""
        params = self.dolly_params
        
        # Apply easing
        t_eased = self._ease(t, params["ease_type"])
        
        position = params["start"] + t_eased * (params["end"] - params["start"])
        
        self.camera.set_position(tuple(position))
        self.camera.set_look_at(tuple(params["look_at"]))
    
    def _update_keyframe(self, frame: int):
        """Update camera from keyframes"""
        if not self.keyframes:
            return
        
        # Find surrounding keyframes
        prev_kf = self.keyframes[0]
        next_kf = self.keyframes[-1]
        
        for i, kf in enumerate(self.keyframes):
            if kf["frame"] <= frame:
                prev_kf = kf
            if kf["frame"] >= frame and i < len(self.keyframes):
                next_kf = kf
                break
        
        # Interpolate
        if prev_kf["frame"] == next_kf["frame"]:
            t = 0
        else:
            t = (frame - prev_kf["frame"]) / (next_kf["frame"] - prev_kf["frame"])
        
        t = self._ease(t, "smooth")
        
        position = prev_kf["position"] + t * (next_kf["position"] - prev_kf["position"])
        look_at = prev_kf["look_at"] + t * (next_kf["look_at"] - prev_kf["look_at"])
        
        self.camera.set_position(tuple(position))
        self.camera.set_look_at(tuple(look_at))
        
        if prev_kf["fov"] is not None and next_kf["fov"] is not None:
            fov = prev_kf["fov"] + t * (next_kf["fov"] - prev_kf["fov"])
            self.camera.set_fov(fov)
    
    def _ease(self, t: float, ease_type: str) -> float:
        """Apply easing function"""
        if ease_type == "linear":
            return t
        elif ease_type == "smooth":
            # Smoothstep
            return t * t * (3 - 2 * t)
        elif ease_type == "smoother":
            # Smootherstep
            return t * t * t * (t * (t * 6 - 15) + 10)
        elif ease_type == "ease_in":
            return t * t
        elif ease_type == "ease_out":
            return 1 - (1 - t) * (1 - t)
        elif ease_type == "ease_in_out":
            if t < 0.5:
                return 2 * t * t
            else:
                return 1 - 2 * (1 - t) * (1 - t)
        else:
            return t
    
    def get_camera_path(
        self,
        num_points: int = 100,
    ) -> np.ndarray:
        """
        Get camera path as array of positions
        Useful for visualization/debugging
        
        Returns:
            Array of positions (num_points, 3)
        """
        positions = []
        
        for i in range(num_points):
            frame = int(i * self.total_frames / num_points)
            self.update(frame)
            positions.append(self.camera.position.copy())
        
        return np.array(positions)


class FlyThroughAnimator(CameraAnimator):
    """
    Specialized animator for fly-through sequences
    Follows particle density or custom paths
    """
    
    def __init__(
        self,
        camera: Camera,
        total_frames: int = 1000,
    ):
        super().__init__(camera, total_frames)
        self.path_points: List[np.ndarray] = []
        self.look_at_points: List[np.ndarray] = []
    
    def generate_path_through_density(
        self,
        particle_positions: np.ndarray,
        num_waypoints: int = 10,
        margin: float = 0.5,
    ):
        """
        Generate camera path through particle cloud
        
        Args:
            particle_positions: Particle positions (N, 3)
            num_waypoints: Number of path waypoints
            margin: Distance margin from particles
        """
        # Compute principal axes of particle distribution
        center = particle_positions.mean(axis=0)
        centered = particle_positions - center
        cov = np.cov(centered.T)
        eigenvalues, eigenvectors = np.linalg.eigh(cov)
        
        # Sort by eigenvalue (largest first)
        order = np.argsort(eigenvalues)[::-1]
        principal_axis = eigenvectors[:, order[0]]
        
        # Generate spiral path along principal axis
        radius = np.sqrt(eigenvalues[order[1]]) * 2 + margin
        length = np.sqrt(eigenvalues[order[0]]) * 3
        
        for i in range(num_waypoints):
            t = i / (num_waypoints - 1)
            
            angle = t * 2 * math.pi * 2  # Two full rotations
            
            # Position along spiral
            pos = center + \
                  principal_axis * (t - 0.5) * length + \
                  eigenvectors[:, order[1]] * radius * math.cos(angle) + \
                  eigenvectors[:, order[2]] * radius * math.sin(angle)
            
            self.path_points.append(pos)
            
            # Look towards center with offset
            look_offset = principal_axis * (t - 0.3) * length * 0.5
            self.look_at_points.append(center + look_offset)
        
        # Convert to keyframe animation
        self.animation_type = "keyframe"
        self.keyframes = []
        
        for i, (pos, look) in enumerate(zip(self.path_points, self.look_at_points)):
            frame = int(i * self.total_frames / (num_waypoints - 1))
            self.add_keyframe(frame, tuple(pos), tuple(look))
