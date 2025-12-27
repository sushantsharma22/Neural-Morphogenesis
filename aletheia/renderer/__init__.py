"""
Rendering Module for Neural Morphogenesis Engine
Headless ray-tracing with post-processing effects
"""

from aletheia.renderer.raytracer import HeadlessRenderer, ParticleRenderer, SimpleSoftwareRenderer
from aletheia.renderer.effects import (
    BloomEffect,
    DepthOfField,
    MotionBlur,
    ColorGrading,
    BioluminescentPalette,
)
from aletheia.renderer.camera import Camera, CameraAnimator, FlyThroughAnimator

__all__ = [
    # Renderers
    "HeadlessRenderer",
    "ParticleRenderer",
    "SimpleSoftwareRenderer",
    
    # Effects
    "BloomEffect",
    "DepthOfField",
    "MotionBlur",
    "ColorGrading",
    "BioluminescentPalette",
    
    # Camera
    "Camera",
    "CameraAnimator",
    "FlyThroughAnimator",
]
