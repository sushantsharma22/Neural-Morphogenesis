"""
Project Aletheia - Neural Morphogenesis Engine
A PyTorch DDP + Taichi Lang framework for growing organic 3D structures from noise
"""

__version__ = "0.1.0"
__author__ = "Project Aletheia Team"

from aletheia.core.config import AletheiaConfig, load_config
from aletheia.core.distributed import DistributedManager
from aletheia.core.memory import MemoryManager

__all__ = [
    "AletheiaConfig",
    "load_config", 
    "DistributedManager",
    "MemoryManager",
]
