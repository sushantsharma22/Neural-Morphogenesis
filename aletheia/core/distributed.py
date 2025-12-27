"""
Distributed Data Parallel (DDP) Management for Project Aletheia
Handles multi-GPU setup, synchronization, and communication
"""

import os
import functools
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from typing import Optional, Callable, Any, List, Tuple


class DistributedManager:
    """
    Manages distributed training/inference across multiple GPUs
    Optimized for 4x NVIDIA A16 setup
    """
    
    def __init__(
        self,
        backend: str = "nccl",
        init_method: str = "env://",
        world_size: Optional[int] = None,
        rank: Optional[int] = None,
    ):
        self.backend = backend
        self.init_method = init_method
        self._world_size = world_size if world_size is not None else 1
        self._rank = rank if rank is not None else 0
        self._local_rank = 0
        self._initialized = False
        self._device: Optional[torch.device] = None
        
    @property
    def is_initialized(self) -> bool:
        return self._initialized
    
    @property
    def world_size(self) -> int:
        if self._initialized:
            return dist.get_world_size()
        return self._world_size
    
    @property
    def rank(self) -> int:
        if self._initialized:
            return dist.get_rank()
        return self._rank
    
    @property
    def local_rank(self) -> int:
        return self._local_rank
    
    @property
    def is_main_process(self) -> bool:
        return self.rank == 0
    
    @property
    def device(self) -> torch.device:
        return self._device or torch.device("cpu")
    
    def initialize(self) -> bool:
        """Initialize distributed environment"""
        if self._initialized:
            return True
        
        # Check for distributed environment variables
        if "WORLD_SIZE" in os.environ:
            self._world_size = int(os.environ["WORLD_SIZE"])
        if "RANK" in os.environ:
            self._rank = int(os.environ["RANK"])
        if "LOCAL_RANK" in os.environ:
            self._local_rank = int(os.environ["LOCAL_RANK"])
        
        # Single GPU fallback
        if self._world_size <= 1:
            self._world_size = 1
            self._rank = 0
            self._local_rank = 0
            self._device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
            print(f"[DDP] Running in single-GPU mode on {self._device}")
            return True
        
        # Initialize process group
        if not dist.is_initialized():
            dist.init_process_group(
                backend=self.backend,
                init_method=self.init_method,
                world_size=self._world_size,
                rank=self._rank,
            )
        
        # Set device
        if torch.cuda.is_available():
            torch.cuda.set_device(self._local_rank)
            self._device = torch.device(f"cuda:{self._local_rank}")
        else:
            self._device = torch.device("cpu")
        
        self._initialized = True
        
        if self.is_main_process:
            print(f"[DDP] Initialized with {self._world_size} processes")
            print(f"[DDP] Backend: {self.backend}")
            
        return True
    
    def cleanup(self):
        """Cleanup distributed environment"""
        if self._initialized and dist.is_initialized():
            dist.destroy_process_group()
            self._initialized = False
    
    def wrap_model(
        self,
        model: torch.nn.Module,
        find_unused_parameters: bool = False,
        broadcast_buffers: bool = True,
    ) -> torch.nn.Module:
        """Wrap model with DDP"""
        model = model.to(self._device)
        
        if self._world_size > 1 and self._initialized:
            model = DDP(
                model,
                device_ids=[self._local_rank],
                output_device=self._local_rank,
                find_unused_parameters=find_unused_parameters,
                broadcast_buffers=broadcast_buffers,
            )
        
        return model
    
    def barrier(self):
        """Synchronization barrier across all processes"""
        if self._initialized and dist.is_initialized():
            dist.barrier()
    
    def all_reduce(
        self,
        tensor: torch.Tensor,
        op: Any = None,
    ) -> torch.Tensor:
        """All-reduce operation across all processes"""
        if op is None:
            op = dist.ReduceOp.SUM
        if self._world_size > 1 and self._initialized:
            dist.all_reduce(tensor, op=op)
        return tensor
    
    def all_gather(
        self,
        tensor: torch.Tensor,
    ) -> torch.Tensor:
        """All-gather operation across all processes"""
        if self._world_size > 1 and self._initialized:
            tensor_list = [torch.zeros_like(tensor) for _ in range(self._world_size)]
            dist.all_gather(tensor_list, tensor)
            return torch.cat(tensor_list, dim=0)
        return tensor
    
    def broadcast(
        self,
        tensor: torch.Tensor,
        src: int = 0,
    ) -> torch.Tensor:
        """Broadcast tensor from source rank to all processes"""
        if self._world_size > 1 and self._initialized:
            dist.broadcast(tensor, src=src)
        return tensor
    
    def reduce_mean(self, tensor: torch.Tensor) -> torch.Tensor:
        """Reduce tensor by mean across all processes"""
        if self._world_size > 1 and self._initialized:
            dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
            tensor = tensor / self._world_size
        return tensor
    
    def get_partition_bounds(
        self,
        total_size: int,
    ) -> Tuple[int, int]:
        """
        Get start and end indices for data partitioning
        Used for distributing particles across GPUs
        """
        chunk_size = total_size // self._world_size
        remainder = total_size % self._world_size
        
        start = self._rank * chunk_size + min(self._rank, remainder)
        end = start + chunk_size + (1 if self._rank < remainder else 0)
        
        return start, end
    
    def partition_data(
        self,
        data: torch.Tensor,
        dim: int = 0,
    ) -> torch.Tensor:
        """Partition data tensor across processes"""
        total_size = data.shape[dim]
        start, end = self.get_partition_bounds(total_size)
        
        indices = torch.arange(start, end, device=data.device)
        return torch.index_select(data, dim, indices)
    
    def gather_data(
        self,
        data: torch.Tensor,
        dim: int = 0,
    ) -> Optional[torch.Tensor]:
        """Gather partitioned data back to main process"""
        if self._world_size <= 1:
            return data
        
        gathered = [torch.zeros_like(data) for _ in range(self._world_size)]
        dist.all_gather(gathered, data)
        
        return torch.cat(gathered, dim=dim)


def main_process_only(func: Callable) -> Callable:
    """Decorator to run function only on main process"""
    @functools.wraps(func)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        rank = int(os.environ.get("RANK", 0))
        if rank == 0:
            return func(*args, **kwargs)
        return None
    return wrapper


def synchronized(func: Callable) -> Callable:
    """Decorator to synchronize function across all processes"""
    @functools.wraps(func)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        result = func(*args, **kwargs)
        if dist.is_initialized():
            dist.barrier()
        return result
    return wrapper


class GradientSynchronizer:
    """
    Handles gradient synchronization for custom operations
    outside of standard DDP autograd
    """
    
    def __init__(self, world_size: int):
        self.world_size = world_size
    
    def sync_gradients(self, model: torch.nn.Module):
        """Synchronize gradients across all processes"""
        if self.world_size <= 1:
            return
        
        for param in model.parameters():
            if param.grad is not None:
                dist.all_reduce(param.grad, op=dist.ReduceOp.SUM)
                param.grad /= self.world_size


class DistributedSampler:
    """
    Custom distributed sampler for particle data
    Ensures each GPU processes unique particle subsets
    """
    
    def __init__(
        self,
        total_size: int,
        world_size: int,
        rank: int,
        shuffle: bool = False,
        seed: int = 42,
    ):
        self.total_size = total_size
        self.world_size = world_size
        self.rank = rank
        self.shuffle = shuffle
        self.seed = seed
        self.epoch = 0
        
        # Calculate local size
        self.local_size = total_size // world_size
        if rank < total_size % world_size:
            self.local_size += 1
    
    def set_epoch(self, epoch: int):
        """Set epoch for shuffling reproducibility"""
        self.epoch = epoch
    
    def __iter__(self):
        """Generate indices for this rank"""
        if self.shuffle:
            g = torch.Generator()
            g.manual_seed(self.seed + self.epoch)
            indices = torch.randperm(self.total_size, generator=g).tolist()
        else:
            indices = list(range(self.total_size))
        
        # Partition indices for this rank
        start = self.rank * (self.total_size // self.world_size)
        end = start + self.local_size
        
        return iter(indices[start:end])
    
    def __len__(self):
        return self.local_size


def setup_for_distributed(is_master: bool):
    """
    Setup print functions for distributed training
    Suppresses output on non-master processes
    """
    import builtins as __builtin__
    
    builtin_print = __builtin__.print
    
    def print(*args: Any, **kwargs: Any):
        force = kwargs.pop('force', False)
        if is_master or force:
            builtin_print(*args, **kwargs)
    
    __builtin__.print = print


def get_world_size() -> int:
    """Get world size, defaulting to 1 if not distributed"""
    if dist.is_initialized():
        return dist.get_world_size()
    return 1


def get_rank() -> int:
    """Get rank, defaulting to 0 if not distributed"""
    if dist.is_initialized():
        return dist.get_rank()
    return 0


def is_main_process() -> bool:
    """Check if current process is main"""
    return get_rank() == 0
