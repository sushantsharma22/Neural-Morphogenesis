"""
GPU Memory Management for Project Aletheia
Strict memory control for shared server environments with 4x A16 GPUs
"""

import gc
import torch
import numpy as np
from typing import Optional, Dict, List, Callable, Any
from dataclasses import dataclass
from contextlib import contextmanager
import threading
import warnings

try:
    import pynvml as _pynvml  # type: ignore[import-untyped]
    PYNVML_AVAILABLE = True
except ImportError:
    _pynvml = None
    PYNVML_AVAILABLE = False

# Type-safe wrapper module - only accessed when PYNVML_AVAILABLE is True
class _PynvmlWrapper:
    """Type-safe wrapper for pynvml to avoid None attribute errors"""
    @staticmethod
    def nvmlInit() -> None:
        if _pynvml is not None:
            _pynvml.nvmlInit()
    
    @staticmethod
    def nvmlDeviceGetHandleByIndex(index: int) -> Any:
        if _pynvml is not None:
            return _pynvml.nvmlDeviceGetHandleByIndex(index)
        return None
    
    @staticmethod
    def nvmlDeviceGetMemoryInfo(handle: Any) -> Any:
        if _pynvml is not None:
            return _pynvml.nvmlDeviceGetMemoryInfo(handle)
        return None

pynvml = _PynvmlWrapper()


@dataclass
class MemoryStats:
    """Memory statistics container"""
    allocated_gb: float
    reserved_gb: float
    free_gb: float
    total_gb: float
    utilization: float


class MemoryManager:
    """
    Manages GPU memory allocation and monitoring
    Critical for shared server environments
    """
    
    # Per-GPU memory limit (A16 has 16GB, we use 15GB for safety)
    DEFAULT_MAX_MEMORY_GB = 15.0
    
    def __init__(
        self,
        max_memory_gb: float = DEFAULT_MAX_MEMORY_GB,
        device_id: int = 0,
        enable_monitoring: bool = True,
    ):
        self.max_memory_gb = max_memory_gb
        self.max_memory_bytes = int(max_memory_gb * 1024**3)
        self.device_id = device_id
        self.device = torch.device(f"cuda:{device_id}")
        self.enable_monitoring = enable_monitoring
        
        self._allocation_history: List[float] = []
        self._warning_threshold = 0.85
        self._critical_threshold = 0.95
        
        # Initialize NVML if available
        self._nvml_initialized = False
        if PYNVML_AVAILABLE and enable_monitoring:
            try:
                pynvml.nvmlInit()
                self._nvml_initialized = True
            except Exception:
                pass
        
        # Set memory fraction limit
        self._set_memory_limit()
    
    def _set_memory_limit(self):
        """Set CUDA memory allocation limit"""
        if torch.cuda.is_available():
            # Calculate fraction based on max_memory_gb
            total_memory = torch.cuda.get_device_properties(self.device_id).total_memory
            fraction = min(self.max_memory_bytes / total_memory, 0.95)
            
            try:
                torch.cuda.set_per_process_memory_fraction(fraction, self.device_id)
            except RuntimeError:
                # Memory fraction can only be set before CUDA initialization
                pass
    
    def get_memory_stats(self) -> MemoryStats:
        """Get current memory statistics"""
        if not torch.cuda.is_available():
            return MemoryStats(0, 0, 0, 0, 0)
        
        allocated = torch.cuda.memory_allocated(self.device_id)
        reserved = torch.cuda.memory_reserved(self.device_id)
        
        if self._nvml_initialized:
            try:
                handle = pynvml.nvmlDeviceGetHandleByIndex(self.device_id)
                info = pynvml.nvmlDeviceGetMemoryInfo(handle)
                total = info.total
                free = info.free
            except Exception:
                props = torch.cuda.get_device_properties(self.device_id)
                total = props.total_memory
                free = total - reserved
        else:
            props = torch.cuda.get_device_properties(self.device_id)
            total = props.total_memory
            free = total - reserved
        
        return MemoryStats(
            allocated_gb=allocated / 1024**3,
            reserved_gb=reserved / 1024**3,
            free_gb=free / 1024**3,
            total_gb=total / 1024**3,
            utilization=allocated / total if total > 0 else 0,
        )
    
    def check_available_memory(self, required_gb: float) -> bool:
        """Check if required memory is available"""
        stats = self.get_memory_stats()
        available = self.max_memory_gb - stats.allocated_gb
        return available >= required_gb
    
    def clear_cache(self):
        """Clear CUDA memory cache"""
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            gc.collect()
    
    def synchronize(self):
        """Synchronize CUDA operations"""
        if torch.cuda.is_available():
            torch.cuda.synchronize(self.device_id)
    
    @contextmanager
    def memory_checkpoint(self, name: str = "checkpoint"):
        """Context manager for memory checkpointing"""
        start_stats = self.get_memory_stats()
        
        try:
            yield
        finally:
            end_stats = self.get_memory_stats()
            delta = end_stats.allocated_gb - start_stats.allocated_gb
            
            if delta > 0.1:  # Log significant allocations
                self._allocation_history.append(delta)
                
                if end_stats.utilization > self._critical_threshold:
                    warnings.warn(
                        f"[Memory] Critical usage after {name}: "
                        f"{end_stats.utilization:.1%} ({end_stats.allocated_gb:.2f}GB)"
                    )
    
    def estimate_particle_memory(
        self,
        num_particles: int,
        bytes_per_particle: int = 64,  # position(12) + velocity(12) + properties(40)
    ) -> float:
        """Estimate memory required for particles in GB"""
        return (num_particles * bytes_per_particle) / 1024**3
    
    def get_optimal_chunk_size(
        self,
        total_particles: int,
        bytes_per_particle: int = 64,
        target_utilization: float = 0.7,
    ) -> int:
        """Calculate optimal chunk size for processing"""
        stats = self.get_memory_stats()
        available_gb = (self.max_memory_gb - stats.allocated_gb) * target_utilization
        available_bytes = available_gb * 1024**3
        
        max_particles = int(available_bytes / bytes_per_particle)
        return min(max_particles, total_particles)
    
    def optimize_batch_size(
        self,
        model: torch.nn.Module,
        sample_input: torch.Tensor,
        max_batch_size: int = 64,
        target_memory_usage: float = 0.8,
    ) -> int:
        """Find optimal batch size through binary search"""
        
        def test_batch_size(batch_size: int) -> bool:
            try:
                self.clear_cache()
                batch = sample_input.repeat(batch_size, *([1] * (sample_input.dim() - 1)))
                batch = batch.to(self.device)
                
                with torch.no_grad():
                    _ = model(batch)
                
                stats = self.get_memory_stats()
                return stats.utilization < target_memory_usage
            except RuntimeError:
                return False
        
        # Binary search for optimal batch size
        low, high = 1, max_batch_size
        optimal = 1
        
        while low <= high:
            mid = (low + high) // 2
            if test_batch_size(mid):
                optimal = mid
                low = mid + 1
            else:
                high = mid - 1
        
        self.clear_cache()
        return optimal


class GradientCheckpointer:
    """
    Implements gradient checkpointing for memory-efficient training
    Essential for fitting large models on A16's 16GB
    """
    
    def __init__(self, segments: int = 4):
        self.segments = segments
    
    def checkpoint_sequential(
        self,
        functions: List[Callable],
        input_tensor: torch.Tensor,
    ) -> torch.Tensor:
        """Apply gradient checkpointing to sequential functions"""
        from torch.utils.checkpoint import checkpoint_sequential
        return checkpoint_sequential(functions, self.segments, input_tensor)
    
    @staticmethod
    def checkpoint(
        function: Callable,
        *args,
        use_reentrant: bool = False,
    ) -> Any:
        """Checkpoint a single function"""
        from torch.utils.checkpoint import checkpoint
        return checkpoint(function, *args, use_reentrant=use_reentrant)


class MemoryEfficientAttention:
    """
    Memory-efficient attention implementations
    Uses chunked computation to reduce peak memory
    """
    
    @staticmethod
    def chunked_attention(
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        chunk_size: int = 1024,
    ) -> torch.Tensor:
        """
        Compute attention in chunks to reduce memory usage
        Shape: (batch, heads, seq_len, dim)
        """
        batch, heads, seq_len, dim = query.shape
        scale = dim ** -0.5
        
        # Process in chunks
        outputs = []
        for i in range(0, seq_len, chunk_size):
            end = min(i + chunk_size, seq_len)
            q_chunk = query[:, :, i:end, :]
            
            # Compute attention scores for chunk
            attn_scores = torch.matmul(q_chunk, key.transpose(-2, -1)) * scale
            attn_probs = torch.softmax(attn_scores, dim=-1)
            
            # Compute output for chunk
            chunk_output = torch.matmul(attn_probs, value)
            outputs.append(chunk_output)
        
        return torch.cat(outputs, dim=2)


class OOMHandler:
    """
    Handles Out-of-Memory situations gracefully
    Implements retry logic with reduced memory footprint
    """
    
    def __init__(
        self,
        memory_manager: MemoryManager,
        max_retries: int = 3,
        reduction_factor: float = 0.8,
    ):
        self.memory_manager = memory_manager
        self.max_retries = max_retries
        self.reduction_factor = reduction_factor
    
    def execute_with_retry(
        self,
        func: Callable,
        *args,
        chunk_size_param: str = "chunk_size",
        initial_chunk_size: int = 500000,
        **kwargs,
    ) -> Any:
        """Execute function with OOM retry logic"""
        chunk_size = initial_chunk_size
        
        for attempt in range(self.max_retries):
            try:
                kwargs[chunk_size_param] = chunk_size
                result = func(*args, **kwargs)
                return result
                
            except RuntimeError as e:
                if "out of memory" in str(e).lower():
                    self.memory_manager.clear_cache()
                    chunk_size = int(chunk_size * self.reduction_factor)
                    
                    if attempt < self.max_retries - 1:
                        print(f"[OOM] Retry {attempt + 1}/{self.max_retries}, "
                              f"reducing chunk size to {chunk_size}")
                    else:
                        raise RuntimeError(
                            f"Out of memory after {self.max_retries} retries"
                        )
                else:
                    raise


class TensorPool:
    """
    Tensor pooling for memory reuse
    Reduces allocation overhead for frequently used tensor shapes
    """
    
    def __init__(self, device: torch.device):
        self.device = device
        self._pools: Dict[tuple, List[torch.Tensor]] = {}
        self._lock = threading.Lock()
    
    def get_tensor(
        self,
        shape: tuple,
        dtype: torch.dtype = torch.float32,
    ) -> torch.Tensor:
        """Get a tensor from pool or create new one"""
        key = (shape, dtype)
        
        with self._lock:
            if key in self._pools and self._pools[key]:
                tensor = self._pools[key].pop()
                tensor.zero_()
                return tensor
        
        return torch.zeros(shape, dtype=dtype, device=self.device)
    
    def return_tensor(self, tensor: torch.Tensor):
        """Return tensor to pool for reuse"""
        key = (tuple(tensor.shape), tensor.dtype)
        
        with self._lock:
            if key not in self._pools:
                self._pools[key] = []
            
            # Limit pool size to prevent memory bloat
            if len(self._pools[key]) < 10:
                self._pools[key].append(tensor.detach())
    
    def clear(self):
        """Clear all pools"""
        with self._lock:
            self._pools.clear()


def estimate_model_memory(model: torch.nn.Module) -> Dict[str, float]:
    """Estimate memory usage of a model"""
    param_memory = sum(p.numel() * p.element_size() for p in model.parameters())
    buffer_memory = sum(b.numel() * b.element_size() for b in model.buffers())
    
    # Gradient memory (same as parameters for standard training)
    grad_memory = param_memory
    
    return {
        "parameters_gb": param_memory / 1024**3,
        "buffers_gb": buffer_memory / 1024**3,
        "gradients_gb": grad_memory / 1024**3,
        "total_gb": (param_memory + buffer_memory + grad_memory) / 1024**3,
    }


def get_memory_summary(device_id: int = 0) -> str:
    """Get formatted memory summary string"""
    if not torch.cuda.is_available():
        return "CUDA not available"
    
    allocated = torch.cuda.memory_allocated(device_id) / 1024**3
    reserved = torch.cuda.memory_reserved(device_id) / 1024**3
    total = torch.cuda.get_device_properties(device_id).total_memory / 1024**3
    
    return (
        f"GPU {device_id}: "
        f"Allocated: {allocated:.2f}GB | "
        f"Reserved: {reserved:.2f}GB | "
        f"Total: {total:.2f}GB | "
        f"Usage: {allocated/total:.1%}"
    )
