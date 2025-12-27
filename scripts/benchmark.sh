#!/bin/bash
#===============================================================================
# Project Aletheia: Neural Morphogenesis Engine
# Benchmark Script
#
# Tests performance across different configurations
#===============================================================================

set -e

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_ROOT="$( cd "$SCRIPT_DIR/.." && pwd )"

cd "$PROJECT_ROOT"

# Activate virtual environment if it exists
if [ -f "$PROJECT_ROOT/venv/bin/activate" ]; then
    source "$PROJECT_ROOT/venv/bin/activate"
fi

echo "=============================================="
echo "  Project Aletheia: Performance Benchmark"
echo "=============================================="
echo ""

# Get GPU info
echo "GPU Configuration:"
nvidia-smi --query-gpu=name,memory.total,compute_cap --format=csv
echo ""

# Run benchmark
python << 'EOF'
import torch
import time
import numpy as np
import sys

def benchmark_pytorch():
    """Benchmark basic PyTorch operations"""
    print("PyTorch Benchmarks")
    print("-" * 40)
    
    if not torch.cuda.is_available():
        print("CUDA not available, skipping GPU benchmarks")
        return
    
    device = torch.device("cuda:0")
    
    # Memory bandwidth test
    sizes = [1024, 2048, 4096, 8192]
    print("\nMemory Bandwidth (GB/s):")
    
    for size in sizes:
        a = torch.randn(size, size, device=device)
        b = torch.randn(size, size, device=device)
        
        # Warmup
        c = a + b
        torch.cuda.synchronize()
        
        # Benchmark
        start = time.time()
        for _ in range(10):
            c = a + b
        torch.cuda.synchronize()
        elapsed = time.time() - start
        
        bytes_moved = 3 * size * size * 4 * 10  # 3 tensors, float32, 10 iterations
        bandwidth = bytes_moved / elapsed / 1e9
        print(f"  {size}x{size}: {bandwidth:.1f} GB/s")
    
    # Matmul throughput
    print("\nMatmul Throughput (TFLOPS):")
    
    for size in [1024, 2048, 4096]:
        a = torch.randn(size, size, device=device)
        b = torch.randn(size, size, device=device)
        
        # Warmup
        c = torch.matmul(a, b)
        torch.cuda.synchronize()
        
        # Benchmark
        start = time.time()
        for _ in range(10):
            c = torch.matmul(a, b)
        torch.cuda.synchronize()
        elapsed = time.time() - start
        
        flops = 2 * size ** 3 * 10  # matmul FLOPs
        tflops = flops / elapsed / 1e12
        print(f"  {size}x{size}: {tflops:.2f} TFLOPS")
    
    # Multi-GPU scaling
    gpu_count = torch.cuda.device_count()
    if gpu_count > 1:
        print(f"\nMulti-GPU Scaling ({gpu_count} GPUs):")
        
        size = 4096
        single_gpu_time = 0
        
        for num_gpus in range(1, gpu_count + 1):
            tensors = []
            for i in range(num_gpus):
                a = torch.randn(size, size, device=f"cuda:{i}")
                b = torch.randn(size, size, device=f"cuda:{i}")
                tensors.append((a, b))
            
            # Warmup
            results = [torch.matmul(a, b) for a, b in tensors]
            for i in range(num_gpus):
                torch.cuda.synchronize(i)
            
            # Benchmark
            start = time.time()
            for _ in range(10):
                results = [torch.matmul(a, b) for a, b in tensors]
            for i in range(num_gpus):
                torch.cuda.synchronize(i)
            elapsed = time.time() - start
            
            if num_gpus == 1:
                single_gpu_time = elapsed
            
            speedup = single_gpu_time / elapsed * num_gpus
            efficiency = speedup / num_gpus * 100
            
            print(f"  {num_gpus} GPU(s): {elapsed:.3f}s, Speedup: {speedup:.2f}x, Efficiency: {efficiency:.1f}%")

def benchmark_taichi():
    """Benchmark Taichi operations"""
    print("\n" + "=" * 40)
    print("Taichi Benchmarks")
    print("-" * 40)
    
    try:
        import taichi as ti
        ti.init(arch=ti.cuda, device_memory_GB=4)
        
        # Particle simulation benchmark
        n_particles = 1_000_000
        
        @ti.kernel
        def particle_step(pos: ti.template(), vel: ti.template(), dt: float):
            for i in pos:
                vel[i] += ti.Vector([0.0, -9.8, 0.0]) * dt
                pos[i] += vel[i] * dt
        
        pos = ti.Vector.field(3, dtype=ti.f32, shape=n_particles)
        vel = ti.Vector.field(3, dtype=ti.f32, shape=n_particles)
        
        @ti.kernel
        def init():
            for i in pos:
                pos[i] = ti.Vector([ti.random(), ti.random(), ti.random()])
                vel[i] = ti.Vector([0.0, 0.0, 0.0])
        
        init()
        
        # Warmup
        for _ in range(10):
            particle_step(pos, vel, 0.001)
        ti.sync()
        
        # Benchmark
        start = time.time()
        for _ in range(1000):
            particle_step(pos, vel, 0.001)
        ti.sync()
        elapsed = time.time() - start
        
        steps_per_sec = 1000 / elapsed
        particles_per_sec = n_particles * steps_per_sec
        
        print(f"\nParticle Simulation ({n_particles:,} particles):")
        print(f"  Steps/sec: {steps_per_sec:.1f}")
        print(f"  Particles/sec: {particles_per_sec/1e9:.2f} billion")
        
    except Exception as e:
        print(f"Taichi benchmark failed: {e}")

def benchmark_memory():
    """Benchmark memory allocation patterns"""
    print("\n" + "=" * 40)
    print("Memory Benchmarks")
    print("-" * 40)
    
    if not torch.cuda.is_available():
        return
    
    device = torch.device("cuda:0")
    
    # Memory allocation speed
    sizes = [64, 256, 512, 1024]
    
    print("\nAllocation Speed (allocs/sec):")
    for size in sizes:
        torch.cuda.empty_cache()
        
        # Warmup
        tensors = [torch.empty(size, size, device=device) for _ in range(100)]
        del tensors
        torch.cuda.synchronize()
        
        # Benchmark
        start = time.time()
        for _ in range(100):
            tensors = [torch.empty(size, size, device=device) for _ in range(10)]
            del tensors
        torch.cuda.synchronize()
        elapsed = time.time() - start
        
        allocs_per_sec = 1000 / elapsed
        print(f"  {size}x{size}: {allocs_per_sec:.0f} allocs/sec")
    
    # Peak memory for different batch sizes
    print("\nPeak Memory Usage:")
    
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    
    # Simulate forward pass memory
    for batch_size in [1, 2, 4, 8]:
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        
        try:
            # Simulate 3D convolution workload (like FNO/UNet)
            x = torch.randn(batch_size, 64, 64, 64, 64, device=device)
            w = torch.randn(64, 64, 3, 3, 3, device=device)
            
            y = torch.nn.functional.conv3d(x, w, padding=1)
            y = torch.nn.functional.gelu(y)
            
            torch.cuda.synchronize()
            
            peak_mb = torch.cuda.max_memory_allocated() / 1024**2
            print(f"  Batch {batch_size}: {peak_mb:.0f} MB")
            
            del x, w, y
            
        except RuntimeError as e:
            print(f"  Batch {batch_size}: OOM")
            break

if __name__ == "__main__":
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA version: {torch.version.cuda}")
        print(f"cuDNN version: {torch.backends.cudnn.version()}")
        print(f"GPU count: {torch.cuda.device_count()}")
    print("")
    
    benchmark_pytorch()
    benchmark_taichi()
    benchmark_memory()
    
    print("\n" + "=" * 40)
    print("Benchmark Complete")
    print("=" * 40)
EOF

echo ""
echo "Benchmark complete!"
