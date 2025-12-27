#!/usr/bin/env python3
"""
Quick test script to verify all imports and basic functionality
Run this after setup to ensure everything is working
"""

import sys
import os
import torch
import numpy as np

# Add project root to path
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJECT_ROOT)

def test_imports():
    """Test all module imports"""
    print("Testing imports...")
    
    errors = []
    
    # Core
    try:
        from aletheia.core import (
            AletheiaConfig, load_config, DistributedManager, MemoryManager
        )
        print("  ✓ Core modules")
    except Exception as e:
        errors.append(f"Core: {e}")
        print(f"  ✗ Core modules: {e}")
    
    # Models
    try:
        from aletheia.models import (
            FourierNeuralOperator3D, MultiscaleFNO3D, TimeConditionedFNO3D,
            UNetNeuralMPM, LightweightUNetMPM,
            LatentDNA, MorphogenesisScheduler
        )
        print("  ✓ Neural models")
    except Exception as e:
        errors.append(f"Models: {e}")
        print(f"  ✗ Neural models: {e}")
    
    # Physics
    try:
        from aletheia.physics import (
            MPMSimulator, TorchMPMSimulator,
            NeuralFieldSolver, GridField,
            CoupledPDESolver, NavierStokesSolver
        )
        print("  ✓ Physics modules")
    except Exception as e:
        errors.append(f"Physics: {e}")
        print(f"  ✗ Physics modules: {e}")
    
    # Renderer
    try:
        from aletheia.renderer import (
            HeadlessRenderer, ParticleRenderer,
            BloomEffect, DepthOfField, MotionBlur, BioluminescentPalette,
            Camera, CameraAnimator
        )
        print("  ✓ Renderer modules")
    except Exception as e:
        errors.append(f"Renderer: {e}")
        print(f"  ✗ Renderer modules: {e}")
    
    # Output
    try:
        from aletheia.output import (
            FrameExporter, MetadataRecorder, VideoEncoder
        )
        print("  ✓ Output modules")
    except Exception as e:
        errors.append(f"Output: {e}")
        print(f"  ✗ Output modules: {e}")
    
    return errors


def test_config():
    """Test configuration loading"""
    print("\nTesting configuration...")
    
    try:
        from aletheia.core import load_config
        
        config = load_config("configs/default.yaml")
        print(f"  ✓ Config loaded: {config.simulation.num_particles:,} particles, {config.simulation.total_frames} frames")
        return None
    except Exception as e:
        print(f"  ✗ Config load failed: {e}")
        return str(e)


def test_pytorch():
    """Test PyTorch and CUDA"""
    print("\nTesting PyTorch...")
    
    import torch
    
    print(f"  PyTorch version: {torch.__version__}")
    print(f"  CUDA available: {torch.cuda.is_available()}")
    
    if torch.cuda.is_available():
        print(f"  CUDA devices: {torch.cuda.device_count()}")
        for i in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(i)
            print(f"    GPU {i}: {props.name} ({props.total_memory / 1024**3:.1f} GB)")
        
        # Quick memory test
        try:
            x = torch.randn(1000, 1000, device="cuda:0")
            y = torch.matmul(x, x)
            del x, y
            torch.cuda.empty_cache()
            print("  ✓ CUDA operations working")
        except Exception as e:
            print(f"  ✗ CUDA operations failed: {e}")
            return str(e)
    
    return None


def test_taichi():
    """Test Taichi initialization"""
    print("\nTesting Taichi...")
    
    try:
        import taichi as ti
        import torch
        
        # Initialize with limited memory
        ti.init(arch=ti.cuda if torch.cuda.is_available() else ti.cpu, 
                device_memory_GB=2)
        
        # Simple kernel test
        @ti.kernel
        def test_kernel() -> ti.f32:  # type: ignore[valid-type]
            return 1.0 + 2.0
        
        result = test_kernel()
        
        if abs(result - 3.0) < 0.001:
            print(f"  ✓ Taichi working (arch: {'cuda' if torch.cuda.is_available() else 'cpu'})")
            return None
        else:
            print(f"  ✗ Taichi computation error")
            return "Computation error"
            
    except Exception as e:
        print(f"  ✗ Taichi failed: {e}")
        return str(e)


def test_model_creation():
    """Test creating neural models"""
    print("\nTesting model creation...")
    
    import torch
    
    try:
        from aletheia.models import TimeConditionedFNO3D, LightweightUNetMPM, LatentDNA
        
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        
        # Create small test models
        fno = TimeConditionedFNO3D(
            in_channels=4,
            out_channels=3,
            modes=(8, 8, 8),
            width=32,
            num_layers=2,
            latent_dim=64,
        ).to(device)
        
        param_count = sum(p.numel() for p in fno.parameters())
        print(f"  ✓ FNO created ({param_count:,} parameters)")
        
        unet = LightweightUNetMPM(
            in_channels=4,
            out_channels=3,
            base_channels=16,
            latent_dim=64,
        ).to(device)
        
        param_count = sum(p.numel() for p in unet.parameters())
        print(f"  ✓ U-Net created ({param_count:,} parameters)")
        
        # Test forward pass
        batch_size = 1
        grid_size = 32
        
        x = torch.randn(batch_size, 4, grid_size, grid_size, grid_size, device=device)
        latent = torch.randn(batch_size, 64, device=device)
        time_step = torch.tensor([0.5], device=device)
        
        with torch.no_grad():
            y_fno = fno(x, time_embedding=time_step, latent=latent)
            print(f"  ✓ FNO forward pass: {list(x.shape)} -> {list(y_fno.shape)}")
            
            y_unet = unet(x, time_step=time_step, latent=latent)
            print(f"  ✓ U-Net forward pass: {list(x.shape)} -> {list(y_unet.shape)}")
        
        del fno, unet, x, latent, y_fno, y_unet
        
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        return None
        
    except Exception as e:
        print(f"  ✗ Model creation failed: {e}")
        import traceback
        traceback.print_exc()
        return str(e)


def test_renderer():
    """Test renderer creation"""
    print("\nTesting renderer...")
    
    try:
        from aletheia.renderer import (
            SimpleSoftwareRenderer, BloomEffect, Camera, CameraAnimator
        )
        import numpy as np
        
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        
        # Create renderer
        renderer = SimpleSoftwareRenderer(resolution=(256, 144))
        print("  ✓ Renderer created")
        
        # Create test particles
        positions = np.random.randn(1000, 3).astype(np.float32)
        colors = np.random.rand(1000, 3).astype(np.float32)
        
        # Create camera
        camera = Camera(position=(0, 0, 5), look_at=(0, 0, 0))
        print("  ✓ Camera created")
        
        # Render
        frame = renderer.render(positions, colors, camera)
        print(f"  ✓ Frame rendered: {frame.shape}")
        
        # Apply bloom
        bloom = BloomEffect(resolution=(256, 144), num_passes=3)
        
        # Convert to tensor (1, 3, H, W)
        frame_tensor = torch.from_numpy(frame).permute(2, 0, 1).unsqueeze(0).float().to(device)
        frame_tensor = frame_tensor / 255.0 if frame_tensor.max() > 1.0 else frame_tensor
        
        frame_bloomed = bloom.apply(frame_tensor, intensity=0.5)
        print(f"  ✓ Bloom applied: {frame_bloomed.shape}")
        
        return None
        
    except Exception as e:
        print(f"  ✗ Renderer failed: {e}")
        import traceback
        traceback.print_exc()
        return str(e)


def main():
    """Run all tests"""
    print("=" * 50)
    print("  Project Aletheia: System Test")
    print("=" * 50)
    print()
    
    import torch  # Import here to have it available in all tests
    
    all_errors = []
    
    # Run tests
    errors = test_imports()
    all_errors.extend(errors)
    
    error = test_config()
    if error:
        all_errors.append(error)
    
    error = test_pytorch()
    if error:
        all_errors.append(error)
    
    error = test_taichi()
    if error:
        all_errors.append(error)
    
    error = test_model_creation()
    if error:
        all_errors.append(error)
    
    error = test_renderer()
    if error:
        all_errors.append(error)
    
    # Summary
    print()
    print("=" * 50)
    
    if all_errors:
        print(f"  ✗ Tests completed with {len(all_errors)} error(s)")
        print()
        for error in all_errors:
            print(f"    - {error}")
        return 1
    else:
        print("  ✓ All tests passed!")
        print()
        print("  Ready to run simulation:")
        print("    python main.py --preview")
        print()
        print("  Or with multi-GPU:")
        print("    ./scripts/run.sh --gpus 4")
        return 0


if __name__ == "__main__":
    sys.exit(main())
