#!/bin/bash
#===============================================================================
# Project Aletheia: Neural Morphogenesis Engine
# Setup Script
#
# This script installs all dependencies using pip install --user
# No sudo required - designed for shared HPC environments
#===============================================================================

set -e

echo "=============================================="
echo "  Project Aletheia Setup"
echo "  Neural Morphogenesis Engine"
echo "=============================================="
echo ""

# Get script directory
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_ROOT="$( cd "$SCRIPT_DIR/.." && pwd )"

echo "Project root: $PROJECT_ROOT"
echo ""

# Check Python version
echo "Checking Python version..."
PYTHON_VERSION=$(python3 --version 2>&1 | cut -d' ' -f2)
PYTHON_MAJOR=$(echo $PYTHON_VERSION | cut -d'.' -f1)
PYTHON_MINOR=$(echo $PYTHON_VERSION | cut -d'.' -f2)

if [ "$PYTHON_MAJOR" -lt 3 ] || ([ "$PYTHON_MAJOR" -eq 3 ] && [ "$PYTHON_MINOR" -lt 8 ]); then
    echo "Error: Python 3.8+ required. Found: Python $PYTHON_VERSION"
    exit 1
fi
echo "Python $PYTHON_VERSION ✓"
echo ""

# Check CUDA availability
echo "Checking CUDA..."
if command -v nvidia-smi &> /dev/null; then
    echo "NVIDIA Driver:"
    nvidia-smi --query-gpu=driver_version,name,memory.total --format=csv,noheader | head -n 4
    echo ""
    
    GPU_COUNT=$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)
    echo "Found $GPU_COUNT GPU(s)"
else
    echo "Warning: nvidia-smi not found. GPU support may be limited."
fi
echo ""

# Check for existing virtual environment or use user install
USE_VENV=false
VENV_PATH="$PROJECT_ROOT/venv"

if [ -d "$VENV_PATH" ]; then
    echo "Found existing virtual environment at $VENV_PATH"
    source "$VENV_PATH/bin/activate"
    USE_VENV=true
elif [ -w "$PROJECT_ROOT" ]; then
    echo "Creating virtual environment..."
    python3 -m venv "$VENV_PATH"
    source "$VENV_PATH/bin/activate"
    USE_VENV=true
    echo "Virtual environment created ✓"
else
    echo "Using pip install --user (no write access to project directory)"
    PIP_FLAGS="--user"
fi
echo ""

# Upgrade pip
echo "Upgrading pip..."
python3 -m pip install --upgrade pip $PIP_FLAGS
echo ""

# Install PyTorch with CUDA support
echo "Installing PyTorch with CUDA support..."
# Check CUDA version from nvidia-smi
if command -v nvidia-smi &> /dev/null; then
    CUDA_VERSION=$(nvidia-smi | grep "CUDA Version" | awk '{print $9}')
    CUDA_MAJOR=$(echo $CUDA_VERSION | cut -d'.' -f1)
    
    if [ "$CUDA_MAJOR" -ge 12 ]; then
        echo "Installing for CUDA 12.x"
        pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121 $PIP_FLAGS
    elif [ "$CUDA_MAJOR" -ge 11 ]; then
        echo "Installing for CUDA 11.x"
        pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118 $PIP_FLAGS
    else
        echo "Installing CPU version (CUDA $CUDA_VERSION not directly supported)"
        pip install torch torchvision torchaudio $PIP_FLAGS
    fi
else
    echo "Installing CPU version (no CUDA detected)"
    pip install torch torchvision torchaudio $PIP_FLAGS
fi
echo ""

# Install Taichi
echo "Installing Taichi..."
pip install taichi $PIP_FLAGS
echo ""

# Install remaining dependencies
echo "Installing remaining dependencies..."
pip install $PIP_FLAGS \
    numpy>=1.21.0 \
    scipy>=1.7.0 \
    pillow>=9.0.0 \
    pyyaml>=6.0 \
    tqdm>=4.62.0 \
    matplotlib>=3.5.0 \
    h5py>=3.6.0 \
    einops>=0.6.0

echo ""

# Optional: Install additional tools
echo "Installing optional tools..."
pip install $PIP_FLAGS \
    tensorboard>=2.8.0 \
    wandb \
    || echo "Optional tools installation skipped"

echo ""

# Verify installations
echo "Verifying installations..."
echo ""

python3 << 'EOF'
import sys

def check_import(name, package=None):
    package = package or name
    try:
        module = __import__(package)
        version = getattr(module, '__version__', 'unknown')
        print(f"  {name}: {version} ✓")
        return True
    except ImportError as e:
        print(f"  {name}: NOT INSTALLED ✗ ({e})")
        return False

print("Core packages:")
check_import("Python", package="sys")
print(f"  Python: {sys.version.split()[0]} ✓")

all_ok = True
all_ok &= check_import("PyTorch", "torch")
all_ok &= check_import("NumPy", "numpy")
all_ok &= check_import("Taichi", "taichi")
all_ok &= check_import("PIL", "PIL")
all_ok &= check_import("YAML", "yaml")
all_ok &= check_import("SciPy", "scipy")

# Check CUDA in PyTorch
import torch
if torch.cuda.is_available():
    print(f"\nCUDA available: {torch.cuda.get_device_name(0)}")
    print(f"CUDA devices: {torch.cuda.device_count()}")
    print(f"cuDNN version: {torch.backends.cudnn.version()}")
else:
    print("\nWarning: CUDA not available in PyTorch")

if not all_ok:
    print("\n⚠ Some packages failed to install")
    sys.exit(1)
else:
    print("\n✓ All core packages installed successfully")
EOF

echo ""

# Create output directories
echo "Creating output directories..."
mkdir -p "$PROJECT_ROOT/data/frames"
mkdir -p "$PROJECT_ROOT/data/output"
mkdir -p "$PROJECT_ROOT/data/checkpoints"
mkdir -p "$PROJECT_ROOT/logs"
echo "Output directories created ✓"
echo ""

# Print activation instructions
echo "=============================================="
echo "  Setup Complete!"
echo "=============================================="
echo ""

if [ "$USE_VENV" = true ]; then
    echo "To activate the virtual environment:"
    echo "  source $VENV_PATH/bin/activate"
    echo ""
fi

echo "To run the simulation:"
echo "  cd $PROJECT_ROOT"
echo "  python main.py --config configs/default.yaml"
echo ""
echo "For multi-GPU (4x A16):"
echo "  torchrun --nproc_per_node=4 main.py --config configs/default.yaml"
echo ""
echo "For a quick preview:"
echo "  python main.py --preview"
echo ""
