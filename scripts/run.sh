#!/bin/bash
#===============================================================================
# Project Aletheia: Neural Morphogenesis Engine
# Run Script
#
# Launches the simulation with optimal settings for 4x NVIDIA A16 GPUs
#===============================================================================

set -e

# Get script directory
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_ROOT="$( cd "$SCRIPT_DIR/.." && pwd )"

cd "$PROJECT_ROOT"

#-------------------------------------------------------------------------------
# Configuration
#-------------------------------------------------------------------------------

# Default values (can be overridden by environment variables)
NUM_GPUS=${NUM_GPUS:-4}
CONFIG=${CONFIG:-"configs/default.yaml"}
OUTPUT_DIR=${OUTPUT_DIR:-"/data/frames"}
PRESET=${PRESET:-""}
SEED=${SEED:-$(date +%s)}
PREVIEW=${PREVIEW:-false}
LOG_LEVEL=${LOG_LEVEL:-"INFO"}

#-------------------------------------------------------------------------------
# Parse command line arguments
#-------------------------------------------------------------------------------

print_usage() {
    echo "Usage: $0 [OPTIONS]"
    echo ""
    echo "Options:"
    echo "  -g, --gpus NUM       Number of GPUs to use (default: 4)"
    echo "  -c, --config PATH    Configuration file (default: configs/default.yaml)"
    echo "  -p, --preset NAME    Use preset: stellar_nursery, coral_lattice, neural_network"
    echo "  -o, --output DIR     Output directory (default: /data/frames)"
    echo "  -s, --seed NUM       Random seed"
    echo "  --preview            Quick preview mode (fewer frames, lower resolution)"
    echo "  -h, --help           Show this help"
    echo ""
    echo "Environment Variables:"
    echo "  NUM_GPUS             Number of GPUs"
    echo "  CONFIG               Config file path"
    echo "  OUTPUT_DIR           Output directory"
    echo "  PRESET               Preset name"
    echo "  SEED                 Random seed"
    echo "  PREVIEW              Set to 'true' for preview mode"
    echo ""
    echo "Examples:"
    echo "  $0 --preset stellar_nursery"
    echo "  $0 --gpus 2 --preview"
    echo "  NUM_GPUS=1 CONFIG=configs/custom.yaml $0"
}

while [[ $# -gt 0 ]]; do
    case $1 in
        -g|--gpus)
            NUM_GPUS="$2"
            shift 2
            ;;
        -c|--config)
            CONFIG="$2"
            shift 2
            ;;
        -p|--preset)
            PRESET="$2"
            shift 2
            ;;
        -o|--output)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        -s|--seed)
            SEED="$2"
            shift 2
            ;;
        --preview)
            PREVIEW=true
            shift
            ;;
        -h|--help)
            print_usage
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            print_usage
            exit 1
            ;;
    esac
done

#-------------------------------------------------------------------------------
# Environment Setup
#-------------------------------------------------------------------------------

# Activate virtual environment if it exists
if [ -f "$PROJECT_ROOT/venv/bin/activate" ]; then
    source "$PROJECT_ROOT/venv/bin/activate"
fi

# CUDA optimizations for A16 (Ampere architecture)
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-"0,1,2,3"}

# PyTorch optimizations
export PYTORCH_CUDA_ALLOC_CONF="max_split_size_mb:512"
export TORCH_CUDA_ARCH_LIST="8.6"  # Ampere

# Enable TF32 for Ampere GPUs (faster matmul)
export NVIDIA_TF32_OVERRIDE=1

# NCCL settings for multi-GPU
export NCCL_DEBUG=${NCCL_DEBUG:-"WARN"}
export NCCL_IB_DISABLE=0
export NCCL_SOCKET_IFNAME=eth0

# Taichi settings
export TI_ARCH=cuda
export TI_DEVICE_MEMORY_GB=15

#-------------------------------------------------------------------------------
# Validate Configuration
#-------------------------------------------------------------------------------

echo "=============================================="
echo "  Project Aletheia: Neural Morphogenesis"
echo "=============================================="
echo ""

# Check GPU availability
GPU_COUNT=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | wc -l || echo "0")

if [ "$GPU_COUNT" -eq 0 ]; then
    echo "Error: No GPUs detected"
    exit 1
fi

if [ "$NUM_GPUS" -gt "$GPU_COUNT" ]; then
    echo "Warning: Requested $NUM_GPUS GPUs but only $GPU_COUNT available"
    NUM_GPUS=$GPU_COUNT
fi

echo "Configuration:"
echo "  GPUs:        $NUM_GPUS x $(nvidia-smi --query-gpu=name --format=csv,noheader | head -n 1)"
echo "  Config:      $CONFIG"
if [ -n "$PRESET" ]; then
    echo "  Preset:      $PRESET"
fi
echo "  Output:      $OUTPUT_DIR"
echo "  Seed:        $SEED"
echo "  Preview:     $PREVIEW"
echo ""

# Validate config file exists
if [ -n "$PRESET" ]; then
    CONFIG="configs/${PRESET}.yaml"
fi

if [ ! -f "$CONFIG" ]; then
    echo "Error: Configuration file not found: $CONFIG"
    exit 1
fi

# Create output directory
mkdir -p "$OUTPUT_DIR"
mkdir -p "$OUTPUT_DIR/metadata"
mkdir -p "$OUTPUT_DIR/../output"
mkdir -p "logs"

#-------------------------------------------------------------------------------
# Build Command
#-------------------------------------------------------------------------------

# Build Python arguments
PYTHON_ARGS="--config $CONFIG --output-dir $OUTPUT_DIR --seed $SEED"

if [ -n "$PRESET" ]; then
    PYTHON_ARGS="$PYTHON_ARGS --preset $PRESET"
fi

if [ "$PREVIEW" = true ]; then
    PYTHON_ARGS="$PYTHON_ARGS --preview"
fi

#-------------------------------------------------------------------------------
# Launch Simulation
#-------------------------------------------------------------------------------

echo "Starting simulation..."
echo "Command: torchrun --nproc_per_node=$NUM_GPUS main.py $PYTHON_ARGS"
echo ""
echo "----------------------------------------------"
echo ""

# Determine launch method
if [ "$NUM_GPUS" -gt 1 ]; then
    # Multi-GPU with torchrun
    torchrun \
        --nproc_per_node=$NUM_GPUS \
        --master_port=${MASTER_PORT:-29500} \
        --rdzv_backend=c10d \
        --rdzv_endpoint=localhost:${RDZV_PORT:-29501} \
        main.py $PYTHON_ARGS
else
    # Single GPU
    python main.py $PYTHON_ARGS
fi

#-------------------------------------------------------------------------------
# Post-processing
#-------------------------------------------------------------------------------

EXIT_CODE=$?

echo ""
echo "----------------------------------------------"

if [ $EXIT_CODE -eq 0 ]; then
    echo "✓ Simulation completed successfully!"
    echo ""
    echo "Output:"
    echo "  Frames:    $OUTPUT_DIR"
    echo "  Metadata:  $OUTPUT_DIR/metadata"
    echo "  Video:     $OUTPUT_DIR/../output/morphogenesis.mp4"
    echo ""
    
    # Check if FFmpeg is available for video encoding
    if ! command -v ffmpeg &> /dev/null; then
        echo "Note: FFmpeg not found. To encode video manually:"
        echo ""
        echo "  ffmpeg -framerate 60 -i $OUTPUT_DIR/frame_%06d.png \\"
        echo "    -c:v libx264 -crf 18 -preset slow -pix_fmt yuv420p \\"
        echo "    $OUTPUT_DIR/../output/morphogenesis.mp4"
    fi
else
    echo "✗ Simulation failed with exit code $EXIT_CODE"
    echo "Check logs in: logs/"
fi

exit $EXIT_CODE
