#!/bin/bash
# =============================================================================
# Neural Morphogenesis - Quick Run & Download
# =============================================================================
# This script runs a simulation preview and prepares files for download
# Run this ON THE SERVER before downloading to your Mac
# =============================================================================

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Colors
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

echo -e "${BLUE}=============================================${NC}"
echo -e "${BLUE}  Neural Morphogenesis - Quick Run${NC}"
echo -e "${BLUE}=============================================${NC}"
echo ""

# Activate virtual environment
if [ -d "venv" ]; then
    source venv/bin/activate
    echo -e "${GREEN}✓${NC} Virtual environment activated"
else
    echo "Creating virtual environment..."
    python3 -m venv venv
    source venv/bin/activate
    pip install --upgrade pip
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
    pip install taichi numpy pillow mcp
fi

# Clean previous output
if [ -d "output" ]; then
    echo -e "${YELLOW}→${NC} Cleaning previous output..."
    rm -rf output/*
fi

# Run preview simulation
echo -e "${YELLOW}→${NC} Running simulation preview..."
echo ""

python3 main.py --preview --output ./output

echo ""
echo -e "${GREEN}=============================================${NC}"
echo -e "${GREEN}  ✓ Simulation Complete!${NC}"
echo -e "${GREEN}=============================================${NC}"
echo ""

# Show output info
echo "Output files:"
ls -lh output/ 2>/dev/null | head -25
echo ""

# Show download instructions
echo -e "${BLUE}=============================================${NC}"
echo -e "${BLUE}  To download to your Mac:${NC}"
echo -e "${BLUE}=============================================${NC}"
echo ""
echo "Run this command ON YOUR MAC (not here):"
echo ""
echo -e "${YELLOW}scp -r $(whoami)@\$(hostname):$(pwd)/output ~/Downloads/NeuralMorphogenesis/ && open ~/Downloads/NeuralMorphogenesis${NC}"
echo ""
echo "Or copy the 'download_to_mac.sh' script to your Mac and run it there."
echo ""
