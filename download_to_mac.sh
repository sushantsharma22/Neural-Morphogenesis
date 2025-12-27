#!/bin/bash
# =============================================================================
# Neural Morphogenesis - Download Script
# =============================================================================
# This script downloads simulation outputs from the SSH server to your Mac.
# 
# USAGE (Run this on your Mac, NOT on the server):
# -----------------------------------------------------------------------------
# 1. Copy this script to your Mac
# 2. Make it executable: chmod +x download_to_mac.sh
# 3. Run it: ./download_to_mac.sh
#
# OR use the one-liner at the bottom of this file
# =============================================================================

# Configuration - EDIT THESE VALUES
SSH_USER="sharmas1"                              # Your SSH username
SSH_HOST="delta.cs.uwindsor.ca"            # Your university SSH server hostname
REMOTE_PATH="/home/sharm2s1/Neural-Morphogenesis/output"  # Path on server
LOCAL_PATH="$HOME/Downloads/NeuralMorphogenesis" # Destination on your Mac

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}=============================================${NC}"
echo -e "${BLUE}  Neural Morphogenesis - Download Script${NC}"
echo -e "${BLUE}=============================================${NC}"
echo ""

# Create local directory if it doesn't exist
mkdir -p "$LOCAL_PATH"
echo -e "${GREEN}✓${NC} Local directory ready: $LOCAL_PATH"

# Check if we can reach the server
echo -e "${YELLOW}→${NC} Connecting to $SSH_HOST..."

# Download simulation outputs
echo -e "${YELLOW}→${NC} Downloading simulation outputs..."
scp -r "${SSH_USER}@${SSH_HOST}:${REMOTE_PATH}/*" "$LOCAL_PATH/"

if [ $? -eq 0 ]; then
    echo ""
    echo -e "${GREEN}=============================================${NC}"
    echo -e "${GREEN}  ✓ Download Complete!${NC}"
    echo -e "${GREEN}=============================================${NC}"
    echo ""
    echo -e "Files saved to: ${BLUE}$LOCAL_PATH${NC}"
    echo ""
    
    # List downloaded files
    echo "Downloaded files:"
    ls -lh "$LOCAL_PATH" 2>/dev/null | tail -20
    
    # Open the folder on Mac
    if [[ "$OSTYPE" == "darwin"* ]]; then
        echo ""
        echo -e "${YELLOW}→${NC} Opening folder in Finder..."
        open "$LOCAL_PATH"
    fi
else
    echo ""
    echo -e "${RED}✗ Download failed!${NC}"
    echo ""
    echo "Troubleshooting:"
    echo "  1. Make sure you can SSH to the server: ssh ${SSH_USER}@${SSH_HOST}"
    echo "  2. Check that the remote path exists: ${REMOTE_PATH}"
    echo "  3. Make sure you have the correct hostname"
    exit 1
fi

# =============================================================================
# ONE-LINER VERSION (Copy and paste directly into Mac terminal):
# =============================================================================
# Replace YOUR_USERNAME, YOUR_SERVER, and run:
#
# scp -r YOUR_USERNAME@YOUR_SERVER:/home/YOUR_USERNAME/Neural-Morphogenesis/output ~/Downloads/NeuralMorphogenesis/ && open ~/Downloads/NeuralMorphogenesis
#
# Example:
# scp -r sharm2s1@ssh.university.edu:/home/sharm2s1/Neural-Morphogenesis/output ~/Downloads/NeuralMorphogenesis/ && open ~/Downloads/NeuralMorphogenesis
# =============================================================================
