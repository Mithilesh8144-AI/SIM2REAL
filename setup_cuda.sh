#!/bin/bash
###############################################################################
# CUDA Environment Setup Script
# Run this script on university systems to set up PyTorch with CUDA support
# Usage: bash setup_cuda.sh
###############################################################################

set -e  # Exit on error

echo "=========================================="
echo "Setting up CUDA PyTorch Environment"
echo "=========================================="

# 1. Create tmp directory in home (where there's space)
echo ""
echo "[1/5] Creating temp directory..."
TMP_DIR="${HOME}/tmp"
mkdir -p "$TMP_DIR"
echo "✓ Created: $TMP_DIR"

# 2. Set TMPDIR environment variable
echo ""
echo "[2/5] Setting TMPDIR environment variable..."
export TMPDIR="$TMP_DIR"
echo "✓ TMPDIR set to: $TMPDIR"

# 3. Add TMPDIR to .bashrc if not already there
echo ""
echo "[3/5] Making TMPDIR permanent in .bashrc..."
if ! grep -q "export TMPDIR=" ~/.bashrc; then
    echo "" >> ~/.bashrc
    echo "# Set TMPDIR to home directory (for large downloads)" >> ~/.bashrc
    echo "export TMPDIR=${TMP_DIR}" >> ~/.bashrc
    echo "✓ Added TMPDIR to ~/.bashrc"
else
    echo "✓ TMPDIR already in ~/.bashrc"
fi

# 4. Install pixi environment (base dependencies)
echo ""
echo "[4/5] Installing pixi environment..."
pixi install
echo "✓ Pixi environment installed"

# 5. Install PyTorch with CUDA support via pip
echo ""
echo "[5/5] Installing PyTorch with CUDA 11.8 support..."
pixi run python -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
echo "✓ PyTorch with CUDA installed"

# 6. Run GPU check
echo ""
echo "=========================================="
echo "Verifying GPU Setup"
echo "=========================================="
pixi run check-gpu

echo ""
echo "=========================================="
echo "✓ Setup Complete!"
echo "=========================================="
echo ""
echo "Next steps:"
echo "  - Run 'pixi run check-gpu' anytime to verify GPU"
echo "  - Run 'pixi run lab' to start Jupyter Lab"
echo "  - Run 'python train.py' to start training"
echo ""
echo "Note: TMPDIR is now permanently set in ~/.bashrc"
echo "      New terminal sessions will automatically use it."
echo "=========================================="
