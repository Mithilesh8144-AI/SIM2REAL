#!/usr/bin/env python3
"""
Quick script to check if PyTorch can access GPU.
Run: pixi run python check_gpu.py
"""

import torch
import sys

print("=" * 60)
print("PyTorch GPU Check")
print("=" * 60)

# Check CUDA availability
cuda_available = torch.cuda.is_available()
print(f"CUDA Available: {cuda_available}")

if cuda_available:
    print(f"✓ CUDA Version: {torch.version.cuda}")
    print(f"✓ GPU Count: {torch.cuda.device_count()}")
    print(f"✓ Current GPU: {torch.cuda.current_device()}")
    print(f"✓ GPU Name: {torch.cuda.get_device_name(0)}")

    # Test GPU operation
    try:
        x = torch.randn(1000, 1000).cuda()
        y = torch.randn(1000, 1000).cuda()
        z = torch.matmul(x, y)
        print(f"✓ GPU Test: Matrix multiplication successful!")
        print(f"  Result shape: {z.shape}")
        print(f"  Result device: {z.device}")
    except Exception as e:
        print(f"✗ GPU Test Failed: {e}")

    print("\n✓ GPU is ready to use!")
    sys.exit(0)
else:
    print("✗ CUDA not available - PyTorch will use CPU")
    print("\nPossible reasons:")
    print("  1. PyTorch installed without CUDA support")
    print("  2. No NVIDIA GPU detected")
    print("  3. CUDA drivers not installed")
    print("\nTo fix:")
    print("  pixi run pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118")
    sys.exit(1)

print("=" * 60)
