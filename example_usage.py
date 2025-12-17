"""
Example usage of the modular frequency analysis library.

This demonstrates how to import and use the different modules.
"""

import torch

# Import from data module
from data import load_imagenet_validation, get_denormalize_transform

# Import from models module
from models import load_resnet18, test_classifier_baseline

# Import from frequency module
from frequency import FrequencyFilterPipeline, apply_fft, apply_ifft

# Import from utils module
from utils import visualize_2d_frequency_mask


def example_pipeline():
    """Example of creating and using the frequency filter pipeline."""

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # 1. Load dataset
    print("\n[1/4] Loading dataset...")
    dataset, dataloader = load_imagenet_validation(subset_size=100, image_size=224)

    # 2. Load classifier
    print("\n[2/4] Loading ResNet-18...")
    classifier = load_resnet18(pretrained=True, device=device)

    # 3. Test baseline
    print("\n[3/4] Testing baseline accuracy...")
    baseline_acc = test_classifier_baseline(classifier, dataloader, device=device)

    # 4. Create frequency filter pipeline
    print("\n[4/4] Creating frequency filter pipeline...")
    mask_config = {
        'image_size': 224,
        'init_value': 1.0,
        'init_std': 0.1
    }
    pipeline = FrequencyFilterPipeline(classifier, mask_config=mask_config).to(device)

    # Test forward pass
    print("\nTesting forward pass...")
    with torch.no_grad():
        for images, labels in dataloader:
            images = images.to(device)
            outputs, reconstructed = pipeline(images)
            print(f"  Input shape: {images.shape}")
            print(f"  Output shape: {outputs.shape}")
            print(f"  Reconstructed shape: {reconstructed.shape}")
            break

    # Visualize initial mask
    print("\nVisualizing initial mask...")
    visualize_2d_frequency_mask(pipeline.freq_mask, title="Initial 2D Frequency Mask")

    print("\n✓ Example complete!")


def example_fft_transforms():
    """Example of using FFT/IFFT transforms directly."""

    print("\nExample: FFT/IFFT Transforms")
    print("-" * 60)

    # Create dummy image
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    test_img = torch.randn(2, 3, 224, 224).to(device)
    print(f"Input image shape: {test_img.shape}")

    # Forward FFT
    fft_result = apply_fft(test_img)
    print(f"FFT output shape: {fft_result.shape}, dtype: {fft_result.dtype}")

    # Inverse FFT
    reconstructed = apply_ifft(fft_result)
    print(f"IFFT output shape: {reconstructed.shape}, dtype: {reconstructed.dtype}")

    # Check reconstruction quality
    reconstruction_error = (test_img - reconstructed).abs().mean()
    print(f"Reconstruction error: {reconstruction_error.item():.6f}")

    print("✓ FFT/IFFT transform test complete!")


if __name__ == "__main__":
    print("=" * 60)
    print("FREQUENCY ANALYSIS LIBRARY - EXAMPLE USAGE")
    print("=" * 60)

    # Run example pipeline
    example_pipeline()

    # Run FFT/IFFT example
    print("\n" + "=" * 60)
    example_fft_transforms()
    print("=" * 60)
