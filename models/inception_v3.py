"""
Inception v3 model loader with pre-trained ImageNet weights.
"""

import torch
import torchvision.models as models


def load_inception_v3(pretrained=True, device='cuda'):
    """
    Load Inception v3 with ImageNet pre-trained weights.

    Args:
        pretrained: Whether to load pre-trained weights (default: True)
        device: Device to load model on ('cuda' or 'cpu')

    Returns:
        model: Inception v3 model in eval mode

    Note:
        Inception v3 requires 299x299 input images (not 224x224).
    """
    print("Loading Inception v3...")

    # Load model with pre-trained weights
    if pretrained:
        model = models.inception_v3(weights=models.Inception_V3_Weights.IMAGENET1K_V1)
        print("  ✓ Loaded pre-trained ImageNet weights")
    else:
        model = models.inception_v3(weights=None)
        print("  ✓ Loaded without pre-trained weights")

    # Disable auxiliary outputs for inference
    model.aux_logits = False

    # Move to device and set to eval mode
    model = model.to(device)
    model.eval()

    print(f"  ✓ Model moved to {device}")
    print(f"  ✓ Model set to eval mode")
    print(f"  ✓ Auxiliary logits disabled")

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    print(f"  ✓ Total parameters: {total_params:,}")

    return model


def test_classifier_baseline(model, dataloader, device='cuda', max_batches=None):
    """
    Test baseline accuracy of classifier on original images.

    Args:
        model: Classifier model
        dataloader: DataLoader for test data
        device: Device to run on
        max_batches: Maximum number of batches to test (None = all)

    Returns:
        accuracy: Accuracy percentage
    """
    model.eval()
    correct = 0
    total = 0

    print("\nTesting baseline classifier accuracy...")

    with torch.no_grad():
        for i, (images, labels) in enumerate(dataloader):
            if max_batches is not None and i >= max_batches:
                break

            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)

            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = 100 * correct / total
    print(f"  ✓ Baseline accuracy: {accuracy:.2f}% ({correct}/{total} correct)")

    return accuracy
