"""
Main training script for 2D frequency mask learning.

This script trains a learnable 2D frequency mask to optimize classification accuracy
on a frozen pre-trained classifier (ResNet-18 by default).
"""

import time
import torch
import torch.nn as nn
import torch.optim as optim

from data.dataset import load_imagenet_validation, get_denormalize_transform
from models.resnet18 import load_resnet18, test_classifier_baseline
from frequency.pipeline import FrequencyFilterPipeline
from utils.visualization import (
    visualize_2d_frequency_mask,
    visualize_reconstructed_images,
    plot_training_history
)


# ============================================
# CONFIGURATION
# ============================================
NUM_EPOCHS = 5
LEARNING_RATE = 0.05
WEIGHT_DECAY = 0.0001
SUBSET_SIZE = 5000
IMAGE_SIZE = 224
BATCH_SIZE = 64

# Logging
LOG_EVERY_N_BATCHES = 10

# Checkpoint
SAVE_CHECKPOINT = True
CHECKPOINT_PATH = "./checkpoints/resnet18_2d_mask.pth"

# Device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def train_2d_mask(pipeline, dataloader, optimizer, criterion, num_epochs, device):
    """
    Train the 2D frequency mask.

    Args:
        pipeline: FrequencyFilterPipeline instance
        dataloader: DataLoader for training data
        optimizer: Optimizer for mask parameters
        criterion: Loss function
        num_epochs: Number of epochs to train
        device: Device to train on

    Returns:
        history: Training history dict
    """
    print("\n" + "=" * 60)
    print("TRAINING 2D FREQUENCY MASK")
    print("=" * 60)

    pipeline.train()
    best_acc = 0.0

    training_history = {
        'epoch': [],
        'loss': [],
        'accuracy': []
    }

    for epoch in range(num_epochs):
        print(f"\n{'=' * 60}")
        print(f"EPOCH {epoch + 1}/{num_epochs}")
        print(f"{'=' * 60}")

        epoch_start = time.time()
        epoch_loss = 0.0
        epoch_correct = 0
        epoch_total = 0
        batch_count = 0

        for batch_idx, (images, labels) in enumerate(dataloader):
            images, labels = images.to(device), labels.to(device)

            # Forward pass
            optimizer.zero_grad()
            outputs, reconstructed = pipeline(images)
            loss = criterion(outputs, labels)

            # Backward pass (only updates frequency mask!)
            loss.backward()
            optimizer.step()

            # Metrics
            _, predicted = torch.max(outputs, 1)
            batch_correct = (predicted == labels).sum().item()
            epoch_correct += batch_correct
            epoch_total += labels.size(0)
            epoch_loss += loss.item()
            batch_count += 1

            # Log every N batches
            if (batch_idx + 1) % LOG_EVERY_N_BATCHES == 0:
                avg_loss = epoch_loss / batch_count
                avg_acc = 100 * epoch_correct / epoch_total
                print(f"  Batch [{batch_idx + 1:3d}] | Loss: {loss.item():.4f} | Avg Acc: {avg_acc:.2f}%")

        # Epoch summary
        epoch_time = time.time() - epoch_start
        avg_epoch_loss = epoch_loss / batch_count
        epoch_accuracy = 100 * epoch_correct / epoch_total

        print(f"\n{'-' * 60}")
        print(f"EPOCH {epoch + 1} SUMMARY:")
        print(f"  Time:     {epoch_time:.1f}s")
        print(f"  Batches:  {batch_count}")
        print(f"  Avg Loss: {avg_epoch_loss:.4f}")
        print(f"  Accuracy: {epoch_accuracy:.2f}%")
        print(f"{'-' * 60}")

        # Save history
        training_history['epoch'].append(epoch + 1)
        training_history['loss'].append(avg_epoch_loss)
        training_history['accuracy'].append(epoch_accuracy)

        # Visualize mask after epoch
        print(f"\n2D Mask after epoch {epoch + 1}:")
        mask_vis = pipeline.freq_mask.get_mask_visualization()
        print(f"Mask stats: min={mask_vis.min():.3f}, max={mask_vis.max():.3f}, mean={mask_vis.mean():.3f}")

        # Save checkpoint if best
        if SAVE_CHECKPOINT and epoch_accuracy > best_acc:
            best_acc = epoch_accuracy
            torch.save({
                'epoch': epoch + 1,
                'mask_state_dict': pipeline.freq_mask.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': avg_epoch_loss,
                'accuracy': epoch_accuracy,
            }, CHECKPOINT_PATH)
            print(f"  ✓ Checkpoint saved (best accuracy: {best_acc:.2f}%)")

    print("\n" + "=" * 60)
    print("TRAINING COMPLETE!")
    print("=" * 60)
    print(f"Best accuracy: {best_acc:.2f}%")

    return training_history


def main():
    """Main training function."""
    print("=" * 60)
    print("2D FREQUENCY MASK TRAINING")
    print("=" * 60)
    print(f"Device: {device}")
    print(f"Dataset: ImageNet validation ({SUBSET_SIZE} images)")
    print(f"Batch size: {BATCH_SIZE}")
    print(f"Epochs: {NUM_EPOCHS}")
    print(f"Learning rate: {LEARNING_RATE}")
    print(f"Weight decay: {WEIGHT_DECAY}")
    print("=" * 60)

    # 1. Load dataset
    print("\n[1/5] Loading dataset...")
    valset, valloader = load_imagenet_validation(subset_size=SUBSET_SIZE, image_size=IMAGE_SIZE)

    # 2. Load classifier
    print("\n[2/5] Loading ResNet-18 classifier...")
    classifier = load_resnet18(pretrained=True, device=device)

    # 3. Test baseline accuracy
    print("\n[3/5] Testing baseline accuracy...")
    baseline_acc = test_classifier_baseline(classifier, valloader, device=device, max_batches=None)

    # 4. Create frequency filter pipeline with 2D mask
    print("\n[4/5] Creating frequency filter pipeline...")
    mask_config = {
        'image_size': IMAGE_SIZE,
        'init_value': 1.0,
        'init_std': 0.1
    }
    pipeline = FrequencyFilterPipeline(classifier, mask_config=mask_config).to(device)

    # Setup optimizer and loss
    optimizer = optim.Adam(
        pipeline.get_trainable_params(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY
    )
    criterion = nn.CrossEntropyLoss()

    trainable_params = sum(p.numel() for p in pipeline.get_trainable_params())
    print(f"\nTraining setup:")
    print(f"  Optimizer: Adam")
    print(f"  Trainable params: {trainable_params:,}")
    print(f"  Baseline accuracy: {baseline_acc:.2f}%")

    # 5. Train the mask
    print("\n[5/5] Training frequency mask...")
    history = train_2d_mask(pipeline, valloader, optimizer, criterion, NUM_EPOCHS, device)

    # Visualizations
    print("\n" + "=" * 60)
    print("GENERATING VISUALIZATIONS")
    print("=" * 60)

    # Plot training history
    plot_training_history(history, save_path="training_history.png")

    # Visualize learned mask
    visualize_2d_frequency_mask(
        pipeline.freq_mask,
        title="Learned 2D Frequency Mask (ResNet-18)",
        save_path="learned_mask_resnet18.png"
    )

    # Visualize reconstructed images
    pipeline.eval()
    with torch.no_grad():
        for images, labels in valloader:
            images, labels = images.to(device), labels.to(device)
            outputs, reconstructed = pipeline(images)
            break

    denormalize = get_denormalize_transform()
    visualize_reconstructed_images(
        images, reconstructed, labels, pipeline.freq_mask,
        denormalize_fn=denormalize,
        num_images=4,
        save_path="reconstructed_images_resnet18.png"
    )

    print("\n✓ Training complete! All visualizations saved.")


if __name__ == "__main__":
    main()
