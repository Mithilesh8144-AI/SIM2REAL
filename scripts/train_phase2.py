#!/usr/bin/env python3
"""
Phase 2: Joint Training Script

Train both the learnable frequency mask AND the classifier from scratch.
This reveals architectural inductive bias for frequency preferences.

Usage:
    python scripts/train_phase2.py --config scripts/config/resnet18.yaml
    python scripts/train_phase2.py --config scripts/config/alexnet.yaml --resume
"""

import argparse
import os
import sys
import time
import yaml
from pathlib import Path
from datetime import datetime

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from data.dataset import load_imagenet_validation, get_denormalize_transform
from frequency.transforms import apply_fft, apply_ifft
from frequency.mask import Learnable2DFrequencyMask
from utils.visualization import (
    visualize_2d_frequency_mask,
    visualize_reconstructed_images,
    plot_training_history
)


class JointTrainingPipeline(nn.Module):
    """
    Phase 2 Pipeline: Image -> FFT -> Learnable Mask -> IFFT -> Trainable Classifier

    Both the frequency mask AND the classifier are trainable.
    This reveals what frequency preferences are built into the architecture.
    """

    def __init__(self, classifier, mask_config=None):
        """
        Args:
            classifier: Classifier model (will be trained from scratch)
            mask_config: Dict with mask configuration
        """
        super(JointTrainingPipeline, self).__init__()

        self.classifier = classifier
        # Enable training for classifier
        for param in self.classifier.parameters():
            param.requires_grad = True
        self.classifier.train()

        # Learnable frequency mask
        if mask_config is None:
            mask_config = {}

        self.freq_mask = Learnable2DFrequencyMask(
            image_size=mask_config.get('image_size', 224),
            init_value=mask_config.get('init_value', 1.0),
            init_std=mask_config.get('init_std', 0.1)
        )

        mask_params = sum(p.numel() for p in self.freq_mask.parameters())
        classifier_params = sum(p.numel() for p in self.classifier.parameters())

        print(f"\nJointTrainingPipeline created:")
        print(f"  Frequency mask: {mask_params:,} trainable params")
        print(f"  Classifier: {classifier_params:,} trainable params")
        print(f"  Total: {mask_params + classifier_params:,} trainable params")

    def forward(self, images):
        """Apply frequency filtering pipeline."""
        fft_result = apply_fft(images)
        masked_fft = self.freq_mask(fft_result)
        reconstructed = apply_ifft(masked_fft)
        outputs = self.classifier(reconstructed)
        return outputs, reconstructed

    def get_mask_params(self):
        """Get frequency mask parameters."""
        return self.freq_mask.parameters()

    def get_classifier_params(self):
        """Get classifier parameters."""
        return self.classifier.parameters()


def load_classifier_from_scratch(arch, num_classes=1000):
    """
    Load a classifier with random initialization (no pretrained weights).

    Args:
        arch: Architecture name ('alexnet', 'vgg16', 'resnet18', 'inception_v3')
        num_classes: Number of output classes

    Returns:
        model: Randomly initialized model
    """
    import torchvision.models as models

    if arch == 'alexnet':
        model = models.alexnet(weights=None, num_classes=num_classes)
    elif arch == 'vgg16':
        model = models.vgg16(weights=None, num_classes=num_classes)
    elif arch == 'resnet18':
        model = models.resnet18(weights=None, num_classes=num_classes)
    elif arch == 'resnet50':
        model = models.resnet50(weights=None, num_classes=num_classes)
    elif arch == 'inception_v3':
        model = models.inception_v3(weights=None, num_classes=num_classes, aux_logits=False)
    else:
        raise ValueError(f"Unknown architecture: {arch}")

    print(f"Loaded {arch} with random initialization")
    return model


def load_config(config_path):
    """Load YAML configuration file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def save_checkpoint(state, checkpoint_dir, filename='checkpoint.pt'):
    """Save training checkpoint."""
    os.makedirs(checkpoint_dir, exist_ok=True)
    filepath = os.path.join(checkpoint_dir, filename)
    torch.save(state, filepath)
    print(f"  Checkpoint saved: {filepath}")


def load_checkpoint(checkpoint_dir, filename='checkpoint.pt'):
    """Load training checkpoint if exists."""
    filepath = os.path.join(checkpoint_dir, filename)
    if os.path.exists(filepath):
        return torch.load(filepath)
    return None


def train_epoch(pipeline, dataloader, optimizer, criterion, device, epoch, log_every=10):
    """Train for one epoch."""
    pipeline.train()

    epoch_loss = 0.0
    epoch_correct = 0
    epoch_total = 0
    batch_count = 0

    for batch_idx, (images, labels) in enumerate(dataloader):
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs, _ = pipeline(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        _, predicted = torch.max(outputs, 1)
        epoch_correct += (predicted == labels).sum().item()
        epoch_total += labels.size(0)
        epoch_loss += loss.item()
        batch_count += 1

        if (batch_idx + 1) % log_every == 0:
            avg_loss = epoch_loss / batch_count
            avg_acc = 100 * epoch_correct / epoch_total
            print(f"  Batch [{batch_idx + 1:4d}] | Loss: {loss.item():.4f} | Acc: {avg_acc:.2f}%")

    return {
        'loss': epoch_loss / batch_count,
        'accuracy': 100 * epoch_correct / epoch_total
    }


def evaluate(pipeline, dataloader, criterion, device):
    """Evaluate on validation set."""
    pipeline.eval()

    total_loss = 0.0
    total_correct = 0
    total_samples = 0
    batch_count = 0

    with torch.no_grad():
        for images, labels in dataloader:
            images, labels = images.to(device), labels.to(device)
            outputs, _ = pipeline(images)
            loss = criterion(outputs, labels)

            _, predicted = torch.max(outputs, 1)
            total_correct += (predicted == labels).sum().item()
            total_samples += labels.size(0)
            total_loss += loss.item()
            batch_count += 1

    return {
        'loss': total_loss / batch_count,
        'accuracy': 100 * total_correct / total_samples
    }


def main():
    parser = argparse.ArgumentParser(description='Phase 2: Joint Training')
    parser.add_argument('--config', type=str, required=True, help='Path to config YAML')
    parser.add_argument('--resume', action='store_true', help='Resume from checkpoint')
    parser.add_argument('--device', type=str, default='cuda', help='Device to use')
    args = parser.parse_args()

    # Load configuration
    config = load_config(args.config)
    arch = config['architecture']

    print("=" * 70)
    print("PHASE 2: JOINT TRAINING (Mask + Classifier from Scratch)")
    print("=" * 70)
    print(f"Architecture: {arch}")
    print(f"Config: {args.config}")
    print(f"Device: {args.device}")
    print("=" * 70)

    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')

    # Setup paths
    results_dir = Path(f"experiments/results/{arch}_phase2")
    checkpoint_dir = results_dir / "checkpoints"
    tensorboard_dir = results_dir / "tensorboard"

    results_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir.mkdir(exist_ok=True)
    tensorboard_dir.mkdir(exist_ok=True)

    # TensorBoard writer
    writer = SummaryWriter(log_dir=str(tensorboard_dir))

    # Training hyperparameters
    train_config = config['training']
    num_epochs = train_config['epochs']
    batch_size = train_config['batch_size']
    mask_lr = train_config['mask_lr']
    classifier_lr = train_config['classifier_lr']
    weight_decay = train_config['weight_decay']
    subset_size = train_config.get('subset_size', 25000)

    print(f"\nTraining Configuration:")
    print(f"  Epochs: {num_epochs}")
    print(f"  Batch size: {batch_size}")
    print(f"  Mask LR: {mask_lr}")
    print(f"  Classifier LR: {classifier_lr}")
    print(f"  Weight decay: {weight_decay}")
    print(f"  Dataset size: {subset_size}")

    # Load dataset
    print("\n[1/4] Loading dataset...")
    _, dataloader = load_imagenet_validation(
        subset_size=subset_size,
        image_size=224,
        batch_size=batch_size
    )

    # Load classifier from scratch
    print("\n[2/4] Loading classifier (random init)...")
    classifier = load_classifier_from_scratch(arch)

    # Create pipeline
    print("\n[3/4] Creating joint training pipeline...")
    mask_config = config.get('mask', {})
    pipeline = JointTrainingPipeline(classifier, mask_config=mask_config).to(device)

    # Optimizer with different LRs for mask and classifier
    optimizer = optim.Adam([
        {'params': pipeline.get_mask_params(), 'lr': mask_lr},
        {'params': pipeline.get_classifier_params(), 'lr': classifier_lr}
    ], weight_decay=weight_decay)

    # Learning rate scheduler
    scheduler = optim.lr_scheduler.StepLR(
        optimizer,
        step_size=train_config.get('lr_step', 30),
        gamma=train_config.get('lr_gamma', 0.1)
    )

    criterion = nn.CrossEntropyLoss()

    # Resume from checkpoint if requested
    start_epoch = 0
    best_acc = 0.0
    history = {'epoch': [], 'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}

    if args.resume:
        checkpoint = load_checkpoint(str(checkpoint_dir))
        if checkpoint:
            pipeline.load_state_dict(checkpoint['pipeline_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            start_epoch = checkpoint['epoch']
            best_acc = checkpoint['best_acc']
            history = checkpoint['history']
            print(f"Resumed from epoch {start_epoch}, best_acc={best_acc:.2f}%")
        else:
            print("No checkpoint found, starting fresh")

    # Training loop
    print("\n[4/4] Starting training...")
    print("=" * 70)

    for epoch in range(start_epoch, num_epochs):
        epoch_start = time.time()
        print(f"\nEpoch {epoch + 1}/{num_epochs}")
        print("-" * 50)

        # Train
        train_metrics = train_epoch(
            pipeline, dataloader, optimizer, criterion, device, epoch,
            log_every=train_config.get('log_every', 20)
        )

        # Evaluate
        val_metrics = evaluate(pipeline, dataloader, criterion, device)

        # Step scheduler
        scheduler.step()

        epoch_time = time.time() - epoch_start

        # Log metrics
        print(f"\nEpoch {epoch + 1} Summary:")
        print(f"  Time: {epoch_time:.1f}s")
        print(f"  Train Loss: {train_metrics['loss']:.4f} | Train Acc: {train_metrics['accuracy']:.2f}%")
        print(f"  Val Loss: {val_metrics['loss']:.4f} | Val Acc: {val_metrics['accuracy']:.2f}%")

        # TensorBoard logging
        writer.add_scalar('Loss/train', train_metrics['loss'], epoch)
        writer.add_scalar('Loss/val', val_metrics['loss'], epoch)
        writer.add_scalar('Accuracy/train', train_metrics['accuracy'], epoch)
        writer.add_scalar('Accuracy/val', val_metrics['accuracy'], epoch)
        writer.add_scalar('LR/mask', optimizer.param_groups[0]['lr'], epoch)
        writer.add_scalar('LR/classifier', optimizer.param_groups[1]['lr'], epoch)

        # Update history
        history['epoch'].append(epoch + 1)
        history['train_loss'].append(train_metrics['loss'])
        history['train_acc'].append(train_metrics['accuracy'])
        history['val_loss'].append(val_metrics['loss'])
        history['val_acc'].append(val_metrics['accuracy'])

        # Save checkpoint
        is_best = val_metrics['accuracy'] > best_acc
        if is_best:
            best_acc = val_metrics['accuracy']

        save_checkpoint({
            'epoch': epoch + 1,
            'pipeline_state_dict': pipeline.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'best_acc': best_acc,
            'history': history
        }, str(checkpoint_dir), 'checkpoint.pt')

        if is_best:
            save_checkpoint({
                'epoch': epoch + 1,
                'pipeline_state_dict': pipeline.state_dict(),
                'best_acc': best_acc
            }, str(checkpoint_dir), 'best_model.pt')
            print(f"  New best model! Acc: {best_acc:.2f}%")

        # Save mask visualization every 10 epochs
        if (epoch + 1) % 10 == 0 or epoch == num_epochs - 1:
            visualize_2d_frequency_mask(
                pipeline.freq_mask,
                title=f"{arch} Phase 2 - Epoch {epoch + 1}",
                save_path=str(results_dir / f"mask_epoch_{epoch + 1}.png")
            )

    # Final outputs
    print("\n" + "=" * 70)
    print("TRAINING COMPLETE")
    print("=" * 70)
    print(f"Best Accuracy: {best_acc:.2f}%")

    # Save final artifacts
    torch.save(pipeline.freq_mask.state_dict(), str(results_dir / "learned_mask.pt"))
    torch.save(history, str(results_dir / "training_history.pt"))

    visualize_2d_frequency_mask(
        pipeline.freq_mask,
        title=f"{arch} Phase 2 - Final Learned Mask",
        save_path=str(results_dir / "learned_mask_final.png")
    )

    # Save summary
    with open(results_dir / "summary.txt", 'w') as f:
        f.write(f"Phase 2 Joint Training Results - {arch}\n")
        f.write("=" * 50 + "\n")
        f.write(f"Architecture: {arch}\n")
        f.write(f"Epochs: {num_epochs}\n")
        f.write(f"Best Accuracy: {best_acc:.2f}%\n")
        f.write(f"Final Train Acc: {history['train_acc'][-1]:.2f}%\n")
        f.write(f"Final Val Acc: {history['val_acc'][-1]:.2f}%\n")
        f.write(f"Mask LR: {mask_lr}\n")
        f.write(f"Classifier LR: {classifier_lr}\n")

    writer.close()
    print(f"\nResults saved to: {results_dir}")


if __name__ == "__main__":
    main()
