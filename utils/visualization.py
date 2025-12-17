"""
Visualization utilities for frequency masks and reconstructed images.
"""

import matplotlib.pyplot as plt
import numpy as np
import torch


def visualize_2d_frequency_mask(mask_model, title="2D Frequency Mask", save_path=None):
    """
    Visualize the 2D frequency mask.

    Args:
        mask_model: Learnable2DFrequencyMask instance
        title: Plot title
        save_path: Optional path to save figure
    """
    mask_vis = mask_model.get_mask_visualization()

    fig, ax = plt.subplots(1, 1, figsize=(10, 9))

    # 2D mask visualization
    im = ax.imshow(mask_vis, cmap='viridis')
    ax.set_title(title, fontsize=14)
    ax.set_xlabel('Frequency X')
    ax.set_ylabel('Frequency Y')
    plt.colorbar(im, ax=ax, label='Mask Weight')

    # Add statistics
    stats_text = f"Min: {mask_vis.min():.3f}\nMax: {mask_vis.max():.3f}\nMean: {mask_vis.mean():.3f}"
    ax.text(0.02, 0.98, stats_text, transform=ax.transAxes,
            fontsize=10, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"  ✓ Saved mask visualization to {save_path}")

    plt.show()


def visualize_reconstructed_images(original, reconstructed, labels, mask_model,
                                   denormalize_fn, num_images=4, save_path=None):
    """
    Visualize original images, reconstructed images, and differences.

    Args:
        original: Original images tensor [B, C, H, W]
        reconstructed: Reconstructed images tensor [B, C, H, W]
        labels: Labels tensor [B]
        mask_model: Learnable2DFrequencyMask instance
        denormalize_fn: Function to denormalize images
        num_images: Number of images to display
        save_path: Optional path to save figure
    """
    fig, axes = plt.subplots(num_images, 4, figsize=(16, 4 * num_images))

    if num_images == 1:
        axes = axes.reshape(1, -1)

    for i in range(num_images):
        # Original
        orig_img = denormalize_fn(original[i]).cpu().numpy()
        axes[i, 0].imshow(np.transpose(orig_img, (1, 2, 0)))
        axes[i, 0].set_title(f'Original\nClass {labels[i].item()}', fontsize=10)
        axes[i, 0].axis('off')

        # Reconstructed (what network sees)
        recon_img = denormalize_fn(reconstructed[i]).cpu().numpy()
        axes[i, 1].imshow(np.transpose(recon_img, (1, 2, 0)))
        axes[i, 1].set_title(f'After Mask\n(Network Input)', fontsize=10)
        axes[i, 1].axis('off')

        # Difference (amplified 5x for visibility)
        diff = np.abs(orig_img - recon_img) * 5
        axes[i, 2].imshow(np.transpose(diff, (1, 2, 0)))
        axes[i, 2].set_title('Difference (5x)', fontsize=10)
        axes[i, 2].axis('off')

        # Frequency mask applied
        mask_vis = mask_model.get_mask_visualization()
        axes[i, 3].imshow(mask_vis, cmap='RdBu_r', vmin=0.5, vmax=1.5)
        axes[i, 3].set_title('2D Freq Mask', fontsize=10)
        axes[i, 3].axis('off')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"  ✓ Saved reconstruction visualization to {save_path}")

    plt.show()


def plot_training_history(history, save_path=None):
    """
    Plot training loss and accuracy curves.

    Args:
        history: Dict with keys 'epoch', 'loss', 'accuracy'
        save_path: Optional path to save figure
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    epochs = history['epoch']

    # Loss curve
    ax1.plot(epochs, history['loss'], 'o-', linewidth=2, markersize=8, color='#e74c3c')
    ax1.set_title('Training Loss', fontsize=14)
    ax1.set_xlabel('Epoch', fontsize=12)
    ax1.set_ylabel('Loss', fontsize=12)
    ax1.grid(True, alpha=0.3)

    # Accuracy curve
    ax2.plot(epochs, history['accuracy'], 'o-', linewidth=2, markersize=8, color='#27ae60')
    ax2.set_title('Training Accuracy', fontsize=14)
    ax2.set_xlabel('Epoch', fontsize=12)
    ax2.set_ylabel('Accuracy (%)', fontsize=12)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"  ✓ Saved training history to {save_path}")

    plt.show()
