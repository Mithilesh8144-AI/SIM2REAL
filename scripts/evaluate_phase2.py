#!/usr/bin/env python3
"""
Phase 2 Evaluation Script

Evaluate trained Phase 2 models and compare learned masks across architectures.

Usage:
    python scripts/evaluate_phase2.py --arch resnet18
    python scripts/evaluate_phase2.py --compare-all
"""

import argparse
import os
import sys
from pathlib import Path

import torch
import numpy as np
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).parent.parent))

from data.dataset import load_imagenet_validation, get_denormalize_transform
from frequency.mask import Learnable2DFrequencyMask
from utils.visualization import visualize_2d_frequency_mask


def load_trained_mask(results_dir):
    """Load a trained frequency mask."""
    mask_path = results_dir / "learned_mask.pt"
    if not mask_path.exists():
        return None

    mask = Learnable2DFrequencyMask(image_size=224)
    mask.load_state_dict(torch.load(mask_path))
    return mask


def compute_mask_statistics(mask):
    """Compute statistics of learned mask."""
    weights = mask.mask_weights[0, 0].detach().cpu().numpy()

    # Compute radial profile (average at each distance from center)
    h, w = weights.shape
    cy, cx = h // 2, w // 2
    y, x = np.ogrid[:h, :w]
    r = np.sqrt((x - cx) ** 2 + (y - cy) ** 2)

    r_max = int(np.sqrt(cx ** 2 + cy ** 2))
    radial_profile = []
    for radius in range(r_max):
        ring_mask = (r >= radius) & (r < radius + 1)
        if ring_mask.sum() > 0:
            radial_profile.append(weights[ring_mask].mean())
        else:
            radial_profile.append(0)

    return {
        'mean': weights.mean(),
        'std': weights.std(),
        'min': weights.min(),
        'max': weights.max(),
        'center_value': weights[cy, cx],
        'edge_value': weights[0, 0],
        'radial_profile': np.array(radial_profile),
        'low_freq_mean': weights[cy-20:cy+20, cx-20:cx+20].mean(),
        'high_freq_mean': np.mean([weights[:20, :].mean(), weights[-20:, :].mean(),
                                   weights[:, :20].mean(), weights[:, -20:].mean()])
    }


def compare_masks(architectures, phase='phase2'):
    """Compare learned masks across architectures."""
    print("=" * 70)
    print(f"MASK COMPARISON - {phase.upper()}")
    print("=" * 70)

    results_base = Path("experiments/results")
    masks = {}
    stats = {}

    for arch in architectures:
        if phase == 'phase2':
            results_dir = results_base / f"{arch}_phase2"
        else:
            results_dir = results_base / arch

        mask = load_trained_mask(results_dir)
        if mask is not None:
            masks[arch] = mask
            stats[arch] = compute_mask_statistics(mask)
            print(f"\n{arch}:")
            print(f"  Mean: {stats[arch]['mean']:.4f}")
            print(f"  Std: {stats[arch]['std']:.4f}")
            print(f"  Low-freq mean: {stats[arch]['low_freq_mean']:.4f}")
            print(f"  High-freq mean: {stats[arch]['high_freq_mean']:.4f}")
            print(f"  Low/High ratio: {stats[arch]['low_freq_mean']/stats[arch]['high_freq_mean']:.2f}")
        else:
            print(f"\n{arch}: No trained mask found")

    if len(masks) < 2:
        print("\nNeed at least 2 trained masks for comparison")
        return

    # Create comparison visualization
    fig, axes = plt.subplots(2, len(masks), figsize=(5 * len(masks), 10))

    for i, (arch, mask) in enumerate(masks.items()):
        weights = mask.mask_weights[0, 0].detach().cpu().numpy()

        # Top row: mask visualization
        im = axes[0, i].imshow(weights, cmap='hot')
        axes[0, i].set_title(f"{arch}\nmean={stats[arch]['mean']:.3f}")
        axes[0, i].axis('off')
        plt.colorbar(im, ax=axes[0, i], fraction=0.046)

        # Bottom row: radial profile
        axes[1, i].plot(stats[arch]['radial_profile'])
        axes[1, i].set_xlabel('Distance from center (freq)')
        axes[1, i].set_ylabel('Average mask value')
        axes[1, i].set_title(f'{arch} Radial Profile')
        axes[1, i].grid(True, alpha=0.3)

    plt.tight_layout()
    save_path = results_base / f"mask_comparison_{phase}.png"
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"\nComparison saved: {save_path}")
    plt.close()

    # Radial profile overlay comparison
    plt.figure(figsize=(10, 6))
    for arch in masks:
        plt.plot(stats[arch]['radial_profile'], label=arch, linewidth=2)
    plt.xlabel('Distance from center (frequency)')
    plt.ylabel('Average mask value')
    plt.title(f'Radial Frequency Profiles - {phase.upper()}')
    plt.legend()
    plt.grid(True, alpha=0.3)

    save_path = results_base / f"radial_profiles_{phase}.png"
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"Radial profiles saved: {save_path}")
    plt.close()


def compare_phases(arch):
    """Compare Phase 1 vs Phase 2 masks for same architecture."""
    print("=" * 70)
    print(f"PHASE COMPARISON: {arch}")
    print("=" * 70)

    results_base = Path("experiments/results")
    phase1_dir = results_base / arch
    phase2_dir = results_base / f"{arch}_phase2"

    mask1 = load_trained_mask(phase1_dir)
    mask2 = load_trained_mask(phase2_dir)

    if mask1 is None:
        print(f"No Phase 1 mask found for {arch}")
        return
    if mask2 is None:
        print(f"No Phase 2 mask found for {arch}")
        return

    stats1 = compute_mask_statistics(mask1)
    stats2 = compute_mask_statistics(mask2)

    print(f"\nPhase 1 (Frozen Classifier):")
    print(f"  Mean: {stats1['mean']:.4f}, Low/High ratio: {stats1['low_freq_mean']/stats1['high_freq_mean']:.2f}")

    print(f"\nPhase 2 (Joint Training):")
    print(f"  Mean: {stats2['mean']:.4f}, Low/High ratio: {stats2['low_freq_mean']/stats2['high_freq_mean']:.2f}")

    # Compute mask difference
    w1 = mask1.mask_weights[0, 0].detach().cpu().numpy()
    w2 = mask2.mask_weights[0, 0].detach().cpu().numpy()
    diff = w2 - w1

    fig, axes = plt.subplots(1, 4, figsize=(20, 5))

    im0 = axes[0].imshow(w1, cmap='hot')
    axes[0].set_title(f'{arch} Phase 1\n(Frozen Classifier)')
    plt.colorbar(im0, ax=axes[0])

    im1 = axes[1].imshow(w2, cmap='hot')
    axes[1].set_title(f'{arch} Phase 2\n(Joint Training)')
    plt.colorbar(im1, ax=axes[1])

    im2 = axes[2].imshow(diff, cmap='coolwarm', vmin=-0.5, vmax=0.5)
    axes[2].set_title('Difference (P2 - P1)')
    plt.colorbar(im2, ax=axes[2])

    axes[3].plot(stats1['radial_profile'], label='Phase 1', linewidth=2)
    axes[3].plot(stats2['radial_profile'], label='Phase 2', linewidth=2)
    axes[3].set_xlabel('Distance from center')
    axes[3].set_ylabel('Mask value')
    axes[3].set_title('Radial Profiles')
    axes[3].legend()
    axes[3].grid(True, alpha=0.3)

    for ax in axes[:3]:
        ax.axis('off')

    plt.tight_layout()
    save_path = results_base / f"{arch}_phase_comparison.png"
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"\nComparison saved: {save_path}")
    plt.close()


def main():
    parser = argparse.ArgumentParser(description='Evaluate Phase 2 Results')
    parser.add_argument('--arch', type=str, help='Architecture to evaluate')
    parser.add_argument('--compare-all', action='store_true', help='Compare all architectures')
    parser.add_argument('--compare-phases', type=str, help='Compare P1 vs P2 for architecture')
    parser.add_argument('--phase', type=str, default='phase2', choices=['phase1', 'phase2'],
                        help='Phase to compare')
    args = parser.parse_args()

    architectures = ['alexnet', 'vgg16', 'resnet18', 'inception_v3']

    if args.compare_all:
        compare_masks(architectures, phase=args.phase)
    elif args.compare_phases:
        compare_phases(args.compare_phases)
    elif args.arch:
        compare_masks([args.arch], phase=args.phase)
    else:
        print("Usage:")
        print("  --compare-all        Compare all architectures")
        print("  --arch ARCH          Evaluate single architecture")
        print("  --compare-phases ARCH  Compare Phase 1 vs Phase 2")


if __name__ == "__main__":
    main()
