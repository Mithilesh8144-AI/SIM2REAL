"""
Script to cache 100k ImageNet images for Phase 3 training.

Uses the same batched approach as the original 25k cache to avoid memory issues.
Downloads from the TRAIN split (validation only has 50k images).

Run with:
    pixi run python scripts/cache_imagenet_100k.py

This will create a local cache at data/imagenet_100k_cache/
"""

import os
import sys
import gc
import shutil
from pathlib import Path
from datasets import load_dataset, Dataset, concatenate_datasets, load_from_disk

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

TOTAL_IMAGES = 100000
BATCH_SIZE = 100  # Download 100 at a time (same as original 25k script)
CACHE_PATH = PROJECT_ROOT / "data" / "imagenet_100k_cache"
TEMP_DIR = PROJECT_ROOT / "data" / "temp_batches_100k"


def main():
    print("=" * 60)
    print(f"ImageNet {TOTAL_IMAGES:,} Image Download (Batched Mode)")
    print("=" * 60)
    print(f"Strategy: Download {BATCH_SIZE} images at a time, save immediately")
    print(f"Source: ILSVRC/imagenet-1k train split")
    print(f"Cache: {CACHE_PATH}")

    if CACHE_PATH.exists():
        print(f"\nCache already exists at: {CACHE_PATH}")
        response = input("Delete and recreate? (y/n): ").strip().lower()
        if response != 'y':
            print("Exiting.")
            return
        shutil.rmtree(CACHE_PATH)
        print("Deleted existing cache.")

    # Create temp directory for batches
    TEMP_DIR.mkdir(parents=True, exist_ok=True)

    # Calculate number of batches
    num_batches = (TOTAL_IMAGES + BATCH_SIZE - 1) // BATCH_SIZE
    print(f"\nWill download in {num_batches} batches of {BATCH_SIZE} images each")

    # Start streaming dataset from TRAIN split (has 1.2M images)
    print("\nConnecting to ImageNet (train split)...")
    streaming_ds = load_dataset(
        "ILSVRC/imagenet-1k",
        split="train",
        streaming=True
    )

    # Download in batches
    current_batch_data = {'image': [], 'label': []}
    total_downloaded = 0
    batch_num = 0

    print("\nStarting download...\n")

    for i, sample in enumerate(streaming_ds):
        # Add sample to current batch
        current_batch_data['image'].append(sample['image'])
        current_batch_data['label'].append(sample['label'])
        total_downloaded += 1

        # When batch is full OR we've hit our total, save it
        if len(current_batch_data['image']) >= BATCH_SIZE or total_downloaded >= TOTAL_IMAGES:
            batch_num += 1

            # Convert batch to dataset
            batch_dataset = Dataset.from_dict(current_batch_data)

            # Save batch to disk
            batch_path = os.path.join(TEMP_DIR, f"batch_{batch_num:04d}")
            batch_dataset.save_to_disk(batch_path)

            print(f"Batch {batch_num}/{num_batches} saved: {len(batch_dataset)} images "
                  f"(Total: {total_downloaded:,}/{TOTAL_IMAGES:,})")

            # Clear memory explicitly
            current_batch_data = {'image': [], 'label': []}
            del batch_dataset
            gc.collect()

            # Stop if we've reached our target
            if total_downloaded >= TOTAL_IMAGES:
                break

    print(f"\nDownload complete! {total_downloaded:,} images in {batch_num} batches")

    # Combine all batches
    print("\nCombining batches...")
    batch_datasets = []
    batch_dirs = sorted(TEMP_DIR.iterdir())

    for i, batch_path in enumerate(batch_dirs):
        if batch_path.is_dir() and batch_path.name.startswith("batch_"):
            ds = load_from_disk(str(batch_path))
            batch_datasets.append(ds)
            if (i + 1) % 50 == 0:
                print(f"  Loaded {i + 1}/{len(batch_dirs)} batches...")

    imagenet_subset = concatenate_datasets(batch_datasets)
    print(f"Combined dataset: {len(imagenet_subset):,} images")

    # Save final dataset
    print(f"\nSaving final dataset to {CACHE_PATH}...")
    imagenet_subset.save_to_disk(str(CACHE_PATH))
    print("Saved!")

    # Clean up temp batches
    print("\nCleaning up temporary files...")
    shutil.rmtree(TEMP_DIR)
    print("Cleaned up!")

    # Verify
    print("\nVerifying cache...")
    loaded = load_from_disk(str(CACHE_PATH))
    print(f"Loaded {len(loaded):,} images from cache")

    print("\n" + "=" * 60)
    print("DONE!")
    print("=" * 60)
    print(f"Cache location: {CACHE_PATH}")
    print(f"Total images: {len(loaded):,}")
    print("\nYou can now run the Phase 3 notebook!")


if __name__ == "__main__":
    main()
