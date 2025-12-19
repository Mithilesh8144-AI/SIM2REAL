# Experiments

This folder contains individual notebooks for Experiment 3 (2D Learnable Frequency Mask) across different architectures.

## Structure

```
experiments/
├── README.md                    # This file
├── resnet18_experiment3.ipynb   # ResNet-18 experiment
├── alexnet_experiment3.ipynb    # (TODO) AlexNet experiment
├── vgg16_experiment3.ipynb      # (TODO) VGG-16 experiment
├── inception_experiment3.ipynb  # (TODO) Inception v3 experiment
└── results/                     # Training results
    ├── resnet18/
    │   ├── learned_mask.pt      # Trained mask weights
    │   ├── learned_mask.png     # Mask visualization
    │   ├── training_history.pt  # Loss & accuracy curves
    │   └── summary.txt          # Experiment summary
    ├── alexnet/
    ├── vgg16/
    └── inception/
```

## Experiment 3: 2D Learnable Frequency Mask

**Goal:** Identify which frequency components each neural network architecture relies on for classification.

**Approach:**
- Input Image → FFT → Learnable 2D Mask (224×224) → IFFT → Frozen Classifier
- Only the frequency mask is trained (50,176 parameters)
- Classifier remains frozen and provides gradients

**Dataset:**
- ImageNet validation subset (10,000 images)
- Cached at `data/imagenet_10k_cache/`

## Running an Experiment

1. Navigate to the experiments folder
2. Open the desired notebook (e.g., `resnet18_experiment3.ipynb`)
3. Run all cells sequentially
4. Results will be saved to `results/<model_name>/`

## Comparing Results

After running experiments for multiple architectures:
1. Compare learned masks visually
2. Analyze center (low freq) vs edges (high freq) weights
3. Compare accuracy improvements
4. Identify architectural patterns in frequency preferences

## Next Steps

1. Complete ResNet-18 experiment
2. Duplicate notebook for AlexNet, VGG-16, Inception v3
3. Run all experiments with same hyperparameters
4. Create comparison notebook to analyze differences
