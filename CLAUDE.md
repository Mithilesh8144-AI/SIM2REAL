# Project Context: Frequency Analysis of Neural Networks

## Research Goal

Replicate "What do Deep Networks Like to See?" (Palacio et al., ECCV 2018) using frequency domain analysis instead of autoencoders.

**Question:** Which frequency components of images do different neural networks rely on for classification?

---

## Methodology

**Pipeline:** Image → FFT → Learnable Frequency Mask → IFFT → Frozen Classifier

- **FFT:** Converts image to frequency domain (fixed, not learned)
- **Frequency Mask:** Learnable weights (THIS is what trains)
- **IFFT:** Converts back to pixels (fixed, not learned)  
- **Frozen Classifier:** Pre-trained model (stays frozen, provides gradients)

Only the frequency mask parameters update during training.

---

## What Has Been Completed (ResNet-18)

### Experiment 1: Radial Frequency Mask
- 10 learnable parameters (one per frequency band)
- Trained to optimize accuracy
- Result: +0.62% improvement
- Learned pattern: slight preference for low frequencies

### Experiment 2: Frequency Band Isolation (Diagnostic)
- Hard masks keeping ONLY specific frequency ranges
- Tested: LOW (0-25%), MID (25-75%), HIGH (75-100%)
- **Key Finding:** LOW only = 49.92% accuracy, MID/HIGH only = <1%
- ResNet-18 heavily depends on low frequencies (shapes/colors)

### Experiment 3: 2D Frequency Mask  
- 50,176 learnable parameters (224×224 spatial mask)
- Full control over frequency domain
- Trained for 5 epochs (LR=0.05, Adam)
- Learned mask shows: bright center (keep low freq), dark edges (suppress high freq)
- Visual reconstructions look nearly identical but improve accuracy

---

## Current State

All three experiments completed successfully for ResNet-18. The approach is validated and working.

Dataset: ImageNet validation (5,000 images), Batch size: 64

---

## Next Phase

Extend the 2D mask training (Experiment 3) to other classifier architectures:
- AlexNet
- VGG-16
- Inception v3
- (Optional) ResNet-50

Then compare the learned masks across architectures to understand:
- Which networks use more/less frequency information
- Differences between older (AlexNet, VGG) and newer (ResNet, Inception) architectures
- Architectural patterns in frequency preferences

This replicates the paper's main contribution: comparing what different networks "like to see."