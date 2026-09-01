# Frequency Analysis of Neural Networks

## Research Goal
Replicate "What do Deep Networks Like to See?" using frequency domain analysis.
**Question:** Which frequency components do different neural networks rely on for classification?

---

## Methodology

**Pipeline:** Image → FFT → Learnable Frequency Mask → IFFT → Classifier

- Only the frequency mask parameters update during training
- Classifier is frozen (Phase 1) or trainable (Phase 2)

---

## Project Structure

```
SIM2REAL/
├── data/           # Dataset loading (ImageNet via HuggingFace)
├── frequency/      # FFT/IFFT transforms, learnable masks, pipeline
├── models/         # Pre-trained classifier loaders
├── utils/          # Visualization utilities
├── experiments/    # Jupyter notebooks and results
└── train.py        # Phase 1 training script
```

---

## Current Status

### Phase 1: Frozen Classifier (COMPLETE)
**Question:** "What frequencies do pre-trained ImageNet models use?"

| Architecture | Status | Baseline | Final | Accuracy Change |
|--------------|--------|----------|-------|-----------------|
| ResNet-18    | DONE   | 65.70%   | 67.24% | +1.54%         |
| AlexNet      | DONE   | 52.48%   | 47.48% | -5.00%         |
| VGG-16       | DONE   | 69.48%   | 65.77% | -3.70%         |
| ResNet-50    | DONE   | 74.18%   | 74.56% | +0.38%         |

### Phase 2: Random Init + Joint Training (FAILED - all versions)
**Question:** "What does a model learn from scratch?"
- v1: Memorization (no validation split)
- v2: Mask collapsed despite regularization
- v3: Massive overfitting (93% train / 9% val) — 20k images too few for 11.7M params
- **Conclusion:** Cannot train from scratch on small data. Abandoned.

### Phase 3: Pretrained + Fine-tuning + Learnable Mask (IN PROGRESS)
**Question:** "What frequency preferences emerge when pretrained classifier and mask co-adapt?"

| Architecture | Status | Data | Notes |
|--------------|--------|------|-------|
| ResNet-18    | IN PROGRESS | 100k | Phase 3.2 with warmup — ready to run |
| AlexNet      | TODO   | -    | - |
| VGG-16       | TODO   | -    | - |
| ResNet-50    | TODO   | -    | - |

- Phase 3 (25k, LR=1e-4): FAILED — val dropped every epoch (99% train / 59% val, 40% gap)
- Phase 3.1 (100k, LR=1e-4): FAILED — val starts below baseline from epoch 1 (67% vs 74% baseline), drops to 61%
- **Root cause:** Even LR=1e-4 too high — classifier overwrites pretrained features from epoch 1
- **Phase 3.2 fix:** Two-stage warmup (freeze classifier for 10 epochs, then unfreeze with LR=1e-5)
- If Phase 3.2 mask correlates with Phase 1 mask → preference is **inductive bias**

---

## Key Results So Far

### Phase 1 Findings

**ResNet-18:**
- Baseline: 65.70% → After mask: 67.24% (+1.54%)
- Artifacts: `experiments/results/resnet18/`

**AlexNet:**
- Baseline: 52.48% → After mask: 47.48% (-5.00%)
- Artifacts: `experiments/results/alexnet/`

**VGG-16:**
- Baseline: 69.48% → After mask: 65.77% (-3.70%)
- Artifacts: `experiments/results/vgg16/`

**ResNet-50:**
- Baseline: 74.18% → After mask: 74.56% (+0.38%)
- Artifacts: `experiments/results/resnet50/`

**Emerging Pattern:**
- ResNet architectures (+1.54%, +0.38%) improve with frequency filtering
- AlexNet (-5.00%) and VGG-16 (-3.70%) degrade
- Hypothesis: Skip connections enable frequency-selective processing

---

## Current TODOs

**Immediate:**
1. Run Phase 3.2 notebook (`experiments/Phase3/resnet18_phase3.2.ipynb`)
   - Stage 1 (ep 1-10): Classifier frozen, mask trains alone
   - Stage 2 (ep 11+): Classifier unfrozen with LR=1e-5
2. Monitor that val stays above baseline during warmup stage
3. Compare Phase 3.2 mask with Phase 1 mask (correlation analysis)

**Next Steps:**
1. If Phase 3.2 works → run for AlexNet, VGG-16, ResNet-50
2. Cross-architecture mask comparison (Phase 1 vs Phase 3.2)
3. Write up findings on inductive bias

---

## Commands Reference

**Data Preparation:**
```bash
pixi run python scripts/cache_imagenet_100k.py  # One-time: cache 100k images
```

**Phase 1 Notebooks:** `experiments/Phase1/`
**Phase 2 Notebooks:** `experiments/Phase2/` (archived, failed)
**Phase 3 Notebooks:** `experiments/Phase3/`
- `resnet18_phase3.2.ipynb` ← current active notebook

**Results:**
- Phase 1: `experiments/results/<arch>/`
- Phase 2: `experiments/results/resnet18_phase2_v*/` (failed)
- Phase 3: `experiments/results/resnet18_phase3.2/`

**Detailed experiment log:** `experiments/EXPERIMENTS.md`

**Technical Notes:**
- PyTorch 2.6+: Use `weights_only=False` in `torch.load()` for checkpoints
- Mask normalization (mean=1.0) prevents collapse without regularization
- 100k cache uses ImageNet **train** split (validation only has 50k)
- 25k cache uses ImageNet **validation** split
