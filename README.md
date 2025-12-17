# Frequency Analysis of Neural Networks

Replicate "What do Deep Networks Like to See?" (Palacio et al., ECCV 2018) using frequency domain analysis instead of autoencoders.

## Research Question

Which frequency components of images do different neural networks rely on for classification?

## Methodology

**Pipeline:** Image → FFT → Learnable Frequency Mask → IFFT → Frozen Classifier

- **FFT:** Converts image to frequency domain (fixed, not learned)
- **Frequency Mask:** Learnable weights (THIS is what trains)
- **IFFT:** Converts back to pixels (fixed, not learned)
- **Frozen Classifier:** Pre-trained model (stays frozen, provides gradients)

Only the frequency mask parameters update during training.

## Project Structure

```
SIM2REAL/
├── data/
│   ├── __init__.py
│   └── dataset.py              # ImageNet dataset loading
│
├── frequency/
│   ├── __init__.py
│   ├── transforms.py           # FFT/IFFT operations
│   ├── mask.py                 # Learnable2DFrequencyMask
│   └── pipeline.py             # FrequencyFilterPipeline
│
├── models/
│   ├── __init__.py
│   └── resnet18.py             # ResNet-18 model (add more models here)
│
├── utils/
│   ├── __init__.py
│   └── visualization.py        # Visualization utilities
│
├── checkpoints/                # Saved model checkpoints
│
├── train.py                    # Main training script
├── FrequencyAnalysis.ipynb     # Original Jupyter notebook
├── CLAUDE.md                   # Project context
└── pixi.toml                   # Dependencies
```

## Usage

### Training

Run the main training script:

```bash
python train.py
```

This will:
1. Load ImageNet validation dataset (5,000 images)
2. Load pre-trained ResNet-18 classifier (frozen)
3. Create 2D frequency mask (50,176 learnable parameters)
4. Train for 5 epochs with Adam optimizer
5. Save visualizations and checkpoints

### Importing Modules

You can import and use the modular components:

```python
from data import load_imagenet_validation
from models import load_resnet18
from frequency import FrequencyFilterPipeline

# Load data
dataset, dataloader = load_imagenet_validation(subset_size=5000)

# Load classifier
classifier = load_resnet18(pretrained=True, device='cuda')

# Create pipeline with 2D mask
pipeline = FrequencyFilterPipeline(classifier).to('cuda')

# Train...
```

## Adding New Models

To test other architectures (AlexNet, VGG-16, Inception v3, etc.):

1. Create a new file in `models/` (e.g., `models/alexnet.py`)
2. Implement similar to `models/resnet18.py`:
   - `load_<model_name>()` function
   - `test_classifier_baseline()` function
3. Update `models/__init__.py` to export the new functions
4. Import and use in your training script

Example for AlexNet:

```python
# models/alexnet.py
import torchvision.models as models

def load_alexnet(pretrained=True, device='cuda'):
    model = models.alexnet(weights=models.AlexNet_Weights.IMAGENET1K_V1 if pretrained else None)
    model = model.to(device)
    model.eval()
    return model
```

## Key Components

### Data Module (`data/`)
- `HFImageNetDataset`: Custom dataset wrapper
- `load_imagenet_validation()`: Load ImageNet validation set
- `get_denormalize_transform()`: For visualization

### Frequency Module (`frequency/`)
- `apply_fft()`: 2D Fourier transform
- `apply_ifft()`: Inverse Fourier transform
- `Learnable2DFrequencyMask`: 224×224 learnable frequency mask
- `FrequencyFilterPipeline`: Complete end-to-end pipeline

### Models Module (`models/`)
- `load_resnet18()`: Load pre-trained ResNet-18
- `test_classifier_baseline()`: Test baseline accuracy
- Add more architectures here for comparison

### Utils Module (`utils/`)
- `visualize_2d_frequency_mask()`: Visualize learned mask
- `visualize_reconstructed_images()`: Compare original/reconstructed
- `plot_training_history()`: Plot loss and accuracy curves

## Results (ResNet-18)

- **Baseline Accuracy:** ~70% on 5K ImageNet validation images
- **2D Mask Parameters:** 50,176 (224×224 grid)
- **Training:** 5 epochs, LR=0.05, Adam optimizer
- **Learned Pattern:** Bright center (preserve low frequencies), dark edges (suppress high frequencies)

## Next Steps

Extend to other architectures:
- AlexNet
- VGG-16
- Inception v3
- ResNet-50

Compare learned frequency preferences across different network architectures.

## Dependencies

Managed via `pixi.toml`:
- PyTorch (>=2.0)
- torchvision
- HuggingFace datasets
- matplotlib
- numpy
- pillow

Install with: `pixi install`
