"""
Learnable frequency masks for filtering in the frequency domain.
"""

import torch
import torch.nn as nn


class Learnable2DFrequencyMask(nn.Module):
    """
    Full 2D learnable mask - 224x224 parameters.
    Each frequency pixel has its own learnable weight.
    """

    def __init__(self, image_size=224, init_value=1.0, init_std=0.1):
        """
        Args:
            image_size: Size of the frequency domain mask (default: 224)
            init_value: Initial mean value for mask weights (default: 1.0)
            init_std: Standard deviation for weight initialization (default: 0.1)
        """
        super(Learnable2DFrequencyMask, self).__init__()

        self.image_size = image_size

        # Learnable 2D mask (50,176 parameters for 224x224)
        initial_mask = torch.ones(1, 1, image_size, image_size) * init_value
        initial_mask += torch.randn(1, 1, image_size, image_size) * init_std

        self.mask_weights = nn.Parameter(initial_mask)

    def forward(self, fft_result):
        """
        Apply 2D mask to frequency domain.

        Args:
            fft_result: Complex tensor of shape [B, C, H, W]

        Returns:
            masked_fft: Masked frequency domain tensor [B, C, H, W]
        """
        # Expand mask to match channels (apply same mask to R, G, B)
        mask_expanded = self.mask_weights.expand(-1, fft_result.size(1), -1, -1)

        # Apply mask
        return fft_result * mask_expanded

    def get_mask_visualization(self):
        """
        Return mask as numpy array for visualization.

        Returns:
            mask_array: Numpy array of shape [H, W]
        """
        return self.mask_weights[0, 0].detach().cpu().numpy()
