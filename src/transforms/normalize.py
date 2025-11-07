import torch
from torch import nn


class NormalizeMouth(nn.Module):
    """
    Normalizes pixel values of the mouth region tensor
    from the [0, 255] range to [0.0, 1.0].

    Args:
        mouth (torch.Tensor): Input tensor containing mouth images,
            expected in the [0, 255] range (uint8 or float).
            Shape: (F, H, W).

    Returns:
        torch.Tensor: Normalized tensor of type float32
            with values in the [0.0, 1.0] range.
    """

    def forward(self, mouth: torch.Tensor) -> torch.Tensor:
        return mouth.float() / 255.0
