import torch.nn as nn
import torch

class GlobalLayerNorm(nn.Module):
    def __init__(self, num_channels: int = 1, eps: float=1e-5):
        super(GlobalLayerNorm, self).__init__()
        self.num_channels = num_channels
        self.eps = eps

        self.norm = nn.GroupNorm(num_groups=1, num_channels=self.num_channels, eps=self.eps)

    def forward(self, x: torch.Tensor):
        return self.norm(x)
