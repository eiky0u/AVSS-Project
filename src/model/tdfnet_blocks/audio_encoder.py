import torch
from torch import nn, Tensor


class AudioEncoder(nn.Module):
    """
    TDFNet audio encoder using a 1D convolution.
    """

    def __init__(
        self,
        emb_dim: int = 512,
        kernel_size: int = 21,
        stride: int = 10,
    ) -> None:
        """
        Args:
            emb_dim (int): Output feature dimension C_a.
            kernel_size (int): Size of the convolution kernel.
            stride (int): Convolution stride.
        """
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv1d(
                in_channels=1,
                out_channels=emb_dim,
                kernel_size=kernel_size,
                stride=stride,
                bias=False,  # no use because of further norm
            ),
            nn.GroupNorm(
                num_groups=1,
                num_channels=emb_dim,
            ),  # analogue of gLN (global layer norm)
            nn.ReLU(),
        )

    def forward(
        self,
        mix: Tensor,
    ) -> Tensor:
        """
        Encode raw audio waveform into a latent representation.

        Args:
            mix (Tensor): Input waveform of shape [B, 1, T].

        Returns:
            Tensor: Encoded audio features of shape [B, C_a, T_a].
        """
        return self.encoder(mix)
