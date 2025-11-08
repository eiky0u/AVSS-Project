import torch
from torch import nn, Tensor
import torch.nn.functional as F


class Decoder(nn.Module):
    """
    Transposed-conv decoder from latent [B, S, C, T_a] to waveforms [B, S, T].
    """

    def __init__(
        self,
        num_speakers: int = 2,
        target_len: int = 32000,
        emb_dim: int = 512,
        kernel_size: int = 21,
        stride: int = 10,
    ) -> None:
        """
        Args:
            num_speakers (int): Number of speakers S.
            emb_dim (int): Latent channels per speaker C.
            kernel_size (int): Transposed convolution kernel size (match encoder).
            stride (int): Transposed convolution stride (match encoder).
        """
        super().__init__()
        self.num_speakers = num_speakers
        self.target_len = target_len

        self.decoder = nn.ConvTranspose1d(
            in_channels=emb_dim * num_speakers,
            out_channels=num_speakers,
            kernel_size=kernel_size,
            stride=stride,
            padding=0,
            output_padding=0,
        )

    def forward(
        self,
        audio_latent: Tensor,
    ) -> Tensor:
        """
        Args:
            audio_latent (Tensor): Latent features of shape [B, S, C, T_a].
            target_length (int): Pad/crop to this length T (right side only).

        Returns:
            Tensor: Waveforms per speaker of shape [B, S, T].
        """
        B, S, C, T_a = audio_latent.shape
        x = audio_latent.contiguous().view(B, S * C, T_a)  # [B, S*C, T_a]
        x = self.decoder(x)  # [B, S, T_out]

        T_out = x.shape[-1]
        if T_out < self.target_len:
            x = F.pad(x, (0, self.target_len - T_out))
        elif T_out > self.target_len:
            x = x[..., : self.target_len]
        return x
