import torch
from torch import nn, Tensor


class Bottleneck(nn.Module):
    """
    TDFNet bottleneck: 1×1 Conv projections for audio and video.
    """

    def __init__(
        self,
        audio_encoder_dim: int = 512,
        audio_bottleneck_dim: int = 512,
        video_encoder_dim: int = 512,
        video_bottleneck_dim: int = 64,
    ) -> None:
        """
        Args:
            audio_encoder_dim (int): Encoded audio channel size C_a.
            audio_bottleneck_dim (int): Bottleneck audio channel size B_a.
            video_encoder_dim (int): Encoded video channel size C_v.
            video_bottleneck_dim (int): Bottleneck video channel size B_v.
        """
        super().__init__()
        self.audio_proj = nn.Conv1d(
            audio_encoder_dim,
            audio_bottleneck_dim,
            kernel_size=1,
            bias=False,
        )
        self.video_proj = nn.Conv1d(
            video_encoder_dim,
            video_bottleneck_dim,
            kernel_size=1,
            bias=False,
        )

    def forward(
        self,
        audio: Tensor,
        video: Tensor,
    ) -> tuple[Tensor, Tensor]:
        """
        Produce compact audio/video representations.

        Args:
            audio (Tensor): Audio features of shape [B, C_a, T_a].
            video (Tensor): Video features of shape [B, C_v, T_v].

        Returns:
            tuple[Tensor, Tensor]:
                Tensor: Compact audio features of shape [B, B_a, T_a].
                Tensor: Compact video features of shape [B, B_v, T_v].
        """
        return self.audio_proj(audio), self.video_proj(video)
