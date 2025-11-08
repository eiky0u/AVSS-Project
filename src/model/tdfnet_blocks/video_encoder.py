import torch
from torch import nn, Tensor

from . import get_lipreading_model


class Backbone(nn.Module):
    """
    Wrapper around a pretrained lipreading backbone.

    The backbone returns [B, T_v, C_v]; this wrapper transposes to [B, C_v, T_v].
    """

    def __init__(
        self,
        config_path: str,
        freeze: bool = True,
    ) -> None:
        """
        Args:
            config_path (str): Path to the lipreading model config.
            freeze (bool): If True, disable grads and keep backbone in eval mode.
        """
        super().__init__()
        self.encoder = get_lipreading_model(config_path=config_path)
        self.freeze = freeze

        if self.freeze:
            self.encoder.requires_grad_(False)
            self.encoder.eval()

    def train(self, mode: bool = True):
        """
        Keep the frozen encoder in eval mode even if parent .train(True) is called.
        """
        super().train(mode)
        if self.freeze:
            self.encoder.eval()
        return self

    def forward(
        self,
        video: Tensor,
    ) -> Tensor:
        """
        Encode video frames into a temporal feature sequence.

        Args:
            video (Tensor): Input video tensor of shape [B, F, H, W]

        Returns:
            Tensor: Encoded video features of shape [B, C_v, T_v], where T_v = F is the
            number of frames and C_v is the video feature channel size.
        """
        if self.freeze:
            with torch.inference_mode():
                feats = self.encoder(video, lengths=video.shape[1])  # [B, F, C_v]
        else:
            feats = self.encoder(video)
        return feats.transpose(1, 2)  # [B, C_v, F]


class VideoEncoder(nn.Module):
    """
    Video encoder that processes S mouths and projects their concatenated features to C_v.
    """

    def __init__(
        self,
        config_path: str,
        freeze: bool = True,
        num_speakers: int = 2,
        video_encoder_dim: int = 512,
    ) -> None:
        """
        Args:
            config_path (str): Path to the lipreading model config.
            freeze (bool): If True, disable grads and keep backbone in eval mode.
            num_speakers (int): Number of speakers S.
            video_encoder_dim (int): Encoded video channel size C_v.
        """
        super().__init__()
        self.backbone = Backbone(config_path=config_path, freeze=freeze)
        self.proj = nn.Sequential(
            nn.Conv1d(
                in_channels=video_encoder_dim * num_speakers,
                out_channels=video_encoder_dim,
                kernel_size=1,
            ),
            nn.GroupNorm(
                num_groups=1,
                num_channels=video_encoder_dim,
            ),
        )

        self.video_encoder_dim = video_encoder_dim

    def forward(
        self,
        mouths: Tensor,
    ) -> Tensor:
        """
        Encode S mouths with a backbone expecting [B, F, H, W], then fuse.

        Args:
            mouths (Tensor): Input tensor of shape [B, S, F, H, W].

        Returns:
            Tensor: Fused video features of shape [B, C_v, T_v], with T_v = F.
        """
        B, S, F, H, W = mouths.shape

        x = mouths.contiguous().view(B * S, 1, F, H, W)  # [B*S, 1, F, H, W]
        feats = self.backbone(x)  # [B*S, C_v, T_v]

        feats = feats.view(B, S, self.video_encoder_dim, -1)  # [B, S, C_v, T_v]
        feats = feats.flatten(1, 2)  # [B, S*C_v, T_v]

        return self.proj(feats)  # [B, C_v, T_v]
