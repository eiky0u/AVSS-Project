import torch
from torch import nn, Tensor
import torch.nn.functional as F

from . import BottomUpDownsampling
from . import RecurrentOperatorGRU
from . import RecurrentOperatorMHSA
from . import TopDownFusion


class AudioSubnetwork(nn.Module):
    """
    Audio refinement sub-network (shared across all iterations in TDFNet).

    Pipeline:
      1) Bottom-up downsampling builds a temporal pyramid and aggregated input `g_in`.
      2) Recurrent operator (BiGRU → projection → residual) produces `g_out`.
      3) Top-down fusion injects `g_out` into all scales and collapses the pyramid
         back to the original resolution, projecting channels to B_a.
    """

    def __init__(
        self,
        bottleneck_dim: int = 512,
        hidden_dim: int = 512,
        kernel_size: int = 5,
        num_sampling_layers: int = 5,
        num_rnn_layers: int = 1,
        dropout: float = 0.1,
    ) -> None:
        """
        Args:
            bottleneck_dim (int): Audio bottleneck channels B_a (I/O of this subnetwork).
            hidden_dim (int): Working channel size D used inside the refinement path.
            kernel_size (int): Depthwise kernel size used in the pyramid and fusion (typically odd).
            num_sampling_layers (int): Number of stride-2 downsampling layers in the pyramid.
            num_rnn_layers (int): Number of GRU layers in the recurrent operator.
            dropout (float): Drop probability in the recurrent operator (after the projection).
        """
        super().__init__()

        self.downsampling = BottomUpDownsampling(
            bottleneck_dim=bottleneck_dim,
            hidden_dim=hidden_dim,
            kernel_size=kernel_size,
            num_layers=num_sampling_layers,
        )
        self.recurrent_operator = RecurrentOperatorGRU(
            hidden_dim=hidden_dim,
            dropout=dropout,
            num_layers=num_rnn_layers,
        )
        self.fusion = TopDownFusion(
            hidden_dim=hidden_dim,
            bottleneck_dim=bottleneck_dim,
            kernel_size=kernel_size,
        )

    def forward(
        self,
        audio: Tensor,
    ) -> Tensor:
        """
        Refine audio features and return the fused output.

        Args:
            audio (Tensor): Audio bottleneck features of shape [B, B_a, T_a].

        Returns:
            Tensor: Refined audio features after top-down fusion,
                shape [B, B_a, T_a].
        """
        feats, g_in = self.downsampling(audio)
        g_out = self.recurrent_operator(g_in)
        return self.fusion(feats, g_out)


class VideoSubnetwork(nn.Module):
    """
    Video refinement sub-network (per-iteration, not shared).

    Pipeline:
      1) Bottom-up downsampling → multi-scale pyramid + aggregated `g_in`.
      2) MHSA-based recurrent operator (MHSA → residual → conv FFN → residual) → `g_out`.
      3) Top-down fusion injects `g_out` and collapses the pyramid back to the original resolution,
         projecting channels to B_v.
    """

    def __init__(
        self,
        bottleneck_dim: int = 512,
        hidden_dim: int = 64,
        kernel_size: int = 3,
        num_sampling_layers: int = 4,
        num_heads: int = 8,
        dropout: float = 0.1,
    ) -> None:
        """
        Args:
            bottleneck_dim (int): Video bottleneck channels B_v (I/O of this subnetwork).
            hidden_dim (int): Working channel size D_v used inside the refinement path.
            kernel_size (int): Middle convolution kernel size for the FFN and pyramid stages.
            num_sampling_layers (int): Number of stride-2 downsampling layers in the pyramid.
            num_heads (int): Number of attention heads (hidden_dim must be divisible by num_heads).
            dropout (float): Drop probability in the MHSA block/FFN.
        """
        super().__init__()

        self.downsampling = BottomUpDownsampling(
            bottleneck_dim=bottleneck_dim,
            hidden_dim=hidden_dim,
            kernel_size=kernel_size,
            num_layers=num_sampling_layers,
        )
        self.recurrent_operator = RecurrentOperatorMHSA(
            hidden_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout,
            kernel_size=kernel_size,
        )
        self.fusion = TopDownFusion(
            hidden_dim=hidden_dim,
            bottleneck_dim=bottleneck_dim,
            kernel_size=kernel_size,
        )

    def forward(
        self,
        video: Tensor,
    ) -> Tensor:
        """
        Refine video features and return the fused output.

        Args:
            video (Tensor): Video bottleneck features of shape [B, B_v, T_v].

        Returns:
            Tensor: Refined video features after top-down fusion,
                shape [B, B_v, T_v].
        """
        feats, g_in = self.downsampling(video)
        g_out = self.recurrent_operator(g_in)
        return self.fusion(feats, g_out)


class FusionSubnetwork(nn.Module):
    """
    Cross-Modal Fusion sub-network γ_j.

    For each iteration it returns two outputs (a_j, v_j):
      a_j = κ_a([X || φ(Y)])  → shape [B, B_a, T_a]
      v_j = κ_v([Y || φ(X)])  → shape [B, B_v, T_v]

    where:
      - X ≡ audio features of shape [B, B_a, T_a]
      - Y ≡ video features of shape [B, B_v, T_v]
      - φ is nearest-neighbor interpolation along time to match lengths
      - κ_a / κ_v are 1×1 Conv1d (no bias) followed by gLN (GroupNorm with num_groups=1).
    """

    def __init__(
        self,
        audio_bottleneck_dim: int = 512,
        video_bottleneck_dim: int = 64,
    ) -> None:
        """
        Args:
            audio_bottleneck_dim (int): Audio channel size B_a.
            video_bottleneck_dim (int): Video channel size B_v.
        """
        super().__init__()
        self.audio_proj = nn.Sequential(
            nn.Conv1d(
                audio_bottleneck_dim + video_bottleneck_dim,
                audio_bottleneck_dim,
                kernel_size=1,
                bias=False,
            ),
            nn.GroupNorm(
                num_groups=1,
                num_channels=audio_bottleneck_dim,
            ),
        )
        self.video_proj = nn.Sequential(
            nn.Conv1d(
                audio_bottleneck_dim + video_bottleneck_dim,
                video_bottleneck_dim,
                kernel_size=1,
                bias=False,
            ),
            nn.GroupNorm(
                num_groups=1,
                num_channels=video_bottleneck_dim,
            ),
        )

    def forward(
        self,
        audio: Tensor,
        video: Tensor,
    ) -> tuple[Tensor, Tensor]:
        """
        Fuse audio into video and video into audio (symmetric cross-modal fusion).

        Args:
            audio (Tensor): Audio features X of shape [B, B_a, T_a].
            video (Tensor): Video features Y of shape [B, B_v, T_v].

        Returns:
            tuple[Tensor, Tensor]:
                - a_j (Tensor): Audio-side fused features κ_a([X || φ(Y)]),
                    shape [B, B_a, T_a].
                - v_j (Tensor): Video-side fused features κ_v([Y || φ(X)]),
                    shape [B, B_v, T_v].
        """
        fused_audio = torch.cat(
            [
                audio,
                F.interpolate(
                    input=video,
                    size=audio.shape[-1],
                    mode="nearest",
                ),
            ],
            dim=1,
        )
        fused_video = torch.cat(
            [
                video,
                F.interpolate(
                    input=audio,
                    size=video.shape[-1],
                    mode="nearest",
                ),
            ],
            dim=1,
        )

        return self.audio_proj(fused_audio), self.video_proj(fused_video)


class RefinementModule(nn.Module):
    """
    Refinement wrapper that assembles per-iteration video TDF sub-network and cross-modal fusion.
    The audio refinement sub-network is NOT owned here — it is shared across all iterations
    and must be passed in via `audio_module` at call time.
    """

    def __init__(
        self,
        only_audio: bool = False,
        audio_bottleneck_dim: int = 512,
        video_bottleneck_dim: int = 64,
        video_hidden_dim: int = 64,
        video_kernel_size: int = 3,
        video_num_sampling_layers: int = 4,
        video_num_heads: int = 8,
        video_dropout: float = 0.1,
    ) -> None:
        """
        Args:
            only_audio (bool): If True, run only the audio path (no video or fusion).
            audio_bottleneck_dim (int): Audio bottleneck channels B_a (used by fusion head).
            video_bottleneck_dim (int): Video bottleneck channels B_v.
            video_hidden_dim (int): Working channels inside the video refinement path (D_v).
            video_kernel_size (int): Depthwise/FFN kernel size for the video path.
            video_num_sampling_layers (int): Number of stride-2 downsampling layers in the video pyramid.
            video_num_heads (int): Number of attention heads in the video MHSA (D_v must be divisible by this).
            video_dropout (float): Drop probability in the video recurrent operator/FFN.
        """
        super().__init__()
        self.only_audio = only_audio

        if not self.only_audio:
            self.video_network = VideoSubnetwork(
                bottleneck_dim=video_bottleneck_dim,
                hidden_dim=video_hidden_dim,
                kernel_size=video_kernel_size,
                num_sampling_layers=video_num_sampling_layers,
                num_heads=video_num_heads,
                dropout=video_dropout,
            )
            self.fusion_network = FusionSubnetwork(
                audio_bottleneck_dim=audio_bottleneck_dim,
                video_bottleneck_dim=video_bottleneck_dim,
            )

    def forward(
        self,
        audio: Tensor,
        video: Tensor | None = None,
        audio_module: AudioSubnetwork | None = None,
    ) -> tuple[Tensor, Tensor] | Tensor:
        """
        Run one refinement iteration.

        Args:
            audio (Tensor): Audio bottleneck features, [B, B_a, T_a].
            video (Tensor | None): Video bottleneck features, [B, B_v, T_v].
                                   Must be provided when `only_audio=False`.
            audio_module (AudioSubnetwork | None): Shared audio subnetwork to use for this iteration.
                                                   Required for both modes.

        Returns:
            Tensor | tuple[Tensor, Tensor]:
                - If `only_audio=True`: refined audio features [B, B_a, T_a].
                - Else: (a_j, v_j) with shapes [B, B_a, T_a] and [B, B_v, T_v], respectively.
        """
        a_ref = audio_module(audio)
        if self.only_audio:
            return a_ref

        v_ref = self.video_network(video)
        return self.fusion_network(audio=a_ref, video=v_ref)
