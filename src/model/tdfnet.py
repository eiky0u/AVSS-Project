import torch
from torch import nn, Tensor

from src.model.tdfnet_blocks.audio_encoder import AudioEncoder
from src.model.tdfnet_blocks.video_encoder import VideoEncoder
from src.model.tdfnet_blocks.bottleneck import Bottleneck
from src.model.tdfnet_blocks.refinement import RefinementModule, AudioSubnetwork
from src.model.tdfnet_blocks.mask_generator import MaskGenerator
from src.model.tdfnet_blocks.decoder import Decoder


class TDFNet(nn.Module):
    """
    Full audiovisual speech separation network (TDFNet-style).

    Pipeline:
      1) Encoders:
         - AudioEncoder: raw waveform → audio features X ∈ ℝ^{B×C_a×T_a}.
         - VideoEncoder: S mouths (frames) → fused video features Y ∈ ℝ^{B×C_v×T_v}.
      2) Bottleneck:
         - Project to compact streams:
           a ∈ ℝ^{B×B_a×T_a}, v ∈ ℝ^{B×B_v×T_v}.
      3) Refinement (iterative):
         - First `num_fusion_blocks` steps (γ_j): shared AudioSubnetwork + per-step VideoSubnetwork + FusionSubnetwork,
           residual skips to initial (a0, v0) each step.
         - Remaining `num_refinement_blocks - num_fusion_blocks` steps (α_j): shared AudioSubnetwork only,
           with residuals to a0.
      4) Masking:
         - MaskGenerator emits S masks over audio encoder features X (C_a channels),
           applies them to get per-speaker latents Z ∈ ℝ^{B×S×C_a×T_a}.
      5) Decoder:
         - ConvTranspose1d upsampling back to waveforms per speaker:
           preds ∈ ℝ^{B×S×T}.
    """

    def __init__(
        self,
        pretrained_config_path: str,
        # general params
        num_speakers: int = 2,
        target_len: int = 32000,
        num_fusion_blocks: int = 3,
        num_refinement_blocks: int = 16,
        # audio params
        audio_encoder_dim: int = 512,
        audio_encoder_kernel_size: int = 21,
        audio_encoder_stride: int = 10,
        audio_bottleneck_dim: int = 512,
        audio_hidden_dim: int = 512,
        audio_kernel_size: int = 5,
        audio_num_sampling_layers: int = 5,
        audio_num_rnn: int = 1,
        audio_dropout: float = 0.1,
        # video params
        video_encoder_dim: int = 512,  # backbone output C_v (always 512)
        video_bottleneck_dim: int = 64,
        video_hidden_dim: int = 64,
        video_kernel_size: int = 3,
        video_num_sampling_layers: int = 4,
        video_num_heads: int = 8,
        video_dropout: float = 0.1,
    ) -> None:
        """
        Args:
            pretrained_config_path (str): Path to the pretrained lipreading backbone config.
            num_speakers (int): Number of speakers / mouths S.
            target_len (int): Target waveform length T for the decoder output (right-pad if needed).
            num_fusion_blocks (int): Number of audiovisual refinement stages.
            num_refinement_blocks (int): Total number of refinement stages.
            audio_encoder_dim (int): Audio encoder output channels C_a.
            audio_encoder_kernel_size (int): Audio encoder Conv1d kernel size.
            audio_encoder_stride (int): Audio encoder Conv1d stride (defines T_a downsampling).
            audio_bottleneck_dim (int): Audio bottleneck channels B_a (refinement I/O).
            audio_hidden_dim (int): Audio refinement working channels D (inside TDF blocks).
            audio_kernel_size (int): Depthwise kernel k for audio pyramid/fusion.
            audio_num_sampling_layers (int): # of stride-2 downsampling layers in audio pyramid.
            audio_num_rnn (int): number of GRU layers in audio recurrent operator.
            audio_dropout (float): Drop probability in audio recurrent operator.
            video_encoder_dim (int): Video backbone output channels C_v (always 512).
            video_bottleneck_dim (int): Video bottleneck channels B_v (refinement I/O).
            video_hidden_dim (int): Video refinement working channels D_v.
            video_kernel_size (int): Depthwise/FFN kernel k for video pyramid/fusion.
            video_num_sampling_layers (int): # of stride-2 downsampling layers in video pyramid.
            video_num_heads (int): number of attention heads in video MHSA (D_v % heads == 0).
            video_dropout (float): Drop probability in video recurrent operator.
        """
        super().__init__()

        self.video_encoder = VideoEncoder(
            config_path=pretrained_config_path,
            freeze=True,
            num_speakers=num_speakers,
            video_encoder_dim=video_encoder_dim,
        )

        self.audio_encoder = AudioEncoder(
            emb_dim=audio_encoder_dim,
            kernel_size=audio_encoder_kernel_size,
            stride=audio_encoder_stride,
        )

        self.bottleneck = Bottleneck(
            audio_encoder_dim=audio_encoder_dim,
            audio_bottleneck_dim=audio_bottleneck_dim,
            video_encoder_dim=video_encoder_dim,
            video_bottleneck_dim=video_bottleneck_dim,
        )

        self.audio_shared = AudioSubnetwork(
            bottleneck_dim=audio_bottleneck_dim,
            hidden_dim=audio_hidden_dim,
            kernel_size=audio_kernel_size,
            num_sampling_layers=audio_num_sampling_layers,
            num_rnn_layers=audio_num_rnn,
            dropout=audio_dropout,
        )

        self.fusion_blocks = nn.ModuleList(
            [
                RefinementModule(
                    only_audio=False,
                    video_bottleneck_dim=video_bottleneck_dim,
                    video_hidden_dim=video_hidden_dim,
                    video_kernel_size=video_kernel_size,
                    video_num_sampling_layers=video_num_sampling_layers,
                    video_num_heads=video_num_heads,
                    video_dropout=video_dropout,
                )
                for _ in range(num_fusion_blocks)
            ]
        )

        self.audio_blocks = nn.ModuleList(
            [
                RefinementModule(
                    only_audio=True,
                )
                for _ in range(num_refinement_blocks - num_fusion_blocks)
            ]
        )

        self.mask_generator = MaskGenerator(
            num_speakers=num_speakers,
            bottleneck_dim=audio_bottleneck_dim,
            out_dim=audio_encoder_dim,
        )

        self.decoder = Decoder(
            num_speakers=num_speakers,
            target_len=target_len,
            emb_dim=audio_encoder_dim,
            kernel_size=audio_encoder_kernel_size,
            stride=audio_encoder_stride,
        )

    def forward(self, mix, mouths, **batch) -> dict[str, Tensor]:
        """
        Forward pass: encode → bottleneck → refine → mask → decode.

        Args:
            mix (Tensor): Mixture waveform, shape [B, 1, T].
            mouths (Tensor): Video mouths, shape [B, S, F, H, W],
                where S is the number of speakers, F—frames.

        Returns:
            dict[str, Tensor]:
                - "preds": Separated waveforms per speaker, Tensor of shape [B, S, T].
        """
        # Encoders
        encoded_audio = self.audio_encoder(mix)  # [B, C_a, T_a]
        encoded_video = self.video_encoder(mouths)  # [B, C_v, T_v]

        # Bottleneck projections
        a, v = self.bottleneck(
            audio=encoded_audio,  # a: [B, B_a, T_a]
            video=encoded_video,  # v: [B, B_v, T_v]
        )

        # Audiovisual refinement with residuals to initial states
        a0, v0 = a, v
        for fusion_block in self.fusion_blocks:
            a, v = fusion_block(audio=a, video=v, audio_module=self.audio_shared)
            a = a + a0
            v = v + v0

        # Audio-only refinement with residuals to initial audio state
        for audio_block in self.audio_blocks:
            a = audio_block(audio=a, video=None, audio_module=self.audio_shared)
            a = a + a0

        # Masking over encoder features and decoding to time domain
        audio_latent = self.mask_generator(audio=encoded_audio, r=a)  # [B, S, C_a, T_a]
        preds = self.decoder(audio_latent=audio_latent)  # [B, S, T]
        return {"preds": preds}

    def __str__(self):
        """
        Summary with parameter counts.
        """
        all_parameters = sum(p.numel() for p in self.parameters())
        trainable_parameters = sum(
            p.numel() for p in self.parameters() if p.requires_grad
        )
        result_info = super().__str__()
        result_info += f"\nAll parameters: {all_parameters}"
        result_info += f"\nTrainable parameters: {trainable_parameters}"
        return result_info
