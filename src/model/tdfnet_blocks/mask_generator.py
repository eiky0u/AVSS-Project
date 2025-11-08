import torch
from torch import nn, Tensor


class MaskGenerator(nn.Module):
    """
    Per-speaker mask generator with gated activation.
    """

    def __init__(
        self,
        num_speakers: int = 2,
        bottleneck_dim: int = 512,
        out_dim: int = 512,
    ) -> None:
        """
        Args:
            num_speakers (int): Number of speakers S.
            bottleneck_dim (int): Channel size of r (B_a).
            out_dim (int): Output channel size per speaker (C_a).
        """
        super().__init__()
        self.pre = nn.Sequential(
            nn.PReLU(
                num_parameters=bottleneck_dim,
            ),
            nn.Conv1d(
                in_channels=bottleneck_dim,
                out_channels=out_dim * num_speakers,
                kernel_size=1,
            ),
            nn.ReLU(),
        )
        self.gate = nn.Conv1d(
            in_channels=out_dim * num_speakers,
            out_channels=2 * out_dim * num_speakers,
            kernel_size=1,
            bias=False,
        )

        self.num_speakers = num_speakers
        self.out_dim = out_dim

    def forward(
        self,
        audio: Tensor,
        r: Tensor,
    ) -> Tensor:
        """
        Apply per-speaker masks to audio features.

        Args:
            audio (Tensor): Audio features of shape [B, C_a, T_a] (C_a == out_dim).
            r (Tensor): Bottleneck features of shape [B, B_a, T_a].

        Returns:
            Tensor: Masked audio per speaker of shape [B, S, C_a, T_a].
        """
        x = self.pre(r)
        u, v = torch.chunk(self.gate(x), 2, dim=1)
        y = torch.tanh(u) * torch.sigmoid(v)

        B, _, T = y.shape
        masks = y.view(B, self.num_speakers, self.out_dim, T)

        return masks * audio.unsqueeze(1)
