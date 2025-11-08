import torch
from torch import nn, Tensor
import torch.nn.functional as F


class NormDWConv(nn.Module):
    """
    Normalized depthwise 1D convolution block H_{k,s}.
    Uses GroupNorm with num_groups=1 (gLN) to normalize across channels and time per sample.
    """

    def __init__(
        self,
        hidden_dim: int,
        kernel_size: int,
        stride: int,
    ) -> None:
        """
        Args:
            hidden_dim (int): Working channel dimension D (input/output channels).
            kernel_size (int): Depthwise kernel size k (typically odd, e.g., 3 or 5).
            stride (int): Convolution stride s. Use 1 to keep length, 2 to downsample.
        """
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(
                in_channels=hidden_dim,
                out_channels=hidden_dim,
                kernel_size=kernel_size,
                stride=stride,
                groups=hidden_dim,
                padding=(kernel_size - 1) // 2,
                bias=False,
            ),
            nn.GroupNorm(
                num_groups=1,
                num_channels=hidden_dim,
            ),  # analogue of gLN (global layer norm)
        )

    def forward(
        self,
        x: Tensor,
    ) -> Tensor:
        """
        Apply depthwise Conv1d + gLN.

        Args:
            x (Tensor): Input of shape [B, D, T_in].

        Returns:
            Tensor: Output of shape [B, D, T_out].
        """
        return self.conv(x)


class BottomUpDownsampling(nn.Module):
    """
    Bottom-up pathway that builds a multi-scale temporal pyramid and aggregates it into `g_in`.

    Structure:
      - H_{k,1}: resolution-preserving depthwise/gLN (stride=1).
      - `num_layers` times H_{k,2}: cascade of depthwise/gLN blocks (stride=2) reducing temporal resolution.
    """

    def __init__(
        self,
        bottleneck_dim: int = 512,
        hidden_dim: int = 512,
        kernel_size: int = 21,
        num_layers: int = 5,
    ):
        """
        Args:
            bottleneck_dim (int): Bottleneck channel size at module I/O (B_a or B_v).
            hidden_dim (int): Working channel size D inside the refinement path.
            kernel_size (int): Depthwise kernel size k for all stages (typically odd).
            num_layers (int): Number of stride-2 downsampling stages.
        """
        super().__init__()
        self.in_pw = nn.Conv1d(
            in_channels=bottleneck_dim,
            out_channels=hidden_dim,
            kernel_size=1,
        )
        self.in_dw = NormDWConv(
            hidden_dim=hidden_dim,
            kernel_size=kernel_size,
            stride=1,
        )
        self.down_convs = nn.ModuleList(
            [
                NormDWConv(
                    hidden_dim=hidden_dim,
                    kernel_size=kernel_size,
                    stride=2,
                )
                for _ in range(num_layers)
            ]
        )

    def forward(
        self,
        x: Tensor,
    ) -> tuple[list[Tensor], Tensor]:
        """
        Downsample and aggregate multi-scale features with adaptive pooling.

        Args:
            x (Tensor): Input of shape [B, B_*, T_in], where B_* is a bottleneck (B_a or B_v).

        Returns:
            (feats, g_in):
                - feats (list[Tensor]): [F^0, F^1, ..., F^L], where F^i ∈ ℝ^{B×D×(T_in/2^i)}.
                - g_in (Tensor): Aggregated coarsest-rate feature of shape [B, D, T_L],
                  obtained by summing {F^i} after adaptive-avg-pooling them to length T_L = len(F^L).
        """
        x = self.in_dw(self.in_pw(x))
        feats = [x]
        for conv in self.down_convs:
            x = conv(x)
            feats.append(x)

        T_L = feats[-1].shape[-1]
        pooled = [F.adaptive_avg_pool1d(f, T_L) for f in feats]
        g_in = torch.stack(pooled, dim=0).sum(dim=0)
        return feats, g_in


class RecurrentOperatorGRU(nn.Module):
    """
    Bidirectional GRU with linear projection and residual.
    """

    def __init__(
        self,
        hidden_dim: int = 512,
        num_layers: int = 1,
        dropout: float = 0.1,
    ) -> None:
        """
        Args:
            hidden_dim (int): Working channel size D (also GRU input/output size).
            num_layers (int): Number of stacked GRU layers.
            dropout (float): Drop probability applied after the linear projection.
        """
        super().__init__()
        self.gru = nn.GRU(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0.0,
            bidirectional=True,
            batch_first=True,
        )
        self.proj = nn.Sequential(
            nn.Linear(
                in_features=hidden_dim * 2,
                out_features=hidden_dim,
            ),
            nn.Dropout(p=dropout),
        )

    def forward(
        self,
        x: Tensor,
    ) -> Tensor:
        """
        Compute the recurrent refinement and return the global output `g_out`.

        Args:
            x (Tensor): Input of shape [B, D, T_L].

        Returns:
            Tensor: g_out ∈ ℝ^{B×D×T_L}, the recurrently refined global representation.
        """
        y, _ = self.gru(x.transpose(1, 2))
        g_out = x + self.proj(y).transpose(1, 2)
        return g_out


class FeedForwardNetwork(nn.Module):
    """
    Convolutional FFN for the MHSA block (1×, k, 1) with channel expansion D → 2D → D.
    """

    def __init__(
        self,
        hidden_dim: int = 512,
        kernel_size: int = 5,
        dropout: float = 0.1,
    ) -> None:
        """
        Args:
            hidden_dim (int): Working channel size D (input/output).
            kernel_size (int): Middle convolution kernel size k.
            dropout (float): Drop probability after the final projection.
        """
        super().__init__()
        self.ffn = nn.Sequential(
            nn.Conv1d(
                in_channels=hidden_dim,
                out_channels=hidden_dim,
                kernel_size=1,
            ),
            nn.ReLU(),
            nn.Conv1d(
                in_channels=hidden_dim,
                out_channels=hidden_dim * 2,
                kernel_size=kernel_size,
                padding=(kernel_size - 1) // 2,
            ),
            nn.ReLU(),
            nn.Conv1d(
                in_channels=hidden_dim * 2,
                out_channels=hidden_dim,
                kernel_size=1,
            ),
            nn.Dropout(p=dropout),
        )

    def forward(self, x: Tensor) -> Tensor:
        """
        Args:
            x (Tensor): [B, D, T]
        Returns:
            Tensor: [B, D, T]
        """
        return self.ffn(x)


class RecurrentOperatorMHSA(nn.Module):
    """
    MHSA-based recurrent operator with residuals and a conv FFN (D → 2D → D).
    """

    def __init__(
        self,
        hidden_dim: int = 512,
        num_heads: int = 8,
        dropout: float = 0.1,
        kernel_size: int = 5,
    ) -> None:
        """
        Args:
            hidden_dim (int): Working channel size D (attention embedding size).
            num_heads (int): Number of attention heads (D must be divisible by num_heads).
            dropout (float): Drop probability after attention and inside the FFN.
            kernel_size (int): Middle convolution kernel size k in the FFN.
        """
        super().__init__()
        assert hidden_dim % num_heads == 0
        self.mhsa = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )

        self.dropout = nn.Dropout(p=dropout)
        self.ffn = FeedForwardNetwork(
            hidden_dim=hidden_dim,
            kernel_size=kernel_size,
            dropout=dropout,
        )

    def forward(
        self,
        x: Tensor,
    ) -> Tensor:
        """
        Apply MHSA and the convolutional FFN with residuals; return `g_out`.

        Args:
            x (Tensor): Input of shape [B, D, T_L].

        Returns:
            Tensor: g_out ∈ ℝ^{B×D×T_L}, the attention-refined global representation.
        """
        x_t = x.transpose(1, 2)
        y, _ = self.mhsa(x_t, x_t, x_t, need_weights=False)
        x = x + self.dropout(y.transpose(1, 2))
        g_out = x + self.ffn(x)
        return g_out


class InjectionSum(nn.Module):
    """
    Gated residual fusion I_k: inject top-down/global features into local features using depthwise H_{k,1}.
    """

    def __init__(
        self,
        hidden_dim: int = 512,
        kernel_size: int = 5,
    ) -> None:
        """
        Args:
            hidden_dim (int): Working channel size D used in all branches.
            kernel_size (int): Depthwise kernel size k for H_k in each branch.
        """
        super().__init__()

        self.q_inj = NormDWConv(
            hidden_dim=hidden_dim,
            kernel_size=kernel_size,
            stride=1,
        )

        self.s_inj = nn.Sequential(
            NormDWConv(
                hidden_dim=hidden_dim,
                kernel_size=kernel_size,
                stride=1,
            ),
            nn.Sigmoid(),
        )

        self.r_inj = NormDWConv(
            hidden_dim=hidden_dim,
            kernel_size=kernel_size,
            stride=1,
        )

    def forward(
        self,
        x: Tensor,
        y: Tensor,
    ) -> Tensor:
        """
        Fuse local features x with top-down features y.

        Args:
            x (Tensor): Local features, shape [B, D, T_x].
            y (Tensor): Top-down/global features, shape [B, D, T_y] with T_y ≤ T_x.

        Returns:
            Tensor: Fused features, shape [B, D, T_x].
        """
        q = self.q_inj(x)
        s = self.s_inj(y)
        r = self.r_inj(y)

        if y.shape[-1] < x.shape[-1]:
            s = F.interpolate(input=s, size=x.shape[-1], mode="nearest")
            r = F.interpolate(input=r, size=x.shape[-1], mode="nearest")

        return q * s + r


class TopDownFusion(nn.Module):
    """
    Two-stage top-down fusion:
      (1) Per-scale injection: F_i' = I_k(F_i, g_out).
      (2) Pyramid collapse: iteratively apply I_1 and add residual skips from {F_i} to recover the finest map.
    """

    def __init__(
        self,
        hidden_dim: int = 512,
        bottleneck_dim: int = 512,
        kernel_size: int = 5,
    ) -> None:
        """
        Args:
            hidden_dim (int): Working channel size D shared across all scales.
            bottleneck_dim (int): Output bottleneck channel size B_* (B_a or B_v), restored at the end by 1×1 conv.
            kernel_size (int): Kernel size k used in I_k (per-scale injection).
        """
        super().__init__()
        self.i_k = InjectionSum(
            hidden_dim=hidden_dim,
            kernel_size=kernel_size,
        )
        self.i_1 = InjectionSum(
            hidden_dim=hidden_dim,
            kernel_size=1,
        )
        self.out_pw = nn.Conv1d(
            in_channels=hidden_dim,
            out_channels=bottleneck_dim,
            kernel_size=1,
        )

    def forward(self, feats: list[Tensor], g_out: Tensor) -> Tensor:
        """
        Perform top-down fusion.

        Args:
            feats (list[Tensor]): Multi-scale features [F_0, ..., F_q], with F_0 finest and F_q coarsest,
                each F_i ∈ ℝ^{B×D×(T/2^i)}.
            g_out (Tensor): Global/top-down context at the coarsest rate, shape [B, D, T/2^q].

        Returns:
            Tensor: Collapsed fused feature at the finest resolution, projected back to bottleneck channels,
                shape [B, B_*, T], where B_* ∈ {B_a, B_v}.
        """
        q = len(feats) - 1

        fused = [self.i_k(Fi, g_out) for Fi in feats]  # list of F'_i

        out = self.i_1(fused[q - 1], fused[q]) + feats[q - 1]
        for j in range(q - 2, -1, -1):
            out = self.i_1(fused[j], out) + feats[j]

        return self.out_pw(out)
