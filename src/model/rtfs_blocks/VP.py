import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Union

class DepthwiseConv1dBN(nn.Module):
    def __init__(
        self,
        channels: int,
        kernel_size: int = 4,
        stride: int = 1,
        padding: Union[str, int] = "same",
        bias: bool = False,
    ):
        super().__init__()
        self.channels = channels
        self.kernel_size = kernel_size
        self.stride = stride

        if isinstance(padding, int):
            self._use_explicit_pad = False
            conv_padding = padding
        elif padding == "same":
            if (kernel_size % 2 == 1) and (stride == 1):
                self._use_explicit_pad = False
                conv_padding = (kernel_size - 1) // 2
            else:
                self._use_explicit_pad = True
                conv_padding = 0
        else:
            raise ValueError("padding must be 'same' or int")

        self.conv = nn.Conv1d(
            channels,
            channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=conv_padding,
            groups=channels,
            bias=bias,
        )
        self.bn = nn.BatchNorm1d(channels)

    def _calc_same_pad(self, L: int) -> tuple[int, int]:
        out_len = math.ceil(L / self.stride)
        pad_needed = max(0, (out_len - 1) * self.stride + (self.kernel_size - 1) + 1 - L)
        pad_left = pad_needed // 2
        pad_right = pad_needed - pad_left
        return pad_left, pad_right

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, C, T]
        if getattr(self, "_use_explicit_pad", False):
            T = x.shape[-1]
            pad_l, pad_r = self._calc_same_pad(T)
            if pad_l or pad_r:
                x = F.pad(x, (pad_l, pad_r))
        x = self.conv(x)
        return self.bn(x)



class PointwiseConv1dBN(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, bias: bool = False):
        super().__init__()
        self.conv = nn.Conv1d(in_ch, out_ch, kernel_size=1, bias=bias)
        self.bn = nn.BatchNorm1d(out_ch)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.bn(self.conv(x))


class MHSAConvFFN1D(nn.Module):
    def __init__(self, dim: int, num_heads: int = 8, ffn_dim: int = 128, dropout: float = 0.1):
        super().__init__()
        self.ln1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(embed_dim=dim, num_heads=num_heads, dropout=dropout, batch_first=True)
        self.drop1 = nn.Dropout(dropout)

        self.ln2 = nn.LayerNorm(dim)
        self.ffn = nn.Sequential(
            nn.Conv1d(dim, ffn_dim, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv1d(ffn_dim, dim, kernel_size=1),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, D, T]
        B, D, T = x.shape
        xt = x.transpose(1, 2)  # [B, T, D]

        y = self.ln1(xt)
        y, _ = self.attn(y, y, y, need_weights=False)  # [B, T, D]
        xt = xt + self.drop1(y)

        z = self.ln2(xt)          # [B, T, D]
        z = z.transpose(1, 2)     # [B, D, T]
        z = self.ffn(z)           # [B, D, T]
        z = z.transpose(1, 2)     # [B, T, D] 
        xt = xt + z               # [B, T, D]

        return xt.transpose(1, 2) # [B, D, T]


class TFAR1D(nn.Module):
    def __init__(self, channels: int, kernel_size: int = 4):
        super().__init__()
        pad = "same"  
        self.W1 = DepthwiseConv1dBN(channels, kernel_size=kernel_size, stride=1, padding=pad)
        self.W2 = DepthwiseConv1dBN(channels, kernel_size=kernel_size, stride=1, padding=pad)
        self.W3 = DepthwiseConv1dBN(channels, kernel_size=kernel_size, stride=1, padding=pad)

    def forward(self, m: torch.Tensor, n: torch.Tensor) -> torch.Tensor:
        B, D, Tm = m.shape
        gate = torch.sigmoid(self.W1(n))             # [B, D, Tn]
        gate_up = F.interpolate(gate, size=Tm, mode="nearest")  # [B, D, Tm]
        n3_up  = F.interpolate(self.W3(n), size=Tm, mode="nearest")
        return gate_up * self.W2(m) + n3_up          # [B, D, Tm]


class VPBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,     
        D: int = 64,         
        q: int = 4,           
        num_heads: int = 8,
        ffn_dim: int = 128,
        dropout: float = 0.1,
        down_kernel: int = 4,  
    ):
        super().__init__()
        assert q >= 1, "q должен быть ≥ 1"

        self.reduce = PointwiseConv1dBN(in_channels, D)
        self.downs = nn.ModuleList([
            DepthwiseConv1dBN(D, kernel_size=down_kernel, stride=2, padding="same")
            for _ in range(q - 1)
        ])

        self.attn = MHSAConvFFN1D(D, num_heads=num_heads, ffn_dim=ffn_dim, dropout=dropout)
        self.tfar = TFAR1D(D, kernel_size=4)

        self.expand = nn.Conv1d(D, in_channels, kernel_size=1)

    def forward(self, v0: torch.Tensor) -> torch.Tensor:
        # v0: [B, C_v, T_v]
        B, C, T = v0.shape

        x = self.reduce(v0)             # [B, D, T]
        levels = [x]
        for down in self.downs:
            x = down(x)                 # [B, D, ceil(T/2)]
            levels.append(x)           

        T_min = levels[-1].shape[-1]
        pooled = [F.adaptive_avg_pool1d(lv, T_min) for lv in levels]
        V_G = torch.stack(pooled, dim=0).sum(dim=0)  # [B, D, T_min]
        V_bar = self.attn(V_G) 

        fused_levels = [self.tfar(levels[i], V_bar) for i in range(len(levels))]
        rec = fused_levels[-1]  
        for i in range(len(levels) - 2, -1, -1):
            rec = self.tfar(fused_levels[i], rec) + levels[i]  

        out = self.expand(rec)       
        out = out + v0     
        return out