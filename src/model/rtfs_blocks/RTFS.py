import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from src.model.modules.GLN import GlobalLayerNorm


class LayerNormChannels(nn.Module):
    def __init__(self, num_channels: int, eps: float = 1e-5):
        super().__init__()
        self.ln = nn.LayerNorm(num_channels, eps=eps)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, C, T, F] -> [B, T, F, C] -> LN -> back
        x = x.permute(0, 2, 3, 1).contiguous()
        x = self.ln(x)
        return x.permute(0, 3, 1, 2).contiguous()

class DepthwiseConv2dGLN(nn.Module):
    def __init__(self, channels: int, kernel_size=(4,4), stride=(1,1), bias=False, eps=1e-5):
        super().__init__()
        ky, kx = kernel_size if isinstance(kernel_size, (tuple, list)) else (kernel_size, kernel_size)
        sy, sx = stride if isinstance(stride, (tuple, list)) else (stride, stride)

        self.kernel_size = (ky, kx)
        self.stride = (sy, sx)
        self.use_explicit_pad = not ((ky % 2 == 1 and sy == 1) and (kx % 2 == 1 and sx == 1))

        conv_padding = (0, 0) if self.use_explicit_pad else ((ky - 1)//2, (kx - 1)//2)
        self.conv = nn.Conv2d(channels, channels, kernel_size=(ky, kx),
                              stride=(sy, sx), padding=conv_padding,
                              groups=channels, bias=bias)
        self.norm = GlobalLayerNorm(channels, eps=eps)

    @staticmethod
    def _calc_same_pad(L, K, S):
        out_len = math.ceil(L / S)
        pad_needed = max(0, (out_len - 1) * S + (K - 1) + 1 - L)
        left = pad_needed // 2
        right = pad_needed - left
        return left, right

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, C, T, F]
        if self.use_explicit_pad:
            B, C, T, Freq = x.shape
            py_l, py_r = self._calc_same_pad(T, self.kernel_size[0], self.stride[0])
            px_l, px_r = self._calc_same_pad(Freq, self.kernel_size[1], self.stride[1])
            if py_l or py_r or px_l or px_r:
                x = F.pad(x, (px_l, px_r, py_l, py_r))
        x = self.conv(x)
        return self.norm(x)

class PointwiseConv2dGLN(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, bias=False):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size=1, bias=bias)
        self.norm = GlobalLayerNorm(out_ch)

    def forward(self, x):
        return self.norm(self.conv(x))



class _BiRNN(nn.Module):
    def __init__(self, in_size: int, hidden_size: int, num_layers: int = 4, dropout: float = 0.0):
        super().__init__()
        self.rnn = nn.LSTM(input_size=in_size,
                              hidden_size=hidden_size,
                              num_layers=num_layers,
                              batch_first=True,
                              dropout=dropout if num_layers > 1 else 0.0,
                              bidirectional=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, L, C]
        y, _ = self.rnn(x)  # [B, L, 2*h]
        return y

# ...existing code...
class TFDomainAttention(nn.Module):
    """
    TF-domain self-attention (аналог TF-GridNet узла): MHSA над токенами T*F.
    Вход/выход: [B, D, T, F].
    """
    def __init__(self, dim: int, num_heads: int = 4, dropout: float = 0.1):
        super().__init__()
        self.ln = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(embed_dim=dim, num_heads=num_heads,
                                          dropout=dropout, batch_first=True)
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, D, T, Freq = x.shape
        # [B, T*F, D]
        tokens = x.permute(0, 2, 3, 1).contiguous().view(B, T * Freq, D)
        y = self.ln(tokens)
        y, _ = self.attn(y, y, y, need_weights=False)
        tokens = tokens + self.drop(y)
        return tokens.view(B, T, Freq, D).permute(0, 3, 1, 2).contiguous()
# class TFDomainAttention(nn.Module):
#     """
#     TF-GridNet style attention with chunking over the (B*F) dimension to control memory.
#     Input: x [B, D, T, F]
#     Output: out [B, D, T, F]
#     Args:
#         dim, num_heads, dropout: as usual
#         chunk_size: number of (B*F) rows to process at once (tune to fit GPU memory)
#     """
#     def __init__(self, dim: int, num_heads: int = 4, dropout: float = 0.1,
#                  ff_hidden_mult: int = 4, chunk_size: int = 32):
#         super().__init__()
#         self.dim = dim
#         self.num_heads = num_heads
#         self.dropout = dropout
#         self.chunk_size = chunk_size

#         self.ln1 = nn.LayerNorm(dim)
#         # batch_first=True so input is [batch, seq, dim]
#         self.attn = nn.MultiheadAttention(embed_dim=dim, num_heads=num_heads,
#                                           dropout=dropout, batch_first=True)
#         self.drop_att = nn.Dropout(dropout)

#         hidden = dim * ff_hidden_mult
#         self.ln2 = nn.LayerNorm(dim)
#         self.ffn = nn.Sequential(
#             nn.Linear(dim, hidden),
#             nn.ReLU(inplace=True),
#             nn.Dropout(dropout),
#             nn.Linear(hidden, dim),
#             nn.Dropout(dropout),
#         )

#     def _chunked_attn(self, xf: torch.Tensor) -> torch.Tensor:
#         # xf: [B*F, T, D]
#         Bf, T, D = xf.shape
#         out = xf.new_empty(Bf, T, D)
#         for i in range(0, Bf, self.chunk_size):
#             xi = xf[i:i + self.chunk_size]                # [cs, T, D]
#             xi_ln = self.ln1(xi)
#             # need_weights=False -> don't store attention weights
#             attn_out, _ = self.attn(xi_ln, xi_ln, xi_ln, need_weights=False)
#             attn_out = self.drop_att(attn_out)
#             out[i:i + attn_out.size(0)] = xi + attn_out   # residual
#         return out

#     def forward(self, x: torch.Tensor) -> torch.Tensor:
#         # x: [B, D, T, F]
#         B, D, T, Freq = x.shape
#         xf = x.permute(0, 3, 2, 1).contiguous().view(B * Freq, T, D)  # [B*F, T, D]

#         # chunked attention (reduces peak memory)
#         attn_out = self._chunked_attn(xf)   # [B*F, T, D]

#         # FFN (we can also chunk FFN if needed, but FFN is linear memory)
#         attn_ln = self.ln2(attn_out)
#         ffn_out = self.ffn(attn_ln)
#         out = attn_out + ffn_out  # residual

#         out = out.view(B, Freq, T, D).permute(0, 3, 2, 1).contiguous()  # [B, D, T, F]
#         return out



# ...existing code...

class TFAR2D(nn.Module):
    def __init__(self, channels: int, kernel_size=(4,4)):
        super().__init__()
        self.W1 = DepthwiseConv2dGLN(channels, kernel_size=kernel_size, stride=(1,1))
        self.W2 = DepthwiseConv2dGLN(channels, kernel_size=kernel_size, stride=(1,1))
        self.W3 = DepthwiseConv2dGLN(channels, kernel_size=kernel_size, stride=(1,1))

    def forward(self, m: torch.Tensor, n: torch.Tensor) -> torch.Tensor:
        # m: [B, D, Tm, Fm], n: [B, D, Tn, Fn]
        B, D, Tm, Fm = m.shape
        gate = torch.sigmoid(self.W1(n))        # [B, D, Tn, Fn]
        n3   = self.W3(n)                       # [B, D, Tn, Fn]
        gate_up = F.interpolate(gate, size=(Tm, Fm), mode="nearest")
        n3_up   = F.interpolate(n3,   size=(Tm, Fm), mode="nearest")
        return gate_up * self.W2(m) + n3_up



class RTFSBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,    
        D: int = 64,    
        q: int = 2,         
        k_unfold: int = 8,   
        rnn_hidden: int = 32,  
        rnn_layers: int = 4,
        attn_heads: int = 4,
        dropout: float = 0.1,
    ):
        super().__init__()
        assert q >= 1, "q must be ≥ 1"

        self.in_channels = in_channels
        self.D = D
        self.q = q
        self.k_unfold = k_unfold
        self.pad_l = (k_unfold - 1) // 2
        self.pad_r = k_unfold - 1 - self.pad_l

        self.reduce = PointwiseConv2dGLN(in_channels, D)
        self.downs = nn.ModuleList([
            DepthwiseConv2dGLN(D, kernel_size=(4,4), stride=(2,2))
            for _ in range(q - 1)
        ])

        self.ln_freq = LayerNormChannels(D * k_unfold)
        self.ln_time = LayerNormChannels(D * k_unfold)

        self.rnn_f = _BiRNN(in_size=D * k_unfold, hidden_size=rnn_hidden,
                            num_layers=rnn_layers, dropout=dropout)
        self.rnn_t = _BiRNN(in_size=D * k_unfold, hidden_size=rnn_hidden,
                            num_layers=rnn_layers, dropout=dropout)

        self.fold_f = nn.ConvTranspose1d(in_channels=2 * rnn_hidden, out_channels=D,
                                         kernel_size=k_unfold, stride=1, padding=0, bias=False)
        self.fold_t = nn.ConvTranspose1d(in_channels=2 * rnn_hidden, out_channels=D,
                                         kernel_size=k_unfold, stride=1, padding=0, bias=False)

        self.attn = TFDomainAttention(D, num_heads=attn_heads, dropout=dropout)

        self.tfar = TFAR2D(D, kernel_size=(4,4))
        self.expand = nn.Conv2d(D, in_channels, kernel_size=1)

    def _unfold_freq(self, AG: torch.Tensor) -> torch.Tensor:
        B, D, T, Freq = AG.shape
        x = F.pad(AG, (self.pad_l, self.pad_r, 0, 0))          # [B, D, T, F'+7]
        x = x.unfold(dimension=-1, size=self.k_unfold, step=1) # [B, D, T, F', K]
        x = x.permute(0, 1, 4, 2, 3).contiguous()              # [B, D, K, T, F']
        return x.view(B, D * self.k_unfold, T, Freq)           # [B, D*K, T, F']

    def _fold_freq(self, y: torch.Tensor, F_out: int, T_: int) -> torch.Tensor:
        B, C2, T, Freq = y.shape
        y = y.permute(0, 2, 1, 3).contiguous().view(B * T, C2, Freq)  # [B*T, 2h, F']
        y = self.fold_f(y)                                            # [B*T, D, F'+K-1] = [B*T, D, F'+7]
        y = y[:, :, self.pad_l:self.pad_l + F_out]                    # [B*T, D, F']
        y = y.view(B, T, self.D, F_out).permute(0, 2, 1, 3).contiguous()  # [B, D, T', F']
        return y




    def _unfold_time(self, Rf: torch.Tensor) -> torch.Tensor:
        # Rf: [B, D, T', F'] → pad T → unfold(T,K=8,S=1) → [B, D*K, T', F']
        B, D, T, Freq = Rf.shape
        x = F.pad(Rf, (0, 0, self.pad_l, self.pad_r))               # [B, D, T'+7, F]
        x = x.unfold(dimension=-2, size=self.k_unfold, step=1)      # [B, D, T', F, K]
        x = x.permute(0, 1, 4, 2, 3).contiguous()                   # [B, D, K, T', F]
        return x.view(B, D * self.k_unfold, T, Freq)                # [B, D*K, T', F']

    def _fold_time(self, y: torch.Tensor, T_out: int, F_: int) -> torch.Tensor:
        # y: [B, 2h, T', F']
        B, C2, T, Freq = y.shape
        y = y.permute(0, 3, 1, 2).contiguous().view(B * Freq, C2, T)   # [B*F, 2h, T']
        y = self.fold_t(y)                                             # [B*F, D, T'+7]
        y = y[:, :, self.pad_l:self.pad_l + T_out]                     # [B*F, D, T']
        y = y.view(B, Freq, self.D, T_out).permute(0, 2, 3, 1).contiguous()  # [B, D, T', F']
        return y

    def forward(self, A: torch.Tensor) -> torch.Tensor:
        """
        A: [B, Ca, T, F]
        return: [B, Ca, T, F]
        """
        B, Ca, T0, F0 = A.shape
        x = self.reduce(A)                    # [B, D, T, F]
        levels = [x]
        for down in self.downs:
            x = down(x)                    
            levels.append(x)

        T_min, F_min = levels[-1].shape[-2:]
        AG = None
        for lv in levels:
            p = F.adaptive_avg_pool2d(lv, (T_min, F_min))
            AG = p if AG is None else AG + p   # [B, D, T', F']

        Rf_in = self._unfold_freq(AG)                 # [B, D*K, T', F']
        Rf_in = self.ln_freq(Rf_in)                   
        B_, DK, T_, F_ = Rf_in.shape
        y_f = Rf_in.permute(0, 2, 3, 1).contiguous().view(B_ * T_, F_, DK)    # [B*T', F', D*K]
        y_f = self.rnn_f(y_f)                                                   # [B*T', F', 2*h]
        y_f = y_f.view(B_, T_, F_, -1).permute(0, 3, 1, 2).contiguous()        # [B, 2h, T', F']
        Rf = self._fold_freq(y_f, F_out=F_, T_=T_) + AG                       

        Rt_in = self._unfold_time(Rf)                 # [B, D*K, T', F']
        Rt_in = self.ln_time(Rt_in)
        y_t = Rt_in.permute(0, 3, 2, 1).contiguous().view(B_ * F_, T_, DK)     # [B*F', T', D*K]
        y_t = self.rnn_t(y_t)                                                   # [B*F', T', 2*h]
        y_t = y_t.view(B_, F_, T_, -1).permute(0, 3, 2, 1).contiguous()        # [B, 2h, T', F']
        Rt = self._fold_time(y_t, T_out=T_, F_=F_) + Rf                   

        AbarG = self.attn(Rt) + Rt                                              # [B, D, T', F']

        fused = [self.tfar(levels[i], AbarG) for i in range(len(levels))]       # A'_i
        rec = fused[-1]
        for i in range(len(levels) - 2, -1, -1):
            rec = self.tfar(fused[i], rec) + levels[i]                          # U-Net skip

        out = self.expand(rec)                                                  # [B, Ca, T, F]
        return out + A
    
class APBlock(nn.Module):
    def __init__(self,
                 in_channels: int,   
                 D: int = 64,
                 q: int = 2,
                 k_unfold: int = 8,
                 rnn_hidden: int = 32,
                 rnn_layers: int = 4,
                 attn_heads: int = 4,
                 dropout: float = 0.1):
        super().__init__()
        self.core = RTFSBlock(in_channels=in_channels, D=D, q=q,
                                k_unfold=k_unfold,
                                rnn_hidden=rnn_hidden, rnn_layers=rnn_layers,
                                attn_heads=attn_heads, dropout=dropout)

    def forward(self, a0: torch.Tensor) -> torch.Tensor:
        return self.core(a0)

class StackedRTFS(nn.Module):
    def __init__(
        self,
        R: int,
        in_channels: int,    
        D: int = 64,    
        q: int = 2,         
        k_unfold: int = 8,   
        rnn_hidden: int = 32,  
        rnn_layers: int = 4,
        attn_heads: int = 4,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.R = R
        self.rtfs = RTFSBlock(
            in_channels=in_channels,
            D=D,
            q=q,
            k_unfold=k_unfold,
            rnn_hidden=rnn_hidden,
            rnn_layers=rnn_layers,
            attn_heads=attn_heads,
            dropout=dropout,
        )

    def forward(self, fused: torch.Tensor, a0: torch.Tensor) -> torch.Tensor:
        x = fused
        for _ in range(self.R):
            x = self.rtfs(x + a0)
        return x