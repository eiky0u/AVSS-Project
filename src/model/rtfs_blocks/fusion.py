import torch
import torch.nn as nn
import torch.nn.functional as F
from src.model.rtfs_blocks.GLN import GlobalLayerNorm

class CAF(nn.Module):

    def __init__(self, audio_channels: int = 256, video_channels: int = 512, heads: int = 4, eps: float = 1e-5):
        super(CAF, self).__init__()
        self.Ca = audio_channels  
        self.Cv = video_channels  
        self.h = heads  
        self.P1 = nn.Sequential(
            nn.Conv2d(self.Ca, self.Ca, kernel_size=1, groups=self.Ca, bias=True),
            GlobalLayerNorm(self.Ca, eps=eps)
        )
        self.P2 = nn.Sequential(
            nn.Conv2d(self.Ca, self.Ca, kernel_size=1, groups=self.Ca, bias=True),
            GlobalLayerNorm(self.Ca, eps=eps)
        )

        self.F1 = nn.Sequential(
            nn.Conv1d(self.Cv, self.Ca * self.h, kernel_size=1, groups=self.Ca, bias=True),
            GlobalLayerNorm(self.Ca * self.h, eps=eps)
        )

        self.F2 = nn.Sequential(
            nn.Conv1d(self.Cv, self.Ca, kernel_size=1, groups=self.Ca, bias=True),
            GlobalLayerNorm(self.Ca, eps=eps)
        )

    def forward(self, audio: torch.Tensor, video: torch.Tensor) -> torch.Tensor:

        b, Ca, Ta, F_dim = audio.shape
        assert Ca == self.Ca, f"Audio channels must be {self.Ca}"
        b_v, Cv, Tv = video.shape
        assert Cv == self.Cv and b == b_v, f"Video shape mismatch"

        aval = self.P1(audio)  # [b, Ca, Ta, F]
        agate = F.relu(self.P2(audio))  # [b, Ca, Ta, F]

        vh = self.F1(video)  # [b, Ca*h, Tv]
        vh_heads = vh.view(b, self.Ca, self.h, Tv)  # [b, Ca, h, Tv]
        vm = vh_heads.mean(dim=2)  # [b, Ca, Tv]
        vm_soft = F.softmax(vm, dim=-1)  # [b, Ca, Tv]
        vattn = F.interpolate(vm_soft, size=Ta, mode='nearest')  # [b, Ca, Ta]
        vattn = vattn.unsqueeze(-1)  # [b, Ca, Ta, 1] for broadcasting over F
        f1 = vattn * aval  # [b, Ca, Ta, F]

        vkey = self.F2(video)  # [b, Ca, Tv]
        vkey = F.interpolate(vkey, size=Ta, mode='nearest')  # [b, Ca, Ta]
        vkey = vkey.unsqueeze(-1)  
        f2 = agate * vkey  # [b, Ca, Ta, F]

        a2 = f1 + f2  # [b, Ca, Ta, F]
        return a2