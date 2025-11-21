import torch
import torch.nn as nn
from src.model.rtfs_blocks.GLN import GlobalLayerNorm

class audio_encoder(nn.Module):
    def __init__(self, n_fft=1024, hop_length=128, win_length=256, av_channels=256):
        super(audio_encoder, self).__init__()
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.win_length = win_length
        self.av_channels = av_channels
        self.conv = nn.Sequential(
            nn.Conv2d(2, av_channels, kernel_size=(3,3), stride=(1,1), padding=(1,1)),
            GlobalLayerNorm(av_channels),
            nn.ReLU(),
        )
        
    def forward(self, x):
        stft_complex = torch.stft(x, 
                                  n_fft=self.n_fft, 
                                  hop_length=self.hop_length, 
                                  win_length=self.win_length, 
                                  window=torch.hann_window(self.win_length, device=x.device), 
                                  return_complex=True
                                  )
        
        x = torch.cat([stft_complex.real.unsqueeze(1), stft_complex.imag.unsqueeze(1)], dim=1).transpose(3,2) # [B, 2, T, F] !! = [B, 2, 251, 513]
        x = self.conv(x)
        return x # [B, 256, 251, 513]