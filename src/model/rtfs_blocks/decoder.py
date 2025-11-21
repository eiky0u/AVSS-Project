import torch
import torch.nn as nn

class Decoder(nn.Module):
    def __init__(
        self,
        n_fft: int,
        hop_length: int,
        input_channels: int,
        win_length: int,
        bias: bool = False,
    ):
        super().__init__()
        self.n_fft = int(n_fft)
        self.hop_length = int(hop_length)
        self.input_channels = int(input_channels)
        self.bias = bias
        self.win_length = win_length 

        self.tconv = nn.ConvTranspose2d(
            in_channels=self.input_channels,
            out_channels=2,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=self.bias,
        )

        
    def forward(self, x: torch.Tensor, target_length):

        x = self.tconv(x)
        x = torch.complex(x[:, 0], x[:, 1])
        x = x.transpose(1, 2).contiguous()

        audio = torch.istft(
            x,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            window=torch.hann_window(self.win_length, device=x.device),
            length=target_length,
            win_length=self.win_length
        ) 

        audio = audio.view(x.shape[0], target_length)
        return audio


