import torch
import torch.nn as nn

class SpectralSourceSeparation(nn.Module):
    def __init__(self, in_channels):
        super(SpectralSourceSeparation, self).__init__()
        self.prelu = nn.PReLU()
        self.conv = nn.Conv2d(in_channels, in_channels, kernel_size=1, stride=1, padding=0)
        self.relu = nn.ReLU()

    def forward(self, aR, a0):
        x = self.prelu(aR)
        m = self.conv(x)
        m = self.relu(m)  #[B, Ca, Ta, F]

        Ca = m.size(1)
        assert Ca % 2 == 0, "Number of channels Ca must be even."

        half_Ca = Ca // 2

        mr = m[:, :half_Ca, :, :]
        mi = m[:, half_Ca:, :, :]
        Er = a0[:, :half_Ca, :, :]
        Ei = a0[:, half_Ca:, :, :]

        zr = mr * Er - mi * Ei
        zi = mr * Ei + mi * Er

        z = torch.cat([zr, zi], dim=1)  #[B, Ca, Ta, F]

        return z

