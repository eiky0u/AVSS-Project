from torch import nn
import torch


class BaselineModel(nn.Module):
    """
    Simple MLP baseline for Target/Multi-speaker Speech Separation.

    Input:
        data_object: mix waveform as Tensor of shape [B, T] or [B, 1, T]
    Output:
        dict with:
            preds: Tensor [B, S, T] — per-speaker waveforms (ordered)
    """

    def __init__(self, clip_len: int, n_speakers: int, fc_hidden: int = 512):
        """
        Args:
            clip_len (int): fixed input length in samples (T).
            n_speakers (int): number of output speakers (S).
            fc_hidden (int): hidden layer width.
        """
        super().__init__()
        self.clip_len = int(clip_len)
        self.n_speakers = int(n_speakers)
        n_feats = self.clip_len
        n_out = self.n_speakers * self.clip_len

        self.net = nn.Sequential(
            nn.Linear(in_features=n_feats, out_features=fc_hidden),
            nn.ReLU(),
            nn.Linear(in_features=fc_hidden, out_features=fc_hidden),
            nn.ReLU(),
            nn.Linear(in_features=fc_hidden, out_features=n_out),
        )

    def forward(self, mix: torch.Tensor, **batch):
        """
        Args:
            mix: mix waveform [B, 1, T]
        Returns:
            {"preds": Tensor [B, S, T]}
        """
        x = mix
        # [B,1,T] -> [B,T]
        if x.ndim == 3 and x.size(1) == 1:
            x = x[:, 0, :]
        else:
            raise ValueError(f"`data_object` must be [B,1,T]; got {tuple(mix.shape)}")

        B, T = x.shape
        if T != self.clip_len:
            raise ValueError(
                f"Got T={T}, but model was initialized with clip_len={self.clip_len}"
            )

        x = x.float()
        y = self.net(x)  # [B, S*T]
        preds = y.view(B, self.n_speakers, self.clip_len)  # [B, S, T]
        return {"preds": preds}

    def __str__(self):
        all_parameters = sum(p.numel() for p in self.parameters())
        trainable_parameters = sum(
            p.numel() for p in self.parameters() if p.requires_grad
        )
        result_info = super().__str__()
        result_info += f"\nAll parameters: {all_parameters}"
        result_info += f"\nTrainable parameters: {trainable_parameters}"
        return result_info
