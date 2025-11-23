import torch
import torch.nn as nn
from torchmetrics.functional.audio import (
    scale_invariant_signal_noise_ratio as si_snr,
)


class SISNRLoss(nn.Module):
    """
    SI-SNR (Scale-Invariant Signal-to-Noise Ratio) Loss.

    Shapes:
        targets: [B, S, T]  — ground-truth sources (S speakers)
        preds:  [B, S, T]  — model estimates (ordered across speakers)

    Returns:
        float: batch-averaged Loss.
    """

    def forward(
        self,
        preds: torch.Tensor,  # [B, S, T]
        targets: torch.Tensor,  # [B, S, T]
        **batch,
    ) -> torch.Tensor:
        loss = -si_snr(preds, targets).mean()
        return {"loss": loss}
