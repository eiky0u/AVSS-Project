import torch
from torchmetrics.functional.audio import (
    scale_invariant_signal_noise_ratio as si_snr,
)

from src.metrics.base_metric import BaseMetric


class SISNRi(BaseMetric):
    """
    SI-SNRi (Scale-Invariant Signal-to-Noise Ratio improvement) in dB.

    Shapes:
        targets: [B, S, T]  — ground-truth sources (S speakers)
        preds:  [B, S, T]  — model estimates (ordered across speakers)
        mix:    [B, 1, T]  — original mixture

    Returns:
        float: batch-averaged SI-SNRi in dB.
    """

    @torch.no_grad()
    def __call__(
        self,
        targets: torch.Tensor,  # [B, S, T]
        preds: torch.Tensor,  # [B, S, T]
        mix: torch.Tensor,  # [B, 1, T]
        **batch,
    ) -> float:

        # Basic shape checks
        if targets.ndim != 3 or preds.ndim != 3:
            raise ValueError(
                f"`target` and `preds` must be [B, S, T]. "
                f"Got target={tuple(targets.shape)}, preds={tuple(preds.shape)}"
            )
        if mix.ndim != 3:
            raise ValueError(f"`mix` must be [B, 1, T]. Got mix={tuple(mix.shape)}")
        if targets.shape != preds.shape:
            raise ValueError(
                f"`target` and `preds` must have identical shapes. "
                f"Got target={tuple(targets.shape)}, preds={tuple(preds.shape)}"
            )

        B, S, T = targets.shape
        mix = mix.expand(-1, S, -1)  # [B, S, T]

        si_snr_mix = si_snr(mix, targets)
        si_snr_preds = si_snr(preds, targets)
        si_snr_i = (si_snr_preds - si_snr_mix).mean()  # scalar

        return float(si_snr_i.item())
