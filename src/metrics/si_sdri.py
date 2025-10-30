import torch
from torchmetrics.functional.audio import (
    scale_invariant_signal_distortion_ratio as si_sdr,
)

from src.metrics.base_metric import BaseMetric


class SISDRi(BaseMetric):
    """
    SI-SDRi (Scale-Invariant Signal-to-Distortion Ratio improvement) in dB.

    Shapes:
        targets: [B, S, T]  — ground-truth sources (S speakers)
        preds:  [B, S, T]  — model estimates (ordered across speakers)
        mix:    [B, 1, T]  — original mixture

    Returns:
        float: batch-averaged SI-SDRi in dB.
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

        si_sdr_mix = si_sdr(mix, targets)
        si_sdr_preds = si_sdr(preds, targets)
        si_sdr_i = (si_sdr_preds - si_sdr_mix).mean()  # scalar

        return float(si_sdr_i.item())
