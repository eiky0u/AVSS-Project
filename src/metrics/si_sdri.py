import torch
from torchmetrics.functional.audio import (
    scale_invariant_signal_distortion_ratio as si_sdr,
    permutation_invariant_training as pit,
)

from src.metrics.base_metric import BaseMetric


class SISDRi(BaseMetric):
    """
    SI-SDRi (Scale-Invariant Signal-to-Distortion Ratio improvement), dB.

    Shapes:
        target: [B, S, T]  — ground-truth sources (S speakers)
        preds:  [B, S, T]  — model estimates (unordered across speakers)
        mix:    [B, T]     — original mixture

    Returns:
        float: batch-averaged SI-SDRi in dB.
    """

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

    @torch.no_grad()
    def __call__(
        self,
        target: torch.Tensor,  # [B, S, T]
        preds: torch.Tensor,  # [B, S, T]
        mix: torch.Tensor,  # [B, T]
        **kwargs,
    ) -> float:

        # Basic shape checks
        if target.ndim != 3 or preds.ndim != 3:
            raise ValueError(
                f"`target` and `preds` must be [B, S, T]. "
                f"Got target={tuple(target.shape)}, preds={tuple(preds.shape)}"
            )
        if mix.ndim != 2:
            raise ValueError(f"`mix` must be [B, T]. Got mix={tuple(mix.shape)}")
        if target.shape != preds.shape:
            raise ValueError(
                f"`target` and `preds` must have identical shapes. "
                f"Got target={tuple(target.shape)}, preds={tuple(preds.shape)}"
            )

        B, S, T = target.shape
        if mix.shape != (B, T):
            raise ValueError(f"Got mix={tuple(mix.shape)}, expected {(B, T)}")

        # PIT over speakers -> [B]
        per_sample_scores, _ = pit(
            preds=preds,
            target=target,
            metric_func=si_sdr,
            mode="speaker-wise",
            eval_func="max",
        )  # [B]
        si_sdr_pred = per_sample_scores.mean()

        # Broadcast mix: [B, T] -> [B, S, T]
        mix_rep = mix.unsqueeze(1).expand(B, S, T)
        scores_mix = si_sdr(
            mix_rep.reshape(B * S, T),
            target.reshape(B * S, T),
        ).view(B, S)
        si_sdr_mix = scores_mix.mean(dim=1).mean()

        si_sdri = si_sdr_pred - si_sdr_mix
        return si_sdri.item()
