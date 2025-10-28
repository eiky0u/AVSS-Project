import torch
from torchmetrics.functional.audio import (
    permutation_invariant_training as pit,
)
from torchmetrics.functional.audio.pesq import (
    perceptual_evaluation_speech_quality as pesq,
)

from src.metrics.base_metric import BaseMetric


class PESQ(BaseMetric):
    """
    PESQ (Perceptual Evaluation of Speech Quality), wideband mode.

    Shapes:
        target: [B, S, T]  — ground-truth sources (S speakers)
        preds:  [B, S, T]  — model estimates (unordered across speakers)

    Returns:
        float: batch-averaged PESQ over PIT-optimal speaker assignment.
    """

    def __init__(self, *args, target_sr: int = 16000, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.target_sr = target_sr

    @torch.no_grad()
    def __call__(
        self,
        target: torch.Tensor,  # [B, S, T]
        preds: torch.Tensor,  # [B, S, T]
        **kwargs,
    ) -> float:

        # Basic shape checks
        if target.ndim != 3 or preds.ndim != 3:
            raise ValueError(
                f"`target` and `preds` must be [B, S, T]. "
                f"Got target={tuple(target.shape)}, preds={tuple(preds.shape)}"
            )
        if target.shape != preds.shape:
            raise ValueError(
                f"`target` and `preds` must have identical shapes. "
                f"Got target={tuple(target.shape)}, preds={tuple(preds.shape)}"
            )

        B, S, T = target.shape

        # Metric function: per-speaker PESQ (wideband) -> [B, S]
        def metric_fn(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
            return pesq(x, y, fs=self.target_sr, mode="wb")

        # PIT over speakers -> [B]
        per_sample_scores, _ = pit(
            preds=preds,
            target=target,
            metric_func=metric_fn,
            mode="speaker-wise",
            eval_func="max",
        )  # [B]

        # Batch-average
        return per_sample_scores.mean().item()
