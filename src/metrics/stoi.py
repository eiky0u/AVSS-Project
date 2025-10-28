import torch
from torchmetrics.functional.audio import (
    permutation_invariant_training as pit,
)
from torchmetrics.functional.audio.stoi import (
    short_time_objective_intelligibility as stoi,
)

from src.metrics.base_metric import BaseMetric


class STOI(BaseMetric):
    """
    STOI (Short-Time Objective Intelligibility).

    Shapes:
        target: [B, S, T]  — ground-truth sources (S speakers)
        preds:  [B, S, T]  — model estimates (unordered across speakers)

    Returns:
        float: batch-averaged STOI (range ~ [0, 1]) over PIT-optimal speaker assignment.
    """

    def __init__(
        self, *args, target_sr: int = 16000, extended: bool = False, **kwargs
    ) -> None:
        super().__init__(*args, **kwargs)
        self.target_sr = target_sr
        self.extended = extended  # eSTOI if True

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

        # Metric function: per-speaker STOI -> [B, S]
        def metric_fn(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
            return stoi(x, y, fs=self.target_sr, extended=self.extended)

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
