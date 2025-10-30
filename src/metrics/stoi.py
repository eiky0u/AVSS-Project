import torch
from torchmetrics.functional.audio.stoi import (
    short_time_objective_intelligibility as stoi,
)
from src.metrics.base_metric import BaseMetric


class STOI(BaseMetric):
    """
    STOI / eSTOI for ordered speakers (no PIT).

    Inputs:
        targets: Tensor [B, S, T] — ground-truth sources (S speakers)
        preds:   Tensor [B, S, T] — model estimates (ordered across speakers)

    Returns:
        float: Mean STOI over speakers and batch (scalar ~ [0, 1]).

    Notes:
        - Computed on CPU; common STOI backends are not CUDA-enabled.
        - Assumes both signals share the same sampling rate `target_sr`.
    """

    def __init__(
        self,
        *args,
        target_sr: int = 16000,
        extended: bool = False,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.target_sr = target_sr
        self.extended = extended  # True -> eSTOI

    @torch.no_grad()
    def __call__(
        self,
        targets: torch.Tensor,
        preds: torch.Tensor,
        **batch,
    ) -> float:
        # Shape checks
        if targets.ndim != 3 or preds.ndim != 3:
            raise ValueError(
                f"`targets` and `preds` must be [B, S, T]. "
                f"Got targets={tuple(targets.shape)}, preds={tuple(preds.shape)}"
            )
        if targets.shape != preds.shape:
            raise ValueError("`targets` and `preds` must match shapes")

        B, S, T = targets.shape

        # Move once to CPU and flatten speaker dim: [B*S, T]
        targets = targets.detach().to("cpu").float().reshape(B * S, T)
        preds = preds.detach().to("cpu").float().reshape(B * S, T)

        scores = []
        for i in range(B * S):
            scores.append(
                stoi(
                    preds[i],
                    targets[i],
                    fs=self.target_sr,
                    extended=self.extended,
                )
            )

        return float(torch.tensor(scores).mean().item())
