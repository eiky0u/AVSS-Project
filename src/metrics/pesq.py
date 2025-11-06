import torch
from torchmetrics.functional.audio.pesq import (
    perceptual_evaluation_speech_quality as pesq,
)
from src.metrics.base_metric import BaseMetric


class PESQ(BaseMetric):
    """
    PESQ (Perceptual Evaluation of Speech Quality) for ordered speakers (no PIT).

    Inputs:
        targets: Tensor [B, S, T] — ground-truth sources (S speakers)
        preds:   Tensor [B, S, T] — model estimates (ordered across speakers)

    Returns:
        float: Mean PESQ over speakers and batch (scalar).

    Requirements:
        - mode="wb" requires fs=16000; mode="nb" requires fs=8000.
        - Computed on CPU; reference implementations are not CUDA-enabled.
    """

    def __init__(
        self,
        *args,
        target_sr: int = 16000,
        mode: str = "wb",
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.target_sr = target_sr
        self.mode = mode  # "wb" (16 kHz) or "nb" (8 kHz)

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

        # SR checks
        if self.mode == "wb" and self.target_sr != 16000:
            raise ValueError(
                "PESQ 'wb' requires target_sr=16000. Resample or use mode='nb' with 8000 Hz."
            )
        if self.mode == "nb" and self.target_sr != 8000:
            raise ValueError("PESQ 'nb' requires target_sr=8000.")

        B, S, T = targets.shape

        # Move once to CPU and flatten: [B*S, T]
        targets = targets.detach().to("cpu").float().reshape(B * S, T)
        preds = preds.detach().to("cpu").float().reshape(B * S, T)

        scores = []
        for i in range(B * S):
            scores.append(
                pesq(
                    preds[i],
                    targets[i],
                    fs=self.target_sr,
                    mode=self.mode,
                )
            )

        return float(torch.tensor(scores).mean().item())
