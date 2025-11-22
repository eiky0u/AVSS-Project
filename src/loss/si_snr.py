# import torch
# import torch.nn as nn
# from torchmetrics.functional.audio import (
#     scale_invariant_signal_noise_ratio as si_snr,
# )


# class SISNRLoss(nn.Module):
#     """
#     SI-SNR (Scale-Invariant Signal-to-Noise Ratio) Loss.

#     Shapes:
#         targets: [B, S, T]  — ground-truth sources (S speakers)
#         preds:  [B, S, T]  — model estimates (ordered across speakers)

#     Returns:
#         float: batch-averaged Loss.
#     """

#     def forward(
#         self,
#         preds: torch.Tensor,  # [B, S, T]
#         targets: torch.Tensor,  # [B, S, T]
#         **batch,
#     ) -> torch.Tensor:
#         loss = -si_snr(preds, targets).mean()
#         return {"loss": loss}
import torch
import torch.nn as nn
from itertools import permutations
from torchmetrics.functional.audio import (
    scale_invariant_signal_noise_ratio as si_snr,
)

class SISNRLoss(nn.Module):
    """
    SI-SNR Loss with PIT (Permutation Invariant Training).
    
    Calculates SI-SNR for all possible speaker permutations and 
    uses the best one (maximum SI-SNR) for backpropagation.
    """

    def __init__(self):
        super().__init__()

    def forward(
        self,
        preds: torch.Tensor,   # [B, S, T]
        targets: torch.Tensor, # [B, S, T]
        **batch,
    ) -> dict[str, torch.Tensor]:
        
        B, S, T = preds.shape
        
        # Список для хранения средних SI-SNR по батчу для каждой перестановки
        # Размерность каждого элемента будет [B]
        snr_permutations = []

        # Перебираем все возможные варианты расстановки таргетов
        # Для 2 спикеров это: (0, 1) и (1, 0)
        for p in permutations(range(S)):
            # Переставляем таргеты в соответствии с текущей перестановкой p
            # targets[:, p, :] меняет порядок спикеров во всех примерах батча
            permuted_targets = targets[:, p, :]
            
            # Считаем SI-SNR. 
            # si_snr возвращает [B, S] (значение для каждого спикера в каждом батче)
            pair_wise_si_snr = si_snr(preds, permuted_targets)
            
            # Усредняем по спикерам (dim=1), чтобы получить качество разделения 
            # для всего примера целиком. Получаем [B].
            batch_avg_si_snr = pair_wise_si_snr.mean(dim=1)
            
            snr_permutations.append(batch_avg_si_snr)

        # Собираем все варианты в один тензор [B, N_perms]
        snr_permutations = torch.stack(snr_permutations, dim=1)

        # Для каждого элемента батча находим ЛУЧШУЮ перестановку (максимальный SNR)
        # best_snr будет иметь размерность [B]
        best_snr, _ = snr_permutations.max(dim=1)

        # Итоговый лосс — это отрицательный средний SNR по всему батчу
        loss = -best_snr.mean()

        return {"loss": loss}