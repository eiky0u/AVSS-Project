import torch
from torch.nn.utils.rnn import pad_sequence
from typing import Dict, List, Any


def collate_fn(items: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Collate function for Speech Separation (SS):

    Returns:
      - utt: List[str]
      - mix: FloatTensor [B, T_max]
      - mix_len: LongTensor [B]
      - targets: FloatTensor [B, 2, T_max]   (sources along the second axis)
      - mouths: FloatTensor [B, 2, F_max, H, W] (F - frames)
      - mouths_len: LongTensor [B]
    """
    assert len(items) > 0, "Empty batch"

    batch: Dict[str, Any] = {"utt": [x["utt"] for x in items]}

    # -------- MIX --------
    mix_list = [x["mix"] for x in items]  # list of [T]
    assert all(t.dim() == 1 for t in mix_list), "mix must be 1D [T]"
    mix_len = torch.as_tensor([t.numel() for t in mix_list], dtype=torch.long)  # [B]
    mix = pad_sequence(mix_list, batch_first=True)  # [B, T_max]
    T_max = mix.size(-1)
    batch["mix"] = mix.unsqueeze(dim=1)  # [B, 1, T_max]
    batch["mix_len"] = mix_len

    # -------- TARGETS --------
    t1_list = [x["target1"] for x in items]  # list of [T]
    t2_list = [x["target2"] for x in items]
    assert all(
        t is not None and t.dim() == 1 for t in t1_list
    ), "target1 must be 1D [T]"
    assert all(
        t is not None and t.dim() == 1 for t in t2_list
    ), "target2 must be 1D [T]"

    for i, (lm, l1, l2) in enumerate(
        zip(
            mix_len.tolist(), [t.numel() for t in t1_list], [t.numel() for t in t2_list]
        )
    ):
        assert (
            l1 == lm and l2 == lm
        ), f"Sample {i}: target len must equal mix len ({l1}, {l2} vs {lm})"

    t1 = pad_sequence(t1_list, batch_first=True)  # [B, T_max]
    t2 = pad_sequence(t2_list, batch_first=True)  # [B, T_max]

    assert (
        t1.size(-1) == T_max and t2.size(-1) == T_max
    ), "Targets padded length must equal mix length in batch"

    targets = torch.stack([t1, t2], dim=1)  # [B, 2, T_max]
    batch["targets"] = targets

    # -------- MOUTHS --------
    m1_list = [x["mouth1"] for x in items]  # [F, H, W]
    m2_list = [x["mouth2"] for x in items]
    assert all(m.dim() == 3 for m in m1_list + m2_list), "mouths must be [F, H, W]"

    mouth1 = pad_sequence(m1_list, batch_first=True)  # [B, F_max, H, W]
    mouth2 = pad_sequence(m2_list, batch_first=True)  # [B, F_max, H, W]
    batch["mouths"] = torch.stack([mouth1, mouth2], dim=1)
    batch["mouths_len"] = torch.as_tensor(
        [m.size(0) for m in m1_list], dtype=torch.long, device=mouth1.device
    )

    return batch
