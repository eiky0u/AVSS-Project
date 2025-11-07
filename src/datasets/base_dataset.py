import logging
import random
from typing import List, Dict, Any, Optional

import torch
import torchaudio
from torch.utils.data import Dataset

import numpy as np

logger = logging.getLogger(__name__)


class BaseDataset(Dataset):
    """
    Base class for audio(-visual) datasets.

    Works with a prebuilt index (list of dicts) and provides common loading
    utilities. Concrete datasets are expected to prepare the index with all
    required metadata (paths, ids, etc.).
    """

    def __init__(
        self,
        index: List[Dict[str, Any]],
        target_sr: int = 16000,
        limit: Optional[int] = None,
        shuffle_index: bool = False,
        instance_transforms: Optional[Dict[str, Any]] = None,
    ):
        """
        Initialize dataset with an index and optional sampling parameters.

        Args:
            index: List of sample dictionaries with required fields (e.g.
                'mix_path', 's1_path', 's2_path', 'mouth1_path', 'mouth2_path').
            target_sr: Target sampling rate for all loaded audio.
            limit: If provided, truncate the index to the first `limit` items
                (after optional shuffling).
            shuffle_index: Shuffle index with a fixed seed (42) before limiting.
            instance_transforms: Optional dict of callables to transform
                instance tensors keyed by their names.
        """
        self.target_sr = target_sr
        index = self._shuffle_and_limit_index(index, limit, shuffle_index)
        self._index: List[dict] = index
        self.instance_transforms = instance_transforms

    def __getitem__(self, ind: int) -> Dict[str, Any]:
        """
        Load a single instance by index and return a dictionary.

        Returned keys:
            - 'utt': str, utterance id.
            - 'mix': torch.FloatTensor [1, T], mono waveform at `target_sr`.
            - 'target1': Optional[torch.FloatTensor] [1, T] or None (test split).
            - 'target2': Optional[torch.FloatTensor] [1, T] or None (test split).
            - 'mouth1': torch.FloatTensor [T_m, D] (e.g., D=1).
            - 'mouth2': torch.FloatTensor [T_m, D].

        Note:
            In test split `target1`/`target2` may be None by design.
        """
        data_dict = self._index[ind]

        mix = self.load_audio(data_dict["mix_path"])
        target1 = self.load_audio(data_dict["s1_path"])
        target2 = self.load_audio(data_dict["s2_path"])

        mouth1 = self.load_mouth(data_dict["mouth1_path"])
        mouth2 = self.load_mouth(data_dict["mouth2_path"])

        instance_data = {
            "utt": data_dict["utt"],
            "mix": mix,  # [T]
            "target1": target1,  # [T]
            "target2": target2,  # [T]
            "mouth1": mouth1,  # [F, H, W]
            "mouth2": mouth2,  # [F, H, W]
        }
        instance_data = self.preprocess_data(instance_data)
        return instance_data

    def __len__(self) -> int:
        """Return the number of indexed samples."""
        return len(self._index)

    def load_audio(self, path: Optional[str]) -> Optional[torch.Tensor]:
        """
        Load audio from `path`, convert to mono [1, T] and resample if needed.

        Args:
            path: Path to .wav file. If None, returns None.

        Returns:
            Tensor of shape [1, T] (float32) at `target_sr`, or None if `path` is None.
        """
        if path is None:
            return None
        audio_tensor, sr = torchaudio.load(path)  # [C, T]
        audio_tensor = audio_tensor.mean(dim=0)  # [T]
        if sr != self.target_sr:
            audio_tensor = torchaudio.functional.resample(
                audio_tensor, sr, self.target_sr
            )
        return audio_tensor

    def load_mouth(self, path: Optional[str]) -> Optional[torch.Tensor]:
        """
        Load mouth features from .npz.

        The first array inside the archive is taken. If 1D, it is expanded to
        shape [T, 1].

        Args:
            path: Path to .npz file. If None, returns None.

        Returns:
            Tensor of shape [T, D] (float32), or None if `path` is None.

        Raises:
            RuntimeError: If the .npz has no arrays inside.
        """
        if path is None:
            return None
        d = np.load(path)
        if len(d.files) == 0:
            raise RuntimeError(f"No arrays inside {path}")
        arr = d[d.files[0]]
        arr = np.asarray(arr, dtype=np.float32)
        if arr.ndim == 1:
            arr = arr[:, None]
        return torch.from_numpy(arr).float()

    def preprocess_data(self, instance_data: Dict[str, Any]) -> Dict[str, Any]:
        if self.instance_transforms is not None:
            for k, transform in self.instance_transforms.items():
                if k in instance_data:
                    instance_data[k] = transform(instance_data[k])
        return instance_data

    @staticmethod
    def _filter_records_from_dataset(index: list) -> list:
        """
        Optional hook to filter index items by custom condition.

        Call from subclass __init__ before shuffling/limiting.
        """
        # Filter logic
        pass

    @staticmethod
    def _sort_index(index):
        """
        Optional hook to sort the index by custom rules.

        Call from subclass __init__ before shuffling/limiting and after filtering.
        """
        return sorted(index, key=lambda x: x["KEY_FOR_SORTING"])

    @staticmethod
    def _shuffle_and_limit_index(index, limit, shuffle_index):
        """
        Optionally shuffle index (seed=42) and truncate to `limit` items.

        Args:
            index: List of sample dicts.
            limit: Optional max length after shuffling.
            shuffle_index: Whether to shuffle deterministically.

        Returns:
            New list after shuffling/limiting.
        """
        if shuffle_index:
            random.seed(42)
            random.shuffle(index)
        if limit is not None:
            index = index[:limit]
        return index
