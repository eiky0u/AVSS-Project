import json
from pathlib import Path

from tqdm import tqdm

from src.datasets.base_dataset import BaseDataset
from src.utils.io_utils import ROOT_PATH


class AudioVisualDataset(BaseDataset):
    """
    Audio-visual dataset for two-speaker mixtures.

    Expects directory structure:
        data/
          audio/
            {split}/
              mix/*.wav
              s1/*.wav
              s2/*.wav
          mouths/*.npz

    Notes:
        - Mouth features are assumed to exist for both speakers.
    """

    def __init__(
        self,
        split: str = "train",
        data_dir=None,
        index_dir=None,
        *args,
        **kwargs,
    ):
        """
        Build or load an index for the given split and initialize the base class.

        Args:
            split: One of {"train", "val"}.
            data_dir: Root directory with expected structure. Defaults to
                `ROOT_PATH / "data"`.
            index_dir: Directory to store/load index files (`{split}_index.json`).
                Defaults to `data_dir`. Must be writable in environments like Kaggle.
            *args, **kwargs: Passed through to `BaseDataset`.
        """
        assert split in ["train", "val"]

        if data_dir is None:
            data_dir = ROOT_PATH / "data"
        else:
            data_dir = Path(data_dir)

        self._data_dir = data_dir

        if index_dir is None:
            index_dir = data_dir
        else:
            index_dir = Path(index_dir)

        index_dir.mkdir(parents=True, exist_ok=True)
        self._index_dir = index_dir

        index = self._get_or_load_index(split)

        super().__init__(index, *args, **kwargs)

    def _get_or_load_index(self, split):
        """
        Load index from `<index_dir>/{split}_index.json` or create it if missing.

        Args:
            split: Split name ("train", "val").

        Returns:
            List of sample dicts ready to be consumed by `BaseDataset`.
        """
        index_path = self._index_dir / f"{split}_index.json"
        if index_path.exists():
            with index_path.open() as f:
                index = json.load(f)
        else:
            index = self._create_index(split)
            with index_path.open("w") as f:
                json.dump(index, f, indent=2)
        return index

    def _create_index(self, split: str):
        """
        Scan filesystem and build an index for the given split.

        Filename convention for mixtures is '<spk1>_<spk2>.wav'.
        Speaker ids are used to locate mouth features '<spk>.npz'.

        Args:
            split: Split name ("train", "val").

        Returns:
            List of dictionaries with keys:
                'utt', 'mix_path', 's1_path', 's2_path',
                'mouth1_path', 'mouth2_path'.
        """
        split = str(split)
        mix_dir = self._data_dir / "audio" / split / "mix"
        s1_dir = self._data_dir / "audio" / split / "s1"
        s2_dir = self._data_dir / "audio" / split / "s2"
        mouths_dir = self._data_dir / "mouths"

        mix_files = sorted(mix_dir.glob("*.wav"))
        index = []

        for mix_path in tqdm(mix_files, desc=f"Creating index [{split}]"):
            base = mix_path.stem  # expected 'spk1_spk2'
            spk_ids = base.split("_")
            spk_1, spk_2 = spk_ids[0], spk_ids[1]

            mouth_1_path = mouths_dir / f"{spk_1}.npz"
            mouth_2_path = mouths_dir / f"{spk_2}.npz"

            s1_path = s1_dir / (base + ".wav")
            s2_path = s2_dir / (base + ".wav")

            item = {
                "utt": base,
                "mix_path": str(mix_path),
                "mouth1_path": str(mouth_1_path),
                "mouth2_path": str(mouth_2_path),
                "s1_path": str(s1_path),
                "s2_path": str(s2_path),
            }
            index.append(item)

        return index
