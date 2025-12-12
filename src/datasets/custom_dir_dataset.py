import json
from pathlib import Path

from tqdm import tqdm

from src.datasets.base_dataset import BaseDataset
from src.utils.io_utils import ROOT_PATH


class CustomDirDataset(BaseDataset):
    """
    Audio-visual dataset for two-speaker mixtures.

    Expects directory structure:
        data/
          audio/
            mix/*.(wav|flac|mp3)
            s1/*.(wav|flac|mp3)
            s2/*.(wav|flac|mp3)
          mouths/*.npz

    Notes:
        - Mouth features are assumed to exist for both speakers.
        - Audio files in mix/s1/s2 can have any of the supported extensions.
    """

    def __init__(
        self,
        data_dir=None,
        index_dir=None,
        *args,
        **kwargs,
    ):
        """
        Build or load an index for the given split and initialize the base class.

        Args:
            data_dir: Root directory with expected structure. This argument is required.
            index_dir: Directory to store/load index files (`index.json`).
                Defaults to `data_dir`. Must be writable in environments like Kaggle.
            *args, **kwargs: Passed through to `BaseDataset`.
        """
        self._AUDIO_EXTENSIONS = (".wav", ".flac", ".mp3")

        if data_dir is None:
            raise ValueError(f"data_dir must be specified")
        else:
            data_dir = Path(data_dir)

        self._data_dir = data_dir

        if index_dir is None:
            index_dir = data_dir
        else:
            index_dir = Path(index_dir)

        index_dir.mkdir(parents=True, exist_ok=True)
        self._index_dir = index_dir

        index = self._get_or_load_index()

        super().__init__(index, *args, **kwargs)

    def _get_or_load_index(self):
        """
        Load index from `<index_dir>/index.json` or create it if missing.

        Returns:
            List of sample dicts ready to be consumed by `BaseDataset`.
        """
        index_path = self._index_dir / "index.json"
        if index_path.exists():
            with index_path.open() as f:
                index = json.load(f)
        else:
            index = self._create_index()
            with index_path.open("w") as f:
                json.dump(index, f, indent=2)
        return index

    def _create_index(self):
        """
        Scan filesystem and build an index.

        Filename convention for mixtures is '<spk1>_<spk2>.<ext>',
        where <ext> is one of (wav, flac, mp3).
        Speaker ids are used to locate mouth features '<spk>.npz'.

        Returns:
            List of dictionaries with keys:
                'utt', 'mix_path', 's1_path', 's2_path',
                'mouth1_path', 'mouth2_path'.
        """
        mix_dir = self._data_dir / "audio" / "mix"
        s1_dir = self._data_dir / "audio" / "s1"
        s2_dir = self._data_dir / "audio" / "s2"
        mouths_dir = self._data_dir / "mouths"

        if not mix_dir.exists():
            raise FileNotFoundError(f"Mix dir not found: {mix_dir}")

        targets_available = s1_dir.exists() and s2_dir.exists()

        mix_files = sorted(
            p
            for p in mix_dir.iterdir()
            if p.is_file() and p.suffix in self._AUDIO_EXTENSIONS
        )
        index = []

        for mix_path in tqdm(mix_files, desc=f"Creating index"):
            base = mix_path.stem  # expected 'spk1_spk2'
            s1_path, s2_path = None, None

            if targets_available:
                for ext in self._AUDIO_EXTENSIONS:
                    cand = f"{base}{ext}"

                    cand_s1 = s1_dir / cand
                    if cand_s1.exists():
                        s1_path = cand_s1

                    cand_s2 = s2_dir / cand
                    if cand_s2.exists():
                        s2_path = cand_s2

                    if s1_path is not None and s2_path is not None:
                        break

                if s1_path is None or s2_path is None:
                    raise FileNotFoundError(
                        f"Targets for '{base}' not found in {s1_dir} / {s2_dir} "
                        f"with extensions {self._AUDIO_EXTENSIONS}"
                    )

            spk_ids = base.split("_")
            spk_1, spk_2 = spk_ids[0], spk_ids[1]

            mouth_1_path = mouths_dir / f"{spk_1}.npz"
            mouth_2_path = mouths_dir / f"{spk_2}.npz"

            if not mouth_1_path.exists() or not mouth_2_path.exists():
                raise FileNotFoundError(
                    f"Mouths for '{base}' not found in {mouths_dir}"
                )

            item = {
                "utt": base,
                "mix_path": str(mix_path),
                "mouth1_path": str(mouth_1_path),
                "mouth2_path": str(mouth_2_path),
                "s1_path": str(s1_path) if s1_path is not None else None,
                "s2_path": str(s2_path) if s2_path is not None else None,
            }
            index.append(item)

        return index
