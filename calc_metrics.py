import argparse
from pathlib import Path
from tqdm import tqdm

import torch
import torchaudio

from torchmetrics.functional.audio import (
    signal_noise_ratio as snr,
    signal_distortion_ratio as sdr,
    scale_invariant_signal_noise_ratio as si_snr,
    scale_invariant_signal_distortion_ratio as si_sdr,
    perceptual_evaluation_speech_quality as pesq,
    short_time_objective_intelligibility as stoi,
)

import matplotlib.pyplot as plt

AUDIO_EXTS = {".wav", ".flac", ".mp3"}


def load_audio(path: Path, target_sr: int = 16000) -> torch.Tensor:
    """
    Load an audio file, convert it to mono (if needed), resample it to `target_sr`,
    and return a 1D float tensor of shape (T,).
    """
    wav, sr = torchaudio.load(path)

    if wav.shape[0] > 1:
        wav = wav.mean(dim=0, keepdim=True)

    if sr != target_sr:
        wav = torchaudio.functional.resample(wav, sr, target_sr)

    return wav.squeeze(0)


def pair_files(gt_root: Path, pred_root: Path, speaker: str):
    """
    Return matched (gt_path, pred_path) pairs for the given speaker ('s1' or 's2').

    Ground-truth files are taken from:
        gt_root / "audio" / speaker

    Prediction files are taken from:
        pred_root / speaker

    Matching is done by filename. Files without corresponding predictions are skipped.
    """
    gt_dir = gt_root / speaker
    pred_dir = pred_root / speaker

    if not gt_dir.exists():
        print(f"[WARN] Ground-truth directory for {speaker} not found: {gt_dir}")
        return []
    if not pred_dir.exists():
        print(f"[WARN] Prediction directory for {speaker} not found: {pred_dir}")
        return []

    pairs = []
    for gt_path in sorted(gt_dir.iterdir()):
        if not gt_path.is_file() or gt_path.suffix.lower() not in AUDIO_EXTS:
            continue

        pred_path = pred_dir / gt_path.name
        if not pred_path.exists():
            print(f"[WARN] No prediction found for {speaker}/{gt_path.name}")
            continue

        pairs.append((gt_path, pred_path))

    return pairs


def compute_metrics_for_pair(
    gt_wav: torch.Tensor,
    pred_wav: torch.Tensor,
    mix_wav: torch.Tensor,
    sr: int = 16000,
):
    """
    Compute separation metrics between ground-truth and predicted waveforms.

    All signals are truncated to the minimum common length before evaluation.

    Metrics computed:
        - SNR
        - SDR
        - SI-SNR
        - SI-SDR
        - SI-SNRi
        - PESQ
        - STOI

    Returns:
        dict: mapping metric name -> torch.Tensor scalar
    """
    metrics = {}
    metrics["snr"] = snr(pred_wav, gt_wav)
    metrics["sdr"] = sdr(pred_wav, gt_wav)
    metrics["si_snr"] = si_snr(pred_wav, gt_wav)
    metrics["si_sdr"] = si_sdr(pred_wav, gt_wav)
    metrics["pesq"] = pesq(pred_wav, gt_wav, fs=sr, mode="wb")
    metrics["stoi"] = stoi(pred_wav, gt_wav, fs=sr, extended=False)

    si_snr_mix = si_snr(mix_wav, gt_wav)
    metrics["si_snri"] = metrics["si_snr"] - si_snr_mix

    return metrics


def mean_metrics(dict_list):
    """
    Compute mean of a list of metric dictionaries (with scalar tensors).

    NaN values are ignored per metric.

    Args:
        dict_list (list[dict]): list of dictionaries with identical keys

    Returns:
        dict | None: dictionary of metric_name -> float, or None if list is empty
    """
    if not dict_list:
        return None

    keys = dict_list[0].keys()
    means = {}
    for k in keys:
        vals = torch.stack([d[k] for d in dict_list if not torch.isnan(d[k])])
        if len(vals) == 0:
            means[k] = float("nan")
        else:
            means[k] = vals.mean().item()
    return means


def plot_metric_histograms(all_results, output_dir: Path):
    """
    Plot histograms of metric distributions for s1 and s2 on the same axes.

    Args:
        all_results (dict): dictionary with lists of metric dicts:
            {
              "s1": [ {metric -> tensor}, ... ],
              "s2": [ {metric -> tensor}, ... ],
            }
        output_dir (Path): directory to save histogram images to.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    non_empty_key = "s1" if all_results["s1"] else "s2"
    if not all_results[non_empty_key]:
        print("[WARN] No results to plot histograms.")
        return

    metric_names = list(all_results[non_empty_key][0].keys())

    for metric in metric_names:
        s1_vals = [
            d[metric].item() for d in all_results["s1"] if not torch.isnan(d[metric])
        ]
        s2_vals = [
            d[metric].item() for d in all_results["s2"] if not torch.isnan(d[metric])
        ]

        if not s1_vals and not s2_vals:
            print(f"[WARN] No valid values for metric {metric}, skipping histogram.")
            continue

        fig, ax = plt.subplots()
        if s1_vals:
            ax.hist(
                s1_vals,
                bins=50,
                alpha=0.5,
                label="s1",
            )
        if s2_vals:
            ax.hist(
                s2_vals,
                bins=50,
                alpha=0.5,
                label="s2",
            )

        ax.set_title(metric.upper())
        ax.set_xlabel("Value")
        ax.set_ylabel("Count")
        ax.legend()
        fig.tight_layout()

        out_path = output_dir / f"hist_{metric}.png"
        fig.savefig(out_path)
        plt.close(fig)

        print(f"[INFO] Saved histogram for {metric} to {out_path}")


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate speech separation predictions against ground-truth "
            "using SNR/SDR/SI-SNR/SI-SDR/SI-SNRi/PESQ/STOI metrics."
        )
    )
    parser.add_argument(
        "--gt-root",
        type=str,
        required=True,
        help=(
            "Path to ground-truth root directory "
            "Ground-truth audio is expected in /s1 and /s2, "
            "mixtures in /mix."
        ),
    )
    parser.add_argument(
        "--pred-root",
        type=str,
        required=True,
        help=(
            "Path to predictions root directory "
            "(containing 's1/' and 's2/' subdirectories with predicted audio)."
        ),
    )
    parser.add_argument(
        "--sr",
        type=int,
        default=16000,
        help="Sample rate used for evaluation (default: 16000).",
    )
    args = parser.parse_args()

    gt_root = Path(args.gt_root)
    pred_root = Path(args.pred_root)
    mix_root = gt_root / "mix"

    required_dirs = {
        "gt_s1": gt_root / "s1",
        "gt_s2": gt_root / "s2",
        "gt_mix": mix_root,
    }

    missing = [
        f"{name}: {path}" for name, path in required_dirs.items() if not path.is_dir()
    ]
    if missing:
        missing_str = "\n  ".join(missing)
        raise FileNotFoundError(
            "The following required ground-truth directories are missing:\n"
            f"  {missing_str}"
        )

    all_results = {"s1": [], "s2": []}
    overall_results = []

    for speaker in ["s1", "s2"]:
        pairs = pair_files(gt_root, pred_root, speaker)
        print(f"\n=== {speaker}: found {len(pairs)} file pairs ===")

        for gt_path, pred_path in tqdm(pairs):
            gt_wav = load_audio(gt_path, target_sr=args.sr)
            pred_wav = load_audio(pred_path, target_sr=args.sr)

            mix_path = mix_root / gt_path.name
            if not mix_path.exists():
                raise FileNotFoundError(
                    f"Mixture file not found for {gt_path.name}: {mix_path}"
                )
            mix_wav = load_audio(mix_path, target_sr=args.sr)

            m = compute_metrics_for_pair(gt_wav, pred_wav, mix_wav=mix_wav, sr=args.sr)
            all_results[speaker].append(m)
            overall_results.append(m)

    print("\n=== Mean metrics per speaker ===")
    for speaker in ["s1", "s2"]:
        means = mean_metrics(all_results[speaker])
        if means is None:
            print(f"{speaker}: no data")
            continue

        print(
            f"{speaker}: " + ", ".join(f"{k.upper()}={v:.3f}" for k, v in means.items())
        )

    print("\n=== Mean metrics over the whole dataset (s1 + s2) ===\n")
    overall_means = mean_metrics(overall_results)
    if overall_means is None:
        print("No data at all.")
    else:
        print(
            "ALL: "
            + ", ".join(f"{k.upper()}={v:.3f}" for k, v in overall_means.items())
        )

    hist_dir = pred_root / "metric_histograms"
    plot_metric_histograms(all_results, hist_dir)


if __name__ == "__main__":
    main()
