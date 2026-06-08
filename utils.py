"""
Utility functions that are used across multiple modules.
"""

import argparse
import json
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import cast
import re
from typing import Any, Callable

import numpy as np
import pandas as pd
import yaml
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parent
RECORDINGS_DIR = ROOT / "recordings"
TSFRESH_DIR = ROOT / "tsfresh"
PLOT_DIR = ROOT / "plots"
PLOT_WIDTH = 1400
PLOT_HEIGHT = 300
PLOT_TITLE_FONTSIZE = 15
PLOT_LABEL_FONTSIZE = 14
PLOT_TICK_FONTSIZE = 12
TOOLS = [
    "pan",
    "wheel_zoom",
    "xwheel_zoom",
    "ywheel_zoom",
    "box_zoom",
    "xbox_zoom",
    "ybox_zoom",
    "reset",
    "save",
]
TIME_COLORS = ["#1f77b4", "#d62728", "#2ca02c", "#ff7f0e"]
FREQ_COLORS = ["#1f77b4", "#d62728", "#2ca02c"]
COLORS = {"healthy": "#2ca02c", "1shim": "#ff7f0e", "2shims": "#d62728"}
MARKERS = {"0p05nm": "o", "0p1nm": "s"}
MODALITY_ORDER = ["vibration", "torque", "ultrasound"]
MODALITY_PREFIXES = {
    "vibration": ("ax__", "ay__", "az__"),
    "torque": ("torque__",),
    "ultrasound": ("us__",),
}
ULTRASOUND_MAX_ABS = 0.5
EMPTY_TS = np.empty((0,), dtype=np.float64)
CONFIG_PATH = ROOT / "config.yaml"
HEALTHY_FILE_SPLIT_KEY = "healthy_file_split"
RECORDING_FILE_RE = re.compile(
    r"^(?P<label>[^-]+)-1000rpm-(?P<config>[^-]+)-run(?P<run>\d+)\.npz$"
)
RECORDING_FILE_RE_NO_RUN = re.compile(
    r"^(?P<label>[^-]+)-1000rpm-(?P<config>[^-]+)-(?P<suffix>[^.]+)\.npz$"
)


def configure_matplotlib_plot_style() -> None:
    """Apply common Matplotlib text sizes for exported analysis plots."""
    import matplotlib.pyplot as plt

    plt.rcParams.update(
        {
            "axes.titlesize": PLOT_TITLE_FONTSIZE,
            "axes.labelsize": PLOT_LABEL_FONTSIZE,
            "xtick.labelsize": PLOT_TICK_FONTSIZE,
            "ytick.labelsize": PLOT_TICK_FONTSIZE,
        }
    )


def max_numeric_value(x: object) -> float | None:
    """Return a useful numeric maximum from scalar/list/dict status values."""
    if x is None:
        return None
    if isinstance(x, (int, float)):
        return float(x)
    if isinstance(x, dict):
        for key in ("max", "val_max", "odr_max", "max_odr"):
            value = x.get(key)
            if isinstance(value, (int, float)):
                return float(value)
        values = [
            float(value) for value in x.values() if isinstance(value, (int, float))
        ]
        return max(values) if values else None
    if isinstance(x, (list, tuple)):
        values = [float(value) for value in x if isinstance(value, (int, float))]
        return max(values) if values else None
    return None


def load_config() -> dict:
    if not CONFIG_PATH.exists():
        return {}
    with CONFIG_PATH.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


@dataclass(frozen=True)
class AutoencoderTuning:
    threshold_pct: float
    ema_alpha: float
    persistence_windows: int
    lambda_h: float
    lambda_shk: float
    calibrate_with_shaking: bool = True


@dataclass(frozen=True)
class TwoComponentCalibration:
    mu_z: np.ndarray
    cov_inv: np.ndarray
    mu_shk: np.ndarray
    cov_inv_shk: np.ndarray
    threshold: float
    persistence_windows: int


def autoencoder_tuning_from_dict(values: dict[str, float | bool]) -> AutoencoderTuning:
    return AutoencoderTuning(
        threshold_pct=float(values["threshold_pct"]),
        ema_alpha=float(values["ema_alpha"]),
        persistence_windows=int(values["persistence_windows"]),
        lambda_h=float(values["lambda_h"]),
        lambda_shk=float(values["lambda_shk"]),
        calibrate_with_shaking=bool(values.get("calibrate_with_shaking", True)),
    )


def parameter_paths(directory: Path, stem: str) -> tuple[Path, Path]:
    return directory / f"{stem}.pt", directory / f"{stem}_meta.pkl"


def save_torch_parameter(
    directory: Path, stem: str, state_dict: dict, meta: dict
) -> None:
    import torch

    directory.mkdir(parents=True, exist_ok=True)
    model_path, meta_path = parameter_paths(directory, stem)
    torch.save(state_dict, model_path)
    with meta_path.open("wb") as handle:
        pickle.dump(meta, handle)


def load_torch_parameter(
    directory: Path, stem: str, *, map_location
) -> tuple[dict, dict]:
    import torch

    model_path, meta_path = parameter_paths(directory, stem)
    if not model_path.exists() or not meta_path.exists():
        raise FileNotFoundError(f"Missing parameter pair for {stem!r} in {directory}")
    with meta_path.open("rb") as handle:
        meta = pickle.load(handle)
    return torch.load(model_path, map_location=map_location), meta


def safe_roc_auc(labels: np.ndarray, scores: np.ndarray) -> float:
    return (
        float(roc_auc_score(labels, scores))
        if np.unique(labels).size == 2
        else float("nan")
    )


def anomaly_group_hit_rates(
    decisions: np.ndarray,
    labels: np.ndarray,
    runs: np.ndarray,
    coupler1_runs: set[int],
    coupler2_runs: set[int],
) -> dict[str, float]:
    groups = {
        "c1_1shim_hit": np.isin(runs, list(coupler1_runs)) & (labels == "1shim"),
        "c1_2shims_hit": np.isin(runs, list(coupler1_runs)) & (labels == "2shims"),
        "c2_1shim_hit": np.isin(runs, list(coupler2_runs)) & (labels == "1shim"),
        "c2_2shims_hit": np.isin(runs, list(coupler2_runs)) & (labels == "2shims"),
    }
    return {
        name: float(decisions[mask].mean())
        for name, mask in groups.items()
        if mask.sum()
    }


def evaluation_metrics(
    labels: np.ndarray,
    scores: np.ndarray,
    decisions: np.ndarray,
    runs: np.ndarray,
    coupler1_runs: set[int],
    coupler2_runs: set[int],
) -> dict[str, float]:
    """Return common AUC/recall metrics for anomaly-detection evaluations."""
    is_anomaly = (labels != "healthy").astype(int)
    c1 = np.isin(runs, list(coupler1_runs))
    c2 = np.isin(runs, list(coupler2_runs))

    def recall(mask: np.ndarray) -> float:
        positives = is_anomaly[mask].astype(bool)
        return (
            float(decisions[mask][positives].mean())
            if positives.any()
            else float("nan")
        )

    return {
        "auc_all": safe_roc_auc(is_anomaly, scores),
        "auc_c1": safe_roc_auc(is_anomaly[c1], scores[c1]),
        "auc_c2": safe_roc_auc(is_anomaly[c2], scores[c2]),
        "recall_all": recall(np.ones_like(is_anomaly, dtype=bool)),
        "recall_c1": recall(c1),
        "recall_c2": recall(c2),
    }


def normalizer_stats(meta: dict, modality: str):
    return meta.get("norms", {}).get(modality) or meta.get(
        {"vib": "norm_vib", "torque": "norm_torque", "us": "norm_us"}[modality]
    )


def save_config(config: dict) -> None:
    with CONFIG_PATH.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(config, handle, sort_keys=False)


def _filename_keys(filename: str | Path) -> set[str]:
    name = Path(str(filename)).name
    return {name, Path(name).stem}


def load_healthy_file_split() -> dict[str, set[str]]:
    """Return configured healthy train/val/test file names.

    File entries may be written either as full names, e.g. ``foo.npz``, or as
    stems, e.g. ``foo``. Empty lists mean the caller should fall back to its
    legacy run-based split.
    """
    split_cfg = load_config().get(HEALTHY_FILE_SPLIT_KEY, {})
    out: dict[str, set[str]] = {}
    for split in ("train", "val", "test"):
        values = split_cfg.get(f"{split}_files", []) or []
        normalized: set[str] = set()
        for value in values:
            normalized.update(_filename_keys(value))
        out[split] = normalized
    return out


def healthy_file_split_configured(split: str | None = None) -> bool:
    split_map = load_healthy_file_split()
    if split is not None:
        return bool(split_map.get(split, set()))
    return any(split_map.values())


def is_healthy_split_member(
    filename: str | Path,
    label: str,
    run: int,
    is_shaking: bool,
    split: str,
    fallback_runs: set[int],
) -> bool:
    if label != "healthy" or is_shaking:
        return False
    configured_files = load_healthy_file_split().get(split, set())
    if configured_files:
        return bool(_filename_keys(filename) & configured_files)
    return int(run) in fallback_runs


def healthy_split_mask(
    filenames,
    labels,
    runs,
    shaking,
    split: str,
    fallback_runs: set[int],
) -> np.ndarray:
    labels_arr = np.asarray(labels)
    runs_arr = np.asarray(runs)
    shaking_arr = np.asarray(shaking, dtype=bool)
    base = (labels_arr == "healthy") & ~shaking_arr
    configured_files = load_healthy_file_split().get(split, set())
    if configured_files:
        file_mask = np.array(
            [bool(_filename_keys(fname) & configured_files) for fname in filenames],
            dtype=bool,
        )
        return base & file_mask
    return base & np.isin(runs_arr, list(fallback_runs))


def healthy_training_pool_mask(
    filenames,
    labels,
    runs,
    shaking,
    train_fallback_runs: set[int],
) -> np.ndarray:
    """Return the healthy rows used before the train/validation split."""
    mask = healthy_split_mask(
        filenames, labels, runs, shaking, "train", train_fallback_runs
    )
    if healthy_file_split_configured("val"):
        return mask | healthy_split_mask(filenames, labels, runs, shaking, "val", set())
    if healthy_file_split_configured("train"):
        return mask
    return mask | healthy_split_mask(
        filenames, labels, runs, shaking, "val", train_fallback_runs
    )


def ema_smooth(scores: np.ndarray, filenames: np.ndarray, alpha: float) -> np.ndarray:
    """Apply per-file EMA smoothing so window order is respected within recordings."""
    smoothed = scores.copy()
    for fname in np.unique(filenames):
        idx = np.where(filenames == fname)[0]
        if idx.size == 0:
            continue
        values = scores[idx]
        out = np.empty_like(values)
        out[0] = values[0]
        for i in range(1, len(values)):
            out[i] = alpha * values[i] + (1.0 - alpha) * out[i - 1]
        smoothed[idx] = out
    return smoothed


def persistence_filter(
    predictions: np.ndarray,
    filenames: np.ndarray,
    min_consecutive: int,
) -> np.ndarray:
    """Require consecutive positive windows within one file before raising an alarm."""
    filtered = np.zeros_like(predictions, dtype=bool)
    for fname in np.unique(filenames):
        idx = np.where(filenames == fname)[0]
        streak = 0
        for pos, i in enumerate(idx):
            streak = streak + 1 if predictions[i] else 0
            if streak >= min_consecutive:
                filtered[idx[pos - min_consecutive + 1 : pos + 1]] = True
    return filtered


def fit_mahalanobis(
    z: np.ndarray,
    regularization: float = 0.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Fit a Mahalanobis distance model to rows of ``z``."""
    mu = z.mean(axis=0)
    cov = np.cov(z, rowvar=False)
    if regularization:
        cov = cov + regularization * np.eye(cov.shape[0])
    return mu, np.linalg.pinv(cov)


def mahalanobis_scores(
    z: np.ndarray,
    mu: np.ndarray,
    cov_inv: np.ndarray,
) -> np.ndarray:
    diff = z - mu
    return np.einsum("ni,ij,nj->n", diff, cov_inv, diff)


def two_component_scores(
    z: np.ndarray,
    mu_a: np.ndarray,
    cov_inv_a: np.ndarray,
    mu_b: np.ndarray,
    cov_inv_b: np.ndarray,
) -> np.ndarray:
    return np.minimum(
        mahalanobis_scores(z, mu_a, cov_inv_a),
        mahalanobis_scores(z, mu_b, cov_inv_b),
    )


def encode_batches(
    encode_fn: Callable[..., Any],
    inputs: tuple[np.ndarray, ...],
    *,
    device,
    batch_size: int = 256,
    latent_dim: int = 0,
) -> np.ndarray:
    """Run a model encoder over one or more aligned numpy inputs."""
    import torch

    n = inputs[0].shape[0] if inputs else 0
    out = []
    with torch.no_grad():
        for start in range(0, n, batch_size):
            tensors = tuple(
                torch.from_numpy(x[start : start + batch_size]).to(device)
                for x in inputs
            )
            encoded = encode_fn(*tensors).detach().cpu().numpy()
            out.append(encoded)
    return np.concatenate(out) if out else np.empty((0, latent_dim), dtype=np.float32)


def train_autoencoder(
    model,
    train_inputs: tuple[np.ndarray, ...],
    val_inputs: tuple[np.ndarray, ...],
    *,
    device,
    epochs: int,
    batch_size: int,
    lr: float,
    patience: int,
    prefix: str = "Epoch",
    weight_decay: float = 1e-5,
) -> list[float]:
    """Train a PyTorch autoencoder model on the given training inputs, using the validation inputs for early stopping and returning the validation loss history."""
    import torch
    import torch.nn as nn

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    mse = nn.MSELoss()
    n = train_inputs[0].shape[0]
    train_tensors = tuple(torch.from_numpy(x).to(device) for x in train_inputs)
    val_tensors = (
        tuple(torch.from_numpy(x).to(device) for x in val_inputs)
        if val_inputs and val_inputs[0].shape[0]
        else None
    )
    best_val, wait, best_state = float("inf"), 0, None
    use_amp = device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)  # type: ignore
    history: list[float] = []

    def reconstruction_loss(inputs: tuple[Any, ...]) -> Any:
        outputs = model(*inputs)
        reconstructions = outputs[1:]
        return sum(mse(recon, target) for recon, target in zip(reconstructions, inputs))

    for epoch in range(1, epochs + 1):
        model.train()
        idx = torch.randperm(n, device=device)
        epoch_loss = torch.zeros((), device=device)
        for start in range(0, n, batch_size):
            batch_idx = idx[start : start + batch_size]
            batch_inputs = tuple(x[batch_idx] for x in train_tensors)
            with torch.amp.autocast("cuda", enabled=use_amp):  # type: ignore
                loss = reconstruction_loss(batch_inputs)
            optimizer.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            epoch_loss += loss.detach() * len(batch_idx)
        epoch_loss_value = float(epoch_loss.cpu()) / n

        model.eval()
        if val_tensors is None:
            val_loss = epoch_loss_value
        else:
            with torch.no_grad():
                with torch.amp.autocast("cuda", enabled=use_amp):  # type: ignore
                    val_loss = float(reconstruction_loss(val_tensors).item())
        history.append(val_loss)
        print(f"{prefix} {epoch:3d}  train={epoch_loss_value:.5f}  val={val_loss:.5f}")

        if val_loss < best_val - 1e-6:
            best_val, wait = val_loss, 0
            best_state = {k: v.clone().cpu() for k, v in model.state_dict().items()}
        else:
            wait += 1
            if wait >= patience:
                print(f"Early stopping at epoch {epoch}")
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    return history


def fit_two_component(
    z_healthy: np.ndarray,
    z_shaking: np.ndarray,
    lambda_h: float,
    lambda_shk: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Fit two Mahalanobis distance components for healthy and shaking windows."""
    mu_z, cov_inv = fit_mahalanobis(z_healthy, lambda_h)
    if z_shaking.shape[0] >= 2:
        mu_shk, cov_inv_shk = fit_mahalanobis(z_shaking, lambda_shk)
    else:
        mu_shk, cov_inv_shk = mu_z, cov_inv
    return mu_z, cov_inv, mu_shk, cov_inv_shk


def calibrate_two_component_threshold(
    z: np.ndarray,
    filenames: np.ndarray,
    mu_z: np.ndarray,
    cov_inv: np.ndarray,
    mu_shk: np.ndarray,
    cov_inv_shk: np.ndarray,
    *,
    ema_alpha: float,
    threshold_percentile: float,
) -> float:
    """Calibrate a Mahalanobis distance threshold based on the healthy distribution component."""
    scores = ema_smooth(
        two_component_scores(z, mu_z, cov_inv, mu_shk, cov_inv_shk),
        filenames,
        ema_alpha,
    )
    return float(np.percentile(scores, threshold_percentile))


def print_score_summary(
    modalities,
    ema_alpha: float,
    threshold: float,
    persistence_windows: int,
    metrics: dict[str, float],
    labels: np.ndarray,
    runs: np.ndarray,
    shaking: np.ndarray,
    decisions: np.ndarray,
    coupler1_runs: set[int],
    coupler2_runs: set[int],
) -> None:
    """Print a concise summary of the scoring results and metrics."""
    is_anomaly = (labels != "healthy").astype(int)
    print(
        f"\nModalities={'+'.join(modalities)}  EMA alpha={ema_alpha}  "
        f"threshold={threshold:.2f}  persistence={persistence_windows}"
    )
    if is_anomaly.sum() > 0 and (~is_anomaly.astype(bool)).sum() > 0:
        print(
            f"ROC-AUC (all): {metrics['auc_all']:.4f}  "
            f"Recall (all): {metrics['recall_all']:.4f}"
        )
    for run_set, name in [
        (coupler1_runs, "coupler-1 runs 1-4"),
        (coupler2_runs, "coupler-2 runs 5-6"),
    ]:
        mask = np.isin(runs, list(run_set))
        if is_anomaly[mask].sum() > 0 and (~is_anomaly[mask].astype(bool)).sum() > 0:
            suffix = "c1" if run_set == coupler1_runs else "c2"
            print(
                f"ROC-AUC ({name}): {metrics[f'auc_{suffix}']:.4f}  "
                f"Recall ({name}): {metrics[f'recall_{suffix}']:.4f}"
            )
    if shaking.sum() > 0:
        print(f"Shaking false-positive rate: {decisions[shaking].mean():.3f}")


def save_score_outputs(
    output_path: Path,
    *,
    modalities,
    ema_alpha: float,
    threshold: float,
    persistence_windows: int,
    metrics: dict[str, float],
    filenames: np.ndarray,
    labels: np.ndarray,
    runs: np.ndarray,
    shaking: np.ndarray,
    raw_scores: np.ndarray,
    scores: np.ndarray,
    decisions: np.ndarray,
    coupler1_runs: set[int],
    coupler2_runs: set[int],
    export_plot: bool,
    score_upper: Callable[[np.ndarray, float], float],
    bins_count: int,
    titles: tuple[str, str, str],
) -> None:
    """Save common autoencoder score histograms and their companion JSON file."""
    if export_plot:
        import matplotlib.pyplot as plt

        output_path.parent.mkdir(exist_ok=True)
        fig, axes = plt.subplots(1, 3, figsize=(18, 4), sharey=True)
        for ax, run_set, title in [
            (axes[0], coupler1_runs, titles[0]),
            (axes[1], coupler2_runs, titles[1]),
        ]:
            panel = np.isin(runs, list(run_set)) & ~shaking
            panel_scores = scores[panel]
            upper = (
                score_upper(panel_scores, threshold)
                if panel_scores.size
                else threshold + 1.0
            )
            bins = np.linspace(0.0, upper, bins_count)
            for label, color in COLORS.items():
                mask = panel & (labels == label)
                if mask.sum():
                    ax.hist(
                        scores[mask],
                        bins=bins,
                        alpha=0.5,
                        label=label,
                        color=color,
                        density=True,
                    )
            ax.axvline(threshold, color="black", linestyle="--", label="threshold")
            ax.set_title(title)
            ax.set_xlabel("Mahalanobis score")
            ax.set_xlim(bins[0], bins[-1])
            ax.legend()

        ax = axes[2]
        panel = (~shaking & (labels == "healthy")) | shaking
        panel_scores = scores[panel]
        upper = (
            score_upper(panel_scores, threshold)
            if panel_scores.size
            else threshold + 1.0
        )
        bins = np.linspace(0.0, upper, bins_count)
        for mask, label, color in [
            (
                ~shaking & (labels == "healthy"),
                "healthy non-shaking",
                COLORS["healthy"],
            ),
            (shaking, "shaking", "#9467bd"),
        ]:
            if mask.sum():
                ax.hist(
                    scores[mask],
                    bins=bins,
                    alpha=0.5,
                    label=label,
                    color=color,
                    density=True,
                )
        ax.axvline(threshold, color="black", linestyle="--")
        ax.set_title(titles[2])
        ax.set_xlabel("Mahalanobis score")
        ax.set_xlim(bins[0], bins[-1])
        ax.legend()
        fig.tight_layout()
        fig.savefig(output_path, bbox_inches="tight")
        plt.close(fig)
        print(f"Saved {output_path}")

    json_out = save_score_json(
        output_path,
        {
            "modalities": list(modalities),
            "ema_alpha": float(ema_alpha),
            "threshold": float(threshold),
            "persistence_windows": int(persistence_windows),
            "metrics": metrics,
            "scores": [
                {
                    "filename": str(filename),
                    "label": str(label),
                    "run": int(run),
                    "shaking": bool(shake),
                    "raw_score": float(raw_score),
                    "score": float(score),
                    "decision": bool(decision),
                }
                for filename, label, run, shake, raw_score, score, decision in zip(
                    filenames, labels, runs, shaking, raw_scores, scores, decisions
                )
            ],
        },
    )
    print(f"Saved {json_out}")


def load_us_cache_features(
    us_features: list[str],
    us_window_tag: str,
    root: Path = ROOT,
) -> pd.DataFrame:
    """Load cached tsfresh features for ultrasound modality from parquet files."""
    frames = []
    cache_dir = root / "tsfresh" / "cache" / us_window_tag / "us"
    for feat_path in sorted(cache_dir.glob("*_feat.parquet")):
        frames.append(pd.read_parquet(feat_path, columns=us_features))
    if not frames:
        return pd.DataFrame(columns=pd.Index(us_features))
    out = pd.concat(frames)
    return out.loc[~out.index.duplicated(keep="first")].copy()


def parse_modalities_arg(
    values: list[str] | None,
    aliases: dict[str, str],
    default=None,
    *,
    as_tuple: bool = False,
):
    if values is None:
        return default
    out = []
    for value in values:
        for token in value.replace(",", " ").split():
            modality = aliases.get(token.strip().lower())
            if modality is None:
                raise ValueError(
                    f"Unknown modality '{token}'. Use vib/vibration, torque/current, or us/ultrasound."
                )
            if modality not in out:
                out.append(modality)
    if not out:
        raise ValueError("At least one modality must be selected.")
    return tuple(out) if as_tuple else out


def modality_parameter_stem(
    modalities,
    prefix: str,
    *,
    default_modalities=None,
    default_stem: str | None = None,
) -> str:
    modalities_tuple = tuple(modalities)
    if default_modalities is not None and modalities_tuple == tuple(default_modalities):
        return default_stem or prefix
    return f"{prefix}_{'_'.join(modalities_tuple)}"


def modality_tuning_key(modalities) -> str:
    return "_".join(tuple(modalities))


def load_modality_tuning(
    section: str,
    modalities,
    defaults: dict[str, Any],
) -> dict[str, Any]:
    config = load_config()
    tuning = config.get("tuning", {}).get(section, {})
    resolved = dict(defaults)
    resolved.update(tuning.get(modality_tuning_key(modalities), {}) or {})
    return resolved


def save_modality_tuning(
    section: str,
    modalities,
    params: dict[str, Any],
) -> None:
    config = load_config()
    tuning = config.setdefault("tuning", {})
    section_cfg = tuning.setdefault(section, {})
    section_cfg[modality_tuning_key(modalities)] = dict(params)
    save_config(config)


def parse_embedding_plot_args(
    include_3d: bool = False,
    include_source_flags: bool = False,
    include_tsfresh_parquet: bool = True,
) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    for modality in MODALITY_ORDER:
        parser.add_argument(f"--{modality}", action="store_true")
    if include_3d:
        parser.add_argument(
            "--3d",
            dest="plot_3d",
            action="store_true",
            help="Plot the embedding with three components instead of two.",
        )
    if include_source_flags:
        source = parser.add_mutually_exclusive_group()
        source.add_argument(
            "--conv",
            action="store_true",
            help="Use convolutional autoencoder features.",
        )
        source.add_argument(
            "--latent",
            action="store_true",
            help="Plot window-level autoencoder latent features.",
        )
        parser.add_argument(
            "--manual", action="store_true", help="Use tsfresh features."
        )
    if include_tsfresh_parquet:
        parser.add_argument(
            "--tsfresh-parquet",
            type=Path,
            help="Optional tsfresh parquet file to aggregate by filename and plot per recording.",
        )
    return parser.parse_args()


def selected_modalities(args: argparse.Namespace) -> list[str]:
    modalities = [name for name in MODALITY_ORDER if getattr(args, name)]
    if not modalities:
        raise ValueError(
            "Select at least one modality: --vibration, --torque, or --ultrasound"
        )
    return modalities


def format_run_label(items: list[tuple[Path, str, str, int]]) -> str:
    runs = sorted({run for _, _, _, run in items if run >= 0})
    if not runs:
        raise ValueError("No runs available in recording config")
    if len(runs) == 1:
        return f"Run {runs[0]}"
    if runs[-1] - runs[0] + 1 == len(runs):
        return f"Runs {runs[0]}-{runs[-1]}"
    return "Runs " + "_".join(str(run) for run in runs)


def default_tsfresh_parquet() -> Path:
    candidates = sorted(
        TSFRESH_DIR.glob("tsfresh_features_*_selected.parquet"),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    if not candidates:
        raise FileNotFoundError(
            f"No tsfresh selected parquet files found in {TSFRESH_DIR}"
        )
    return candidates[0]


def default_tsfresh_parquet_for(modality: str) -> Path:
    direct_map = {
        "vibration": TSFRESH_DIR
        / "tsfresh_features_vib_efficient_w4096_selected.parquet",
        "torque": TSFRESH_DIR
        / "tsfresh_features_vib_torque_efficient_vibw4096_torquew154_selected.parquet",
        "ultrasound": TSFRESH_DIR
        / "tsfresh_features_vib_torque_us_efficient_vibw4096_torquew154_usw29487_selected.parquet",
    }
    preferred = direct_map[modality]
    if preferred.exists():
        return preferred
    return default_tsfresh_parquet()


def load_recording_config(
    recordings_dir: Path | None = None,
) -> list[tuple[Path, str, str, int]]:
    config = load_config()["npz_selection"]

    recordings_root = (
        recordings_dir
        if recordings_dir is not None
        else Path(__file__).resolve().parent / "recordings"
    )
    labels = set(config["labels"])
    runs = {int(run_id) for run_id in config["runs"]}
    loads = {str(load).replace("p", ".").removesuffix("nm") for load in config["loads"]}
    excluded_files = set(config.get("excluded_files", []))
    included_files = set(config.get("included_files", []))

    items: list[tuple[Path, str, str, int]] = []
    for path in sorted(recordings_root.glob("*.npz")):
        if path.name in excluded_files:
            continue
        match = RECORDING_FILE_RE.match(path.name)
        if not match:
            if path.name not in included_files:
                continue
            match2 = RECORDING_FILE_RE_NO_RUN.match(path.name)
            if not match2:
                continue
            suffix = match2.group("suffix")
            label = match2.group("label")
            config_name = match2.group("config")
            run_match = re.search(r"run(?P<run>\d+)", suffix)
            run = int(run_match.group("run")) if run_match else -1
            items.append((path, label, config_name, run))
            continue
        label = match.group("label")
        config_name = match.group("config")
        run = int(match.group("run"))
        if path.name not in included_files:
            if label not in labels:
                continue
            if run not in runs:
                continue
            if _normalize_load_token(config_name) not in loads:
                continue
        items.append((path, label, config_name, run))
    return items


def load_data(
    recordings_dir: Path | None = None,
) -> list[dict]:
    """Load selected recording metadata."""
    return [
        {
            "path": path,
            "filename": path.stem,
            "label": label,
            "load": load,
            "run": run,
            "is_shaking": "shaking" in path.stem,
        }
        for path, label, load, run in load_recording_config(recordings_dir)
    ]


def _normalize_load_token(token: str) -> str:
    normalized = token.replace("p", ".")
    if normalized.endswith("nm"):
        normalized = normalized[:-2]
    return normalized


def build_time_axis(
    fallback_t: np.ndarray, ts_arr: np.ndarray, n_rows: int
) -> np.ndarray:
    if ts_arr.ndim != 2 or ts_arr.shape[0] < n_rows or ts_arr.shape[1] < 3:
        return fallback_t

    tx_col = 3 if ts_arr.shape[1] >= 4 else 2
    tx = ts_arr[:n_rows, tx_col]
    valid = np.isfinite(tx) & (tx >= 946684800.0) & (tx <= 4102444800.0)
    if np.count_nonzero(valid) < 2:
        return fallback_t

    rel = tx[valid] - tx[valid][0]
    if not np.all(np.isfinite(rel)) or float(np.nanmax(rel)) > 1e6:
        return fallback_t

    out = np.full(n_rows, np.nan, dtype=np.float64)
    out[valid] = rel
    if np.any(~valid):
        out[~valid] = np.interp(
            np.flatnonzero(~valid).astype(np.float64),
            np.flatnonzero(valid).astype(np.float64),
            out[valid],
        )
    return out


def discard_corrupt_ultrasound_readings(us: np.ndarray) -> np.ndarray:
    if us.ndim != 2 or us.shape[1] < 2:
        return us
    valid = np.isfinite(us[:, 0]) & np.isfinite(us[:, 1])
    valid &= np.abs(us[:, 1]) <= ULTRASOUND_MAX_ABS
    return us[valid]


def plot_embedding_points(
    ax, points: np.ndarray, labels: list[str], configs: list[str], runs: list[int]
) -> None:
    for (x0, x1), label, config, run in zip(points, labels, configs, runs):
        annotation = f"{label}-{config}" if run < 0 else f"{label}-{config}-r{run}"
        ax.scatter(
            x0,
            x1,
            s=90,
            c=COLORS.get(label, "#7f7f7f"),
            marker=MARKERS.get(config, "o"),
            edgecolors="black",
            linewidths=0.6,
        )
        ax.annotate(
            annotation,
            (x0, x1),
            xytext=(5, 4),
            textcoords="offset points",
            fontsize=8,
        )


def add_embedding_legend(ax, configs: list[str]) -> None:
    from matplotlib.lines import Line2D

    class_handles = [
        Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            label=label,
            markerfacecolor=color,
            markeredgecolor="black",
            markersize=8,
        )
        for label, color in COLORS.items()
    ]
    config_handles = [
        Line2D(
            [0],
            [0],
            marker=MARKERS.get(config, "o"),
            color="black",
            label=config,
            linestyle="None",
            markersize=8,
        )
        for config in sorted(set(configs))
    ]
    ax.legend(handles=class_handles + config_handles, loc="best", fontsize=8)


def save_embedding_json(
    output_path: Path,
    method: str,
    modalities: list[str],
    run_label: str,
    points: np.ndarray,
    items: list[tuple[Path, str, str, int]],
    axes: list[str],
    metadata: dict | None = None,
) -> None:
    records = []
    for point, (path, label, config, run) in zip(points, items):
        point_arr = np.asarray(point, dtype=np.float64)
        record = {
            "filename": path.name,
            "recording": path.stem,
            "label": label,
            "config": config,
            "run": None if run < 0 else run,
            "x": float(point_arr[0]),
            "y": float(point_arr[1]),
        }
        if point_arr.size >= 3:
            record["z"] = float(point_arr[2])
        records.append(record)

    payload = {
        "method": method,
        "modalities": modalities,
        "run_label": run_label,
        "axes": axes,
        "metadata": metadata or {},
        "points": records,
    }
    output_path.with_suffix(".json").write_text(
        json.dumps(payload, indent=2),
        encoding="utf-8",
    )


def save_score_json(output_path: Path, payload: dict[str, Any]) -> Path:
    """Write score data next to a plotted score parameter."""
    json_path = output_path.with_suffix(".json")
    json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return json_path


def load_capture_data(path: Path) -> dict[str, object]:
    data = np.load(path, allow_pickle=True)

    sensor_name = str(np.asarray(data.get("sensor_name", [""]), dtype=object)[0])
    ultrasound_name = str(
        np.asarray(data.get("ultrasound_name", ["imp23absu_mic"]), dtype=object)[0]
    )
    plc = np.asarray(data.get("plc", np.empty((0, 2))), dtype=np.float64)
    sk_source = data.get("sensorkit", data.get("vibration", np.empty((0, 4))))
    sk = np.asarray(sk_source, dtype=np.float64)
    us = np.asarray(data.get("ultrasound", np.empty((0, 2))), dtype=np.float64)
    us = discard_corrupt_ultrasound_readings(us)
    plc_ts = np.asarray(data.get("plc_ts", np.empty((0, 3))), dtype=np.float64)
    sk_ts_source = data.get("sensorkit_ts", data.get("vibration_ts", np.empty((0, 3))))
    sk_ts = np.asarray(sk_ts_source, dtype=np.float64)

    plc_t = build_time_axis(plc[:, 0], plc_ts, plc.shape[0]) if plc.size else EMPTY_TS
    sk_t = build_time_axis(sk[:, 0], sk_ts, sk.shape[0]) if sk.size else EMPTY_TS
    us_t = us[:, 0] if us.size else EMPTY_TS

    return {
        "sensor_name": sensor_name,
        "ultrasound_name": ultrasound_name,
        "plc": plc,
        "plc_t": plc_t,
        "sk": sk,
        "sk_t": sk_t,
        "us": us,
        "us_t": us_t,
    }


def _compute_features_for_recording(
    path: Path,
    modalities: list[str],
    kind_to_fc: dict,
    window_size: int = 4096,
) -> pd.Series:
    """Extract tsfresh features for a single recording using a fixed feature schema."""
    from tsfresh import extract_features
    from tsfresh.utilities.dataframe_functions import impute

    capture = load_capture_data(path)
    signal_map = {
        "ax": (lambda c: c["sk"][:, 1], lambda c: c["sk_t"]),
        "ay": (lambda c: c["sk"][:, 2], lambda c: c["sk_t"]),
        "az": (lambda c: c["sk"][:, 3], lambda c: c["sk_t"]),
        "torque": (lambda c: c["plc"][:, -1], lambda c: c["plc_t"]),
        "us": (lambda c: c["us"][:, 1], lambda c: c["us_t"]),
    }
    active_kinds = set(kind_to_fc.keys())
    chunk_dfs: list[pd.DataFrame] = []
    for kind, (get_sig, get_t) in signal_map.items():
        if kind not in active_kinds:
            continue
        try:
            sig = get_sig(capture)
            t = get_t(capture)
        except (KeyError, IndexError):
            continue
        valid = np.isfinite(sig) & np.isfinite(t)
        sig, t = sig[valid], t[valid]
        n = len(sig)
        if n == 0:
            continue
        effective_window = window_size if window_size > 0 else n
        starts = list(range(0, n - effective_window + 1, effective_window)) or [0]
        for w_idx, start in enumerate(starts):
            end = min(start + effective_window, n)
            win_id = f"{path.stem}__w{w_idx:04d}"
            chunk_dfs.append(
                pd.DataFrame(
                    {
                        "id": win_id,
                        "time": t[start:end] - t[start],
                        "kind": kind,
                        "value": sig[start:end],
                    }
                )
            )
    if not chunk_dfs:
        raise ValueError(f"No signal data for {path.name}")
    ts_df = pd.concat(chunk_dfs, ignore_index=True)
    feat = extract_features(
        ts_df,
        column_id="id",
        column_sort="time",
        column_kind="kind",
        column_value="value",
        kind_to_fc_parameters=kind_to_fc,
        impute_function=impute,
        n_jobs=1,
        disable_progressbar=True,
    )
    return cast(pd.Series, pd.DataFrame(feat).mean())


def load_tsfresh_recording_features(
    parquet_path: Path,
    items: list[tuple[Path, str, str, int]],
    modalities: list[str],
    selected_features: list[str] | None = None,
) -> tuple[np.ndarray, list[str]]:
    import pyarrow.parquet as pq

    parquet_cols = pq.ParquetFile(parquet_path).schema.names
    if "filename" not in parquet_cols:
        raise ValueError(f"Expected 'filename' column in {parquet_path}")

    meta_cols = {"label", "load", "run", "filename"}
    prefixes = tuple(
        prefix for name in modalities for prefix in MODALITY_PREFIXES[name]
    )
    feature_source_cols = [
        col for col in parquet_cols if col not in meta_cols and col.startswith(prefixes)
    ]
    if selected_features is not None:
        requested = set(selected_features)
        feature_source_cols = [col for col in feature_source_cols if col in requested]
        missing_selected = [
            col for col in selected_features if col not in set(feature_source_cols)
        ]
        if missing_selected:
            raise ValueError(
                f"{parquet_path} is missing configured tsfresh features: "
                f"{missing_selected[:5]}"
            )
    feature_cols = feature_source_cols
    if not feature_cols:
        raise ValueError(
            f"No tsfresh feature columns found in {parquet_path} for modalities {modalities}"
        )
    source_by_normalized = dict(zip(feature_cols, feature_source_cols))

    cache_dirs = {
        "vibration": TSFRESH_DIR / "cache" / "w4096" / "vib",
        "torque": TSFRESH_DIR / "cache" / "vibw4096_torquew154" / "torque",
        "ultrasound": TSFRESH_DIR / "cache" / "vibw4096_torquew154_usw29487" / "us",
    }
    if all(cache_dirs[modality].exists() for modality in modalities):
        rows: list[pd.Series] = []
        available: list[str] = []
        missing_items: list[tuple[Path, str, str, int]] = []
        for item in items:
            stem = item[0].stem
            cache_frames = []
            cache_complete = True
            for modality in modalities:
                feat_path = cache_dirs[modality] / f"{stem}_feat.parquet"
                if not feat_path.exists():
                    cache_complete = False
                    break
                cache_cols = pq.ParquetFile(feat_path).schema.names
                source_cols = [
                    source_by_normalized[col]
                    for col in feature_cols
                    if col.startswith(MODALITY_PREFIXES[modality])
                    and source_by_normalized[col] in cache_cols
                ]
                if not source_cols:
                    cache_complete = False
                    break
                cache_frames.append(pd.read_parquet(feat_path, columns=source_cols))
            if cache_complete:
                row = cast(pd.Series, pd.concat(cache_frames, axis=1).mean())
                rows.append(row.reindex(feature_cols))
                available.append(stem)
            else:
                missing_items.append(item)

        if missing_items:
            print(
                f"  Skipping {len(missing_items)} recordings without cached tsfresh "
                f"{'/'.join(modalities)} features."
            )

        if rows:
            recording_df = pd.DataFrame(rows, index=pd.Index(available))
            valid_cols = recording_df.columns[recording_df.notna().all()].tolist()
            return recording_df[valid_cols].to_numpy(dtype=np.float64), available

    read_cols = [
        col for col in (*meta_cols, *feature_source_cols) if col in parquet_cols
    ]
    df = pd.read_parquet(parquet_path, columns=read_cols)

    recording_df = df.groupby("filename", sort=False)[feature_cols].mean()
    valid_cols = recording_df.columns[recording_df.notna().all()].tolist()

    names = [path.stem for path, _, _, _ in items]
    missing_items = [item for item in items if item[0].stem not in recording_df.index]

    if missing_items:
        from tsfresh.feature_extraction.settings import from_columns

        kind_to_fc = from_columns(valid_cols)
        extra_rows: list[pd.Series] = []
        for path, *_ in missing_items:
            print(f"  Computing features on-the-fly for {path.name} ...")
            row = _compute_features_for_recording(path, modalities, kind_to_fc)
            row.name = path.stem
            extra_rows.append(row.loc[valid_cols])
        extra_df = pd.DataFrame(extra_rows)
        recording_df = cast(
            pd.DataFrame, pd.concat([recording_df.loc[:, valid_cols], extra_df])
        )
    else:
        recording_df = cast(pd.DataFrame, recording_df.loc[:, valid_cols])

    available = [name for name in names if name in recording_df.index]
    if not available:
        raise ValueError(
            f"No tsfresh rows found for any requested recordings in {parquet_path}"
        )

    return recording_df.loc[available].to_numpy(dtype=np.float64), available


def load_combined_tsfresh_recording_features(
    items: list[tuple[Path, str, str, int]],
    modalities: list[str],
    selected_features: dict[str, list[str]] | None = None,
) -> tuple[np.ndarray, list[str], list[Path]]:
    parquet_to_modalities: dict[Path, list[str]] = {}
    for modality in modalities:
        parquet_path = default_tsfresh_parquet_for(modality)
        parquet_to_modalities.setdefault(parquet_path, []).append(modality)

    arrays: list[np.ndarray] = []
    available_per_parquet: list[list[str]] = []
    parquet_paths: list[Path] = list(parquet_to_modalities.keys())
    for parquet_path, grouped_modalities in parquet_to_modalities.items():
        requested = None
        if selected_features is not None:
            requested = [
                feature
                for modality in grouped_modalities
                for feature in selected_features.get(modality, [])
            ]
        arr, avail = load_tsfresh_recording_features(
            parquet_path, items, grouped_modalities, requested
        )
        arrays.append(arr)
        available_per_parquet.append(avail)

    common: list[str] = available_per_parquet[0]
    for avail in available_per_parquet[1:]:
        avail_set = set(avail)
        common = [name for name in common if name in avail_set]

    aligned = []
    for arr, avail in zip(arrays, available_per_parquet):
        idx = [avail.index(name) for name in common]
        aligned.append(arr[idx])

    return np.concatenate(aligned, axis=1), common, parquet_paths
