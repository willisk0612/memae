"""
Convolutional autoencoder for shaft misalignment anomaly detection. Uses raw vibration, torque, ultrasound as input.

Anomaly score = Mahalanobis distance of latent code from train distribution. Decoder is used only for training.

Usage:
  python conv_autoencoder.py
  python conv_autoencoder.py --latent-dim 8 --epochs 100 --modalities vib torque us
  python conv_autoencoder.py --eval-only
"""

from __future__ import annotations

import argparse
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import cast
from zipfile import BadZipFile

import matplotlib

matplotlib.use("Agg")
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from utils import (
    autoencoder_tuning_from_dict,
    calibrate_two_component_threshold,
    configure_matplotlib_plot_style,
    ema_smooth,
    encode_batches,
    evaluation_metrics,
    fit_two_component,
    is_healthy_split_member,
    load_capture_data,
    load_data,
    load_modality_tuning,
    modality_parameter_stem as make_modality_parameter_stem,
    parse_modalities_arg,
    persistence_filter as persistence,
    print_score_summary,
    save_score_outputs,
    load_torch_parameter,
    normalizer_stats,
    save_torch_parameter,
    train_autoencoder as train_autoencoder_model,
    two_component_scores,
)

RANDOM_STATE = 42
COUPLER1_RUNS = {1, 2, 3, 4}
COUPLER2_RUNS = {5, 6}
TRAIN_HEALTHY_RUNS = {1, 2, 3, 5}
VAL_HEALTHY_RUNS = {6}
TEST_HEALTHY_RUNS = {4}

ROOT = Path(__file__).resolve().parent
CACHE_DIR = ROOT / "conv_ae_cache"
MODELS_DIR = ROOT / "models" / "fusion"
PLOT_DIR = ROOT / "plots"
configure_matplotlib_plot_style()


# Ensure each window is ~153.58 ms
VIB_WINDOW = 4096  # 153.58ms*26.67kHz
US_WINDOW = 29487  # 153.58ms*192.0kHz
TORQUE_WINDOW = 154  # 153.58ms*1.0kHz
LATENT_DIM_DEFAULT = 8

torch.manual_seed(RANDOM_STATE)
np.random.seed(RANDOM_STATE)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if DEVICE.type == "cuda":
    torch.backends.cudnn.benchmark = True  # noqa
MODALITY_ALIASES = {
    "vib": "vib",
    "vibration": "vib",
    "torque": "torque",
    "current": "torque",
    "us": "us",
    "ultrasound": "us",
}

DEFAULT_TUNING = load_modality_tuning(
    "autoencoder_conv",
    ("vib", "torque"),
    {},
)
TUNING_DEFAULTS = DEFAULT_TUNING
DEFAULT_TUNING_STATE = autoencoder_tuning_from_dict(DEFAULT_TUNING)
THRESHOLD_PERCENTILE = DEFAULT_TUNING_STATE.threshold_pct
PERSISTENCE_WINDOWS = DEFAULT_TUNING_STATE.persistence_windows
LAMBDA_H = DEFAULT_TUNING_STATE.lambda_h
LAMBDA_SHK = DEFAULT_TUNING_STATE.lambda_shk
EMA_ALPHA_DEFAULT = DEFAULT_TUNING_STATE.ema_alpha
CALIB_WITH_SHAKING = DEFAULT_TUNING_STATE.calibrate_with_shaking


def modality_model_parameter_stem(modalities: tuple[str, ...]) -> str:
    """Return the canonical parameter stem shared with ae_modality_test.py."""
    case_stems = {
        ("vib",): "vib_only",
        ("torque",): "torque_only",
        ("us",): "us_only",
        ("vib", "torque"): "vib_torque",
        ("vib", "us"): "vib_us",
    }
    return case_stems.get(
        modalities,
        make_modality_parameter_stem(
            modalities,
            "fusion_conv_ae",
            default_modalities=("vib", "torque"),
            default_stem="fusion_conv_ae",
        ),
    )


def set_tuning_for_modalities(modalities: tuple[str, ...]) -> dict[str, float | bool]:
    global THRESHOLD_PERCENTILE, PERSISTENCE_WINDOWS, LAMBDA_H, LAMBDA_SHK, EMA_ALPHA_DEFAULT, CALIB_WITH_SHAKING

    tuning = load_modality_tuning("autoencoder_conv", modalities, TUNING_DEFAULTS)
    state = autoencoder_tuning_from_dict(tuning)
    THRESHOLD_PERCENTILE = state.threshold_pct
    PERSISTENCE_WINDOWS = state.persistence_windows
    LAMBDA_H = state.lambda_h
    LAMBDA_SHK = state.lambda_shk
    EMA_ALPHA_DEFAULT = state.ema_alpha
    CALIB_WITH_SHAKING = state.calibrate_with_shaking
    return tuning


@dataclass
class WindowBatch:
    vib: np.ndarray
    torque: np.ndarray
    us: np.ndarray
    filenames: np.ndarray
    labels: np.ndarray
    runs: np.ndarray
    shaking: np.ndarray


# ---------------------------------------------------------------------------
# Data caching: convert npz files to per-file tensors
# ---------------------------------------------------------------------------


def windowize(arr: np.ndarray, win: int) -> np.ndarray:
    """Slice a time-series array into non-overlapping fixed-length windows.

    Args:
        arr: Input array of shape (T, C) where T is time steps and C is channels.
        win: Window length in samples. Trailing samples that don't fill a full
            window are discarded.

    Returns:
        Array of shape (N, C, win) where N = T // win.
    """
    n = arr.shape[0] // win
    if n == 0:
        return np.empty((0, arr.shape[1], win), dtype=np.float32)
    truncated = arr[: n * win]
    return truncated.reshape(n, win, arr.shape[1]).transpose(0, 2, 1).astype(np.float32)


def build_or_load_cache(path: Path) -> dict | None:
    """Load windowed vibration and torque data for a recording, caching to disk.

    On first call for a given file the raw npz is loaded, windowed, and saved to
    CACHE_DIR. Subsequent calls read directly from the cache. Returns None when
    the recording yields zero usable windows (e.g. too short or missing channels).

    Args:
        path: Path to the raw recording .npz file.

    Returns:
        Dict with keys 'vib' (N, 3, VIB_WINDOW), 'torque' (N, 1, TORQUE_WINDOW), and
        'filename' (stem string), or None if no windows could be extracted.
    """
    cache_path = CACHE_DIR / f"{path.stem}.npz"

    def save_cache(**arrays: np.ndarray) -> None:
        """Write a cache file atomically so concurrent readers never see half a zip."""
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        fd, tmp_name = tempfile.mkstemp(
            prefix=f"{path.stem}.", suffix=".npz", dir=CACHE_DIR
        )
        os.close(fd)
        tmp_path = Path(tmp_name)
        try:
            np.savez(str(tmp_path), **arrays)  # pyright: ignore[reportArgumentType]
            os.replace(tmp_path, cache_path)
        finally:
            if tmp_path.exists():
                tmp_path.unlink()

    if cache_path.exists():
        try:
            with np.load(cache_path) as d:
                vib = d["vib"]
                torque = d["torque"]
                if vib.shape[0] == 0:
                    return None
                cached_us_valid = (
                    "us" in d
                    and d["us"].ndim == 3
                    and d["us"].shape[1:] == (1, US_WINDOW)
                )
                us = d["us"] if cached_us_valid else None
        except (BadZipFile, EOFError, OSError, ValueError):
            vib = torque = us = None
        else:
            if us is not None:
                return {
                    "vib": vib,
                    "torque": torque,
                    "us": us,
                    "filename": path.stem,
                }
            capture = load_capture_data(path)
            us_raw = np.asarray(capture["us"])[:, 1:2]
            us = windowize(us_raw, US_WINDOW)
            save_cache(vib=vib, torque=torque, us=us)
            return {
                "vib": vib,
                "torque": torque,
                "us": us,
                "filename": path.stem,
            }

    capture = load_capture_data(path)
    sk = np.asarray(capture["sk"])
    plc = np.asarray(capture["plc"])
    if sk.shape[0] == 0 or sk.shape[1] < 4 or plc.shape[0] == 0:
        save_cache(
            vib=np.empty((0, 3, VIB_WINDOW), dtype=np.float32),
            torque=np.empty((0, 1, TORQUE_WINDOW), dtype=np.float32),
        )
        return None

    vib_raw = sk[:, 1:4]  # (T_v, 3)
    torque_raw = plc[:, -1:]  # (T_c, 1)
    us_raw = np.asarray(capture["us"])[:, 1:2]  # (T_us, 1)
    vib = windowize(vib_raw, VIB_WINDOW)  # (N_v, 3, VIB_WINDOW)
    torque = windowize(torque_raw, TORQUE_WINDOW)  # (N_c, 1, TORQUE_WINDOW)
    us = windowize(us_raw, US_WINDOW)  # (N_us, 1, US_WINDOW)
    n_vib_torque = min(vib.shape[0], torque.shape[0])
    vib, torque = vib[:n_vib_torque], torque[:n_vib_torque]
    save_cache(vib=vib, torque=torque, us=us)
    if n_vib_torque == 0:
        return None
    return {"vib": vib, "torque": torque, "us": us, "filename": path.stem}


def load_dataset() -> list[dict]:
    """Load all recordings listed in the config plus any shaking files found on disk.

    Returns:
        List of dicts, one per file, each containing 'vib', 'torque', 'filename',
        'label', 'run', and 'is_shaking' fields.
    """
    out = []
    for item in load_data():
        path = item["path"]
        d = build_or_load_cache(path)
        if d is None:
            continue
        d["label"] = item["label"]
        d["run"] = item["run"]
        d["is_shaking"] = item["is_shaking"]
        out.append(d)
    return out


def stack_for_modalities(
    records: list[dict], modalities: tuple[str, ...]
) -> WindowBatch:
    """Stack records aligned only across the modalities used by a model."""
    vibs, torques, uss, fns, labels, runs, shakes = [], [], [], [], [], [], []
    for r in records:
        counts = []
        if "vib" in modalities:
            counts.append(r["vib"].shape[0])
        if "torque" in modalities:
            counts.append(r["torque"].shape[0])
        if "us" in modalities:
            counts.append(r["us"].shape[0])
        n = min(counts) if counts else 0
        if "vib" in modalities:
            vibs.append(r["vib"][:n])
        if "torque" in modalities:
            torques.append(r["torque"][:n])
        if "us" in modalities:
            uss.append(r["us"][:n])
        fns.append(np.array([r["filename"]] * n))
        labels.append(np.array([r["label"]] * n))
        runs.append(np.full(n, r["run"], dtype=np.int32))
        shakes.append(np.full(n, r["is_shaking"], dtype=bool))

    return WindowBatch(
        vib=(
            np.concatenate(vibs)
            if vibs
            else np.empty((0, 3, VIB_WINDOW), dtype=np.float32)
        ),
        torque=(
            np.concatenate(torques)
            if torques
            else np.empty((0, 1, TORQUE_WINDOW), dtype=np.float32)
        ),
        us=(
            np.concatenate(uss)
            if uss
            else np.empty((0, 1, US_WINDOW), dtype=np.float32)
        ),
        filenames=np.concatenate(fns) if fns else np.array([]),
        labels=np.concatenate(labels) if labels else np.array([]),
        runs=np.concatenate(runs) if runs else np.array([], dtype=np.int32),
        shaking=np.concatenate(shakes) if shakes else np.array([], dtype=bool),
    )


# ---------------------------------------------------------------------------
# Per-channel z-score normalization (computed on training set only)
# ---------------------------------------------------------------------------


class ChannelNorm:
    def __init__(self) -> None:
        self.mu: np.ndarray | None = None
        self.sd: np.ndarray | None = None

    def fit(self, x: np.ndarray) -> "ChannelNorm":
        """Compute per-channel mean and std from training windows.

        Args:
            x: Array of shape (N, C, T). Statistics are pooled over N and T.

        Returns:
            self, for thaining.
        """
        # x shape (N, C, T): mean+std per channel across N and T
        self.mu = x.mean(axis=(0, 2), keepdims=True).astype(np.float32)
        self.sd = (x.std(axis=(0, 2), keepdims=True) + 1e-6).astype(np.float32)
        return self

    def transform(self, x: np.ndarray) -> np.ndarray:
        """Apply z-score normalization using the fitted statistics.

        Args:
            x: Array of shape (N, C, T).

        Returns:
            Normalized array of the same shape as x.
        """
        return ((x - self.mu) / self.sd).astype(np.float32)


# ---------------------------------------------------------------------------
# Model: intermediate-fusion 1D convolutional autoencoder with two component Mahalanobis distance scoring in the latent space
# ---------------------------------------------------------------------------


def conv_block(c_in: int, c_out: int, k: int, pool: int) -> nn.Sequential:
    """Applies Conv1d + BatchNorm + ReLU + MaxPool1d, with padding to preserve length before pooling."""
    return nn.Sequential(
        nn.Conv1d(c_in, c_out, kernel_size=k, padding=k // 2),
        nn.BatchNorm1d(c_out),
        nn.ReLU(inplace=True),
        nn.MaxPool1d(pool),
    )


def deconv_block(c_in: int, c_out: int, k: int, scale: int) -> nn.Sequential:
    """Applies Upsample + Conv1d + BatchNorm + ReLU, with padding to preserve length after upsampling."""
    return nn.Sequential(
        nn.Upsample(scale_factor=scale, mode="linear", align_corners=False),
        nn.Conv1d(c_in, c_out, kernel_size=k, padding=k // 2),
        nn.BatchNorm1d(c_out),
        nn.ReLU(inplace=True),
    )


class ConvAutoencoder(nn.Module):
    # Hyperparameters uses the args c_in, c_out, k, pool for encoder specs and c_in, c_out, k, scale for decoder specs
    _MODALITY_CONFIG = {
        "vib": {
            "encoder": (
                (3, 16, 7, 4),
                (16, 32, 5, 4),
                (32, 64, 3, 4),
                (64, 64, 3, 4),
            ),
            "embed_dim": 64,
            "decoder_in": (64, 16),
            "decoder": (
                (64, 64, 3, 4),
                (64, 32, 3, 4),
                (32, 16, 5, 4),
                (16, 16, 7, 4),
            ),
            "out_channels": 3,
            "window": VIB_WINDOW,
        },
        "torque": {
            "encoder": (
                (1, 8, 5, 4),
                (8, 16, 3, 2),
                (16, 16, 3, 2),
            ),
            "embed_dim": 16,
            "decoder_in": (16, 9),
            "decoder": (
                (16, 16, 3, 2),
                (16, 8, 3, 2),
                (8, 8, 5, 4),
            ),
            "out_channels": 1,
            "window": TORQUE_WINDOW,
        },
        "us": {
            "encoder": (
                (1, 16, 7, 4),
                (16, 32, 5, 4),
                (32, 64, 3, 4),
                (64, 64, 3, 4),
            ),
            "embed_dim": 64,
            "decoder_in": (64, 16),
            "decoder": (
                (64, 64, 3, 4),
                (64, 32, 3, 4),
                (32, 16, 5, 4),
                (16, 16, 7, 4),
            ),
            "out_channels": 1,
            "window": US_WINDOW,
        },
    }

    def __init__(
        self,
        latent_dim: int = LATENT_DIM_DEFAULT,
        modalities: tuple[str, ...] = ("vib", "torque"),
    ) -> None:
        super().__init__()
        self.modalities = modalities
        if not modalities:
            raise ValueError("ConvAutoencoder needs at least one modality.")

        self.encoders = nn.ModuleDict()
        self.pools = nn.ModuleDict()
        self.decoder_inputs = nn.ModuleDict()
        self.decoders = nn.ModuleDict()
        embed_dims: dict[str, int] = {}

        for modality in modalities:
            embed_dims[modality] = self._build_modality_branch(modality, latent_dim)

        self.bottleneck = nn.Linear(sum(embed_dims.values()), latent_dim)

    def _build_modality_branch(self, modality: str, latent_dim: int) -> int:
        if modality not in self._MODALITY_CONFIG:
            raise ValueError(f"Unknown modality: {modality}")

        config = self._MODALITY_CONFIG[modality]
        decoder_channels, decoder_length = config["decoder_in"]
        decoder_specs = config["decoder"]
        last_decoder_channels = decoder_specs[-1][1]

        self.encoders[modality] = nn.Sequential(
            *(conv_block(*spec) for spec in config["encoder"])
        )
        self.pools[modality] = nn.AdaptiveAvgPool1d(1)
        self.decoder_inputs[modality] = nn.Linear(
            latent_dim, decoder_channels * decoder_length
        )
        self.decoders[modality] = nn.Sequential(
            *(deconv_block(*spec) for spec in decoder_specs),
            nn.Conv1d(last_decoder_channels, config["out_channels"], kernel_size=1),
        )
        return config["embed_dim"]

    def _modality_embedding(self, modality: str, x: torch.Tensor) -> torch.Tensor:
        """Return the fixed-width branch embedding for one modality input."""
        return self.pools[modality](self.encoders[modality](x)).squeeze(-1)

    def to_latent(self, *inputs: torch.Tensor) -> torch.Tensor:
        if len(inputs) != len(self.modalities):
            raise ValueError(
                f"Expected {len(self.modalities)} inputs, got {len(inputs)}"
            )
        encoded = [
            self._modality_embedding(modality, x)
            for modality, x in zip(self.modalities, inputs)
        ]
        return self.bottleneck(torch.cat(encoded, dim=1))

    def _decode_modality(self, modality: str, z: torch.Tensor) -> torch.Tensor:
        config = self._MODALITY_CONFIG[modality]
        channels, length = config["decoder_in"]
        out = self.decoders[modality](
            self.decoder_inputs[modality](z).view(-1, channels, length)
        )
        return F.interpolate(
            out, size=config["window"], mode="linear", align_corners=False
        )

    def from_latent(self, z: torch.Tensor) -> tuple[torch.Tensor, ...]:
        return tuple(self._decode_modality(modality, z) for modality in self.modalities)

    def forward(self, *inputs: torch.Tensor):  # noqa
        z = self.to_latent(*inputs)
        return (z, *self.decode(z))

    encode = to_latent
    decode = from_latent


# ---------------------------------------------------------------------------
# Mahalanobis & post-processing
# ---------------------------------------------------------------------------


def encode_modalities(
    model: ConvAutoencoder, inputs: tuple[np.ndarray, ...], batch: int = 256
) -> np.ndarray:
    model.eval()
    latent_dim = model.bottleneck.out_features
    return encode_batches(
        model.encode,
        inputs,
        device=DEVICE,
        batch_size=batch,
        latent_dim=latent_dim,
    )


def modality_inputs(
    batch: WindowBatch,
    modalities: tuple[str, ...],
    norms: dict[str, ChannelNorm],
) -> tuple[np.ndarray, ...]:
    arrays = {"vib": batch.vib, "torque": batch.torque, "us": batch.us}
    return tuple(
        (
            norms[modality].transform(arrays[modality])
            if arrays[modality].shape[0]
            else arrays[modality]
        )
        for modality in modalities
    )


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------


def evaluate_modalities(
    model,
    inputs: tuple[np.ndarray, ...],
    fnames,
    labels,
    runs,
    shaking,
    mu_z,
    cov_inv,
    mu_shk,
    cov_inv_shk,
    threshold,
    ema_alpha,
    persist_k,
    modalities: tuple[str, ...],
    export_plot: bool = True,
):
    z = encode_modalities(model, inputs)
    raw = two_component_scores(z, mu_z, cov_inv, mu_shk, cov_inv_shk)
    scores = ema_smooth(raw, fnames, ema_alpha)
    decisions = persistence(scores > threshold, fnames, persist_k)
    metrics = evaluation_metrics(
        labels, scores, decisions, runs, COUPLER1_RUNS, COUPLER2_RUNS
    )
    print_score_summary(
        modalities,
        ema_alpha,
        threshold,
        persist_k,
        metrics,
        labels,
        runs,
        shaking,
        decisions,
        COUPLER1_RUNS,
        COUPLER2_RUNS,
    )

    PLOT_DIR.mkdir(exist_ok=True)
    suffix = "_".join(modalities)
    out = PLOT_DIR / f"autoencoder_conv_scores_{suffix}.pdf"
    save_score_outputs(
        out,
        modalities=modalities,
        ema_alpha=ema_alpha,
        threshold=threshold,
        persistence_windows=persist_k,
        metrics=metrics,
        filenames=fnames,
        labels=labels,
        runs=runs,
        shaking=shaking,
        raw_scores=raw,
        scores=scores,
        decisions=decisions,
        coupler1_runs=COUPLER1_RUNS,
        coupler2_runs=COUPLER2_RUNS,
        export_plot=export_plot,
        score_upper=lambda values, t: max(float(np.percentile(values, 99.5)), t + 1.0),
        bins_count=201,
        titles=("Coupler 1", "Coupler 2", "Shaking"),
    )


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------


def save_modality_parameters(
    model: ConvAutoencoder,
    norms: dict[str, ChannelNorm],
    mu_z,
    cov_inv,
    mu_shk,
    cov_inv_shk,
    threshold,
    latent_dim,
    ema_alpha,
    persist_k,
    modalities: tuple[str, ...],
) -> None:
    stem = modality_model_parameter_stem(modalities)
    metadata = {
        "modalities": modalities,
        "norms": {
            modality: (norms[modality].mu, norms[modality].sd)
            for modality in modalities
        },
        "mu_z": mu_z,
        "cov_inv": cov_inv,
        "mu_shk": mu_shk,
        "cov_inv_shk": cov_inv_shk,
        "threshold": threshold,
        "latent_dim": latent_dim,
        "ema_alpha": ema_alpha,
        "persistence_windows": persist_k,
    }
    if "vib" in norms:
        metadata["norm_vib"] = (norms["vib"].mu, norms["vib"].sd)
    if "torque" in norms:
        metadata["norm_torque"] = (norms["torque"].mu, norms["torque"].sd)
    save_torch_parameter(MODELS_DIR, stem, model.state_dict(), metadata)
    print(f"Saved {stem} to {MODELS_DIR}/")


def load_modality_parameters(
    requested_modalities: tuple[str, ...] | None = None,
) -> tuple[ConvAutoencoder, dict, dict[str, ChannelNorm]]:
    modalities = requested_modalities or ("vib", "torque")
    stem = modality_model_parameter_stem(modalities)
    state, meta = load_torch_parameter(MODELS_DIR, stem, map_location=DEVICE)
    modalities = tuple(meta.get("modalities", modalities))
    model = ConvAutoencoder(meta["latent_dim"], modalities=modalities).to(DEVICE)
    model.load_state_dict(state)
    model.eval()
    norms = {}
    for modality, stats in meta.get("norms", {}).items():
        norm = ChannelNorm()
        norm.mu, norm.sd = stats
        norms[modality] = norm
    for modality in modalities:
        if modality in norms:
            continue
        stats = normalizer_stats(meta, modality)
        if stats is not None:
            norm = ChannelNorm()
            norm.mu, norm.sd = stats
            norms[modality] = norm
    return model, meta, norms


def load_recording_latent_features(
    items: list[tuple[Path, str, str, int]], batch: int = 256
) -> tuple[np.ndarray, list[tuple[Path, str, str, int]], dict]:
    """Load latent features for each recording. Returns features, available items, and metadata for the model and norms used."""
    modalities = ("vib", "torque")
    model, meta, norms = load_modality_parameters(modalities)
    stem = modality_model_parameter_stem(modalities)
    features: list[np.ndarray] = []
    available_items: list[tuple[Path, str, str, int]] = []
    window_counts: list[int] = []

    for item in items:
        path, *_ = item
        cached = build_or_load_cache(path)
        if cached is None:
            continue

        vib_n = norms["vib"].transform(cached["vib"])
        torque_n = norms["torque"].transform(cached["torque"])
        z = encode_modalities(model, (vib_n, torque_n), batch=batch)
        if z.shape[0] == 0:
            continue

        features.append(z.mean(axis=0))
        available_items.append(item)
        window_counts.append(int(z.shape[0]))

    if features:
        x = np.vstack(features)
    else:
        latent_dim = int(meta.get("latent_dim", 0))
        x = np.empty((0, latent_dim), dtype=np.float32)

    metadata = {
        "feature_source": "fusion_conv_autoencoder_latent",
        "model_path": str(MODELS_DIR / f"{stem}.pt"),
        "metadata_path": str(MODELS_DIR / f"{stem}_meta.pkl"),
        "aggregation": "mean_latent_over_windows",
        "window_counts": window_counts,
        "latent_dim": int(meta.get("latent_dim", x.shape[1] if x.ndim == 2 else 0)),
        "vibration_window": VIB_WINDOW,
        "torque_window": TORQUE_WINDOW,
    }
    return x, available_items, metadata


def split_train_val_records(records: list[dict]) -> tuple[list[dict], list[dict]]:
    """Split records into train and validation sets based on run membership and shaking status."""
    train_recs = [
        r
        for r in records
        if (
            is_healthy_split_member(
                r["filename"],
                r["label"],
                r["run"],
                r["is_shaking"],
                "train",
                TRAIN_HEALTHY_RUNS,
            )
        )
        or (r["is_shaking"] and "run1" in r["filename"])
    ]
    val_recs = [
        r
        for r in records
        if is_healthy_split_member(
            r["filename"],
            r["label"],
            r["run"],
            r["is_shaking"],
            "val",
            VAL_HEALTHY_RUNS,
        )
    ]
    return train_recs, val_recs


def train_autoencoder(
    args,
    modalities: tuple[str, ...],
    train_batch: WindowBatch,
    val_batch: WindowBatch,
) -> tuple[
    ConvAutoencoder,
    dict[str, ChannelNorm],
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    float,
    int,
]:
    """Load modalities from the training and validation batches, fit normalization, train the autoencoder, fit the two-component model, and calibrate the threshold."""
    raw_arrays = {
        "vib": train_batch.vib,
        "torque": train_batch.torque,
        "us": train_batch.us,
    }
    norms = {
        modality: ChannelNorm().fit(raw_arrays[modality]) for modality in modalities
    }
    train_inputs = modality_inputs(train_batch, modalities, norms)
    val_inputs = modality_inputs(val_batch, modalities, norms)

    model = ConvAutoencoder(args.latent_dim, modalities=modalities).to(DEVICE)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model: {n_params/1e3:.1f}k params, latent_dim={args.latent_dim}")
    train_autoencoder_model(
        model,
        train_inputs,
        val_inputs,
        device=DEVICE,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        patience=8,
        prefix=f"{'+'.join(modalities)} epoch",
        weight_decay=1e-5,
    )

    train_is_shk = train_batch.shaking.astype(bool)
    z_train = encode_modalities(model, train_inputs)
    z_healthy = z_train[~train_is_shk]
    z_shaking = z_train[train_is_shk]
    mu_z, cov_inv, mu_shk, cov_inv_shk = fit_two_component(
        z_healthy, z_shaking, LAMBDA_H, LAMBDA_SHK
    )
    if z_shaking.shape[0] < 2:
        print("WARNING: <2 shaking-run1 windows; falling back to a single component.")
    print(
        f"Component sizes - healthy: {z_healthy.shape[0]}  shaking: {z_shaking.shape[0]}"
    )

    if CALIB_WITH_SHAKING:
        calib_train_z = z_train
        calib_train_fn = train_batch.filenames
    else:
        calib_train_z = z_healthy
        calib_train_fn = train_batch.filenames[~train_is_shk]

    if val_inputs[0].shape[0]:
        z_val = encode_modalities(model, val_inputs)
        calib_z = np.concatenate([calib_train_z, z_val])
        calib_fn = np.concatenate([calib_train_fn, val_batch.filenames])
    else:
        calib_z, calib_fn = calib_train_z, calib_train_fn
    threshold = calibrate_two_component_threshold(
        calib_z,
        calib_fn,
        mu_z,
        cov_inv,
        mu_shk,
        cov_inv_shk,
        ema_alpha=args.ema_alpha,
        threshold_percentile=THRESHOLD_PERCENTILE,
    )
    persist_k = PERSISTENCE_WINDOWS
    print(
        f"Threshold ({THRESHOLD_PERCENTILE:.2f} pct of 2-component train scores): {threshold:.4f}"
    )
    save_modality_parameters(
        model,
        norms,
        mu_z,
        cov_inv,
        mu_shk,
        cov_inv_shk,
        threshold,
        args.latent_dim,
        args.ema_alpha,
        persist_k,
        modalities,
    )
    return model, norms, mu_z, cov_inv, mu_shk, cov_inv_shk, threshold, persist_k


def train_and_eval_autoencoder(args, modalities: tuple[str, ...]) -> None:
    """Train and evaluate the convolutional autoencoder for the specified modalities."""
    records = load_dataset()
    print(f"Loaded {len(records)} files")

    train_recs, val_recs = split_train_val_records(records)
    train_batch = stack_for_modalities(train_recs, modalities)
    val_batch = stack_for_modalities(val_recs, modalities)
    all_batch = stack_for_modalities(records, modalities)

    print(
        f"Train: {train_batch.filenames.shape[0]} windows ({len([r for r in train_recs if not r['is_shaking']])} healthy files + "
        f"{len([r for r in train_recs if r['is_shaking']])} shaking-run1 files)"
    )
    print(f"Val:   {val_batch.filenames.shape[0]} windows")
    print(f"Test:  {all_batch.filenames.shape[0]} windows (full set)")

    if args.eval_only:
        model, meta, norms = load_modality_parameters(modalities)
        modalities = tuple(meta.get("modalities", modalities))
        threshold = meta["threshold"]
        mu_z = meta["mu_z"]
        cov_inv = meta["cov_inv"]
        mu_shk = meta["mu_shk"]
        cov_inv_shk = meta["cov_inv_shk"]
        args.ema_alpha = meta.get("ema_alpha", args.ema_alpha)
        persist_k = meta.get("persistence_windows", PERSISTENCE_WINDOWS)
    else:
        (
            model,
            norms,
            mu_z,
            cov_inv,
            mu_shk,
            cov_inv_shk,
            threshold,
            persist_k,
        ) = train_autoencoder(args, modalities, train_batch, val_batch)

    all_inputs = modality_inputs(all_batch, modalities, norms)
    evaluate_modalities(
        model,
        all_inputs,
        all_batch.filenames,
        all_batch.labels,
        all_batch.runs,
        all_batch.shaking,
        mu_z,
        cov_inv,
        mu_shk,
        cov_inv_shk,
        threshold,
        args.ema_alpha,
        persist_k,
        modalities,
        args.export_plot,
    )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--latent-dim", type=int, default=LATENT_DIM_DEFAULT)
    p.add_argument("--epochs", type=int, default=60)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--ema-alpha", type=float, default=None)
    p.add_argument("--eval-only", action="store_true")
    p.add_argument(
        "--ultrasound",
        action="store_true",
        help="Train/evaluate the ultrasound-only convolutional autoencoder.",
    )
    p.add_argument(
        "--modalities",
        nargs="+",
        default=None,
        help="Modalities to use: vib/vibration, torque/current, us/ultrasound. Example: --modalities vib torque",
    )
    p.add_argument(
        "--no-export-plot",
        dest="export_plot",
        action="store_false",
        help="Do not export score or latent-space plot files.",
    )
    p.set_defaults(export_plot=True)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    modalities = cast(
        tuple[str, ...],
        tuple(
            parse_modalities_arg(
                args.modalities,
                MODALITY_ALIASES,
                default=("vib", "torque"),
                as_tuple=True,
            )
            or ("vib", "torque")
        ),
    )
    if args.ultrasound:
        modalities = ("us",)
    set_tuning_for_modalities(modalities)
    if args.ema_alpha is None:
        args.ema_alpha = EMA_ALPHA_DEFAULT

    print(f"Device: {DEVICE}")
    print(f"Modalities: {'+'.join(modalities)}")
    print("Loading recordings (caching raw windows) ...")
    train_and_eval_autoencoder(args, modalities)


if __name__ == "__main__":
    main()
