"""
Hand crafted feature autoencoder for shaft misalignment anomaly detection using configured tsfresh features.

Example usage:
  python hcf_autoencoder.py
  python hcf_autoencoder.py --latent-dim 4
  python hcf_autoencoder.py --modalities vib torque us
  python hcf_autoencoder.py --eval-only
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import cast

import matplotlib

matplotlib.use("Agg")
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from utils import (
    autoencoder_tuning_from_dict,
    calibrate_two_component_threshold,
    configure_matplotlib_plot_style,
    ema_smooth,
    encode_batches,
    evaluation_metrics,
    fit_two_component,
    healthy_file_split_configured,
    healthy_split_mask,
    healthy_training_pool_mask,
    load_config,
    load_modality_tuning,
    load_us_cache_features,
    modality_parameter_stem as make_modality_parameter_stem,
    parse_modalities_arg,
    persistence_filter as apply_persistence_filter,
    print_score_summary,
    load_torch_parameter,
    save_score_outputs,
    save_torch_parameter,
    train_autoencoder,
    two_component_scores,
)

RANDOM_STATE = 42
COUPLER1_RUNS = {1, 2, 3, 4}
COUPLER2_RUNS = {5, 6}
TEST_HEALTHY_RUNS = {4}

ROOT = Path(__file__).resolve().parent
TSFRESH_US_WINDOW_TAG = "vibw4096_torquew154_usw29487"
TSFRESH_WINDOW_TAG = TSFRESH_US_WINDOW_TAG
TSFRESH_VIB_TORQUE_WINDOW_TAG = "vibw4096_torquew154"
PARQUET = (
    ROOT
    / "tsfresh"
    / f"tsfresh_features_vib_torque_us_efficient_{TSFRESH_US_WINDOW_TAG}.parquet"
)
MODELS_DIR = ROOT / "models"
MANUAL_MODELS_DIR = ROOT / "models" / "manual"
PLOT_DIR = ROOT / "plots"
configure_matplotlib_plot_style()

torch.manual_seed(RANDOM_STATE)
np.random.seed(RANDOM_STATE)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if DEVICE.type == "cuda":
    torch.backends.cudnn.benchmark = True  # noqa
MODALITY_ALIASES = {
    "vib": "vibration",
    "vibration": "vibration",
    "torque": "torque",
    "current": "torque",
    "us": "ultrasound",
    "ultrasound": "ultrasound",
}

LEARNING_RATE = 1e-4
WEIGHT_DECAY = 1e-5
HEALTHY_TRAIN_RUNS = COUPLER1_RUNS | COUPLER2_RUNS
MANUAL_MODALITIES = ["vibration", "torque"]
MANUAL_MODALITIES_WITH_ULTRASOUND = ["vibration", "torque", "ultrasound"]
MANUAL_FEATURE_DIMS = {
    "vibration": 44,
    "torque": 11,
    "ultrasound": 11,
}

DEFAULT_TUNING = load_modality_tuning(
    "autoencoder",
    MANUAL_MODALITIES,
    {},
)
TUNING_DEFAULTS = DEFAULT_TUNING
DEFAULT_TUNING_STATE = autoencoder_tuning_from_dict(DEFAULT_TUNING)
THRESHOLD_PERCENTILE = DEFAULT_TUNING_STATE.threshold_pct
EMA_ALPHA_DEFAULT = DEFAULT_TUNING_STATE.ema_alpha
PERSISTENCE_WINDOWS = DEFAULT_TUNING_STATE.persistence_windows
LAMBDA_H = DEFAULT_TUNING_STATE.lambda_h
LAMBDA_SHK = DEFAULT_TUNING_STATE.lambda_shk
CALIB_WITH_SHAKING = DEFAULT_TUNING_STATE.calibrate_with_shaking


def set_tuning_for_modalities(modalities: list[str]) -> dict[str, float | bool]:
    global THRESHOLD_PERCENTILE, EMA_ALPHA_DEFAULT, PERSISTENCE_WINDOWS, LAMBDA_H, LAMBDA_SHK, CALIB_WITH_SHAKING

    tuning = load_modality_tuning("autoencoder", modalities, TUNING_DEFAULTS)
    state = autoencoder_tuning_from_dict(tuning)
    THRESHOLD_PERCENTILE = state.threshold_pct
    EMA_ALPHA_DEFAULT = state.ema_alpha
    PERSISTENCE_WINDOWS = state.persistence_windows
    LAMBDA_H = state.lambda_h
    LAMBDA_SHK = state.lambda_shk
    CALIB_WITH_SHAKING = state.calibrate_with_shaking
    return tuning


def manual_feature_slices(
    modalities: list[str],
    feature_dims: dict[str, int] | None = None,
) -> dict[str, slice]:
    """Return a dict mapping modality to slice object for indexing into the combined feature vector."""
    feature_dims = feature_dims or MANUAL_FEATURE_DIMS
    slices: dict[str, slice] = {}
    start = 0
    for modality in modalities:
        width = feature_dims[modality]
        slices[modality] = slice(start, start + width)
        start += width
    return slices


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------


class HCFAutoencoder(nn.Module):
    """Hand-crafted feature autoencoder, selects tsfresh features from config.yaml. Fits two-component Mahalanobis distance on the latent space for anomaly scoring."""

    def __init__(
        self,
        input_dim: int,
        latent_dim: int = 6,
        modality_dims: dict[str, int] | None = None,
    ) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.modality_dims = dict(modality_dims or {})
        self.modality_slices = manual_feature_slices(
            list(self.modality_dims), self.modality_dims
        )

        if len(self.modality_dims) > 1:
            self.branch_encoders = nn.ModuleDict()
            branch_widths: list[int] = []
            for modality, width in self.modality_dims.items():
                hidden_dim = max(16, min(64, width * 2))
                embed_dim = max(8, min(32, width * 2))
                self.branch_encoders[modality] = nn.Sequential(
                    nn.Linear(width, hidden_dim),
                    nn.LayerNorm(hidden_dim),
                    nn.ReLU(),
                    nn.Linear(hidden_dim, embed_dim),
                )
                branch_widths.append(embed_dim)
            self.encoder = nn.Sequential(
                nn.Linear(sum(branch_widths), latent_dim), # Concatenate modalities
            )
        else:
            self.branch_encoders = None
            hidden_dim = max(16, min(64, input_dim * 2))
            self.encoder = nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, latent_dim),
            )

        hidden_dim = max(16, min(64, input_dim * 2))
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim),
        )

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        if self.branch_encoders is not None:
            parts = [
                self.branch_encoders[modality](x[:, self.modality_slices[modality]])
                for modality in self.modality_dims
            ]
            x = torch.cat(parts, dim=1)
        return self.encoder(x)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:  # noqa
        z = self.encode(x)
        return z, self.decoder(z)


def encode_feature_all(
    model: HCFAutoencoder,
    x: np.ndarray,
    batch_size: int = 256,
) -> np.ndarray:
    """Encode the input features using the trained autoencoder model in batches."""
    model.eval()
    final_layer = cast(nn.Linear, model.encoder[-1])
    latent_dim = final_layer.out_features
    return encode_batches(
        model.encode,
        (x,),
        device=DEVICE,
        batch_size=batch_size,
        latent_dim=latent_dim,
    )


# ---------------------------------------------------------------------------
# Data helpers
# ---------------------------------------------------------------------------


def configured_modality_features(
    cfg: dict, modalities: list[str]
) -> dict[str, list[str]]:
    """Return the configured tsfresh features to use for each selected modality."""
    configured = cfg.get("autoencoder_features")
    if not isinstance(configured, dict):
        raise ValueError("config.yaml must define autoencoder_features.")

    feature_map: dict[str, list[str]] = {}
    missing: list[str] = []
    for modality in modalities:
        features = configured.get(modality)
        if not features:
            missing.append(modality)
            continue
        feature_map[modality] = list(features)

    if missing:
        raise ValueError(
            "config.yaml autoencoder_features is missing entries for: "
            + ", ".join(missing)
        )
    return feature_map


def load_selected_tsfresh_data(feature_map: dict[str, list[str]]) -> pd.DataFrame:
    """Load selected tsfresh features and supplement missing cache/ultrasound rows."""
    feature_cols = [col for cols in feature_map.values() for col in cols]
    if not feature_cols:
        raise ValueError(
            "No configured tsfresh features found for selected modalities."
        )

    cfg = load_config()
    configured = configured_modality_features(cfg, MANUAL_MODALITIES)
    vib_features = list(
        dict.fromkeys(
            configured["vibration"] + list(feature_map.get("vibration", []))
        )
    )
    torque_features = list(
        dict.fromkeys(configured["torque"] + list(feature_map.get("torque", [])))
    )
    us_features = list(feature_map.get("ultrasound", []))

    parquet_features = list(dict.fromkeys(vib_features + torque_features))
    df = pd.read_parquet(
        PARQUET, columns=parquet_features + ["filename", "run", "label"]
    )
    df = normalize_run_from_filename(df)
    df = df.dropna(subset=parquet_features)
    extra_df = load_extra_cache_data(df, vib_features, torque_features)
    df = pd.concat([df, extra_df], ignore_index=False)

    if us_features:
        us_df = pd.read_parquet(
            ROOT
            / "tsfresh"
            / f"tsfresh_features_vib_torque_us_efficient_{TSFRESH_US_WINDOW_TAG}_selected.parquet",
            columns=us_features,
        )
        us_cache_df = load_us_cache_features(us_features, TSFRESH_US_WINDOW_TAG, ROOT)
        us_df = pd.concat([us_df, us_cache_df[~us_cache_df.index.isin(us_df.index)]])
        df = df.join(us_df.loc[:, us_features], how="left")
        missing_us = int(df.loc[:, us_features].isna().any(axis=1).sum())
        if missing_us:
            print(
                f"Warning: imputing {missing_us} windows without configured ultrasound features."
            )

    return df


def normalize_run_from_filename(df: pd.DataFrame) -> pd.DataFrame:
    if "filename" not in df.columns:
        return df
    out = df.copy()
    parsed = out["filename"].astype(str).str.extract(r"run(?P<run>\d+)")["run"]
    mask = parsed.notna()
    if mask.any():
        out.loc[mask, "run"] = parsed[mask].astype(int).to_numpy()
    return out


def load_cache_stems(
    stems: list[str], vib_features: list[str], torque_features: list[str]
) -> pd.DataFrame:
    """Load specific cache stems by name and return a combined DataFrame."""
    cache = ROOT / "tsfresh" / "cache" / TSFRESH_VIB_TORQUE_WINDOW_TAG
    torque_dir = cache / "torque"
    frames = []
    for stem in stems:
        vib_feat_path = cache / "vib" / f"{stem}_feat.parquet"
        torque_feat_path = torque_dir / f"{stem}_feat.parquet"
        meta_path = cache / "vib" / f"{stem}_meta.parquet"
        if not (
            vib_feat_path.exists() and torque_feat_path.exists() and meta_path.exists()
        ):
            print(f"Skipping incomplete cache stem: {stem}")
            continue
        vib_feat = pd.read_parquet(vib_feat_path)
        torque_feat = pd.read_parquet(torque_feat_path)
        meta = pd.read_parquet(meta_path).set_index("id")
        combined = vib_feat.join(
            torque_feat, how="inner", lsuffix="", rsuffix="_torque"
        )
        combined = combined.join(meta[["label", "run", "filename"]])
        frames.append(combined)
    if not frames:
        return pd.DataFrame(
            columns=pd.Index(
                vib_features + torque_features + ["run", "label", "filename"]
            )
        )
    df = pd.concat(frames)
    missing = [c for c in vib_features + torque_features if c not in df.columns]
    if missing:
        raise KeyError(f"Cache missing features: {missing[:5]} ...")
    selected = cast(
        pd.DataFrame,
        df.loc[:, vib_features + torque_features + ["run", "label", "filename"]],
    )
    return normalize_run_from_filename(selected)


def load_extra_cache_data(
    df_parquet: pd.DataFrame, vib_features: list[str], torque_features: list[str]
) -> pd.DataFrame:
    """Load any cache stems not already present in the main parquet."""
    cache = ROOT / "tsfresh" / "cache" / TSFRESH_VIB_TORQUE_WINDOW_TAG
    known = (
        set(df_parquet["filename"].unique())
        if "filename" in df_parquet.columns
        else set()
    )
    torque_dir = cache / "torque"
    torque_stems = {
        p.stem.removesuffix("_feat") for p in torque_dir.glob("*_feat.parquet")
    }
    stems = sorted(
        {
            p.stem.removesuffix("_feat")
            for p in (cache / "vib").glob("*_feat.parquet")
            if p.stem.removesuffix("_feat") not in known
            and p.stem.removesuffix("_feat") in torque_stems
        }
    )
    if not stems:
        return pd.DataFrame(
            columns=pd.Index(
                vib_features + torque_features + ["run", "label", "filename"]
            )
        )
    return load_cache_stems(stems, vib_features, torque_features)


def split_healthy_training_data(
    healthy_df: pd.DataFrame, val_fraction: float = 0.15
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split healthy files into train/val subsets without mixing windows from one file."""
    if healthy_file_split_configured("train") or healthy_file_split_configured("val"):
        shaking = (
            healthy_df["filename"]
            .astype(str)
            .str.contains("shaking", case=False, na=False)
        )
        train_mask = healthy_split_mask(
            healthy_df["filename"],
            healthy_df["label"],
            healthy_df["run"],
            shaking,
            "train",
            HEALTHY_TRAIN_RUNS,
        )
        val_mask = healthy_split_mask(
            healthy_df["filename"],
            healthy_df["label"],
            healthy_df["run"],
            shaking,
            "val",
            set(),
        )
        return healthy_df.loc[train_mask].copy(), healthy_df.loc[val_mask].copy()

    filenames = np.array(sorted(healthy_df["filename"].dropna().unique()))
    rng = np.random.default_rng(RANDOM_STATE)
    shuffled = filenames.copy()
    rng.shuffle(shuffled)
    n_val = max(1, int(round(len(shuffled) * val_fraction)))
    val_files = shuffled[:n_val].tolist()
    train_mask = ~healthy_df["filename"].isin(val_files)
    train_df = healthy_df.loc[train_mask].copy()
    val_df = healthy_df.loc[~train_mask].copy()
    return train_df, val_df


def select_test_evaluation_data(df: pd.DataFrame) -> pd.DataFrame:
    """Return rows that belong in the held-out reporting plot."""
    shaking = df["filename"].astype(str).str.contains("shaking", case=False, na=False)
    test_healthy = healthy_split_mask(
        df["filename"],
        df["label"],
        df["run"],
        shaking,
        "test",
        TEST_HEALTHY_RUNS,
    )
    anomalies = df["label"] != "healthy"
    return df.loc[test_healthy | anomalies].copy()


# ---------------------------------------------------------------------------
# Mahalanobis calibration
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------


def save_selected_modalities(
    model: HCFAutoencoder,
    scaler: StandardScaler,
    imputer: SimpleImputer,
    mu_z,
    cov_inv,
    mu_shk,
    cov_inv_shk,
    threshold,
    latent_dim,
    ema_alpha,
    persistence_windows,
    modalities: list[str],
    feature_map: dict[str, list[str]],
) -> None:
    """Save model weights and evaluation parameters with a modality-specific stem."""
    stem = make_modality_parameter_stem(modalities, "fusion_ae")
    save_torch_parameter(
        MANUAL_MODELS_DIR,
        stem,
        model.state_dict(),
        {
            "modalities": modalities,
            "feature_map": feature_map,
            "features": [col for cols in feature_map.values() for col in cols],
            "modality_dims": {
                modality: len(cols) for modality, cols in feature_map.items()
            },
            "scaler": scaler,
            "imputer": imputer,
            "mu_z": mu_z,
            "cov_inv": cov_inv,
            "mu_shk": mu_shk,
            "cov_inv_shk": cov_inv_shk,
            "threshold": threshold,
            "latent_dim": latent_dim,
            "ema_alpha": ema_alpha,
            "persistence_windows": persistence_windows,
        },
    )
    print(f"Saved model to {MANUAL_MODELS_DIR / stem}")


def load_selected_modalities(modalities: list[str]) -> tuple[HCFAutoencoder, dict]:
    """Load the model matching the selected modalities."""
    stem = make_modality_parameter_stem(modalities, "fusion_ae")
    meta_path = MANUAL_MODELS_DIR / f"{stem}_meta.pkl"
    model_path = MANUAL_MODELS_DIR / f"{stem}.pt"
    state, meta = load_torch_parameter(
        meta_path.parent, model_path.stem, map_location=DEVICE
    )
    if "feature_map" not in meta:
        features = list(meta.get("features") or meta.get("case_features") or [])
        if len(modalities) == 1 and features:
            meta["feature_map"] = {modalities[0]: features}
            meta["features"] = features
    model = HCFAutoencoder(
        len(meta["features"]),
        meta["latent_dim"],
        meta.get("modality_dims"),
    ).to(DEVICE)
    model.load_state_dict(state)
    model.eval()
    return model, meta


@dataclass
class FeatureRunParameters:
    model: HCFAutoencoder
    imputer: SimpleImputer
    scaler: StandardScaler
    mu_z: np.ndarray
    cov_inv: np.ndarray
    mu_shk: np.ndarray
    cov_inv_shk: np.ndarray
    threshold: float
    persistence_windows: int
    features: list[str]
    feature_map: dict[str, list[str]]


def load_training_dataframe(
    feature_map: dict[str, list[str]],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load the main parquet with only the configured features for the selected modalities, and split healthy training data into train/val subsets."""
    df = load_selected_tsfresh_data(feature_map)
    shaking = df["filename"].astype(str).str.contains("shaking", case=False, na=False)
    healthy_mask = healthy_training_pool_mask(
        df["filename"], df["label"], df["run"], shaking, HEALTHY_TRAIN_RUNS
    )
    healthy_df = df.loc[healthy_mask].copy()
    train_df, val_df = split_healthy_training_data(healthy_df)
    return df, train_df, val_df


def fit_feature_preprocessors(
    train_df: pd.DataFrame, val_df: pd.DataFrame, features: list[str]
) -> tuple[SimpleImputer, StandardScaler, np.ndarray, np.ndarray]:
    """Fit imputer and scaler on training data and return transformed train and val arrays."""
    imputer = SimpleImputer(strategy="mean").fit(
        train_df.loc[:, features].to_numpy(dtype=np.float32)
    )
    train_x = np.asarray(
        imputer.transform(train_df.loc[:, features].to_numpy(dtype=np.float32)),
        dtype=np.float32,
    )
    val_x = np.asarray(
        imputer.transform(val_df.loc[:, features].to_numpy(dtype=np.float32)),
        dtype=np.float32,
    )
    scaler = StandardScaler().fit(train_x)
    return (
        imputer,
        scaler,
        np.asarray(scaler.transform(train_x), dtype=np.float32),
        np.asarray(scaler.transform(val_x), dtype=np.float32),
    )


def load_shaking_features(
    cfg: dict,
    features: list[str],
    imputer: SimpleImputer,
    scaler: StandardScaler,
) -> tuple[np.ndarray, np.ndarray]:
    """Load shaking run1 features from cache, preprocess with imputer and scaler, and return with corresponding filenames."""
    shaking_run1_stems = [
        Path(f).stem
        for f in cfg.get("npz_selection", {}).get("included_files", [])
        if "run1" in f
    ]
    configured = configured_modality_features(cfg, MANUAL_MODALITIES)
    base_vib_features = configured["vibration"]
    base_torque_features = configured["torque"]
    shaking_df = load_cache_stems(
        shaking_run1_stems, base_vib_features, base_torque_features
    )
    shaking_df = shaking_df.dropna(subset=base_vib_features + base_torque_features)
    feature_df = shaking_df.reindex(columns=features).dropna()
    if feature_df.empty:
        return np.empty((0, len(features)), dtype=np.float32), np.array([])
    x = np.asarray(
        scaler.transform(
            np.asarray(
                imputer.transform(feature_df.to_numpy(dtype=np.float32)),
                dtype=np.float32,
            )
        ),
        dtype=np.float32,
    )
    return x, shaking_df.loc[feature_df.index, "filename"].to_numpy()


def fit_selected_modalities(
    args,
    cfg: dict,
    modalities: list[str],
    feature_map: dict[str, list[str]],
    features: list[str],
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
) -> FeatureRunParameters:
    """Fit the feature autoencoder and Mahalanobis parameters for the selected modalities, and return all necessary components for evaluation."""
    imputer, scaler, train_x, val_x = fit_feature_preprocessors(
        train_df, val_df, features
    )
    model = HCFAutoencoder(
        train_x.shape[1],
        args.latent_dim,
        {modality: len(cols) for modality, cols in feature_map.items()},
    ).to(DEVICE)
    print(
        f"Training FeatureAE (modalities={'+'.join(modalities)}, "
        f"latent_dim={args.latent_dim}, device={DEVICE}) ..."
    )
    train_autoencoder(
        model,
        (train_x,),
        (val_x,),
        device=DEVICE,
        epochs=300,
        batch_size=256,
        lr=LEARNING_RATE,
        patience=20,
        prefix="Epoch",
        weight_decay=WEIGHT_DECAY,
    )

    z_train = encode_feature_all(model, train_x)
    shaking_x, shaking_filenames = load_shaking_features(
        cfg, features, imputer, scaler
    )
    z_shaking = (
        encode_feature_all(model, shaking_x)
        if shaking_x.shape[0]
        else np.empty((0, z_train.shape[1]), dtype=z_train.dtype)
    )
    mu_z, cov_inv, mu_shk, cov_inv_shk = fit_two_component(
        z_train, z_shaking, LAMBDA_H, LAMBDA_SHK
    )

    calib_z = z_train
    calib_filenames = train_df["filename"].to_numpy()
    if CALIB_WITH_SHAKING and z_shaking.shape[0]:
        calib_z = np.concatenate([calib_z, z_shaking])
        calib_filenames = np.concatenate([calib_filenames, shaking_filenames])
    threshold = calibrate_two_component_threshold(
        calib_z,
        calib_filenames,
        mu_z,
        cov_inv,
        mu_shk,
        cov_inv_shk,
        ema_alpha=args.ema_alpha,
        threshold_percentile=THRESHOLD_PERCENTILE,
    )
    persistence_windows = PERSISTENCE_WINDOWS

    save_selected_modalities(
        model,
        scaler,
        imputer,
        mu_z,
        cov_inv,
        mu_shk,
        cov_inv_shk,
        threshold,
        args.latent_dim,
        args.ema_alpha,
        persistence_windows,
        modalities,
        feature_map,
    )
    return FeatureRunParameters(
        model,
        imputer,
        scaler,
        mu_z,
        cov_inv,
        mu_shk,
        cov_inv_shk,
        threshold,
        persistence_windows,
        features,
        feature_map,
    )


def load_selected_run_parameters(
    args,
    modalities: list[str],
) -> FeatureRunParameters:
    """Load the model and evaluation parameters for the selected modalities from disk."""
    model, meta = load_selected_modalities(modalities)
    args.ema_alpha = meta.get("ema_alpha", args.ema_alpha)
    return FeatureRunParameters(
        model=model,
        imputer=meta["imputer"],
        scaler=meta["scaler"],
        mu_z=meta["mu_z"],
        cov_inv=meta["cov_inv"],
        mu_shk=meta.get("mu_shk", meta["mu_z"]),
        cov_inv_shk=meta.get("cov_inv_shk", meta["cov_inv"]),
        threshold=meta["threshold"],
        persistence_windows=meta.get("persistence_windows", PERSISTENCE_WINDOWS),
        features=meta["features"],
        feature_map=meta["feature_map"],
    )


def evaluate_selected_modalities(
    model: HCFAutoencoder,
    df_test: pd.DataFrame,
    features: list[str],
    imputer: SimpleImputer,
    scaler: StandardScaler,
    mu_z,
    cov_inv,
    mu_shk,
    cov_inv_shk,
    threshold: float,
    ema_alpha: float,
    persistence_windows: int,
    modalities: list[str],
    export_plot: bool = True,
) -> None:
    """Evaluate the selected modalities on the test set and print/save the results."""
    x = np.asarray(
        imputer.transform(df_test.loc[:, features].to_numpy(dtype=np.float32)),
        dtype=np.float32,
    )
    x = np.asarray(scaler.transform(x), dtype=np.float32)
    z_all = encode_feature_all(model, x)
    raw_scores = two_component_scores(z_all, mu_z, cov_inv, mu_shk, cov_inv_shk)
    filenames = df_test["filename"].fillna("unknown").to_numpy()
    scores = ema_smooth(raw_scores, filenames, ema_alpha)
    decisions = apply_persistence_filter(
        scores > threshold, filenames, persistence_windows
    )

    labels = df_test["label"].to_numpy()
    runs = df_test["run"].to_numpy()
    metrics = evaluation_metrics(
        labels, scores, decisions, runs, COUPLER1_RUNS, COUPLER2_RUNS
    )

    shaking_mask = np.array([isinstance(f, str) and "shaking" in f for f in filenames])
    print_score_summary(
        modalities,
        ema_alpha,
        threshold,
        persistence_windows,
        metrics,
        labels,
        runs,
        shaking_mask,
        decisions,
        COUPLER1_RUNS,
        COUPLER2_RUNS,
    )

    PLOT_DIR.mkdir(exist_ok=True)
    out = PLOT_DIR / f"autoencoder_scores_{'_'.join(modalities)}.pdf"
    save_score_outputs(
        out,
        modalities=modalities,
        ema_alpha=ema_alpha,
        threshold=threshold,
        persistence_windows=persistence_windows,
        metrics=metrics,
        filenames=filenames,
        labels=labels,
        runs=runs,
        shaking=shaking_mask,
        raw_scores=raw_scores,
        scores=scores,
        decisions=decisions,
        coupler1_runs=COUPLER1_RUNS,
        coupler2_runs=COUPLER2_RUNS,
        export_plot=export_plot,
        score_upper=lambda values, t: max(
            float(np.percentile(values, 99.5)), t * 1.5, 1.0
        ),
        bins_count=151,
        titles=(
            "Coupler 1 (runs 1-4)",
            "Coupler 2 (runs 5-6)",
            "Shaking recordings" if shaking_mask.any() else "Held-out healthy test",
        ),
    )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments for the feature autoencoder script."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--latent-dim", type=int, default=6)
    parser.add_argument("--ema-alpha", type=float, default=None)
    parser.add_argument("--eval-only", action="store_true")
    parser.add_argument(
        "--no-export-plot",
        dest="export_plot",
        action="store_false",
        help="Do not export the autoencoder score PDF.",
    )
    parser.set_defaults(export_plot=True)
    parser.add_argument(
        "--modalities",
        nargs="+",
        default=None,
        help="Modalities to use: vib/vibration, torque/current, us/ultrasound. Example: --modalities vib torque",
    )
    parser.add_argument(
        "--max-windows-per-recording",
        type=int,
        default=0,
        help=argparse.SUPPRESS,
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    modalities = list(
        parse_modalities_arg(args.modalities, MODALITY_ALIASES) or MANUAL_MODALITIES
    )

    set_tuning_for_modalities(modalities)
    if args.ema_alpha is None:
        args.ema_alpha = EMA_ALPHA_DEFAULT

    cfg = load_config()
    feature_map = configured_modality_features(cfg, modalities)
    features = [col for cols in feature_map.values() for col in cols]
    print(
        "Loading selected tsfresh features: "
        + ", ".join(
            f"{modality}={len(feature_map[modality])}" for modality in modalities
        )
    )

    parameters: FeatureRunParameters | None = None
    if args.eval_only:
        parameters = load_selected_run_parameters(args, modalities)
        feature_map = parameters.feature_map
        features = parameters.features

    df, train_df, val_df = load_training_dataframe(feature_map)
    print(f"Train: {len(train_df)}  Val: {len(val_df)}  Test: {len(df)}")

    if parameters is None:
        parameters = fit_selected_modalities(
            args,
            cfg,
            modalities,
            feature_map,
            features,
            train_df,
            val_df,
        )

    eval_df = select_test_evaluation_data(df)
    print(f"Plot/eval rows (held-out test only): {len(eval_df)}")

    evaluate_selected_modalities(
        parameters.model,
        eval_df,
        parameters.features,
        parameters.imputer,
        parameters.scaler,
        parameters.mu_z,
        parameters.cov_inv,
        parameters.mu_shk,
        parameters.cov_inv_shk,
        parameters.threshold,
        args.ema_alpha,
        parameters.persistence_windows,
        modalities,
        args.export_plot,
    )


if __name__ == "__main__":
    main()
