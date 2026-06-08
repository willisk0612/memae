"""
Extract features from recordings using tsfresh, with both time and frequency domain features. The output is a parquet file with one row per window and columns for each feature, along with metadata columns for label, load, run, and filename.

Example usage:
    python feature_extraction.py --fc-params efficient --fdr-level 0.05 --modalities vibration torque ultrasound
"""


from __future__ import annotations

import argparse
from pathlib import Path
from typing import cast

import time

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from tsfresh import extract_features, select_features
from tsfresh.feature_extraction import (
    ComprehensiveFCParameters,
    EfficientFCParameters,
    MinimalFCParameters,
)
from tsfresh.utilities.dataframe_functions import impute

from utils import load_capture_data, load_recording_config

ROOT = Path(__file__).resolve().parent
TSFRESH_DIR = ROOT / "tsfresh"
DEFAULT_WINDOW_SIZES = {
    "vibration": 4096,
    "torque": 154,
    "ultrasound": 29487,
}
DEFAULT_FDR_LEVEL = 0.05

FC_PARAMS_MAP = {
    "minimal": MinimalFCParameters,
    "efficient": EfficientFCParameters,
    "comprehensive": ComprehensiveFCParameters,
}
MODALITY_ORDER = ["vibration", "torque", "ultrasound"]
MODALITY_ALIASES = {"sensorkit": "vibration"}
MODALITY_TAGS = {"vibration": "vib", "torque": "torque", "ultrasound": "us"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract tsfresh features from recordings."
    )
    parser.add_argument(
        "--window-size",
        type=int,
        default=None,
        help="Override all modality window sizes in samples. Use 0 for whole-recording mode.",
    )
    parser.add_argument(
        "--vibration-window-size",
        type=int,
        default=DEFAULT_WINDOW_SIZES["vibration"],
        help="Vibration samples per window (default: %(default)s).",
    )
    parser.add_argument(
        "--torque-window-size",
        type=int,
        default=DEFAULT_WINDOW_SIZES["torque"],
        help="Torque samples per window (default: %(default)s).",
    )
    parser.add_argument(
        "--ultrasound-window-size",
        type=int,
        default=DEFAULT_WINDOW_SIZES["ultrasound"],
        help="Ultrasound samples per window (default: %(default)s).",
    )
    parser.add_argument(
        "--fc-params",
        choices=list(FC_PARAMS_MAP.keys()),
        default="efficient",
        help="tsfresh feature calculator set (default: %(default)s).",
    )
    parser.add_argument(
        "--fdr-level",
        type=float,
        default=DEFAULT_FDR_LEVEL,
        help="FDR threshold for feature selection (default: %(default)s).",
    )
    parser.add_argument(
        "--modalities",
        nargs="+",
        default=["vibration", "torque"],
        choices=MODALITY_ORDER + list(MODALITY_ALIASES.keys()),
        help="Sensor modalities to include (default: vibration).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=TSFRESH_DIR,
        help="Directory to write output parquet files (default: tsfresh/).",
    )
    parser.add_argument(
        "--skip-selection",
        action="store_true",
        help="Skip tsfresh feature selection step.",
    )
    parser.add_argument(
        "--n-jobs",
        type=int,
        default=4,
        help="Parallel jobs for tsfresh (default: %(default)s).",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-extract all recordings, ignoring cached results.",
    )
    return parser.parse_args()


def modality_tag(modalities: list[str]) -> str:
    ordered = [name for name in MODALITY_ORDER if name in set(modalities)]
    return "_".join(MODALITY_TAGS[name] for name in ordered)


def normalize_modalities(modalities: list[str]) -> list[str]:
    normalized = [MODALITY_ALIASES.get(name, name) for name in modalities]
    return [name for name in MODALITY_ORDER if name in set(normalized)]


def resolve_window_sizes(args: argparse.Namespace) -> dict[str, int]:
    if args.window_size is not None:
        return {modality: args.window_size for modality in MODALITY_ORDER}
    return {
        "vibration": args.vibration_window_size,
        "torque": args.torque_window_size,
        "ultrasound": args.ultrasound_window_size,
    }


def window_tag(modalities: list[str], window_sizes: dict[str, int]) -> str:
    sizes = [window_sizes[modality] for modality in modalities]
    if len(set(sizes)) == 1:
        return f"w{sizes[0]}"
    return "_".join(f"{MODALITY_TAGS[modality]}w{window_sizes[modality]}" for modality in modalities)


def _build_recording_ts_df(
    capture: dict,
    stem: str,
    label: str,
    load_str: str,
    run: int,
    window_sizes: dict[str, int],
    modalities: list[str],
) -> tuple[pd.DataFrame, list[dict]]:
    """Build the long-format tsfresh DataFrame for a single recording.

    Returns (ts_df, meta_rows) where ts_df has columns [id, time, kind, value]
    and meta_rows is a list of {id, label, load, run, filename} dicts.
    """
    chunk_dfs: list[pd.DataFrame] = []
    meta_rows: list[dict] = []

    if "vibration" in modalities:
        sk: np.ndarray = capture["sk"]
        sk_t: np.ndarray = capture["sk_t"]
        if sk.shape[0] > 0 and sk.shape[1] >= 4:
            _append_signal_windows(
                signals=sk[:, 1:4],
                times=sk_t,
                kind_names=["ax", "ay", "az"],
                stem=stem,
                label=label,
                load_str=load_str,
                run=run,
                window_size=window_sizes["vibration"],
                chunk_dfs=chunk_dfs,
                meta_rows=meta_rows,
            )

    if "torque" in modalities:
        plc: np.ndarray = capture["plc"]
        plc_t: np.ndarray = capture["plc_t"]
        if plc.shape[0] > 0 and plc.shape[1] >= 2:
            _append_signal_windows(
                signals=plc[:, -1:],
                times=plc_t,
                kind_names=["torque"],
                stem=stem,
                label=label,
                load_str=load_str,
                run=run,
                window_size=window_sizes["torque"],
                chunk_dfs=chunk_dfs,
                meta_rows=meta_rows,
            )

    if "ultrasound" in modalities:
        us: np.ndarray = capture["us"]
        us_t: np.ndarray = capture["us_t"]
        if us.shape[0] > 0 and us.shape[1] >= 2:
            _append_signal_windows(
                signals=us[:, 1:2],
                times=us_t,
                kind_names=["us"],
                stem=stem,
                label=label,
                load_str=load_str,
                run=run,
                window_size=window_sizes["ultrasound"],
                chunk_dfs=chunk_dfs,
                meta_rows=meta_rows,
            )

    if not chunk_dfs:
        return pd.DataFrame(), []

    ts_df = pd.concat(chunk_dfs, ignore_index=True)
    return ts_df, meta_rows


def _append_signal_windows(
    signals: np.ndarray,
    times: np.ndarray,
    kind_names: list[str],
    stem: str,
    label: str,
    load_str: str,
    run: int,
    window_size: int,
    chunk_dfs: list[pd.DataFrame],
    meta_rows: list[dict],
) -> None:
    # Filter non-finite rows
    valid = np.isfinite(times)
    for col in range(signals.shape[1]):
        valid &= np.isfinite(signals[:, col])
    signals = signals[valid]
    times = times[valid]

    n = signals.shape[0]
    if n == 0:
        return

    effective_window = window_size if window_size > 0 else n
    starts = list(range(0, n - effective_window + 1, effective_window))
    if not starts:
        starts = [0]
        effective_window = n

    n_kinds = len(kind_names)
    for w_idx, start in enumerate(starts):
        end = min(start + effective_window, n)
        n_pts = end - start
        t_rel = times[start:end] - times[start]
        win_id = f"{stem}__w{w_idx:04d}"

        ids = np.repeat(win_id, n_pts * n_kinds)
        time_col = np.tile(t_rel, n_kinds)
        kind_col = np.repeat(kind_names, n_pts)
        value_col = np.concatenate([signals[start:end, k] for k in range(n_kinds)])

        chunk_dfs.append(
            pd.DataFrame({"id": ids, "time": time_col, "kind": kind_col, "value": value_col})
        )
        meta_rows.append(
            {"id": win_id, "label": label, "load": load_str, "run": run, "filename": stem}
        )


def main() -> None:
    args = parse_args()
    modalities = normalize_modalities(args.modalities)
    window_sizes = resolve_window_sizes(args)
    tag = f"{modality_tag(modalities)}_{args.fc_params}_{window_tag(modalities, window_sizes)}"
    fc_params = FC_PARAMS_MAP[args.fc_params]()

    print("Loading recording config...")
    items = load_recording_config()
    labels_found = sorted({label for _, label, _, _ in items})
    print(f"Found {len(items)} recordings: {labels_found}")

    cache_base = args.output_dir / "cache" / window_tag(modalities, window_sizes)
    cache_base.mkdir(parents=True, exist_ok=True)

    all_feat_dfs: list[pd.DataFrame] = []
    all_meta_rows: list[dict] = []
    n_cached = 0

    for i, (path, label, config_name, run) in enumerate(items):
        prefix = f"  [{i + 1}/{len(items)}] {path.name}"
        load_str = config_name.replace("p", ".").removesuffix("nm")

        # Check per-modality caches; collect which modalities still need extraction.
        modality_feat_dfs: list[pd.DataFrame] = []
        missing_modalities: list[str] = []
        cached_meta_rows: list[dict] | None = None

        for mod in modalities:
            mod_tag = MODALITY_TAGS[mod]
            mod_cache_dir = cache_base / mod_tag
            cache_feat = mod_cache_dir / f"{path.stem}_feat.parquet"
            cache_meta = mod_cache_dir / f"{path.stem}_meta.parquet"
            if not args.force and cache_feat.exists() and cache_meta.exists():
                modality_feat_dfs.append(pd.read_parquet(cache_feat))
                if cached_meta_rows is None:
                    cached_meta_rows = pd.read_parquet(cache_meta).to_dict("records")
            else:
                missing_modalities.append(mod)

        if not missing_modalities:
            feat_df = pd.concat(modality_feat_dfs, axis=1) if len(modality_feat_dfs) > 1 else modality_feat_dfs[0]
            all_feat_dfs.append(feat_df)
            all_meta_rows.extend(cached_meta_rows or [])
            n_cached += 1
            print(f"{prefix} â€” cached ({feat_df.shape[0]} windows x {feat_df.shape[1]} features)")
            continue

        print(f"{prefix} â€” label={label} load={config_name} run={run}")

        t0 = time.perf_counter()
        print("    Loading data ...")
        try:
            capture = load_capture_data(path)
        except Exception as e:
            print(f"    WARNING: failed to load {path.name} ({e}), skipping.")
            continue

        new_feat_dfs: list[pd.DataFrame] = []
        new_meta_rows: list[dict] = []

        for mod in missing_modalities:
            mod_tag = MODALITY_TAGS[mod]
            print(f"    Building time-series windows for {mod} (window_size={window_sizes[mod]}) ...")
            ts_df, meta_rows = _build_recording_ts_df(
                capture, path.stem, label, load_str, run, window_sizes, [mod]
            )
            if ts_df.empty:
                print(f"    WARNING: no {mod} data, skipping modality.")
                continue

            n_windows = ts_df["id"].nunique()
            n_kinds = ts_df["kind"].nunique()
            print(f"    Extracting features for {n_windows} windows x {n_kinds} signals ({mod}) ...")
            t1 = time.perf_counter()
            feat = extract_features(
                ts_df,
                column_id="id",
                column_sort="time",
                column_kind="kind",
                column_value="value",
                default_fc_parameters=fc_params,
                impute_function=impute,
                n_jobs=args.n_jobs,
                disable_progressbar=True,
            )
            extract_time = time.perf_counter() - t1
            feat_df = cast(
                pd.DataFrame,
                feat if isinstance(feat, pd.DataFrame) else pd.DataFrame(feat),
            )

            mod_cache_dir = cache_base / mod_tag
            mod_cache_dir.mkdir(parents=True, exist_ok=True)
            cache_feat = mod_cache_dir / f"{path.stem}_feat.parquet"
            cache_meta = mod_cache_dir / f"{path.stem}_meta.parquet"
            feat_df.to_parquet(cache_feat, index=True)
            pd.DataFrame(meta_rows).to_parquet(cache_meta, index=False)

            new_feat_dfs.append(feat_df)
            if not new_meta_rows:
                new_meta_rows = meta_rows
            print(f"    -> {feat_df.shape[0]} windows x {feat_df.shape[1]} features "
                  f"(extract: {extract_time:.1f}s, total: {time.perf_counter() - t0:.1f}s)")

        combined_feat_dfs = modality_feat_dfs + new_feat_dfs
        if not combined_feat_dfs:
            print("    WARNING: no data extracted, skipping.")
            continue

        feat_df = pd.concat(combined_feat_dfs, axis=1) if len(combined_feat_dfs) > 1 else combined_feat_dfs[0]
        meta_rows_final = (cached_meta_rows or []) + new_meta_rows
        # Deduplicate meta rows by id (prefer first occurrence)
        seen: set[str] = set()
        meta_rows_deduped = [r for r in meta_rows_final if not (r["id"] in seen or seen.add(r["id"]))]  # type: ignore[func-returns-value]

        all_feat_dfs.append(feat_df)
        all_meta_rows.extend(meta_rows_deduped)

    if n_cached:
        print(f"\n  {n_cached}/{len(items)} recordings loaded from cache.")

    if not all_feat_dfs:
        print("No features extracted. Exiting.")
        return

    feature_df = pd.concat(all_feat_dfs)
    feature_df.index.name = "id"

    meta_df = pd.DataFrame(all_meta_rows).drop_duplicates(subset=["id"]).set_index("id")
    full_df = feature_df.join(meta_df, how="left")
    print(f"\nFull feature matrix: {full_df.shape[0]} windows x {full_df.shape[1]} columns")

    meta_cols = {"label", "load", "run", "filename"}
    feature_cols = [c for c in full_df.columns if c not in meta_cols]

    if not args.skip_selection:
        print(f"\nRunning feature selection (fdr_level={args.fdr_level}, multiclass) ...")
        le = LabelEncoder()
        label_values = full_df["label"].astype(str).to_numpy()
        encoded_labels = np.asarray(le.fit_transform(label_values), dtype=np.int64)
        y_enc = pd.Series(
            data=encoded_labels, index=full_df.index, name="label", dtype="int64"
        )
        X = full_df[feature_cols]
        impute(X)
        X_selected = select_features(
            X,
            y_enc,
            fdr_level=args.fdr_level,
            multiclass=True,
            n_significant=1,
            n_jobs=args.n_jobs,
        )
        selected_cols = list(X_selected.columns)
        print(f"Feature selection: {len(feature_cols)} -> {len(selected_cols)} features retained")
    else:
        selected_cols = feature_cols

    args.output_dir.mkdir(parents=True, exist_ok=True)

    full_path = args.output_dir / f"tsfresh_features_{tag}.parquet"
    full_df.to_parquet(full_path, index=True)
    print(f"\nSaved full features:     {full_path}")

    present_meta_cols = list(meta_cols & set(full_df.columns))
    selected_df = full_df[selected_cols + present_meta_cols]
    sel_path = args.output_dir / f"tsfresh_features_{tag}_selected.parquet"
    selected_df.to_parquet(sel_path, index=True)
    print(f"Saved selected features: {sel_path}")

    names_path = args.output_dir / f"tsfresh_feature_names_{tag}.txt"
    names_path.write_text("\n".join(selected_cols))
    print(f"Saved feature names:     {names_path}")
    print("\nDone.")


if __name__ == "__main__":
    main()
