"""
One-time data preparation for autoresearch time-series experiments.

Usage:
    uv run prepare.py
    uv run prepare.py --input-csv /path/to/sol.csv

Prepared artifacts are stored under ~/.cache/autoresearch/timeseries by default.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from contextlib import nullcontext
from dataclasses import dataclass

import numpy as np
import pandas as pd
import torch

# ---------------------------------------------------------------------------
# Fixed benchmark contract (only change with explicit human approval)
# ---------------------------------------------------------------------------

TIME_BUDGET = 600
MAX_TIME_BUDGET = 1_200
HORIZON_BARS = 60
VAL_RATIO = 0.15
TEST_RATIO = 0.15
NUM_VAL_FOLDS = 3
EVAL_SAMPLES = 131_072
PRIMARY_METRIC = "val_corr"
PRIMARY_METRIC_DIRECTION = "higher_is_better"
FEATURE_NAMES = (
    "log_close",
    "log_volume",
    "log_return_1",
    "log_return_5",
    "log_return_15",
    "log_return_60",
    "log_return_300",
    "log_return_900",
    "body_frac",
    "range_frac",
    "upper_wick_frac",
    "lower_wick_frac",
    "close_vs_sma_60",
    "close_vs_sma_300",
    "close_vs_sma_900",
    "volume_zscore_60",
    "volume_zscore_300",
    "realized_vol_60",
    "realized_vol_300",
    "realized_vol_900",
    "vol_ratio_60_300",
    "log_seconds_since_trade",
    "is_synthetic_bar",
    "synthetic_frac_60",
)

# ---------------------------------------------------------------------------
# Signal-engineering defaults (future experiment agents may modify)
# ---------------------------------------------------------------------------

LOOKBACK_BARS = 300
FEATURE_WINDOWS = (10, 60, 300, 900)

# Derived from the benchmark contract plus the current signal settings.
FEATURE_WARMUP_BARS = max(FEATURE_WINDOWS)
PURGE_BARS = LOOKBACK_BARS + HORIZON_BARS

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

REPO_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_INPUT_CSV = os.path.join(REPO_DIR, "gecko_pool_1s_history.csv")
CACHE_DIR = os.environ.get(
    "AUTORESEARCH_CACHE_DIR",
    os.path.join(os.path.expanduser("~"), ".cache", "autoresearch"),
)
PREPARED_DIR = os.environ.get(
    "AUTORESEARCH_PREPARED_DIR",
    os.path.join(CACHE_DIR, "timeseries"),
)
FEATURES_FILENAME = "features.npy"
TARGETS_FILENAME = "targets.npy"
METADATA_FILENAME = "metadata.json"
MIN_STD = 1e-6
EPS = 1e-12


# ---------------------------------------------------------------------------
# Metadata and dataset loading
# ---------------------------------------------------------------------------

@dataclass
class PreparedDataset:
    features: np.ndarray
    targets: np.ndarray
    metadata: dict

    def __post_init__(self):
        self.lookback_bars = int(self.metadata["lookback_bars"])
        self.window_offsets = np.arange(self.lookback_bars, dtype=np.int64)

    @property
    def input_dim(self):
        return int(self.features.shape[1])

    def split_bounds(self, split):
        info = self.metadata["splits"][split]
        return int(info["sample_start"]), int(info["sample_stop"])


def load_prepared_dataset(prepared_dir=PREPARED_DIR):
    metadata_path = os.path.join(prepared_dir, METADATA_FILENAME)
    features_path = os.path.join(prepared_dir, FEATURES_FILENAME)
    targets_path = os.path.join(prepared_dir, TARGETS_FILENAME)
    if not os.path.exists(metadata_path):
        raise FileNotFoundError(
            f"No prepared dataset found at {prepared_dir}. Run `uv run prepare.py` first."
        )
    with open(metadata_path, "r", encoding="utf-8") as f:
        metadata = json.load(f)
    features = np.load(features_path, mmap_mode="r")
    targets = np.load(targets_path, mmap_mode="r")
    return PreparedDataset(features=features, targets=targets, metadata=metadata)


# ---------------------------------------------------------------------------
# Numeric helpers
# ---------------------------------------------------------------------------

def lagged_difference(values, lag):
    out = np.full(values.shape, np.nan, dtype=np.float32)
    out[lag:] = values[lag:] - values[:-lag]
    return out


def rolling_mean(values, window):
    values64 = np.asarray(values, dtype=np.float64)
    out = np.full(values64.shape, np.nan, dtype=np.float32)
    if window <= 1:
        out[:] = values64
        return out
    cumsum = np.cumsum(values64, dtype=np.float64)
    totals = cumsum[window - 1 :].copy()
    totals[1:] -= cumsum[:-window]
    out[window - 1 :] = totals / window
    return out


def rolling_std(values, window):
    values64 = np.asarray(values, dtype=np.float64)
    out = np.full(values64.shape, np.nan, dtype=np.float32)
    if window <= 1:
        out[:] = 0.0
        return out
    cumsum = np.cumsum(values64, dtype=np.float64)
    cumsum_sq = np.cumsum(values64 * values64, dtype=np.float64)
    totals = cumsum[window - 1 :].copy()
    totals[1:] -= cumsum[:-window]
    sq_totals = cumsum_sq[window - 1 :].copy()
    sq_totals[1:] -= cumsum_sq[:-window]
    means = totals / window
    variances = np.maximum(sq_totals / window - means * means, 0.0)
    out[window - 1 :] = np.sqrt(variances)
    return out


def safe_zscore(values, mean_values, std_values):
    return (values - mean_values) / np.maximum(std_values, MIN_STD)


def dense_row_to_timestamp(metadata, row_idx):
    return int(metadata["dense_start_time"]) + int(row_idx)


def summarize_split(name, split_info):
    return (
        f"{name:>5s}: {split_info['num_samples']:,} samples | "
        f"{split_info['timestamp_start']} -> {split_info['timestamp_stop']}"
    )


# ---------------------------------------------------------------------------
# Data preparation
# ---------------------------------------------------------------------------

def load_raw_bars(input_csv):
    usecols = ["time", "open", "high", "low", "close", "volume"]
    dtypes = {
        "time": "int64",
        "open": "float32",
        "high": "float32",
        "low": "float32",
        "close": "float32",
        "volume": "float32",
    }
    df = pd.read_csv(input_csv, usecols=usecols, dtype=dtypes)
    if df.empty:
        raise ValueError(f"No rows found in {input_csv}")
    df = df.sort_values("time").drop_duplicates("time", keep="last").reset_index(drop=True)
    times = df["time"].to_numpy(dtype=np.int64, copy=True)
    if np.any(np.diff(times) <= 0):
        raise ValueError("Expected strictly increasing timestamps after dedupe.")
    bars = {
        "open": df["open"].to_numpy(dtype=np.float32, copy=True),
        "high": df["high"].to_numpy(dtype=np.float32, copy=True),
        "low": df["low"].to_numpy(dtype=np.float32, copy=True),
        "close": df["close"].to_numpy(dtype=np.float32, copy=True),
        "volume": df["volume"].to_numpy(dtype=np.float32, copy=True),
    }
    return times, bars


def densify_bars(times, raw_bars):
    segment_lengths = np.diff(np.append(times, times[-1] + 1)).astype(np.int64, copy=False)
    dense_length = int(segment_lengths.sum())
    dense_close = np.repeat(raw_bars["close"], segment_lengths).astype(np.float32, copy=False)
    dense_open = dense_close.copy()
    dense_high = dense_close.copy()
    dense_low = dense_close.copy()
    dense_volume = np.zeros(dense_length, dtype=np.float32)
    observed = np.zeros(dense_length, dtype=np.float32)

    start_rows = np.empty(times.shape[0], dtype=np.int32)
    start_rows[0] = 0
    if len(times) > 1:
        np.cumsum(segment_lengths[:-1], out=start_rows[1:])

    dense_open[start_rows] = raw_bars["open"]
    dense_high[start_rows] = raw_bars["high"]
    dense_low[start_rows] = raw_bars["low"]
    dense_volume[start_rows] = raw_bars["volume"]
    observed[start_rows] = 1.0

    dense_rows = np.arange(dense_length, dtype=np.int32)
    seconds_since_trade = dense_rows - np.repeat(start_rows, segment_lengths)

    return {
        "open": dense_open,
        "high": dense_high,
        "low": dense_low,
        "close": dense_close,
        "volume": dense_volume,
        "observed": observed,
        "seconds_since_trade": seconds_since_trade.astype(np.float32, copy=False),
        "dense_start_time": int(times[0]),
        "dense_end_time": int(times[-1]),
        "num_raw_rows": int(len(times)),
    }


def build_splits(num_rows, dense_start_time):
    min_sample_end = LOOKBACK_BARS + FEATURE_WARMUP_BARS - 2
    max_sample_end = num_rows - HORIZON_BARS
    if max_sample_end <= min_sample_end:
        raise ValueError("Dataset is too short for the configured lookback, horizon, and features.")

    usable_samples = max_sample_end - min_sample_end
    val_len = max(1, int(usable_samples * VAL_RATIO))
    test_len = max(1, int(usable_samples * TEST_RATIO))

    test_start = max_sample_end - test_len
    val_stop = test_start - PURGE_BARS
    val_start = val_stop - val_len
    train_stop = val_start - PURGE_BARS
    if train_stop <= min_sample_end:
        raise ValueError("Not enough room for train/val/test splits after purge gaps.")

    splits = {
        "train": {"sample_start": int(min_sample_end), "sample_stop": int(train_stop)},
        "val": {"sample_start": int(val_start), "sample_stop": int(val_stop)},
        "test": {"sample_start": int(test_start), "sample_stop": int(max_sample_end)},
    }

    for split_info in splits.values():
        split_info["num_samples"] = int(split_info["sample_stop"] - split_info["sample_start"])
        split_info["timestamp_start"] = dense_start_time + split_info["sample_start"]
        split_info["timestamp_stop"] = dense_start_time + split_info["sample_stop"] - 1

    fold_len = max(1, (splits["val"]["sample_stop"] - splits["val"]["sample_start"]) // NUM_VAL_FOLDS)
    validation_folds = []
    fold_start = splits["val"]["sample_start"]
    for fold_idx in range(NUM_VAL_FOLDS):
        fold_stop = splits["val"]["sample_stop"] if fold_idx == NUM_VAL_FOLDS - 1 else min(
            splits["val"]["sample_stop"], fold_start + fold_len
        )
        fold_train_stop = fold_start - PURGE_BARS
        if fold_train_stop <= min_sample_end:
            break
        validation_folds.append(
            {
                "fold": fold_idx,
                "train_start": int(min_sample_end),
                "train_stop": int(fold_train_stop),
                "val_start": int(fold_start),
                "val_stop": int(fold_stop),
                "train_timestamp_start": dense_start_time + int(min_sample_end),
                "train_timestamp_stop": dense_start_time + int(fold_train_stop) - 1,
                "val_timestamp_start": dense_start_time + int(fold_start),
                "val_timestamp_stop": dense_start_time + int(fold_stop) - 1,
            }
        )
        fold_start = fold_stop

    return splits, validation_folds, int(min_sample_end), int(max_sample_end)


def build_target(log_close):
    target = np.full(log_close.shape, np.nan, dtype=np.float32)
    target[:-HORIZON_BARS] = log_close[HORIZON_BARS:] - log_close[:-HORIZON_BARS]
    return target


def generate_feature_columns(dense_bars):
    open_ = dense_bars["open"]
    high = dense_bars["high"]
    low = dense_bars["low"]
    close = dense_bars["close"]
    volume = dense_bars["volume"]
    observed = dense_bars["observed"]
    seconds_since_trade = dense_bars["seconds_since_trade"]

    safe_close = np.maximum(close, EPS)
    safe_open = np.maximum(open_, EPS)
    safe_volume = np.maximum(volume, 0.0)

    log_close = np.log(safe_close).astype(np.float32)
    log_volume = np.log1p(safe_volume).astype(np.float32)
    log_return_1 = lagged_difference(log_close, 1)
    realized_vol_input = np.nan_to_num(log_return_1, nan=0.0)

    yield "log_close", log_close
    yield "log_volume", log_volume
    yield "log_return_1", log_return_1
    yield "log_return_5", lagged_difference(log_close, 5)
    yield "log_return_15", lagged_difference(log_close, 15)
    yield "log_return_60", lagged_difference(log_close, 60)
    yield "log_return_300", lagged_difference(log_close, 300)
    yield "log_return_900", lagged_difference(log_close, 900)
    yield "body_frac", ((close - open_) / safe_open).astype(np.float32)
    yield "range_frac", ((high - low) / safe_close).astype(np.float32)
    yield "upper_wick_frac", ((high - np.maximum(open_, close)) / safe_close).astype(np.float32)
    yield "lower_wick_frac", ((np.minimum(open_, close) - low) / safe_close).astype(np.float32)

    close_sma_60 = rolling_mean(close, 60)
    close_sma_300 = rolling_mean(close, 300)
    close_sma_900 = rolling_mean(close, 900)
    yield "close_vs_sma_60", ((close / np.maximum(close_sma_60, EPS)) - 1.0).astype(np.float32)
    yield "close_vs_sma_300", ((close / np.maximum(close_sma_300, EPS)) - 1.0).astype(np.float32)
    yield "close_vs_sma_900", ((close / np.maximum(close_sma_900, EPS)) - 1.0).astype(np.float32)

    volume_mean_60 = rolling_mean(log_volume, 60)
    volume_std_60 = rolling_std(log_volume, 60)
    volume_mean_300 = rolling_mean(log_volume, 300)
    volume_std_300 = rolling_std(log_volume, 300)
    realized_vol_60 = rolling_std(realized_vol_input, 60)
    realized_vol_300 = rolling_std(realized_vol_input, 300)
    realized_vol_900 = rolling_std(realized_vol_input, 900)
    synthetic_frac_60 = rolling_mean(1.0 - observed, 60)
    yield "volume_zscore_60", safe_zscore(log_volume, volume_mean_60, volume_std_60).astype(np.float32)
    yield "volume_zscore_300", safe_zscore(log_volume, volume_mean_300, volume_std_300).astype(np.float32)
    yield "realized_vol_60", realized_vol_60
    yield "realized_vol_300", realized_vol_300
    yield "realized_vol_900", realized_vol_900
    yield "vol_ratio_60_300", (
        (realized_vol_60 / np.maximum(realized_vol_300, MIN_STD)) - 1.0
    ).astype(np.float32)
    yield "log_seconds_since_trade", np.log1p(seconds_since_trade).astype(np.float32)
    yield "is_synthetic_bar", (1.0 - observed).astype(np.float32)
    yield "synthetic_frac_60", synthetic_frac_60.astype(np.float32)


def write_prepared_arrays(dense_bars, splits, output_dir):
    num_rows = len(dense_bars["close"])
    feature_path = os.path.join(output_dir, FEATURES_FILENAME)
    target_path = os.path.join(output_dir, TARGETS_FILENAME)

    log_close = np.log(np.maximum(dense_bars["close"], EPS)).astype(np.float32)
    target = build_target(log_close)
    target_memmap = np.lib.format.open_memmap(
        target_path, mode="w+", dtype=np.float32, shape=target.shape
    )
    target_memmap[:] = target
    target_memmap.flush()

    scaler_row_start = FEATURE_WARMUP_BARS - 1
    scaler_row_stop = splits["train"]["sample_stop"]

    feature_names = []
    scaler = {}
    features_memmap = np.lib.format.open_memmap(
        feature_path, mode="w+", dtype=np.float32, shape=(num_rows, len(FEATURE_NAMES))
    )
    for feature_idx, (name, values) in enumerate(generate_feature_columns(dense_bars)):
        train_slice = values[scaler_row_start:scaler_row_stop]
        mean = float(np.nanmean(train_slice))
        std = float(np.nanstd(train_slice))
        if not np.isfinite(std) or std < MIN_STD:
            std = 1.0
        scaled = ((values - mean) / std).astype(np.float32, copy=False)
        np.nan_to_num(scaled, copy=False, nan=0.0, posinf=0.0, neginf=0.0)
        features_memmap[:, feature_idx] = scaled
        feature_names.append(name)
        scaler[name] = {"mean": mean, "std": std}
    features_memmap.flush()
    if tuple(feature_names) != FEATURE_NAMES:
        raise ValueError("Feature generator order drifted from the benchmark feature list.")
    return feature_names, scaler


def save_metadata(output_dir, metadata):
    metadata_path = os.path.join(output_dir, METADATA_FILENAME)
    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, sort_keys=True)


def prepare_dataset(input_csv=DEFAULT_INPUT_CSV, output_dir=PREPARED_DIR):
    t0 = time.time()
    if not os.path.exists(input_csv):
        raise FileNotFoundError(f"Input CSV not found: {input_csv}")

    os.makedirs(output_dir, exist_ok=True)

    raw_times, raw_bars = load_raw_bars(input_csv)
    dense_bars = densify_bars(raw_times, raw_bars)
    num_dense_rows = len(dense_bars["close"])
    splits, validation_folds, min_sample_end, max_sample_end = build_splits(
        num_dense_rows, dense_bars["dense_start_time"]
    )
    feature_names, scaler = write_prepared_arrays(dense_bars, splits, output_dir)

    metadata = {
        "version": 1,
        "input_csv": os.path.abspath(input_csv),
        "prepared_dir": os.path.abspath(output_dir),
        "time_budget": TIME_BUDGET,
        "default_time_budget_seconds": TIME_BUDGET,
        "max_time_budget_seconds": MAX_TIME_BUDGET,
        "primary_metric": PRIMARY_METRIC,
        "primary_metric_direction": PRIMARY_METRIC_DIRECTION,
        "lookback_bars": LOOKBACK_BARS,
        "horizon_bars": HORIZON_BARS,
        "purge_bars": PURGE_BARS,
        "eval_samples": EVAL_SAMPLES,
        "feature_windows": list(FEATURE_WINDOWS),
        "feature_warmup_bars": FEATURE_WARMUP_BARS,
        "feature_names": feature_names,
        "scaler": scaler,
        "dense_start_time": dense_bars["dense_start_time"],
        "dense_end_time": dense_bars["dense_end_time"],
        "num_raw_rows": dense_bars["num_raw_rows"],
        "num_dense_rows": int(num_dense_rows),
        "num_observed_rows": int(np.count_nonzero(dense_bars["observed"])),
        "num_synthetic_rows": int(num_dense_rows - np.count_nonzero(dense_bars["observed"])),
        "sample_end_start": min_sample_end,
        "sample_end_stop": max_sample_end,
        "splits": splits,
        "validation_folds": validation_folds,
    }
    save_metadata(output_dir, metadata)

    elapsed = time.time() - t0
    print(f"Prepared dataset in {elapsed:.1f}s")
    print(f"Input CSV: {os.path.abspath(input_csv)}")
    print(f"Prepared dir: {os.path.abspath(output_dir)}")
    print(f"Observed bars: {metadata['num_observed_rows']:,}")
    print(f"Synthetic bars: {metadata['num_synthetic_rows']:,}")
    print(f"Features: {len(feature_names)} -> {', '.join(feature_names)}")
    print(summarize_split("train", splits["train"]))
    print(summarize_split("val", splits["val"]))
    print(summarize_split("test", splits["test"]))
    return metadata


# ---------------------------------------------------------------------------
# Runtime helpers for train.py
# ---------------------------------------------------------------------------

def build_window_batch(dataset, endpoints, device):
    row_idx = endpoints[:, None] - dataset.lookback_bars + 1 + dataset.window_offsets[None, :]
    x_np = np.asarray(dataset.features[row_idx], dtype=np.float32)
    y_np = np.asarray(dataset.targets[endpoints], dtype=np.float32)

    x = torch.from_numpy(x_np)
    y = torch.from_numpy(y_np)
    if device.type == "cuda":
        x = x.pin_memory().to(device, non_blocking=True)
        y = y.pin_memory().to(device, non_blocking=True)
    else:
        x = x.to(device)
        y = y.to(device)
    return x, y


def sample_batch(dataset, split, batch_size, device, rng):
    start, stop = dataset.split_bounds(split)
    endpoints = rng.integers(start, stop, size=batch_size, endpoint=False, dtype=np.int64)
    return build_window_batch(dataset, endpoints, device)


def select_eval_endpoints(dataset, split, max_samples=None):
    start, stop = dataset.split_bounds(split)
    total = stop - start
    limit = total if max_samples is None else min(total, int(max_samples))
    if limit <= 0:
        raise ValueError(f"Split {split} has no available samples.")
    if limit == total:
        return np.arange(start, stop, dtype=np.int64)
    return np.linspace(start, stop - 1, num=limit, dtype=np.int64)


def iter_eval_batches(dataset, split, batch_size, device, max_samples=None):
    endpoints = select_eval_endpoints(dataset, split, max_samples=max_samples)
    for start_idx in range(0, len(endpoints), batch_size):
        batch_endpoints = endpoints[start_idx : start_idx + batch_size]
        yield build_window_batch(dataset, batch_endpoints, device)


def regression_metrics(predictions, targets):
    diff = predictions - targets
    mse = float(np.mean(diff * diff))
    mae = float(np.mean(np.abs(diff)))
    pred_centered = predictions - predictions.mean()
    target_centered = targets - targets.mean()
    denom = float(np.sqrt(np.mean(pred_centered * pred_centered) * np.mean(target_centered * target_centered)))
    corr = 0.0 if denom < MIN_STD else float(np.mean(pred_centered * target_centered) / denom)
    sign_acc = float(np.mean((predictions >= 0.0) == (targets >= 0.0)))
    return {
        "rmse": float(np.sqrt(mse)),
        "mae": mae,
        "corr": corr,
        "sign_accuracy": sign_acc,
    }


@torch.no_grad()
def evaluate_regression(model, dataset, split, batch_size, device, autocast_ctx=None, max_samples=None):
    ctx = autocast_ctx if autocast_ctx is not None else nullcontext()
    pred_chunks = []
    target_chunks = []
    for x, y in iter_eval_batches(dataset, split, batch_size, device, max_samples=max_samples):
        with ctx:
            preds = model(x)
        pred_chunks.append(preds.float().cpu())
        target_chunks.append(y.float().cpu())
    predictions = torch.cat(pred_chunks).numpy()
    targets = torch.cat(target_chunks).numpy()
    return regression_metrics(predictions, targets)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prepare dense SOL OHLCV features for forecasting.")
    parser.add_argument(
        "--input-csv",
        type=str,
        default=DEFAULT_INPUT_CSV,
        help="Path to the raw SOL OHLCV CSV.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=PREPARED_DIR,
        help="Directory for prepared arrays and metadata.",
    )
    args = parser.parse_args()
    prepare_dataset(input_csv=args.input_csv, output_dir=args.output_dir)
