"""
One-time data preparation for autoresearch Orca minute-level experiments.

Usage:
    uv run prepare.py
    uv run prepare.py --input-dir /path/to/orca_hist_last_year

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
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq
import torch

# ---------------------------------------------------------------------------
# Fixed benchmark contract (only change with explicit human approval)
# ---------------------------------------------------------------------------

PREPARED_VERSION = 4
MAX_TIME_BUDGET = 1_200
BAR_SECONDS = 60
HORIZON_BARS = 3
VAL_RATIO = 0.15
TEST_RATIO = 0.15
NUM_VAL_FOLDS = 3
EVAL_SAMPLES = 131_072
PRIMARY_METRIC = "val_corr"
PRIMARY_METRIC_DIRECTION = "higher_is_better"
SOURCE_KIND = "orca_parquet_dir"
SPLIT_STRATEGY = "anchored_test_walk_forward_val"
FEATURE_NAMES = (
    "log_close",
    "log_volume",
    "log_return_1",
    "log_return_3",
    "log_return_15",
    "log_return_60",
    "body_frac",
    "range_frac",
    "upper_wick_frac",
    "lower_wick_frac",
    "close_vs_sma_15",
    "close_vs_sma_60",
    "close_vs_sma_300",
    "volume_zscore_15",
    "volume_zscore_60",
    "realized_vol_15",
    "realized_vol_60",
    "log_minutes_since_observed",
    "is_synthetic_bar",
    "log_swap_count",
    "log_swap_amount_in_sum",
    "log_swap_amount_out_sum",
    "log_swap_amount_in_mean",
    "log_swap_amount_out_mean",
    "log_swap_amount_in_max",
    "log_swap_amount_out_max",
    "swap_tick_delta_sum",
    "swap_tick_delta_abs_sum",
    "swap_tick_delta_max_abs",
    "log_liq_event_count",
    "log_liq_delta_abs_sum",
    "liq_delta_net",
    "log_liq_increase_count",
    "log_liq_decrease_count",
    "log_liq_open_count",
    "log_liq_close_count",
    "log_liq_reset_range_count",
    "liq_range_width_mean",
    "daily_tick_current_index",
    "daily_log_liquidity",
    "daily_tick_change_1d",
    "daily_log_liquidity_change_1d",
)

# ---------------------------------------------------------------------------
# Signal-engineering defaults (future experiment agents may modify)
# ---------------------------------------------------------------------------

LOOKBACK_BARS = 300
FEATURE_WINDOWS = (15, 60, 300)

# Derived from the benchmark contract plus the current signal settings.
FEATURE_WARMUP_BARS = max(FEATURE_WINDOWS)
PURGE_BARS = LOOKBACK_BARS + HORIZON_BARS

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

REPO_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_INPUT_DIR = os.path.join(REPO_DIR, "orca_hist_last_year")
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
OHLCV_FILENAME = "ohlcv_minutely.parquet"
SWAPS_FILENAME = "swaps.parquet"
LIQUIDITY_EVENTS_FILENAME = "liquidity_events.parquet"
DAILY_STATE_FILENAME = "daily_state.parquet"
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
    return int(metadata["dense_start_time"]) + int(row_idx) * int(metadata.get("bar_seconds", BAR_SECONDS))


def summarize_split(name, split_info):
    return (
        f"{name:>5s}: {split_info['num_samples']:,} samples | "
        f"{split_info['timestamp_start']} -> {split_info['timestamp_stop']}"
    )


# ---------------------------------------------------------------------------
# Data preparation
# ---------------------------------------------------------------------------

def arrow_array_to_numpy(array, dtype):
    values = array.to_numpy(zero_copy_only=False)
    return np.asarray(values, dtype=dtype)


def arrow_decimal_to_float64(array):
    values = pc.cast(array, pa.float64(), safe=False).to_numpy(zero_copy_only=False)
    return np.asarray(values, dtype=np.float64)


def validate_input_dir(input_dir):
    if not os.path.isdir(input_dir):
        raise FileNotFoundError(f"Input directory not found: {input_dir}")
    required_files = (
        METADATA_FILENAME,
        OHLCV_FILENAME,
        SWAPS_FILENAME,
        LIQUIDITY_EVENTS_FILENAME,
        DAILY_STATE_FILENAME,
    )
    for filename in required_files:
        path = os.path.join(input_dir, filename)
        if not os.path.exists(path):
            raise FileNotFoundError(f"Missing required input file: {path}")


def load_source_manifest(input_dir):
    manifest_path = os.path.join(input_dir, METADATA_FILENAME)
    with open(manifest_path, "r", encoding="utf-8") as f:
        return json.load(f)


def compute_steps_since_observed(observed):
    observed = np.asarray(observed, dtype=bool)
    if observed.size == 0:
        return np.zeros(0, dtype=np.float32)
    row_idx = np.arange(observed.size, dtype=np.int64)
    last_observed = np.maximum.accumulate(np.where(observed, row_idx, 0))
    return (row_idx - last_observed).astype(np.float32, copy=False)


def load_ohlcv_minutely_bars(input_dir):
    path = os.path.join(input_dir, OHLCV_FILENAME)
    usecols = ["timestamp", "open", "high", "low", "close", "volume"]
    df = pd.read_parquet(path, columns=usecols)
    if df.empty:
        raise ValueError(f"No rows found in {path}")

    for col in usecols:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df["timestamp"] = df["timestamp"].astype(np.int64, copy=False)
    df = df.sort_values("timestamp").drop_duplicates("timestamp", keep="last").reset_index(drop=True)

    timestamps = df["timestamp"].to_numpy(dtype=np.int64, copy=True)
    if np.any(np.diff(timestamps) <= 0):
        raise ValueError("Expected strictly increasing minute timestamps after dedupe.")
    if np.any(timestamps % BAR_SECONDS != 0):
        raise ValueError("Expected minute timestamps aligned to 60-second boundaries.")

    full_index = pd.Index(
        np.arange(timestamps[0], timestamps[-1] + BAR_SECONDS, BAR_SECONDS, dtype=np.int64),
        name="timestamp",
    )
    minute_frame = df.set_index("timestamp").reindex(full_index)
    observed = minute_frame["close"].notna().to_numpy(dtype=np.float32, copy=False)

    close = pd.to_numeric(minute_frame["close"], errors="coerce").ffill().bfill()
    open_ = pd.to_numeric(minute_frame["open"], errors="coerce").fillna(close)
    high = pd.to_numeric(minute_frame["high"], errors="coerce").fillna(close)
    low = pd.to_numeric(minute_frame["low"], errors="coerce").fillna(close)
    volume = pd.to_numeric(minute_frame["volume"], errors="coerce").fillna(0.0).clip(lower=0.0)

    return {
        "open": open_.to_numpy(dtype=np.float32, copy=True),
        "high": high.to_numpy(dtype=np.float32, copy=True),
        "low": low.to_numpy(dtype=np.float32, copy=True),
        "close": close.to_numpy(dtype=np.float32, copy=True),
        "volume": volume.to_numpy(dtype=np.float32, copy=True),
        "observed": observed,
        "minutes_since_observed": compute_steps_since_observed(observed),
        "dense_start_time": int(full_index[0]),
        "dense_end_time": int(full_index[-1]),
        "num_raw_rows": int(len(df)),
    }


def aggregate_swaps(input_dir, dense_start_time, num_rows):
    path = os.path.join(input_dir, SWAPS_FILENAME)
    parquet_file = pq.ParquetFile(path)
    columns = [
        "block_time",
        "amount_in",
        "amount_out",
        "tick_current_index_pre",
        "tick_current_index_post",
    ]
    col_idx = {name: idx for idx, name in enumerate(columns)}

    swap_count = np.zeros(num_rows, dtype=np.float64)
    amount_in_sum = np.zeros(num_rows, dtype=np.float64)
    amount_out_sum = np.zeros(num_rows, dtype=np.float64)
    amount_in_max = np.zeros(num_rows, dtype=np.float64)
    amount_out_max = np.zeros(num_rows, dtype=np.float64)
    tick_delta_sum = np.zeros(num_rows, dtype=np.float64)
    tick_delta_abs_sum = np.zeros(num_rows, dtype=np.float64)
    tick_delta_abs_max = np.zeros(num_rows, dtype=np.float64)
    total_rows = 0

    for batch in parquet_file.iter_batches(batch_size=1_000_000, columns=columns):
        total_rows += batch.num_rows
        block_time = arrow_array_to_numpy(batch.column(col_idx["block_time"]), np.int64)
        row_idx = ((block_time - dense_start_time) // BAR_SECONDS).astype(np.int64, copy=False)
        valid = (row_idx >= 0) & (row_idx < num_rows)
        if not np.any(valid):
            continue

        row_idx = row_idx[valid]
        amount_in = np.nan_to_num(
            arrow_decimal_to_float64(batch.column(col_idx["amount_in"]))[valid],
            nan=0.0,
            posinf=0.0,
            neginf=0.0,
        )
        amount_out = np.nan_to_num(
            arrow_decimal_to_float64(batch.column(col_idx["amount_out"]))[valid],
            nan=0.0,
            posinf=0.0,
            neginf=0.0,
        )
        tick_pre = arrow_array_to_numpy(batch.column(col_idx["tick_current_index_pre"]), np.float64)[valid]
        tick_post = arrow_array_to_numpy(batch.column(col_idx["tick_current_index_post"]), np.float64)[valid]
        tick_delta = np.nan_to_num(tick_post - tick_pre, nan=0.0, posinf=0.0, neginf=0.0)
        tick_delta_abs = np.abs(tick_delta)

        swap_count += np.bincount(row_idx, minlength=num_rows)
        amount_in_sum += np.bincount(row_idx, weights=amount_in, minlength=num_rows)
        amount_out_sum += np.bincount(row_idx, weights=amount_out, minlength=num_rows)
        tick_delta_sum += np.bincount(row_idx, weights=tick_delta, minlength=num_rows)
        tick_delta_abs_sum += np.bincount(row_idx, weights=tick_delta_abs, minlength=num_rows)
        np.maximum.at(amount_in_max, row_idx, amount_in)
        np.maximum.at(amount_out_max, row_idx, amount_out)
        np.maximum.at(tick_delta_abs_max, row_idx, tick_delta_abs)

    count_denom = np.maximum(swap_count, 1.0)
    return {
        "swap_count": swap_count.astype(np.float32, copy=False),
        "swap_amount_in_sum": amount_in_sum.astype(np.float32, copy=False),
        "swap_amount_out_sum": amount_out_sum.astype(np.float32, copy=False),
        "swap_amount_in_mean": (amount_in_sum / count_denom).astype(np.float32, copy=False),
        "swap_amount_out_mean": (amount_out_sum / count_denom).astype(np.float32, copy=False),
        "swap_amount_in_max": amount_in_max.astype(np.float32, copy=False),
        "swap_amount_out_max": amount_out_max.astype(np.float32, copy=False),
        "swap_tick_delta_sum": tick_delta_sum.astype(np.float32, copy=False),
        "swap_tick_delta_abs_sum": tick_delta_abs_sum.astype(np.float32, copy=False),
        "swap_tick_delta_max_abs": tick_delta_abs_max.astype(np.float32, copy=False),
    }, int(total_rows)


def aggregate_liquidity_events(input_dir, dense_start_time, num_rows):
    path = os.path.join(input_dir, LIQUIDITY_EVENTS_FILENAME)
    parquet_file = pq.ParquetFile(path)
    columns = [
        "block_time",
        "instruction",
        "tick_lower_index",
        "tick_upper_index",
        "liquidity_delta",
    ]
    col_idx = {name: idx for idx, name in enumerate(columns)}

    liq_event_count = np.zeros(num_rows, dtype=np.float64)
    liq_delta_net = np.zeros(num_rows, dtype=np.float64)
    liq_delta_abs_sum = np.zeros(num_rows, dtype=np.float64)
    liq_range_width_sum = np.zeros(num_rows, dtype=np.float64)
    liq_increase_count = np.zeros(num_rows, dtype=np.float64)
    liq_decrease_count = np.zeros(num_rows, dtype=np.float64)
    liq_open_count = np.zeros(num_rows, dtype=np.float64)
    liq_close_count = np.zeros(num_rows, dtype=np.float64)
    liq_reset_range_count = np.zeros(num_rows, dtype=np.float64)
    total_rows = 0

    for batch in parquet_file.iter_batches(batch_size=1_000_000, columns=columns):
        total_rows += batch.num_rows
        block_time = arrow_array_to_numpy(batch.column(col_idx["block_time"]), np.int64)
        row_idx = ((block_time - dense_start_time) // BAR_SECONDS).astype(np.int64, copy=False)
        valid = (row_idx >= 0) & (row_idx < num_rows)
        if not np.any(valid):
            continue

        row_idx = row_idx[valid]
        instructions = batch.column(col_idx["instruction"]).to_numpy(zero_copy_only=False)[valid]
        tick_lower = arrow_array_to_numpy(batch.column(col_idx["tick_lower_index"]), np.float64)[valid]
        tick_upper = arrow_array_to_numpy(batch.column(col_idx["tick_upper_index"]), np.float64)[valid]
        range_width = np.nan_to_num(tick_upper - tick_lower, nan=0.0, posinf=0.0, neginf=0.0)
        liq_delta = np.nan_to_num(
            arrow_decimal_to_float64(batch.column(col_idx["liquidity_delta"]))[valid],
            nan=0.0,
            posinf=0.0,
            neginf=0.0,
        )

        liq_event_count += np.bincount(row_idx, minlength=num_rows)
        liq_delta_net += np.bincount(row_idx, weights=liq_delta, minlength=num_rows)
        liq_delta_abs_sum += np.bincount(row_idx, weights=np.abs(liq_delta), minlength=num_rows)
        liq_range_width_sum += np.bincount(row_idx, weights=range_width, minlength=num_rows)

        for instruction_name, accumulator in (
            ("increaseLiquidity", liq_increase_count),
            ("decreaseLiquidity", liq_decrease_count),
            ("openPosition", liq_open_count),
            ("closePosition", liq_close_count),
            ("resetPositionRange", liq_reset_range_count),
        ):
            mask = instructions == instruction_name
            if np.any(mask):
                accumulator += np.bincount(row_idx[mask], minlength=num_rows)

    count_denom = np.maximum(liq_event_count, 1.0)
    return {
        "liq_event_count": liq_event_count.astype(np.float32, copy=False),
        "liq_delta_abs_sum": liq_delta_abs_sum.astype(np.float32, copy=False),
        "liq_delta_net": liq_delta_net.astype(np.float32, copy=False),
        "liq_increase_count": liq_increase_count.astype(np.float32, copy=False),
        "liq_decrease_count": liq_decrease_count.astype(np.float32, copy=False),
        "liq_open_count": liq_open_count.astype(np.float32, copy=False),
        "liq_close_count": liq_close_count.astype(np.float32, copy=False),
        "liq_reset_range_count": liq_reset_range_count.astype(np.float32, copy=False),
        "liq_range_width_mean": (liq_range_width_sum / count_denom).astype(np.float32, copy=False),
    }, int(total_rows)


def load_daily_state_overlay(input_dir, dense_start_time, num_rows):
    path = os.path.join(input_dir, DAILY_STATE_FILENAME)
    df = pd.read_parquet(path, columns=["date", "tick_current_index", "liquidity"])
    if df.empty:
        raise ValueError(f"No rows found in {path}")

    df["timestamp"] = (
        pd.to_datetime(df["date"], utc=True).astype("int64") // 1_000_000_000
    ).astype(np.int64, copy=False)
    df["tick_current_index"] = pd.to_numeric(df["tick_current_index"], errors="coerce")
    df["liquidity"] = pd.to_numeric(df["liquidity"], errors="coerce")
    df = df.sort_values("timestamp").drop_duplicates("timestamp", keep="last").reset_index(drop=True)

    minute_index = pd.Index(
        np.arange(dense_start_time, dense_start_time + num_rows * BAR_SECONDS, BAR_SECONDS, dtype=np.int64),
        name="timestamp",
    )
    overlay = df.set_index("timestamp").reindex(minute_index, method="ffill")

    daily_tick = overlay["tick_current_index"].to_numpy(dtype=np.float32, copy=True)
    daily_log_liquidity = np.log1p(np.maximum(overlay["liquidity"].to_numpy(dtype=np.float64, copy=False), 0.0))
    daily_tick_change_1d = lagged_difference(daily_tick.astype(np.float32, copy=False), 24 * 60)
    daily_log_liquidity_change_1d = lagged_difference(daily_log_liquidity.astype(np.float32, copy=False), 24 * 60)

    return {
        "daily_tick_current_index": daily_tick.astype(np.float32, copy=False),
        "daily_log_liquidity": daily_log_liquidity.astype(np.float32, copy=False),
        "daily_tick_change_1d": daily_tick_change_1d.astype(np.float32, copy=False),
        "daily_log_liquidity_change_1d": daily_log_liquidity_change_1d.astype(np.float32, copy=False),
    }, int(len(df))


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
        split_info["timestamp_start"] = dense_start_time + split_info["sample_start"] * BAR_SECONDS
        split_info["timestamp_stop"] = dense_start_time + (split_info["sample_stop"] - 1) * BAR_SECONDS

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
                "train_timestamp_start": dense_start_time + int(min_sample_end) * BAR_SECONDS,
                "train_timestamp_stop": dense_start_time + (int(fold_train_stop) - 1) * BAR_SECONDS,
                "val_timestamp_start": dense_start_time + int(fold_start) * BAR_SECONDS,
                "val_timestamp_stop": dense_start_time + (int(fold_stop) - 1) * BAR_SECONDS,
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
    minutes_since_observed = dense_bars["minutes_since_observed"]
    swap_count = np.maximum(dense_bars["swap_count"], 0.0)
    swap_amount_in_sum = np.maximum(dense_bars["swap_amount_in_sum"], 0.0)
    swap_amount_out_sum = np.maximum(dense_bars["swap_amount_out_sum"], 0.0)
    swap_amount_in_mean = np.maximum(dense_bars["swap_amount_in_mean"], 0.0)
    swap_amount_out_mean = np.maximum(dense_bars["swap_amount_out_mean"], 0.0)
    swap_amount_in_max = np.maximum(dense_bars["swap_amount_in_max"], 0.0)
    swap_amount_out_max = np.maximum(dense_bars["swap_amount_out_max"], 0.0)
    liq_event_count = np.maximum(dense_bars["liq_event_count"], 0.0)
    liq_delta_abs_sum = np.maximum(dense_bars["liq_delta_abs_sum"], 0.0)
    liq_increase_count = np.maximum(dense_bars["liq_increase_count"], 0.0)
    liq_decrease_count = np.maximum(dense_bars["liq_decrease_count"], 0.0)
    liq_open_count = np.maximum(dense_bars["liq_open_count"], 0.0)
    liq_close_count = np.maximum(dense_bars["liq_close_count"], 0.0)
    liq_reset_range_count = np.maximum(dense_bars["liq_reset_range_count"], 0.0)

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
    yield "log_return_3", lagged_difference(log_close, 3)
    yield "log_return_15", lagged_difference(log_close, 15)
    yield "log_return_60", lagged_difference(log_close, 60)
    yield "body_frac", ((close - open_) / safe_open).astype(np.float32)
    yield "range_frac", ((high - low) / safe_close).astype(np.float32)
    yield "upper_wick_frac", ((high - np.maximum(open_, close)) / safe_close).astype(np.float32)
    yield "lower_wick_frac", ((np.minimum(open_, close) - low) / safe_close).astype(np.float32)

    close_sma_15 = rolling_mean(close, 15)
    close_sma_60 = rolling_mean(close, 60)
    close_sma_300 = rolling_mean(close, 300)
    yield "close_vs_sma_15", ((close / np.maximum(close_sma_15, EPS)) - 1.0).astype(np.float32)
    yield "close_vs_sma_60", ((close / np.maximum(close_sma_60, EPS)) - 1.0).astype(np.float32)
    yield "close_vs_sma_300", ((close / np.maximum(close_sma_300, EPS)) - 1.0).astype(np.float32)

    volume_mean_15 = rolling_mean(log_volume, 15)
    volume_std_15 = rolling_std(log_volume, 15)
    volume_mean_60 = rolling_mean(log_volume, 60)
    volume_std_60 = rolling_std(log_volume, 60)
    yield "volume_zscore_15", safe_zscore(log_volume, volume_mean_15, volume_std_15).astype(np.float32)
    yield "volume_zscore_60", safe_zscore(log_volume, volume_mean_60, volume_std_60).astype(np.float32)
    yield "realized_vol_15", rolling_std(realized_vol_input, 15)
    yield "realized_vol_60", rolling_std(realized_vol_input, 60)
    yield "log_minutes_since_observed", np.log1p(minutes_since_observed).astype(np.float32)
    yield "is_synthetic_bar", (1.0 - observed).astype(np.float32)
    yield "log_swap_count", np.log1p(swap_count).astype(np.float32)
    yield "log_swap_amount_in_sum", np.log1p(swap_amount_in_sum).astype(np.float32)
    yield "log_swap_amount_out_sum", np.log1p(swap_amount_out_sum).astype(np.float32)
    yield "log_swap_amount_in_mean", np.log1p(swap_amount_in_mean).astype(np.float32)
    yield "log_swap_amount_out_mean", np.log1p(swap_amount_out_mean).astype(np.float32)
    yield "log_swap_amount_in_max", np.log1p(swap_amount_in_max).astype(np.float32)
    yield "log_swap_amount_out_max", np.log1p(swap_amount_out_max).astype(np.float32)
    yield "swap_tick_delta_sum", dense_bars["swap_tick_delta_sum"].astype(np.float32)
    yield "swap_tick_delta_abs_sum", dense_bars["swap_tick_delta_abs_sum"].astype(np.float32)
    yield "swap_tick_delta_max_abs", dense_bars["swap_tick_delta_max_abs"].astype(np.float32)
    yield "log_liq_event_count", np.log1p(liq_event_count).astype(np.float32)
    yield "log_liq_delta_abs_sum", np.log1p(liq_delta_abs_sum).astype(np.float32)
    yield "liq_delta_net", dense_bars["liq_delta_net"].astype(np.float32)
    yield "log_liq_increase_count", np.log1p(liq_increase_count).astype(np.float32)
    yield "log_liq_decrease_count", np.log1p(liq_decrease_count).astype(np.float32)
    yield "log_liq_open_count", np.log1p(liq_open_count).astype(np.float32)
    yield "log_liq_close_count", np.log1p(liq_close_count).astype(np.float32)
    yield "log_liq_reset_range_count", np.log1p(liq_reset_range_count).astype(np.float32)
    yield "liq_range_width_mean", dense_bars["liq_range_width_mean"].astype(np.float32)
    yield "daily_tick_current_index", dense_bars["daily_tick_current_index"].astype(np.float32)
    yield "daily_log_liquidity", dense_bars["daily_log_liquidity"].astype(np.float32)
    yield "daily_tick_change_1d", dense_bars["daily_tick_change_1d"].astype(np.float32)
    yield "daily_log_liquidity_change_1d", dense_bars["daily_log_liquidity_change_1d"].astype(np.float32)


def scaler_row_bounds(train_start, train_stop):
    row_start = max(FEATURE_WARMUP_BARS - 1, int(train_start) - LOOKBACK_BARS + 1)
    row_stop = int(train_stop)
    if row_stop <= row_start:
        raise ValueError("Not enough rows for scaler fitting.")
    return row_start, row_stop


def compute_scaler(values, row_start, row_stop):
    train_slice = np.asarray(values[row_start:row_stop], dtype=np.float64)
    finite_values = train_slice[np.isfinite(train_slice)]
    if finite_values.size == 0:
        return 0.0, 1.0
    mean = float(finite_values.mean())
    std = float(finite_values.std())
    if not np.isfinite(std) or std < MIN_STD:
        std = 1.0
    return mean, std


def write_prepared_arrays(dense_bars, splits, validation_folds, output_dir):
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

    train_scaler_row_start, train_scaler_row_stop = scaler_row_bounds(
        splits["train"]["sample_start"], splits["train"]["sample_stop"]
    )
    fold_scaler_bounds = [
        scaler_row_bounds(fold["train_start"], fold["train_stop"]) for fold in validation_folds
    ]

    feature_names = []
    train_scaler = {
        "row_start": int(train_scaler_row_start),
        "row_stop": int(train_scaler_row_stop),
        "mean": [],
        "std": [],
    }
    fold_scalers = [
        {
            "row_start": int(row_start),
            "row_stop": int(row_stop),
            "mean": [],
            "std": [],
        }
        for row_start, row_stop in fold_scaler_bounds
    ]
    features_memmap = np.lib.format.open_memmap(
        feature_path, mode="w+", dtype=np.float32, shape=(num_rows, len(FEATURE_NAMES))
    )
    for feature_idx, (name, values) in enumerate(generate_feature_columns(dense_bars)):
        raw_values = np.asarray(values, dtype=np.float32)
        features_memmap[:, feature_idx] = raw_values
        feature_names.append(name)
        mean, std = compute_scaler(raw_values, train_scaler_row_start, train_scaler_row_stop)
        train_scaler["mean"].append(mean)
        train_scaler["std"].append(std)
        for fold_scaler, (row_start, row_stop) in zip(fold_scalers, fold_scaler_bounds):
            mean, std = compute_scaler(raw_values, row_start, row_stop)
            fold_scaler["mean"].append(mean)
            fold_scaler["std"].append(std)
    features_memmap.flush()
    if tuple(feature_names) != FEATURE_NAMES:
        raise ValueError("Feature generator order drifted from the benchmark feature list.")
    for fold_info, fold_scaler in zip(validation_folds, fold_scalers):
        fold_info["scaler"] = fold_scaler
    return feature_names, train_scaler, validation_folds


def save_metadata(output_dir, metadata):
    metadata_path = os.path.join(output_dir, METADATA_FILENAME)
    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, sort_keys=True)


def prepare_dataset(input_dir=DEFAULT_INPUT_DIR, output_dir=PREPARED_DIR):
    t0 = time.time()
    validate_input_dir(input_dir)

    os.makedirs(output_dir, exist_ok=True)

    source_manifest = load_source_manifest(input_dir)
    dense_bars = load_ohlcv_minutely_bars(input_dir)
    swap_features, num_swap_rows = aggregate_swaps(
        input_dir=input_dir,
        dense_start_time=dense_bars["dense_start_time"],
        num_rows=len(dense_bars["close"]),
    )
    liquidity_features, num_liquidity_event_rows = aggregate_liquidity_events(
        input_dir=input_dir,
        dense_start_time=dense_bars["dense_start_time"],
        num_rows=len(dense_bars["close"]),
    )
    daily_features, num_daily_state_rows = load_daily_state_overlay(
        input_dir=input_dir,
        dense_start_time=dense_bars["dense_start_time"],
        num_rows=len(dense_bars["close"]),
    )
    dense_bars.update(swap_features)
    dense_bars.update(liquidity_features)
    dense_bars.update(daily_features)

    num_dense_rows = len(dense_bars["close"])
    splits, validation_folds, min_sample_end, max_sample_end = build_splits(
        num_dense_rows, dense_bars["dense_start_time"]
    )
    feature_names, train_scaler, validation_folds = write_prepared_arrays(
        dense_bars, splits, validation_folds, output_dir
    )

    abs_input_dir = os.path.abspath(input_dir)

    metadata = {
        "version": PREPARED_VERSION,
        "source_kind": SOURCE_KIND,
        "input_dir": abs_input_dir,
        "prepared_dir": os.path.abspath(output_dir),
        "max_time_budget_seconds": MAX_TIME_BUDGET,
        "primary_metric": PRIMARY_METRIC,
        "primary_metric_direction": PRIMARY_METRIC_DIRECTION,
        "bar_seconds": BAR_SECONDS,
        "lookback_bars": LOOKBACK_BARS,
        "horizon_bars": HORIZON_BARS,
        "purge_bars": PURGE_BARS,
        "split_strategy": SPLIT_STRATEGY,
        "val_ratio": VAL_RATIO,
        "test_ratio": TEST_RATIO,
        "num_val_folds": NUM_VAL_FOLDS,
        "feature_scaling": "per_fold_standardize",
        "eval_samples": EVAL_SAMPLES,
        "feature_windows": list(FEATURE_WINDOWS),
        "feature_warmup_bars": FEATURE_WARMUP_BARS,
        "feature_names": feature_names,
        "train_scaler": train_scaler,
        "dense_start_time": dense_bars["dense_start_time"],
        "dense_end_time": dense_bars["dense_end_time"],
        "num_raw_rows": dense_bars["num_raw_rows"],
        "num_dense_rows": int(num_dense_rows),
        "num_observed_rows": int(np.count_nonzero(dense_bars["observed"])),
        "num_synthetic_rows": int(num_dense_rows - np.count_nonzero(dense_bars["observed"])),
        "num_swap_rows": int(num_swap_rows),
        "num_liquidity_event_rows": int(num_liquidity_event_rows),
        "num_daily_state_rows": int(num_daily_state_rows),
        "sample_end_start": min_sample_end,
        "sample_end_stop": max_sample_end,
        "splits": splits,
        "validation_folds": validation_folds,
        "source_manifest": source_manifest,
        "source_files": {
            METADATA_FILENAME: os.path.join(abs_input_dir, METADATA_FILENAME),
            OHLCV_FILENAME: os.path.join(abs_input_dir, OHLCV_FILENAME),
            SWAPS_FILENAME: os.path.join(abs_input_dir, SWAPS_FILENAME),
            LIQUIDITY_EVENTS_FILENAME: os.path.join(abs_input_dir, LIQUIDITY_EVENTS_FILENAME),
            DAILY_STATE_FILENAME: os.path.join(abs_input_dir, DAILY_STATE_FILENAME),
        },
    }
    save_metadata(output_dir, metadata)

    elapsed = time.time() - t0
    print(f"Prepared dataset in {elapsed:.1f}s")
    print(f"Input dir: {abs_input_dir}")
    print(f"Prepared dir: {os.path.abspath(output_dir)}")
    print(f"Observed bars: {metadata['num_observed_rows']:,}")
    print(f"Synthetic bars: {metadata['num_synthetic_rows']:,}")
    print(
        "Source rows: "
        f"ohlcv={metadata['num_raw_rows']:,}, "
        f"swaps={metadata['num_swap_rows']:,}, "
        f"liquidity_events={metadata['num_liquidity_event_rows']:,}, "
        f"daily_state={metadata['num_daily_state_rows']:,}"
    )
    print(f"Features: {len(feature_names)} -> {', '.join(feature_names)}")
    print(summarize_split("train", splits["train"]))
    print(summarize_split("val", splits["val"]))
    print(summarize_split("test", splits["test"]))
    return metadata


# ---------------------------------------------------------------------------
# Runtime helpers for train.py
# ---------------------------------------------------------------------------

def scaler_arrays_from_info(scaler_info):
    mean = np.asarray(scaler_info["mean"], dtype=np.float32)
    std = np.asarray(scaler_info["std"], dtype=np.float32)
    return mean, std


def build_window_batch(dataset, endpoints, device, scaler_mean=None, scaler_std=None):
    row_idx = endpoints[:, None] - dataset.lookback_bars + 1 + dataset.window_offsets[None, :]
    x_np = np.asarray(dataset.features[row_idx], dtype=np.float32)
    if scaler_mean is not None and scaler_std is not None:
        x_np = (x_np - scaler_mean[None, None, :]) / scaler_std[None, None, :]
    y_np = np.asarray(dataset.targets[endpoints], dtype=np.float32)

    np.nan_to_num(x_np, copy=False, nan=0.0, posinf=0.0, neginf=0.0)

    x = torch.from_numpy(x_np)
    y = torch.from_numpy(y_np)
    if device.type == "cuda":
        x = x.pin_memory().to(device, non_blocking=True)
        y = y.pin_memory().to(device, non_blocking=True)
    else:
        x = x.to(device)
        y = y.to(device)
    return x, y


def sample_batch_range(dataset, sample_start, sample_stop, batch_size, device, rng, scaler_mean=None, scaler_std=None):
    start = int(sample_start)
    stop = int(sample_stop)
    endpoints = rng.integers(start, stop, size=batch_size, endpoint=False, dtype=np.int64)
    return build_window_batch(
        dataset,
        endpoints,
        device,
        scaler_mean=scaler_mean,
        scaler_std=scaler_std,
    )


def sample_batch(dataset, split, batch_size, device, rng, scaler_mean=None, scaler_std=None):
    start, stop = dataset.split_bounds(split)
    return sample_batch_range(
        dataset,
        start,
        stop,
        batch_size,
        device,
        rng,
        scaler_mean=scaler_mean,
        scaler_std=scaler_std,
    )


def select_eval_endpoints_range(sample_start, sample_stop, max_samples=None):
    start = int(sample_start)
    stop = int(sample_stop)
    total = stop - start
    limit = total if max_samples is None else min(total, int(max_samples))
    if limit <= 0:
        raise ValueError("Requested evaluation range has no available samples.")
    if limit == total:
        return np.arange(start, stop, dtype=np.int64)
    return np.linspace(start, stop - 1, num=limit, dtype=np.int64)


def select_eval_endpoints(dataset, split, max_samples=None):
    start, stop = dataset.split_bounds(split)
    return select_eval_endpoints_range(start, stop, max_samples=max_samples)


def iter_eval_batches_range(
    dataset,
    sample_start,
    sample_stop,
    batch_size,
    device,
    max_samples=None,
    scaler_mean=None,
    scaler_std=None,
):
    endpoints = select_eval_endpoints_range(sample_start, sample_stop, max_samples=max_samples)
    for start_idx in range(0, len(endpoints), batch_size):
        batch_endpoints = endpoints[start_idx : start_idx + batch_size]
        yield build_window_batch(
            dataset,
            batch_endpoints,
            device,
            scaler_mean=scaler_mean,
            scaler_std=scaler_std,
        )


def iter_eval_batches(dataset, split, batch_size, device, max_samples=None, scaler_mean=None, scaler_std=None):
    start, stop = dataset.split_bounds(split)
    yield from iter_eval_batches_range(
        dataset,
        start,
        stop,
        batch_size,
        device,
        max_samples=max_samples,
        scaler_mean=scaler_mean,
        scaler_std=scaler_std,
    )


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
def predict_regression_range(
    model,
    dataset,
    sample_start,
    sample_stop,
    batch_size,
    device,
    autocast_ctx=None,
    max_samples=None,
    scaler_mean=None,
    scaler_std=None,
):
    ctx = autocast_ctx if autocast_ctx is not None else nullcontext()
    pred_chunks = []
    target_chunks = []
    for x, y in iter_eval_batches_range(
        dataset,
        sample_start,
        sample_stop,
        batch_size,
        device,
        max_samples=max_samples,
        scaler_mean=scaler_mean,
        scaler_std=scaler_std,
    ):
        with ctx:
            preds = model(x)
        pred_chunks.append(preds.float().cpu())
        target_chunks.append(y.float().cpu())
    predictions = torch.cat(pred_chunks).numpy()
    targets = torch.cat(target_chunks).numpy()
    return predictions, targets


@torch.no_grad()
def evaluate_regression_range(
    model,
    dataset,
    sample_start,
    sample_stop,
    batch_size,
    device,
    autocast_ctx=None,
    max_samples=None,
    scaler_mean=None,
    scaler_std=None,
):
    predictions, targets = predict_regression_range(
        model,
        dataset,
        sample_start,
        sample_stop,
        batch_size,
        device,
        autocast_ctx=autocast_ctx,
        max_samples=max_samples,
        scaler_mean=scaler_mean,
        scaler_std=scaler_std,
    )
    return regression_metrics(predictions, targets)


@torch.no_grad()
def evaluate_regression(
    model,
    dataset,
    split,
    batch_size,
    device,
    autocast_ctx=None,
    max_samples=None,
    scaler_mean=None,
    scaler_std=None,
):
    sample_start, sample_stop = dataset.split_bounds(split)
    predictions, targets = predict_regression_range(
        model,
        dataset,
        sample_start,
        sample_stop,
        batch_size,
        device,
        autocast_ctx=autocast_ctx,
        max_samples=max_samples,
        scaler_mean=scaler_mean,
        scaler_std=scaler_std,
    )
    return regression_metrics(predictions, targets)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prepare Orca minute-level features for forecasting.")
    parser.add_argument(
        "--input-dir",
        type=str,
        default=DEFAULT_INPUT_DIR,
        help="Path to the Orca historical parquet directory.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=PREPARED_DIR,
        help="Directory for prepared arrays and metadata.",
    )
    args = parser.parse_args()
    prepare_dataset(input_dir=args.input_dir, output_dir=args.output_dir)
