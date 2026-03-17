"""
Autoresearch Orca minute-level regression with a causal TCN.

Usage:
    uv run train.py --time-budget-seconds 600
    uv run train.py --time-budget-seconds 1200
"""

from __future__ import annotations

import argparse
import gc
import math
import os
import time
from contextlib import nullcontext
from dataclasses import asdict, dataclass

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from prepare import (
    BAR_SECONDS,
    DEFAULT_INPUT_DIR,
    EVAL_SAMPLES,
    HORIZON_BARS,
    LOOKBACK_BARS,
    MAX_TIME_BUDGET,
    NUM_VAL_FOLDS,
    PREPARED_DIR,
    PREPARED_VERSION,
    PRIMARY_METRIC,
    PURGE_BARS,
    SPLIT_STRATEGY,
    SOURCE_KIND,
    load_prepared_dataset,
    predict_regression_range,
    regression_metrics,
    sample_batch_range,
    scaler_arrays_from_info,
)

os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"

# ---------------------------------------------------------------------------
# Model and training knobs
# ---------------------------------------------------------------------------

# Model
HIDDEN_SIZE = 128
NUM_LAYERS = 5
DROPOUT = 0.10
KERNEL_SIZE = 7
NORM_GROUPS = 8

# Optimization
DEVICE_BATCH_SIZE = 512
TOTAL_BATCH_SIZE = 2048
LEARNING_RATE = 2e-4
WEIGHT_DECAY = 1e-4
ADAM_BETAS = (0.9, 0.95)
WARMUP_RATIO = 0.05
FINAL_LR_FRAC = 0.10
GRAD_CLIP_NORM = 1.0
LOSS_NAME = "huber"  # "huber" or "mse"
HUBER_DELTA = 0.01
CORR_LOSS_WEIGHT = 3e-4
SEED = 42


# ---------------------------------------------------------------------------
# Configs
# ---------------------------------------------------------------------------

@dataclass
class ModelConfig:
    input_dim: int
    hidden_size: int = HIDDEN_SIZE
    num_layers: int = NUM_LAYERS
    dropout: float = DROPOUT
    kernel_size: int = KERNEL_SIZE
    norm_groups: int = NORM_GROUPS


@dataclass
class TrainConfig:
    device_batch_size: int = DEVICE_BATCH_SIZE
    total_batch_size: int = TOTAL_BATCH_SIZE
    learning_rate: float = LEARNING_RATE
    weight_decay: float = WEIGHT_DECAY
    adam_betas: tuple = ADAM_BETAS
    warmup_ratio: float = WARMUP_RATIO
    final_lr_frac: float = FINAL_LR_FRAC
    grad_clip_norm: float = GRAD_CLIP_NORM
    loss_name: str = LOSS_NAME
    huber_delta: float = HUBER_DELTA
    corr_loss_weight: float = CORR_LOSS_WEIGHT
    seed: int = SEED


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

def make_group_norm(num_channels, max_groups):
    num_groups = min(max_groups, num_channels)
    while num_groups > 1 and num_channels % num_groups != 0:
        num_groups -= 1
    return nn.GroupNorm(num_groups, num_channels)


class ResidualTemporalBlock(nn.Module):
    def __init__(self, channels, kernel_size, dilation, dropout, norm_groups):
        super().__init__()
        padding = (kernel_size - 1) * dilation
        self.trim = padding
        self.conv1 = nn.Conv1d(
            channels,
            channels,
            kernel_size=kernel_size,
            dilation=dilation,
            padding=padding,
        )
        self.norm1 = make_group_norm(channels, norm_groups)
        self.conv2 = nn.Conv1d(
            channels,
            channels,
            kernel_size=kernel_size,
            dilation=dilation,
            padding=padding,
        )
        self.norm2 = make_group_norm(channels, norm_groups)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        residual = x
        x = self.conv1(x)
        if self.trim > 0:
            x = x[:, :, :-self.trim]
        x = self.norm1(x)
        x = F.gelu(x)
        x = self.dropout(x)

        x = self.conv2(x)
        if self.trim > 0:
            x = x[:, :, :-self.trim]
        x = self.norm2(x)
        x = self.dropout(x)
        return F.gelu(x + residual)


class TCNRegressor(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.input_norm = nn.LayerNorm(config.input_dim)
        self.input_proj = nn.Conv1d(config.input_dim, config.hidden_size, kernel_size=1)
        self.input_proj_norm = make_group_norm(config.hidden_size, config.norm_groups)
        self.blocks = nn.ModuleList(
            [
                ResidualTemporalBlock(
                    channels=config.hidden_size,
                    kernel_size=config.kernel_size,
                    dilation=2**layer_idx,
                    dropout=config.dropout,
                    norm_groups=config.norm_groups,
                )
                for layer_idx in range(config.num_layers)
            ]
        )
        summary_dim = config.hidden_size * 2
        self.head = nn.Sequential(
            nn.LayerNorm(summary_dim),
            nn.Linear(summary_dim, config.hidden_size),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.hidden_size, 1),
        )
        self.reset_parameters()

    def reset_parameters(self):
        for module in self.modules():
            if isinstance(module, nn.Conv1d):
                nn.init.kaiming_normal_(module.weight, nonlinearity="linear")
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.zeros_(module.bias)

    def forward(self, x):
        x = self.input_norm(x)
        x = x.transpose(1, 2)
        x = self.input_proj(x)
        x = self.input_proj_norm(x)
        x = F.gelu(x)
        for block in self.blocks:
            x = block(x)
        last_state = x[:, :, -1]
        pooled_state = x.mean(dim=2)
        summary = torch.cat((last_state, pooled_state), dim=1)
        prediction = self.head(summary)
        return prediction.squeeze(-1)


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def resolve_device(device_arg):
    if device_arg != "auto":
        return torch.device(device_arg)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def get_autocast_context(device):
    if device.type == "cuda":
        return torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16)
    return nullcontext()


def seed_everything(seed, device):
    torch.manual_seed(seed)
    np.random.seed(seed)
    if device.type == "cuda":
        torch.cuda.manual_seed(seed)


def build_model(model_config):
    return TCNRegressor(model_config)


def build_optimizer(model, train_config):
    return torch.optim.AdamW(
        model.parameters(),
        lr=train_config.learning_rate,
        betas=train_config.adam_betas,
        weight_decay=train_config.weight_decay,
    )


def get_lr_multiplier(progress, train_config):
    if progress < train_config.warmup_ratio:
        if train_config.warmup_ratio == 0:
            return 1.0
        return progress / train_config.warmup_ratio
    decay_progress = (progress - train_config.warmup_ratio) / max(1e-8, 1.0 - train_config.warmup_ratio)
    decay_progress = min(max(decay_progress, 0.0), 1.0)
    cosine = 0.5 * (1.0 + math.cos(math.pi * decay_progress))
    return train_config.final_lr_frac + (1.0 - train_config.final_lr_frac) * cosine


def compute_loss(predictions, targets, train_config):
    if train_config.loss_name == "mse":
        base_loss = F.mse_loss(predictions, targets)
    elif train_config.loss_name == "huber":
        base_loss = F.huber_loss(predictions, targets, delta=train_config.huber_delta)
    else:
        raise ValueError(f"Unsupported loss: {train_config.loss_name}")

    if train_config.corr_loss_weight <= 0:
        return base_loss

    predictions32 = predictions.float()
    targets32 = targets.float()
    pred_centered = predictions32 - predictions32.mean()
    target_centered = targets32 - targets32.mean()
    denom = torch.sqrt(
        torch.clamp(pred_centered.square().mean() * target_centered.square().mean(), min=1e-12)
    )
    corr = (pred_centered * target_centered).mean() / denom
    return base_loss + train_config.corr_loss_weight * (1.0 - corr)


def allocate_fold_budgets(total_budget_seconds, num_folds):
    if total_budget_seconds < num_folds:
        raise ValueError(
            f"time budget {total_budget_seconds}s must be at least {num_folds}s for walk-forward validation."
        )
    base_budget = total_budget_seconds // num_folds
    remainder = total_budget_seconds % num_folds
    return [base_budget + (1 if fold_idx < remainder else 0) for fold_idx in range(num_folds)]


def allocate_eval_limits(validation_folds, max_eval_samples):
    total_val_samples = sum(int(fold["val_stop"]) - int(fold["val_start"]) for fold in validation_folds)
    if max_eval_samples is None or max_eval_samples >= total_val_samples:
        return [None] * len(validation_folds)
    if max_eval_samples < len(validation_folds):
        raise ValueError(
            f"eval_samples {max_eval_samples} must be at least the number of validation folds "
            f"({len(validation_folds)})."
        )
    base_limit = max_eval_samples // len(validation_folds)
    remainder = max_eval_samples % len(validation_folds)
    return [base_limit + (1 if fold_idx < remainder else 0) for fold_idx in range(len(validation_folds))]


def count_parameters(model):
    return sum(param.numel() for param in model.parameters())


def get_primary_metric_value(metrics):
    metric_key = {
        "val_corr": "corr",
        "val_rmse": "rmse",
        "val_mae": "mae",
        "val_sign_acc": "sign_accuracy",
    }[PRIMARY_METRIC]
    return float(metrics[metric_key])


def resolve_time_budget_seconds(requested_time_budget_seconds):
    time_budget_seconds = int(requested_time_budget_seconds)
    if time_budget_seconds <= 0:
        raise ValueError("time budget must be positive.")
    if time_budget_seconds > MAX_TIME_BUDGET:
        raise ValueError(
            f"time budget {time_budget_seconds}s exceeds max allowed {MAX_TIME_BUDGET}s."
        )
    return time_budget_seconds


def validate_prepared_input(dataset):
    actual_version = dataset.metadata.get("version")
    actual_source_kind = dataset.metadata.get("source_kind")
    actual_input_dir = dataset.metadata.get("input_dir")
    actual_bar_seconds = dataset.metadata.get("bar_seconds")
    actual_lookback_bars = dataset.metadata.get("lookback_bars")
    actual_horizon_bars = dataset.metadata.get("horizon_bars")
    actual_purge_bars = dataset.metadata.get("purge_bars")
    actual_split_strategy = dataset.metadata.get("split_strategy")
    actual_num_val_folds = dataset.metadata.get("num_val_folds")
    actual_validation_folds = dataset.metadata.get("validation_folds") or []

    mismatches = []
    if actual_version != PREPARED_VERSION:
        mismatches.append(f"version expected {PREPARED_VERSION} but found {actual_version!r}")
    if actual_source_kind != SOURCE_KIND:
        mismatches.append(f"source_kind expected {SOURCE_KIND!r} but found {actual_source_kind!r}")

    expected_input_dir = os.path.abspath(DEFAULT_INPUT_DIR)
    normalized_actual_input_dir = None if not actual_input_dir else os.path.abspath(actual_input_dir)
    if normalized_actual_input_dir != expected_input_dir:
        mismatches.append(
            f"input_dir expected {expected_input_dir!r} but found {normalized_actual_input_dir!r}"
        )

    if actual_bar_seconds != BAR_SECONDS:
        mismatches.append(f"bar_seconds expected {BAR_SECONDS} but found {actual_bar_seconds!r}")
    if actual_lookback_bars != LOOKBACK_BARS:
        mismatches.append(f"lookback_bars expected {LOOKBACK_BARS} but found {actual_lookback_bars!r}")
    if actual_horizon_bars != HORIZON_BARS:
        mismatches.append(f"horizon_bars expected {HORIZON_BARS} but found {actual_horizon_bars!r}")
    if actual_purge_bars != PURGE_BARS:
        mismatches.append(f"purge_bars expected {PURGE_BARS} but found {actual_purge_bars!r}")
    if actual_split_strategy != SPLIT_STRATEGY:
        mismatches.append(
            f"split_strategy expected {SPLIT_STRATEGY!r} but found {actual_split_strategy!r}"
        )
    if actual_num_val_folds != NUM_VAL_FOLDS:
        mismatches.append(f"num_val_folds expected {NUM_VAL_FOLDS} but found {actual_num_val_folds!r}")
    if len(actual_validation_folds) != NUM_VAL_FOLDS:
        mismatches.append(
            f"validation_folds expected {NUM_VAL_FOLDS} entries but found {len(actual_validation_folds)!r}"
        )

    if mismatches:
        details = "\n".join(f"- {item}" for item in mismatches)
        raise RuntimeError(
            "Prepared artifacts were built for a different data source or benchmark contract.\n"
            f"{details}\n"
            "Run `uv run prepare.py` to rebuild the prepared dataset."
        )


def maybe_sync(device):
    if device.type == "cuda":
        torch.cuda.synchronize()
    elif device.type == "mps" and hasattr(torch, "mps") and hasattr(torch.mps, "synchronize"):
        torch.mps.synchronize()


def train_one_run(
    model,
    dataset,
    optimizer,
    train_config,
    device,
    time_budget_seconds,
    sample_start,
    sample_stop,
    scaler_mean=None,
    scaler_std=None,
    progress_label="train",
):
    if train_config.total_batch_size % train_config.device_batch_size != 0:
        raise ValueError("TOTAL_BATCH_SIZE must be divisible by DEVICE_BATCH_SIZE.")

    grad_accum_steps = train_config.total_batch_size // train_config.device_batch_size
    rng = np.random.default_rng(train_config.seed)
    smooth_train_loss = 0.0
    total_training_time = 0.0
    step = 0
    debiased_smooth_loss = float("nan")
    last_train_loss_value = float("nan")
    last_grad_norm_value = float("nan")
    model.train()

    while True:
        maybe_sync(device)
        t0 = time.time()
        progress = min(total_training_time / time_budget_seconds, 1.0)
        lr_multiplier = get_lr_multiplier(progress, train_config)
        optimizer.param_groups[0]["lr"] = train_config.learning_rate * lr_multiplier
        optimizer.zero_grad(set_to_none=True)

        for _ in range(grad_accum_steps):
            x, y = sample_batch_range(
                dataset,
                sample_start,
                sample_stop,
                train_config.device_batch_size,
                device,
                rng,
                scaler_mean=scaler_mean,
                scaler_std=scaler_std,
            )
            with get_autocast_context(device):
                predictions = model(x)
                loss = compute_loss(predictions, y, train_config)
            train_loss = loss.detach()
            (loss / grad_accum_steps).backward()

        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), train_config.grad_clip_norm)
        optimizer.step()

        maybe_sync(device)
        dt = time.time() - t0
        total_training_time += dt
        step += 1

        train_loss_value = float(train_loss.item())
        grad_norm_value = float(grad_norm)
        last_train_loss_value = train_loss_value
        last_grad_norm_value = grad_norm_value
        if not math.isfinite(train_loss_value) or not math.isfinite(grad_norm_value) or train_loss_value > 100:
            raise RuntimeError("Training diverged.")

        ema_beta = 0.9
        smooth_train_loss = ema_beta * smooth_train_loss + (1.0 - ema_beta) * train_loss_value
        debiased_smooth_loss = smooth_train_loss / (1.0 - ema_beta ** step)
        samples_per_second = int(train_config.total_batch_size / max(dt, 1e-6))
        remaining = max(0.0, time_budget_seconds - total_training_time)
        pct_done = 100.0 * progress
        print(
            f"\r{progress_label} step {step:05d} ({pct_done:5.1f}%) | "
            f"loss: {debiased_smooth_loss:.6f} | "
            f"lr_mult: {lr_multiplier:.3f} | "
            f"grad_norm: {grad_norm_value:.3f} | "
            f"samples/s: {samples_per_second:,} | "
            f"remaining: {remaining:6.1f}s    ",
            end="",
            flush=True,
        )

        if step == 1:
            gc.collect()
            gc.freeze()
            gc.disable()
        elif step % 1000 == 0:
            gc.collect()

        if total_training_time >= time_budget_seconds:
            break

    print()
    total_windows = step * train_config.total_batch_size
    return {
        "training_seconds": total_training_time,
        "num_steps": step,
        "total_windows": total_windows,
        "avg_samples_per_second": total_windows / max(total_training_time, 1e-6),
        "last_train_loss": last_train_loss_value,
        "smoothed_train_loss": debiased_smooth_loss,
        "last_grad_norm": last_grad_norm_value,
        "grad_accum_steps": grad_accum_steps,
    }


def run_walk_forward_validation(model_config, dataset, train_config, device, time_budget_seconds):
    validation_folds = dataset.metadata["validation_folds"]
    fold_budgets = allocate_fold_budgets(time_budget_seconds, len(validation_folds))
    eval_limits = allocate_eval_limits(validation_folds, EVAL_SAMPLES)

    fold_summaries = []
    prediction_chunks = []
    target_chunks = []
    total_training_seconds = 0.0
    total_eval_seconds = 0.0
    total_windows = 0
    total_steps = 0
    final_run_stats = None

    for fold_idx, (fold_info, fold_budget, eval_limit) in enumerate(
        zip(validation_folds, fold_budgets, eval_limits)
    ):
        fold_label = f"fold {fold_idx + 1}/{len(validation_folds)}"
        train_samples = int(fold_info["train_stop"]) - int(fold_info["train_start"])
        val_samples = int(fold_info["val_stop"]) - int(fold_info["val_start"])
        print(
            f"{fold_label}: train {train_samples:,} samples | "
            f"val {val_samples:,} samples | budget {fold_budget}s"
        )

        seed_everything(train_config.seed, device)
        model = build_model(model_config).to(device)
        optimizer = build_optimizer(model, train_config)
        scaler_mean, scaler_std = scaler_arrays_from_info(fold_info["scaler"])

        fold_run_stats = train_one_run(
            model,
            dataset,
            optimizer,
            train_config,
            device,
            fold_budget,
            sample_start=fold_info["train_start"],
            sample_stop=fold_info["train_stop"],
            scaler_mean=scaler_mean,
            scaler_std=scaler_std,
            progress_label=fold_label,
        )

        t_eval_start = time.time()
        predictions, targets = predict_regression_range(
            model,
            dataset,
            fold_info["val_start"],
            fold_info["val_stop"],
            train_config.device_batch_size,
            device,
            autocast_ctx=get_autocast_context(device),
            max_samples=eval_limit,
            scaler_mean=scaler_mean,
            scaler_std=scaler_std,
        )
        fold_eval_seconds = time.time() - t_eval_start
        fold_metrics = regression_metrics(predictions, targets)
        print(
            f"{fold_label} val_corr: {fold_metrics['corr']:.6f} | "
            f"val_rmse: {fold_metrics['rmse']:.6f} | "
            f"val_mae: {fold_metrics['mae']:.6f} | "
            f"val_sign_acc: {fold_metrics['sign_accuracy']:.6f}"
        )

        fold_summaries.append(
            {
                "fold": int(fold_info["fold"]),
                "train_samples": train_samples,
                "val_samples": val_samples,
                "eval_samples": int(len(targets)),
                "budget_seconds": int(fold_budget),
                "training_seconds": float(fold_run_stats["training_seconds"]),
                "eval_seconds": float(fold_eval_seconds),
                "metrics": fold_metrics,
            }
        )
        prediction_chunks.append(predictions)
        target_chunks.append(targets)
        total_training_seconds += float(fold_run_stats["training_seconds"])
        total_eval_seconds += float(fold_eval_seconds)
        total_windows += int(fold_run_stats["total_windows"])
        total_steps += int(fold_run_stats["num_steps"])
        final_run_stats = fold_run_stats

        del model, optimizer
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()

    all_predictions = np.concatenate(prediction_chunks)
    all_targets = np.concatenate(target_chunks)
    aggregate_metrics = regression_metrics(all_predictions, all_targets)
    aggregate_stats = {
        "training_seconds": total_training_seconds,
        "num_steps": total_steps,
        "total_windows": total_windows,
        "avg_samples_per_second": total_windows / max(total_training_seconds, 1e-6),
        "last_train_loss": final_run_stats["last_train_loss"],
        "smoothed_train_loss": final_run_stats["smoothed_train_loss"],
        "last_grad_norm": final_run_stats["last_grad_norm"],
        "grad_accum_steps": final_run_stats["grad_accum_steps"],
        "fold_budgets": fold_budgets,
    }
    return aggregate_metrics, aggregate_stats, fold_summaries, total_eval_seconds


def main():
    parser = argparse.ArgumentParser(description="Train a causal TCN on prepared Orca minute time-series data.")
    parser.add_argument(
        "--prepared-dir",
        type=str,
        default=PREPARED_DIR,
        help="Directory containing features.npy, targets.npy, and metadata.json.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        help="Device override, e.g. cuda, cpu, or mps.",
    )
    parser.add_argument(
        "--time-budget-seconds",
        type=int,
        required=True,
        help=(
            "Training time budget in seconds. "
            f"Required for each run, max {MAX_TIME_BUDGET}."
        ),
    )
    args = parser.parse_args()

    t_start = time.time()
    time_budget_seconds = resolve_time_budget_seconds(args.time_budget_seconds)
    device = resolve_device(args.device)
    seed_everything(SEED, device)
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True
        torch.cuda.reset_peak_memory_stats()
    torch.set_float32_matmul_precision("high")

    dataset = load_prepared_dataset(args.prepared_dir)
    validate_prepared_input(dataset)
    model_config = ModelConfig(input_dim=dataset.input_dim)
    train_config = TrainConfig()
    probe_model = build_model(model_config)
    num_params = count_parameters(probe_model)
    del probe_model

    print(f"Device: {device}")
    print(f"Prepared dir: {os.path.abspath(args.prepared_dir)}")
    print(f"Primary metric: {PRIMARY_METRIC}")
    print(f"Time budget seconds: {time_budget_seconds}")
    print(f"Feature dim: {dataset.input_dim}")
    print(f"Lookback bars: {dataset.metadata['lookback_bars']}")
    print(f"Split strategy: {dataset.metadata['split_strategy']}")
    print(f"Validation folds: {dataset.metadata['num_val_folds']}")
    print(f"Model config: {asdict(model_config)}")
    print(f"Train config: {asdict(train_config)}")

    t_training_start = time.time()
    val_metrics, run_stats, fold_summaries, eval_seconds = run_walk_forward_validation(
        model_config,
        dataset,
        train_config,
        device,
        time_budget_seconds,
    )

    t_end = time.time()
    peak_vram_mb = (
        torch.cuda.max_memory_allocated() / 1024 / 1024 if device.type == "cuda" else 0.0
    )
    startup_seconds = t_training_start - t_start
    primary_value = get_primary_metric_value(val_metrics)
    splits = dataset.metadata["splits"]
    eval_samples = sum(fold_summary["eval_samples"] for fold_summary in fold_summaries)
    max_train_samples = max(fold_summary["train_samples"] for fold_summary in fold_summaries)

    print("---")
    print(f"primary_metric:    {PRIMARY_METRIC}")
    print(f"primary_value:     {primary_value:.6f}")
    print(f"val_corr:          {val_metrics['corr']:.6f}")
    print(f"val_rmse:          {val_metrics['rmse']:.6f}")
    print(f"val_mae:           {val_metrics['mae']:.6f}")
    print(f"val_sign_acc:      {val_metrics['sign_accuracy']:.6f}")
    print(f"last_train_loss:   {run_stats['last_train_loss']:.6f}")
    print(f"smooth_train_loss: {run_stats['smoothed_train_loss']:.6f}")
    print(f"last_grad_norm:    {run_stats['last_grad_norm']:.6f}")
    print(f"time_budget_s:     {time_budget_seconds}")
    print(f"max_time_budget_s: {MAX_TIME_BUDGET}")
    print(f"startup_seconds:   {startup_seconds:.1f}")
    print(f"training_seconds:  {run_stats['training_seconds']:.1f}")
    print(f"eval_seconds:      {eval_seconds:.1f}")
    print(f"total_seconds:     {t_end - t_start:.1f}")
    print(f"avg_samples_sec:   {run_stats['avg_samples_per_second']:.1f}")
    print(f"peak_vram_mb:      {peak_vram_mb:.1f}")
    print(f"total_windows_K:   {run_stats['total_windows'] / 1e3:.1f}")
    print(f"num_steps:         {run_stats['num_steps']}")
    print(f"grad_accum_steps:  {run_stats['grad_accum_steps']}")
    print(f"split_strategy:    {dataset.metadata['split_strategy']}")
    print(f"num_val_folds:     {len(fold_summaries)}")
    print(f"fold_budgets_s:    {','.join(str(budget) for budget in run_stats['fold_budgets'])}")
    print(f"device_batch_size: {train_config.device_batch_size}")
    print(f"total_batch_size:  {train_config.total_batch_size}")
    print(f"num_params_M:      {num_params / 1e6:.3f}")
    print(f"device:            {device}")
    print(f"feature_dim:       {dataset.input_dim}")
    print(f"train_samples:     {splits['train']['num_samples']}")
    print(f"max_train_samples: {max_train_samples}")
    print(f"val_samples:       {splits['val']['num_samples']}")
    print(f"test_samples:      {splits['test']['num_samples']}")
    print(f"eval_samples:      {eval_samples}")
    print(f"observed_bars:     {dataset.metadata['num_observed_rows']}")
    print(f"synthetic_bars:    {dataset.metadata['num_synthetic_rows']}")
    print(f"lookback_bars:     {dataset.metadata['lookback_bars']}")
    print(f"horizon_bars:      {dataset.metadata['horizon_bars']}")
    for fold_summary in fold_summaries:
        fold_name = fold_summary["fold"] + 1
        print(f"fold{fold_name}_val_corr:   {fold_summary['metrics']['corr']:.6f}")


if __name__ == "__main__":
    main()
