"""
Autoresearch time-series regression with a causal TCN.

Usage:
    uv run train.py
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
    DEFAULT_INPUT_CSV,
    EVAL_SAMPLES,
    MAX_TIME_BUDGET,
    PREPARED_DIR,
    PRIMARY_METRIC,
    TIME_BUDGET,
    evaluate_regression,
    load_prepared_dataset,
    sample_batch,
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
CORR_LOSS_WEIGHT = 0.0
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


def evaluate(model, dataset, batch_size, device):
    model.eval()
    metrics = evaluate_regression(
        model,
        dataset,
        split="val",
        batch_size=batch_size,
        device=device,
        autocast_ctx=get_autocast_context(device),
        max_samples=EVAL_SAMPLES,
    )
    model.train()
    return metrics


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
    expected_input_csv = os.path.abspath(DEFAULT_INPUT_CSV)
    actual_input_csv = os.path.abspath(dataset.metadata.get("input_csv", ""))
    if actual_input_csv != expected_input_csv:
        raise RuntimeError(
            "Prepared artifacts were built from a different CSV.\n"
            f"Expected: {expected_input_csv}\n"
            f"Found:    {actual_input_csv}\n"
            "Run `uv run prepare.py` to rebuild the prepared dataset."
        )


def maybe_sync(device):
    if device.type == "cuda":
        torch.cuda.synchronize()
    elif device.type == "mps" and hasattr(torch, "mps") and hasattr(torch.mps, "synchronize"):
        torch.mps.synchronize()


def train_one_run(model, dataset, optimizer, train_config, device, time_budget_seconds):
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
            x, y = sample_batch(dataset, "train", train_config.device_batch_size, device, rng)
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
            f"\rstep {step:05d} ({pct_done:5.1f}%) | "
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


def main():
    parser = argparse.ArgumentParser(description="Train a causal TCN on prepared SOL time-series data.")
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
        default=TIME_BUDGET,
        help=f"Training time budget in seconds. Default {TIME_BUDGET}, max {MAX_TIME_BUDGET}.",
    )
    args = parser.parse_args()

    t_start = time.time()
    time_budget_seconds = resolve_time_budget_seconds(args.time_budget_seconds)
    device = resolve_device(args.device)
    torch.manual_seed(SEED)
    if device.type == "cuda":
        torch.cuda.manual_seed(SEED)
        torch.backends.cudnn.benchmark = True
    torch.set_float32_matmul_precision("high")

    dataset = load_prepared_dataset(args.prepared_dir)
    validate_prepared_input(dataset)
    model_config = ModelConfig(input_dim=dataset.input_dim)
    train_config = TrainConfig()

    print(f"Device: {device}")
    print(f"Prepared dir: {os.path.abspath(args.prepared_dir)}")
    print(f"Primary metric: {PRIMARY_METRIC}")
    print(f"Time budget seconds: {time_budget_seconds}")
    print(f"Feature dim: {dataset.input_dim}")
    print(f"Lookback bars: {dataset.metadata['lookback_bars']}")
    print(f"Model config: {asdict(model_config)}")
    print(f"Train config: {asdict(train_config)}")

    model = build_model(model_config).to(device)
    optimizer = build_optimizer(model, train_config)
    num_params = count_parameters(model)

    t_training_start = time.time()
    run_stats = train_one_run(model, dataset, optimizer, train_config, device, time_budget_seconds)
    t_eval_start = time.time()
    val_metrics = evaluate(model, dataset, train_config.device_batch_size, device)
    eval_seconds = time.time() - t_eval_start

    t_end = time.time()
    peak_vram_mb = (
        torch.cuda.max_memory_allocated() / 1024 / 1024 if device.type == "cuda" else 0.0
    )
    startup_seconds = t_training_start - t_start
    primary_value = get_primary_metric_value(val_metrics)
    splits = dataset.metadata["splits"]
    eval_samples = min(EVAL_SAMPLES, splits["val"]["num_samples"])

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
    print(f"device_batch_size: {train_config.device_batch_size}")
    print(f"total_batch_size:  {train_config.total_batch_size}")
    print(f"num_params_M:      {num_params / 1e6:.3f}")
    print(f"device:            {device}")
    print(f"feature_dim:       {dataset.input_dim}")
    print(f"train_samples:     {splits['train']['num_samples']}")
    print(f"val_samples:       {splits['val']['num_samples']}")
    print(f"test_samples:      {splits['test']['num_samples']}")
    print(f"eval_samples:      {eval_samples}")
    print(f"observed_bars:     {dataset.metadata['num_observed_rows']}")
    print(f"synthetic_bars:    {dataset.metadata['num_synthetic_rows']}")
    print(f"lookback_bars:     {dataset.metadata['lookback_bars']}")
    print(f"horizon_bars:      {dataset.metadata['horizon_bars']}")


if __name__ == "__main__":
    main()
