"""
Autoresearch CIFAR-10 training script. Single-device, single-file.
Usage: uv run train.py
"""

import gc
import math
import time
from contextlib import nullcontext
from dataclasses import asdict, dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from prepare import (
    IMAGE_SIZE,
    NUM_CHANNELS,
    NUM_CLASSES,
    TIME_BUDGET,
    evaluate_classifier,
    get_default_device,
    get_device_name,
    get_peak_memory_mb,
    make_dataloader,
    reset_peak_memory_stats,
    synchronize_device,
)

# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------


@dataclass
class ModelConfig:
    image_size: int = IMAGE_SIZE
    in_channels: int = NUM_CHANNELS
    num_classes: int = NUM_CLASSES
    widths: tuple[int, int, int] = (64, 128, 256)
    blocks_per_stage: tuple[int, int, int] = (2, 2, 2)
    group_norm_groups: int = 8
    dropout: float = 0.0


def make_group_norm(num_channels, max_groups):
    num_groups = min(max_groups, num_channels)
    while num_groups > 1 and num_channels % num_groups != 0:
        num_groups -= 1
    return nn.GroupNorm(num_groups, num_channels)


class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride, group_norm_groups, dropout):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.norm1 = make_group_norm(out_channels, group_norm_groups)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.norm2 = make_group_norm(out_channels, group_norm_groups)
        self.act = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                make_group_norm(out_channels, group_norm_groups),
            )
        else:
            self.shortcut = nn.Identity()

    def forward(self, x):
        residual = self.shortcut(x)
        x = self.conv1(x)
        x = self.norm1(x)
        x = self.act(x)
        x = self.dropout(x)
        x = self.conv2(x)
        x = self.norm2(x)
        x = self.act(x + residual)
        return x


class SmallResNet(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        widths = config.widths
        blocks = config.blocks_per_stage
        groups = config.group_norm_groups

        self.stem = nn.Sequential(
            nn.Conv2d(config.in_channels, widths[0], kernel_size=3, stride=1, padding=1, bias=False),
            make_group_norm(widths[0], groups),
            nn.ReLU(inplace=True),
        )
        self.stage1 = self._make_stage(widths[0], widths[0], blocks[0], stride=1)
        self.stage2 = self._make_stage(widths[0], widths[1], blocks[1], stride=2)
        self.stage3 = self._make_stage(widths[1], widths[2], blocks[2], stride=2)
        self.head_norm = make_group_norm(widths[2], groups)
        self.head_dropout = nn.Dropout(config.dropout) if config.dropout > 0 else nn.Identity()
        self.head = nn.Linear(widths[2], config.num_classes)

    def _make_stage(self, in_channels, out_channels, num_blocks, stride):
        blocks = [
            ResidualBlock(
                in_channels,
                out_channels,
                stride=stride,
                group_norm_groups=self.config.group_norm_groups,
                dropout=self.config.dropout,
            )
        ]
        for _ in range(1, num_blocks):
            blocks.append(
                ResidualBlock(
                    out_channels,
                    out_channels,
                    stride=1,
                    group_norm_groups=self.config.group_norm_groups,
                    dropout=self.config.dropout,
                )
            )
        return nn.Sequential(*blocks)

    def forward(self, x):
        x = self.stem(x)
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.head_norm(x)
        x = F.relu(x, inplace=True)
        x = x.mean(dim=(2, 3))
        x = self.head_dropout(x)
        return self.head(x)


def count_parameters(model):
    return sum(p.numel() for p in model.parameters())


def build_model_config(depth):
    widths = tuple(BASE_WIDTH * (2 ** idx) for idx in range(3))
    blocks_per_stage = (depth, depth, depth)
    return ModelConfig(
        widths=widths,
        blocks_per_stage=blocks_per_stage,
        group_norm_groups=GROUP_NORM_GROUPS,
        dropout=DROPOUT,
    )


def build_autocast_context(device):
    if device.type == "cuda":
        return torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16)
    # MPS mixed precision is still less predictable than CUDA, so default to
    # full precision on Mac for a safer baseline.
    return nullcontext()


def configure_runtime(device):
    torch.manual_seed(42)
    torch.set_float32_matmul_precision("high")
    if device.type == "cuda":
        torch.cuda.manual_seed(42)
        torch.backends.cudnn.benchmark = True


def get_lr_multiplier(progress):
    if progress < WARMUP_RATIO:
        return progress / WARMUP_RATIO if WARMUP_RATIO > 0 else 1.0
    if progress < 1.0 - WARMDOWN_RATIO:
        return 1.0
    cooldown = (1.0 - progress) / WARMDOWN_RATIO
    return cooldown + (1.0 - cooldown) * FINAL_LR_FRAC


# ---------------------------------------------------------------------------
# Hyperparameters
# ---------------------------------------------------------------------------

# Model architecture
BASE_WIDTH = 64
DEPTH = 2  # residual blocks per stage
GROUP_NORM_GROUPS = 8
DROPOUT = 0.0

# Optimization
TOTAL_BATCH_SIZE = 512  # images per optimizer step
DEVICE_BATCH_SIZE = 64  # per-device microbatch size
LEARNING_RATE = 0.3
MOMENTUM = 0.9
WEIGHT_DECAY = 5e-4
LABEL_SMOOTHING = 0.0
WARMUP_RATIO = 0.05
WARMDOWN_RATIO = 0.65
FINAL_LR_FRAC = 0.05

# Runtime
USE_COMPILE = False
TIMING_WARMUP_STEPS = 10


def main():
    device = get_default_device()
    configure_runtime(device)
    autocast_ctx = build_autocast_context(device)
    t_start = time.time()

    print(f"Device: {get_device_name(device)}")

    config = build_model_config(DEPTH)
    print(f"Model config: {asdict(config)}")

    model = SmallResNet(config).to(device)
    num_params = count_parameters(model)
    print(f"Number of parameters: {num_params:,}")

    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=LEARNING_RATE,
        momentum=MOMENTUM,
        weight_decay=WEIGHT_DECAY,
        nesterov=True,
    )

    if USE_COMPILE:
        model = torch.compile(model, dynamic=False)

    assert TOTAL_BATCH_SIZE % DEVICE_BATCH_SIZE == 0
    grad_accum_steps = TOTAL_BATCH_SIZE // DEVICE_BATCH_SIZE

    train_loader = make_dataloader(DEVICE_BATCH_SIZE, "train", augment=True, device=device)
    x, y, epoch = next(train_loader)

    print(f"Time budget: {TIME_BUDGET}s")
    print(f"Gradient accumulation steps: {grad_accum_steps}")

    model.train()
    optimizer.zero_grad(set_to_none=True)
    reset_peak_memory_stats(device)

    # -----------------------------------------------------------------------
    # Training loop
    # -----------------------------------------------------------------------

    smooth_train_loss = 0.0
    total_training_time = 0.0
    step = 0
    peak_memory_mb = get_peak_memory_mb(device)

    def sample_peak_memory():
        nonlocal peak_memory_mb
        peak_memory_mb = max(peak_memory_mb, get_peak_memory_mb(device))

    while True:
        synchronize_device(device)
        t0 = time.time()
        batch_correct = 0
        batch_examples = 0

        for _ in range(grad_accum_steps):
            with autocast_ctx:
                logits = model(x)
                loss = F.cross_entropy(logits, y, label_smoothing=LABEL_SMOOTHING)
            sample_peak_memory()
            train_loss = loss.detach()
            loss = loss / grad_accum_steps
            loss.backward()
            sample_peak_memory()
            batch_correct += (logits.detach().argmax(dim=1) == y).sum().item()
            batch_examples += y.size(0)
            x, y, epoch = next(train_loader)

        progress = min(total_training_time / TIME_BUDGET, 1.0)
        lr = LEARNING_RATE * get_lr_multiplier(progress)
        for group in optimizer.param_groups:
            group["lr"] = lr
        optimizer.step()
        sample_peak_memory()
        optimizer.zero_grad(set_to_none=True)

        train_loss_f = train_loss.item()
        train_acc = batch_correct / batch_examples

        if math.isnan(train_loss_f) or train_loss_f > 100:
            print("FAIL")
            raise SystemExit(1)

        synchronize_device(device)
        t1 = time.time()
        dt = t1 - t0
        sample_peak_memory()

        if step > TIMING_WARMUP_STEPS:
            total_training_time += dt

        ema_beta = 0.9
        smooth_train_loss = ema_beta * smooth_train_loss + (1 - ema_beta) * train_loss_f
        debiased_smooth_loss = smooth_train_loss / (1 - ema_beta ** (step + 1))
        pct_done = 100 * progress
        images_per_sec = int(TOTAL_BATCH_SIZE / dt)
        remaining = max(0.0, TIME_BUDGET - total_training_time)

        print(
            f"\rstep {step:05d} ({pct_done:.1f}%) | "
            f"loss: {debiased_smooth_loss:.4f} | "
            f"acc: {train_acc:.3f} | "
            f"lr: {lr:.4f} | "
            f"dt: {dt * 1000:.0f}ms | "
            f"img/sec: {images_per_sec:,} | "
            f"epoch: {epoch} | "
            f"remaining: {remaining:.0f}s    ",
            end="",
            flush=True,
        )

        if step == 0:
            gc.collect()
            gc.freeze()
            gc.disable()
        elif (step + 1) % 5000 == 0:
            gc.collect()

        step += 1

        if step > TIMING_WARMUP_STEPS and total_training_time >= TIME_BUDGET:
            break

    print()

    total_images = step * TOTAL_BATCH_SIZE

    # Final eval
    val_metrics = evaluate_classifier(model, DEVICE_BATCH_SIZE, split="val", device=device)
    sample_peak_memory()

    # Final summary
    t_end = time.time()
    steady_state_img_per_sec = (
        TOTAL_BATCH_SIZE * max(step - TIMING_WARMUP_STEPS, 0) / total_training_time
        if total_training_time > 0
        else 0.0
    )

    print("---")
    print(f"device:           {get_device_name(device)}")
    print(f"val_acc:          {val_metrics['accuracy']:.6f}")
    print(f"val_loss:         {val_metrics['loss']:.6f}")
    print(f"training_seconds: {total_training_time:.1f}")
    print(f"total_seconds:    {t_end - t_start:.1f}")
    print(f"peak_memory_mb:   {peak_memory_mb:.1f}")
    print(f"images_per_sec:   {steady_state_img_per_sec:.1f}")
    print(f"total_images_k:   {total_images / 1e3:.1f}")
    print(f"num_steps:        {step}")
    print(f"num_params_M:     {num_params / 1e6:.1f}")
    print(f"depth:            {DEPTH}")


if __name__ == "__main__":
    main()
