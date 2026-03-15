# autoresearch

Autonomous CIFAR-10 classifier research in a tiny repo.

The idea: give an AI agent a small but real image classification setup and let it experiment autonomously overnight. It modifies the model/training code, trains for 5 minutes, checks whether validation accuracy improved, keeps or discards the result, and repeats. You wake up to a log of experiments and, ideally, a better CIFAR-10 model.

## How it works

The repo is deliberately kept small and only really has three files that matter:

- **`prepare.py`**: fixed constants, one-time CIFAR-10 prep, fixed train/validation split, dataloader utilities, and evaluation. Not modified during experiments.
- **`train.py`**: the single file the agent edits. Contains the full model, optimizer, schedules, and training loop.
- **`program.md`**: baseline instructions for an autonomous research agent. This is the human-edited playbook that defines the experiment loop.

By design, training runs for a **fixed 5-minute time budget** (wall clock training time, excluding startup/compilation overhead). The primary metric is **`val_acc`** on a pinned validation split from the CIFAR-10 training set, so higher is better. Validation loss is the tie-breaker. The official CIFAR-10 test split is reserved for final reporting and should not be used in the autonomous keep/discard loop.

## Quick start

**Requirements:** A Mac with Python 3.10+ and [uv](https://docs.astral.sh/uv/). Apple Silicon with `mps` is the primary target, and CPU fallback is supported for slower smoke-test runs.

```bash
# 1. Install dependencies
uv sync

# 2. Download CIFAR-10 and create the fixed split (one-time)
uv run prepare.py

# 3. Manually run a single training experiment (~5 min)
uv run train.py
```

If those commands work, the repo is ready for autonomous experimentation.

## Running the agent

Point your coding agent at this repo and `program.md`, then prompt it with something like:

```text
Read program.md and let's start a new CIFAR-10 experiment run. Do the setup first.
```

The `program.md` file is effectively a lightweight skill that tells the agent how to branch, run experiments, compare results, and log outcomes.

## Project structure

```text
prepare.py       fixed CIFAR-10 harness (do not modify during experiments)
train.py         model, optimizer, training loop (agent modifies this)
program.md       autonomous research instructions
analysis.ipynb   results.tsv analysis and plots
pyproject.toml   dependencies
```

## Design choices

- **Single file to modify.** The agent only edits `train.py`, which keeps diffs small and reviewable.
- **Fixed time budget.** Every experiment gets the same 5-minute wall clock budget, so architecture and optimization changes are compared under the same constraint.
- **Fixed split and metric.** `prepare.py` defines the deterministic train/validation split and evaluation harness so experiments do not silently drift onto a different target.
- **Small, self-contained setup.** One dataset, one local device, one editable training script, one scoreboard.

## Current baseline

The baseline in `train.py` is a compact ResNet-style image classifier for 32x32 RGB images. It uses:

- GroupNorm-based residual blocks
- Standard CIFAR-10 augmentation from the fixed harness
- SGD with momentum
- Mac-first execution on `mps` when available, with CPU fallback
- A fixed-time training loop with gradient accumulation

The baseline is intentionally simple so the autonomous loop can explore improvements in architecture, optimization, and schedule from a clean starting point.

## Analysis

`analysis.ipynb` reads `results.tsv`, summarizes keep/discard/crash outcomes, and plots validation accuracy over time.

## Platform support

This repo is Mac-first. On Apple Silicon it prefers the `mps` backend; on other Macs it falls back to CPU. The training harness is intentionally conservative on macOS: DataLoader workers default to a safer local setup, and the baseline runs in full precision on Mac for stability.

## License

MIT
