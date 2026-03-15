# autoresearch

This is an experiment to have the LLM do its own CIFAR-10 research.

## Setup

To set up a new experiment, work with the user to:

1. **Agree on a run tag**: propose a tag based on today's date (for example `mar13`). The branch `autoresearch/<tag>` must not already exist.
2. **Create the branch**: `git checkout -b autoresearch/<tag>` from the current main branch.
3. **Read the in-scope files**: The repo is intentionally small. Read these files for full context:
   - `README.md` — repository context
   - `prepare.py` — fixed CIFAR-10 harness, split policy, dataloader, evaluation. Do not modify.
   - `train.py` — the file you modify. Model architecture, optimizer, schedule, training loop.
4. **Verify data exists**: Check that `~/.cache/autoresearch/cifar10/` contains CIFAR-10 and that `~/.cache/autoresearch/splits/cifar10_train_val_split.pt` exists. If not, tell the human to run `uv run prepare.py`.
5. **Initialize `results.tsv`**: Create `results.tsv` with just the header row. The baseline will be recorded after the first run.
6. **Confirm and go**: Confirm setup looks good.

Once you get confirmation, kick off the experimentation.

## Experimentation

Each experiment runs on a single GPU. The training script runs for a **fixed time budget of 5 minutes** (wall clock training time, excluding startup/compilation overhead). You launch it simply as:

`uv run train.py`

**What you CAN do:**
- Modify `train.py`. This is the only file you edit during the research loop. Everything in it is fair game: architecture, optimizer, hyperparameters, schedules, batch size, width, depth, regularization, and training logic.

**What you CANNOT do:**
- Modify `prepare.py`. It is the fixed harness containing the CIFAR-10 split policy, preprocessing, dataloading, and evaluation.
- Modify the evaluation harness. The `evaluate_classifier` function in `prepare.py` is the ground truth metric path.
- Use the official CIFAR-10 **test** split in the autonomous keep/discard loop. The loop optimizes only on the pinned validation split.
- Install new packages or add dependencies during the loop. Use only what is already in `pyproject.toml`.

**The goal is simple: get the highest `val_acc`.** Since the time budget is fixed, you do not need to optimize for elapsed runtime beyond fitting comfortably inside the budget. Everything in `train.py` is fair game as long as the code runs without crashing and finishes within the time budget.

**Tie-breaker**: If two experiments have effectively the same `val_acc`, prefer the lower `val_loss`. If that is also similar, prefer the simpler implementation.

**VRAM** is a soft constraint. Some increase is acceptable for a meaningful accuracy gain, but it should not blow up dramatically.

**Simplicity criterion**: All else being equal, simpler is better. A tiny gain that adds ugly complexity is not worth it. Conversely, removing code and getting equal or better accuracy is a win.

**The first run**: Your very first run should always be the baseline, so run the training script as-is before changing anything.

## Output format

Once the script finishes it prints a summary like this:

```text
---
val_acc:          0.842600
val_loss:         0.505300
training_seconds: 300.0
total_seconds:    313.4
peak_vram_mb:     1820.5
images_per_sec:   24750.2
total_images_k:   3072.0
num_steps:        3000
num_params_M:     2.8
depth:            2
```

You can extract the key metrics from the log file with:

```bash
rg "^(val_acc|val_loss|peak_vram_mb):" run.log
```

## Logging results

When an experiment is done, log it to `results.tsv` (tab-separated, not comma-separated).

The TSV has a header row and 5 columns:

```text
commit	val_acc	memory_gb	status	description
```

1. git commit hash (short, 7 chars)
2. `val_acc` achieved (for example `0.842600`) — use `0.000000` for crashes
3. peak memory in GB, round to one decimal place (divide `peak_vram_mb` by 1024) — use `0.0` for crashes
4. status: `keep`, `discard`, or `crash`
5. short text description of what this experiment tried

Example:

```text
commit	val_acc	memory_gb	status	description
a1b2c3d	0.781200	1.8	keep	baseline
b2c3d4e	0.798500	1.9	keep	increase base width to 96
c3d4e5f	0.794000	1.8	discard	switch off data augmentation
d4e5f6g	0.000000	0.0	crash	double batch size too far and OOM
```

## The experiment loop

The experiment runs on a dedicated branch (for example `autoresearch/mar13` or `autoresearch/mar13-gpu0`).

LOOP FOREVER:

1. Look at the git state: the current branch/commit you are on.
2. Tune `train.py` with one experimental idea.
3. Git commit.
4. Run the experiment: `uv run train.py > run.log 2>&1`
5. Read out the results: `rg "^(val_acc|val_loss|peak_vram_mb):" run.log`
6. If the metric lines are missing, the run crashed. Read the end of `run.log`, understand the failure, and decide whether to fix-and-rerun or log a crash and move on.
7. Record the result in `results.tsv` (do not commit `results.tsv`).
8. If `val_acc` improved, advance the branch and keep the commit.
9. If `val_acc` is worse, reset back to where you started.
10. If `val_acc` is tied, use `val_loss`, simplicity, and VRAM as tie-breakers.

The idea is that you are a completely autonomous researcher trying things out. If they work, keep them. If they do not, discard them.

**Timeout**: Each experiment should take about 5 minutes total plus startup/eval overhead. If a run exceeds 10 minutes, kill it and treat it as a failure.

**Crashes**: If a run crashes due to a simple bug, fix it and re-run. If the idea itself is fundamentally bad, log a crash and move on.

**NEVER STOP**: Once the experiment loop has begun, do not pause to ask the human whether to continue. Keep running experiments until manually interrupted.
