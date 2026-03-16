# autoresearch

This repo now runs autonomous **time-series forecasting** experiments on SOL OHLCV data.

The job of the experiment agent is to improve the forecasting pipeline by iterating on:

- the **model** in `train.py`
- the **derived signals / feature engineering** in `prepare.py`

The job of the human is to keep the **benchmark definition** stable so experiments remain comparable.

## Setup

To set up a new experiment, work with the user to:

1. **Agree on a run tag**: propose a tag based on today's date (for example `mar15`). The branch `autoresearch/<tag>` must not already exist.
2. **Create the branch**: create `autoresearch/<tag>` from the current default branch.
3. **Read the in-scope files**:
   - `README.md` for repository context.
   - `prepare.py` for the data contract, feature pipeline, split logic, and evaluation helpers.
   - `train.py` for the model, optimizer, and training loop.
   - `program.md` for the autonomous research rules.
4. **Verify the raw dataset exists**: confirm `gecko_pool_1s_history.csv` is present in the repo root.
5. **Verify prepared artifacts exist**: confirm `~/.cache/autoresearch/timeseries/` contains `features.npy`, `targets.npy`, and `metadata.json`, and if they were built from an older CSV, rerun `uv run prepare.py`.
6. **Initialize local experiment artifacts**:
   - Create `results.tsv` if it does not exist.
   - Ensure `findings/` exists locally.
   - Both `results.tsv` and `findings/` must remain untracked by git.
7. **Confirm and go**: once setup looks correct, start the experiment loop.

## Benchmark Contract

These rules are fixed unless a human explicitly changes them:

- The raw dataset is `gecko_pool_1s_history.csv`.
- Missing seconds are **densified** into synthetic 1-second bars.
- Synthetic bars must use:
  - `open = previous_close`
  - `high = previous_close`
  - `low = previous_close`
  - `close = previous_close`
  - `volume = 0`
- The prediction target is:
  - `log(close[t+60] / close[t])`
- Splits are chronological.
- Purge gaps remain part of the split contract.
- The newest block remains the untouched final test set.
- The primary keep/discard metric is **`val_corr`**.
- The default wall-clock training budget is **10 minutes**.
- The agent may rerun a promising candidate for up to **20 minutes** by explicitly setting a longer time budget, but should do this sparingly.
- Do not add dependencies or change repo-wide instructions without human approval.

Do not change the benchmark to make numbers look better. Improve the forecaster, not the game.

## What You May Modify

- `train.py`
  - Model family and architecture.
  - Learned feature extraction / representation learning inside the model from the allowed input window.
  - Hidden size, layer count, dropout, regularization.
  - Optimizer, LR schedule, batch sizing, and training loop details.
  - Loss implementation and diagnostics.
- `prepare.py`
  - Derived signals from the raw OHLCV bars.
  - Feature transforms and rolling statistics.
  - Normalization method, as long as it is fit on train-only data.
  - Input lookback length and other signal-side knobs.

If you modify feature engineering or sequence construction in `prepare.py`, rerun `uv run prepare.py` before the next training run.

Clarification:

- If you change the prepared input columns, rolling computations, normalization, or sequence-construction contract, do it in `prepare.py`.
- If the model in `train.py` learns its own transformations of the existing input window, that is allowed.
- Learned feature extraction must not use future information, alter the benchmark contract, or introduce train/val/test leakage.

## What You May Not Modify

- The raw dataset path or data source.
- The gap-fill semantics for missing seconds.
- The target definition.
- The chronological split semantics.
- The purge-gap logic as part of the benchmark contract.
- The primary evaluation metric `val_corr`.
- `program.md`, `README.md`, or `pyproject.toml` unless a human explicitly asks.
- Dependencies.

## Goal

The goal is simple: get the **highest `val_corr`** under the configured time budget.

Secondary metrics such as `val_rmse`, `val_mae`, and `val_sign_acc` are for diagnosis and calibration checks. They help you understand *why* a run behaved the way it did, but they do not override the primary keep/discard rule.

Unless there is a clear reason to do otherwise:

- Use the default `10 minute` budget for ordinary exploration runs.
- Optionally rerun a promising candidate at `15-20 minutes` for confirmation.
- Do not compare a `10 minute` run directly against a `20 minute` run without noting the different budget in the findings and `results.tsv`.

**Simplicity criterion**: all else equal, simpler is better. A tiny improvement that adds brittle complexity is usually not worth keeping. A similarly good or better result with simpler code is a win.

**The first run** should establish the baseline with the current code:

1. Run `uv run prepare.py` if needed.
2. Run `uv run train.py` exactly as-is.
3. Record the baseline before trying ideas.

## Output Format

Each training run prints a summary like:

```text
---
primary_metric:    val_corr
primary_value:     0.123456
val_corr:          0.123456
val_rmse:          0.012345
val_mae:           0.009876
val_sign_acc:      0.531250
last_train_loss:   0.000456
smooth_train_loss: 0.000512
last_grad_norm:    3.142857
time_budget_s:     600
max_time_budget_s: 1200
startup_seconds:   1.8
training_seconds:  600.0
eval_seconds:      18.4
total_seconds:     620.2
avg_samples_sec:   3820.5
peak_vram_mb:      1420.7
total_windows_K:   1228.8
num_steps:         600
grad_accum_steps:  8
device_batch_size: 256
total_batch_size:  2048
num_params_M:      0.185
device:            cuda
feature_dim:       17
train_samples:     3561212
val_samples:       763271
test_samples:      763271
eval_samples:      131072
observed_bars:     5055033
synthetic_bars:    34099
lookback_bars:     300
horizon_bars:      60
```

Extract the key metrics from `run.log` after each run. Higher `val_corr` is better.

## Logging Results

Keep a lightweight structured index in `results.tsv` (tab-separated, not comma-separated).

Header:

```text
commit	time_budget_s	training_seconds	val_corr	val_rmse	val_mae	val_sign_acc	memory_gb	status	description	findings_file
```

Columns:

1. Short git commit hash.
2. Configured time budget in seconds, such as `600` or `1200`.
3. Actual `training_seconds` achieved. Use `0.0` for crashes.
4. `val_corr` achieved. Higher is better. Use `nan` for crashes.
5. `val_rmse` achieved. Use `nan` for crashes.
6. `val_mae` achieved. Use `nan` for crashes.
7. `val_sign_acc` achieved. Use `nan` for crashes.
8. Peak memory in GB, rounded to one decimal place. Use `0.0` for crashes.
9. Status: `keep`, `discard`, or `crash`.
10. Short description of the experiment.
11. Relative path to the markdown report in `findings/`.

Example:

```text
commit	time_budget_s	training_seconds	val_corr	val_rmse	val_mae	val_sign_acc	memory_gb	status	description	findings_file
a1b2c3d	600	600.1	0.012300	0.012345	0.009876	0.5312	1.4	keep	baseline lstm findings/2026-03-15-baseline.md
b2c3d4e	600	600.0	0.018900	0.011980	0.009410	0.5386	1.5	keep	add realized vol features findings/2026-03-15-realized-vol.md
c3d4e5f	1200	1200.2	0.021400	0.011500	0.009100	0.5410	1.5	keep	confirm promising realized vol run findings/2026-03-15-confirm-realized-vol.md
d4e5f6g	600	0.0	nan	nan	nan	nan	0.0	crash	bad feature normalization findings/2026-03-15-crash-normalization.md
```

## Findings Reports

After **every** experiment, create a detailed markdown report under `findings/`, even if the run crashes or is discarded.

Recommended filename format:

```text
findings/YYYY-MM-DD-short-slug.md
```

Each report should include:

- `Hypothesis`
- `Benchmark Contract`
- `Data And Split`
- `Signals`
- `Model`
- `Training Setup`
- `Metrics`
- `Outcome`
- `Failure Modes`
- `Next Step`

Prefer adding YAML front matter at the top with fields such as:

- `date`
- `commit`
- `status`
- `val_corr`
- `val_rmse`
- `val_mae`
- `val_sign_acc`
- `time_budget_s`
- `description`

Findings must remain local untracked artifacts so they survive discarded code revisions.

## The Experiment Loop

The experiment runs on a dedicated branch such as `autoresearch/mar15`.

LOOP FOREVER:

1. Check the current branch and commit.
2. Decide whether the next idea is:
   - model-focused
   - signal-focused
   - a combined model + signal change
3. Edit `train.py` and, when justified, the signal-engineering parts of `prepare.py`.
4. If `prepare.py` changed in a way that affects prepared data, rerun:
   - `uv run prepare.py`
5. Commit the tracked code changes.
6. Run the experiment:
   - Default screen: `uv run train.py > run.log 2>&1`
   - Optional confirmation rerun: `uv run train.py --time-budget-seconds 1200 > run.log 2>&1`
7. Read out the results from `run.log`:
   - `time_budget_s`
   - `training_seconds`
   - `val_corr`
   - `val_rmse`
   - `val_mae`
   - `val_sign_acc`
   - `peak_vram_mb`
8. If the metrics are missing, treat the run as a crash and inspect the traceback.
9. Write the detailed markdown report to `findings/`.
10. Append the structured result to `results.tsv`.
11. Compare runs fairly at the same budget whenever possible.
12. If a `10 minute` run is promising, you may rerun it at up to `20 minutes` to confirm the signal before deciding whether to keep it.
13. If `val_corr` improved on a fair comparison, keep the change.
14. If `val_corr` is worse or equal, restore the tracked code to the last kept commit, but keep the local findings and result logs.

## Crash Handling

If a run crashes:

- If the issue is trivial and clearly fixable, fix it and rerun.
- If the idea itself is broken, log it as `crash`, write the findings report, and move on.

## Timeout

Each experiment should take about 10 minutes of training time by default, or up to 20 minutes for an explicit confirmation rerun, plus startup and evaluation overhead. If a run goes meaningfully beyond its configured budget and is clearly hung, stop it, log the failure, and move on.

## Never Stop

Once the experiment loop begins, do not pause to ask the human if you should continue. Keep iterating until explicitly interrupted.
