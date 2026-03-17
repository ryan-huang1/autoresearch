# autoresearch

This repo runs autonomous **minute-level time-series forecasting** experiments on Orca Whirlpool data stored in `orca_hist_last_year/`.

The job of the experiment agent is to improve the forecasting pipeline by iterating on:

- the **model** in `train.py`
- the **feature pipeline** in `prepare.py`
- the **aggregations and derived signals** built from all Orca source files
- the **lookback window** and related signal-side knobs in `prepare.py`

## Setup

To set up a new experiment, work with the user to:

1. **Agree on a run tag**: propose a tag based on today's date, for example `mar17`. The branch `autoresearch/<tag>` must not already exist.
2. **Create the branch**: create `autoresearch/<tag>` from the current default branch.
3. **Read the in-scope files**:
   - `README.md` for general repository context only.
   - `prepare.py` for the data contract, feature pipeline, split logic, and evaluation helpers.
   - `train.py` for the model, optimizer, and training loop.
   - `program.md` for the autonomous research rules.
4. **Treat the current Python code and this file as source of truth**:
   - The README may lag behind the benchmark. If it disagrees with `prepare.py`, `train.py`, or `program.md`, follow the current code and this file.
5. **Verify the raw dataset directory exists**:
   - Confirm `orca_hist_last_year/` exists in the repo root.
   - Confirm it contains:
     - `metadata.json`
     - `ohlcv_minutely.parquet`
     - `swaps.parquet`
     - `liquidity_events.parquet`
     - `daily_state.parquet`
6. **Verify prepared artifacts exist**:
   - Confirm the prepared directory for this run contains `features.npy`, `targets.npy`, and `metadata.json`.
   - By default this is `AUTORESEARCH_PREPARED_DIR` if set, otherwise `~/.cache/autoresearch/timeseries/` on the machine that will run training.
   - If you override the prepared directory, keep `uv run prepare.py --output-dir <dir>` and `uv run train.py --prepared-dir <dir>` aligned.
   - If the prepared artifacts were built from older benchmark code, rerun `uv run prepare.py`.
7. **Initialize local experiment artifacts**:
   - Create `results.tsv` if it does not exist.
   - Ensure `findings/` exists locally.
   - Both `results.tsv` and `findings/` must remain untracked by git.
8. **Confirm and go**: once setup looks correct, start the experiment loop.

## Remote GPU Machine

A remote GPU machine is available for experiments. Prefer that machine over the local CPU machine for heavy data preparation and all real training runs.

Connect with:

- `ssh -i ~/.ssh/tensordock -o IdentitiesOnly=yes user@38.224.253.249`

Operational rules:

- Default execution target for `uv run prepare.py` and `uv run train.py` is the remote GPU machine.
- Use the local machine mainly for editing, inspection, lightweight checks, and coordination.
- Use `scp` or `rsync` to transfer the repo, datasets, caches, logs, `results.tsv`, `findings/`, and other experiment artifacts back and forth between the local machine and the remote machine whenever useful.
- You may install packages, Python environments, CUDA/tooling dependencies, and other machine-local software on the remote machine without asking first.
- After remote runs, use `scp` or `rsync` to sync the important outputs back so the local repo remains the canonical home for untracked experiment artifacts.
- This permission does not override repo rules about committed dependency-manifest changes unless a human explicitly asks.

## Orca Data Inventory

The raw dataset is the entire `orca_hist_last_year/` directory. Use all of it when useful.

### `metadata.json`

This is the dataset manifest and provenance file.

- It identifies the pool, token mints, fee rate, date range, row counts, and repair history.
- It is useful for sanity checks, reporting, and benchmark provenance.
- It is not the main predictive table, but it can inform constants and data validation.

Important fields include:

- `pool`
- `token_mint_a`
- `token_mint_b`
- `fee_rate`
- `date_start`
- `date_end`
- `daily_state_rows`
- `swap_rows`
- `liquidity_event_rows`
- `ohlcv_minutely_rows`
- `repair_history`

### `ohlcv_minutely.parquet`

This is the canonical **1-minute OHLCV bar table** and the base time grid for the benchmark.

Columns:

- `timestamp`
- `open`
- `high`
- `low`
- `close`
- `volume`

How to use it:

- Sort by `timestamp` and deduplicate deterministically.
- Reindex to a complete 1-minute grid across the covered range.
- Use it as the price target source and as the base timeline for all joins.
- Derive baseline price, candle, return, volume, momentum, mean-reversion, volatility, and missingness features from it.

Important caveats:

- Missing minutes exist and must be handled explicitly.
- Under this benchmark, a missing minute should be interpreted as a **no-trade minute** for the pool.
- Treat no-trade minutes as signal, not as noise to be ignored away.
- Preserve explicit missingness / no-trade indicators so the model can learn from market inactivity.

### `swaps.parquet`

This is the raw event-level swap tape for the same pool.

Columns:

- `slot`
- `block_time`
- `instruction`
- `amount_in`
- `amount_out`
- `sqrt_price_pre`
- `sqrt_price_post`
- `tick_current_index_pre`
- `tick_current_index_post`
- `liquidity_pre`
- `liquidity_post`
- `fee_rate`

How to use it:

- Aggregate it onto the canonical minute grid.
- Build robust swap-flow features such as:
  - trade count
  - summed and averaged swap sizes
  - max swap size
  - tick-move summaries
  - event-intensity proxies

Important caveats:

- Several numeric fields are raw on-chain values.
- These values may be large and may need `log1p`, normalization, or careful aggregation.
- Do not use future information when aggregating.

### `liquidity_events.parquet`

This is the raw event-level LP and liquidity-management table.

Columns:

- `slot`
- `block_time`
- `instruction`
- `tick_lower_index`
- `tick_upper_index`
- `liquidity_delta`
- `sqrt_price_pre`
- `sqrt_price_post`
- `tick_current_index_pre`
- `tick_current_index_post`
- `liquidity_pre`
- `liquidity_post`
- `fee_rate`

How to use it:

- Aggregate it onto the canonical minute grid.
- Build robust liquidity-regime features such as:
  - counts by instruction
  - net liquidity change
  - absolute liquidity churn
  - average LP range width
  - position open/close activity

Important caveats:

- Some instruction types naturally have nulls in some fields.
- Handle nulls in an instruction-aware way.
- Favor stable simple aggregates over brittle event-order assumptions.

### `daily_state.parquet`

This is the slow-moving daily pool-state snapshot table.

Columns:

- `date`
- `tick_spacing`
- `fee_rate`
- `sqrt_price`
- `tick_current_index`
- `liquidity`

How to use it:

- Join it as a slower daily overlay on the minute grid.
- Forward-fill daily state within each day as needed.
- Derive slow regime features such as:
  - daily liquidity level
  - daily tick regime
  - daily state changes

Important caveats:

- These are not minute bars.
- Use them as regime context, not as a replacement for the minute grid.

### General Data Notes

- `ohlcv_minutely.parquet` is the benchmark anchor.
- `swaps.parquet` and `liquidity_events.parquet` are auxiliary event streams that must be aggregated onto the minute grid.
- `daily_state.parquet` is a slower regime overlay.
- `daily_state.parquet` should be treated conservatively: when aligned to the minute grid, lag it by one full day unless snapshot timing is explicitly verified.
- Raw on-chain numeric fields can be large; robust transforms are encouraged.
- Sort or group data deterministically before any lagged, rolling, or aggregation logic.

## Benchmark Contract

These rules are fixed unless a human explicitly changes them:

- The raw dataset is the directory `orca_hist_last_year/`.
- The canonical bar grid is **1-minute bars** anchored on `ohlcv_minutely.parquet`.
- Missing minutes are **densified** into synthetic 1-minute bars.
- For benchmark purposes, a missing minute means there were **no trades in that minute**.
- Synthetic minute bars must use:
  - `open = previous_close`
  - `high = previous_close`
  - `low = previous_close`
  - `close = previous_close`
  - `volume = 0`
- Keep an explicit missingness indicator for synthetic minutes.
- Treat synthetic / no-trade minutes as predictive state and preserve that information in the feature set.
- Auxiliary features may be derived from `swaps.parquet`, `liquidity_events.parquet`, `daily_state.parquet`, and `metadata.json`, but they must align causally to the minute grid.
- The prediction target is:
  - `log(close[t+3] / close[t])`
- Splits are chronological.
- The newest block remains the untouched anchored final test set.
- Validation uses walk-forward folds immediately before the test block.
- Each walk-forward fold trains only on history available before that fold, with purge gaps preserved.
- `daily_state.parquet` is lagged by one full day before being exposed to the minute grid.
- The purge gap is fixed separately from lookback.
- Purge gaps remain part of the split contract.
- The primary keep/discard metric is **`val_corr`**.
- The agent chooses the wall-clock training budget for each experiment, with no repo-level maximum cap.
- In general, prefer shorter runs for faster iteration loops, but run long enough to get reliable results and useful insight.
- Do not add dependencies or change repo-wide instructions without human approval.

Do not change the benchmark to make numbers look better. Improve the forecaster, not the game.

## What You May Modify

- `train.py`
  - Model family and architecture.
  - Hidden size, layer count, dropout, regularization, sequence summarization, and overall model size / parameter count.
  - Experiments with materially different architectures, representation styles, depths, widths, and capacities.
  - Optimizer, LR schedule, batch sizing, and training loop details.
  - Loss implementation and diagnostics.
- `prepare.py`
  - Minute-grid feature engineering from `ohlcv_minutely.parquet`.
  - Aggregations derived from `swaps.parquet`.
  - Aggregations derived from `liquidity_events.parquet`.
  - Regime overlays derived from `daily_state.parquet`.
  - Any robust use of `metadata.json` for validation or constant side information.
  - Rolling statistics, transforms, and normalization, as long as normalization is fit on train-only data.
  - `LOOKBACK_BARS`, feature windows, and other signal-side knobs.
  - Feature engineering as a first-class research lever, not just minor cleanup around the model.

Explicit instruction:

- Your job is **not only** to tune architecture and optimization.
- You are explicitly allowed to edit **both** `train.py` and `prepare.py`.
- The human expects you to treat **model engineering and feature engineering as equally important levers** for improving `val_corr`.
- Your job **also explicitly includes feature engineering from all Orca source files** when doing so may improve `val_corr`.
- Do not limit yourself to minor hyperparameter tweaks. Actively experiment with different model architectures, different model sizes, and different parameter budgets when justified.
- You are **free to modify the lookback** and related signal-side settings when justified.
- You should treat missing / synthetic minutes as meaningful **no-trade state** and engineer features that let the model use that information.

If you modify feature engineering, aggregation logic, normalization, missingness handling, or sequence construction in `prepare.py`, rerun `uv run prepare.py` before the next training run.

Clarification:

- If you change prepared input columns, rolling computations, event aggregation logic, normalization, or lookback, do it in `prepare.py`.
- If the model in `train.py` learns its own transformations of the existing input window, that is allowed.
- Learned feature extraction must not use future information, alter the benchmark contract, or introduce train/val/test leakage.

## What You May Not Modify

- The raw dataset directory or file set.
- The canonical minute-grid anchoring on `ohlcv_minutely.parquet`.
- The missing-minute fill semantics.
- The target definition.
- The chronological split semantics.
- The fixed purge gap as part of the benchmark contract.
- The primary evaluation metric `val_corr`.
- `program.md`, `README.md`, or `pyproject.toml` unless a human explicitly asks.
- Dependencies.

## Goal

The goal is simple: get the **highest `val_corr`** at the chosen training budget.

Secondary metrics such as `val_rmse`, `val_mae`, and `val_sign_acc` are for diagnosis and calibration checks. They help you understand *why* a run behaved the way it did, but they do not override the primary keep/discard rule.

Unless there is a clear reason to do otherwise:

- Start with shorter runs so you can complete more iteration loops.
- Increase the training budget when a stronger signal, a fairer confirmation, or better insight is worth the extra time.
- When comparing runs, note the chosen budget in the findings and `results.tsv` and avoid pretending that short and long runs are directly equivalent.

**Simplicity criterion**: all else equal, simpler is better. A tiny improvement that adds brittle complexity is usually not worth keeping. A similarly good or better result with simpler code is a win.

**The first run** should establish the baseline with the current code:

1. Run `uv run prepare.py`.
2. Choose an explicit training budget and run `uv run train.py --time-budget-seconds <seconds>`.
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
startup_seconds:   2.1
training_seconds:  600.0
eval_seconds:      8.4
total_seconds:     610.8
avg_samples_sec:   3820.5
peak_vram_mb:      1420.7
total_windows_K:   1228.8
num_steps:         1200
grad_accum_steps:  4
split_strategy:    anchored_test_walk_forward_val
num_val_folds:     3
fold_budgets_s:    200,200,200
device_batch_size: 512
total_batch_size:  2048
num_params_M:      0.185
device:            cuda
feature_dim:       42
train_samples:     366000
max_train_samples: 418000
val_samples:       78300
test_samples:      78300
eval_samples:      78300
observed_bars:     520138
synthetic_bars:    5462
lookback_bars:     300
horizon_bars:      3
fold1_val_corr:    0.101010
fold2_val_corr:    0.123456
fold3_val_corr:    0.118118
```

The `val_*` metrics above are the aggregate out-of-sample walk-forward validation metrics across all validation folds. The final test block remains untouched during experiment selection.

Extract the key metrics from `run.log` after each run. Higher `val_corr` is better.

## Logging Results

Keep a lightweight structured index in `results.tsv` (tab-separated, not comma-separated).

Header:

```text
commit	time_budget_s	training_seconds	val_corr	val_rmse	val_mae	val_sign_acc	memory_gb	status	description	findings_file
```

Columns:

1. Short git commit hash.
2. Chosen time budget in seconds for that run.
3. Actual `training_seconds` achieved. Use `0.0` for crashes.
4. `val_corr` achieved. Higher is better. Use `nan` for crashes.
5. `val_rmse` achieved. Use `nan` for crashes.
6. `val_mae` achieved. Use `nan` for crashes.
7. `val_sign_acc` achieved. Use `nan` for crashes.
8. Peak memory in GB, rounded to one decimal place. Use `0.0` for crashes.
9. Status: `keep`, `discard`, or `crash`.
10. Short description of the experiment.
11. Relative path to the markdown report in `findings/`.

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

The experiment runs on a dedicated branch such as `autoresearch/mar17`.

LOOP FOREVER:

1. Check the current branch and commit.
2. Decide whether the next idea is:
   - model-focused
   - feature-focused
   - a combined model + feature change
3. Edit `train.py` and, when justified, the signal-engineering parts of `prepare.py`.
4. In `prepare.py`, consider all available raw sources:
   - `ohlcv_minutely.parquet`
   - `swaps.parquet`
   - `liquidity_events.parquet`
   - `daily_state.parquet`
   - `metadata.json`
5. If `prepare.py` changed in a way that affects prepared data, rerun:
   - `uv run prepare.py`
6. Commit the tracked code changes.
7. Run the experiment on the remote GPU machine:
   - Run with an explicit budget: `uv run train.py --time-budget-seconds <seconds> > run.log 2>&1`
8. Read out the results from `run.log`:
   - `time_budget_s`
   - `training_seconds`
   - `val_corr`
   - `val_rmse`
   - `val_mae`
   - `val_sign_acc`
   - `peak_vram_mb`
9. If the metrics are missing, treat the run as a crash and inspect the traceback.
10. Write the detailed markdown report to `findings/`.
11. Append the structured result to `results.tsv`.
12. Compare runs fairly at the same budget whenever possible.
13. If a shorter run is promising, you may rerun it at a longer budget to confirm the signal before deciding whether to keep it.
14. If `val_corr` improved on a fair comparison, keep the change.
15. If `val_corr` is worse or equal, restore the tracked code to the last kept commit, but keep the local findings and result logs.

## Timeout

The agent may choose the training budget for any experiment. There is no repo-level maximum cap. In general, prefer shorter runs so you can complete more iteration loops and improve the model faster, but make sure each run is long enough to produce good results and useful insight. Pass `--time-budget-seconds <seconds>` on every training run. The total budget must still be large enough to allocate at least one second per validation fold. If a run goes meaningfully beyond its chosen budget and is clearly hung, stop it, log the failure, and move on.

Crashes: If a run crashes (OOM, or a bug, or etc.), use your judgment: If it's something dumb and easy to fix (e.g. a typo, a missing import), fix it and re-run. If the idea itself is fundamentally broken, just skip it, log "crash" as the status in the tsv, and move on.

NEVER STOP: Once the experiment loop has begun (after the initial setup), do NOT pause to ask the human if you should continue. Do NOT ask "should I keep going?" or "is this a good stopping point?". The human might be asleep, or gone from a computer and expects you to continue working indefinitely until you are manually stopped. You are autonomous. If you run out of ideas, think harder — read papers referenced in the code, re-read the in-scope files for new angles, try combining previous near-misses, try more radical architectural changes. The loop runs until the human interrupts you, period.

As an example use case, a user might leave you running while they sleep. The user then wakes up to experimental results, all completed by you while they slept!

