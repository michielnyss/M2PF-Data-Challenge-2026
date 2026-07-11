# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a QRT (Quantitative Research Tournament) challenge: a binary classification task predicting whether to go long (1) or short (0) on financial asset allocations. The metric is directional accuracy (`sign(pred) == sign(real)`).

## Running the Code

All scripts expect to be run from the `data/` directory (or with the working directory adjusted), since `get_data()` uses relative paths like `pd.read_csv("X_train.csv")`. The `strategies/` scripts resolve paths via `BASE_DIR = Path(__file__).resolve().parent.parent`.

```bash
# Run the data processing pipeline (produces X_train_cleaned.csv, y_train_cleaned.csv)
python data/data_processing.py

# Run feature engineering exploration
python data/feature_engineering.py

# Run the GLM/model strategy
python strategies/generalized_models.py

# Run the benchmark notebook
jupyter notebook benchmark_submission.ipynb
```

## Data

Raw data lives in `data/`:
- `X_train.csv`, `X_test.csv`: features indexed by `ROW_ID`
- `y_train.csv`: targets indexed by `ROW_ID`, column `target` (continuous return)
- `sample_submission.csv`: submission template
- `X_train_cleaned.csv`, `y_train_cleaned.csv`: processed output from `data_processing.py`

**Key columns:**
- `RET_1` through `RET_20`: trailing 20-day return time series (RET_20 is oldest, RET_1 is most recent)
- `SIGNED_VOLUME_1` through `SIGNED_VOLUME_20`: signed volume time series
- `MEDIAN_DAILY_TURNOVER`: scalar liquidity measure
- `ALLOCATION`: identifier grouping rows into asset allocation batches
- `TS`: timestamp
- `GROUP`: categorical group label

## Architecture & Pipeline

The pipeline has three stages:

### 1. Data Processing (`data/data_processing.py`) — `PipelineState`
Groups rows by `ALLOCATION`, then within each allocation:
1. **Computes cross-correlations** between all pairs of return time series (`_compute_lag`)
2. **Clusters** time series that have high cross-correlation (corr > 0.9) using graph-connected-components (`_cluster_lag`)
3. **Orders** time series within each cluster by their lag structure to find the "lead-lag" sequence (`_order_timeseries`)
4. **Generates aligned features** by time-shifting series to align them, then averaging across the cluster (`_generated_ordered_data`)

Output: each row in `X_train_cleaned.csv` represents one cluster (not one original asset), with up to 70-column wide aligned price/volume/turnover arrays, right-aligned. The target is the return of the **last** series in the ordered cluster.

### 2. Feature Engineering (`data/feature_engineering.py`) — `featureEngineeringPipeline`
Operates on cleaned data. Key methods:
- `trim_cols(indices, prefix)`: drops columns by index range and prefix
- `n_day_avg(n_ret, n_vol, n_turn)`: computes rolling N-day averages across the time axis (columns), optionally replacing originals

Column naming convention after processing: `RET_{t}`, `SIGNED_VOLUME_{t}`, `MEDIAN_DAILY_TURNOVER_{t}` where `t` counts back from 1 (most recent).

### 3. Modeling (`strategies/generalized_models.py`, `benchmark_submission.ipynb`)
- **Benchmark notebook**: Ridge regression and LightGBM (MSE objective, 8-fold CV by date) on naive features + rolling averages. Baseline ~52% accuracy.
- **GLM strategy**: `statsmodels` logistic regression (`smf.glm` with `Binomial` family) on mean return/volume/turnover aggregates, with `GROUP` as a categorical covariate.

### Evaluation (`evaluation.py`)
- `calculate_accuracy(pred, real)`: directional accuracy — fraction where `sign(pred) == sign(real)`
- `confusion_matrix(pred, real)`: returns a labeled `pd.DataFrame` (binary 0/1 predictions)
