# M2PF Data Challenge 2026

Predicting the sign of an allocation's next return from its 20 previous returns,
signed volumes, median daily turnover and two categorical descriptors.

## Data

The challenge data is not in this repository. Download it from the organizer and
place the files in `data/`:

| File | Contents |
| --- | --- |
| `X_train.csv` | Features for the training rows |
| `y_train.csv` | `target`: the return at time 21 |
| `X_test.csv` | Features for the rows to predict |
| `sample_submission.csv` | Submission template (needed by `00_benchmark.ipynb`) |

Feature columns:

- `TS` anonymized, shuffled timestamp
- `ALLOCATION` label per allocation (i.e. different trading strategies)
- `RET_i` return at time `i = 1 … 20`
- `SIGNED_VOLUME_i` signed volume at time `i = 1 … 20`
- `MEDIAN_DAILY_TURNOVER` median daily turnover over the 20 fixings
- `GROUP` anonymized allocation group (long-short, momentum, …)

## Notebooks

Run them from inside `notebooks/` the paths are relative to that folder.

| Notebook | What it does |
| --- | --- |
| `00_benchmark.ipynb` | Organizer's benchmark: a Ridge regression and a cross-validated LightGBM on engineered features |
| `01_simple_net.ipynb` | `simpleNet` a feed-forward net over the flattened continuous features plus learned embeddings for `ALLOCATION` and `GROUP` |
| `02_transformer.ipynb` | Work in progress: a transformer encoder over the (batch × time × channel) sequence, to be combined with the categorical embeddings |

`02_transformer.ipynb` was written for Google Colab and still reads its data from
Google Drive, the other two read from `data/`.

## Layout

```
data/         challenge csv files (git-ignored)
models/       trained checkpoints — simple_net.pt holds the 01 baseline weights
notebooks/    the analysis, in order
resources/    figures used in the notebooks
submissions/  generated prediction files (git-ignored)
```

## Setup

```bash
pip install -r requirements.txt
jupyter lab
```
