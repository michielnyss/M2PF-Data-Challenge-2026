# -*- coding: utf-8 -*-
"""
Run the lead-lag pipeline on X_train and write cleaned features + targets.

Usage
-----
    python data/run_train.py

Checkpointing
-------------
Each allocation is written to data/checkpoints/train/ immediately after
processing.  On restart, already-done allocations are skipped automatically.
Delete an allocation's checkpoint files to force a reprocess.
After every run the final CSVs are rebuilt from all available checkpoints.

To run all allocations, change allocation[:3] to allocation below.
"""

from pathlib import Path
import time

import pandas as pd

from data_processing import AllocationPipeline, get_data, DATA_DIR

CHECKPOINT_DIR  = DATA_DIR / "checkpoints" / "train"
OUTPUT_FEATURES = DATA_DIR / "X_train_cleaned.csv"
OUTPUT_TARGETS  = DATA_DIR / "y_train_cleaned.csv"


def _alloc_number(p: Path) -> int:
    return int(p.stem.split("_")[1])


def run(
    output_features: Path = OUTPUT_FEATURES,
    output_targets:  Path = OUTPUT_TARGETS,
    corr_threshold:  float = 0.9,
    nan_threshold:   float = 0.2,
):
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)

    X_train, _, y_train = get_data()

    allocation = sorted(
        X_train["ALLOCATION"].unique(),
        key=lambda x: int(x.split("_")[1]),
    )

    for alloc in allocation:
        feat_ckpt = CHECKPOINT_DIR / f"{alloc}_features.csv"
        tgt_ckpt  = CHECKPOINT_DIR / f"{alloc}_targets.csv"

        if feat_ckpt.exists() and tgt_ckpt.exists():
            print(f"{alloc} already processed, skipping.")
            continue

        t0         = time.time()
        alloc_mask = X_train["ALLOCATION"] == alloc
        alloc_ids  = X_train.loc[alloc_mask, "ROW_ID"]
        y_alloc    = y_train.loc[y_train["ROW_ID"].isin(alloc_ids)]

        pipeline = AllocationPipeline(
            raw_data=X_train.loc[alloc_mask].copy(),
            allocation=alloc,
            y_data=y_alloc,
            corr_threshold=corr_threshold,
            nan_threshold=nan_threshold,
        )
        pipeline.fit()

        pipeline.features.to_csv(feat_ckpt, index=False)
        pipeline.targets.to_csv(tgt_ckpt,   index=False)
        print(f"{alloc} done in {time.time() - t0:.2f}s")

    feat_files = sorted(CHECKPOINT_DIR.glob("*_features.csv"), key=_alloc_number)
    tgt_files  = sorted(CHECKPOINT_DIR.glob("*_targets.csv"),  key=_alloc_number)

    pd.concat([pd.read_csv(f) for f in feat_files], ignore_index=True).to_csv(output_features, index=False)
    pd.concat([pd.read_csv(f) for f in tgt_files],  ignore_index=True).to_csv(output_targets,  index=False)
    print(f"Merged {len(feat_files)} allocations -> {output_features.name}, {output_targets.name}")


if __name__ == "__main__":
    run()
