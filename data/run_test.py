# -*- coding: utf-8 -*-
"""
Run the lead-lag pipeline on X_test and write cleaned features.

Usage
-----
    python data/run_test.py

No targets are produced (X_test has no labels).

Checkpointing
-------------
Each allocation is written to data/checkpoints/test/ immediately after
processing.  On restart, already-done allocations are skipped automatically.
Delete an allocation's checkpoint file to force a reprocess.
After every run the final CSV is rebuilt from all available checkpoints.

To run all allocations, change allocation[:3] to allocation below.
"""

from pathlib import Path
import time

import pandas as pd

from data_processing import AllocationPipeline, get_data, DATA_DIR

CHECKPOINT_DIR  = DATA_DIR / "checkpoints" / "test"
OUTPUT_FEATURES = DATA_DIR / "X_test_cleaned.csv"


def _alloc_number(p: Path) -> int:
    return int(p.stem.split("_")[1])


def run(
    output_features: Path = OUTPUT_FEATURES,
    corr_threshold:  float = 0.9,
    nan_threshold:   float = 0.2,
):
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)

    _, X_test, _ = get_data()

    allocation = sorted(
        X_test["ALLOCATION"].unique(),
        key=lambda x: int(x.split("_")[1]),
    )

    for alloc in allocation:
        feat_ckpt = CHECKPOINT_DIR / f"{alloc}_features.csv"

        if feat_ckpt.exists():
            print(f"{alloc} already processed, skipping.")
            continue

        t0         = time.time()
        alloc_mask = X_test["ALLOCATION"] == alloc

        pipeline = AllocationPipeline(
            raw_data=X_test.loc[alloc_mask].copy(),
            allocation=alloc,
            y_data=None,
            corr_threshold=corr_threshold,
            nan_threshold=nan_threshold,
        )
        pipeline.fit()

        pipeline.features.to_csv(feat_ckpt, index=False)
        print(f"{alloc} done in {time.time() - t0:.2f}s")

    feat_files = sorted(CHECKPOINT_DIR.glob("*_features.csv"), key=_alloc_number)

    pd.concat([pd.read_csv(f) for f in feat_files], ignore_index=True).to_csv(output_features, index=False)
    print(f"Merged {len(feat_files)} allocations -> {output_features.name}")


if __name__ == "__main__":
    run()
