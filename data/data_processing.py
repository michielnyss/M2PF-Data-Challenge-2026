# -*- coding: utf-8 -*-
"""
Lead-lag clustering pipeline for financial return series.

Each allocation contains multiple rows, where each row represents the same
asset at a different timestamp. Some rows have return windows that look like
time-shifted versions of each other — they are observing the same underlying
market signal, but one asset reacts earlier than the other (the "lead-lag"
effect).

This module finds those lagged pairs, clusters them into connected groups,
orders each cluster from the most-leading to the most-lagging series, and
then aggregates the temporally aligned signals into a single feature row per
cluster. The prediction target is the future return of the most-lagging
(most recent) series in the cluster, using the earlier series as leading
indicators.

Pipeline stages
---------------
1. _compute_lag_matrix  — FFT cross-correlation → pairwise lag & correlation matrices
2. _cluster             — connected-components clustering on the correlation graph
3. _order_all_clusters  — greedy temporal ordering within each cluster
4. _align_and_aggregate — time-shift alignment and nanmean aggregation → feature rows
"""

from pathlib import Path

import pandas as pd
import numpy as np
import time
import warnings

DATA_DIR   = Path(__file__).parent
WINDOW_LEN = 20   # number of trading days per return / volume observation

PRICE_COLS = [f"RET_{i}" for i in range(20, 0, -1)]
VOL_COLS   = [f"SIGNED_VOLUME_{i}" for i in range(20, 0, -1)]


def get_data():
    X_train = pd.read_csv(DATA_DIR / "X_train_cleaned.csv")
    X_test  = pd.read_csv(DATA_DIR / "X_test_cleaned.csv")
    y_train = pd.read_csv(DATA_DIR / "y_train_cleaned.csv")
    return X_train, X_test, y_train


class AllocationPipeline:
    """Process one allocation through the lead-lag clustering pipeline.

    Rows in `raw_data` represent the same asset at sequential timestamps.
    The pipeline detects which rows lead others in terms of their return
    pattern, clusters co-moving rows together, aligns them on a shared time
    axis, and produces one aggregated feature row per cluster.

    Parameters
    ----------
    raw_data        : DataFrame slice for a single allocation (from X_train).
    allocation      : Allocation identifier string (e.g. "ALLOC_3").
    y_data          : Matching rows from y_train (same ROW_IDs as raw_data).
    corr_threshold  : Minimum peak cross-correlation to connect two series.
    nan_threshold   : Maximum fraction of NaN in a return window before the
                      series is excluded from cross-correlation computation.

    Results (available after calling fit())
    ----------------------------------------
    features    : Aggregated feature DataFrame (one row per cluster).
    targets     : DataFrame with TARGET (continuous) and LABEL (binary) columns.
    diagnostic  : Per-cluster diagnostic dict; None unless fit(compute_diagnostics=True).
    time_report : Timing breakdown per pipeline stage.
    """

    def __init__(self, raw_data, allocation, y_data,
                 corr_threshold: float = 0.9,
                 nan_threshold: float = 0.2):
        self.raw_data       = raw_data
        self.allocation     = allocation
        self.y_data         = y_data
        self.corr_threshold = corr_threshold
        self.nan_threshold  = nan_threshold

        # Computed state — populated by fit()
        self.lag_matrix  = None
        self.corr_matrix = None
        self.clusters    = None
        self.ranked      = []   # list of (ordered_series, lags) tuples
        self.features    = None
        self.targets     = None
        self.diagnostic  = None
        self.time_report = None

    # ------------------------------------------------------------------
    # Public entry point
    # ------------------------------------------------------------------

    def fit(self, compute_diagnostics: bool = False):
        """Run the full pipeline: lag matrix → clusters → ordering → features."""
        # Reset all computed state so calling fit() twice is safe
        self.lag_matrix  = None
        self.corr_matrix = None
        self.clusters    = None
        self.ranked      = []
        self.features    = None
        self.targets     = None
        self.diagnostic  = None
        self.time_report = None

        t0 = time.time()
        self._compute_lag_matrix()
        t1 = time.time()

        self._cluster()
        t2 = time.time()

        self._order_all_clusters()
        self.features, self.targets, self.diagnostic = self._align_and_aggregate(compute_diagnostics)
        t3 = time.time()

        t_tot = t3 - t0
        self.time_report = {
            "Cleaning":   [t1 - t0,   100 * (t1 - t0) / t_tot],
            "Clustering": [t2 - t1,   100 * (t2 - t1) / t_tot],
            "Ordering":   [t3 - t2,   100 * (t3 - t2) / t_tot],
            "Total time": t_tot,
        }

    # ------------------------------------------------------------------
    # Pipeline stages
    # ------------------------------------------------------------------

    def _compute_lag_matrix(self):
        """Drop sparse series, compute pairwise cross-correlations, apply threshold.

        Builds self.lag_matrix and self.corr_matrix. Entries below
        corr_threshold are set to NaN so they are ignored by clustering.
        """
        price_data       = self.raw_data.loc[:, PRICE_COLS]
        nan_fraction     = price_data.isna().mean(axis=1)
        clean_price_data = price_data.loc[nan_fraction <= self.nan_threshold].fillna(0)

        lag_matrix, corr_matrix = self._compute_pairwise_cross_correlation(clean_price_data)

        lag_np  = lag_matrix.to_numpy()
        corr_np = corr_matrix.to_numpy()

        mask = corr_np <= self.corr_threshold
        lag_np[mask]  = np.nan
        corr_np[mask] = np.nan

        self.lag_matrix  = pd.DataFrame(lag_np,  index=lag_matrix.index,  columns=lag_matrix.columns)
        self.corr_matrix = pd.DataFrame(corr_np, index=corr_matrix.index, columns=corr_matrix.columns)

    def _cluster(self):
        """Group series into connected components based on the lag matrix."""
        self.clusters = self._find_connected_clusters(self.lag_matrix)

    def _order_all_clusters(self):
        """Order series within each cluster from most-leading to most-lagging."""
        self.ranked = []
        for cluster in self.clusters:
            lags_sub = self.lag_matrix.loc[cluster, cluster]
            ordered_series, lags = self._order_by_cumulative_lag(lags_sub)
            self.ranked.append((ordered_series, lags))

    def _align_and_aggregate(self, compute_diagnostics: bool = False):
        """Align each cluster on a shared time axis and aggregate into feature rows.

        For each cluster, series are time-shifted by their cumulative lag so
        that co-moving windows overlap. Returns (prices, volume, turnover) are
        then averaged column-wise across the cluster, producing one feature row.

        The prediction target is the return of ordered_series[-1] — the
        most-lagging (most recent timestamp) series, which is the one we want
        to predict using the leading series as signals.
        """
        rows        = []
        y_list      = []
        diagnostics = [] if compute_diagnostics else None

        if self.y_data is not None and "target" in self.y_data.columns:
            row_id_to_target = self.y_data.set_index("ROW_ID")["target"].to_dict()
        else:
            row_id_to_target = {}

        for ordered_series, lags in self.ranked:
            cluster_size = len(ordered_series)
            # Size the window exactly to this cluster — no truncation, no wasted columns
            feature_width = int(max(lags)) + WINDOW_LEN

            prices   = np.full((cluster_size, feature_width), np.nan)
            volume   = np.full((cluster_size, feature_width), np.nan)
            med_turn = np.full(feature_width, np.nan)
            cluster_row_ids = []

            for idx, series_index in enumerate(ordered_series):
                time_offset  = int(lags[idx])
                series_data  = self.raw_data.loc[series_index]

                prices[idx, time_offset:time_offset + WINDOW_LEN] = series_data.loc[PRICE_COLS].values
                volume[idx, time_offset:time_offset + WINDOW_LEN] = series_data.loc[VOL_COLS].values
                med_turn[time_offset]                              = series_data.loc["MEDIAN_DAILY_TURNOVER"]
                cluster_row_ids.append(series_data["ROW_ID"])

            # Target: future return of the most-lagging series.
            # ordered_series[-1] is the most-lagging (latest timestamp) in the cluster —
            # it is the one we want to predict, using the earlier series as leading indicators.
            last_row_id  = self.raw_data.loc[ordered_series[-1], "ROW_ID"]
            target_value = row_id_to_target.get(last_row_id, np.nan)
            y_list.append(target_value)

            with warnings.catch_warnings():
                warnings.simplefilter("ignore", category=RuntimeWarning)
                prices_mean = np.nanmean(prices, axis=0)
                prices_std  = np.nanstd(prices,  axis=0)
                vol_mean    = np.nanmean(volume,  axis=0)
                vol_std     = np.nanstd(volume,   axis=0)

            prices_mean = self._trim_and_right_align(prices_mean)
            vol_mean    = self._trim_and_right_align(vol_mean)
            med_turn    = self._trim_and_right_align(med_turn)

            feature_width = len(prices_mean)
            price_cols = [f"RET_{feature_width - j}"                   for j in range(feature_width)]
            vol_cols   = [f"SIGNED_VOLUME_{feature_width - j}"         for j in range(feature_width)]
            turn_cols  = [f"MEDIAN_DAILY_TURNOVER_{feature_width - j}" for j in range(feature_width)]

            row_df = pd.DataFrame(
                [np.concatenate([prices_mean, vol_mean, med_turn])],
                columns=price_cols + vol_cols + turn_cols,
            )
            row_df["SOURCE_ROW_IDS"] = ";".join(str(r) for r in cluster_row_ids)
            rows.append(row_df)

            if compute_diagnostics:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", category=RuntimeWarning)
                    abs_price_mean = np.nanmean(np.abs(prices_mean))
                    abs_vol_mean   = np.nanmean(np.abs(vol_mean))
                    diagnostics.append({
                        "prices_full": prices,
                        "volume_full": volume,
                        "relative_prices_spread": self._trim_and_right_align(
                            np.nanmax(prices, axis=0) - np.nanmin(prices, axis=0)
                        ) / abs_price_mean,
                        "relative_volume_spread": self._trim_and_right_align(
                            (np.nanmax(volume, axis=0) - np.nanmin(volume, axis=0)) / abs_vol_mean
                        ),
                        "prices_std": self._trim_and_right_align(prices_std),
                        "volume_std": self._trim_and_right_align(vol_std),
                        "prices_cv": np.nanmean(prices_std) / abs_price_mean,
                        "volume_cv": np.nanmean(vol_std)    / abs_vol_mean,
                    })

        features_out = pd.concat(rows, ignore_index=True)
        features_out["ALLOCATION"] = self.allocation
        features_out["GROUP"]      = self.raw_data.iloc[0]["GROUP"]

        targets_out = pd.DataFrame({"TARGET": y_list})
        targets_out["LABEL"] = (targets_out["TARGET"] > 0).astype(int)

        return features_out, targets_out, diagnostics

    # ------------------------------------------------------------------
    # Internal algorithms
    # ------------------------------------------------------------------

    def _compute_pairwise_cross_correlation(
        self,
        time_series: pd.DataFrame,
        min_overlap: int = 15,
        min_std: float = 1e-8,
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Compute pairwise lag and peak correlation for all series via FFT.

        Returns two (n × n) DataFrames:
          lag_matrix  — lag[r, c] > 0 means row r leads row c by that many steps
          corr_matrix — peak normalized cross-correlation for each pair, in [-1, 1]

        Only lags where the overlapping window is >= min_overlap are considered,
        preventing unreliable estimates from near-zero overlap windows.
        """
        idx    = time_series.index
        values = time_series.to_numpy()
        stds   = values.std(axis=1)

        # Drop series with no variance — they carry no lag information and cause division by zero
        valid_mask = stds > min_std
        if not valid_mask.all():
            dropped = idx[~valid_mask].tolist()
            warnings.warn(
                f"_compute_pairwise_cross_correlation: dropping {len(dropped)} "
                f"constant/near-constant series at indices {dropped}"
            )
            values = values[valid_mask]
            idx    = idx[valid_mask]
            stds   = stds[valid_mask]

        n, t            = values.shape
        means           = values.mean(axis=1)
        values_centered = values - means[:, np.newaxis]  # (n, t)

        n_fft = 2 * t

        # Map each FFT output index to its lag value and overlap size.
        # fftfreq with d=1/n_fft gives integer lags: [0, 1, ..., t-1, -t, -(t-1), ..., -1]
        # Positive lag k → row r leads row c by k steps (same convention as np.correlate)
        lags_fft     = np.fft.fftfreq(n_fft, d=1.0 / n_fft).astype(int)  # (n_fft,)
        overlaps_fft = (t - np.abs(lags_fft)).astype(float)                # (n_fft,)
        valid_fft    = overlaps_fft >= min_overlap                          # (n_fft,)

        # FFT all series simultaneously — shape (n, n_fft//2+1) complex
        F = np.fft.rfft(values_centered, n=n_fft, axis=1)

        # All pairwise cross-correlations in one shot.
        # F[r] * conj(F[c]) → irfft gives sum_t c[t]*r[t+k], maximised at positive k when r leads c.
        xcorr_fft = F[:, np.newaxis, :] * np.conj(F[np.newaxis, :, :])  # (n, n, n_fft//2+1)
        xcorr_all = np.fft.irfft(xcorr_fft, n=n_fft, axis=2)             # (n, n, n_fft)

        std_outer  = stds[:, np.newaxis] * stds[np.newaxis, :]  # (n, n)
        xcorr_norm = xcorr_all / (std_outer[:, :, np.newaxis] * t)

        xcorr_norm[:, :, ~valid_fft] = -np.inf

        # Find the lag of maximum correlation for every pair simultaneously
        max_idx = np.argmax(xcorr_norm, axis=2)  # (n, n)
        lag  = lags_fft[max_idx].astype(float)
        corr = xcorr_norm[
            np.arange(n)[:, np.newaxis],
            np.arange(n)[np.newaxis, :],
            max_idx,
        ]

        np.fill_diagonal(lag,  0.0)
        np.fill_diagonal(corr, 1.0)

        return pd.DataFrame(lag,  index=idx, columns=idx), \
               pd.DataFrame(corr, index=idx, columns=idx)

    def _find_connected_clusters(self, pairwise_lags: pd.DataFrame) -> list:
        """Return connected components of the lag graph as sorted lists of indices.

        Two series belong to the same cluster if there is a non-NaN lag between
        them (directly or transitively). Uses iterative DFS to avoid recursion
        limits on large allocations.
        """
        labels = pairwise_lags.index
        neighbors_dict = {
            row: pairwise_lags.loc[row].dropna().index.tolist()
            for row in labels
        }

        unvisited = set(labels)
        clusters  = []

        while unvisited:
            seed    = unvisited.pop()
            cluster = {seed}
            stack   = [seed]

            while stack:
                current = stack.pop()
                for neighbor in neighbors_dict[current]:
                    if neighbor in unvisited:
                        unvisited.remove(neighbor)
                        stack.append(neighbor)
                        cluster.add(neighbor)

            clusters.append(sorted(cluster))

        return clusters

    def _order_by_cumulative_lag(self, pairwise_lags: pd.DataFrame):
        """Order series from most-leading to most-lagging using greedy graph traversal.

        Starting from the series closest to the median lag sum (the "middle"
        of the cluster), the algorithm greedily places the next series by
        accumulating pairwise lags along the shortest available chain. The
        final positions are normalized so the most-leading series has lag 0.

        Returns
        -------
        ordered_series : list of index labels, most-leading first
        lags           : np.ndarray of cumulative lag offsets, starting at 0
        """
        labels     = pairwise_lags.index.tolist()
        label_to_i = {lbl: i for i, lbl in enumerate(labels)}
        lag_np     = pairwise_lags.to_numpy()  # convert once; avoids O(n²) pandas .loc overhead

        col_sums   = pairwise_lags.sum(skipna=True)
        start_node = (col_sums - col_sums.median()).abs().idxmin()

        unvisited      = set(labels)
        sequence       = []
        cumulative_lag = {start_node: 0}
        sequence.append(start_node)
        unvisited.remove(start_node)

        while unvisited:
            candidates = {}

            # Iterate most-recently-placed first to minimise cumulative chain length
            for placed_node in reversed(sequence):
                pi = label_to_i[placed_node]
                for neighbor in unvisited:
                    if neighbor not in candidates:
                        lag_value = lag_np[pi, label_to_i[neighbor]]
                        if not np.isnan(lag_value):
                            # Negate: lag[r,c] < 0 when r leads c, so subtracting gives
                            # earlier series a smaller cumulative lag (placed left/first).
                            candidates[neighbor] = cumulative_lag[placed_node] - lag_value

            if candidates:
                next_node = max(candidates, key=lambda k: abs(candidates[k]))
                cumulative_lag[next_node] = candidates[next_node]
                sequence.append(next_node)
                unvisited.remove(next_node)
            else:
                # Disconnected node — place at lag 0 (treated as a separate leading series)
                disconnected = unvisited.pop()
                cumulative_lag[disconnected] = 0
                sequence.append(disconnected)

        lag_series   = pd.Series(cumulative_lag).sort_values()
        ordered_lags = lag_series.values - lag_series.values.min()

        return lag_series.index.tolist(), ordered_lags

    @staticmethod
    def _trim_and_right_align(array: np.ndarray) -> np.ndarray:
        """Remove leading and trailing NaN, then right-pad back to original length.

        Right-alignment ensures that index -1 (RET_1) always corresponds to
        the most recent data point, regardless of cluster length.
        """
        original_len = len(array)
        valid_idx    = ~np.isnan(array)
        if valid_idx.any():
            first   = np.argmax(valid_idx)
            last    = len(valid_idx) - np.argmax(valid_idx[::-1])
            trimmed = array[first:last]
            padded  = np.full(original_len, np.nan)
            padded[-len(trimmed):] = trimmed
            return padded
        return np.full(original_len, np.nan)
