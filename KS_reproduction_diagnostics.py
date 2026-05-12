"""
Diagnose how closely a Python port of the TiPES/Rousseau augmented KS
workflow reproduces the published Rousseau et al. (2023) Chinese speleothem
transition catalogue.

This script is deliberately a reproduction diagnostic, not a new event
catalogue generator. The published transition ages are used only after the KS
detections have been produced, in order to identify which implementation or
parameter details might explain the remaining mismatch between our local port
and the published tables.

Outputs
-------
data/processed/rousseau2023_ks_reproduction_diagnostics/
    parameter_sweep_summary.csv
    best_candidate_detected_transitions.csv
    best_candidate_matches.csv
    best_candidate_unmatched_detected.csv
    best_candidate_unmatched_published.csv
    declared_parameter_matches.csv
    parameters.csv

figures/rousseau2023_ks_reproduction_diagnostics/
    fig01_best_reproduction_overlay.{png,pdf}
    fig02_top_parameter_sweep.{png,pdf}
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parent
RUN_NAME = "rousseau2023_ks_reproduction_diagnostics"
OUT_DATA_DIR = PROJECT_ROOT / "data" / "processed" / RUN_NAME
OUT_FIG_DIR = PROJECT_ROOT / "figures" / RUN_NAME

CHENG_XLSX = PROJECT_ROOT / "data/raw/Cheng_2016.xlsx"

PUBLISHED_CATALOGUES = {
    ("0.4-4", "strong_monsoon_start"): PROJECT_ROOT
    / "data/raw/rousseau_2023_ks_0p4_4kyr_strong_monsoon_start_times.csv",
    ("0.4-4", "weak_monsoon_start"): PROJECT_ROOT
    / "data/raw/rousseau_2023_ks_0p4_4kyr_weak_monsoon_start_times.csv",
    ("0.6-4", "strong_monsoon_start"): PROJECT_ROOT
    / "data/raw/rousseau_2023_ks_0p6_4kyr_strong_monsoon_start_times.csv",
    ("0.6-4", "weak_monsoon_start"): PROJECT_ROOT
    / "data/raw/rousseau_2023_ks_0p6_4kyr_weak_monsoon_start_times.csv",
}

EVENT_SETTINGS = {
    "strong_monsoon_start": {
        "label": "Strong monsoon starts",
        "short_label": "strong",
        "color": "#d95f02",
        # Cheng speleothem d18O becomes more negative for stronger monsoon.
        # In TiPES, jump_direction=-1 means the mean after the candidate time
        # is lower than the mean before it.
        "jump_direction": -1,
    },
    "weak_monsoon_start": {
        "label": "Weak monsoon starts",
        "short_label": "weak",
        "color": "#6a3d9a",
        "jump_direction": 1,
    },
}

ANALYSIS_START_KA = 0.0
ANALYSIS_END_KA = 640.0
WINDOW_RANGES = {
    "0.4-4": (0.4, 4.0),
    "0.6-4": (0.6, 4.0),
}
N_WINDOWS = 15
N_CUT = 3

# The paper states that KS values above 0.7 are significant. TiPES defaults
# are d_c=0.75, s_c=1.5, and x_c=std(x)*0.1+range(x)*0.05. Because Rousseau
# et al. do not report every auxiliary threshold in the text, we sweep a small
# transparent grid around TiPES defaults to localize the mismatch.
D_CUT_GRID = [0.68, 0.70, 0.72, 0.75, 0.77]
SIGMA_CUT_GRID = [1.0, 1.5, 2.0]
CHANGE_CUT_GRID: list[float | str] = ["default", 0.0, 0.2, 0.4]
MATCH_TOLERANCES_KA = [0.8]
PRIMARY_MATCH_TOLERANCE_KA = 0.8
EDGE_NAN_BRANCHES = [False]


plt.rcParams.update(
    {
        "font.family": "DejaVu Sans",
        "font.size": 9,
        "axes.labelsize": 9,
        "axes.titlesize": 10,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "legend.fontsize": 8,
        "axes.linewidth": 0.8,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    }
)


@dataclass(frozen=True)
class DetectorParams:
    window_label: str
    w_min: float
    w_max: float
    n_windows: int
    d_cut: float
    n_cut: int
    sigma_cut: float
    change_cut: float | str
    edge_nan_ksstat: bool = False


@dataclass
class KSArrays:
    window_label: str
    age_ka: np.ndarray
    values: np.ndarray
    kswindow: np.ndarray
    ksstat: np.ndarray
    change_ks: np.ndarray
    kslen1: np.ndarray
    kslen2: np.ndarray
    ksstd1: np.ndarray
    ksstd2: np.ndarray
    default_change_cut: float


def ensure_dirs() -> None:
    OUT_DATA_DIR.mkdir(parents=True, exist_ok=True)
    OUT_FIG_DIR.mkdir(parents=True, exist_ok=True)


def load_cheng_composite() -> pd.DataFrame:
    raw = pd.read_excel(CHENG_XLSX, sheet_name="Composite record")
    age_col = next(col for col in raw.columns if "age" in str(col).lower())
    value_col = next(col for col in raw.columns if "18" in str(col).lower() or "δ" in str(col))
    out = pd.DataFrame(
        {
            "age_ka": pd.to_numeric(raw[age_col], errors="coerce"),
            "d18o": pd.to_numeric(raw[value_col], errors="coerce"),
        }
    )
    out = out.dropna(subset=["age_ka", "d18o"])
    out = out[out["age_ka"].between(ANALYSIS_START_KA, ANALYSIS_END_KA, inclusive="both")]
    out = out.groupby("age_ka", as_index=False)["d18o"].mean()
    return out.sort_values("age_ka").reset_index(drop=True)


def load_published_catalogues() -> pd.DataFrame:
    rows = []
    for (window_label, event_type), path in PUBLISHED_CATALOGUES.items():
        df = pd.read_csv(path, encoding="utf-8-sig")
        ages = pd.to_numeric(df["start_time_ka_BP"], errors="coerce")
        sub = pd.DataFrame(
            {
                "window_label": window_label,
                "event_type": event_type,
                "event_label": EVENT_SETTINGS[event_type]["label"],
                "published_age_ka": ages,
                "published_order": pd.to_numeric(df.get("order"), errors="coerce"),
                "source_file": str(path.relative_to(PROJECT_ROOT)),
            }
        )
        rows.append(sub.dropna(subset=["published_age_ka"]))
    return pd.concat(rows, ignore_index=True).sort_values(
        ["window_label", "event_type", "published_age_ka"]
    )


def tipes_window_grid(w_min: float, w_max: float, n_windows: int) -> np.ndarray:
    if w_min == w_max or n_windows == 1:
        return np.array([w_min], dtype=float)
    idx = np.arange(n_windows, dtype=float)
    return w_min * (w_max / w_min) ** (idx / (n_windows - 1.0))


def two_sample_ks_statistic(sample_a: np.ndarray, sample_b: np.ndarray) -> float:
    a = np.sort(np.asarray(sample_a, dtype=float))
    b = np.sort(np.asarray(sample_b, dtype=float))
    values = np.sort(np.concatenate([a, b]))
    cdf_a = np.searchsorted(a, values, side="right") / float(len(a))
    cdf_b = np.searchsorted(b, values, side="right") / float(len(b))
    return float(np.max(np.abs(cdf_a - cdf_b)))


def rowwise_nanmean(frame: np.ndarray) -> np.ndarray:
    with np.errstate(invalid="ignore"):
        return np.nanmean(frame, axis=1)


def compute_ks_arrays(
    composite: pd.DataFrame,
    window_label: str,
    edge_nan_ksstat: bool,
) -> KSArrays:
    w_min, w_max = WINDOW_RANGES[window_label]
    frame = composite[["age_ka", "d18o"]].dropna().copy()
    frame = frame.groupby("age_ka", as_index=False)["d18o"].mean().sort_values("age_ka")
    tt = frame["age_ka"].to_numpy(dtype=float)
    xx = frame["d18o"].to_numpy(dtype=float)

    # Faithful to TiPES: if a record crosses zero, shift it positive before
    # computing std/change thresholds. The Cheng composite is all negative, so
    # this is usually inactive but harmless.
    if np.sign(np.nanmax(xx)) > np.sign(np.nanmin(xx)):
        xx = xx - np.nanmin(xx) + 0.1

    kswindow = tipes_window_grid(w_min, w_max, N_WINDOWS)
    n = len(tt)
    n_w = len(kswindow)
    ksstat = np.zeros((n, n_w), dtype=float)
    change_ks = np.full((n, n_w), np.nan, dtype=float)
    kslen1 = np.zeros((n, n_w), dtype=float)
    kslen2 = np.zeros((n, n_w), dtype=float)
    ksstd1 = np.full((n, n_w), np.nan, dtype=float)
    ksstd2 = np.full((n, n_w), np.nan, dtype=float)

    # Use contiguous slices rather than fancy-index arrays. This keeps the
    # Python port much faster while preserving TiPES' window definitions:
    # sample 1 is (t-w, t] and sample 2 is (t, t+w].
    for col, window in enumerate(kswindow):
        left_start = np.searchsorted(tt, tt - window, side="right")
        right_end = np.searchsorted(tt, tt + window, side="right")
        for idx in range(n):
            r1_start = int(left_start[idx])
            r1_stop = idx + 1
            r2_start = idx + 1
            r2_stop = int(right_end[idx])
            len1 = r1_stop - r1_start
            len2 = r2_stop - r2_start
            full_window_available = (
                len1 > 0
                and len2 > 0
                and np.nanmin(tt) <= tt[idx] - window
                and np.nanmax(tt) >= tt[idx] + window
            )
            ks = (
                two_sample_ks_statistic(xx[r1_start:r1_stop], xx[r2_start:r2_stop])
                if full_window_available
                else 0.0
            )
            if len1 > 0 and len2 > 0:
                before = xx[r1_start:r1_stop]
                after = xx[r2_start:r2_stop]
                change = float(np.nanmean(before) - np.nanmean(after))
                std1 = float(np.nanstd(before, ddof=1)) if len1 > 1 else 0.0
                std2 = float(np.nanstd(after, ddof=1)) if len2 > 1 else 0.0
            else:
                change = np.nan
                std1 = np.nan
                std2 = np.nan

            if np.isfinite(change):
                ksstat[idx, col] = ks * np.sign(change)
            else:
                # MATLAB's 0*sign(NaN) can propagate NaN. The default branch
                # follows our cleaner earlier port; the alternate branch lets
                # us test whether boundary NaN propagation matters.
                ksstat[idx, col] = np.nan if edge_nan_ksstat else 0.0
            change_ks[idx, col] = change
            kslen1[idx, col] = len1
            kslen2[idx, col] = len2
            ksstd1[idx, col] = std1
            ksstd2[idx, col] = std2

    default_change_cut = float(np.nanstd(xx, ddof=1) * 0.1 + (np.nanmax(xx) - np.nanmin(xx)) * 0.05)
    return KSArrays(
        window_label=window_label,
        age_ka=tt,
        values=xx,
        kswindow=kswindow,
        ksstat=ksstat,
        change_ks=change_ks,
        kslen1=kslen1,
        kslen2=kslen2,
        ksstd1=ksstd1,
        ksstd2=ksstd2,
        default_change_cut=default_change_cut,
    )


def sign_change_segments(values: np.ndarray) -> list[tuple[int, int]]:
    segments = []
    start = 0
    signs = np.sign(values)
    for idx in range(1, len(signs)):
        if signs[idx - 1] != signs[idx]:
            segments.append((start, idx))
            start = idx
    return segments


def extract_jumps(arrays: KSArrays, params: DetectorParams) -> pd.DataFrame:
    ksstat = arrays.ksstat
    change_ks = arrays.change_ks
    kslen1 = arrays.kslen1
    kslen2 = arrays.kslen2
    ksstd1 = arrays.ksstd1
    ksstd2 = arrays.ksstd2
    tt = arrays.age_ka
    n_w = len(arrays.kswindow)

    ksstat2 = ksstat.copy()
    if n_w > 2:
        for col in range(n_w - 1, 1, -1):
            smaller = rowwise_nanmean(ksstat[:, :col])
            ksstat2[:, col] = rowwise_nanmean(np.column_stack([ksstat[:, col], smaller]))
        ksstat2[:, 1] = rowwise_nanmean(np.column_stack([ksstat[:, 1], ksstat[:, 0]]))

    if params.change_cut == "default":
        x_cut = arrays.default_change_cut
    else:
        x_cut = float(params.change_cut)

    th_si = (kslen1 >= params.n_cut) & (kslen2 >= params.n_cut)
    th_st = (params.sigma_cut * ksstd1 <= np.abs(change_ks)) & (
        params.sigma_cut * ksstd2 <= np.abs(change_ks)
    )
    th_change = x_cut <= np.abs(change_ks)
    with np.errstate(divide="ignore", invalid="ignore"):
        finite_sample = np.sqrt((kslen1 + kslen2) / (kslen1 * kslen2))
        adjusted = 1.0 - (1.0 - np.abs(ksstat2)) / (1.0 - finite_sample)
    adjusted[np.abs(ksstat) == 1.0] = 1.0
    adjusted[adjusted < 0.0] = 0.0
    adjusted = adjusted * np.sign(change_ks)
    adjusted[~th_si] = 0.0
    if len(tt) >= 2:
        adjusted[0, :] = 0.0
        adjusted[-2, :] = 0.0
    th_ks = np.abs(adjusted) >= params.d_cut
    ks_all = adjusted * th_st * th_si * th_change * th_ks

    ksstat_m = ksstat.copy()
    if n_w > 1:
        ksstat_m = np.column_stack([ksstat_m, np.nanmean(ksstat_m, axis=1)])
        adjusted = np.column_stack([adjusted, np.nanmean(adjusted, axis=1)])
        ks_all = np.column_stack([ks_all, np.nanmean(ks_all, axis=1)])
        ks_all[np.abs(ks_all) < params.d_cut] = 0.0

    peaks_by_col: list[pd.DataFrame] = []
    for col in range(ks_all.shape[1]):
        peaks = []
        for start, stop in sign_change_segments(ksstat_m[:, col]):
            segment = ks_all[start:stop, col]
            valid_nonzero = np.isfinite(segment) & (segment != 0.0)
            if len(segment) == 0 or not np.any(valid_nonzero):
                continue
            finite_segment = np.where(np.isfinite(segment), segment, 0.0)
            candidate_local = np.flatnonzero(
                finite_segment == finite_segment[np.argmax(np.abs(finite_segment))]
            )
            candidate_idx = candidate_local + start
            row_score = np.abs(np.nansum(adjusted[candidate_idx, :], axis=1))
            chosen = candidate_idx[np.flatnonzero(row_score == np.nanmax(row_score))[-1]]
            peaks.append({"index": int(chosen), "value": float(ks_all[chosen, col])})
        peaks_by_col.append(pd.DataFrame(peaks))

    if not any(not peaks.empty for peaks in peaks_by_col):
        return empty_detection_frame()

    def peak_times(peaks: pd.DataFrame, sign: int) -> np.ndarray:
        if peaks.empty:
            return np.array([], dtype=float)
        sub = peaks[peaks["value"] > 0.0] if sign > 0 else peaks[peaks["value"] < 0.0]
        if sub.empty:
            return np.array([], dtype=float)
        idx = sub["index"].to_numpy(dtype=int)
        idx = idx[idx + 1 < len(tt)]
        return 0.5 * (tt[idx] + tt[idx + 1])

    n_cols = len(peaks_by_col)
    jump_up = peak_times(peaks_by_col[-1], sign=1)
    jump_down = peak_times(peaks_by_col[-1], sign=-1)
    if n_cols > 1:
        for col in range(n_cols - 2, -1, -1):
            window = arrays.kswindow[col] if col < len(arrays.kswindow) else arrays.kswindow[-1]
            up = peak_times(peaks_by_col[col], sign=1)
            down = peak_times(peaks_by_col[col], sign=-1)
            for existing in jump_up:
                up = up[(up >= existing + window) | (up <= existing - window)]
            for existing in jump_down:
                down = down[(down >= existing + window) | (down <= existing - window)]
            jump_up = np.concatenate([jump_up, up])
            jump_down = np.concatenate([jump_down, down])

    rows = []
    for age in jump_up:
        rows.append(
            {
                "age_ka": float(age),
                "jump_direction": 1,
                "event_type": "weak_monsoon_start",
                "window_label": params.window_label,
                "d_cut": params.d_cut,
                "sigma_cut": params.sigma_cut,
                "change_cut": x_cut,
                "change_cut_setting": str(params.change_cut),
                "edge_nan_ksstat": params.edge_nan_ksstat,
            }
        )
    for age in jump_down:
        rows.append(
            {
                "age_ka": float(age),
                "jump_direction": -1,
                "event_type": "strong_monsoon_start",
                "window_label": params.window_label,
                "d_cut": params.d_cut,
                "sigma_cut": params.sigma_cut,
                "change_cut": x_cut,
                "change_cut_setting": str(params.change_cut),
                "edge_nan_ksstat": params.edge_nan_ksstat,
            }
        )
    if not rows:
        return empty_detection_frame()
    out = pd.DataFrame(rows).sort_values(["event_type", "age_ka"]).reset_index(drop=True)
    out.insert(0, "transition_id", np.arange(1, len(out) + 1))
    return out


def empty_detection_frame() -> pd.DataFrame:
    return pd.DataFrame(
        columns=[
            "transition_id",
            "age_ka",
            "jump_direction",
            "event_type",
            "window_label",
            "d_cut",
            "sigma_cut",
            "change_cut",
            "change_cut_setting",
            "edge_nan_ksstat",
        ]
    )


def greedy_match(
    detected: pd.DataFrame,
    published: pd.DataFrame,
    window_label: str,
    event_type: str,
    tolerance_ka: float,
) -> pd.DataFrame:
    det = detected[
        detected["window_label"].eq(window_label) & detected["event_type"].eq(event_type)
    ].copy()
    ref = published[
        published["window_label"].eq(window_label) & published["event_type"].eq(event_type)
    ].copy()
    pairs = []
    for det_idx, drow in det.iterrows():
        for ref_idx, rrow in ref.iterrows():
            error = float(drow["age_ka"] - rrow["published_age_ka"])
            if abs(error) <= tolerance_ka:
                pairs.append(
                    {
                        "window_label": window_label,
                        "event_type": event_type,
                        "detected_index": det_idx,
                        "published_index": ref_idx,
                        "detected_age_ka": float(drow["age_ka"]),
                        "published_age_ka": float(rrow["published_age_ka"]),
                        "age_error_ka": error,
                        "abs_age_error_ka": abs(error),
                        "tolerance_ka": tolerance_ka,
                    }
                )
    if not pairs:
        return pd.DataFrame(columns=["window_label", "event_type"])
    pairs = sorted(pairs, key=lambda row: (row["abs_age_error_ka"], row["detected_age_ka"]))
    used_det: set[int] = set()
    used_ref: set[int] = set()
    matched = []
    for row in pairs:
        if row["detected_index"] in used_det or row["published_index"] in used_ref:
            continue
        used_det.add(row["detected_index"])
        used_ref.add(row["published_index"])
        matched.append(row)
    return pd.DataFrame(matched)


def summarize_detection(
    detected: pd.DataFrame,
    published: pd.DataFrame,
    params: DetectorParams,
    tolerance_ka: float,
) -> tuple[list[dict], pd.DataFrame]:
    rows = []
    match_frames = []
    for event_type in EVENT_SETTINGS:
        det_count = int(
            detected[
                detected["window_label"].eq(params.window_label)
                & detected["event_type"].eq(event_type)
            ].shape[0]
        )
        pub_count = int(
            published[
                published["window_label"].eq(params.window_label)
                & published["event_type"].eq(event_type)
            ].shape[0]
        )
        matched = greedy_match(detected, published, params.window_label, event_type, tolerance_ka)
        if not matched.empty:
            match_frames.append(matched)
        n_matched = int(len(matched))
        precision = n_matched / det_count if det_count else np.nan
        recall = n_matched / pub_count if pub_count else np.nan
        rows.append(
            {
                "window_label": params.window_label,
                "event_type": event_type,
                "event_label": EVENT_SETTINGS[event_type]["label"],
                "d_cut": params.d_cut,
                "sigma_cut": params.sigma_cut,
                "change_cut_setting": str(params.change_cut),
                "edge_nan_ksstat": params.edge_nan_ksstat,
                "effective_change_cut": np.nan
                if detected.empty
                else float(detected["change_cut"].dropna().iloc[0]),
                "match_tolerance_ka": tolerance_ka,
                "n_detected": det_count,
                "n_published": pub_count,
                "n_matched": n_matched,
                "n_unmatched_detected": det_count - n_matched,
                "n_unmatched_published": pub_count - n_matched,
                "precision": precision,
                "recall": recall,
                "f1": 2 * precision * recall / (precision + recall)
                if np.isfinite(precision) and np.isfinite(recall) and precision + recall > 0
                else np.nan,
                "median_abs_age_error_ka": float(matched["abs_age_error_ka"].median())
                if n_matched
                else np.nan,
                "mean_abs_age_error_ka": float(matched["abs_age_error_ka"].mean())
                if n_matched
                else np.nan,
            }
        )
    all_matches = pd.concat(match_frames, ignore_index=True) if match_frames else pd.DataFrame()
    return rows, all_matches


def aggregate_sweep_rows(rows: list[dict]) -> pd.DataFrame:
    frame = pd.DataFrame(rows)
    group_cols = [
        "window_label",
        "d_cut",
        "sigma_cut",
        "change_cut_setting",
        "edge_nan_ksstat",
        "match_tolerance_ka",
    ]
    agg = (
        frame.groupby(group_cols, as_index=False)
        .agg(
            n_detected_total=("n_detected", "sum"),
            n_published_total=("n_published", "sum"),
            n_matched_total=("n_matched", "sum"),
            n_unmatched_detected_total=("n_unmatched_detected", "sum"),
            n_unmatched_published_total=("n_unmatched_published", "sum"),
            median_abs_age_error_ka=("median_abs_age_error_ka", "median"),
            mean_abs_age_error_ka=("mean_abs_age_error_ka", "mean"),
        )
        .reset_index(drop=True)
    )
    agg["precision_total"] = agg["n_matched_total"] / agg["n_detected_total"]
    agg["recall_total"] = agg["n_matched_total"] / agg["n_published_total"]
    agg["f1_total"] = (
        2
        * agg["precision_total"]
        * agg["recall_total"]
        / (agg["precision_total"] + agg["recall_total"])
    )
    agg["unmatched_total"] = (
        agg["n_unmatched_detected_total"] + agg["n_unmatched_published_total"]
    )
    return agg


def build_unmatched_tables(
    detected: pd.DataFrame,
    published: pd.DataFrame,
    matches: pd.DataFrame,
    window_label: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    unmatched_detected = []
    unmatched_published = []
    for event_type in EVENT_SETTINGS:
        det = detected[
            detected["window_label"].eq(window_label) & detected["event_type"].eq(event_type)
        ].copy()
        ref = published[
            published["window_label"].eq(window_label) & published["event_type"].eq(event_type)
        ].copy()
        matched = matches[matches["event_type"].eq(event_type)] if not matches.empty else pd.DataFrame()
        used_det = set(matched["detected_index"].astype(int)) if not matched.empty else set()
        used_ref = set(matched["published_index"].astype(int)) if not matched.empty else set()

        det_un = det.loc[[idx for idx in det.index if idx not in used_det]].copy()
        ref_un = ref.loc[[idx for idx in ref.index if idx not in used_ref]].copy()
        for frame, age_col in [(det_un, "age_ka"), (ref_un, "published_age_ka")]:
            if frame.empty:
                continue
            other_ages = (
                ref["published_age_ka"].to_numpy(dtype=float)
                if age_col == "age_ka"
                else det["age_ka"].to_numpy(dtype=float)
            )
            if len(other_ages):
                nearest = []
                nearest_error = []
                for age in frame[age_col].to_numpy(dtype=float):
                    idx = int(np.argmin(np.abs(other_ages - age)))
                    nearest.append(float(other_ages[idx]))
                    nearest_error.append(float(age - other_ages[idx]))
                frame["nearest_opposite_catalogue_age_ka"] = nearest
                frame["nearest_error_ka"] = nearest_error
            else:
                frame["nearest_opposite_catalogue_age_ka"] = np.nan
                frame["nearest_error_ka"] = np.nan
        if not det_un.empty:
            unmatched_detected.append(det_un)
        if not ref_un.empty:
            unmatched_published.append(ref_un)

    return (
        pd.concat(unmatched_detected, ignore_index=True) if unmatched_detected else pd.DataFrame(),
        pd.concat(unmatched_published, ignore_index=True) if unmatched_published else pd.DataFrame(),
    )


def iter_parameter_grid(window_label: str, edge_nan_ksstat: bool) -> Iterable[DetectorParams]:
    w_min, w_max = WINDOW_RANGES[window_label]
    for d_cut in D_CUT_GRID:
        for sigma_cut in SIGMA_CUT_GRID:
            for change_cut in CHANGE_CUT_GRID:
                yield DetectorParams(
                    window_label=window_label,
                    w_min=w_min,
                    w_max=w_max,
                    n_windows=N_WINDOWS,
                    d_cut=d_cut,
                    n_cut=N_CUT,
                    sigma_cut=sigma_cut,
                    change_cut=change_cut,
                    edge_nan_ksstat=edge_nan_ksstat,
                )


def select_best_candidate(aggregate: pd.DataFrame, window_label: str) -> pd.Series:
    sub = aggregate[
        aggregate["window_label"].eq(window_label)
        & aggregate["match_tolerance_ka"].eq(PRIMARY_MATCH_TOLERANCE_KA)
    ].copy()
    sub = sub.sort_values(
        [
            "unmatched_total",
            "median_abs_age_error_ka",
            "n_unmatched_published_total",
            "n_unmatched_detected_total",
            "d_cut",
            "sigma_cut",
        ],
        ascending=[True, True, True, True, True, True],
    )
    return sub.iloc[0]


def plot_best_overlay(
    composite: pd.DataFrame,
    detected: pd.DataFrame,
    published: pd.DataFrame,
    best_rows: pd.DataFrame,
    write_pdf: bool = True,
) -> None:
    fig, axes = plt.subplots(2, 1, figsize=(8.2, 5.8), sharex=True, constrained_layout=True)
    for ax, (_, best) in zip(axes, best_rows.iterrows()):
        window_label = best["window_label"]
        ax.plot(composite["age_ka"], composite["d18o"], color="#222222", lw=0.55)
        ax.invert_yaxis()
        y0, y1 = ax.get_ylim()
        height = abs(y1 - y0)
        for event_type, settings in EVENT_SETTINGS.items():
            det_ages = detected[
                detected["window_label"].eq(window_label) & detected["event_type"].eq(event_type)
            ]["age_ka"].to_numpy(dtype=float)
            pub_ages = published[
                published["window_label"].eq(window_label) & published["event_type"].eq(event_type)
            ]["published_age_ka"].to_numpy(dtype=float)
            color = settings["color"]
            ax.vlines(pub_ages, y0, y0 + 0.10 * height, color=color, lw=0.9, alpha=0.45)
            ax.vlines(det_ages, y1 - 0.10 * height, y1, color=color, lw=0.9, alpha=0.95)
        ax.set_ylabel(r"$\delta^{18}$O")
        ax.set_title(
            f"{window_label} kyr, best diagnostic parameters: "
            f"D={best['d_cut']:.2f}, s={best['sigma_cut']:.2f}, "
            f"x={best['change_cut_setting']}"
        )
        ax.text(
            0.01,
            0.04,
            "bottom ticks: detected; top ticks: published",
            transform=ax.transAxes,
            ha="left",
            va="bottom",
            fontsize=8,
        )
    axes[-1].set_xlabel("Age (kyr BP)")
    handles = [
        plt.Line2D([0], [0], color=EVENT_SETTINGS[event]["color"], lw=1.6, label=settings["label"])
        for event, settings in EVENT_SETTINGS.items()
    ]
    axes[0].legend(handles=handles, loc="upper right", frameon=False)
    fig.savefig(OUT_FIG_DIR / "fig01_best_reproduction_overlay.png", dpi=300, bbox_inches="tight")
    if write_pdf:
        fig.savefig(OUT_FIG_DIR / "fig01_best_reproduction_overlay.pdf", bbox_inches="tight")
    plt.close(fig)


def plot_top_sweep(aggregate: pd.DataFrame, write_pdf: bool = True) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(8.4, 3.2), constrained_layout=True)
    for ax, window_label in zip(axes, WINDOW_RANGES):
        sub = aggregate[
            aggregate["window_label"].eq(window_label)
            & aggregate["match_tolerance_ka"].eq(PRIMARY_MATCH_TOLERANCE_KA)
            & aggregate["edge_nan_ksstat"].eq(False)
        ].copy()
        sub = sub.sort_values(["unmatched_total", "median_abs_age_error_ka"]).head(20)
        labels = [
            f"D={row.d_cut:.2f}, s={row.sigma_cut:.2f}, x={row.change_cut_setting}"
            for row in sub.itertuples()
        ]
        y = np.arange(len(sub))
        ax.barh(y, sub["unmatched_total"], color="#4c78a8", alpha=0.85)
        ax.set_yticks(y)
        ax.set_yticklabels(labels, fontsize=6.5)
        ax.invert_yaxis()
        ax.set_xlabel("Total unmatched events")
        ax.set_title(f"{window_label} kyr window")
        ax.grid(axis="x", color="0.88", lw=0.6)
    fig.savefig(OUT_FIG_DIR / "fig02_top_parameter_sweep.png", dpi=300, bbox_inches="tight")
    if write_pdf:
        fig.savefig(OUT_FIG_DIR / "fig02_top_parameter_sweep.pdf", bbox_inches="tight")
    plt.close(fig)


def run() -> None:
    ensure_dirs()
    composite = load_cheng_composite()
    published = load_published_catalogues()
    published.to_csv(OUT_DATA_DIR / "published_catalogues_used.csv", index=False)

    all_detail_rows: list[dict] = []
    all_detections: dict[tuple[str, float, float, str, bool], pd.DataFrame] = {}
    declared_rows = []
    declared_matches = []

    # Compute KS matrices once for each window and edge-handling branch.
    arrays_by_key: dict[tuple[str, bool], KSArrays] = {}
    for window_label in WINDOW_RANGES:
        for edge_nan in EDGE_NAN_BRANCHES:
            arrays_by_key[(window_label, edge_nan)] = compute_ks_arrays(
                composite, window_label, edge_nan_ksstat=edge_nan
            )

    for window_label in WINDOW_RANGES:
        for edge_nan in EDGE_NAN_BRANCHES:
            arrays = arrays_by_key[(window_label, edge_nan)]
            for params in iter_parameter_grid(window_label, edge_nan):
                detected = extract_jumps(arrays, params)
                key = (
                    params.window_label,
                    params.d_cut,
                    params.sigma_cut,
                    str(params.change_cut),
                    params.edge_nan_ksstat,
                )
                all_detections[key] = detected
                for tolerance in MATCH_TOLERANCES_KA:
                    rows, matches = summarize_detection(detected, published, params, tolerance)
                    all_detail_rows.extend(rows)

                if (
                    params.d_cut == 0.70
                    and params.sigma_cut == 1.5
                    and params.change_cut == "default"
                    and params.edge_nan_ksstat is False
                ):
                    rows, matches = summarize_detection(
                        detected, published, params, PRIMARY_MATCH_TOLERANCE_KA
                    )
                    declared_rows.extend(rows)
                    if not matches.empty:
                        matches = matches.copy()
                        matches["d_cut"] = params.d_cut
                        matches["sigma_cut"] = params.sigma_cut
                        matches["change_cut_setting"] = str(params.change_cut)
                        declared_matches.append(matches)

    detail = pd.DataFrame(all_detail_rows)
    detail.to_csv(OUT_DATA_DIR / "parameter_sweep_detail_by_event.csv", index=False)
    aggregate = aggregate_sweep_rows(all_detail_rows)
    aggregate = aggregate.sort_values(
        ["window_label", "match_tolerance_ka", "unmatched_total", "median_abs_age_error_ka"]
    )
    aggregate.to_csv(OUT_DATA_DIR / "parameter_sweep_summary.csv", index=False)

    pd.DataFrame(declared_rows).to_csv(OUT_DATA_DIR / "declared_parameter_summary.csv", index=False)
    if declared_matches:
        pd.concat(declared_matches, ignore_index=True).to_csv(
            OUT_DATA_DIR / "declared_parameter_matches.csv", index=False
        )

    best_rows = []
    best_detected_frames = []
    best_match_frames = []
    best_unmatched_detected = []
    best_unmatched_published = []
    for window_label in WINDOW_RANGES:
        best = select_best_candidate(aggregate, window_label)
        best_rows.append(best)
        key = (
            best["window_label"],
            float(best["d_cut"]),
            float(best["sigma_cut"]),
            str(best["change_cut_setting"]),
            bool(best["edge_nan_ksstat"]),
        )
        detected = all_detections[key].copy()
        detected["best_for_window"] = window_label
        rows, matches = summarize_detection(
            detected,
            published,
            DetectorParams(
                window_label=window_label,
                w_min=WINDOW_RANGES[window_label][0],
                w_max=WINDOW_RANGES[window_label][1],
                n_windows=N_WINDOWS,
                d_cut=float(best["d_cut"]),
                n_cut=N_CUT,
                sigma_cut=float(best["sigma_cut"]),
                change_cut=str(best["change_cut_setting"])
                if best["change_cut_setting"] == "default"
                else float(best["change_cut_setting"]),
                edge_nan_ksstat=bool(best["edge_nan_ksstat"]),
            ),
            PRIMARY_MATCH_TOLERANCE_KA,
        )
        if not matches.empty:
            matches["best_for_window"] = window_label
            best_match_frames.append(matches)
        un_det, un_pub = build_unmatched_tables(detected, published, matches, window_label)
        if not un_det.empty:
            un_det["best_for_window"] = window_label
            best_unmatched_detected.append(un_det)
        if not un_pub.empty:
            un_pub["best_for_window"] = window_label
            best_unmatched_published.append(un_pub)
        best_detected_frames.append(detected)

    best_frame = pd.DataFrame(best_rows)
    best_frame.to_csv(OUT_DATA_DIR / "best_parameter_by_window.csv", index=False)
    best_detected = pd.concat(best_detected_frames, ignore_index=True)
    best_detected.to_csv(OUT_DATA_DIR / "best_candidate_detected_transitions.csv", index=False)
    best_matches = (
        pd.concat(best_match_frames, ignore_index=True) if best_match_frames else pd.DataFrame()
    )
    best_matches.to_csv(OUT_DATA_DIR / "best_candidate_matches.csv", index=False)
    (
        pd.concat(best_unmatched_detected, ignore_index=True)
        if best_unmatched_detected
        else pd.DataFrame()
    ).to_csv(OUT_DATA_DIR / "best_candidate_unmatched_detected.csv", index=False)
    (
        pd.concat(best_unmatched_published, ignore_index=True)
        if best_unmatched_published
        else pd.DataFrame()
    ).to_csv(OUT_DATA_DIR / "best_candidate_unmatched_published.csv", index=False)

    plot_best_overlay(composite, best_detected, published, best_frame)
    plot_top_sweep(aggregate)

    params_rows = []
    for window_label, (w_min, w_max) in WINDOW_RANGES.items():
        arrays = arrays_by_key[(window_label, False)]
        params_rows.append(
            {
                "window_label": window_label,
                "w_min": w_min,
                "w_max": w_max,
                "n_windows": N_WINDOWS,
                "n_cut": N_CUT,
                "default_change_cut": arrays.default_change_cut,
                "n_cheng_points": len(arrays.age_ka),
                "age_min_ka": arrays.age_ka.min(),
                "age_max_ka": arrays.age_ka.max(),
                "primary_match_tolerance_ka": PRIMARY_MATCH_TOLERANCE_KA,
            }
        )
    pd.DataFrame(params_rows).to_csv(OUT_DATA_DIR / "parameters.csv", index=False)

    print("\nBest diagnostic reproduction candidates:")
    print(
        best_frame[
            [
                "window_label",
                "d_cut",
                "sigma_cut",
                "change_cut_setting",
                "edge_nan_ksstat",
                "n_detected_total",
                "n_published_total",
                "n_matched_total",
                "unmatched_total",
                "precision_total",
                "recall_total",
                "median_abs_age_error_ka",
            ]
        ].to_string(index=False)
    )
    print("\nDeclared paper/TiPES-style settings (D=0.70, s=1.5, x=default):")
    declared = pd.DataFrame(declared_rows)
    print(
        declared[
            [
                "window_label",
                "event_type",
                "n_detected",
                "n_published",
                "n_matched",
                "precision",
                "recall",
                "median_abs_age_error_ka",
            ]
        ].to_string(index=False)
    )
    print(f"\nWrote diagnostics to {OUT_DATA_DIR.relative_to(PROJECT_ROOT)}")
    print(f"Wrote figures to {OUT_FIG_DIR.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    run()
