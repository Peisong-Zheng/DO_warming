"""
Sensitivity test for precession-band locking in the Rousseau et al. (2023)
transition catalogue.

Concern addressed
-----------------
The Rousseau catalogue is detected from the Cheng et al. (2016) composite
speleothem d18O curve, which contains strong precession-band variance. If a
KS-style detector preferentially flags the steep flanks of that orbital-band
waveform, apparent precession-phase organization of detected events could be a
catalogue-construction artifact rather than a property of abrupt millennial
transitions.

This script therefore:

1. implements a local Rousseau-style augmented two-sample KS detector;
2. applies fixed, pre-declared detector settings to the raw composite and to
   composite curves with precession-band variance removed;
3. compares the raw-composite detections against the published 0.4-4 kyr
   Rousseau catalogue only as a post-hoc reproduction check;
4. reruns only lightweight downstream checks on each detected catalogue:
   Rayleigh phase tests and the history/resolution-adjusted predictive
   Poisson information test for precession phase after LR04+CO2.

This is not intended to replace the published Rousseau catalogue. It is a
diagnostic for whether the main phase result survives when the orbital-scale
component of the input curve is suppressed before event detection. To avoid
information leakage, the published Rousseau transition ages are never used to
choose detection thresholds.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd
from scipy.signal import butter, sosfiltfilt

import Bin_hazard_phase_poisson as base
import Orbital_phase_rayleigh as rayleigh
import Predictive_hazard_history_resolution as predictive


PROJECT_ROOT = Path(__file__).resolve().parent
RUN_NAME = "rousseau2023_monsoon_ks_precession_band_removal_sensitivity"
OUT_DATA_DIR = PROJECT_ROOT / "data" / "processed" / RUN_NAME
OUT_FIG_DIR = PROJECT_ROOT / "figures" / RUN_NAME

CHENG_XLSX = PROJECT_ROOT / "data/raw/Cheng_2016.xlsx"
PUBLISHED_CATALOGUES = {
    "strong_monsoon_start": PROJECT_ROOT
    / "data/raw/rousseau_2023_ks_0p4_4kyr_strong_monsoon_start_times.csv",
    "weak_monsoon_start": PROJECT_ROOT
    / "data/raw/rousseau_2023_ks_0p4_4kyr_weak_monsoon_start_times.csv",
}

ANALYSIS_START_KA = 0.0
ANALYSIS_END_KA = 640.0
RESAMPLE_STEP_KA = 0.1
WINDOW_MIN_KA = 0.4
WINDOW_MAX_KA = 4.0
MATCH_TOLERANCE_KA = 0.8

# Fixed detector settings. D >= 0.7 follows the Rousseau et al. (2023) method
# description. Other settings follow the TiPES KS_detection.m defaults unless
# stated otherwise. They are not tuned against the published transition ages.
DETECTOR_N_WINDOWS = 15
DETECTOR_D_CUT = 0.70
DETECTOR_N_CUT = 3
DETECTOR_SIGMA_CUT = 1.50
DETECTOR_CHANGE_CUT = None

FILTER_VARIANTS = {
    "raw": {
        "label": "Raw composite",
        "bandstop_period_kyr": None,
        "color": "#222222",
    },
    "no_pre_18_25kyr": {
        "label": "18-25 kyr removed",
        "bandstop_period_kyr": (18.0, 25.0),
        "color": "#0072B2",
    },
    "no_pre_15_30kyr": {
        "label": "15-30 kyr removed",
        "bandstop_period_kyr": (15.0, 30.0),
        "color": "#D55E00",
    },
}

EVENT_SETTINGS = {
    "strong_monsoon_start": {
        "label": "Strong monsoon starts",
        "color": "#d95f02",
    },
    "weak_monsoon_start": {
        "label": "Weak monsoon starts",
        "color": "#6a3d9a",
    },
}


plt.rcParams.update(
    {
        "font.family": "DejaVu Sans",
        "font.size": 10,
        "axes.titlesize": 11,
        "axes.labelsize": 10,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "legend.fontsize": 8,
        "axes.linewidth": 0.9,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    }
)


@dataclass(frozen=True)
class DetectorParams:
    n_windows: int
    d_cut: float
    n_cut: int
    sigma_cut: float
    change_cut: float | None


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def save_figure(fig: plt.Figure, stem: str, write_pdf: bool) -> None:
    ensure_dir(OUT_FIG_DIR)
    fig.savefig(OUT_FIG_DIR / f"{stem}.png", dpi=300, bbox_inches="tight")
    if write_pdf:
        fig.savefig(OUT_FIG_DIR / f"{stem}.pdf", bbox_inches="tight")
    plt.close(fig)


def format_p_value(value: float) -> str:
    if not np.isfinite(value):
        return "NA"
    if value < 1e-4:
        return f"{value:.1e}"
    if value < 0.01:
        return f"{value:.4f}"
    return f"{value:.3f}"


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


def regularize_cheng(composite: pd.DataFrame) -> pd.DataFrame:
    start = max(
        ANALYSIS_START_KA,
        float(np.ceil(composite["age_ka"].min() / RESAMPLE_STEP_KA) * RESAMPLE_STEP_KA),
    )
    end = min(
        ANALYSIS_END_KA,
        float(np.floor(composite["age_ka"].max() / RESAMPLE_STEP_KA) * RESAMPLE_STEP_KA),
    )
    age_grid = np.round(np.arange(start, end + RESAMPLE_STEP_KA / 2.0, RESAMPLE_STEP_KA), 10)
    d18o = np.interp(age_grid, composite["age_ka"].to_numpy(), composite["d18o"].to_numpy())
    return pd.DataFrame({"age_ka": age_grid, "d18o_raw": d18o})


def bandpass_component(values: np.ndarray, period_band_kyr: tuple[float, float]) -> np.ndarray:
    min_period, max_period = period_band_kyr
    if min_period <= 0.0 or max_period <= min_period:
        raise ValueError("period_band_kyr must be ordered as (min_period, max_period).")
    sample_rate_per_kyr = 1.0 / RESAMPLE_STEP_KA
    nyquist = 0.5 * sample_rate_per_kyr
    low_frequency = 1.0 / max_period
    high_frequency = 1.0 / min_period
    sos = butter(
        4,
        [low_frequency / nyquist, high_frequency / nyquist],
        btype="bandpass",
        output="sos",
    )
    return sosfiltfilt(sos, values)


def build_filtered_composites(regular: pd.DataFrame) -> pd.DataFrame:
    out = regular.copy()
    raw = out["d18o_raw"].to_numpy(dtype=float)
    for variant_id, settings in FILTER_VARIANTS.items():
        band = settings["bandstop_period_kyr"]
        if variant_id == "raw":
            out["d18o_raw_variant"] = raw
            continue
        component = bandpass_component(raw, band)
        out[f"d18o_{variant_id}_precession_component"] = component
        out[f"d18o_{variant_id}"] = raw - component
    return out


def build_detection_inputs(composite: pd.DataFrame, filtered_regular: pd.DataFrame) -> pd.DataFrame:
    """Evaluate raw and filtered records on the original Cheng age grid.

    TiPES ``KS_detection.m`` operates on the supplied, possibly irregular,
    sampling grid. For the raw reproduction check we therefore keep the Cheng
    composite ages rather than detecting on an artificial regular grid. The
    band-removed curves are constructed on a regular grid for filtering, then
    interpolated back to the original ages so that all variants share the same
    detection sampling.
    """

    out = composite.rename(columns={"d18o": "d18o_raw_variant"}).copy()
    for variant_id in FILTER_VARIANTS:
        if variant_id == "raw":
            continue
        out[f"d18o_{variant_id}"] = np.interp(
            out["age_ka"].to_numpy(dtype=float),
            filtered_regular["age_ka"].to_numpy(dtype=float),
            filtered_regular[value_column_for_variant(variant_id)].to_numpy(dtype=float),
        )
    return out


def value_column_for_variant(variant_id: str) -> str:
    return "d18o_raw_variant" if variant_id == "raw" else f"d18o_{variant_id}"


def two_sample_ks_statistic(sample_a: np.ndarray, sample_b: np.ndarray) -> float:
    a = np.sort(np.asarray(sample_a, dtype=float))
    b = np.sort(np.asarray(sample_b, dtype=float))
    values = np.sort(np.concatenate([a, b]))
    cdf_a = np.searchsorted(a, values, side="right") / float(len(a))
    cdf_b = np.searchsorted(b, values, side="right") / float(len(b))
    return float(np.max(np.abs(cdf_a - cdf_b)))


def tipes_window_grid(params: DetectorParams) -> np.ndarray:
    if WINDOW_MIN_KA == WINDOW_MAX_KA or params.n_windows == 1:
        return np.array([WINDOW_MIN_KA], dtype=float)
    idx = np.arange(params.n_windows, dtype=float)
    return WINDOW_MIN_KA * (WINDOW_MAX_KA / WINDOW_MIN_KA) ** (idx / (params.n_windows - 1.0))


def rowwise_nanmean(frame: np.ndarray) -> np.ndarray:
    with np.errstate(invalid="ignore"):
        return np.nanmean(frame, axis=1)


def sign_change_segments(values: np.ndarray) -> list[tuple[int, int]]:
    segments = []
    start = 0
    signs = np.sign(values)
    for idx in range(1, len(signs)):
        if signs[idx - 1] != signs[idx]:
            segments.append((start, idx))
            start = idx
    return segments


def tipes_ks_detection(series: pd.DataFrame, value_col: str, params: DetectorParams) -> pd.DataFrame:
    """Port the TiPES KS_detection.m workflow to Python.

    Important details copied from the MATLAB code:
    - windows are geometric between w_min and w_max;
    - the two samples are (t-w, t] and (t, t+w];
    - the KS statistic is signed by the mean change between samples;
    - larger windows are blended with smaller-window information and corrected
      for finite sample size;
    - peaks are selected within same-sign segments;
    - the mean across all windows is used first, then shorter-window detections
      are added only if not close to detections from larger windows.
    """

    frame = series[["age_ka", value_col]].dropna().copy()
    frame = frame.groupby("age_ka", as_index=False)[value_col].mean().sort_values("age_ka")
    tt = frame["age_ka"].to_numpy(dtype=float)
    xx = frame[value_col].to_numpy(dtype=float)
    if len(tt) < 10:
        return pd.DataFrame(columns=["transition_id", "age_ka", "jump_direction", "event_type"])

    # MATLAB shifts mixed-sign records positive before applying std/change
    # thresholds. Cheng d18O values are all negative, so this normally has no
    # effect, but keeping the step makes the port faithful.
    if np.sign(np.nanmax(xx)) > np.sign(np.nanmin(xx)):
        xx = xx - np.nanmin(xx) + 0.1

    kswindow = tipes_window_grid(params)
    n = len(tt)
    n_w = len(kswindow)
    ksstat = np.zeros((n, n_w), dtype=float)
    change_ks = np.full((n, n_w), np.nan, dtype=float)
    kslen1 = np.zeros((n, n_w), dtype=float)
    kslen2 = np.zeros((n, n_w), dtype=float)
    ksstd1 = np.full((n, n_w), np.nan, dtype=float)
    ksstd2 = np.full((n, n_w), np.nan, dtype=float)

    for col, window in enumerate(kswindow):
        left_start = np.searchsorted(tt, tt - window, side="right")
        left_end = np.arange(n) + 1
        right_start = np.arange(n) + 1
        right_end = np.searchsorted(tt, tt + window, side="right")
        for idx in range(n):
            r1 = np.arange(left_start[idx], left_end[idx])
            r2 = np.arange(right_start[idx], right_end[idx])
            if (
                len(r1) > 0
                and len(r2) > 0
                and np.nanmin(tt) <= tt[idx] - window
                and np.nanmax(tt) >= tt[idx] + window
            ):
                ks = two_sample_ks_statistic(xx[r1], xx[r2])
            else:
                ks = 0.0
            if len(r1) > 0 and len(r2) > 0:
                change = float(np.nanmean(xx[r1]) - np.nanmean(xx[r2]))
                std1 = float(np.nanstd(xx[r1], ddof=1)) if len(r1) > 1 else 0.0
                std2 = float(np.nanstd(xx[r2], ddof=1)) if len(r2) > 1 else 0.0
            else:
                change = np.nan
                std1 = np.nan
                std2 = np.nan
            ksstat[idx, col] = ks * np.sign(change) if np.isfinite(change) else 0.0
            change_ks[idx, col] = change
            kslen1[idx, col] = len(r1)
            kslen2[idx, col] = len(r2)
            ksstd1[idx, col] = std1
            ksstd2[idx, col] = std2

    ksstat2 = ksstat.copy()
    if n_w > 2:
        for col in range(n_w - 1, 1, -1):
            smaller = rowwise_nanmean(ksstat[:, :col])
            ksstat2[:, col] = rowwise_nanmean(np.column_stack([ksstat[:, col], smaller]))
        ksstat2[:, 1] = rowwise_nanmean(np.column_stack([ksstat[:, 1], ksstat[:, 0]]))

    x_cut = params.change_cut
    if x_cut is None:
        x_cut = float(np.nanstd(xx, ddof=1) * 0.1 + (np.nanmax(xx) - np.nanmin(xx)) * 0.05)

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
    if n >= 2:
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
            if len(segment) == 0 or not np.any(np.isfinite(segment) & (segment != 0.0)):
                continue
            candidate_local = np.flatnonzero(segment == segment[np.nanargmax(np.abs(segment))])
            candidate_idx = candidate_local + start
            row_score = np.abs(np.nansum(adjusted[candidate_idx, :], axis=1))
            chosen = candidate_idx[np.flatnonzero(row_score == np.nanmax(row_score))[-1]]
            peaks.append({"index": int(chosen), "value": float(ks_all[chosen, col])})
        peaks_by_col.append(pd.DataFrame(peaks))

    if not any(not peaks.empty for peaks in peaks_by_col):
        return pd.DataFrame(columns=["transition_id", "age_ka", "jump_direction", "event_type"])

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
            window = kswindow[col] if col < len(kswindow) else kswindow[-1]
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
                "detector_change_cut": float(x_cut),
            }
        )
    for age in jump_down:
        rows.append(
            {
                "age_ka": float(age),
                "jump_direction": -1,
                "event_type": "strong_monsoon_start",
                "detector_change_cut": float(x_cut),
            }
        )
    out = pd.DataFrame(rows).sort_values("age_ka").reset_index(drop=True)
    out.insert(0, "transition_id", np.arange(1, len(out) + 1))
    return out


def load_published_catalogue() -> pd.DataFrame:
    rows = []
    for event_type, path in PUBLISHED_CATALOGUES.items():
        df = pd.read_csv(path, encoding="utf-8-sig")
        ages = pd.to_numeric(df["start_time_ka_BP"], errors="coerce")
        sub = pd.DataFrame(
            {
                "event_type": event_type,
                "event_label": EVENT_SETTINGS[event_type]["label"],
                "published_age_ka": ages,
                "published_order": df.get("order", pd.Series(np.arange(1, len(df) + 1))),
                "published_source": str(path.relative_to(PROJECT_ROOT)),
            }
        )
        rows.append(sub.dropna(subset=["published_age_ka"]))
    return pd.concat(rows, ignore_index=True).sort_values(["event_type", "published_age_ka"])


def greedy_match_to_published(detected: pd.DataFrame, published: pd.DataFrame) -> pd.DataFrame:
    pairs = []
    for event_type in EVENT_SETTINGS:
        det = detected[detected["event_type"].eq(event_type)].copy()
        ref = published[published["event_type"].eq(event_type)].copy()
        for det_idx, drow in det.iterrows():
            for ref_idx, rrow in ref.iterrows():
                error = float(drow["age_ka"] - rrow["published_age_ka"])
                if abs(error) <= MATCH_TOLERANCE_KA:
                    pairs.append(
                        {
                            "event_type": event_type,
                            "detected_index": det_idx,
                            "published_index": ref_idx,
                            "detected_age_ka": float(drow["age_ka"]),
                            "published_age_ka": float(rrow["published_age_ka"]),
                            "age_error_ka": error,
                            "abs_age_error_ka": abs(error),
                        }
                    )
    if not pairs:
        return pd.DataFrame()

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


def build_event_frame(detected: pd.DataFrame, variant_id: str) -> pd.DataFrame:
    out = detected.copy()
    if out.empty:
        return out
    out["event_age_ka"] = out["age_ka"].astype(float)
    out["event_label"] = out["event_type"].map(
        {event_type: settings["label"] for event_type, settings in EVENT_SETTINGS.items()}
    )
    out["variant_id"] = variant_id
    out["variant_label"] = FILTER_VARIANTS[variant_id]["label"]
    out["source"] = f"{RUN_NAME}:{variant_id}"
    out = out.sort_values(["event_type", "event_age_ka"]).reset_index(drop=True)
    out["event_index"] = out.groupby("event_type").cumcount() + 1
    return out


def run_rayleigh_for_variant(events: pd.DataFrame, variant_id: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    phase_products = {
        driver: rayleigh.build_phase_series(driver, settings)
        for driver, settings in rayleigh.DRIVER_SETTINGS.items()
    }
    event_phases = rayleigh.sample_event_phases(events, phase_products)
    results = rayleigh.build_rayleigh_results(event_phases)
    for frame in (event_phases, results):
        frame.insert(0, "variant_id", variant_id)
        frame.insert(1, "variant_label", FILTER_VARIANTS[variant_id]["label"])
    return event_phases, results


def hazard_events_from_frame(events: pd.DataFrame, variant_id: str) -> list[base.EventDataset]:
    datasets = []
    for event_type, settings in EVENT_SETTINGS.items():
        ages = (
            events.loc[events["event_type"].eq(event_type), "event_age_ka"]
            .dropna()
            .to_numpy(dtype=float)
        )
        datasets.append(
            base.EventDataset(
                dataset_id=event_type,
                label=settings["label"],
                color=settings["color"],
                ages_ka=np.sort(ages),
                source=f"{RUN_NAME}:{variant_id}",
            )
        )
    return datasets


def run_predictive_for_variant(events: pd.DataFrame, variant_id: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    event_datasets = hazard_events_from_frame(events, variant_id)
    binned, _, _ = base.build_binned_inputs(event_datasets)
    binned, _, _ = predictive.add_resolution_control(binned)
    binned = predictive.add_same_type_history(binned, predictive.MAIN_HISTORY_WINDOW_KA)
    fit_frame = predictive.model_frame(binned)
    models = []
    for _, group in fit_frame.groupby("dataset_id", sort=False):
        for model_id, terms, label in predictive.MODEL_SPECS:
            models.append(base.fit_poisson_model(group, model_id, terms, label))
    summary = base.build_model_summary(models, fit_frame)
    lrt = predictive.build_adjusted_likelihood_tests(
        models,
        fit_frame,
        predictive.MAIN_HISTORY_WINDOW_KA,
    )
    for frame in (summary, lrt):
        frame.insert(0, "variant_id", variant_id)
        frame.insert(1, "variant_label", FILTER_VARIANTS[variant_id]["label"])
    return summary, lrt


def summarize_variant_results(
    catalogue_summary: pd.DataFrame,
    rayleigh_results: pd.DataFrame,
    predictive_lrt: pd.DataFrame,
    predictive_summary: pd.DataFrame,
) -> pd.DataFrame:
    rows = []
    for variant_id in FILTER_VARIANTS:
        for event_type in EVENT_SETTINGS:
            row = {
                "variant_id": variant_id,
                "variant_label": FILTER_VARIANTS[variant_id]["label"],
                "event_type": event_type,
                "event_label": EVENT_SETTINGS[event_type]["label"],
            }
            cat = catalogue_summary[
                catalogue_summary["variant_id"].eq(variant_id)
                & catalogue_summary["event_type"].eq(event_type)
            ]
            if not cat.empty:
                row["n_detected_events"] = int(cat["n_detected_events"].iloc[0])
                row["raw_reproduction_precision"] = float(cat["precision"].iloc[0])
                row["raw_reproduction_recall"] = float(cat["recall"].iloc[0])
            ray = rayleigh_results[
                rayleigh_results["variant_id"].eq(variant_id)
                & rayleigh_results["event_type"].eq(event_type)
                & rayleigh_results["driver"].eq("pre")
            ]
            if not ray.empty:
                row["rayleigh_pre_n"] = int(ray["n_phase_events_used"].iloc[0])
                row["rayleigh_pre_mean_phase_deg"] = float(ray["mean_phase_deg"].iloc[0])
                row["rayleigh_pre_Rbar"] = float(ray["mean_resultant_length"].iloc[0])
                row["rayleigh_pre_p"] = float(ray["rayleigh_p"].iloc[0])
            lrt = predictive_lrt[
                predictive_lrt["variant_id"].eq(variant_id)
                & predictive_lrt["dataset_id"].eq(event_type)
                & predictive_lrt["comparison_id"].eq("phase_after_adjusted_climate")
            ]
            if not lrt.empty:
                row["pi_phase_after_lr04_co2_LR"] = float(lrt["LR_statistic"].iloc[0])
                row["pi_phase_after_lr04_co2_p"] = float(lrt["LR_p_value"].iloc[0])
                row["pi_phase_after_lr04_co2_bits_per_event"] = float(
                    lrt["info_bits_per_event"].iloc[0]
                )
                row["pi_phase_after_lr04_co2_delta_AICc"] = float(
                    lrt["delta_AICc_full_minus_reduced"].iloc[0]
                )
            full = predictive_summary[
                predictive_summary["variant_id"].eq(variant_id)
                & predictive_summary["dataset_id"].eq(event_type)
                & predictive_summary["model_id"].eq("baseline_climate_lr04_co2_pre_phase")
            ]
            if not full.empty:
                row["pi_full_preferred_phase_deg"] = float(full["pre_phase_preferred_deg"].iloc[0])
                row["pi_full_rate_ratio"] = float(
                    full["pre_phase_rate_ratio_max_vs_min"].iloc[0]
                )
            rows.append(row)
    return pd.DataFrame(rows)


def build_catalogue_summary(
    variant_id: str,
    detected: pd.DataFrame,
    published: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    matches = greedy_match_to_published(detected, published)
    rows = []
    for event_type, settings in EVENT_SETTINGS.items():
        n_pub = int(published["event_type"].eq(event_type).sum())
        n_det = int(detected["event_type"].eq(event_type).sum()) if not detected.empty else 0
        n_tp = int(matches["event_type"].eq(event_type).sum()) if not matches.empty else 0
        n_fp = n_det - n_tp
        n_fn = n_pub - n_tp
        rows.append(
            {
                "variant_id": variant_id,
                "variant_label": FILTER_VARIANTS[variant_id]["label"],
                "event_type": event_type,
                "event_label": settings["label"],
                "n_detected_events": n_det,
                "n_published_events": n_pub,
                "matched_to_published": n_tp,
                "unmatched_detected": n_fp,
                "unmatched_published": n_fn,
                "precision": n_tp / n_det if n_det else 0.0,
                "recall": n_tp / n_pub if n_pub else 0.0,
                "match_tolerance_ka": MATCH_TOLERANCE_KA,
            }
        )
    return pd.DataFrame(rows), matches


def plot_reproduction_check(
    matches: pd.DataFrame,
    raw_catalogue_summary: pd.DataFrame,
    detector_params: DetectorParams,
    write_pdf: bool,
) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.4))
    ax = axes[0]
    colors = {event_type: settings["color"] for event_type, settings in EVENT_SETTINGS.items()}
    labels = {event_type: settings["label"] for event_type, settings in EVENT_SETTINGS.items()}
    if not matches.empty:
        for event_type, group in matches.groupby("event_type", sort=False):
            ax.scatter(
                group["published_age_ka"],
                group["detected_age_ka"],
                s=17,
                color=colors[event_type],
                alpha=0.75,
                label=labels[event_type],
            )
    ax.plot([ANALYSIS_START_KA, ANALYSIS_END_KA], [ANALYSIS_START_KA, ANALYSIS_END_KA], color="#222222", lw=0.8)
    ax.set_xlabel("Published age (kyr BP)")
    ax.set_ylabel("Detected age (kyr BP)")
    ax.set_title("Raw composite reproduction", loc="left")
    ax.legend(frameon=False, loc="lower right")

    ax = axes[1]
    by_type = raw_catalogue_summary.set_index("event_type")
    terms = ["strong_monsoon_start", "weak_monsoon_start"]
    x = np.arange(len(terms))
    width = 0.34
    precision = by_type.reindex(terms)["precision"].to_numpy(dtype=float)
    recall = by_type.reindex(terms)["recall"].to_numpy(dtype=float)
    ax.bar(x - width / 2, precision, width=width, color="#4daf4a", alpha=0.86, label="precision")
    ax.bar(x + width / 2, recall, width=width, color="#377eb8", alpha=0.86, label="recall")
    ax.set_xticks(x)
    ax.set_xticklabels(["Strong", "Weak"])
    ax.set_ylim(0.0, 1.0)
    ax.set_ylabel("score")
    ax.set_title("Post-hoc match to published catalogue", loc="left")
    ax.legend(frameon=False, loc="upper right")
    text = (
        f"n_w={detector_params.n_windows}\n"
        f"D>={detector_params.d_cut:.2f}\n"
        f"n>={detector_params.n_cut}\n"
        f"s_c={detector_params.sigma_cut:.2f}\n"
        "not tuned to published ages"
    )
    ax.text(
        0.98,
        0.05,
        text,
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        bbox={"boxstyle": "round,pad=0.25", "facecolor": "white", "edgecolor": "#bbbbbb", "alpha": 0.92},
    )
    for idx, ax in enumerate(axes):
        ax.text(
            -0.12,
            1.04,
            chr(ord("a") + idx),
            transform=ax.transAxes,
            fontweight="bold",
            fontsize=12,
            va="bottom",
        )
    fig.subplots_adjust(left=0.08, right=0.98, bottom=0.15, top=0.90, wspace=0.28)
    save_figure(fig, "fig01_raw_ks_reproduction_check", write_pdf)


def plot_phase_sensitivity(summary: pd.DataFrame, write_pdf: bool) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(11.5, 6.8), sharex="col")
    variant_order = list(FILTER_VARIANTS)
    x = np.arange(len(variant_order))
    width = 0.34
    event_order = ["strong_monsoon_start", "weak_monsoon_start"]
    offsets = [-width / 2, width / 2]
    for event_type, offset in zip(event_order, offsets):
        sub = summary[summary["event_type"].eq(event_type)].set_index("variant_id").reindex(variant_order)
        color = EVENT_SETTINGS[event_type]["color"]
        label = EVENT_SETTINGS[event_type]["label"]
        axes[0, 0].bar(x + offset, sub["n_detected_events"], width=width, color=color, alpha=0.82, label=label)
        axes[0, 1].bar(x + offset, -np.log10(np.maximum(sub["rayleigh_pre_p"], 1e-300)), width=width, color=color, alpha=0.82)
        axes[1, 0].bar(x + offset, -np.log10(np.maximum(sub["pi_phase_after_lr04_co2_p"], 1e-300)), width=width, color=color, alpha=0.82)
        axes[1, 1].bar(x + offset, sub["pi_full_preferred_phase_deg"], width=width, color=color, alpha=0.82)

    axes[0, 0].set_ylabel("Detected events")
    axes[0, 1].set_ylabel("-log10 Rayleigh p")
    axes[1, 0].set_ylabel("-log10 PI p")
    axes[1, 1].set_ylabel("Preferred phase (deg)")
    axes[0, 0].set_title("Catalogue size", loc="left")
    axes[0, 1].set_title("Rayleigh precession phase", loc="left")
    axes[1, 0].set_title("Precession PI after baseline+LR04+CO2", loc="left")
    axes[1, 1].set_title("Full-model preferred phase", loc="left")
    for ax in (axes[0, 1], axes[1, 0]):
        ax.axhline(-np.log10(0.05), color="#222222", ls="--", lw=0.8)
    axes[1, 1].set_ylim(0.0, 360.0)
    axes[1, 1].set_yticks([0, 90, 180, 270, 360])
    for ax in axes.ravel():
        ax.set_xticks(x)
        ax.set_xticklabels([FILTER_VARIANTS[v]["label"] for v in variant_order], rotation=18, ha="right")
        ax.grid(True, axis="y", color="#e6e6e6", lw=0.6)
    axes[0, 0].legend(frameon=False, loc="upper right")
    for idx, ax in enumerate(axes.ravel()):
        ax.text(
            -0.12,
            1.04,
            chr(ord("a") + idx),
            transform=ax.transAxes,
            fontweight="bold",
            fontsize=12,
            va="bottom",
        )
    fig.subplots_adjust(left=0.08, right=0.98, bottom=0.16, top=0.92, hspace=0.35, wspace=0.28)
    save_figure(fig, "fig02_precession_band_removal_phase_sensitivity", write_pdf)


def plot_filtered_composites(
    filtered: pd.DataFrame,
    detected_by_variant: pd.DataFrame,
    write_pdf: bool,
) -> None:
    fig, axes = plt.subplots(len(FILTER_VARIANTS), 1, figsize=(12.0, 7.2), sharex=True)
    for ax_idx, (ax, (variant_id, settings)) in enumerate(zip(axes, FILTER_VARIANTS.items())):
        value_col = value_column_for_variant(variant_id)
        ax.plot(filtered["age_ka"], filtered[value_col], color=settings["color"], lw=0.75)
        ymin, ymax = ax.get_ylim()
        for event_type, event_settings in EVENT_SETTINGS.items():
            sub = detected_by_variant[
                detected_by_variant["variant_id"].eq(variant_id)
                & detected_by_variant["event_type"].eq(event_type)
            ]
            ax.vlines(
                sub["age_ka"],
                ymin,
                ymax,
                color=event_settings["color"],
                lw=0.45,
                alpha=0.35,
            )
        ax.set_ylim(ymin, ymax)
        ax.set_ylabel(r"$\delta^{18}$O")
        ax.set_title(settings["label"], loc="left")
        ax.text(
            -0.035,
            1.03,
            chr(ord("a") + ax_idx),
            transform=ax.transAxes,
            fontweight="bold",
            fontsize=12,
            va="bottom",
        )
    axes[-1].set_xlabel("Age (kyr BP)")
    axes[-1].set_xlim(ANALYSIS_START_KA, ANALYSIS_END_KA)
    legend_handles = [
        Line2D([0], [0], color=EVENT_SETTINGS["strong_monsoon_start"]["color"], lw=1.5, label="Strong starts"),
        Line2D([0], [0], color=EVENT_SETTINGS["weak_monsoon_start"]["color"], lw=1.5, label="Weak starts"),
    ]
    axes[0].legend(handles=legend_handles, frameon=False, loc="upper right", ncol=2)
    fig.subplots_adjust(left=0.07, right=0.98, top=0.95, bottom=0.08, hspace=0.26)
    save_figure(fig, "fig03_filtered_catalogues_on_composite", write_pdf)


def write_outputs(
    filtered: pd.DataFrame,
    detected_by_variant: pd.DataFrame,
    catalogue_summary: pd.DataFrame,
    matches_by_variant: pd.DataFrame,
    event_phases: pd.DataFrame,
    rayleigh_results: pd.DataFrame,
    predictive_summary: pd.DataFrame,
    predictive_lrt: pd.DataFrame,
    paper_summary: pd.DataFrame,
    detector_params: DetectorParams,
) -> None:
    ensure_dir(OUT_DATA_DIR)
    filtered.to_csv(OUT_DATA_DIR / "cheng_composite_regularized_filtered.csv", index=False)
    detected_by_variant.to_csv(OUT_DATA_DIR / "detected_transitions_by_filter_variant.csv", index=False)
    catalogue_summary.to_csv(OUT_DATA_DIR / "catalogue_match_summary_by_filter_variant.csv", index=False)
    matches_by_variant.to_csv(OUT_DATA_DIR / "matches_to_published_by_filter_variant.csv", index=False)
    event_phases.to_csv(OUT_DATA_DIR / "event_orbital_phases_by_filter_variant.csv", index=False)
    rayleigh_results.to_csv(OUT_DATA_DIR / "rayleigh_results_by_filter_variant.csv", index=False)
    predictive_summary.to_csv(OUT_DATA_DIR / "predictive_model_summary_by_filter_variant.csv", index=False)
    predictive_lrt.to_csv(OUT_DATA_DIR / "predictive_likelihood_tests_by_filter_variant.csv", index=False)
    paper_summary.to_csv(OUT_DATA_DIR / "precession_band_removal_summary.csv", index=False)
    pd.DataFrame(
        [
            {
                "run_name": RUN_NAME,
                "method": "Rousseau-style two-sample KS with fixed pre-declared thresholds",
                "analysis_start_ka": ANALYSIS_START_KA,
                "analysis_end_ka": ANALYSIS_END_KA,
                "filter_resample_step_ka": RESAMPLE_STEP_KA,
                "window_min_ka": WINDOW_MIN_KA,
                "window_max_ka": WINDOW_MAX_KA,
                "window_spacing": "geometric",
                "match_tolerance_ka": MATCH_TOLERANCE_KA,
                "n_windows": detector_params.n_windows,
                "d_cut": detector_params.d_cut,
                "n_cut": detector_params.n_cut,
                "sigma_cut": detector_params.sigma_cut,
                "change_cut": detector_params.change_cut,
                "leakage_control": (
                    "Published Rousseau event ages are used only for post-hoc reproduction diagnostics, "
                    "not to choose detector parameters."
                ),
                "filter_variants": "; ".join(FILTER_VARIANTS.keys()),
            }
        ]
    ).to_csv(OUT_DATA_DIR / "parameters.csv", index=False)


def run_analysis(write_pdf: bool = True) -> pd.DataFrame:
    composite = load_cheng_composite()
    regular = regularize_cheng(composite)
    filtered = build_filtered_composites(regular)
    detection_inputs = build_detection_inputs(composite, filtered)
    published = load_published_catalogue()
    detector_params = DetectorParams(
        DETECTOR_N_WINDOWS,
        DETECTOR_D_CUT,
        DETECTOR_N_CUT,
        DETECTOR_SIGMA_CUT,
        DETECTOR_CHANGE_CUT,
    )

    detected_frames = []
    catalogue_summary_frames = []
    match_frames = []
    event_phase_frames = []
    rayleigh_frames = []
    predictive_summary_frames = []
    predictive_lrt_frames = []

    for variant_id in FILTER_VARIANTS:
        detected = tipes_ks_detection(
            detection_inputs,
            value_column_for_variant(variant_id),
            detector_params,
        )
        detected.insert(0, "variant_id", variant_id)
        detected.insert(1, "variant_label", FILTER_VARIANTS[variant_id]["label"])
        detected_frames.append(detected)

        catalogue_summary, matches = build_catalogue_summary(variant_id, detected, published)
        catalogue_summary_frames.append(catalogue_summary)
        if not matches.empty:
            matches.insert(0, "variant_id", variant_id)
            matches.insert(1, "variant_label", FILTER_VARIANTS[variant_id]["label"])
            match_frames.append(matches)

        events = build_event_frame(detected, variant_id)
        event_phases, rayleigh_result = run_rayleigh_for_variant(events, variant_id)
        model_summary, lrt = run_predictive_for_variant(events, variant_id)
        event_phase_frames.append(event_phases)
        rayleigh_frames.append(rayleigh_result)
        predictive_summary_frames.append(model_summary)
        predictive_lrt_frames.append(lrt)

    detected_by_variant = pd.concat(detected_frames, ignore_index=True)
    catalogue_summary = pd.concat(catalogue_summary_frames, ignore_index=True)
    matches_by_variant = (
        pd.concat(match_frames, ignore_index=True)
        if match_frames
        else pd.DataFrame(columns=["variant_id", "variant_label"])
    )
    event_phases = pd.concat(event_phase_frames, ignore_index=True)
    rayleigh_results = pd.concat(rayleigh_frames, ignore_index=True)
    predictive_summary = pd.concat(predictive_summary_frames, ignore_index=True)
    predictive_lrt = pd.concat(predictive_lrt_frames, ignore_index=True)
    paper_summary = summarize_variant_results(
        catalogue_summary,
        rayleigh_results,
        predictive_lrt,
        predictive_summary,
    )

    write_outputs(
        filtered,
        detected_by_variant,
        catalogue_summary,
        matches_by_variant,
        event_phases,
        rayleigh_results,
        predictive_summary,
        predictive_lrt,
        paper_summary,
        detector_params,
    )
    raw_matches = matches_by_variant[matches_by_variant["variant_id"].eq("raw")]
    raw_catalogue_summary = catalogue_summary[catalogue_summary["variant_id"].eq("raw")]
    plot_reproduction_check(raw_matches, raw_catalogue_summary, detector_params, write_pdf)
    plot_phase_sensitivity(paper_summary, write_pdf)
    plot_filtered_composites(filtered, detected_by_variant, write_pdf)
    return paper_summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Test whether precession-phase results survive removing precession-band variance before KS detection."
    )
    parser.add_argument("--no-pdf", action="store_true", help="Only write PNG figures.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary = run_analysis(write_pdf=not args.no_pdf)
    print("Precession-band removal sensitivity summary:")
    columns = [
        "variant_id",
        "event_type",
        "n_detected_events",
        "rayleigh_pre_mean_phase_deg",
        "rayleigh_pre_p",
        "pi_phase_after_lr04_co2_p",
        "pi_phase_after_lr04_co2_bits_per_event",
        "pi_full_preferred_phase_deg",
    ]
    print(summary[columns].to_string(index=False))
    print(f"\nWrote tables to: {OUT_DATA_DIR.relative_to(PROJECT_ROOT)}")
    print(f"Wrote figures to: {OUT_FIG_DIR.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    main()
