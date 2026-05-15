"""
Event-process-baseline predictive Poisson hazard models.

This script is a stricter extension of ``Bin_hazard_phase_poisson.py``. It keeps
the same 0.2 kyr event-bin likelihood and the same LR04/CO2/precession-phase
predictors, but adds two nuisance controls before testing external forcing:

1. same-type event history, i.e. the number of events of the same type in the
   older W kyr interval (t, t + W];
2. local sampling resolution estimated from the Cheng et al. (2016) composite
   speleothem age grid.

The main model uses W = 5 kyr. Other history windows are treated as a
sensitivity experiment. Nested-model gains are reported both as likelihood-
ratio tests and as bits/event, matching the predictive-information language of
the lagged scan.

Method sketch
-------------
The event catalogue is represented on short time bins. For bin i, the response
is the number of starts observed in that bin:

    Y_i ~ Poisson(mu_i)
    mu_i = dt_i * lambda_i
    log(lambda_i) = beta0 + x_i beta.

The log(dt_i) term is therefore an offset in the Poisson likelihood. It converts
the fitted per-kyr hazard, lambda_i, into the expected event count, mu_i, in a
finite bin. The summed log-likelihood is:

    logL = sum_i [Y_i log(mu_i) - mu_i - log(Y_i!)].

Here "event-process baseline" means the event-process and detection baseline rather than a
climate-forcing baseline: same-type event history controls local clustering or
inhibition in the event sequence, while the Cheng composite age-spacing term
controls the possibility that better sampled parts of the speleothem stack have
more detected transitions. The scientific comparison of interest is then:

    climate-state model  = event-process baseline + LR04 + CO2
    full predictive model = climate-state model + sin(precession phase) + cos(precession phase).

The log-likelihood gain, logL_full - logL_reduced, is reported as nats and as
bits per bin/event. The likelihood-ratio statistic is 2 times the same gain;
under Wilks' theorem it is compared with a chi-square distribution with degrees
of freedom equal to the number of added coefficients. For the precession-phase
test, df = 2 because the circular phase is represented by sine and cosine.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd

import Bin_hazard_phase_poisson as base
from paper_figure_export import save_paper_pdf
from toolbox.model_stats import nested_likelihood_metrics


RUN_NAME = "Predictive_information_model"
OUT_DATA_DIR = base.PROJECT_ROOT / "data" / "processed" / RUN_NAME
OUT_FIG_DIR = base.PROJECT_ROOT / "figures" / RUN_NAME

CHENG_XLSX = base.PROJECT_ROOT / "data/raw/Cheng_2016.xlsx"

MAIN_HISTORY_WINDOW_KA = 5.0
HISTORY_WINDOW_GRID_KA = (1.0, 2.0, 5.0, 10.0)

HISTORY_TERM = "same_type_history_count"
RESOLUTION_TERM = "cheng_log_resolution_scaled"
# In this script, the baseline terms are nuisance controls for event-process
# memory and local detection/sampling resolution. Climate/orbital predictors are
# tested after this baseline has already been included.
BASELINE_TERMS = (HISTORY_TERM, RESOLUTION_TERM)
CLIMATE_TERMS = ("lr04_scaled", "co2_scaled")
PHASE_TERMS = ("pre_phase_sin", "pre_phase_cos")
FULL_TERMS = BASELINE_TERMS + CLIMATE_TERMS + PHASE_TERMS

MODEL_SPECS: list[tuple[str, tuple[str, ...], str]] = [
    ("stationary", (), "Stationary"),
    ("history_resolution_baseline", BASELINE_TERMS, "Event-process baseline"),
    ("baseline_lr04", BASELINE_TERMS + ("lr04_scaled",), "Event-process baseline + LR04"),
    ("baseline_co2", BASELINE_TERMS + ("co2_scaled",), "Event-process baseline + CO$_2$"),
    (
        "baseline_climate_lr04_co2",
        BASELINE_TERMS + CLIMATE_TERMS,
        "Climate-state model",
    ),
    (
        "baseline_pre_phase",
        BASELINE_TERMS + PHASE_TERMS,
        "Event-process baseline + precession phase",
    ),
    (
        "baseline_lr04_pre_phase",
        BASELINE_TERMS + ("lr04_scaled",) + PHASE_TERMS,
        "Event-process baseline + LR04 + precession phase",
    ),
    (
        "baseline_co2_pre_phase",
        BASELINE_TERMS + ("co2_scaled",) + PHASE_TERMS,
        "Event-process baseline + CO$_2$ + precession phase",
    ),
    (
        "baseline_climate_lr04_co2_pre_phase",
        FULL_TERMS,
        "Full predictive model",
    ),
]

LR_TEST_SPECS: list[tuple[str, str, str, str]] = [
    (
        "baseline_vs_stationary",
        "stationary",
        "history_resolution_baseline",
        "Do same-type history and sampling resolution improve over constant rate?",
    ),
    (
        "lr04_after_baseline",
        "history_resolution_baseline",
        "baseline_lr04",
        "Does LR04 improve over the event-process baseline?",
    ),
    (
        "co2_after_baseline",
        "history_resolution_baseline",
        "baseline_co2",
        "Does CO2 improve over the event-process baseline?",
    ),
    (
        "climate_lr04_co2_after_baseline",
        "history_resolution_baseline",
        "baseline_climate_lr04_co2",
        "Does LR04+CO2 improve over the event-process baseline?",
    ),
    (
        "pre_phase_after_baseline",
        "history_resolution_baseline",
        "baseline_pre_phase",
        "Does precession phase improve over the event-process baseline?",
    ),
    (
        "phase_after_adjusted_climate",
        "baseline_climate_lr04_co2",
        "baseline_climate_lr04_co2_pre_phase",
        "Does precession phase add information after the climate-state model?",
    ),
    (
        "lr04_after_adjusted_co2_phase",
        "baseline_co2_pre_phase",
        "baseline_climate_lr04_co2_pre_phase",
        "Does LR04 add information within the full predictive model?",
    ),
    (
        "co2_after_adjusted_lr04_phase",
        "baseline_lr04_pre_phase",
        "baseline_climate_lr04_co2_pre_phase",
        "Does CO2 add information within the full predictive model?",
    ),
    (
        "full_vs_baseline",
        "history_resolution_baseline",
        "baseline_climate_lr04_co2_pre_phase",
        "Does LR04+CO2+precession phase improve over the event-process baseline?",
    ),
]

MODEL_COLORS = {
    "stationary": "#777777",
    "history_resolution_baseline": "#4d4d4d",
    "baseline_lr04": "#66c2a5",
    "baseline_co2": "#fc8d62",
    "baseline_climate_lr04_co2": "#1865C3",
    "baseline_pre_phase": "#8da0cb",
    "baseline_lr04_pre_phase": "#a6cee3",
    "baseline_co2_pre_phase": "#fdbf6f",
    "baseline_climate_lr04_co2_pre_phase": "#C51B7D",
}

LR_TEST_SHORT_LABELS = {
    "baseline_vs_stationary": "EP baseline vs stationary",
    "lr04_after_baseline": "+LR04 vs EP baseline",
    "co2_after_baseline": "+CO$_2$ vs EP baseline",
    "climate_lr04_co2_after_baseline": "Climate-state vs EP baseline",
    "pre_phase_after_baseline": "+phase vs EP baseline",
    "phase_after_adjusted_climate": "+phase vs climate-state",
    "lr04_after_adjusted_co2_phase": "Full predictive model vs -LR04",
    "co2_after_adjusted_lr04_phase": "Full predictive model vs -CO$_2$",
    "full_vs_baseline": "Full predictive model vs EP baseline",
}

MODEL_PLOT_LABELS = {
    "stationary": "Stationary",
    "history_resolution_baseline": "EP baseline",
    "baseline_lr04": "+ LR04",
    "baseline_co2": "+ CO$_2$",
    "baseline_climate_lr04_co2": "Climate-state",
    "baseline_pre_phase": "+ phase",
    "baseline_lr04_pre_phase": "+ LR04 + phase",
    "baseline_co2_pre_phase": "+ CO$_2$ + phase",
    "baseline_climate_lr04_co2_pre_phase": "Full predictive model",
}

LR_TEST_PLOT_LABELS = {
    "baseline_vs_stationary": "EP baseline vs stationary",
    "lr04_after_baseline": "+LR04 vs EP baseline",
    "co2_after_baseline": "+CO$_2$ vs EP baseline",
    "climate_lr04_co2_after_baseline": "Climate-state vs EP baseline",
    "pre_phase_after_baseline": "+phase vs EP baseline",
    "phase_after_adjusted_climate": "+phase vs climate-state",
    "lr04_after_adjusted_co2_phase": "Full predictive model vs -LR04",
    "co2_after_adjusted_lr04_phase": "Full predictive model vs -CO$_2$",
    "full_vs_baseline": "Full predictive model vs EP baseline",
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


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def save_figure(fig: plt.Figure, stem: str, write_pdf: bool) -> None:
    ensure_dir(OUT_FIG_DIR)
    fig.savefig(OUT_FIG_DIR / f"{stem}.png", dpi=300, bbox_inches="tight")
    if write_pdf:
        fig.savefig(OUT_FIG_DIR / f"{stem}.pdf", bbox_inches="tight")
        save_paper_pdf(fig, base.PROJECT_ROOT, stem)
    plt.close(fig)


def format_p_value(value: float) -> str:
    if not np.isfinite(value):
        return "NA"
    if value < 1e-4:
        return f"{value:.1e}"
    if value < 0.01:
        return f"{value:.4f}"
    return f"{value:.3f}"


def load_cheng_composite_resolution(centers_ka: np.ndarray) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Estimate local sampling resolution from the Cheng composite age grid.

    The transition catalogue is extracted from an unevenly sampled composite
    speleothem record. Dense parts of that record can support more reliable
    detection of abrupt changes than sparse parts. To keep this detection
    structure from being absorbed by LR04, CO2, or orbital terms, we estimate a
    local age spacing around each Cheng composite point, interpolate it to the
    0.2 kyr event-bin centers, log-transform it, and range-scale it like the
    other predictors.
    """

    raw = pd.read_excel(CHENG_XLSX, sheet_name="Composite record")
    age_col = next(col for col in raw.columns if "age" in str(col).lower())
    value_col = next(col for col in raw.columns if "18" in str(col).lower() or "δ" in str(col))
    ages = pd.to_numeric(raw[age_col], errors="coerce").to_numpy(dtype=float)
    values = pd.to_numeric(raw[value_col], errors="coerce").to_numpy(dtype=float)
    ok = np.isfinite(ages) & np.isfinite(values)
    frame = pd.DataFrame({"age_ka": ages[ok], "d18o": values[ok]})
    frame = frame[
        frame["age_ka"].between(base.ANALYSIS_START_KA, base.ANALYSIS_END_KA, inclusive="both")
    ].copy()
    frame = frame.groupby("age_ka", as_index=False)["d18o"].mean().sort_values("age_ka")
    age = frame["age_ka"].to_numpy(dtype=float)
    if len(age) < 3:
        raise ValueError("Too few Cheng composite ages to estimate sampling resolution.")

    previous_gap = np.r_[np.nan, np.diff(age)]
    next_gap = np.r_[np.diff(age), np.nan]
    local_spacing = np.nanmedian(np.vstack([previous_gap, next_gap]), axis=0)
    fallback = np.nanmedian(np.diff(age))
    local_spacing = np.where(np.isfinite(local_spacing) & (local_spacing > 0.0), local_spacing, fallback)
    frame["local_spacing_ka"] = local_spacing
    frame["local_sampling_density_per_kyr"] = 1.0 / frame["local_spacing_ka"]

    spacing_at_center = np.interp(centers_ka, age, frame["local_spacing_ka"].to_numpy(dtype=float))
    log_spacing = np.log(np.clip(spacing_at_center, 1e-6, None))
    scaled, mean, vmin, vmax, value_range = base.scale_to_zero_mean_range_one(log_spacing)
    by_bin = pd.DataFrame(
        {
            "bin_center_ka": centers_ka,
            "cheng_local_resolution_ka": spacing_at_center,
            "cheng_log_resolution": log_spacing,
            RESOLUTION_TERM: scaled,
        }
    )
    meta = pd.DataFrame(
        [
            {
                "forcing_id": RESOLUTION_TERM,
                "forcing_label": "log Cheng composite local age spacing",
                "source": str(CHENG_XLSX.relative_to(base.PROJECT_ROOT)),
                "mean": mean,
                "min": vmin,
                "max": vmax,
                "range": value_range,
            }
        ]
    )
    return by_bin, meta


def add_resolution_control(binned: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    centers = (
        binned[["bin_center_ka"]]
        .drop_duplicates()
        .sort_values("bin_center_ka")["bin_center_ka"]
        .to_numpy(dtype=float)
    )
    resolution_by_bin, resolution_meta = load_cheng_composite_resolution(centers)
    out = binned.merge(resolution_by_bin, on="bin_center_ka", how="left")
    if out[RESOLUTION_TERM].isna().any():
        raise ValueError("Resolution control contains missing values after interpolation.")
    return out, resolution_by_bin, resolution_meta


def add_same_type_history(binned: pd.DataFrame, history_window_ka: float) -> pd.DataFrame:
    """Count same-type events in the older interval (t, t + W].

    Age increases into the past. For a bin centered at age t, the interval
    (t, t + W] is therefore the local past of the event process. The current
    bin is excluded so the model cannot predict an event with itself. The
    fitted coefficient on this term can absorb clustering, inhibition, or
    catalogue-resolution artifacts before the model asks whether external
    forcings add information.
    """

    frames = []
    for _, group in binned.groupby("dataset_id", sort=False):
        group = group.sort_values("bin_center_ka").copy()
        centers = group["bin_center_ka"].to_numpy(dtype=float)
        counts = group["event_count"].to_numpy(dtype=float)
        # Cumulative sums make rolling event counts exact on the binned grid.
        # ``left_idx`` uses side="right" with a tiny offset to exclude the
        # current bin; ``right_idx`` includes bins up to t + W.
        cumulative = np.concatenate([[0.0], np.cumsum(counts)])
        left_idx = np.searchsorted(centers, centers + 1e-9, side="right")
        right_idx = np.searchsorted(centers, centers + history_window_ka, side="right")
        history = cumulative[right_idx] - cumulative[left_idx]
        coverage = np.minimum(centers[-1], centers + history_window_ka) - centers
        group[HISTORY_TERM] = history
        group["same_type_history_rate_per_kyr"] = history / history_window_ka
        group["same_type_history_window_ka"] = float(history_window_ka)
        group["same_type_history_coverage_ka"] = coverage
        group["same_type_history_complete"] = coverage >= (
            history_window_ka - 0.5 * base.BIN_WIDTH_KA
        )
        frames.append(group)
    return pd.concat(frames, ignore_index=True)


def load_cheng_composite_record() -> pd.DataFrame:
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
    out = out[
        out["age_ka"].between(base.ANALYSIS_START_KA, base.ANALYSIS_END_KA, inclusive="both")
    ]
    return out.sort_values("age_ka").reset_index(drop=True)


def build_adjusted_inputs(history_window_ka: float) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Build the adjusted design table used by the main and sensitivity fits.

    This starts from the original LR04/CO2/precession event-bin table, then
    appends the two nuisance controls introduced in this script: Cheng
    composite sampling resolution and same-type event history.
    """

    events = base.load_all_events()
    binned, scale_summary, phase_extrema = base.build_binned_inputs(events)
    binned, resolution_by_bin, resolution_meta = add_resolution_control(binned)
    binned = add_same_type_history(binned, history_window_ka)
    scale_summary = pd.concat([scale_summary, resolution_meta], ignore_index=True)
    return binned, scale_summary, phase_extrema, resolution_by_bin


def model_frame(binned: pd.DataFrame) -> pd.DataFrame:
    """Keep only bins whose history window is fully covered.

    Bins near the oldest edge do not have a complete older W-kyr history
    interval. Dropping them keeps the fitted history coefficient comparable
    across the analysis interval.
    """

    return binned[binned["same_type_history_complete"].astype(bool)].copy()


def fit_adjusted_models(binned: pd.DataFrame) -> list[base.FittedPoissonModel]:
    """Fit all adjusted candidate models for each event type."""

    fit_frame = model_frame(binned)
    models: list[base.FittedPoissonModel] = []
    for _, group in fit_frame.groupby("dataset_id", sort=False):
        for model_id, terms, label in MODEL_SPECS:
            models.append(base.fit_poisson_model(group, model_id, terms, label))
    return models


def build_adjusted_likelihood_tests(
    models: list[base.FittedPoissonModel],
    fit_frame: pd.DataFrame,
    history_window_ka: float,
) -> pd.DataFrame:
    """Compare nested adjusted models with LR and predictive-information scores.

    Each comparison uses two models fitted on the same rows. The log-likelihood
    gain is first kept in nats, then divided by log(2) to express the gain in
    bits. Dividing by the number of events gives an event-normalized score;
    dividing by the number of bins gives a time-grid-normalized score.
    """

    lookup = base.model_lookup(models)
    rows = []
    for dataset_id, group in fit_frame.groupby("dataset_id", sort=False):
        n_events = int(group["event_count"].sum())
        n_bins = int(len(group))
        for comparison_id, reduced_id, full_id, question in LR_TEST_SPECS:
            reduced = lookup[(dataset_id, reduced_id)]
            full = lookup[(dataset_id, full_id)]
            df = len(full.beta) - len(reduced.beta)
            metrics = nested_likelihood_metrics(
                loglik_full=full.log_likelihood,
                loglik_reduced=reduced.log_likelihood,
                df=df,
                n_bins=n_bins,
                n_events=n_events,
                aicc_full=full.aicc,
                aicc_reduced=reduced.aicc,
            )
            rows.append(
                {
                    "dataset_id": dataset_id,
                    "dataset_label": full.dataset_label,
                    "history_window_ka": float(history_window_ka),
                    "comparison_id": comparison_id,
                    "question": question,
                    "reduced_model_id": reduced_id,
                    "full_model_id": full_id,
                    **metrics,
                    "reject_LR_at_0p05": metrics["LR_p_value"] < 0.05,
                }
            )
    return pd.DataFrame(rows)


def build_sensitivity_tables() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Repeat the adjusted analysis for each candidate history-window length."""

    binned_rows = []
    summary_rows = []
    lrt_rows = []
    for history_window in HISTORY_WINDOW_GRID_KA:
        binned, _, _, _ = build_adjusted_inputs(history_window)
        fit_frame = model_frame(binned)
        models = fit_adjusted_models(binned)
        summary = base.build_model_summary(models, fit_frame)
        lrt = build_adjusted_likelihood_tests(models, fit_frame, history_window)
        summary.insert(0, "history_window_ka", float(history_window))
        fit_frame = fit_frame.copy()
        fit_frame.insert(0, "history_window_ka", float(history_window))
        binned_rows.append(fit_frame)
        summary_rows.append(summary)
        lrt_rows.append(lrt)
    return (
        pd.concat(binned_rows, ignore_index=True),
        pd.concat(summary_rows, ignore_index=True),
        pd.concat(lrt_rows, ignore_index=True),
    )


def build_history_sensitivity_summary(model_summary: pd.DataFrame, likelihood_tests: pd.DataFrame) -> pd.DataFrame:
    """Extract the paper-facing history-window sensitivity metrics."""

    rows = []
    for history_window in HISTORY_WINDOW_GRID_KA:
        for dataset_id in base.DATASET_SETTINGS:
            phase_test = likelihood_tests[
                likelihood_tests["history_window_ka"].eq(history_window)
                & likelihood_tests["dataset_id"].eq(dataset_id)
                & likelihood_tests["comparison_id"].eq("phase_after_adjusted_climate")
            ].iloc[0]
            full = model_summary[
                model_summary["history_window_ka"].eq(history_window)
                & model_summary["dataset_id"].eq(dataset_id)
                & model_summary["model_id"].eq("baseline_climate_lr04_co2_pre_phase")
            ].iloc[0]
            baseline = model_summary[
                model_summary["history_window_ka"].eq(history_window)
                & model_summary["dataset_id"].eq(dataset_id)
                & model_summary["model_id"].eq("history_resolution_baseline")
            ].iloc[0]
            rows.append(
                {
                    "history_window_ka": float(history_window),
                    "dataset_id": dataset_id,
                    "dataset_label": full["dataset_label"],
                    "n_bins": int(full["n_bins"]),
                    "n_events": int(full["n_events"]),
                    "baseline_AICc": float(baseline["AICc"]),
                    "full_AICc": float(full["AICc"]),
                    "full_delta_AICc_ranked_within_window": float(full["delta_AICc"]),
                    "full_rank_AICc": float(full["rank_AICc"]),
                    "phase_after_adjusted_climate_LR": float(phase_test["LR_statistic"]),
                    "phase_after_adjusted_climate_p": float(phase_test["LR_p_value"]),
                    "phase_after_adjusted_climate_delta_AICc": float(
                        phase_test["delta_AICc_full_minus_reduced"]
                    ),
                    "phase_after_adjusted_climate_bits_per_event": float(
                        phase_test["info_bits_per_event"]
                    ),
                    "pre_phase_preferred_deg": float(full["pre_phase_preferred_deg"]),
                    "pre_phase_rate_ratio_max_vs_min": float(
                        full["pre_phase_rate_ratio_max_vs_min"]
                    ),
                }
            )
    return pd.DataFrame(rows)


def write_outputs(
    *,
    binned: pd.DataFrame,
    fit_frame: pd.DataFrame,
    scale_summary: pd.DataFrame,
    phase_extrema: pd.DataFrame,
    resolution_by_bin: pd.DataFrame,
    model_summary: pd.DataFrame,
    coefficients: pd.DataFrame,
    likelihood_tests: pd.DataFrame,
    fitted_rates: pd.DataFrame,
    sensitivity_binned: pd.DataFrame,
    sensitivity_summary: pd.DataFrame,
    sensitivity_lrt: pd.DataFrame,
    history_sensitivity: pd.DataFrame,
) -> None:
    ensure_dir(OUT_DATA_DIR)
    binned.to_csv(OUT_DATA_DIR / "adjusted_binned_inputs_0p2kyr_full_support.csv", index=False)
    fit_frame.to_csv(OUT_DATA_DIR / "adjusted_binned_inputs_0p2kyr_model_support.csv", index=False)
    scale_summary.to_csv(OUT_DATA_DIR / "adjusted_forcing_scale_summary.csv", index=False)
    phase_extrema.to_csv(OUT_DATA_DIR / "adjusted_precession_phase_extrema.csv", index=False)
    resolution_by_bin.to_csv(OUT_DATA_DIR / "cheng_composite_sampling_resolution_by_bin.csv", index=False)
    model_summary.to_csv(OUT_DATA_DIR / "adjusted_poisson_model_summary_5kyr_history.csv", index=False)
    coefficients.to_csv(OUT_DATA_DIR / "adjusted_poisson_coefficients_5kyr_history.csv", index=False)
    likelihood_tests.to_csv(OUT_DATA_DIR / "adjusted_likelihood_tests_5kyr_history.csv", index=False)
    fitted_rates.to_csv(OUT_DATA_DIR / "adjusted_fitted_rates_5kyr_history.csv", index=False)
    sensitivity_binned.to_csv(OUT_DATA_DIR / "history_window_binned_inputs.csv", index=False)
    sensitivity_summary.to_csv(OUT_DATA_DIR / "history_window_model_summary.csv", index=False)
    sensitivity_lrt.to_csv(OUT_DATA_DIR / "history_window_likelihood_tests.csv", index=False)
    history_sensitivity.to_csv(OUT_DATA_DIR / "history_window_phase_sensitivity_summary.csv", index=False)
    pd.DataFrame(
        [
            {
                "run_name": RUN_NAME,
                "analysis_start_ka": base.ANALYSIS_START_KA,
                "analysis_end_ka": base.ANALYSIS_END_KA,
                "bin_width_ka": base.BIN_WIDTH_KA,
                "main_history_window_ka": MAIN_HISTORY_WINDOW_KA,
                "history_window_grid_ka": ",".join(f"{v:g}" for v in HISTORY_WINDOW_GRID_KA),
                "history_definition": "same-type event count in older interval (t, t + W]",
                "resolution_definition": "log local age spacing interpolated from Cheng 2016 composite record",
                "baseline_model": "event-process baseline: same-type history + Cheng log-resolution",
                "main_model": "full predictive model: event-process baseline + LR04 + CO2 + sin(pre_phase) + cos(pre_phase)",
                "predictive_information": "logL_full - logL_reduced, reported as bits/event and bits/bin",
                "pre_phase_convention": "precession-index minima=0 rad; maxima=pi rad",
            }
        ]
    ).to_csv(OUT_DATA_DIR / "parameters.csv", index=False)


def plot_inputs_and_rates(
    binned: pd.DataFrame,
    fit_frame: pd.DataFrame,
    fitted_rates: pd.DataFrame,
    likelihood_tests: pd.DataFrame,
    write_pdf: bool,
) -> None:
    fig = plt.figure(figsize=(13, 9.6))
    outer = fig.add_gridspec(
        2,
        1,
        height_ratios=[3.35, 2.70],
        hspace=0.18,
    )
    top_grid = outer[0].subgridspec(
        5,
        1,
        height_ratios=[0.95, 0.72, 0.72, 0.92, 0.62],
        hspace=0.05,
    )
    top_axes: list[plt.Axes] = []
    for idx in range(5):
        ax = fig.add_subplot(top_grid[idx, 0], sharex=top_axes[0] if top_axes else None)
        top_axes.append(ax)
    rate_grid = outer[1].subgridspec(2, 1, hspace=0.10)
    rate_axes = [
        fig.add_subplot(rate_grid[0, 0], sharex=top_axes[0]),
        fig.add_subplot(rate_grid[1, 0], sharex=top_axes[0]),
    ]
    axes = top_axes + rate_axes

    cheng = load_cheng_composite_record()
    base_rows = binned[binned["dataset_id"].eq("strong_monsoon_start")]

    pre_ax = axes[0].twinx()
    axes[0].set_zorder(2)
    pre_ax.set_zorder(1)
    axes[0].patch.set_visible(False)
    pre_ax.plot(
        base_rows["bin_center_ka"],
        base_rows["precession_index"],
        color="#4d4d4d",
        lw=0.75,
        alpha=0.32,
    )
    pre_ax.set_ylabel("precession\nindex", color="#5f5f5f")
    pre_ax.tick_params(axis="y", colors="#5f5f5f", labelsize=8, length=2.5)

    axes[0].plot(base_rows["bin_center_ka"], base_rows["pre_phase_deg"], color="#7570b3", lw=0.9)
    axes[0].set_ylabel("precession\nphase")
    axes[0].set_yticks([0, 180, 360])

    axes[1].plot(base_rows["bin_center_ka"], base_rows["lr04"], color="#1b9e77", lw=1.0)
    axes[1].set_ylabel(r"LR04 $\delta^{18}$O")
    axes[1].invert_yaxis()

    axes[2].plot(base_rows["bin_center_ka"], base_rows["co2"], color="#d95f02", lw=1.0)
    axes[2].set_ylabel("CO$_2$")

    axes[3].plot(cheng["age_ka"], cheng["d18o"], color="#202020", lw=0.75)
    axes[3].set_ylabel(r"$\delta^{18}$O")
    axes[3].invert_yaxis()

    density_values = 1.0 / base_rows["cheng_local_resolution_ka"]
    axes[4].plot(
        base_rows["bin_center_ka"],
        density_values,
        color="#4d4d4d",
        lw=0.9,
    )
    axes[4].set_ylabel("density")
    axes[4].set_ylim(0.0, np.nanpercentile(density_values, 98.0) * 1.05)

    for ax_idx, dataset_id in enumerate(base.DATASET_SETTINGS, start=5):
        ax = axes[ax_idx]
        settings = base.DATASET_SETTINGS[dataset_id]
        data = fit_frame[fit_frame["dataset_id"].eq(dataset_id)]
        event_bins = data[data["event_count"] > 0]
        ax.vlines(
            event_bins["bin_center_ka"],
            0.0,
            1.0,
            transform=ax.get_xaxis_transform(),
            color=settings["color"],
            lw=0.75,
            alpha=0.55,
            zorder=1,
        )
        rate_style = {
            "history_resolution_baseline": {
                "color": "#8c8c8c",
                "lw": 1.0,
                "ls": "-",
                "alpha": 0.72,
                "zorder": 2,
            },
            "baseline_climate_lr04_co2": {
                "color": MODEL_COLORS["baseline_climate_lr04_co2"],
                "lw": 1.25,
                "ls": "-",
                "alpha": 1.0,
                "zorder": 3,
            },
            "baseline_climate_lr04_co2_pre_phase": {
                "color": MODEL_COLORS["baseline_climate_lr04_co2_pre_phase"],
                "lw": 1.25,
                "ls": "-",
                "alpha": 1.0,
                "zorder": 4,
            },
        }
        for model_id in (
            "history_resolution_baseline",
            "baseline_climate_lr04_co2",
            "baseline_climate_lr04_co2_pre_phase",
        ):
            rates = fitted_rates[
                fitted_rates["dataset_id"].eq(dataset_id)
                & fitted_rates["model_id"].eq(model_id)
            ]
            style = rate_style[model_id]
            ax.plot(
                rates["bin_center_ka"],
                rates["lambda_per_kyr"],
                color=style["color"],
                lw=style["lw"],
                ls=style["ls"],
                alpha=style["alpha"],
                zorder=style["zorder"],
                label=rates["model_label"].iloc[0],
            )
        phase_test = likelihood_tests[
            likelihood_tests["dataset_id"].eq(dataset_id)
            & likelihood_tests["comparison_id"].eq("phase_after_adjusted_climate")
        ].iloc[0]
        ax.text(
            0.99,
            0.94,
            "Precession phase after climate-state model: "
            f"LR={phase_test['LR_statistic']:.2f}, p={format_p_value(phase_test['LR_p_value'])}",
            transform=ax.transAxes,
            ha="right",
            va="top",
            bbox={"boxstyle": "round,pad=0.25", "facecolor": "white", "edgecolor": "#bbbbbb", "alpha": 0.88},
        )
        ax.set_ylabel("rate / kyr")
        if ax_idx == 5:
            legend_handles = [
                Line2D(
                    [0],
                    [0],
                    marker="|",
                    linestyle="None",
                    markersize=13,
                    markeredgewidth=1.6,
                    color=base.DATASET_SETTINGS["strong_monsoon_start"]["color"],
                    label="Strong monsoon starts",
                ),
                Line2D(
                    [0],
                    [0],
                    marker="|",
                    linestyle="None",
                    markersize=13,
                    markeredgewidth=1.6,
                    color=base.DATASET_SETTINGS["weak_monsoon_start"]["color"],
                    label="Weak monsoon starts",
                ),
                Line2D(
                    [0],
                    [0],
                    color="#8c8c8c",
                    lw=1.2,
                    ls="-",
                    alpha=0.78,
                    label="Event-process baseline",
                ),
                Line2D(
                    [0],
                    [0],
                    color=MODEL_COLORS["baseline_climate_lr04_co2"],
                    lw=1.4,
                    label="Climate-state model",
                ),
                Line2D(
                    [0],
                    [0],
                    color=MODEL_COLORS["baseline_climate_lr04_co2_pre_phase"],
                    lw=1.4,
                    label="Full predictive model",
                ),
            ]
            ax.legend(
                handles=legend_handles,
                loc="lower right",
                bbox_to_anchor=(1.0, 1.05),
                ncol=3,
                frameon=False,
                borderaxespad=0.0,
            )

    for ax in top_axes:
        ax.grid(False)
        ax.set_xlim(base.ANALYSIS_START_KA, base.ANALYSIS_END_KA)
        ax.tick_params(axis="x", labelbottom=False, length=0)
        ax.tick_params(axis="y", length=2.5, pad=2)
        for spine in ax.spines.values():
            spine.set_visible(False)
    # Panel a is a stacked input panel. Show only tick marks along its lower
    # edge so readers can align it with panels b/c, while keeping labels only on
    # the final shared x axis.
    top_axes[-1].tick_params(
        axis="x",
        labelbottom=False,
        bottom=True,
        top=False,
        length=3.0,
        width=0.8,
        direction="out",
    )
    for spine in pre_ax.spines.values():
        spine.set_visible(False)

    for ax in rate_axes:
        ax.grid(False)
        ax.set_xlim(base.ANALYSIS_START_KA, base.ANALYSIS_END_KA)
    pre_ax.grid(False)
    pre_ax.set_xlim(base.ANALYSIS_START_KA, base.ANALYSIS_END_KA)
    rate_axes[0].tick_params(axis="x", labelbottom=False)
    axes[-1].set_xlabel("Age (kyr BP)")

    top_axes[0].text(
        -0.045,
        1.08,
        "a",
        transform=top_axes[0].transAxes,
        ha="right",
        va="bottom",
        fontsize=11,
        fontweight="bold",
        clip_on=False,
    )
    for label, ax in zip(("b", "c"), rate_axes):
        ax.text(
            -0.045,
            1.02,
            label,
            transform=ax.transAxes,
            ha="right",
            va="bottom",
            fontsize=11,
            fontweight="bold",
            clip_on=False,
        )

    fig.subplots_adjust(left=0.10, right=0.90, top=0.985, bottom=0.075)
    fig.canvas.draw()
    top_positions = [ax.get_position() for ax in top_axes]
    x0 = min(pos.x0 for pos in top_positions)
    y0 = min(pos.y0 for pos in top_positions)
    x1 = max(pos.x1 for pos in top_positions)
    y1 = max(pos.y1 for pos in top_positions)
    fig.add_artist(
        Rectangle(
            (x0, y0),
            x1 - x0,
            y1 - y0,
            transform=fig.transFigure,
            fill=False,
            edgecolor="#1f1f1f",
            linewidth=0.9,
            zorder=20,
        )
    )
    save_figure(fig, "fig01_adjusted_inputs_and_fitted_hazards", write_pdf)


def plot_model_comparison(model_summary: pd.DataFrame, likelihood_tests: pd.DataFrame, write_pdf: bool) -> None:
    fig, axes = plt.subplots(
        2,
        2,
        figsize=(13.6, 9.6),
        gridspec_kw={"height_ratios": [1.0, 1.08]},
    )
    for ax_idx, (ax, dataset_id) in enumerate(zip(axes[0], base.DATASET_SETTINGS)):
        sub = model_summary[model_summary["dataset_id"].eq(dataset_id)].sort_values("AICc")
        colors = [MODEL_COLORS.get(mid, "#777777") for mid in sub["model_id"]]
        y = np.arange(len(sub)) * 1.30
        labels = [
            MODEL_PLOT_LABELS.get(model_id, label)
            for model_id, label in zip(sub["model_id"], sub["model_label"])
        ]
        ax.barh(y, sub["delta_AICc"], height=0.76, color=colors, alpha=0.85)
        ax.set_yticks(y)
        ax.set_yticklabels(labels if ax_idx == 0 else [])
        if ax_idx == 1:
            ax.tick_params(axis="y", length=0)
        ax.set_ylim(y[-1] + 0.70, -0.70)
        ax.set_xlabel("Delta AICc from best model")
        ax.set_title(base.DATASET_SETTINGS[dataset_id]["label"], loc="left", fontsize=15.0)
        ax.tick_params(axis="both", labelsize=13.0)
        ax.xaxis.label.set_size(13.5)
        ax.grid(False)
        ax.text(
            -0.08,
            1.04,
            chr(ord("a") + ax_idx),
            transform=ax.transAxes,
            ha="right",
            va="bottom",
            fontsize=16,
            fontweight="bold",
            clip_on=False,
        )

    ordered_tests = [comparison_id for comparison_id, _, _, _ in LR_TEST_SPECS]
    max_x = float(
        -np.log10(
            np.maximum(
                likelihood_tests[likelihood_tests["comparison_id"].isin(ordered_tests)]["LR_p_value"].min(),
                1e-300,
            )
        )
    )
    xlim = max(max_x, -np.log10(0.05)) + 0.8
    threshold = -np.log10(0.05)
    for ax_idx, (ax, dataset_id) in enumerate(zip(axes[1], base.DATASET_SETTINGS)):
        sub = (
            likelihood_tests[
                likelihood_tests["dataset_id"].eq(dataset_id)
                & likelihood_tests["comparison_id"].isin(ordered_tests)
            ]
            .copy()
            .set_index("comparison_id")
            .reindex(ordered_tests)
            .reset_index()
        )
        sub["minus_log10_p"] = -np.log10(np.maximum(sub["LR_p_value"].astype(float), 1e-300))
        labels = sub["comparison_id"].map(LR_TEST_PLOT_LABELS)
        y = np.arange(len(sub)) * 1.28
        colors = ["#3182bd" if p < 0.05 else "#bdbdbd" for p in sub["LR_p_value"]]
        ax.barh(y, sub["minus_log10_p"], height=0.76, color=colors, alpha=0.86)
        ax.axvline(threshold, color="#222222", ls="--", lw=0.9)
        ax.set_yticks(y)
        ax.set_yticklabels(labels if ax_idx == 0 else [])
        if ax_idx == 1:
            ax.tick_params(axis="y", length=0)
        ax.set_ylim(y[-1] + 0.70, -0.70)
        ax.set_xlim(0.0, xlim)
        ax.set_xlabel("-log10 LR p value")
        ax.tick_params(axis="both", labelsize=13.0)
        ax.xaxis.label.set_size(13.5)
        ax.grid(False)
        ax.text(
            -0.08,
            1.04,
            chr(ord("c") + ax_idx),
            transform=ax.transAxes,
            ha="right",
            va="bottom",
            fontsize=16,
            fontweight="bold",
            clip_on=False,
        )
    fig.subplots_adjust(left=0.33, right=0.985, top=0.94, bottom=0.08, hspace=0.34, wspace=0.28)
    save_figure(fig, "fig02_adjusted_model_comparison_delta_aicc", write_pdf)


def plot_phase_response(model_summary: pd.DataFrame, write_pdf: bool) -> None:
    phase = np.linspace(0.0, 2.0 * np.pi, 361)
    fig, ax = plt.subplots(figsize=(8.8, 4.8))
    for dataset_id, settings in base.DATASET_SETTINGS.items():
        row = model_summary[
            model_summary["dataset_id"].eq(dataset_id)
            & model_summary["model_id"].eq("baseline_climate_lr04_co2_pre_phase")
        ].iloc[0]
        b_sin = float(row["beta_pre_phase_sin"])
        b_cos = float(row["beta_pre_phase_cos"])
        multiplier = np.exp(b_sin * np.sin(phase) + b_cos * np.cos(phase))
        multiplier = multiplier / np.mean(multiplier)
        preferred = float(row["pre_phase_preferred_rad"])
        ax.plot(np.degrees(phase), multiplier, color=settings["color"], lw=2.0, label=settings["label"])
        ax.axvline(np.degrees(preferred), color=settings["color"], lw=1.0, ls="--", alpha=0.7)
        ax.text(
            np.degrees(preferred),
            multiplier.max(),
            f"{np.degrees(preferred):.0f} deg",
            color=settings["color"],
            ha="center",
            va="bottom",
            fontsize=8,
        )
    ax.axhline(1.0, color="#666666", lw=0.8, ls=":")
    ax.set_xlim(0, 360)
    ax.set_xticks([0, 90, 180, 270, 360])
    ax.set_xlabel("Precession phase (deg; 0=min, 180=max)")
    ax.set_ylabel("Relative rate multiplier\n(adjusted terms held constant)")
    ax.set_title("Adjusted precession-phase effect", loc="left")
    ax.grid(True, color="#e6e6e6", lw=0.6)
    ax.legend(frameon=False, loc="upper right")
    save_figure(fig, "fig03_adjusted_phase_response", write_pdf)


def plot_history_window_sensitivity(history_sensitivity: pd.DataFrame, write_pdf: bool) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(13.2, 4.2))
    for dataset_id, settings in base.DATASET_SETTINGS.items():
        sub = history_sensitivity[history_sensitivity["dataset_id"].eq(dataset_id)]
        axes[0].plot(
            sub["history_window_ka"],
            sub["phase_after_adjusted_climate_p"],
            marker="o",
            color=settings["color"],
            lw=1.6,
            label=settings["label"],
        )
        axes[1].plot(
            sub["history_window_ka"],
            sub["phase_after_adjusted_climate_bits_per_event"],
            marker="o",
            color=settings["color"],
            lw=1.6,
        )
        axes[2].plot(
            sub["history_window_ka"],
            sub["pre_phase_preferred_deg"],
            marker="o",
            color=settings["color"],
            lw=1.6,
        )
    axes[0].axhline(0.05, color="#333333", lw=0.9, ls="--")
    axes[0].set_yscale("log")
    axes[0].set_ylabel("LR p value\nphase after climate-state model")
    axes[1].set_ylabel("Information gain\n(bits/event)")
    axes[2].set_ylabel("Preferred phase (deg)")
    axes[2].set_ylim(0, 360)
    axes[2].set_yticks([0, 90, 180, 270, 360])
    for idx, ax in enumerate(axes):
        ax.set_xlabel("History window (kyr)")
        ax.set_xticks(list(HISTORY_WINDOW_GRID_KA))
        ax.grid(True, color="#e6e6e6", lw=0.6)
        ax.text(
            -0.10,
            1.04,
            chr(ord("a") + idx),
            transform=ax.transAxes,
            ha="right",
            va="bottom",
            fontsize=11,
            fontweight="bold",
            clip_on=False,
        )
    axes[0].legend(frameon=False, loc="lower left")
    fig.subplots_adjust(left=0.08, right=0.98, top=0.92, bottom=0.16, wspace=0.34)
    save_figure(fig, "fig04_history_window_sensitivity", write_pdf)


def plot_history_window_table(history_sensitivity: pd.DataFrame, write_pdf: bool) -> None:
    rows = []
    for _, row in history_sensitivity.sort_values(["dataset_id", "history_window_ka"]).iterrows():
        rows.append(
            [
                row["dataset_label"].replace(" monsoon starts", ""),
                f"{row['history_window_ka']:.0f}",
                f"{row['n_events']:.0f}",
                f"{row['phase_after_adjusted_climate_LR']:.2f}",
                format_p_value(float(row["phase_after_adjusted_climate_p"])),
                f"{row['phase_after_adjusted_climate_delta_AICc']:.2f}",
                f"{row['phase_after_adjusted_climate_bits_per_event']:.3f}",
                f"{row['pre_phase_preferred_deg']:.1f}",
                f"{row['pre_phase_rate_ratio_max_vs_min']:.2f}",
            ]
        )
    columns = ["Event", "W", "N", "LR", "p", "Delta AICc", "bits/event", "phase", "rate ratio"]
    fig, ax = plt.subplots(figsize=(11.6, 0.42 * len(rows) + 1.2))
    ax.axis("off")
    table = ax.table(
        cellText=rows,
        colLabels=columns,
        cellLoc="center",
        colLoc="center",
        colColours=["#e6e6e6"] * len(columns),
        bbox=[0.0, 0.0, 1.0, 1.0],
    )
    table.auto_set_font_size(False)
    table.set_fontsize(8.5)
    table.scale(1.0, 1.16)
    for (row_idx, _), cell in table.get_celld().items():
        cell.set_edgecolor("#bdbdbd")
        cell.set_linewidth(0.45)
        if row_idx == 0:
            cell.set_text_props(weight="bold")
    save_figure(fig, "fig05_history_window_summary_table", write_pdf)


def run_analysis(write_pdf: bool) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    ensure_dir(OUT_DATA_DIR)
    ensure_dir(OUT_FIG_DIR)

    binned, scale_summary, phase_extrema, resolution_by_bin = build_adjusted_inputs(
        MAIN_HISTORY_WINDOW_KA
    )
    fit_frame = model_frame(binned)
    models = fit_adjusted_models(binned)
    model_summary = base.build_model_summary(models, fit_frame)
    coefficients = base.build_coefficient_table(models)
    likelihood_tests = build_adjusted_likelihood_tests(models, fit_frame, MAIN_HISTORY_WINDOW_KA)
    fitted_rates = base.build_fitted_rate_table(models, fit_frame)

    sensitivity_binned, sensitivity_model_summary, sensitivity_lrt = build_sensitivity_tables()
    history_sensitivity = build_history_sensitivity_summary(
        sensitivity_model_summary,
        sensitivity_lrt,
    )
    write_outputs(
        binned=binned,
        fit_frame=fit_frame,
        scale_summary=scale_summary,
        phase_extrema=phase_extrema,
        resolution_by_bin=resolution_by_bin,
        model_summary=model_summary,
        coefficients=coefficients,
        likelihood_tests=likelihood_tests,
        fitted_rates=fitted_rates,
        sensitivity_binned=sensitivity_binned,
        sensitivity_summary=sensitivity_model_summary,
        sensitivity_lrt=sensitivity_lrt,
        history_sensitivity=history_sensitivity,
    )

    plot_inputs_and_rates(binned, fit_frame, fitted_rates, likelihood_tests, write_pdf)
    plot_model_comparison(model_summary, likelihood_tests, write_pdf)
    plot_phase_response(model_summary, write_pdf)
    plot_history_window_sensitivity(history_sensitivity, write_pdf)
    plot_history_window_table(history_sensitivity, write_pdf)
    return model_summary, likelihood_tests, history_sensitivity


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fit event-process-baseline predictive Poisson hazard models."
    )
    parser.add_argument("--no-pdf", action="store_true", help="Only write PNG figures.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    model_summary, likelihood_tests, history_sensitivity = run_analysis(
        write_pdf=not args.no_pdf
    )
    print("Main 5 kyr history-window model ranking:")
    print(
        model_summary[
            [
                "dataset_id",
                "model_id",
                "n_events",
                "n_parameters",
                "log_likelihood",
                "AICc",
                "delta_AICc",
                "rank_AICc",
                "pre_phase_preferred_deg",
                "pre_phase_rate_ratio_max_vs_min",
            ]
        ].to_string(index=False)
    )
    print("\nMain likelihood-ratio / predictive-information tests:")
    print(
        likelihood_tests[
            [
                "dataset_id",
                "comparison_id",
                "df",
                "LR_statistic",
                "LR_p_value",
                "delta_AICc_full_minus_reduced",
                "info_bits_per_event",
            ]
        ].to_string(index=False)
    )
    print("\nHistory-window sensitivity: phase after climate-state model")
    print(
        history_sensitivity[
            [
                "history_window_ka",
                "dataset_id",
                "n_events",
                "phase_after_adjusted_climate_LR",
                "phase_after_adjusted_climate_p",
                "phase_after_adjusted_climate_delta_AICc",
                "phase_after_adjusted_climate_bits_per_event",
                "pre_phase_preferred_deg",
                "pre_phase_rate_ratio_max_vs_min",
            ]
        ].to_string(index=False)
    )
    print(f"\nWrote tables to: {OUT_DATA_DIR.relative_to(base.PROJECT_ROOT)}")
    print(f"Wrote figures to: {OUT_FIG_DIR.relative_to(base.PROJECT_ROOT)}")


if __name__ == "__main__":
    main()
