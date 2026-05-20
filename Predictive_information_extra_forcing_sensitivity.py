"""
Extra-forcing sensitivity experiment for the Rousseau et al. (2023)
predictive-information event-rate models.

The main predictive-information script fits a compact binned Poisson model:

    Y_i ~ Poisson(lambda_i * dt)
    log(lambda_i) = beta0 + history_i + resolution_i
                    + LR04_i + CO2_i + sin(pre_phase_i) + cos(pre_phase_i)

This script keeps the same binning, likelihood, and precession-phase
convention, then asks whether additional forcings contribute beyond the full
predictive model:

    AT, obliquity, eccentricity, and 65N summer-solstice insolation.

The main sensitivity checks are nested likelihood-ratio tests:

1. Add each extra forcing to the full predictive model one at a time.
2. Add all extra forcings jointly to the full predictive model.
3. In the full model, drop each extra forcing one at a time to estimate its
   unique contribution after the other extra forcings are also present.

The third check is useful because orbital terms and insolation can be strongly
correlated, so a variable can look helpful alone but add little once related
variables are already in the model.

Method sketch
-------------
This script does not introduce a new likelihood. It reuses the shared Poisson
event-rate fitter and changes only the candidate model list. The full
predictive model is treated as the scientific reference model:

    same-type history + Cheng composite sampling resolution
    + LR04 + CO2 + sin(pre_phase) + cos(pre_phase).

The extra predictors are range-scaled in exactly the same way as LR04 and CO2,
so their coefficients are effects per observed predictor range. The key output
is not the raw coefficient size, but the nested log-likelihood improvement
obtained when an extra term is added to, or removed from, a specified reference
model.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd
import xarray as xr
from paper_figure_export import save_paper_pdf
from toolbox.model_stats import nested_likelihood_metrics

import toolbox.event_inputs as event_inputs
import toolbox.poisson as poisson
from toolbox.project_config import (
    ANALYSIS_END_KA,
    ANALYSIS_START_KA,
    BIN_WIDTH_KA,
    DATASET_SETTINGS,
    PROJECT_ROOT,
)
import Predictive_information_model as predictive


RUN_NAME = "Predictive_information_extra_forcing_sensitivity"
OUT_DATA_DIR = PROJECT_ROOT / "data" / "processed" / RUN_NAME
OUT_FIG_DIR = PROJECT_ROOT / "figures" / RUN_NAME

AT_CSV = PROJECT_ROOT / "data/raw/AT.csv"
OBL_TXT = PROJECT_ROOT / "data/raw/obl_1000_60_inter100.txt"
ECC_TXT = PROJECT_ROOT / "data/raw/ecc_1000_60_inter100.txt"
INSOLATION_NC = PROJECT_ROOT / "data/raw/solstice_insolation_NH.nc"
INSOLATION_LATITUDE_DEG_N = 65.0

BASE_TERMS = predictive.FULL_TERMS
EXTRA_TERMS = ("AT_scaled", "obl_scaled", "ecc_scaled", "insol65N_scaled")
EXTRA_TERM_LABELS = {
    "AT_scaled": "Antarctic T",
    "obl_scaled": "obliquity",
    "ecc_scaled": "eccentricity",
    "insol65N_scaled": "65N solstice insolation",
}

MODEL_SPECS: list[tuple[str, tuple[str, ...], str]] = [
    ("stationary", (), "Stationary"),
    (
        "history_resolution_baseline",
        predictive.BASELINE_TERMS,
        "Event-process baseline",
    ),
    (
        "climate_state",
        predictive.BASELINE_TERMS + predictive.CLIMATE_TERMS,
        "Climate-state model",
    ),
    (
        "extended_baseline",
        BASE_TERMS,
        "Extended baseline\n(full predictive model)",
    ),
    ("base_plus_AT", BASE_TERMS + ("AT_scaled",), "Extended baseline + Antarctic T"),
    ("base_plus_obl", BASE_TERMS + ("obl_scaled",), "Extended baseline + obliquity"),
    ("base_plus_ecc", BASE_TERMS + ("ecc_scaled",), "Extended baseline + eccentricity"),
    (
        "base_plus_insol65N",
        BASE_TERMS + ("insol65N_scaled",),
        "Extended baseline + 65$^{\\circ}$N insolation",
    ),
    ("extended_all", BASE_TERMS + EXTRA_TERMS, "Extended baseline + all extra forcings"),
    (
        "all_without_AT",
        BASE_TERMS + ("obl_scaled", "ecc_scaled", "insol65N_scaled"),
        "All extra except Antarctic T",
    ),
    (
        "all_without_obl",
        BASE_TERMS + ("AT_scaled", "ecc_scaled", "insol65N_scaled"),
        "All extra except obliquity",
    ),
    (
        "all_without_ecc",
        BASE_TERMS + ("AT_scaled", "obl_scaled", "insol65N_scaled"),
        "All extra except eccentricity",
    ),
    (
        "all_without_insol65N",
        BASE_TERMS + ("AT_scaled", "obl_scaled", "ecc_scaled"),
        "All extra except 65$^{\\circ}$N insolation",
    ),
]

MODEL_COLORS = {
    "stationary": "#777777",
    "history_resolution_baseline": "#4d4d4d",
    "climate_state": "#1865C3",
    "extended_baseline": "#C51B7D",
    "base_plus_AT": "#8dd3c7",
    "base_plus_obl": "#80b1d3",
    "base_plus_ecc": "#fdb462",
    "base_plus_insol65N": "#bebada",
    "extended_all": "#202020",
    "all_without_AT": "#bdbdbd",
    "all_without_obl": "#bdbdbd",
    "all_without_ecc": "#bdbdbd",
    "all_without_insol65N": "#bdbdbd",
}

LR_TEST_ORDER = [
    "AT_after_base",
    "obl_after_base",
    "ecc_after_base",
    "insol65N_after_base",
    "all_extras_after_base",
    "AT_unique_in_full",
    "obl_unique_in_full",
    "ecc_unique_in_full",
    "insol65N_unique_in_full",
]

LR_TEST_SHORT_LABELS = {
    "AT_after_base": "Add Antarctic T to extended baseline",
    "obl_after_base": "Add obliquity to extended baseline",
    "ecc_after_base": "Add eccentricity to extended baseline",
    "insol65N_after_base": "Add 65$^{\\circ}$N insolation to extended baseline",
    "all_extras_after_base": "Add all extras to extended baseline",
    "AT_unique_in_full": "Antarctic T unique in full",
    "obl_unique_in_full": "Obliquity unique in full",
    "ecc_unique_in_full": "Eccentricity unique in full",
    "insol65N_unique_in_full": "65$^{\\circ}$N insolation unique in full",
}

MODEL_PLOT_LABELS = {
    "stationary": "Stationary",
    "history_resolution_baseline": "EP baseline",
    "climate_state": "Climate-state model",
    "extended_baseline": "Extended baseline",
    "base_plus_AT": "+ Antarctic T",
    "base_plus_obl": "+ obliquity",
    "base_plus_ecc": "+ eccentricity",
    "base_plus_insol65N": "+ $65^{\\circ}$N insolation",
    "extended_all": "+ all extra forcings",
    "all_without_AT": "All except Antarctic T",
    "all_without_obl": "All except obliquity",
    "all_without_ecc": "All except eccentricity",
    "all_without_insol65N": "All except $65^{\\circ}$N insolation",
}

LR_TEST_PLOT_LABELS = {
    "AT_after_base": "+Antarctic T vs extended baseline",
    "obl_after_base": "+obliquity vs extended baseline",
    "ecc_after_base": "+eccentricity vs extended baseline",
    "insol65N_after_base": "+$65^{\\circ}$N insolation vs extended baseline",
    "all_extras_after_base": "+all extras vs extended baseline",
    "AT_unique_in_full": "all extras vs -Antarctic T",
    "obl_unique_in_full": "all extras vs -obliquity",
    "ecc_unique_in_full": "all extras vs -eccentricity",
    "insol65N_unique_in_full": "all extras vs -$65^{\\circ}$N insolation",
}

PREDICTOR_TERMS = BASE_TERMS + EXTRA_TERMS
FOCUSED_CORRELATION_TERMS = BASE_TERMS
PREDICTOR_LABELS = {
    predictive.HISTORY_TERM: "history",
    predictive.RESOLUTION_TERM: "resolution",
    "lr04_scaled": "LR04",
    "co2_scaled": "CO2",
    "pre_phase_sin": "sin phase",
    "pre_phase_cos": "cos phase",
    "AT_scaled": "Antarctic T",
    "obl_scaled": "obliquity",
    "ecc_scaled": "eccentricity",
    "insol65N_scaled": "65$^{\\circ}$N insol",
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


# ---------------------------------------------------------------------------
# General utilities
# ---------------------------------------------------------------------------


def ensure_dir(path: Path) -> None:
    """Create an output directory if it is missing."""

    path.mkdir(parents=True, exist_ok=True)


def save_figure(fig: plt.Figure, stem: str, write_pdf: bool) -> None:
    """Save a sensitivity figure and optional paper-facing PDF."""

    ensure_dir(OUT_FIG_DIR)
    fig.savefig(OUT_FIG_DIR / f"{stem}.png", dpi=300, bbox_inches="tight")
    if write_pdf:
        fig.savefig(OUT_FIG_DIR / f"{stem}.pdf", bbox_inches="tight")
        save_paper_pdf(fig, PROJECT_ROOT, stem)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Input preparation and model fitting
# ---------------------------------------------------------------------------


def scale_interpolated_forcing(
    forcing_id: str,
    forcing_label: str,
    age_ka: np.ndarray,
    value: np.ndarray,
    centers_ka: np.ndarray,
    source: Path,
) -> tuple[np.ndarray, np.ndarray, dict]:
    """Interpolate one forcing to bin centers and range-scale it.

    The baseline GLM uses one row per 0.2 kyr bin, so every external forcing
    must be evaluated on that exact grid before it can be added as a predictor.
    Scaling follows the baseline script: subtract the mean and divide by the
    full observed range. This keeps the sensitivity coefficients comparable to
    LR04 and CO2 coefficients without changing model likelihoods in any
    qualitative way.
    """

    age, cleaned = event_inputs.clean_series(age_ka, value)
    interpolated = np.interp(centers_ka, age, cleaned)
    scaled, mean, vmin, vmax, value_range = event_inputs.scale_to_zero_mean_range_one(interpolated)
    meta = {
        "forcing_id": forcing_id,
        "forcing_label": forcing_label,
        "source": str(source.relative_to(PROJECT_ROOT)),
        "mean": mean,
        "min": vmin,
        "max": vmax,
        "range": value_range,
    }
    return interpolated, scaled, meta


def load_at(centers_ka: np.ndarray) -> tuple[np.ndarray, np.ndarray, dict]:
    """Load Antarctic temperature and interpolate it to bin centers."""

    raw = pd.read_csv(AT_CSV)
    return scale_interpolated_forcing(
        "AT",
        "Antarctic temperature",
        raw["age"].to_numpy(dtype=float) / 1000.0,
        raw["AT"].to_numpy(dtype=float),
        centers_ka,
        AT_CSV,
    )


def load_orbital_text(
    forcing_id: str,
    forcing_label: str,
    path: Path,
    centers_ka: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, dict]:
    """Load an orbital text file and interpolate/range-scale it."""

    raw = pd.read_csv(path, sep=r"\s+", header=None, names=["age_raw_ka", "value"])
    return scale_interpolated_forcing(
        forcing_id,
        forcing_label,
        -raw["age_raw_ka"].to_numpy(dtype=float),
        raw["value"].to_numpy(dtype=float),
        centers_ka,
        path,
    )


def load_insolation_65n(centers_ka: np.ndarray) -> tuple[np.ndarray, np.ndarray, dict]:
    """Load 65N summer-solstice insolation from the NetCDF grid."""

    ds = xr.open_dataset(INSOLATION_NC)
    try:
        latitudes = np.asarray(ds["latitude_degN"].to_numpy(), dtype=float)
        lat_index = int(np.argmin(np.abs(latitudes - INSOLATION_LATITUDE_DEG_N)))
        age = np.asarray(ds["age_kyr_BP"].to_numpy(), dtype=float)
        value = np.asarray(
            ds["daily_mean_insolation_Wm2"].isel(latitude=lat_index).to_numpy(),
            dtype=float,
        )
        forcing_label = f"{latitudes[lat_index]:g}N summer solstice insolation"
        raw, scaled, meta = scale_interpolated_forcing(
            "insol65N",
            forcing_label,
            age,
            value,
            centers_ka,
            INSOLATION_NC,
        )
    finally:
        ds.close()
    return raw, scaled, meta


def build_extra_forcing_frame(centers_ka: np.ndarray) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build extra covariates for the sensitivity experiment.

    The returned frame has one row per bin center and both raw and scaled
    versions of each added forcing. Only scaled columns enter the GLMs. Raw
    columns are kept in the output table so the preprocessing is auditable.
    """

    loaders = [
        ("AT", "AT_scaled", load_at),
        (
            "obl",
            "obl_scaled",
            lambda centers: load_orbital_text("obl", "obliquity", OBL_TXT, centers),
        ),
        (
            "ecc",
            "ecc_scaled",
            lambda centers: load_orbital_text("ecc", "eccentricity", ECC_TXT, centers),
        ),
        ("insol65N", "insol65N_scaled", load_insolation_65n),
    ]
    frame = pd.DataFrame({"bin_center_ka": centers_ka})
    scale_rows = []
    for raw_col, scaled_col, loader in loaders:
        raw, scaled, meta = loader(centers_ka)
        frame[raw_col] = raw
        frame[scaled_col] = scaled
        scale_rows.append(meta)
    return frame, pd.DataFrame(scale_rows)


def build_sensitivity_inputs() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Reuse the baseline binned inputs and append the extra forcings.

    This preserves the event binning, precession phase convention, LR04/CO2
    preprocessing, and response vector from the baseline analysis. The only
    difference is the addition of AT, obliquity, eccentricity, and 65N
    solstice-insolation predictors.
    """

    binned_inputs, base_scale_summary, phase_extrema, _ = predictive.build_predictive_inputs(
        predictive.MAIN_HISTORY_WINDOW_KA
    )
    first_dataset = next(iter(DATASET_SETTINGS))
    centers = (
        binned_inputs.loc[binned_inputs["dataset_id"].eq(first_dataset), "bin_center_ka"]
        .to_numpy(dtype=float)
    )
    extra_frame, extra_scale_summary = build_extra_forcing_frame(centers)
    binned_inputs = binned_inputs.merge(extra_frame, on="bin_center_ka", how="left", validate="many_to_one")
    scale_summary = pd.concat([base_scale_summary, extra_scale_summary], ignore_index=True)
    return binned_inputs, scale_summary, phase_extrema


def fit_sensitivity_models(binned_inputs: pd.DataFrame) -> list:
    """Fit every sensitivity model for strong and weak event datasets.

    The optimizer and Poisson likelihood are imported from the core model
    script. This function only loops over the expanded model specification.
    """

    fit_frame = predictive.model_frame(binned_inputs)
    models = []
    for _, group in fit_frame.groupby("dataset_id", sort=False):
        for model_id, terms, label in MODEL_SPECS:
            models.append(poisson.fit_poisson_model(group, model_id, terms, label))
    return models


def build_sensitivity_likelihood_tests(models: list, fit_frame: pd.DataFrame) -> pd.DataFrame:
    """Nested LR tests for extra-forcing contribution.

    In this script "base" means the extended scientific reference model:

        event-process baseline + LR04 + CO2 + precession phase.

    It is different from the event-process baseline in
    ``Predictive_information_model.py``, which contains only history
    and sampling resolution.

    There are three logically different comparison families:

    1. add_to_base: add one extra forcing to the extended baseline. This answers
       whether that one predictor helps when LR04, CO2, and precession phase
       are already present.
    2. joint_add_to_base: add all extra forcings together. This asks whether
       the extra block has any collective information.
    3. drop_one_from_full: compare the full model against a model missing one
       extra forcing. This estimates the unique contribution of that forcing
       after the other extras are also included.

    In every row, the reported LR p value comes from the same Wilks-style
    nested-model approximation used in the main script:

        LR = 2 * (logL_full - logL_reduced)
        df = number of added coefficients.
    """

    lookup = poisson.model_lookup(models)
    comparisons = [
        (
            "full_vs_ep_baseline",
            "main_context",
            "history_resolution_baseline",
            "extended_baseline",
            "Full predictive model vs event-process baseline",
        ),
        (
            "phase_after_climate_state",
            "main_context",
            "climate_state",
            "extended_baseline",
            "Precession phase after climate-state model",
        ),
        (
            "AT_after_base",
            "add_to_base",
            "extended_baseline",
            "base_plus_AT",
            "Add Antarctic T to extended baseline",
        ),
        (
            "obl_after_base",
            "add_to_base",
            "extended_baseline",
            "base_plus_obl",
            "Add obliquity to extended baseline",
        ),
        (
            "ecc_after_base",
            "add_to_base",
            "extended_baseline",
            "base_plus_ecc",
            "Add eccentricity to extended baseline",
        ),
        (
            "insol65N_after_base",
            "add_to_base",
            "extended_baseline",
            "base_plus_insol65N",
            "Add 65$^{\\circ}$N insolation to extended baseline",
        ),
        (
            "all_extras_after_base",
            "joint_add_to_base",
            "extended_baseline",
            "extended_all",
            "Add all extra forcings to extended baseline",
        ),
        (
            "AT_unique_in_full",
            "drop_one_from_full",
            "all_without_AT",
            "extended_all",
            "Antarctic T unique in full",
        ),
        (
            "obl_unique_in_full",
            "drop_one_from_full",
            "all_without_obl",
            "extended_all",
            "Obliquity unique in full",
        ),
        (
            "ecc_unique_in_full",
            "drop_one_from_full",
            "all_without_ecc",
            "extended_all",
            "Eccentricity unique in full",
        ),
        (
            "insol65N_unique_in_full",
            "drop_one_from_full",
            "all_without_insol65N",
            "extended_all",
            "65$^{\\circ}$N insolation unique in full",
        ),
    ]
    rows = []
    support = {
        dataset_id: {
            "n_events": int(group["event_count"].sum()),
            "n_bins": int(len(group)),
        }
        for dataset_id, group in fit_frame.groupby("dataset_id", sort=False)
    }
    for dataset_id in DATASET_SETTINGS:
        n_events = support[dataset_id]["n_events"]
        n_bins = support[dataset_id]["n_bins"]
        for comparison_id, family, reduced_id, full_id, label in comparisons:
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
                    "comparison_id": comparison_id,
                    "comparison_family": family,
                    "comparison_label": label,
                    "reduced_model_id": reduced_id,
                    "full_model_id": full_id,
                    **metrics,
                }
            )
    return pd.DataFrame(rows)


def build_predictor_correlation_table(binned_inputs: pd.DataFrame) -> pd.DataFrame:
    """Measure collinearity among baseline and extra predictors.

    The same predictor grid is used for strong and weak datasets, so one
    representative dataset is enough. This table helps interpret the nested
    tests: if two orbital predictors are highly correlated, their individual
    coefficients and drop-one p values can be unstable even when the model is
    correctly specified.
    """

    first_dataset = next(iter(DATASET_SETTINGS))
    base = binned_inputs[binned_inputs["dataset_id"].eq(first_dataset)]
    corr = base.loc[:, list(PREDICTOR_TERMS)].corr()
    corr.index.name = "term_1"
    out = corr.reset_index().melt(id_vars="term_1", var_name="term_2", value_name="correlation")
    return out


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------


def draw_model_delta_aicc(ax: plt.Axes, summary: pd.DataFrame, dataset_id: str) -> None:
    """Draw one Delta AICc panel relative to the extended baseline."""

    model_order = [model_id for model_id, _, _ in MODEL_SPECS]
    sub = summary[summary["dataset_id"].eq(dataset_id)].set_index("model_id").reindex(model_order)
    model_labels = [MODEL_PLOT_LABELS.get(model_id, label) for model_id, label in zip(sub.index, sub["model_label"])]
    base_aicc = float(sub.loc["extended_baseline", "AICc"])
    delta_aicc_from_base = sub["AICc"] - base_aicc
    colors = [MODEL_COLORS.get(model_id, "#999999") for model_id in sub.index]
    y = np.arange(len(sub)) * 1.25
    ax.barh(y, delta_aicc_from_base, height=0.76, color=colors, alpha=0.86)
    ax.axvline(0.0, color="#222222", lw=0.8)
    ax.set_yticks(y)
    ax.set_yticklabels(model_labels)
    ax.set_ylim(y[-1] + 0.68, -0.68)
    ax.set_xlabel("Delta AICc relative to extended baseline")
    ax.set_title(DATASET_SETTINGS[dataset_id]["label"], loc="left", fontsize=15.0)
    ax.tick_params(axis="both", labelsize=12.5)
    ax.xaxis.label.set_size(13.5)
    ax.set_xlim(min(-2.0, float(delta_aicc_from_base.min()) - 0.8), float(delta_aicc_from_base.max()) + 2.0)
    ax.grid(False)


def ordered_likelihood_tests(likelihood_tests: pd.DataFrame, dataset_id: str) -> pd.DataFrame:
    """Return likelihood tests in the order used by sensitivity figures."""

    order = {comparison_id: idx for idx, comparison_id in enumerate(LR_TEST_ORDER)}
    sub = likelihood_tests[
        likelihood_tests["dataset_id"].eq(dataset_id)
        & likelihood_tests["comparison_id"].isin(LR_TEST_ORDER)
    ].copy()
    sub["sort_key"] = sub["comparison_id"].map(order)
    sub = sub.sort_values("sort_key").reset_index(drop=True)
    sub["minus_log10_p"] = -np.log10(np.maximum(sub["LR_p_value"], 1e-300))
    sub["plot_label"] = sub["comparison_id"].map(LR_TEST_PLOT_LABELS).fillna(
        sub["comparison_id"].map(LR_TEST_SHORT_LABELS)
    ).fillna(
        sub["comparison_label"]
    )

    y_positions: list[float] = []
    y_pos = 0.0
    for comparison_id in sub["comparison_id"]:
        y_positions.append(y_pos)
        y_pos += 1.25
        if comparison_id in {"insol65N_after_base", "all_extras_after_base"}:
            y_pos += 0.88
    sub["y_position"] = y_positions
    return sub


def likelihood_test_axis_limit(likelihood_tests: pd.DataFrame) -> float:
    """Choose an x-axis limit for -log10 p-value likelihood-test panels."""

    sub = likelihood_tests[likelihood_tests["comparison_id"].isin(LR_TEST_ORDER)].copy()
    if sub.empty:
        return -np.log10(0.05) + 1.0
    minus_log10_p = -np.log10(np.maximum(sub["LR_p_value"], 1e-300))
    return max(float(minus_log10_p.max()), -np.log10(0.05)) + 2.1


def draw_likelihood_tests_merged(
    ax: plt.Axes,
    likelihood_tests: pd.DataFrame,
    dataset_id: str,
    xlim: float,
    show_title: bool = True,
    annotate: bool = True,
) -> None:
    """Draw one merged likelihood-test bar panel."""

    sub = ordered_likelihood_tests(likelihood_tests, dataset_id)
    threshold = -np.log10(0.05)
    colors = ["#3182bd" if p < 0.05 else "#bdbdbd" for p in sub["LR_p_value"]]
    ax.barh(sub["y_position"], sub["minus_log10_p"], height=0.78, color=colors, alpha=0.86)
    ax.axvline(threshold, color="#222222", ls="--", lw=0.9)
    ax.set_yticks(sub["y_position"])
    ax.set_yticklabels(sub["plot_label"])
    ax.set_ylim(float(sub["y_position"].max()) + 0.72, -0.72)
    ax.set_xlim(0.0, xlim)
    ax.set_xlabel("-log10 LR p value")
    if show_title:
        ax.set_title(DATASET_SETTINGS[dataset_id]["label"], loc="left", fontsize=15.0)
    ax.tick_params(axis="both", labelsize=12.0)
    ax.xaxis.label.set_size(13.5)
    ax.grid(False)

    if annotate:
        for _, test in sub.iterrows():
            ax.text(
                test["minus_log10_p"] + 0.05,
                test["y_position"],
                f"p={test['LR_p_value']:.3g}, dAICc={test['delta_AICc_full_minus_reduced']:.2f}",
                va="center",
                fontsize=10.5,
            )


def plot_model_delta_aicc(summary: pd.DataFrame, write_pdf: bool) -> None:
    """Plot sensitivity-model Delta AICc for both event catalogues."""

    fig, axes = plt.subplots(1, 2, figsize=(17.2, 6.4))
    for ax, dataset_id in zip(axes, DATASET_SETTINGS):
        draw_model_delta_aicc(ax, summary, dataset_id)
    fig.subplots_adjust(left=0.23, right=0.98, top=0.92, bottom=0.12, wspace=0.78)
    save_figure(fig, "fig01_sensitivity_delta_AICc", write_pdf)


def plot_likelihood_tests(likelihood_tests: pd.DataFrame, write_pdf: bool) -> None:
    """Plot sensitivity likelihood tests as standalone panels."""

    fig, axes = plt.subplots(1, 2, figsize=(15.2, 6.2), sharex=True)
    xlim = 1.6
    for ax, dataset_id in zip(axes, DATASET_SETTINGS):
        draw_likelihood_tests_merged(ax, likelihood_tests, dataset_id, xlim)
    fig.subplots_adjust(left=0.23, right=0.98, top=0.92, bottom=0.12, wspace=0.58)
    save_figure(fig, "fig02_sensitivity_likelihood_tests", write_pdf)


def plot_aicc_and_likelihood_tests_combined(
    summary: pd.DataFrame, likelihood_tests: pd.DataFrame, write_pdf: bool
) -> None:
    """Plot AICc and likelihood-test results in the combined SI figure."""

    fig, axes = plt.subplots(
        2,
        2,
        figsize=(13.8, 11.4),
        gridspec_kw={"height_ratios": [1.0, 1.05]},
    )
    xlim = 1.48
    for col_idx, dataset_id in enumerate(DATASET_SETTINGS):
        draw_model_delta_aicc(axes[0, col_idx], summary, dataset_id)
        draw_likelihood_tests_merged(
            axes[1, col_idx],
            likelihood_tests,
            dataset_id,
            xlim,
            show_title=False,
            annotate=False,
        )
    for ax in axes[:, 1]:
        ax.set_yticklabels([])
        ax.tick_params(axis="y", length=0)
    axes[0, 0].text(
        -0.24, 1.05, "a", transform=axes[0, 0].transAxes, fontweight="bold", fontsize=16
    )
    axes[0, 1].text(
        -0.24, 1.05, "b", transform=axes[0, 1].transAxes, fontweight="bold", fontsize=16
    )
    axes[1, 0].text(
        -0.24, 1.05, "c", transform=axes[1, 0].transAxes, fontweight="bold", fontsize=16
    )
    axes[1, 1].text(
        -0.24, 1.05, "d", transform=axes[1, 1].transAxes, fontweight="bold", fontsize=16
    )
    fig.subplots_adjust(left=0.31, right=0.985, top=0.95, bottom=0.08, hspace=0.32, wspace=0.48)
    save_figure(fig, "fig06_sensitivity_aicc_and_likelihood_tests", write_pdf)


def plot_extended_coefficients(coefficients: pd.DataFrame, write_pdf: bool) -> None:
    """Plot coefficients for the extra forcings in the full sensitivity model."""

    terms = list(EXTRA_TERMS)
    labels = [EXTRA_TERM_LABELS[term] for term in terms]
    datasets = list(DATASET_SETTINGS)
    x = np.arange(len(terms))
    width = 0.32
    fig, ax = plt.subplots(figsize=(8.8, 4.8))
    offsets = np.linspace(-width / 2, width / 2, len(datasets))
    for offset, dataset_id in zip(offsets, datasets):
        sub = (
            coefficients[
                coefficients["dataset_id"].eq(dataset_id)
                & coefficients["model_id"].eq("extended_all")
            ]
            .set_index("term")
            .reindex(terms)
        )
        ax.bar(
            x + offset,
            sub["beta"],
            width=width,
            label=DATASET_SETTINGS[dataset_id]["label"],
            alpha=0.86,
        )
    ax.axhline(0.0, color="#222222", lw=0.9)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=20, ha="right")
    ax.set_ylabel("Full-model beta in log(lambda)")
    ax.set_title("Extra forcing coefficients in the full sensitivity model", loc="left")
    ax.grid(True, axis="y", color="#e6e6e6", lw=0.6)
    ax.legend(frameon=False, loc="upper right")
    fig.subplots_adjust(bottom=0.24)
    save_figure(fig, "fig03_extended_extra_coefficients", write_pdf)


def plot_base_vs_extended_rates(
    binned_inputs: pd.DataFrame,
    fitted_rates: pd.DataFrame,
    likelihood_tests: pd.DataFrame,
    write_pdf: bool,
) -> None:
    """Compare fitted rates from the extended baseline and all-extra model."""

    fig, axes = plt.subplots(2, 1, figsize=(9.2, 5.6), sharex=True)
    for ax_idx, (ax, dataset_id) in enumerate(zip(axes, DATASET_SETTINGS)):
        settings = DATASET_SETTINGS[dataset_id]
        data = binned_inputs[binned_inputs["dataset_id"].eq(dataset_id)]
        events = data[data["event_count"] > 0]
        ax.vlines(
            events["bin_center_ka"],
            0.0,
            1.0,
            transform=ax.get_xaxis_transform(),
            color=settings["color"],
            lw=0.8,
            alpha=0.45,
            label="observed event bins",
            zorder=1,
        )
        for model_id, color, label in [
            ("extended_baseline", "#C51B7D", "extended baseline"),
            ("extended_all", "#202020", "extended baseline + extras"),
        ]:
            rates = fitted_rates[
                fitted_rates["dataset_id"].eq(dataset_id)
                & fitted_rates["model_id"].eq(model_id)
            ]
            ax.plot(
                rates["bin_center_ka"],
                rates["lambda_per_kyr"],
                color=color,
                lw=1.8,
                label=label,
                zorder=3,
            )
        test = likelihood_tests[
            likelihood_tests["dataset_id"].eq(dataset_id)
            & likelihood_tests["comparison_id"].eq("all_extras_after_base")
        ].iloc[0]
        ax.text(
            0.01,
            0.94,
            f"{settings['label']}: N={int(data['event_count'].sum())}, "
            f"all extras after extended baseline p={test['LR_p_value']:.3g}",
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=10.5,
            bbox={"boxstyle": "round,pad=0.25", "facecolor": "white", "edgecolor": "#bbbbbb", "alpha": 0.86},
        )
        ax.set_ylabel("rate / kyr")
        ax.tick_params(axis="both", labelsize=11.5)
        ax.xaxis.label.set_size(12.5)
        ax.yaxis.label.set_size(12.5)
        ax.grid(False)
        ax.text(
            -0.045,
            1.03,
            chr(ord("a") + ax_idx),
            transform=ax.transAxes,
            ha="right",
            va="bottom",
            fontsize=14,
            fontweight="bold",
            clip_on=False,
        )
    legend_handles = [
        Line2D(
            [0],
            [0],
            color=DATASET_SETTINGS["strong_monsoon_start"]["color"],
            lw=1.0,
            alpha=0.45,
            label="Strong monsoon start",
        ),
        Line2D(
            [0],
            [0],
            color=DATASET_SETTINGS["weak_monsoon_start"]["color"],
            lw=1.0,
            alpha=0.45,
            label="Weak monsoon start",
        ),
        Line2D([0], [0], color="#C51B7D", lw=1.8, label="extended baseline"),
        Line2D([0], [0], color="#202020", lw=1.8, label="extended baseline + extras"),
    ]
    axes[0].legend(
        handles=legend_handles,
        frameon=False,
        loc="lower right",
        bbox_to_anchor=(1.0, 1.06),
        ncol=4,
        fontsize=8.5,
        borderaxespad=0.0,
    )
    axes[-1].set_xlim(ANALYSIS_END_KA, ANALYSIS_START_KA)
    axes[-1].set_xlabel("Age (ka BP)")
    fig.subplots_adjust(left=0.11, right=0.98, top=0.86, bottom=0.12, hspace=0.22)
    save_figure(fig, "fig04_base_vs_extended_hazards", write_pdf)


def plot_predictor_correlation(correlation_table: pd.DataFrame, write_pdf: bool) -> None:
    """Plot the full predictor correlation matrix on the analysis grid."""

    corr = correlation_table.pivot(index="term_1", columns="term_2", values="correlation")
    corr = corr.loc[list(PREDICTOR_TERMS), list(PREDICTOR_TERMS)]
    fig, ax = plt.subplots(figsize=(7.4, 6.2))
    image = ax.imshow(corr.to_numpy(dtype=float), vmin=-1.0, vmax=1.0, cmap="coolwarm")
    labels = [PREDICTOR_LABELS[term] for term in PREDICTOR_TERMS]
    ax.set_xticks(np.arange(len(labels)))
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_yticks(np.arange(len(labels)))
    ax.set_yticklabels(labels)
    for i in range(corr.shape[0]):
        for j in range(corr.shape[1]):
            value = corr.iloc[i, j]
            ax.text(j, i, f"{value:.2f}", ha="center", va="center", fontsize=7)
    cbar = fig.colorbar(image, ax=ax, pad=0.015)
    cbar.set_label("Pearson correlation")
    ax.set_title("Predictor correlation on the 0.2 kyr analysis grid", loc="left")
    fig.subplots_adjust(bottom=0.22, left=0.18)
    save_figure(fig, "fig05_predictor_correlation", write_pdf)


def plot_focused_baseline_correlation(correlation_table: pd.DataFrame, write_pdf: bool) -> None:
    """Plot correlations among the variables in the extended baseline only."""

    corr = correlation_table.pivot(index="term_1", columns="term_2", values="correlation")
    corr = corr.loc[list(FOCUSED_CORRELATION_TERMS), list(FOCUSED_CORRELATION_TERMS)]
    # For display only, flip LR04 so that the slow-background pair
    # (-LR04, CO2) has a positive correlation in the matrix. The fitted models
    # and saved likelihood tables still use the original LR04 orientation.
    display_sign = pd.Series(1.0, index=corr.index)
    display_sign.loc["lr04_scaled"] = -1.0
    corr = corr.mul(display_sign, axis=0).mul(display_sign, axis=1)
    fig, ax = plt.subplots(figsize=(5.9, 5.2))
    image = ax.imshow(corr.to_numpy(dtype=float), vmin=-1.0, vmax=1.0, cmap="coolwarm")
    labels = [
        "-LR04" if term == "lr04_scaled" else PREDICTOR_LABELS[term]
        for term in FOCUSED_CORRELATION_TERMS
    ]
    ax.set_xticks(np.arange(len(labels)))
    ax.set_xticklabels(labels, rotation=35, ha="right")
    ax.set_yticks(np.arange(len(labels)))
    ax.set_yticklabels(labels)
    for i in range(corr.shape[0]):
        for j in range(corr.shape[1]):
            value = corr.iloc[i, j]
            text_color = "white" if abs(value) >= 0.65 else "#202020"
            ax.text(j, i, f"{value:.2f}", ha="center", va="center", fontsize=8, color=text_color)
    cbar = fig.colorbar(image, ax=ax, fraction=0.025, pad=0.09, shrink=0.68, aspect=35)
    cbar.set_label("Pearson correlation")
    cbar.ax.tick_params(labelsize=8, length=2.5)
    fig.subplots_adjust(bottom=0.20, left=0.23, right=0.84, top=0.98)
    fig.canvas.draw()
    cbar_pos = cbar.ax.get_position()
    cbar.ax.set_position([cbar_pos.x0, cbar_pos.y0 - 0.025, cbar_pos.width, cbar_pos.height])
    save_figure(fig, "fig07_extended_baseline_predictor_correlation", write_pdf)


def write_outputs(
    binned_inputs: pd.DataFrame,
    scale_summary: pd.DataFrame,
    phase_extrema: pd.DataFrame,
    summary: pd.DataFrame,
    coefficients: pd.DataFrame,
    likelihood_tests: pd.DataFrame,
    fitted_rates: pd.DataFrame,
    predictor_correlation: pd.DataFrame,
) -> None:
    """Write sensitivity input, model, test, and diagnostic tables."""

    ensure_dir(OUT_DATA_DIR)
    binned_inputs.to_csv(OUT_DATA_DIR / "sensitivity_binned_inputs_0p2kyr.csv", index=False)
    scale_summary.to_csv(OUT_DATA_DIR / "sensitivity_forcing_scale_summary.csv", index=False)
    phase_extrema.to_csv(OUT_DATA_DIR / "sensitivity_precession_phase_extrema.csv", index=False)
    summary.to_csv(OUT_DATA_DIR / "sensitivity_poisson_model_summary.csv", index=False)
    coefficients.to_csv(OUT_DATA_DIR / "sensitivity_poisson_coefficients.csv", index=False)
    likelihood_tests.to_csv(OUT_DATA_DIR / "sensitivity_likelihood_tests.csv", index=False)
    fitted_rates.to_csv(OUT_DATA_DIR / "sensitivity_fitted_rates_by_bin.csv", index=False)
    predictor_correlation.to_csv(OUT_DATA_DIR / "sensitivity_predictor_correlation.csv", index=False)

    params = pd.DataFrame(
        [
            {
                "analysis_start_ka": ANALYSIS_START_KA,
                "analysis_end_ka": ANALYSIS_END_KA,
                "bin_width_ka": BIN_WIDTH_KA,
                "history_window_ka": predictive.MAIN_HISTORY_WINDOW_KA,
                "response": "event_count_per_0p2kyr_bin",
                "baseline_model": "same-type history + Cheng log-resolution + LR04_scaled + CO2_scaled + sin(pre_phase) + cos(pre_phase)",
                "extra_forcings": "AT_scaled + obl_scaled + ecc_scaled + insol65N_scaled",
                "main_question": "Do extra forcings improve the extended baseline predictive-information model?",
            }
        ]
    )
    params.to_csv(OUT_DATA_DIR / "parameters.csv", index=False)


# ---------------------------------------------------------------------------
# Command-line workflow
# ---------------------------------------------------------------------------


def run_analysis(write_pdf: bool) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Run the complete sensitivity experiment.

    The execution order mirrors the statistical question:
    build a baseline-compatible design matrix, fit the expanded candidate model
    set, summarize AICc/coefficient evidence, run nested tests, then generate
    figures and tables.
    """

    ensure_dir(OUT_DATA_DIR)
    ensure_dir(OUT_FIG_DIR)
    binned_inputs, scale_summary, phase_extrema = build_sensitivity_inputs()
    fit_frame = predictive.model_frame(binned_inputs)
    models = fit_sensitivity_models(binned_inputs)
    summary = poisson.build_model_summary(models, fit_frame)
    coefficients = poisson.build_coefficient_table(models)
    likelihood_tests = build_sensitivity_likelihood_tests(models, fit_frame)
    fitted_rates = poisson.build_fitted_rate_table(models, fit_frame)
    predictor_correlation = build_predictor_correlation_table(fit_frame)

    write_outputs(
        fit_frame,
        scale_summary,
        phase_extrema,
        summary,
        coefficients,
        likelihood_tests,
        fitted_rates,
        predictor_correlation,
    )

    plot_model_delta_aicc(summary, write_pdf)
    plot_likelihood_tests(likelihood_tests, write_pdf)
    plot_aicc_and_likelihood_tests_combined(summary, likelihood_tests, write_pdf)
    plot_extended_coefficients(coefficients, write_pdf)
    plot_base_vs_extended_rates(fit_frame, fitted_rates, likelihood_tests, write_pdf)
    plot_predictor_correlation(predictor_correlation, write_pdf)
    plot_focused_baseline_correlation(predictor_correlation, write_pdf)
    return summary, likelihood_tests


def parse_args() -> argparse.Namespace:
    """Parse command-line options for the extra-forcing sensitivity script."""

    parser = argparse.ArgumentParser(
        description="Sensitivity GLM for extra forcings in Rousseau 2023 binned monsoon-start hazards."
    )
    parser.add_argument("--no-pdf", action="store_true", help="Only write PNG figures.")
    return parser.parse_args()


def main() -> None:
    """Command-line entry point."""

    args = parse_args()
    summary, likelihood_tests = run_analysis(write_pdf=not args.no_pdf)
    best = (
        summary.sort_values(["dataset_id", "AICc"])
        .groupby("dataset_id", as_index=False)
        .first()
    )
    print("Best model by AICc:")
    print(
        best[
            [
                "dataset_id",
                "model_id",
                "n_events",
                "n_parameters",
                "AICc",
                "delta_AICc",
                "pre_phase_preferred_deg",
                "pre_phase_rate_ratio_max_vs_min",
            ]
        ].to_string(index=False)
    )
    print("\nExtra-forcing likelihood-ratio tests:")
    print(
        likelihood_tests[
            likelihood_tests["comparison_family"].isin(
                ["add_to_base", "joint_add_to_base", "drop_one_from_full"]
            )
        ][
            [
                "dataset_id",
                "comparison_id",
                "reduced_model_id",
                "full_model_id",
                "df",
                "LR_statistic",
                "LR_p_value",
                "delta_AICc_full_minus_reduced",
            ]
        ].to_string(index=False)
    )
    print(f"\nWrote tables to: {OUT_DATA_DIR.relative_to(PROJECT_ROOT)}")
    print(f"Wrote figures to: {OUT_FIG_DIR.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    main()
