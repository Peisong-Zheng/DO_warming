"""
Poisson event-hazard models for Rousseau et al. (2023) strong/weak monsoon
starts on 0.2 kyr bins.

This script implements the bin-level alternative to the 20 kyr event-count
models. Each 0.2 kyr bin is treated as a small time interval with an observed
event count:

    Y_i ~ Poisson(lambda_i * dt)

and the main model is:

    log(lambda_i) = beta0 + beta1 LR04_i + beta2 CO2_i
                    + beta3 sin(pre_phase_i) + beta4 cos(pre_phase_i)

The precession phase convention is the same as in
``Orbital_phase_rayleigh.py``:
local minima of the precession index are phase 0 and local maxima are phase pi.

Method sketch
-------------
The analysis is a discrete approximation to an event-time hazard model. The
timeline is cut into short bins. For each bin, the observed response is the
number of monsoon-start events in that bin, and the expected count is the
duration of the bin times an instantaneous event rate:

    mu_i = dt_i * lambda_i.

The GLM is fit on log(lambda_i), not directly on mu_i. This is why the
likelihood below contains log(dt_i) as an offset term: longer bins should have
larger expected counts even if their underlying per-kyr rate is the same.

The phase terms are represented by sin(theta) and cos(theta) instead of theta
itself because phase is circular. A phase of 359 degrees is close to 0 degrees,
not far away from it. The pair of sine/cosine coefficients is later converted
back into a preferred phase and an amplitude.
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
from paper_figure_export import save_paper_pdf
from toolbox.model_stats import likelihood_gain, likelihood_ratio_p_value
import toolbox.event_inputs as event_inputs
import toolbox.poisson as poisson
from toolbox.project_config import (
    ANALYSIS_END_KA,
    ANALYSIS_START_KA,
    BIN_WIDTH_KA,
    CO2_XLSX,
    DATASET_SETTINGS,
    LR04_XLSX,
    PRE_TXT,
    PROJECT_ROOT,
)


RUN_NAME = "Bin_hazard_phase_poisson"
OUT_DATA_DIR = PROJECT_ROOT / "data" / "processed" / RUN_NAME
OUT_FIG_DIR = PROJECT_ROOT / "figures" / RUN_NAME

MODEL_SPECS: list[tuple[str, tuple[str, ...], str]] = [
    ("stationary", (), "Stationary"),
    ("climate_lr04_co2", ("lr04_scaled", "co2_scaled"), "LR04 + CO2"),
    ("pre_phase", ("pre_phase_sin", "pre_phase_cos"), "Precession phase"),
    (
        "climate_lr04_co2_pre_phase",
        ("lr04_scaled", "co2_scaled", "pre_phase_sin", "pre_phase_cos"),
        "LR04 + CO2 + precession phase",
    ),
]

MODEL_COLORS = {
    "stationary": "#777777",
    "climate_lr04_co2": "#1865C3",
    "pre_phase": "#0e470c",
    "climate_lr04_co2_pre_phase": "#F20D0D",
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
    """Save a figure to the run directory and, when requested, paper exports."""

    ensure_dir(OUT_FIG_DIR)
    fig.savefig(OUT_FIG_DIR / f"{stem}.png", dpi=300, bbox_inches="tight")
    if write_pdf:
        fig.savefig(OUT_FIG_DIR / f"{stem}.pdf", bbox_inches="tight")
        save_paper_pdf(fig, PROJECT_ROOT, stem)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Model fitting and likelihood comparisons
# ---------------------------------------------------------------------------


def fit_all_models(binned_inputs: pd.DataFrame) -> list[poisson.FittedPoissonModel]:
    """Fit every candidate model separately for strong and weak starts."""

    models = []
    for _, group in binned_inputs.groupby("dataset_id", sort=False):
        for model_id, terms, label in MODEL_SPECS:
            models.append(poisson.fit_poisson_model(group, model_id, terms, label))
    return models


def build_likelihood_tests(models: list[poisson.FittedPoissonModel]) -> pd.DataFrame:
    """Compare nested models with likelihood-ratio tests.

    The most important comparison is ``phase_after_climate``:

        reduced = LR04 + CO2
        full    = LR04 + CO2 + sin(pre_phase) + cos(pre_phase)

    Its p value asks whether adding the two phase coefficients improves the
    likelihood more than expected from two extra parameters under the nested
    chi-square approximation.
    """

    lookup = poisson.model_lookup(models)
    rows = []
    comparisons = [
        ("climate_vs_stationary", "stationary", "climate_lr04_co2"),
        ("pre_phase_vs_stationary", "stationary", "pre_phase"),
        ("phase_after_climate", "climate_lr04_co2", "climate_lr04_co2_pre_phase"),
        ("full_vs_stationary", "stationary", "climate_lr04_co2_pre_phase"),
    ]
    for dataset_id in DATASET_SETTINGS:
        for comparison_id, reduced_id, full_id in comparisons:
            reduced = lookup[(dataset_id, reduced_id)]
            full = lookup[(dataset_id, full_id)]
            ll_gain = likelihood_gain(full.log_likelihood, reduced.log_likelihood)
            df = len(full.beta) - len(reduced.beta)
            lr_stat, p_value = likelihood_ratio_p_value(ll_gain, df)
            rows.append(
                {
                    "dataset_id": dataset_id,
                    "dataset_label": full.dataset_label,
                    "comparison_id": comparison_id,
                    "reduced_model_id": reduced_id,
                    "full_model_id": full_id,
                    "df": int(df),
                    "loglik_reduced": reduced.log_likelihood,
                    "loglik_full": full.log_likelihood,
                    "LR_statistic": lr_stat,
                    "LR_p_value": p_value,
                    "delta_AICc_full_minus_reduced": full.aicc - reduced.aicc,
                }
            )
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------


def plot_inputs_and_rates(
    binned_inputs: pd.DataFrame,
    fitted_rates: pd.DataFrame,
    likelihood_tests: pd.DataFrame,
    write_pdf: bool,
) -> None:
    """Plot model inputs and fitted climate/phase hazards for both event types."""

    fig, axes = plt.subplots(5, 1, figsize=(13, 10), sharex=True, height_ratios=[1.0, 1.0, 1.0, 1.4, 1.4])
    base = binned_inputs[binned_inputs["dataset_id"].eq("strong_monsoon_start")]
    axes[0].plot(base["bin_center_ka"], base["lr04"], color="#1b9e77", lw=1.0)
    axes[0].set_ylabel("LR04")
    axes[0].invert_yaxis()

    axes[1].plot(base["bin_center_ka"], base["co2"], color="#d95f02", lw=1.0)
    axes[1].set_ylabel("CO2")

    pre_raw = event_inputs.load_precession_series(PRE_TXT, project_root=PROJECT_ROOT)
    pre_raw = pre_raw[
        (pre_raw["age_ka"] >= ANALYSIS_START_KA)
        & (pre_raw["age_ka"] <= ANALYSIS_END_KA)
    ]
    pre_ax = axes[2].twinx()
    axes[2].set_zorder(2)
    pre_ax.set_zorder(1)
    axes[2].patch.set_visible(False)
    pre_ax.plot(
        pre_raw["age_ka"],
        pre_raw["precession_index"],
        color="#5f5f5f",
        lw=0.75,
        alpha=0.38,
        zorder=1,
    )
    pre_ax.axhline(0.0, color="#777777", lw=0.55, alpha=0.28, zorder=0)
    pre_ax.set_ylabel("pre index", color="#5f5f5f")
    pre_ax.tick_params(axis="y", colors="#5f5f5f", labelsize=8)
    pre_ax.spines["right"].set_color("#8a8a8a")

    axes[2].plot(base["bin_center_ka"], base["pre_phase_deg"], color="#7570b3", lw=0.9, zorder=3)
    axes[2].set_ylabel("pre phase", color="#7570b3")
    axes[2].tick_params(axis="y", colors="#7570b3")
    axes[2].set_yticks([0, 90, 180, 270, 360])

    for ax_idx, dataset_id in enumerate(DATASET_SETTINGS, start=3):
        ax = axes[ax_idx]
        settings = DATASET_SETTINGS[dataset_id]
        data = binned_inputs[binned_inputs["dataset_id"].eq(dataset_id)]
        event_bins = data[data["event_count"] > 0]
        ax.vlines(
            event_bins["bin_center_ka"],
            0.0,
            event_bins["event_count"],
            color=settings["color"],
            lw=0.8,
            alpha=0.55,
            label="observed event bins",
        )
        for model_id in ("climate_lr04_co2", "climate_lr04_co2_pre_phase"):
            rates = fitted_rates[
                fitted_rates["dataset_id"].eq(dataset_id)
                & fitted_rates["model_id"].eq(model_id)
            ]
            ax.plot(
                rates["bin_center_ka"],
                rates["lambda_per_kyr"],
                color=MODEL_COLORS[model_id],
                lw=1.2,
                label=rates["model_label"].iloc[0],
            )
        phase_test = likelihood_tests[
            likelihood_tests["dataset_id"].eq(dataset_id)
            & likelihood_tests["comparison_id"].eq("phase_after_climate")
        ].iloc[0]
        ax.text(
            0.01,
            0.94,
            f"{settings['label']}: N={int(data['event_count'].sum())}, "
            f"phase-after-climate p={phase_test['LR_p_value']:.3g}",
            transform=ax.transAxes,
            ha="left",
            va="top",
            bbox={"boxstyle": "round,pad=0.25", "facecolor": "white", "edgecolor": "#bbbbbb", "alpha": 0.85},
        )
        ax.set_ylabel("rate / kyr")
        ax.legend(
            loc="lower right",
            bbox_to_anchor=(1.0, 1.03),
            ncol=3,
            frameon=False,
            borderaxespad=0.0,
        )

    for ax in axes:
        ax.grid(True, color="#e6e6e6", lw=0.6)
        ax.set_xlim(ANALYSIS_END_KA, ANALYSIS_START_KA)
    for idx, ax in enumerate(axes):
        ax.text(
            -0.045,
            1.02,
            chr(ord("a") + idx),
            transform=ax.transAxes,
            ha="right",
            va="bottom",
            fontsize=11,
            fontweight="bold",
            clip_on=False,
        )
    axes[-1].set_xlabel("Age (ka BP)")
    fig.subplots_adjust(left=0.10, right=0.90, top=0.985, bottom=0.075, hspace=0.30)
    save_figure(fig, "fig01_binned_inputs_and_fitted_hazards", write_pdf)


def format_p_value(p_value: float) -> str:
    """Format p values consistently for compact figure annotations."""

    return f"{p_value:.2e}"


def plot_fitted_hazards_only(
    binned_inputs: pd.DataFrame,
    fitted_rates: pd.DataFrame,
    likelihood_tests: pd.DataFrame,
    summary: pd.DataFrame,
    write_pdf: bool,
) -> None:
    """Plot the two fitted hazard panels used as a standalone paper figure."""

    fig, axes = plt.subplots(2, 1, figsize=(13, 5.6), sharex=True)
    dataset_event_counts = {
        dataset_id: int(
            binned_inputs.loc[binned_inputs["dataset_id"].eq(dataset_id), "event_count"].sum()
        )
        for dataset_id in DATASET_SETTINGS
    }

    for ax_idx, dataset_id in enumerate(DATASET_SETTINGS):
        ax = axes[ax_idx]
        settings = DATASET_SETTINGS[dataset_id]
        data = binned_inputs[binned_inputs["dataset_id"].eq(dataset_id)]
        event_bins = data[data["event_count"] > 0]
        ax.vlines(
            event_bins["bin_center_ka"],
            0.0,
            event_bins["event_count"],
            color=settings["color"],
            lw=0.8,
            alpha=0.55,
            label="_nolegend_",
        )
        for model_id in ("climate_lr04_co2", "climate_lr04_co2_pre_phase"):
            rates = fitted_rates[
                fitted_rates["dataset_id"].eq(dataset_id)
                & fitted_rates["model_id"].eq(model_id)
            ]
            ax.plot(
                rates["bin_center_ka"],
                rates["lambda_per_kyr"],
                color=MODEL_COLORS[model_id],
                lw=1.2,
                label="_nolegend_",
            )

        phase_test = likelihood_tests[
            likelihood_tests["dataset_id"].eq(dataset_id)
            & likelihood_tests["comparison_id"].eq("phase_after_climate")
        ].iloc[0]
        ax.text(
            0.01,
            0.94,
            f"Precession phase after LR04+CO$_2$: LR={phase_test['LR_statistic']:.2f}, "
            f"p={format_p_value(phase_test['LR_p_value'])}",
            transform=ax.transAxes,
            ha="left",
            va="top",
            bbox={"boxstyle": "round,pad=0.25", "facecolor": "white", "edgecolor": "#bbbbbb", "alpha": 0.85},
        )
        ax.set_ylabel("rate / kyr")
        ax.grid(True, color="#e6e6e6", lw=0.6)
        ax.set_xlim(ANALYSIS_START_KA, ANALYSIS_END_KA)
        ax.text(
            -0.045,
            1.02,
            chr(ord("a") + ax_idx),
            transform=ax.transAxes,
            ha="right",
            va="bottom",
            fontsize=11,
            fontweight="bold",
            clip_on=False,
        )

    legend_handles = [
        Line2D(
            [0],
            [0],
            color=DATASET_SETTINGS["strong_monsoon_start"]["color"],
            lw=1.2,
            alpha=0.55,
            label="Strong monsoon start",
        ),
        Line2D(
            [0],
            [0],
            color=DATASET_SETTINGS["weak_monsoon_start"]["color"],
            lw=1.2,
            alpha=0.55,
            label="Weak monsoon start",
        ),
        Line2D([0], [0], color=MODEL_COLORS["climate_lr04_co2"], lw=1.2, label="LR04 + CO$_2$"),
        Line2D(
            [0],
            [0],
            color=MODEL_COLORS["climate_lr04_co2_pre_phase"],
            lw=1.2,
            label="LR04 + CO$_2$ + precession phase",
        ),
    ]
    axes[0].legend(
        handles=legend_handles,
        loc="lower center",
        bbox_to_anchor=(0.5, 1.03),
        ncol=2,
        frameon=False,
        fontsize=10.5,
        borderaxespad=0.0,
    )
    # set y lim to [0, 0.6] 
    for ax in axes:
        ax.set_ylim(0, 0.6)
    axes[-1].set_xlabel("Age (kyr BP)")
    fig.subplots_adjust(left=0.10, right=0.90, top=0.94, bottom=0.12, hspace=0.35)
    save_figure(fig, "fig03_fitted_hazards_from_fig01de", write_pdf)


def plot_model_comparison(summary: pd.DataFrame, likelihood_tests: pd.DataFrame, write_pdf: bool) -> None:
    """Plot candidate-model Delta AICc values and phase likelihood tests."""

    fig, axes = plt.subplots(1, 2, figsize=(12.8, 4.8), sharey=True)
    for ax, dataset_id in zip(axes, DATASET_SETTINGS):
        sub = summary[summary["dataset_id"].eq(dataset_id)].sort_values("AICc")
        colors = [MODEL_COLORS.get(mid, "#777777") for mid in sub["model_id"]]
        ax.barh(sub["model_label"], sub["delta_AICc"], color=colors, alpha=0.85)
        ax.invert_yaxis()
        ax.set_xlabel("Delta AICc")
        ax.set_title(DATASET_SETTINGS[dataset_id]["label"], loc="left")
        ax.grid(True, axis="x", color="#e6e6e6", lw=0.6)
        phase_test = likelihood_tests[
            likelihood_tests["dataset_id"].eq(dataset_id)
            & likelihood_tests["comparison_id"].eq("phase_after_climate")
        ].iloc[0]
        ax.text(
            0.98,
            0.95,
            f"LR test phase after LR04+CO2\np={phase_test['LR_p_value']:.3g}",
            transform=ax.transAxes,
            ha="right",
            va="top",
            bbox={"boxstyle": "round,pad=0.25", "facecolor": "white", "edgecolor": "#bbbbbb", "alpha": 0.9},
        )
    fig.suptitle("Poisson hazard model comparison on 0.2 kyr bins", y=0.995, fontsize=14)
    fig.subplots_adjust(top=0.84, wspace=0.35)
    save_figure(fig, "fig02_model_comparison_delta_aicc", write_pdf)


def plot_phase_response(summary: pd.DataFrame, write_pdf: bool) -> None:
    """Plot the fitted rate multiplier over the precession cycle."""

    phase = np.linspace(0.0, 2.0 * np.pi, 361)
    fig, ax = plt.subplots(figsize=(8.8, 4.8))
    for dataset_id, settings in DATASET_SETTINGS.items():
        row = summary[
            summary["dataset_id"].eq(dataset_id)
            & summary["model_id"].eq("climate_lr04_co2_pre_phase")
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
    ax.set_ylabel("Relative rate multiplier\n(climate terms held constant)")
    ax.set_title("Estimated independent precession-phase effect", loc="left")
    ax.grid(True, color="#e6e6e6", lw=0.6)
    ax.legend(frameon=False, loc="upper right")
    save_figure(fig, "fig03_phase_response_after_climate", write_pdf)


def plot_event_phase_histograms(binned_inputs: pd.DataFrame, summary: pd.DataFrame, write_pdf: bool) -> None:
    """Plot observed event phases together with fitted phase-response maxima."""

    fig, axes = plt.subplots(1, 2, figsize=(12.4, 4.8), sharey=True)
    bins = np.linspace(0, 360, 19)
    for ax, dataset_id in zip(axes, DATASET_SETTINGS):
        settings = DATASET_SETTINGS[dataset_id]
        sub = binned_inputs[
            binned_inputs["dataset_id"].eq(dataset_id)
            & (binned_inputs["event_count"] > 0)
        ]
        phases = np.repeat(sub["pre_phase_deg"].to_numpy(dtype=float), sub["event_count"].to_numpy(dtype=int))
        ax.hist(phases, bins=bins, color=settings["color"], alpha=0.65, edgecolor="white")
        row = summary[
            summary["dataset_id"].eq(dataset_id)
            & summary["model_id"].eq("climate_lr04_co2_pre_phase")
        ].iloc[0]
        ax.axvline(row["pre_phase_preferred_deg"], color="#111111", lw=1.2, ls="--", label="model max")
        ax.set_xlim(0, 360)
        ax.set_xticks([0, 90, 180, 270, 360])
        ax.set_xlabel("Precession phase (deg)")
        ax.set_title(settings["label"], loc="left")
        ax.grid(True, axis="y", color="#e6e6e6", lw=0.6)
        ax.legend(frameon=False, loc="upper right")
    axes[0].set_ylabel("Event count")
    fig.suptitle("Observed event phases and fitted phase-response maxima", y=0.995, fontsize=14)
    fig.subplots_adjust(top=0.84, wspace=0.18)
    save_figure(fig, "fig04_event_phase_histograms", write_pdf)


def write_outputs(
    binned_inputs: pd.DataFrame,
    scale_summary: pd.DataFrame,
    phase_extrema: pd.DataFrame,
    models: list[poisson.FittedPoissonModel],
    summary: pd.DataFrame,
    coefficients: pd.DataFrame,
    likelihood_tests: pd.DataFrame,
    fitted_rates: pd.DataFrame,
) -> None:
    """Write analysis tables and run metadata to the processed-data directory."""

    ensure_dir(OUT_DATA_DIR)
    binned_inputs.to_csv(OUT_DATA_DIR / "binned_event_hazard_inputs_0p2kyr.csv", index=False)
    scale_summary.to_csv(OUT_DATA_DIR / "forcing_scale_summary.csv", index=False)
    phase_extrema.to_csv(OUT_DATA_DIR / "precession_phase_extrema.csv", index=False)
    summary.to_csv(OUT_DATA_DIR / "poisson_hazard_model_summary.csv", index=False)
    coefficients.to_csv(OUT_DATA_DIR / "poisson_hazard_coefficients.csv", index=False)
    likelihood_tests.to_csv(OUT_DATA_DIR / "poisson_hazard_likelihood_tests.csv", index=False)
    fitted_rates.to_csv(OUT_DATA_DIR / "poisson_hazard_fitted_rates_by_bin.csv", index=False)

    params = pd.DataFrame(
        [
            {
                "analysis_start_ka": ANALYSIS_START_KA,
                "analysis_end_ka": ANALYSIS_END_KA,
                "bin_width_ka": BIN_WIDTH_KA,
                "response": "event_count_per_0p2kyr_bin",
                "main_model": "log(lambda)=intercept+LR04_scaled+CO2_scaled+sin(pre_phase)+cos(pre_phase)",
                "pre_phase_convention": "precession-index minima=0 rad; maxima=pi rad",
            }
        ]
    )
    params.to_csv(OUT_DATA_DIR / "parameters.csv", index=False)


# ---------------------------------------------------------------------------
# Command-line workflow
# ---------------------------------------------------------------------------


def run_analysis(write_pdf: bool) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Run the full binned-hazard workflow and return the main result tables."""

    ensure_dir(OUT_DATA_DIR)
    ensure_dir(OUT_FIG_DIR)
    events = event_inputs.load_event_catalogues(
        DATASET_SETTINGS,
        analysis_start_ka=ANALYSIS_START_KA,
        analysis_end_ka=ANALYSIS_END_KA,
        project_root=PROJECT_ROOT,
    )
    binned_inputs, scale_summary, phase_extrema = event_inputs.build_binned_inputs(
        events,
        analysis_start_ka=ANALYSIS_START_KA,
        analysis_end_ka=ANALYSIS_END_KA,
        bin_width_ka=BIN_WIDTH_KA,
        lr04_path=LR04_XLSX,
        co2_path=CO2_XLSX,
        precession_path=PRE_TXT,
        project_root=PROJECT_ROOT,
    )
    models = fit_all_models(binned_inputs)
    summary = poisson.build_model_summary(models, binned_inputs, bin_width_ka=BIN_WIDTH_KA)
    coefficients = poisson.build_coefficient_table(models)
    likelihood_tests = build_likelihood_tests(models)
    fitted_rates = poisson.build_fitted_rate_table(models, binned_inputs)
    write_outputs(
        binned_inputs,
        scale_summary,
        phase_extrema,
        models,
        summary,
        coefficients,
        likelihood_tests,
        fitted_rates,
    )
    plot_inputs_and_rates(binned_inputs, fitted_rates, likelihood_tests, write_pdf)
    plot_fitted_hazards_only(binned_inputs, fitted_rates, likelihood_tests, summary, write_pdf)
    plot_model_comparison(summary, likelihood_tests, write_pdf)
    plot_phase_response(summary, write_pdf)
    plot_event_phase_histograms(binned_inputs, summary, write_pdf)
    return summary, likelihood_tests


def parse_args() -> argparse.Namespace:
    """Parse command-line options for this script."""

    parser = argparse.ArgumentParser(
        description="Fit 0.2 kyr bin Poisson hazard models for Rousseau monsoon starts."
    )
    parser.add_argument("--no-pdf", action="store_true", help="Only write PNG figures.")
    return parser.parse_args()


def main() -> None:
    """Command-line entry point."""

    args = parse_args()
    summary, likelihood_tests = run_analysis(write_pdf=not args.no_pdf)
    print(
        summary[
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
    print("\nLikelihood-ratio tests:")
    print(
        likelihood_tests[
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
