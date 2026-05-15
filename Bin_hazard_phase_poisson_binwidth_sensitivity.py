"""
Bin-width sensitivity for the Rousseau et al. (2023) binned Poisson hazard
phase model.

The baseline script uses 0.2 kyr bins. That choice is a compromise: bins should
be short enough that the event hazard is roughly constant within each bin, but
not so short that almost every bin is empty. This script repeats the core GLM
analysis for a range of bin widths and checks whether the main qualitative
conclusions remain stable:

1. whether LR04, CO2, and precession phase improve event-hazard models after
   controlling same-type event history and Cheng composite sampling resolution;
2. whether precession phase still adds information after the adjusted
   LR04 + CO2 model;
3. whether the preferred precession phase is close to the 0.2 kyr result;
4. how sparse the event counts are at each bin width.

The likelihood is the same as in the core binned hazard scripts:

    Y_i ~ Poisson(lambda_i * dt_i)
    log(lambda_i) = beta0 + beta X_i

Only the bin grid changes; the 5 kyr history window and Cheng sampling-
resolution control are recomputed on each candidate grid. The main paper-ready table is
``bin_width_paper_summary.csv``; detailed model and test tables are also
written for auditability.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import Bin_hazard_phase_poisson as base
import Predictive_information_model as predictive


RUN_NAME = "Bin_hazard_phase_poisson_binwidth_sensitivity"
OUT_DATA_DIR = base.PROJECT_ROOT / "data" / "processed" / RUN_NAME
OUT_FIG_DIR = base.PROJECT_ROOT / "figures" / RUN_NAME

DEFAULT_BIN_WIDTHS_KA = (0.2, 0.4, 0.6, 0.8, 1.0)
PHASE_MATCH_TOLERANCE_DEG = 30.0
CORE_MODEL_IDS = (
    "stationary",
    "history_resolution_baseline",
    "baseline_pre_phase",
    "baseline_climate_lr04_co2",
    "baseline_climate_lr04_co2_pre_phase",
)

MODEL_SPECS = predictive.MODEL_SPECS
MODEL_COLORS = predictive.MODEL_COLORS


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
    plt.close(fig)


def circular_signed_difference_deg(angle_deg: pd.Series, reference_deg: pd.Series) -> pd.Series:
    return ((angle_deg - reference_deg + 180.0) % 360.0) - 180.0


def fit_models_for_width(bin_width_ka: float) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Fit all sensitivity models after temporarily changing the bin width.

    The original helper functions read ``base.BIN_WIDTH_KA`` when constructing
    bin edges. Temporarily replacing that module-level value lets the
    sensitivity test reuse exactly the same input construction and likelihood
    code for every grid, then restore the default width before returning.
    """

    original_width = base.BIN_WIDTH_KA
    try:
        base.BIN_WIDTH_KA = float(bin_width_ka)
        binned_inputs, _, phase_extrema, _ = predictive.build_adjusted_inputs(
            predictive.MAIN_HISTORY_WINDOW_KA
        )
        fit_frame = predictive.model_frame(binned_inputs)
        models = predictive.fit_adjusted_models(binned_inputs)
        summary = base.build_model_summary(models, fit_frame)
        likelihood_tests = predictive.build_adjusted_likelihood_tests(
            models,
            fit_frame,
            predictive.MAIN_HISTORY_WINDOW_KA,
        )
        likelihood_tests.insert(0, "bin_width_ka", float(bin_width_ka))
        occupancy = build_occupancy_summary(fit_frame, bin_width_ka)
    finally:
        base.BIN_WIDTH_KA = original_width

    binned_inputs = binned_inputs.copy()
    binned_inputs["bin_width_ka"] = float(bin_width_ka)
    summary["bin_width_ka"] = float(bin_width_ka)
    summary["history_window_ka"] = predictive.MAIN_HISTORY_WINDOW_KA
    phase_extrema = phase_extrema.copy()
    phase_extrema["bin_width_ka"] = float(bin_width_ka)
    return binned_inputs, summary, likelihood_tests, occupancy


def model_lookup(models: list[base.FittedPoissonModel]) -> dict[tuple[str, str], base.FittedPoissonModel]:
    return {(model.dataset_id, model.model_id): model for model in models}


def build_occupancy_summary(binned_inputs: pd.DataFrame, bin_width_ka: float) -> pd.DataFrame:
    rows = []
    for dataset_id, group in binned_inputs.groupby("dataset_id", sort=False):
        counts = group["event_count"].to_numpy(dtype=int)
        n_bins = int(len(group))
        n_events = int(counts.sum())
        nonempty = int(np.count_nonzero(counts > 0))
        multi = int(np.count_nonzero(counts > 1))
        rows.append(
            {
                "bin_width_ka": float(bin_width_ka),
                "dataset_id": dataset_id,
                "dataset_label": group["dataset_label"].iloc[0],
                "n_bins": n_bins,
                "n_events": n_events,
                "mean_events_per_bin": n_events / n_bins,
                "nonempty_bin_count": nonempty,
                "empty_bin_fraction": 1.0 - nonempty / n_bins,
                "multi_event_bin_count": multi,
                "multi_event_bin_fraction": multi / n_bins,
                "max_event_count_per_bin": int(counts.max(initial=0)),
            }
        )
    return pd.DataFrame(rows)


def collect_test_value(
    tests: pd.DataFrame,
    comparison_id: str,
    value_col: str,
    out_col: str,
) -> pd.DataFrame:
    return tests[tests["comparison_id"].eq(comparison_id)][
        ["bin_width_ka", "dataset_id", value_col]
    ].rename(columns={value_col: out_col})


def build_paper_summary(
    model_summary: pd.DataFrame,
    likelihood_tests: pd.DataFrame,
    occupancy: pd.DataFrame,
) -> pd.DataFrame:
    core = model_summary[model_summary["model_id"].isin(CORE_MODEL_IDS)].copy()
    core["delta_AICc_core"] = core["AICc"] - core.groupby(["dataset_id", "bin_width_ka"])["AICc"].transform("min")
    core["rank_AICc_core"] = core.groupby(["dataset_id", "bin_width_ka"])["AICc"].rank(method="first")
    full = core[core["model_id"].eq("baseline_climate_lr04_co2_pre_phase")].copy()
    full = full[
        [
            "bin_width_ka",
            "dataset_id",
            "dataset_label",
            "rank_AICc_core",
            "delta_AICc_core",
            "AICc",
            "pre_phase_preferred_deg",
            "pre_phase_rate_ratio_max_vs_min",
        ]
    ].rename(
        columns={
            "rank_AICc_core": "full_model_core_rank_AICc",
            "delta_AICc_core": "full_model_core_delta_AICc",
            "AICc": "full_model_AICc",
        }
    )

    best = (
        core.sort_values(["dataset_id", "bin_width_ka", "AICc"])
        .groupby(["dataset_id", "bin_width_ka"], as_index=False)
        .first()[["dataset_id", "bin_width_ka", "model_id", "model_label"]]
        .rename(columns={"model_id": "best_core_model_id", "model_label": "best_core_model_label"})
    )
    out = occupancy.merge(full, on=["bin_width_ka", "dataset_id", "dataset_label"], how="left")
    out = out.merge(best, on=["bin_width_ka", "dataset_id"], how="left")

    for comparison_id, prefix in [
        ("lr04_after_baseline", "lr04_after_baseline"),
        ("co2_after_baseline", "co2_after_baseline"),
        ("pre_phase_after_baseline", "pre_phase_after_baseline"),
        ("climate_lr04_co2_after_baseline", "climate_after_baseline"),
        ("phase_after_adjusted_climate", "phase_after_adjusted_climate"),
        ("lr04_after_adjusted_co2_phase", "lr04_unique"),
        ("co2_after_adjusted_lr04_phase", "co2_unique"),
    ]:
        out = out.merge(
            collect_test_value(likelihood_tests, comparison_id, "LR_p_value", f"{prefix}_p"),
            on=["bin_width_ka", "dataset_id"],
            how="left",
        )
        out = out.merge(
            collect_test_value(
                likelihood_tests,
                comparison_id,
                "delta_AICc_full_minus_reduced",
                f"{prefix}_delta_AICc_full_minus_reduced",
            ),
            on=["bin_width_ka", "dataset_id"],
            how="left",
        )

    ref = out[out["bin_width_ka"].eq(0.2)][
        ["dataset_id", "pre_phase_preferred_deg"]
    ].rename(columns={"pre_phase_preferred_deg": "pre_phase_preferred_deg_0p2_ref"})
    out = out.merge(ref, on="dataset_id", how="left")
    out["preferred_phase_shift_vs_0p2_deg"] = circular_signed_difference_deg(
        out["pre_phase_preferred_deg"],
        out["pre_phase_preferred_deg_0p2_ref"],
    )
    # Backward-compatible aliases keep the plotting code readable while the
    # output names make clear that the reduced model includes history,
    # resolution, LR04, and CO2.
    out["phase_after_climate_p"] = out["phase_after_adjusted_climate_p"]
    out["phase_after_climate_delta_AICc_full_minus_reduced"] = out[
        "phase_after_adjusted_climate_delta_AICc_full_minus_reduced"
    ]
    out["phase_after_climate_robust_p05"] = out["phase_after_adjusted_climate_p"] < 0.05
    out["phase_after_climate_AICc_support"] = (
        out["phase_after_adjusted_climate_delta_AICc_full_minus_reduced"] < 0.0
    )
    out["preferred_phase_within_30deg_of_0p2"] = (
        out["preferred_phase_shift_vs_0p2_deg"].abs() <= PHASE_MATCH_TOLERANCE_DEG
    )
    out["qualitatively_matches_0p2"] = (
        out["phase_after_climate_robust_p05"]
        & out["phase_after_climate_AICc_support"]
        & out["preferred_phase_within_30deg_of_0p2"]
    )

    preferred_cols = [
        "dataset_id",
        "dataset_label",
        "bin_width_ka",
        "n_bins",
        "n_events",
        "empty_bin_fraction",
        "multi_event_bin_count",
        "max_event_count_per_bin",
        "best_core_model_id",
        "full_model_core_rank_AICc",
        "full_model_core_delta_AICc",
        "lr04_after_baseline_p",
        "co2_after_baseline_p",
        "pre_phase_after_baseline_p",
        "phase_after_climate_p",
        "phase_after_climate_delta_AICc_full_minus_reduced",
        "pre_phase_preferred_deg",
        "preferred_phase_shift_vs_0p2_deg",
        "pre_phase_rate_ratio_max_vs_min",
        "qualitatively_matches_0p2",
    ]
    return out[preferred_cols].sort_values(["dataset_id", "bin_width_ka"]).reset_index(drop=True)


def format_p_value(value: float) -> str:
    if not np.isfinite(value):
        return ""
    if value < 0.001:
        return f"{value:.1e}"
    return f"{value:.3f}"


def build_paper_summary_formatted(paper: pd.DataFrame) -> pd.DataFrame:
    out = paper.copy()
    out["empty bins (%)"] = (100.0 * out["empty_bin_fraction"]).round(1)
    out["full Delta AICc among core models"] = out["full_model_core_delta_AICc"].round(2)
    out["phase-after-climate Delta AICc"] = out[
        "phase_after_climate_delta_AICc_full_minus_reduced"
    ].round(2)
    out["preferred phase (deg)"] = out["pre_phase_preferred_deg"].round(1)
    out["phase shift vs 0.2 kyr (deg)"] = out["preferred_phase_shift_vs_0p2_deg"].round(1)
    out["phase rate ratio"] = out["pre_phase_rate_ratio_max_vs_min"].round(2)
    for src, dst in [
        ("lr04_after_baseline_p", "LR04 p"),
        ("co2_after_baseline_p", "CO2 p"),
        ("pre_phase_after_baseline_p", "pre phase p"),
        ("phase_after_climate_p", "phase after adjusted climate p"),
    ]:
        out[dst] = out[src].map(format_p_value)
    out["matches 0.2 kyr"] = np.where(out["qualitatively_matches_0p2"], "yes", "no")
    return out[
        [
            "dataset_label",
            "bin_width_ka",
            "n_bins",
            "empty bins (%)",
            "multi_event_bin_count",
            "max_event_count_per_bin",
            "best_core_model_id",
            "full Delta AICc among core models",
            "LR04 p",
            "CO2 p",
            "pre phase p",
            "phase after adjusted climate p",
            "phase-after-climate Delta AICc",
            "preferred phase (deg)",
            "phase shift vs 0.2 kyr (deg)",
            "phase rate ratio",
            "matches 0.2 kyr",
        ]
    ].rename(
        columns={
            "dataset_label": "Dataset",
            "bin_width_ka": "Bin width (kyr)",
            "n_bins": "Bins",
            "multi_event_bin_count": "Multi-event bins",
            "max_event_count_per_bin": "Max events/bin",
            "best_core_model_id": "Best core model",
        }
    )


def plot_phase_stability(paper: pd.DataFrame, write_pdf: bool) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(12.5, 7.8), sharex=True)
    for dataset_id, settings in base.DATASET_SETTINGS.items():
        sub = paper[paper["dataset_id"].eq(dataset_id)].sort_values("bin_width_ka")
        color = settings["color"]
        label = settings["label"]
        axes[0, 0].plot(sub["bin_width_ka"], sub["phase_after_climate_p"], marker="o", color=color, label=label)
        axes[0, 1].plot(
            sub["bin_width_ka"],
            sub["phase_after_climate_delta_AICc_full_minus_reduced"],
            marker="o",
            color=color,
            label=label,
        )
        axes[1, 0].plot(sub["bin_width_ka"], sub["pre_phase_preferred_deg"], marker="o", color=color, label=label)
        axes[1, 1].plot(sub["bin_width_ka"], sub["empty_bin_fraction"] * 100.0, marker="o", color=color, label=label)

    axes[0, 0].axhline(0.05, color="#555555", lw=0.8, ls="--")
    axes[0, 0].set_yscale("log")
    axes[0, 0].set_ylabel("LR p value")
    axes[0, 0].set_title("Precession phase after adjusted LR04 + CO2", loc="left")

    axes[0, 1].axhline(0.0, color="#555555", lw=0.8, ls="--")
    axes[0, 1].set_ylabel("Delta AICc full - adjusted climate")
    axes[0, 1].set_title("AICc support for adding phase", loc="left")

    axes[1, 0].set_ylim(0, 360)
    axes[1, 0].set_yticks([0, 90, 180, 270, 360])
    axes[1, 0].set_ylabel("Preferred phase (deg)")
    axes[1, 0].set_title("Preferred precession phase", loc="left")

    axes[1, 1].set_ylabel("Empty bins (%)")
    axes[1, 1].set_title("Event-count sparsity", loc="left")

    for ax in axes.ravel():
        ax.grid(True, color="#e6e6e6", lw=0.6)
        ax.set_xlabel("Bin width (kyr)")
    axes[0, 0].legend(frameon=False, loc="best")
    fig.subplots_adjust(hspace=0.32, wspace=0.25)
    save_figure(fig, "fig01_phase_and_sparsity_stability", write_pdf)


def plot_driver_pvalues(likelihood_tests: pd.DataFrame, write_pdf: bool) -> None:
    comparisons = [
        ("lr04_after_baseline", "LR04 | baseline"),
        ("co2_after_baseline", "CO2 | baseline"),
        ("pre_phase_after_baseline", "pre phase | baseline"),
        ("phase_after_adjusted_climate", "pre phase | adjusted LR04+CO2"),
    ]
    fig, axes = plt.subplots(1, 2, figsize=(12.8, 4.8), sharey=True)
    for ax, dataset_id in zip(axes, base.DATASET_SETTINGS):
        sub = likelihood_tests[likelihood_tests["dataset_id"].eq(dataset_id)]
        for comparison_id, label in comparisons:
            g = sub[sub["comparison_id"].eq(comparison_id)].sort_values("bin_width_ka")
            ax.plot(g["bin_width_ka"], -np.log10(np.maximum(g["LR_p_value"], 1e-300)), marker="o", lw=1.5, label=label)
        ax.axhline(-np.log10(0.05), color="#555555", lw=0.8, ls="--", label="p=0.05")
        ax.set_title(base.DATASET_SETTINGS[dataset_id]["label"], loc="left")
        ax.set_xlabel("Bin width (kyr)")
        ax.grid(True, color="#e6e6e6", lw=0.6)
    axes[0].set_ylabel("-log10(LR p value)")
    axes[1].legend(frameon=False, loc="upper right")
    fig.subplots_adjust(wspace=0.18)
    save_figure(fig, "fig02_driver_lr_pvalues_by_bin_width", write_pdf)


def plot_model_delta_aicc(model_summary: pd.DataFrame, write_pdf: bool) -> None:
    model_ids = [
        "stationary",
        "history_resolution_baseline",
        "baseline_lr04",
        "baseline_co2",
        "baseline_pre_phase",
        "baseline_climate_lr04_co2",
        "baseline_climate_lr04_co2_pre_phase",
    ]
    fig, axes = plt.subplots(1, 2, figsize=(12.8, 4.8), sharey=True)
    for ax, dataset_id in zip(axes, base.DATASET_SETTINGS):
        sub = model_summary[model_summary["dataset_id"].eq(dataset_id)]
        for model_id in model_ids:
            g = sub[sub["model_id"].eq(model_id)].sort_values("bin_width_ka")
            if g.empty:
                continue
            ax.plot(
                g["bin_width_ka"],
                g["delta_AICc"],
                marker="o",
                lw=1.4,
                color=MODEL_COLORS.get(model_id, "#555555"),
                label=g["model_label"].iloc[0],
            )
        ax.axhline(0.0, color="#555555", lw=0.8, ls=":")
        ax.set_title(base.DATASET_SETTINGS[dataset_id]["label"], loc="left")
        ax.set_xlabel("Bin width (kyr)")
        ax.grid(True, color="#e6e6e6", lw=0.6)
    axes[0].set_ylabel("Delta AICc within bin width")
    axes[1].legend(frameon=False, loc="upper right")
    fig.subplots_adjust(wspace=0.18)
    save_figure(fig, "fig03_model_delta_aicc_by_bin_width", write_pdf)


def write_outputs(
    binned_inputs: pd.DataFrame,
    model_summary: pd.DataFrame,
    likelihood_tests: pd.DataFrame,
    occupancy: pd.DataFrame,
    paper_summary: pd.DataFrame,
    paper_summary_formatted: pd.DataFrame,
    bin_widths: tuple[float, ...],
) -> None:
    ensure_dir(OUT_DATA_DIR)
    binned_inputs.to_csv(OUT_DATA_DIR / "bin_width_binned_inputs.csv", index=False)
    model_summary.to_csv(OUT_DATA_DIR / "bin_width_model_summary.csv", index=False)
    likelihood_tests.to_csv(OUT_DATA_DIR / "bin_width_likelihood_tests.csv", index=False)
    occupancy.to_csv(OUT_DATA_DIR / "bin_width_event_count_occupancy.csv", index=False)
    paper_summary.to_csv(OUT_DATA_DIR / "bin_width_paper_summary.csv", index=False)
    paper_summary_formatted.to_csv(OUT_DATA_DIR / "bin_width_paper_summary_formatted.csv", index=False)
    pd.DataFrame(
        [
            {
                "bin_widths_ka": " ".join(f"{width:g}" for width in bin_widths),
                "phase_match_tolerance_deg": PHASE_MATCH_TOLERANCE_DEG,
                "history_window_ka": predictive.MAIN_HISTORY_WINDOW_KA,
                "baseline_model_id": "history_resolution_baseline",
                "main_model_id": "baseline_climate_lr04_co2_pre_phase",
                "phase_test": "phase_after_adjusted_climate",
                "phase_convention": "precession-index minima=0 rad; maxima=pi rad",
                "note": "qualitatively_matches_0p2 requires adjusted phase p<0.05, Delta AICc<0, and preferred phase within tolerance of the 0.2 kyr result",
            }
        ]
    ).to_csv(OUT_DATA_DIR / "parameters.csv", index=False)


def run_analysis(bin_widths: tuple[float, ...], write_pdf: bool) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    ensure_dir(OUT_DATA_DIR)
    ensure_dir(OUT_FIG_DIR)

    binned_tables = []
    summary_tables = []
    test_tables = []
    occupancy_tables = []
    for width in bin_widths:
        binned, summary, tests, occupancy = fit_models_for_width(width)
        binned_tables.append(binned)
        summary_tables.append(summary)
        test_tables.append(tests)
        occupancy_tables.append(occupancy)

    binned_inputs = pd.concat(binned_tables, ignore_index=True)
    model_summary = pd.concat(summary_tables, ignore_index=True)
    likelihood_tests = pd.concat(test_tables, ignore_index=True)
    occupancy = pd.concat(occupancy_tables, ignore_index=True)
    paper_summary = build_paper_summary(model_summary, likelihood_tests, occupancy)
    paper_summary_formatted = build_paper_summary_formatted(paper_summary)

    write_outputs(
        binned_inputs,
        model_summary,
        likelihood_tests,
        occupancy,
        paper_summary,
        paper_summary_formatted,
        bin_widths,
    )
    plot_phase_stability(paper_summary, write_pdf)
    plot_driver_pvalues(likelihood_tests, write_pdf)
    plot_model_delta_aicc(model_summary, write_pdf)
    return paper_summary, paper_summary_formatted, likelihood_tests


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Bin-width sensitivity for Rousseau monsoon binned Poisson hazard models."
    )
    parser.add_argument(
        "--bin-widths",
        type=float,
        nargs="+",
        default=DEFAULT_BIN_WIDTHS_KA,
        help="Bin widths in kyr to scan. Default: 0.2 0.4 0.6 0.8 1.0",
    )
    parser.add_argument("--no-pdf", action="store_true", help="Only write PNG figures.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    bin_widths = tuple(float(width) for width in args.bin_widths)
    paper_summary, paper_summary_formatted, _ = run_analysis(
        bin_widths=bin_widths,
        write_pdf=not args.no_pdf,
    )
    print("Paper-oriented bin-width sensitivity table:")
    print(paper_summary_formatted.to_string(index=False))
    print("\nKey robustness flags:")
    print(
        paper_summary[
            [
                "dataset_id",
                "bin_width_ka",
                "phase_after_climate_p",
                "phase_after_climate_delta_AICc_full_minus_reduced",
                "pre_phase_preferred_deg",
                "preferred_phase_shift_vs_0p2_deg",
                "qualitatively_matches_0p2",
            ]
        ].to_string(index=False)
    )
    print(f"\nWrote tables to: {OUT_DATA_DIR.relative_to(base.PROJECT_ROOT)}")
    print(f"Wrote figures to: {OUT_FIG_DIR.relative_to(base.PROJECT_ROOT)}")


if __name__ == "__main__":
    main()
