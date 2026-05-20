"""
Sensitivity of Rousseau et al. (2023) event-timing tests to KS window.

This script compares the complete 0.4-4 kyr KS transition catalogue against
the 0.6-4 kyr catalogue for:

1. Rayleigh tests for orbital phase preference.
2. Event-process-baseline predictive models with
   LR04, CO2, and precession phase controls.

The implementation reuses the core definitions from the main analysis scripts
while avoiding their figure-writing wrappers.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

import Orbital_phase_rayleigh as rayleigh
import toolbox.event_inputs as event_inputs
import toolbox.poisson as poisson
from toolbox.project_config import (
    ANALYSIS_END_KA,
    ANALYSIS_START_KA,
    BIN_WIDTH_KA,
    CO2_XLSX,
    LR04_XLSX,
    PRE_TXT,
    PROJECT_ROOT,
)
import Predictive_information_model as predictive


RUN_NAME = "KS_window_core_experiment_sensitivity"
OUT_DATA_DIR = PROJECT_ROOT / "data" / "processed" / RUN_NAME
OUT_FIG_DIR = PROJECT_ROOT / "figures" / RUN_NAME

KS_WINDOW_CATALOGUES = {
    "ks_0p4_4kyr": {
        "ks_window_kyr": "0.4-4",
        "label": "KS 0.4-4 kyr",
        "weak_monsoon_start": PROJECT_ROOT
        / "data/raw/rousseau_2023_ks_0p4_4kyr_weak_monsoon_start_times.csv",
        "strong_monsoon_start": PROJECT_ROOT
        / "data/raw/rousseau_2023_ks_0p4_4kyr_strong_monsoon_start_times.csv",
    },
    "ks_0p6_4kyr": {
        "ks_window_kyr": "0.6-4",
        "label": "KS 0.6-4 kyr",
        "weak_monsoon_start": PROJECT_ROOT
        / "data/raw/rousseau_2023_ks_0p6_4kyr_weak_monsoon_start_times.csv",
        "strong_monsoon_start": PROJECT_ROOT
        / "data/raw/rousseau_2023_ks_0p6_4kyr_strong_monsoon_start_times.csv",
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

POISSON_MODEL_SPECS = predictive.MODEL_SPECS


@dataclass
class EventCatalogue:
    ks_window_id: str
    ks_window_label: str
    ks_window_kyr: str
    event_type: str
    event_label: str
    source_path: Path
    events: pd.DataFrame
    ages_ka: np.ndarray


# ---------------------------------------------------------------------------
# Input preparation and model fitting
# ---------------------------------------------------------------------------


def ensure_dir(path: Path) -> None:
    """Create an output directory if it is missing."""

    path.mkdir(parents=True, exist_ok=True)


def relative_path(path: Path) -> str:
    """Return a project-relative path for metadata tables."""

    return str(path.relative_to(PROJECT_ROOT))


def unique_sorted(values: np.ndarray, decimals: int = 6) -> np.ndarray:
    """Return finite sorted unique values after light decimal rounding."""

    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    values = np.round(values, decimals=decimals)
    return np.unique(np.sort(values))


def load_event_catalogue(ks_window_id: str, event_type: str) -> EventCatalogue:
    """Load one event catalogue for a specific KS detection window."""

    window = KS_WINDOW_CATALOGUES[ks_window_id]
    settings = EVENT_SETTINGS[event_type]
    path = window[event_type]
    df = pd.read_csv(path, encoding="utf-8-sig")
    if "start_time_ka_BP" not in df.columns:
        raise ValueError(f"{path} must contain start_time_ka_BP.")

    events = df.copy()
    events["event_age_ka"] = pd.to_numeric(events["start_time_ka_BP"], errors="coerce")
    events = events.dropna(subset=["event_age_ka"])
    events = events[
        (events["event_age_ka"] >= ANALYSIS_START_KA)
        & (events["event_age_ka"] <= ANALYSIS_END_KA)
    ].copy()
    events = events.sort_values("event_age_ka").reset_index(drop=True)
    events["event_index"] = np.arange(1, len(events) + 1)
    events["event_type"] = event_type
    events["event_label"] = settings["label"]
    events["ks_window_id"] = ks_window_id
    events["ks_window_label"] = window["label"]
    events["ks_window_kyr"] = window["ks_window_kyr"]
    events["source"] = relative_path(path)

    ages = unique_sorted(events["event_age_ka"].to_numpy(dtype=float))
    return EventCatalogue(
        ks_window_id=ks_window_id,
        ks_window_label=window["label"],
        ks_window_kyr=window["ks_window_kyr"],
        event_type=event_type,
        event_label=settings["label"],
        source_path=path,
        events=events,
        ages_ka=ages,
    )


def load_catalogues_for_window(ks_window_id: str) -> list[EventCatalogue]:
    """Load strong and weak catalogues for one KS-window setting."""

    return [
        load_event_catalogue(ks_window_id, "strong_monsoon_start"),
        load_event_catalogue(ks_window_id, "weak_monsoon_start"),
    ]


def build_events_used_table(catalogues: list[EventCatalogue]) -> pd.DataFrame:
    """Concatenate event rows used in the KS-window sensitivity experiment."""

    rows = []
    for catalogue in catalogues:
        rows.append(catalogue.events)
    return pd.concat(rows, ignore_index=True)


def rayleigh_events_frame(catalogues: list[EventCatalogue]) -> pd.DataFrame:
    """Return the event table layout expected by the Rayleigh helpers."""

    return pd.concat([catalogue.events for catalogue in catalogues], ignore_index=True)


def run_rayleigh_for_all(catalogues_by_window: dict[str, list[EventCatalogue]]) -> pd.DataFrame:
    """Run Rayleigh phase tests for every KS-window catalogue."""

    phase_products = {
        driver: rayleigh.build_phase_series(driver, settings)
        for driver, settings in rayleigh.DRIVER_SETTINGS.items()
    }
    rows = []
    for ks_window_id, catalogues in catalogues_by_window.items():
        events = rayleigh_events_frame(catalogues)
        event_phases = rayleigh.sample_event_phases(events, phase_products)
        result = rayleigh.build_rayleigh_results(event_phases)
        window = KS_WINDOW_CATALOGUES[ks_window_id]
        result.insert(0, "ks_window_id", ks_window_id)
        result.insert(1, "ks_window_label", window["label"])
        result.insert(2, "ks_window_kyr", window["ks_window_kyr"])
        result["reject_rayleigh_at_0p05"] = result["rayleigh_p"] < 0.05
        rows.append(result)
    return pd.concat(rows, ignore_index=True)


def hazard_events_for_window(catalogues: list[EventCatalogue]) -> list[event_inputs.EventDataset]:
    """Convert loaded catalogues into the EventDataset format used by models."""

    datasets = []
    for catalogue in catalogues:
        settings = EVENT_SETTINGS[catalogue.event_type]
        datasets.append(
            event_inputs.EventDataset(
                dataset_id=catalogue.event_type,
                label=settings["label"],
                color=settings["color"],
                ages_ka=catalogue.ages_ka,
                source=relative_path(catalogue.source_path),
            )
        )
    return datasets


def fit_sensitivity_poisson_models(
    binned_inputs: pd.DataFrame,
) -> list[poisson.FittedPoissonModel]:
    """Fit the predictive model set for one KS catalogue.

    The KS window changes which transition ages enter the event histogram. The
    model frame is otherwise identical to the main predictive analysis: bins with
    incomplete same-type history are removed before fitting.
    """

    models: list[poisson.FittedPoissonModel] = []
    fit_frame = predictive.model_frame(binned_inputs)
    for _, group in fit_frame.groupby("dataset_id", sort=False):
        for model_id, terms, label in POISSON_MODEL_SPECS:
            models.append(poisson.fit_poisson_model(group, model_id, terms, label))
    return models


def build_sensitivity_likelihood_tests(
    models: list[poisson.FittedPoissonModel],
    fit_frame: pd.DataFrame,
) -> pd.DataFrame:
    """Use the main predictive-model comparisons for the KS sensitivity."""

    return predictive.build_predictive_likelihood_tests(
        models,
        fit_frame,
        predictive.MAIN_HISTORY_WINDOW_KA,
    )


def add_window_columns(frame: pd.DataFrame, ks_window_id: str) -> pd.DataFrame:
    """Attach KS-window identifiers and labels to a result table."""

    window = KS_WINDOW_CATALOGUES[ks_window_id]
    out = frame.copy()
    out.insert(0, "ks_window_id", ks_window_id)
    out.insert(1, "ks_window_label", window["label"])
    out.insert(2, "ks_window_kyr", window["ks_window_kyr"])
    return out


def run_poisson_for_all(
    catalogues_by_window: dict[str, list[EventCatalogue]],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Run the predictive-information workflow for each KS window.

    For each catalogue we rebuild the event histogram, append the Cheng
    sampling-resolution control and same-type history term, then reuse the
    central predictive likelihood-test definitions. This keeps the KS experiment
    focused on catalogue construction rather than on changing the hazard model.
    """

    binned_frames = []
    summary_frames = []
    coefficient_frames = []
    likelihood_frames = []

    for ks_window_id, catalogues in catalogues_by_window.items():
        events = hazard_events_for_window(catalogues)
        binned_inputs, _, _ = event_inputs.build_binned_inputs(
            events,
            analysis_start_ka=ANALYSIS_START_KA,
            analysis_end_ka=ANALYSIS_END_KA,
            bin_width_ka=BIN_WIDTH_KA,
            lr04_path=LR04_XLSX,
            co2_path=CO2_XLSX,
            precession_path=PRE_TXT,
            project_root=PROJECT_ROOT,
        )
        binned_inputs, _, _ = predictive.add_resolution_control(binned_inputs)
        binned_inputs = predictive.add_same_type_history(
            binned_inputs,
            predictive.MAIN_HISTORY_WINDOW_KA,
        )
        fit_frame = predictive.model_frame(binned_inputs)
        models = fit_sensitivity_poisson_models(binned_inputs)
        summary = poisson.build_model_summary(models, fit_frame)
        coefficients = poisson.build_coefficient_table(models)
        likelihood_tests = build_sensitivity_likelihood_tests(models, fit_frame)

        binned_frames.append(add_window_columns(fit_frame, ks_window_id))
        summary_frames.append(add_window_columns(summary, ks_window_id))
        coefficient_frames.append(add_window_columns(coefficients, ks_window_id))
        likelihood_frames.append(add_window_columns(likelihood_tests, ks_window_id))

    return (
        pd.concat(binned_frames, ignore_index=True),
        pd.concat(summary_frames, ignore_index=True),
        pd.concat(coefficient_frames, ignore_index=True),
        pd.concat(likelihood_frames, ignore_index=True),
    )


def consistency_row(
    *,
    experiment: str,
    metric_id: str,
    event_type: str,
    p_04: float,
    p_06: float,
    extra_04: dict | None = None,
    extra_06: dict | None = None,
) -> dict:
    """Build one row comparing p-value decisions between two KS windows."""

    sig_04 = bool(np.isfinite(p_04) and p_04 < 0.05)
    sig_06 = bool(np.isfinite(p_06) and p_06 < 0.05)
    row = {
        "experiment": experiment,
        "metric_id": metric_id,
        "event_type": event_type,
        "ks_0p4_4kyr_p_value": p_04,
        "ks_0p6_4kyr_p_value": p_06,
        "ks_0p4_4kyr_significant_at_0p05": sig_04,
        "ks_0p6_4kyr_significant_at_0p05": sig_06,
        "consistent_significance_decision": sig_04 == sig_06,
    }
    if extra_04:
        for key, value in extra_04.items():
            row[f"ks_0p4_4kyr_{key}"] = value
    if extra_06:
        for key, value in extra_06.items():
            row[f"ks_0p6_4kyr_{key}"] = value
    return row


def build_consistency_summary(
    rayleigh_results: pd.DataFrame,
    poisson_lrt: pd.DataFrame,
    poisson_summary: pd.DataFrame,
) -> pd.DataFrame:
    """Summarize whether core conclusions are stable across KS windows."""

    rows = []
    for event_type in EVENT_SETTINGS:
        pre04 = rayleigh_results[
            rayleigh_results["ks_window_id"].eq("ks_0p4_4kyr")
            & rayleigh_results["event_type"].eq(event_type)
            & rayleigh_results["driver"].eq("pre")
        ].iloc[0]
        pre06 = rayleigh_results[
            rayleigh_results["ks_window_id"].eq("ks_0p6_4kyr")
            & rayleigh_results["event_type"].eq(event_type)
            & rayleigh_results["driver"].eq("pre")
        ].iloc[0]
        rows.append(
            consistency_row(
                experiment="rayleigh_phase",
                metric_id="precession_phase_uniformity",
                event_type=event_type,
                p_04=float(pre04["rayleigh_p"]),
                p_06=float(pre06["rayleigh_p"]),
                extra_04={
                    "n_events": int(pre04["n_phase_events_used"]),
                    "mean_phase_deg": float(pre04["mean_phase_deg"]),
                    "mean_resultant_length": float(pre04["mean_resultant_length"]),
                },
                extra_06={
                    "n_events": int(pre06["n_phase_events_used"]),
                    "mean_phase_deg": float(pre06["mean_phase_deg"]),
                    "mean_resultant_length": float(pre06["mean_resultant_length"]),
                },
            )
        )

        for comparison_id in [
            "climate_lr04_co2_after_baseline",
            "phase_after_climate_state",
            "lr04_after_co2_phase",
            "co2_after_lr04_phase",
            "full_vs_baseline",
        ]:
            p04 = poisson_lrt[
                poisson_lrt["ks_window_id"].eq("ks_0p4_4kyr")
                & poisson_lrt["dataset_id"].eq(event_type)
                & poisson_lrt["comparison_id"].eq(comparison_id)
            ].iloc[0]
            p06 = poisson_lrt[
                poisson_lrt["ks_window_id"].eq("ks_0p6_4kyr")
                & poisson_lrt["dataset_id"].eq(event_type)
                & poisson_lrt["comparison_id"].eq(comparison_id)
            ].iloc[0]
            rows.append(
                consistency_row(
                    experiment="predictive_model",
                    metric_id=comparison_id,
                    event_type=event_type,
                    p_04=float(p04["LR_p_value"]),
                    p_06=float(p06["LR_p_value"]),
                    extra_04={"LR_statistic": float(p04["LR_statistic"])},
                    extra_06={"LR_statistic": float(p06["LR_statistic"])},
                )
            )

        full04 = poisson_summary[
            poisson_summary["ks_window_id"].eq("ks_0p4_4kyr")
            & poisson_summary["dataset_id"].eq(event_type)
            & poisson_summary["model_id"].eq("baseline_climate_lr04_co2_pre_phase")
        ].iloc[0]
        full06 = poisson_summary[
            poisson_summary["ks_window_id"].eq("ks_0p6_4kyr")
            & poisson_summary["dataset_id"].eq(event_type)
            & poisson_summary["model_id"].eq("baseline_climate_lr04_co2_pre_phase")
        ].iloc[0]
        rows.append(
            {
                "experiment": "predictive_model",
                "metric_id": "full_model_preferred_phase_and_rate_ratio",
                "event_type": event_type,
                "ks_0p4_4kyr_p_value": np.nan,
                "ks_0p6_4kyr_p_value": np.nan,
                "ks_0p4_4kyr_significant_at_0p05": np.nan,
                "ks_0p6_4kyr_significant_at_0p05": np.nan,
                "consistent_significance_decision": np.nan,
                "ks_0p4_4kyr_pre_phase_preferred_deg": float(
                    full04["pre_phase_preferred_deg"]
                ),
                "ks_0p6_4kyr_pre_phase_preferred_deg": float(
                    full06["pre_phase_preferred_deg"]
                ),
                "ks_0p4_4kyr_pre_phase_rate_ratio_max_vs_min": float(
                    full04["pre_phase_rate_ratio_max_vs_min"]
                ),
                "ks_0p6_4kyr_pre_phase_rate_ratio_max_vs_min": float(
                    full06["pre_phase_rate_ratio_max_vs_min"]
                ),
            }
        )

    return pd.DataFrame(rows)


def write_outputs(
    *,
    events_used: pd.DataFrame,
    rayleigh_results: pd.DataFrame,
    binned_inputs: pd.DataFrame,
    poisson_summary: pd.DataFrame,
    poisson_coefficients: pd.DataFrame,
    poisson_likelihood_tests: pd.DataFrame,
    consistency: pd.DataFrame,
) -> None:
    """Write KS-window sensitivity outputs."""

    ensure_dir(OUT_DATA_DIR)
    events_used.to_csv(OUT_DATA_DIR / "events_used_by_ks_window.csv", index=False)
    rayleigh_results.to_csv(OUT_DATA_DIR / "rayleigh_phase_results_by_ks_window.csv", index=False)
    binned_inputs.to_csv(OUT_DATA_DIR / "poisson_binned_inputs_by_ks_window.csv", index=False)
    poisson_summary.to_csv(OUT_DATA_DIR / "poisson_model_summary_by_ks_window.csv", index=False)
    poisson_coefficients.to_csv(
        OUT_DATA_DIR / "poisson_coefficients_by_ks_window.csv",
        index=False,
    )
    poisson_likelihood_tests.to_csv(
        OUT_DATA_DIR / "poisson_likelihood_tests_by_ks_window.csv",
        index=False,
    )
    consistency.to_csv(OUT_DATA_DIR / "core_experiment_consistency_summary.csv", index=False)


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------


def format_p_value(value: float) -> str:
    """Format p values for table figures."""

    if not np.isfinite(value):
        return ""
    if value < 1e-4:
        return f"{value:.1e}"
    if value < 0.01:
        return f"{value:.4f}"
    return f"{value:.3f}"


def significance_label(value: float) -> str:
    """Return a compact significance label for p-value table cells."""

    if not np.isfinite(value):
        return ""
    return "sig" if value < 0.05 else "n.s."


def metric_display_name(metric_id: str) -> str:
    """Human-readable metric labels for the consistency table."""

    labels = {
        "precession_phase_uniformity": "Rayleigh: precession phase",
        "climate_lr04_co2_after_baseline": "Predictive: climate-state vs EP baseline",
        "phase_after_climate_state": "Predictive: phase vs climate-state",
        "lr04_after_co2_phase": "Predictive: full model vs -LR04",
        "co2_after_lr04_phase": "Predictive: full model vs -CO2",
        "full_vs_baseline": "Predictive: full model vs EP baseline",
    }
    return labels.get(metric_id, metric_id)


def event_display_name(event_type: str) -> str:
    """Short event labels for the consistency table."""

    labels = {
        "strong_monsoon_start": "Strong",
        "weak_monsoon_start": "Weak",
    }
    return labels.get(event_type, event_type)


def plot_consistency_table(consistency: pd.DataFrame) -> None:
    """Render the KS-window consistency summary as a table figure."""

    ensure_dir(OUT_FIG_DIR)
    plot_rows = consistency[
        consistency["metric_id"].isin(
            [
                "precession_phase_uniformity",
                "climate_lr04_co2_after_baseline",
                "phase_after_climate_state",
                "lr04_after_co2_phase",
                "co2_after_lr04_phase",
                "full_vs_baseline",
            ]
        )
    ].copy()
    plot_rows["event_order"] = plot_rows["event_type"].map(
        {"strong_monsoon_start": 0, "weak_monsoon_start": 1}
    )
    metric_order = {
        "precession_phase_uniformity": 0,
        "climate_lr04_co2_after_baseline": 1,
        "phase_after_climate_state": 2,
        "lr04_after_co2_phase": 3,
        "co2_after_lr04_phase": 4,
        "full_vs_baseline": 5,
    }
    plot_rows["metric_order"] = plot_rows["metric_id"].map(metric_order)
    plot_rows = plot_rows.sort_values(["event_order", "metric_order"]).reset_index(drop=True)

    columns = ["Event", "Test", "0.4-4 kyr", "0.6-4 kyr", "Same decision"]
    cell_text = []
    cell_colours = []
    for _, row in plot_rows.iterrows():
        p04 = float(row["ks_0p4_4kyr_p_value"])
        p06 = float(row["ks_0p6_4kyr_p_value"])
        same = bool(row["consistent_significance_decision"])
        cell_text.append(
            [
                event_display_name(str(row["event_type"])),
                metric_display_name(str(row["metric_id"])),
                f"{format_p_value(p04)} ({significance_label(p04)})",
                f"{format_p_value(p06)} ({significance_label(p06)})",
                "yes" if same else "no",
            ]
        )
        cell_colours.append(
            [
                "#f7f7f7",
                "#ffffff",
                "#d9ead3" if p04 < 0.05 else "#f4cccc",
                "#d9ead3" if p06 < 0.05 else "#f4cccc",
                "#d9ead3" if same else "#fce5cd",
            ]
        )

    fig_height = 0.42 * len(cell_text) + 1.45
    fig, ax = plt.subplots(figsize=(11.6, fig_height))
    ax.axis("off")
    ax.set_title(
        "Sensitivity of core results to Rousseau et al. KS-window catalogue",
        loc="left",
        fontsize=13,
        fontweight="bold",
        pad=14,
    )
    subtitle = (
        "Green p-value cells are significant at 0.05; red cells are not significant. "
        "The final column flags whether the significance decision is unchanged."
    )
    ax.text(0.0, 0.985, subtitle, transform=ax.transAxes, fontsize=8.8, va="top")

    table = ax.table(
        cellText=cell_text,
        colLabels=columns,
        cellColours=cell_colours,
        colColours=["#e6e6e6"] * len(columns),
        cellLoc="left",
        colLoc="left",
        colWidths=[0.08, 0.43, 0.16, 0.16, 0.13],
        bbox=[0.0, 0.0, 1.0, 0.93],
    )
    table.auto_set_font_size(False)
    table.set_fontsize(8.5)
    table.scale(1.0, 1.18)
    for (row_idx, col_idx), cell in table.get_celld().items():
        cell.set_edgecolor("#bdbdbd")
        cell.set_linewidth(0.45)
        if row_idx == 0:
            cell.set_text_props(weight="bold")
            cell.set_height(cell.get_height() * 1.2)
        if col_idx in {2, 3, 4} and row_idx > 0:
            cell.set_text_props(ha="center")

    fig.savefig(OUT_FIG_DIR / "core_experiment_consistency_table.png", dpi=300, bbox_inches="tight")
    fig.savefig(OUT_FIG_DIR / "core_experiment_consistency_table.pdf", bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Command-line workflow
# ---------------------------------------------------------------------------


def run_analysis() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Run the KS-window sensitivity workflow."""

    catalogues_by_window = {
        ks_window_id: load_catalogues_for_window(ks_window_id)
        for ks_window_id in KS_WINDOW_CATALOGUES
    }
    all_catalogues = [
        catalogue
        for catalogues in catalogues_by_window.values()
        for catalogue in catalogues
    ]

    events_used = build_events_used_table(all_catalogues)
    rayleigh_results = run_rayleigh_for_all(catalogues_by_window)
    (
        binned_inputs,
        poisson_summary,
        poisson_coefficients,
        poisson_likelihood_tests,
    ) = run_poisson_for_all(catalogues_by_window)
    consistency = build_consistency_summary(
        rayleigh_results,
        poisson_likelihood_tests,
        poisson_summary,
    )
    write_outputs(
        events_used=events_used,
        rayleigh_results=rayleigh_results,
        binned_inputs=binned_inputs,
        poisson_summary=poisson_summary,
        poisson_coefficients=poisson_coefficients,
        poisson_likelihood_tests=poisson_likelihood_tests,
        consistency=consistency,
    )
    plot_consistency_table(consistency)
    return rayleigh_results, poisson_likelihood_tests, consistency


def parse_args() -> argparse.Namespace:
    """Parse command-line options for the KS-window sensitivity script."""

    parser = argparse.ArgumentParser(
        description="Compare Rousseau 2023 core experiments across KS window catalogues."
    )
    return parser.parse_args()


def main() -> None:
    """Command-line entry point."""

    parse_args()
    rayleigh_results, poisson_lrt, consistency = run_analysis()

    print("\nPrecession Rayleigh p values:")
    print(
        rayleigh_results[rayleigh_results["driver"].eq("pre")][
            [
                "ks_window_id",
                "event_type",
                "n_phase_events_used",
                "mean_phase_deg",
                "rayleigh_p",
            ]
        ].to_string(index=False)
    )

    print("\nPoisson likelihood tests for controls:")
    print(
        poisson_lrt[
            poisson_lrt["comparison_id"].isin(
                [
                    "climate_lr04_co2_after_baseline",
                    "phase_after_climate_state",
                    "lr04_after_co2_phase",
                    "co2_after_lr04_phase",
                ]
            )
        ][
            [
                "ks_window_id",
                "dataset_id",
                "comparison_id",
                "LR_statistic",
                "LR_p_value",
                "delta_AICc_full_minus_reduced",
            ]
        ].to_string(index=False)
    )

    inconsistent = consistency[
        consistency["consistent_significance_decision"].eq(False)
    ]
    if len(inconsistent):
        print("\nSignificance decisions that change between KS windows:")
        print(
            inconsistent[
                [
                    "experiment",
                    "metric_id",
                    "event_type",
                    "ks_0p4_4kyr_p_value",
                    "ks_0p6_4kyr_p_value",
                ]
            ].to_string(index=False)
        )
    else:
        print("\nNo p<0.05 significance decisions changed between KS windows.")

    print(f"\nWrote sensitivity tables to: {OUT_DATA_DIR.relative_to(PROJECT_ROOT)}")
    print(f"Wrote summary figure to: {OUT_FIG_DIR.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    main()
