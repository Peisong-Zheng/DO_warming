"""
Rayleigh and predictive-information checks for Barker et al. (2011) predicted
D-O warming events.

The Barker et al. supplementary workbook lists predicted D-O event occurrence
from the synthetic Greenland record. The variable-threshold picks are treated
here as an independent analogue of strong-monsoon starts: transitions toward
warmer Greenland/interstadial-like conditions.

Two age variants are analysed:

1. variable_threshold_edc3_0_640:
   Event ages from the EDC3 column, restricted to 0-640 ka.
2. variable_threshold_speleo_0_400:
   Event ages from the Barker SpeleoAge column, where available, restricted to
   0-400 ka. This keeps the age scale aligned with the Chinese speleothem-tuned
   part of Barker et al. (2011), but necessarily uses fewer events.

For each variant this script runs:
- Rayleigh tests of precession and obliquity phase preference;
- binned Poisson predictive-information models with same-type history as the
  event-process baseline, then LR04, CO2, and precession phase as predictors.

Unlike the Rousseau KS catalogue, Barker events were not detected from the
Cheng composite, so the Cheng sampling-resolution nuisance term is not used.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import chi2

import Bin_hazard_phase_poisson as base
import Orbital_phase_rayleigh as rayleigh
import Predictive_hazard_history_resolution as predictive


RUN_NAME = "barker2011_do_predictive_information"
OUT_DATA_DIR = base.PROJECT_ROOT / "data" / "processed" / RUN_NAME
OUT_FIG_DIR = base.PROJECT_ROOT / "figures" / RUN_NAME

BARKER_XLS = base.PROJECT_ROOT / "data/raw/Barker et al-2011-SOM.xls"

MAIN_HISTORY_WINDOW_KA = 5.0
RAYLEIGH_ALPHA = 0.05

HISTORY_TERM = predictive.HISTORY_TERM
CLIMATE_TERMS = predictive.CLIMATE_TERMS
PHASE_TERMS = predictive.PHASE_TERMS

MODEL_SPECS: list[tuple[str, tuple[str, ...], str]] = [
    ("stationary", (), "Stationary"),
    ("history_baseline", (HISTORY_TERM,), "History"),
    ("history_climate_lr04_co2", (HISTORY_TERM,) + CLIMATE_TERMS, "History + LR04 + CO2"),
    ("history_pre_phase", (HISTORY_TERM,) + PHASE_TERMS, "History + precession phase"),
    (
        "history_climate_lr04_co2_pre_phase",
        (HISTORY_TERM,) + CLIMATE_TERMS + PHASE_TERMS,
        "History + LR04 + CO2 + precession phase",
    ),
]

LR_TEST_SPECS: list[tuple[str, str, str, str]] = [
    (
        "history_vs_stationary",
        "stationary",
        "history_baseline",
        "Does same-type event history improve over a constant rate?",
    ),
    (
        "climate_lr04_co2_after_history",
        "history_baseline",
        "history_climate_lr04_co2",
        "Does LR04+CO2 improve over same-type history?",
    ),
    (
        "pre_phase_after_history",
        "history_baseline",
        "history_pre_phase",
        "Does precession phase improve over same-type history?",
    ),
    (
        "phase_after_history_climate",
        "history_climate_lr04_co2",
        "history_climate_lr04_co2_pre_phase",
        "Does precession phase add information after history+LR04+CO2?",
    ),
    (
        "full_after_history",
        "history_baseline",
        "history_climate_lr04_co2_pre_phase",
        "Does LR04+CO2+precession phase improve over same-type history?",
    ),
]

CATALOGUE_VARIANTS = [
    {
        "dataset_id": "barker_variable_threshold_edc3_0_640",
        "event_type": "barker_do_warming_variable_threshold_edc3",
        "label": "Barker variable-threshold D-O warmings, EDC3 0-640 ka",
        "age_column": "Age kyr (EDC3)",
        "pick_column": "DO pick variable threshold",
        "analysis_start_ka": 0.0,
        "analysis_end_ka": 640.0,
        "color": "#d95f02",
        "primary": True,
    },
    {
        "dataset_id": "barker_variable_threshold_speleo_0_400",
        "event_type": "barker_do_warming_variable_threshold_speleo",
        "label": "Barker variable-threshold D-O warmings, SpeleoAge 0-400 ka",
        "age_column": "SpeloAge (kyr).1",
        "pick_column": "DO pick variable threshold",
        "analysis_start_ka": 0.0,
        "analysis_end_ka": 400.0,
        "color": "#cc6677",
        "primary": False,
    },
    {
        "dataset_id": "barker_fixed_threshold_edc3_0_640",
        "event_type": "barker_do_warming_fixed_threshold_edc3",
        "label": "Barker fixed-threshold D-O warmings, EDC3 0-640 ka",
        "age_column": "Age kyr (EDC3)",
        "pick_column": "DO pick",
        "analysis_start_ka": 0.0,
        "analysis_end_ka": 640.0,
        "color": "#0072B2",
        "primary": False,
    },
]


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


def save_figure(fig: plt.Figure, stem: str, write_pdf: bool = True) -> None:
    ensure_dir(OUT_FIG_DIR)
    fig.savefig(OUT_FIG_DIR / f"{stem}.png", dpi=300, bbox_inches="tight")
    if write_pdf:
        fig.savefig(OUT_FIG_DIR / f"{stem}.pdf", bbox_inches="tight")
    plt.close(fig)


def format_p(value: float) -> str:
    if not np.isfinite(value):
        return "NA"
    if value < 1e-4:
        return f"{value:.1e}"
    if value < 0.01:
        return f"{value:.4f}"
    return f"{value:.3f}"


def load_barker_table_s3() -> pd.DataFrame:
    """Load Barker et al. Table S3 from Sheet1 of the supplementary workbook."""

    raw = pd.read_excel(BARKER_XLS, sheet_name="Sheet1", header=8)
    required = [
        "Age kyr (EDC3)",
        "SpeloAge (kyr).1",
        "DO pick",
        "DO pick variable threshold",
    ]
    missing = [column for column in required if column not in raw.columns]
    if missing:
        raise ValueError(f"Missing expected Barker columns: {missing}")
    out = raw[required].copy()
    for column in required:
        out[column] = pd.to_numeric(out[column], errors="coerce")
    out = out.dropna(subset=["Age kyr (EDC3)"], how="all")
    return out.reset_index(drop=True)


def build_event_catalogues() -> tuple[pd.DataFrame, list[base.EventDataset]]:
    table = load_barker_table_s3()
    event_frames = []
    datasets: list[base.EventDataset] = []
    for settings in CATALOGUE_VARIANTS:
        age = pd.to_numeric(table[settings["age_column"]], errors="coerce")
        pick = pd.to_numeric(table[settings["pick_column"]], errors="coerce")
        frame = pd.DataFrame(
            {
                "event_age_ka": age,
                "pick_value": pick,
            }
        )
        frame = frame.dropna(subset=["event_age_ka", "pick_value"])
        frame = frame[frame["pick_value"].eq(1.0)].copy()
        frame = frame[
            frame["event_age_ka"].between(
                settings["analysis_start_ka"],
                settings["analysis_end_ka"],
                inclusive="both",
            )
        ].copy()
        frame = frame.sort_values("event_age_ka").reset_index(drop=True)
        frame["event_index"] = np.arange(1, len(frame) + 1)
        frame["dataset_id"] = settings["dataset_id"]
        frame["event_type"] = settings["event_type"]
        frame["event_label"] = settings["label"]
        frame["age_column"] = settings["age_column"]
        frame["pick_column"] = settings["pick_column"]
        frame["analysis_start_ka"] = settings["analysis_start_ka"]
        frame["analysis_end_ka"] = settings["analysis_end_ka"]
        frame["source"] = str(BARKER_XLS.relative_to(base.PROJECT_ROOT))
        event_frames.append(frame)

        datasets.append(
            base.EventDataset(
                dataset_id=settings["dataset_id"],
                label=settings["label"],
                color=settings["color"],
                ages_ka=frame["event_age_ka"].to_numpy(dtype=float),
                source=str(BARKER_XLS.relative_to(base.PROJECT_ROOT)),
            )
        )

    return pd.concat(event_frames, ignore_index=True), datasets


def build_rayleigh_tables(events: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    phase_products = {
        driver: rayleigh.build_phase_series(driver, settings)
        for driver, settings in rayleigh.DRIVER_SETTINGS.items()
    }
    event_phases = rayleigh.sample_event_phases(events, phase_products)
    results = rayleigh.build_rayleigh_results(event_phases)
    return event_phases, results


def build_variant_binned_inputs(dataset: base.EventDataset, analysis_end_ka: float) -> pd.DataFrame:
    """Build base predictor bins and keep only the intended age interval."""

    binned, _, _ = base.build_binned_inputs([dataset])
    binned = binned[
        binned["bin_center_ka"].between(0.0, analysis_end_ka, inclusive="both")
    ].copy()
    binned = predictive.add_same_type_history(binned, MAIN_HISTORY_WINDOW_KA)
    return binned


def fit_predictive_models(binned: pd.DataFrame) -> tuple[list[base.FittedPoissonModel], pd.DataFrame]:
    fit_frame = predictive.model_frame(binned)
    models: list[base.FittedPoissonModel] = []
    for _, group in fit_frame.groupby("dataset_id", sort=False):
        for model_id, terms, label in MODEL_SPECS:
            models.append(base.fit_poisson_model(group, model_id, terms, label))
    return models, fit_frame


def build_likelihood_tests(
    models: list[base.FittedPoissonModel],
    fit_frame: pd.DataFrame,
) -> pd.DataFrame:
    lookup = base.model_lookup(models)
    rows = []
    for dataset_id, group in fit_frame.groupby("dataset_id", sort=False):
        n_events = int(group["event_count"].sum())
        n_bins = int(len(group))
        for comparison_id, reduced_id, full_id, question in LR_TEST_SPECS:
            reduced = lookup[(dataset_id, reduced_id)]
            full = lookup[(dataset_id, full_id)]
            ll_gain = full.log_likelihood - reduced.log_likelihood
            lr_stat = 2.0 * ll_gain
            df = len(full.beta) - len(reduced.beta)
            p_value = float(chi2.sf(max(lr_stat, 0.0), df))
            rows.append(
                {
                    "dataset_id": dataset_id,
                    "dataset_label": full.dataset_label,
                    "history_window_ka": MAIN_HISTORY_WINDOW_KA,
                    "comparison_id": comparison_id,
                    "question": question,
                    "reduced_model_id": reduced_id,
                    "full_model_id": full_id,
                    "df": int(df),
                    "n_bins": n_bins,
                    "n_events": n_events,
                    "loglik_reduced": reduced.log_likelihood,
                    "loglik_full": full.log_likelihood,
                    "ll_gain_nats": ll_gain,
                    "info_bits_per_event": ll_gain / np.log(2.0) / max(n_events, 1),
                    "info_bits_per_bin": ll_gain / np.log(2.0) / max(n_bins, 1),
                    "LR_statistic": lr_stat,
                    "LR_p_value": p_value,
                    "reject_LR_at_0p05": p_value < 0.05,
                    "delta_AICc_full_minus_reduced": full.aicc - reduced.aicc,
                }
            )
    return pd.DataFrame(rows)


def build_predictive_tables(
    datasets: list[base.EventDataset],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    binned_frames = []
    summary_frames = []
    lrt_frames = []
    fitted_frames = []

    end_lookup = {settings["dataset_id"]: settings["analysis_end_ka"] for settings in CATALOGUE_VARIANTS}
    for dataset in datasets:
        binned = build_variant_binned_inputs(dataset, end_lookup[dataset.dataset_id])
        models, fit_frame = fit_predictive_models(binned)
        summary = base.build_model_summary(models, fit_frame)
        lrt = build_likelihood_tests(models, fit_frame)
        fitted = base.build_fitted_rate_table(models, fit_frame)

        binned_frames.append(fit_frame)
        summary_frames.append(summary)
        lrt_frames.append(lrt)
        fitted_frames.append(fitted)

    return (
        pd.concat(binned_frames, ignore_index=True),
        pd.concat(summary_frames, ignore_index=True),
        pd.concat(lrt_frames, ignore_index=True),
        pd.concat(fitted_frames, ignore_index=True),
    )


def plot_rayleigh_precession(event_phases: pd.DataFrame, results: pd.DataFrame) -> None:
    variants = [settings for settings in CATALOGUE_VARIANTS if settings["pick_column"] == "DO pick variable threshold"]
    fig, axes = plt.subplots(
        1,
        len(variants),
        figsize=(7.2, 3.7),
        subplot_kw={"projection": "polar"},
        constrained_layout=True,
    )
    if len(variants) == 1:
        axes = [axes]
    bins = np.linspace(0.0, 2.0 * np.pi, 19)
    for ax, settings in zip(axes, variants):
        event_type = settings["event_type"]
        phases = event_phases[
            event_phases["driver"].eq("pre")
            & event_phases["event_type"].eq(event_type)
            & ~event_phases["phase_extrapolated"].astype(bool)
        ]["phase_rad"].to_numpy(dtype=float)
        counts, edges = np.histogram(phases, bins=bins)
        widths = np.diff(edges)
        ax.bar(
            edges[:-1],
            counts,
            width=widths,
            align="edge",
            color=settings["color"],
            alpha=0.55,
            edgecolor="white",
            linewidth=0.6,
        )
        row = results[results["driver"].eq("pre") & results["event_type"].eq(event_type)].iloc[0]
        mean_phase = float(row["mean_phase_rad"])
        rbar = float(row["mean_resultant_length"])
        n = int(row["n_phase_events_used"])
        max_count = max(int(counts.max()), 1)
        ax.annotate(
            "",
            xy=(mean_phase, max_count * rbar),
            xytext=(mean_phase, 0.0),
            arrowprops={"arrowstyle": "-|>", "lw": 1.6, "color": "#222222"},
        )
        ax.set_ylim(0, max_count + 1.2)
        ax.set_theta_zero_location("E")
        ax.set_theta_direction(1)
        ax.set_yticklabels([])
        ax.grid(color="0.82", alpha=0.45, lw=0.6)
        title = "EDC3 0-640 ka" if "edc3" in settings["dataset_id"] else "SpeleoAge 0-400 ka"
        ax.set_title(
            f"{title}\nN={n}, mean={row['mean_phase_deg']:.1f} deg, p={format_p(row['rayleigh_p'])}",
            pad=12,
        )
    save_figure(fig, "fig01_barker_precession_rayleigh_polar")


def plot_predictive_summary(lrt: pd.DataFrame) -> None:
    wanted = [
        "climate_lr04_co2_after_history",
        "phase_after_history_climate",
        "full_after_history",
    ]
    labels = {
        "climate_lr04_co2_after_history": "LR04+CO2\nafter history",
        "phase_after_history_climate": "Precession phase\nafter history+LR04+CO2",
        "full_after_history": "Full model\nafter history",
    }
    variants = [settings["dataset_id"] for settings in CATALOGUE_VARIANTS]
    fig, ax = plt.subplots(figsize=(8.0, 3.4), constrained_layout=True)
    width = 0.24
    x = np.arange(len(wanted))
    for idx, dataset_id in enumerate(variants):
        sub = lrt[lrt["dataset_id"].eq(dataset_id)].set_index("comparison_id")
        values = [-np.log10(max(float(sub.loc[item, "LR_p_value"]), 1e-300)) for item in wanted]
        label = next(s["label"] for s in CATALOGUE_VARIANTS if s["dataset_id"] == dataset_id)
        short = label.replace("Barker ", "").replace(" D-O warmings, ", "\n")
        color = next(s["color"] for s in CATALOGUE_VARIANTS if s["dataset_id"] == dataset_id)
        ax.bar(x + (idx - 1) * width, values, width=width, label=short, color=color, alpha=0.82)
    ax.axhline(-np.log10(0.05), color="0.25", lw=0.8, ls="--")
    ax.set_xticks(x)
    ax.set_xticklabels([labels[item] for item in wanted])
    ax.set_ylabel(r"$-\log_{10}(p)$")
    ax.legend(frameon=False, loc="upper right", fontsize=7)
    ax.grid(axis="y", color="0.88", lw=0.6)
    save_figure(fig, "fig02_barker_predictive_likelihood_tests")


def write_outputs(
    events: pd.DataFrame,
    event_phases: pd.DataFrame,
    rayleigh_results: pd.DataFrame,
    binned: pd.DataFrame,
    summary: pd.DataFrame,
    lrt: pd.DataFrame,
    fitted: pd.DataFrame,
) -> None:
    ensure_dir(OUT_DATA_DIR)
    events.to_csv(OUT_DATA_DIR / "barker2011_event_catalogues_used.csv", index=False)
    event_phases.to_csv(OUT_DATA_DIR / "barker2011_event_orbital_phases.csv", index=False)
    rayleigh_results.to_csv(OUT_DATA_DIR / "barker2011_rayleigh_phase_results.csv", index=False)
    binned.to_csv(OUT_DATA_DIR / "barker2011_binned_predictive_inputs.csv", index=False)
    summary.to_csv(OUT_DATA_DIR / "barker2011_predictive_model_summary.csv", index=False)
    lrt.to_csv(OUT_DATA_DIR / "barker2011_predictive_likelihood_tests.csv", index=False)
    fitted.to_csv(OUT_DATA_DIR / "barker2011_fitted_rates.csv", index=False)
    pd.DataFrame(
        [
            {
                "run_name": RUN_NAME,
                "source": str(BARKER_XLS.relative_to(base.PROJECT_ROOT)),
                "bin_width_ka": base.BIN_WIDTH_KA,
                "history_window_ka": MAIN_HISTORY_WINDOW_KA,
                "note": "Barker events use same-type history only; Cheng resolution control is not included.",
            }
        ]
    ).to_csv(OUT_DATA_DIR / "parameters.csv", index=False)


def print_summary(rayleigh_results: pd.DataFrame, summary: pd.DataFrame, lrt: pd.DataFrame) -> None:
    print("\nBarker et al. (2011) Rayleigh results:")
    ray = rayleigh_results[rayleigh_results["driver"].eq("pre")].copy()
    print(
        ray[
            [
                "event_type",
                "n_phase_events_used",
                "mean_phase_deg",
                "mean_resultant_length",
                "rayleigh_z",
                "rayleigh_p",
            ]
        ].to_string(index=False)
    )

    print("\nPredictive-information likelihood tests:")
    key_tests = lrt[
        lrt["comparison_id"].isin(
            [
                "climate_lr04_co2_after_history",
                "phase_after_history_climate",
                "full_after_history",
            ]
        )
    ].copy()
    print(
        key_tests[
            [
                "dataset_id",
                "comparison_id",
                "n_events",
                "LR_statistic",
                "df",
                "LR_p_value",
                "delta_AICc_full_minus_reduced",
                "info_bits_per_event",
            ]
        ].to_string(index=False)
    )

    full = summary[summary["model_id"].eq("history_climate_lr04_co2_pre_phase")].copy()
    cols = [
        "dataset_id",
        "n_events",
        "delta_AICc",
        "pre_phase_preferred_deg",
        "pre_phase_rate_ratio_max_vs_min",
    ]
    print("\nFull-model precession-phase estimates:")
    print(full[cols].to_string(index=False))
    print(f"\nWrote outputs to {OUT_DATA_DIR.relative_to(base.PROJECT_ROOT)}")
    print(f"Wrote figures to {OUT_FIG_DIR.relative_to(base.PROJECT_ROOT)}")


def main() -> None:
    events, datasets = build_event_catalogues()
    event_phases, rayleigh_results = build_rayleigh_tables(events)
    binned, model_summary, lrt, fitted = build_predictive_tables(datasets)
    write_outputs(events, event_phases, rayleigh_results, binned, model_summary, lrt, fitted)
    plot_rayleigh_precession(event_phases, rayleigh_results)
    plot_predictive_summary(lrt)
    print_summary(rayleigh_results, model_summary, lrt)


if __name__ == "__main__":
    main()
