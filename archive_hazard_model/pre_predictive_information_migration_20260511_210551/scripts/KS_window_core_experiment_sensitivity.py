"""
Sensitivity of the three core Rousseau et al. (2023) experiments to KS window.

This script compares the complete 0.4-4 kyr KS transition catalogue against
the 0.6-4 kyr catalogue for:

1. Lohmann-style stationary-random occurrence tests.
2. Rayleigh tests for orbital phase preference.
3. Binned Poisson hazard models with LR04, CO2, and precession phase controls.

The implementation reuses the core definitions from the three main analysis
scripts while avoiding their figure-writing wrappers.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import chi2

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

import Lohmann_style_randomness_test as lohmann
import Orbital_phase_rayleigh as rayleigh
import archive_hazard_model.pre_predictive_information_migration_20260511_210551.scripts.Bin_hazard_phase_poisson as hazard


PROJECT_ROOT = Path(__file__).resolve().parent
RUN_NAME = "rousseau2023_ks_window_core_experiment_sensitivity"
OUT_DATA_DIR = PROJECT_ROOT / "data" / "processed" / RUN_NAME
OUT_FIG_DIR = PROJECT_ROOT / "figures" / RUN_NAME

ANALYSIS_START_KA = 0.0
ANALYSIS_END_KA = 640.0

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
        "seed_offset": 101,
    },
    "weak_monsoon_start": {
        "label": "Weak monsoon starts",
        "color": "#6a3d9a",
        "seed_offset": 211,
    },
}

POISSON_MODEL_SPECS: list[tuple[str, tuple[str, ...], str]] = [
    ("stationary", (), "Stationary"),
    ("lr04", ("lr04_scaled",), "LR04"),
    ("co2", ("co2_scaled",), "CO2"),
    ("climate_lr04_co2", ("lr04_scaled", "co2_scaled"), "LR04 + CO2"),
    ("pre_phase", ("pre_phase_sin", "pre_phase_cos"), "Precession phase"),
    (
        "lr04_pre_phase",
        ("lr04_scaled", "pre_phase_sin", "pre_phase_cos"),
        "LR04 + precession phase",
    ),
    (
        "co2_pre_phase",
        ("co2_scaled", "pre_phase_sin", "pre_phase_cos"),
        "CO2 + precession phase",
    ),
    (
        "climate_lr04_co2_pre_phase",
        ("lr04_scaled", "co2_scaled", "pre_phase_sin", "pre_phase_cos"),
        "LR04 + CO2 + precession phase",
    ),
]

POISSON_LRT_COMPARISONS: list[tuple[str, str, str, str]] = [
    (
        "lr04_after_stationary",
        "stationary",
        "lr04",
        "Does LR04 improve over a stationary event rate?",
    ),
    (
        "co2_after_stationary",
        "stationary",
        "co2",
        "Does CO2 improve over a stationary event rate?",
    ),
    (
        "climate_lr04_co2_vs_stationary",
        "stationary",
        "climate_lr04_co2",
        "Does the joint LR04+CO2 climate model improve over stationary?",
    ),
    (
        "pre_phase_vs_stationary",
        "stationary",
        "pre_phase",
        "Does precession phase improve over stationary?",
    ),
    (
        "phase_after_lr04_co2",
        "climate_lr04_co2",
        "climate_lr04_co2_pre_phase",
        "Does precession phase add information after LR04+CO2?",
    ),
    (
        "lr04_after_co2_and_phase",
        "co2_pre_phase",
        "climate_lr04_co2_pre_phase",
        "Does LR04 add information after CO2+precession phase?",
    ),
    (
        "co2_after_lr04_and_phase",
        "lr04_pre_phase",
        "climate_lr04_co2_pre_phase",
        "Does CO2 add information after LR04+precession phase?",
    ),
    (
        "full_vs_stationary",
        "stationary",
        "climate_lr04_co2_pre_phase",
        "Does LR04+CO2+precession phase improve over stationary?",
    ),
]


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


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def relative_path(path: Path) -> str:
    return str(path.relative_to(PROJECT_ROOT))


def load_event_catalogue(ks_window_id: str, event_type: str) -> EventCatalogue:
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

    ages = lohmann.unique_sorted(events["event_age_ka"].to_numpy(dtype=float))
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
    return [
        load_event_catalogue(ks_window_id, "strong_monsoon_start"),
        load_event_catalogue(ks_window_id, "weak_monsoon_start"),
    ]


def build_events_used_table(catalogues: list[EventCatalogue]) -> pd.DataFrame:
    rows = []
    for catalogue in catalogues:
        rows.append(catalogue.events)
    return pd.concat(rows, ignore_index=True)


def stable_seed(base_seed: int, ks_window_id: str, event_type: str) -> int:
    window_offset = 0 if ks_window_id == "ks_0p4_4kyr" else 10_000
    return int(base_seed + window_offset + EVENT_SETTINGS[event_type]["seed_offset"])


def run_lohmann_randomness(
    catalogue: EventCatalogue,
    *,
    n_monte_carlo: int,
    base_seed: int,
    window_ka: float,
    grid_step_ka: float,
) -> dict[str, float | int | str]:
    ages = catalogue.ages_ka[
        (catalogue.ages_ka >= ANALYSIS_START_KA) & (catalogue.ages_ka <= ANALYSIS_END_KA)
    ]
    if ages.size < 3:
        raise ValueError(f"{catalogue.ks_window_id}/{catalogue.event_type} has too few events.")

    duration_ka = ANALYSIS_END_KA - ANALYSIS_START_KA
    rate_per_kyr = ages.size / duration_ka
    expected_count = rate_per_kyr * window_ka
    centers = lohmann.centered_time_grid(
        ANALYSIS_START_KA,
        ANALYSIS_END_KA,
        window_ka,
        grid_step_ka,
    )
    observed_counts = lohmann.moving_event_counts(ages, centers, window_ka)
    observed_es = lohmann.rms_deviation(observed_counts, expected_count)

    rng = np.random.default_rng(stable_seed(base_seed, catalogue.ks_window_id, catalogue.event_type))
    conditional_es = np.empty(n_monte_carlo, dtype=float)
    poisson_es = np.empty(n_monte_carlo, dtype=float)
    for i in range(n_monte_carlo):
        conditional_events = np.sort(rng.uniform(ANALYSIS_START_KA, ANALYSIS_END_KA, size=ages.size))
        conditional_counts = lohmann.moving_event_counts(conditional_events, centers, window_ka)
        conditional_es[i] = lohmann.rms_deviation(conditional_counts, expected_count)

        n_events = int(rng.poisson(rate_per_kyr * duration_ka))
        poisson_events = np.sort(rng.uniform(ANALYSIS_START_KA, ANALYSIS_END_KA, size=n_events))
        poisson_counts = lohmann.moving_event_counts(poisson_events, centers, window_ka)
        poisson_es[i] = lohmann.rms_deviation(poisson_counts, expected_count)

    poisson_p = lohmann.p_value_greater_or_equal(poisson_es, observed_es)
    conditional_p = lohmann.p_value_greater_or_equal(conditional_es, observed_es)

    return {
        "ks_window_id": catalogue.ks_window_id,
        "ks_window_label": catalogue.ks_window_label,
        "ks_window_kyr": catalogue.ks_window_kyr,
        "event_type": catalogue.event_type,
        "event_label": catalogue.event_label,
        "source": relative_path(catalogue.source_path),
        "analysis_start_ka": ANALYSIS_START_KA,
        "analysis_end_ka": ANALYSIS_END_KA,
        "duration_ka": duration_ka,
        "n_events": int(ages.size),
        "rate_per_kyr_from_event_count": rate_per_kyr,
        f"expected_events_per_{window_ka:g}kyr": expected_count,
        "lohmann_event_count_ES": observed_es,
        "lohmann_poisson_ES_p_value": poisson_p,
        "conditional_fixed_n_ES_p_value": conditional_p,
        "reject_poisson_ES_at_0p05": poisson_p < 0.05,
        "reject_fixed_n_ES_at_0p05": conditional_p < 0.05,
        "n_monte_carlo": int(n_monte_carlo),
        "event_count_window_ka": window_ka,
        "time_grid_step_ka": grid_step_ka,
        "random_seed": stable_seed(base_seed, catalogue.ks_window_id, catalogue.event_type),
    }


def run_randomness_for_all(
    catalogues: list[EventCatalogue],
    *,
    n_monte_carlo: int,
    seed: int,
    window_ka: float,
    grid_step_ka: float,
) -> pd.DataFrame:
    rows = [
        run_lohmann_randomness(
            catalogue,
            n_monte_carlo=n_monte_carlo,
            base_seed=seed,
            window_ka=window_ka,
            grid_step_ka=grid_step_ka,
        )
        for catalogue in catalogues
    ]
    return pd.DataFrame(rows)


def rayleigh_events_frame(catalogues: list[EventCatalogue]) -> pd.DataFrame:
    return pd.concat([catalogue.events for catalogue in catalogues], ignore_index=True)


def run_rayleigh_for_all(catalogues_by_window: dict[str, list[EventCatalogue]]) -> pd.DataFrame:
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


def hazard_events_for_window(catalogues: list[EventCatalogue]) -> list[hazard.EventDataset]:
    datasets = []
    for catalogue in catalogues:
        settings = EVENT_SETTINGS[catalogue.event_type]
        datasets.append(
            hazard.EventDataset(
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
) -> list[hazard.FittedPoissonModel]:
    models: list[hazard.FittedPoissonModel] = []
    for _, group in binned_inputs.groupby("dataset_id", sort=False):
        for model_id, terms, label in POISSON_MODEL_SPECS:
            models.append(hazard.fit_poisson_model(group, model_id, terms, label))
    return models


def build_sensitivity_likelihood_tests(
    models: list[hazard.FittedPoissonModel],
) -> pd.DataFrame:
    lookup = hazard.model_lookup(models)
    rows = []
    dataset_ids = sorted({model.dataset_id for model in models})
    for dataset_id in dataset_ids:
        for comparison_id, reduced_id, full_id, question in POISSON_LRT_COMPARISONS:
            reduced = lookup[(dataset_id, reduced_id)]
            full = lookup[(dataset_id, full_id)]
            lr_stat = 2.0 * (full.log_likelihood - reduced.log_likelihood)
            df = len(full.beta) - len(reduced.beta)
            p_value = float(chi2.sf(max(lr_stat, 0.0), df))
            rows.append(
                {
                    "dataset_id": dataset_id,
                    "dataset_label": full.dataset_label,
                    "comparison_id": comparison_id,
                    "question": question,
                    "reduced_model_id": reduced_id,
                    "full_model_id": full_id,
                    "df": int(df),
                    "loglik_reduced": reduced.log_likelihood,
                    "loglik_full": full.log_likelihood,
                    "LR_statistic": lr_stat,
                    "LR_p_value": p_value,
                    "reject_LR_at_0p05": p_value < 0.05,
                    "delta_AICc_full_minus_reduced": full.aicc - reduced.aicc,
                }
            )
    return pd.DataFrame(rows)


def add_window_columns(frame: pd.DataFrame, ks_window_id: str) -> pd.DataFrame:
    window = KS_WINDOW_CATALOGUES[ks_window_id]
    out = frame.copy()
    out.insert(0, "ks_window_id", ks_window_id)
    out.insert(1, "ks_window_label", window["label"])
    out.insert(2, "ks_window_kyr", window["ks_window_kyr"])
    return out


def run_poisson_for_all(
    catalogues_by_window: dict[str, list[EventCatalogue]],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    binned_frames = []
    summary_frames = []
    coefficient_frames = []
    likelihood_frames = []

    for ks_window_id, catalogues in catalogues_by_window.items():
        events = hazard_events_for_window(catalogues)
        binned_inputs, _, _ = hazard.build_binned_inputs(events)
        models = fit_sensitivity_poisson_models(binned_inputs)
        summary = hazard.build_model_summary(models, binned_inputs)
        coefficients = hazard.build_coefficient_table(models)
        likelihood_tests = build_sensitivity_likelihood_tests(models)

        binned_frames.append(add_window_columns(binned_inputs, ks_window_id))
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
    randomness: pd.DataFrame,
    rayleigh_results: pd.DataFrame,
    poisson_lrt: pd.DataFrame,
    poisson_summary: pd.DataFrame,
) -> pd.DataFrame:
    rows = []
    for event_type in EVENT_SETTINGS:
        r04 = randomness[
            randomness["ks_window_id"].eq("ks_0p4_4kyr")
            & randomness["event_type"].eq(event_type)
        ].iloc[0]
        r06 = randomness[
            randomness["ks_window_id"].eq("ks_0p6_4kyr")
            & randomness["event_type"].eq(event_type)
        ].iloc[0]
        rows.append(
            consistency_row(
                experiment="lohmann_randomness",
                metric_id="poisson_ES_null",
                event_type=event_type,
                p_04=float(r04["lohmann_poisson_ES_p_value"]),
                p_06=float(r06["lohmann_poisson_ES_p_value"]),
                extra_04={"n_events": int(r04["n_events"])},
                extra_06={"n_events": int(r06["n_events"])},
            )
        )
        rows.append(
            consistency_row(
                experiment="lohmann_randomness",
                metric_id="fixed_n_ES_null",
                event_type=event_type,
                p_04=float(r04["conditional_fixed_n_ES_p_value"]),
                p_06=float(r06["conditional_fixed_n_ES_p_value"]),
                extra_04={"n_events": int(r04["n_events"])},
                extra_06={"n_events": int(r06["n_events"])},
            )
        )

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
            "climate_lr04_co2_vs_stationary",
            "phase_after_lr04_co2",
            "lr04_after_co2_and_phase",
            "co2_after_lr04_and_phase",
            "full_vs_stationary",
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
                    experiment="poisson_hazard",
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
            & poisson_summary["model_id"].eq("climate_lr04_co2_pre_phase")
        ].iloc[0]
        full06 = poisson_summary[
            poisson_summary["ks_window_id"].eq("ks_0p6_4kyr")
            & poisson_summary["dataset_id"].eq(event_type)
            & poisson_summary["model_id"].eq("climate_lr04_co2_pre_phase")
        ].iloc[0]
        rows.append(
            {
                "experiment": "poisson_hazard",
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
    randomness: pd.DataFrame,
    rayleigh_results: pd.DataFrame,
    binned_inputs: pd.DataFrame,
    poisson_summary: pd.DataFrame,
    poisson_coefficients: pd.DataFrame,
    poisson_likelihood_tests: pd.DataFrame,
    consistency: pd.DataFrame,
) -> None:
    ensure_dir(OUT_DATA_DIR)
    events_used.to_csv(OUT_DATA_DIR / "events_used_by_ks_window.csv", index=False)
    randomness.to_csv(OUT_DATA_DIR / "lohmann_randomness_by_ks_window.csv", index=False)
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


def format_p_value(value: float) -> str:
    if not np.isfinite(value):
        return ""
    if value < 1e-4:
        return f"{value:.1e}"
    if value < 0.01:
        return f"{value:.4f}"
    return f"{value:.3f}"


def significance_label(value: float) -> str:
    if not np.isfinite(value):
        return ""
    return "sig" if value < 0.05 else "n.s."


def metric_display_name(metric_id: str) -> str:
    labels = {
        "poisson_ES_null": "Randomness: Poisson null",
        "fixed_n_ES_null": "Randomness: fixed-N null",
        "precession_phase_uniformity": "Rayleigh: precession phase",
        "climate_lr04_co2_vs_stationary": "Poisson: LR04+CO2 vs stationary",
        "phase_after_lr04_co2": "Poisson: phase after LR04+CO2",
        "lr04_after_co2_and_phase": "Poisson: LR04 after CO2+phase",
        "co2_after_lr04_and_phase": "Poisson: CO2 after LR04+phase",
        "full_vs_stationary": "Poisson: full model vs stationary",
    }
    return labels.get(metric_id, metric_id)


def event_display_name(event_type: str) -> str:
    labels = {
        "strong_monsoon_start": "Strong",
        "weak_monsoon_start": "Weak",
    }
    return labels.get(event_type, event_type)


def plot_consistency_table(consistency: pd.DataFrame) -> None:
    ensure_dir(OUT_FIG_DIR)
    plot_rows = consistency[
        consistency["metric_id"].isin(
            [
                "poisson_ES_null",
                "fixed_n_ES_null",
                "precession_phase_uniformity",
                "climate_lr04_co2_vs_stationary",
                "phase_after_lr04_co2",
                "lr04_after_co2_and_phase",
                "co2_after_lr04_and_phase",
                "full_vs_stationary",
            ]
        )
    ].copy()
    plot_rows["event_order"] = plot_rows["event_type"].map(
        {"strong_monsoon_start": 0, "weak_monsoon_start": 1}
    )
    metric_order = {
        "poisson_ES_null": 0,
        "fixed_n_ES_null": 1,
        "precession_phase_uniformity": 2,
        "climate_lr04_co2_vs_stationary": 3,
        "phase_after_lr04_co2": 4,
        "lr04_after_co2_and_phase": 5,
        "co2_after_lr04_and_phase": 6,
        "full_vs_stationary": 7,
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


def run_analysis(
    *,
    n_monte_carlo: int,
    seed: int,
    window_ka: float,
    grid_step_ka: float,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
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
    randomness = run_randomness_for_all(
        all_catalogues,
        n_monte_carlo=n_monte_carlo,
        seed=seed,
        window_ka=window_ka,
        grid_step_ka=grid_step_ka,
    )
    rayleigh_results = run_rayleigh_for_all(catalogues_by_window)
    (
        binned_inputs,
        poisson_summary,
        poisson_coefficients,
        poisson_likelihood_tests,
    ) = run_poisson_for_all(catalogues_by_window)
    consistency = build_consistency_summary(
        randomness,
        rayleigh_results,
        poisson_likelihood_tests,
        poisson_summary,
    )
    write_outputs(
        events_used=events_used,
        randomness=randomness,
        rayleigh_results=rayleigh_results,
        binned_inputs=binned_inputs,
        poisson_summary=poisson_summary,
        poisson_coefficients=poisson_coefficients,
        poisson_likelihood_tests=poisson_likelihood_tests,
        consistency=consistency,
    )
    plot_consistency_table(consistency)
    return randomness, rayleigh_results, poisson_likelihood_tests, consistency


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare Rousseau 2023 core experiments across KS window catalogues."
    )
    parser.add_argument("--n-monte-carlo", type=int, default=lohmann.N_MONTE_CARLO)
    parser.add_argument("--seed", type=int, default=lohmann.RANDOM_SEED)
    parser.add_argument("--window-ka", type=float, default=lohmann.EVENT_COUNT_WINDOW_KA)
    parser.add_argument("--grid-step-ka", type=float, default=lohmann.TIME_GRID_STEP_KA)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    randomness, rayleigh_results, poisson_lrt, consistency = run_analysis(
        n_monte_carlo=args.n_monte_carlo,
        seed=args.seed,
        window_ka=args.window_ka,
        grid_step_ka=args.grid_step_ka,
    )

    print("\nLohmann randomness p values:")
    print(
        randomness[
            [
                "ks_window_id",
                "event_type",
                "n_events",
                "lohmann_poisson_ES_p_value",
                "conditional_fixed_n_ES_p_value",
            ]
        ].to_string(index=False)
    )

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
                    "climate_lr04_co2_vs_stationary",
                    "phase_after_lr04_co2",
                    "lr04_after_co2_and_phase",
                    "co2_after_lr04_and_phase",
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
