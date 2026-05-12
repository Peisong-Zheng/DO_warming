"""
Sensitivity of Rousseau et al. (2023) core event-timing tests to a rough
Cheng et al. (2016) composite age-uncertainty envelope.

The age-uncertainty model is intentionally simple: the 2-sigma age range grows
exponentially from 0.1 kyr at 0 ka BP to 7.4 kyr at 640 ka BP. The envelope is
evaluated on the Cheng composite age grid, interpolated to the 0.4-4 kyr
Rousseau strong/weak start catalogues, and then used to generate randomized
event catalogues with rank-preserving truncated Gaussian draws.

For each randomized catalogue, the script runs a simplified version of the
core experiments:

1. Lohmann-style stationary-random occurrence test.
2. Rayleigh test for precession-phase clustering.
3. Likelihood-ratio test of the full LR04 + CO2 + precession-phase Poisson
   hazard model against a stationary hazard model.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import numpy as np
import pandas as pd
from scipy.stats import chi2, norm, truncnorm

import Lohmann_style_randomness_test as lohmann
import Orbital_phase_rayleigh as rayleigh
import archive_hazard_model.pre_predictive_information_migration_20260511_210551.scripts.Bin_hazard_phase_poisson as hazard


PROJECT_ROOT = Path(__file__).resolve().parent
RUN_NAME = "rousseau2023_composite_age_uncertainty_sensitivity"
OUT_DATA_DIR = PROJECT_ROOT / "data" / "processed" / RUN_NAME
OUT_FIG_DIR = PROJECT_ROOT / "figures" / RUN_NAME

CHENG_XLSX = PROJECT_ROOT / "data/raw/Cheng_2016.xlsx"
CHENG_COMPOSITE_SHEET = "Composite record"
BEST_MATCH_UNCERTAINTY_CSV = (
    PROJECT_ROOT
    / "age_scale_uncertainty"
    / "cheng2016_composite_age_uncertainty_from_best_match.csv"
)
STRONG_CSV = PROJECT_ROOT / "data/raw/rousseau_2023_ks_0p4_4kyr_strong_monsoon_start_times.csv"
WEAK_CSV = PROJECT_ROOT / "data/raw/rousseau_2023_ks_0p4_4kyr_weak_monsoon_start_times.csv"

ANALYSIS_START_KA = 0.0
ANALYSIS_END_KA = 640.0
ERROR_2SIGMA_START_KA = 1
ERROR_2SIGMA_END_KA = 7.4  #7.4 #10

DEFAULT_N_REALIZATIONS = 500
DEFAULT_N_LOHMANN_NULL = 5000
DEFAULT_RANDOM_SEED = 20260506

EVENT_SETTINGS = {
    "strong_monsoon_start": {
        "label": "Strong monsoon starts",
        "short_label": "Strong",
        "path": STRONG_CSV,
        "color": "#d95f02",
    },
    "weak_monsoon_start": {
        "label": "Weak monsoon starts",
        "short_label": "Weak",
        "path": WEAK_CSV,
        "color": "#6a3d9a",
    },
}

FULL_MODEL_ID = "climate_lr04_co2_pre_phase"
FULL_MODEL_LABEL = "LR04 + CO2 + precession phase"
FULL_MODEL_TERMS = ("lr04_scaled", "co2_scaled", "pre_phase_sin", "pre_phase_cos")


@dataclass(frozen=True)
class LohmannNull:
    event_type: str
    n_events: int
    centers_ka: np.ndarray
    expected_count: float
    conditional_es_samples: np.ndarray
    poisson_es_samples: np.ndarray


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def save_figure(fig: plt.Figure, stem: str, write_pdf: bool) -> None:
    ensure_dir(OUT_FIG_DIR)
    fig.savefig(OUT_FIG_DIR / f"{stem}.png", dpi=300, bbox_inches="tight")
    if write_pdf:
        fig.savefig(OUT_FIG_DIR / f"{stem}.pdf", bbox_inches="tight")
    plt.close(fig)


def relative_path(path: Path) -> str:
    return str(path.relative_to(PROJECT_ROOT))


def find_column(columns: pd.Index, *needles: str) -> str:
    normalized = {str(column).strip().lower(): column for column in columns}
    for key, column in normalized.items():
        if all(needle.lower() in key for needle in needles):
            return str(column)
    raise ValueError(f"Cannot find a column containing {needles}.")


def age_error_2sigma_kyr(age_ka: np.ndarray | float) -> np.ndarray:
    age = np.asarray(age_ka, dtype=float)
    clipped = np.clip(age, ANALYSIS_START_KA, ANALYSIS_END_KA)
    fraction = (clipped - ANALYSIS_START_KA) / (ANALYSIS_END_KA - ANALYSIS_START_KA)
    growth = np.log(ERROR_2SIGMA_END_KA / ERROR_2SIGMA_START_KA)
    return ERROR_2SIGMA_START_KA * np.exp(growth * fraction)


def load_cheng_composite_with_error() -> pd.DataFrame:
    raw = pd.read_excel(CHENG_XLSX, sheet_name=CHENG_COMPOSITE_SHEET)
    age_col = find_column(raw.columns, "age")
    value_col = find_column(raw.columns, "18")
    out = pd.DataFrame(
        {
            "age_ka": pd.to_numeric(raw[age_col], errors="coerce"),
            "d18o": pd.to_numeric(raw[value_col], errors="coerce"),
        }
    )
    out = out.dropna(subset=["age_ka", "d18o"])
    out = out[(out["age_ka"] >= ANALYSIS_START_KA) & (out["age_ka"] <= ANALYSIS_END_KA)]
    out = out.groupby("age_ka", as_index=False)["d18o"].mean().sort_values("age_ka")

    anchors = pd.DataFrame({"age_ka": [ANALYSIS_START_KA, ANALYSIS_END_KA], "d18o": [np.nan, np.nan]})
    out = pd.concat([out, anchors], ignore_index=True)
    out = out.sort_values("age_ka").drop_duplicates("age_ka", keep="first").reset_index(drop=True)
    out["age_error_2sigma_kyr"] = age_error_2sigma_kyr(out["age_ka"].to_numpy(dtype=float))
    out["age_sigma_kyr"] = out["age_error_2sigma_kyr"] / 2.0
    out["source"] = relative_path(CHENG_XLSX)
    out["source_sheet"] = CHENG_COMPOSITE_SHEET
    return out


def load_best_match_uncertainty() -> pd.DataFrame:
    raw = pd.read_csv(BEST_MATCH_UNCERTAINTY_CSV)
    out = pd.DataFrame(
        {
            "age_ka": pd.to_numeric(raw["composite_age_ka"], errors="coerce"),
            "age_uncertainty_2sigma_kyr": pd.to_numeric(
                raw["age_uncertainty_2sigma_ka"],
                errors="coerce",
            ),
            "uncertainty_status": raw.get("uncertainty_status", "unknown"),
            "uncertainty_method": raw.get("uncertainty_method", "unknown"),
        }
    )
    out = out.dropna(subset=["age_ka", "age_uncertainty_2sigma_kyr"])
    out = out[
        (out["age_ka"] >= ANALYSIS_START_KA)
        & (out["age_ka"] <= ANALYSIS_END_KA)
        & (out["age_uncertainty_2sigma_kyr"] > 0.0)
    ].copy()
    out["source"] = relative_path(BEST_MATCH_UNCERTAINTY_CSV)
    return out.sort_values("age_ka").reset_index(drop=True)


def load_event_catalogues(error_series: pd.DataFrame) -> pd.DataFrame:
    rows = []
    error_age = error_series["age_ka"].to_numpy(dtype=float)
    error_2sigma = error_series["age_error_2sigma_kyr"].to_numpy(dtype=float)

    for event_type, settings in EVENT_SETTINGS.items():
        raw = pd.read_csv(settings["path"], encoding="utf-8-sig")
        if "start_time_ka_BP" not in raw.columns:
            raise ValueError(f"{settings['path']} must contain start_time_ka_BP.")

        ages = pd.to_numeric(raw["start_time_ka_BP"], errors="coerce")
        frame = raw.copy()
        frame["published_start_age_ka"] = ages
        frame = frame.dropna(subset=["published_start_age_ka"])
        frame = frame[
            (frame["published_start_age_ka"] >= ANALYSIS_START_KA)
            & (frame["published_start_age_ka"] <= ANALYSIS_END_KA)
        ].copy()
        frame = frame.sort_values("published_start_age_ka").reset_index(drop=True)
        frame["event_index"] = np.arange(1, len(frame) + 1)
        frame["event_type"] = event_type
        frame["event_label"] = settings["label"]
        frame["event_short_label"] = settings["short_label"]
        frame["color"] = settings["color"]
        frame["source"] = relative_path(settings["path"])
        frame["age_error_2sigma_kyr"] = np.interp(
            frame["published_start_age_ka"].to_numpy(dtype=float),
            error_age,
            error_2sigma,
        )
        frame["age_sigma_kyr"] = frame["age_error_2sigma_kyr"] / 2.0
        rows.append(frame)

    events = pd.concat(rows, ignore_index=True)
    bounds = []
    for event_type, group in events.groupby("event_type", sort=False):
        ages = group["published_start_age_ka"].to_numpy(dtype=float)
        error = group["age_error_2sigma_kyr"].to_numpy(dtype=float)
        sigma = group["age_sigma_kyr"].to_numpy(dtype=float)
        lower_mid = np.r_[ANALYSIS_START_KA, 0.5 * (ages[:-1] + ages[1:])]
        upper_mid = np.r_[0.5 * (ages[:-1] + ages[1:]), ANALYSIS_END_KA]
        lower = np.maximum.reduce([np.full_like(ages, ANALYSIS_START_KA), ages - error, lower_mid])
        upper = np.minimum.reduce([np.full_like(ages, ANALYSIS_END_KA), ages + error, upper_mid])
        if np.any(lower >= upper):
            bad = np.where(lower >= upper)[0][:5] + 1
            raise ValueError(f"Invalid sampling bounds for {event_type}, event indices {bad}.")
        a = (lower - ages) / sigma
        b = (upper - ages) / sigma
        mass_kept = norm.cdf(b) - norm.cdf(a)
        group_bounds = pd.DataFrame(
            {
                "event_type": event_type,
                "event_index": group["event_index"].to_numpy(dtype=int),
                "sampling_lower_bound_ka": lower,
                "sampling_upper_bound_ka": upper,
                "sampling_bound_width_ka": upper - lower,
                "truncated_gaussian_mass_kept": mass_kept,
                "rank_lower_midpoint_ka": lower_mid,
                "rank_upper_midpoint_ka": upper_mid,
            }
        )
        bounds.append(group_bounds)

    bound_frame = pd.concat(bounds, ignore_index=True)
    events = events.merge(bound_frame, on=["event_type", "event_index"], how="left")
    return events


def sample_rank_preserving_catalogues(
    event_uncertainties: pd.DataFrame,
    *,
    n_realizations: int,
    seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(seed)
    sample_frames = []
    diagnostic_rows = []

    for event_type, group in event_uncertainties.groupby("event_type", sort=False):
        group = group.sort_values("event_index").reset_index(drop=True)
        published = group["published_start_age_ka"].to_numpy(dtype=float)
        sigma = group["age_sigma_kyr"].to_numpy(dtype=float)
        lower = group["sampling_lower_bound_ka"].to_numpy(dtype=float)
        upper = group["sampling_upper_bound_ka"].to_numpy(dtype=float)
        a = (lower - published) / sigma
        b = (upper - published) / sigma
        samples = truncnorm.rvs(
            a=a[None, :],
            b=b[None, :],
            loc=published[None, :],
            scale=sigma[None, :],
            size=(n_realizations, len(group)),
            random_state=rng,
        )
        order_violations = np.sum(np.diff(samples, axis=1) < 0.0, axis=1)
        if np.any(order_violations):
            raise RuntimeError(f"Rank-preserving sampler produced crossings for {event_type}.")

        wide = pd.DataFrame(samples)
        wide.insert(0, "realization_id", np.arange(1, n_realizations + 1))
        long = wide.melt(id_vars="realization_id", var_name="event_zero_index", value_name="sampled_age_ka")
        long["event_index"] = long["event_zero_index"].astype(int) + 1
        long = long.drop(columns=["event_zero_index"])
        long["event_type"] = event_type
        long = long.merge(
            group[
                [
                    "event_index",
                    "event_label",
                    "event_short_label",
                    "published_start_age_ka",
                    "age_error_2sigma_kyr",
                    "age_sigma_kyr",
                    "sampling_lower_bound_ka",
                    "sampling_upper_bound_ka",
                    "truncated_gaussian_mass_kept",
                ]
            ],
            on="event_index",
            how="left",
        )
        long["sampled_minus_published_ka"] = (
            long["sampled_age_ka"] - long["published_start_age_ka"]
        )
        sample_frames.append(long)

        sample_mean = samples.mean(axis=0)
        sample_std = samples.std(axis=0, ddof=1)
        q025, q500, q975 = np.quantile(samples, [0.025, 0.5, 0.975], axis=0)
        diag = group[
            [
                "event_type",
                "event_label",
                "event_short_label",
                "event_index",
                "published_start_age_ka",
                "age_error_2sigma_kyr",
                "age_sigma_kyr",
                "sampling_lower_bound_ka",
                "sampling_upper_bound_ka",
                "truncated_gaussian_mass_kept",
            ]
        ].copy()
        diag["sample_mean_age_ka"] = sample_mean
        diag["sample_std_age_ka"] = sample_std
        diag["sample_q025_age_ka"] = q025
        diag["sample_q500_age_ka"] = q500
        diag["sample_q975_age_ka"] = q975
        diag["mean_minus_published_ka"] = sample_mean - published
        diag["std_over_requested_sigma"] = sample_std / sigma
        diagnostic_rows.append(diag)

    samples_long = pd.concat(sample_frames, ignore_index=True)
    diagnostics = pd.concat(diagnostic_rows, ignore_index=True)
    return samples_long.sort_values(["realization_id", "event_type", "event_index"]).reset_index(drop=True), diagnostics


def precompute_lohmann_nulls(
    event_uncertainties: pd.DataFrame,
    *,
    n_null: int,
    seed: int,
    window_ka: float,
    grid_step_ka: float,
) -> dict[str, LohmannNull]:
    duration_ka = ANALYSIS_END_KA - ANALYSIS_START_KA
    centers = lohmann.centered_time_grid(
        ANALYSIS_START_KA,
        ANALYSIS_END_KA,
        window_ka,
        grid_step_ka,
    )
    out: dict[str, LohmannNull] = {}

    for offset, (event_type, group) in enumerate(event_uncertainties.groupby("event_type", sort=False)):
        n_events = int(len(group))
        rate_per_kyr = n_events / duration_ka
        expected_count = rate_per_kyr * window_ka
        conditional_es = np.empty(n_null, dtype=float)
        poisson_es = np.empty(n_null, dtype=float)
        rng = np.random.default_rng(seed + 10_000 + offset * 101)
        for i in range(n_null):
            conditional_events = np.sort(
                rng.uniform(ANALYSIS_START_KA, ANALYSIS_END_KA, size=n_events)
            )
            conditional_counts = lohmann.moving_event_counts(conditional_events, centers, window_ka)
            conditional_es[i] = lohmann.rms_deviation(conditional_counts, expected_count)

            n_poisson = int(rng.poisson(rate_per_kyr * duration_ka))
            poisson_events = np.sort(
                rng.uniform(ANALYSIS_START_KA, ANALYSIS_END_KA, size=n_poisson)
            )
            poisson_counts = lohmann.moving_event_counts(poisson_events, centers, window_ka)
            poisson_es[i] = lohmann.rms_deviation(poisson_counts, expected_count)

        out[event_type] = LohmannNull(
            event_type=event_type,
            n_events=n_events,
            centers_ka=centers,
            expected_count=expected_count,
            conditional_es_samples=conditional_es,
            poisson_es_samples=poisson_es,
        )
    return out


def build_hazard_base() -> tuple[pd.DataFrame, np.ndarray]:
    edges = hazard.make_bin_edges()
    centers = 0.5 * (edges[:-1] + edges[1:])
    dt = np.diff(edges)
    lr04_scaled, lr04_info = hazard.load_lr04(centers)
    co2_scaled, co2_info = hazard.load_co2(centers)
    phase_table, _ = hazard.build_precession_phase(centers)
    base = pd.DataFrame(
        {
            "bin_start_ka": edges[:-1],
            "bin_end_ka": edges[1:],
            "bin_center_ka": centers,
            "dt_ka": dt,
            "lr04": lr04_info["raw"],
            "lr04_scaled": lr04_scaled,
            "co2": co2_info["raw"],
            "co2_scaled": co2_scaled,
        }
    )
    base = pd.concat([base, phase_table.drop(columns=["age_ka"])], axis=1)
    return base, edges


def lohmann_metrics(ages: np.ndarray, null: LohmannNull, window_ka: float) -> dict[str, float]:
    observed_counts = lohmann.moving_event_counts(ages, null.centers_ka, window_ka)
    observed_es = lohmann.rms_deviation(observed_counts, null.expected_count)
    return {
        "lohmann_event_count_ES": observed_es,
        "lohmann_poisson_ES_p_value": lohmann.p_value_greater_or_equal(
            null.poisson_es_samples,
            observed_es,
        ),
        "lohmann_fixed_n_ES_p_value": lohmann.p_value_greater_or_equal(
            null.conditional_es_samples,
            observed_es,
        ),
    }


def rayleigh_precession_metrics(
    ages: np.ndarray,
    phase_product: rayleigh.OrbitalPhase,
) -> dict[str, float]:
    phase_unwrapped, extrapolated = rayleigh.interpolate_unwrapped_phase(
        ages,
        phase_product.extrema["age_ka"].to_numpy(dtype=float),
        phase_product.extrema["anchor_phase_unwrapped_rad"].to_numpy(dtype=float),
    )
    phase = np.mod(phase_unwrapped, 2.0 * np.pi)
    used = phase[~extrapolated]
    result = rayleigh.rayleigh_test(used)
    return {
        "rayleigh_pre_n_phase_events_used": int(result["n_phase_events_used"]),
        "rayleigh_pre_n_extrapolated": int(np.sum(extrapolated)),
        "rayleigh_pre_mean_phase_deg": float(result["mean_phase_deg"]),
        "rayleigh_pre_mean_resultant_length": float(result["mean_resultant_length"]),
        "rayleigh_pre_z": float(result["rayleigh_z"]),
        "rayleigh_pre_p_value": float(result["rayleigh_p"]),
    }


def full_hazard_metrics(
    ages: np.ndarray,
    *,
    event_type: str,
    event_label: str,
    base_binned: pd.DataFrame,
    bin_edges: np.ndarray,
) -> dict[str, float | bool]:
    frame = base_binned.copy()
    counts, _ = np.histogram(ages, bins=bin_edges)
    frame.insert(0, "dataset_id", event_type)
    frame.insert(1, "dataset_label", event_label)
    frame["event_count"] = counts.astype(int)
    frame["n_events_total"] = int(len(ages))
    frame["source"] = "age_uncertainty_realization"
    stationary = hazard.fit_poisson_model(frame, "stationary", (), "Stationary")
    climate = hazard.fit_poisson_model(
        frame,
        "climate_lr04_co2",
        ("lr04_scaled", "co2_scaled"),
        "LR04 + CO2",
    )
    full = hazard.fit_poisson_model(frame, FULL_MODEL_ID, FULL_MODEL_TERMS, FULL_MODEL_LABEL)
    full_lr_stat = 2.0 * (full.log_likelihood - stationary.log_likelihood)
    full_p_value = float(
        chi2.sf(max(full_lr_stat, 0.0), len(full.beta) - len(stationary.beta))
    )
    phase_lr_stat = 2.0 * (full.log_likelihood - climate.log_likelihood)
    phase_p_value = float(
        chi2.sf(max(phase_lr_stat, 0.0), len(full.beta) - len(climate.beta))
    )
    beta_map = dict(zip(full.terms, full.beta[1:]))
    b_sin = float(beta_map["pre_phase_sin"])
    b_cos = float(beta_map["pre_phase_cos"])
    amplitude = float(np.hypot(b_sin, b_cos))
    preferred = float(np.mod(np.arctan2(b_sin, b_cos), 2.0 * np.pi))
    return {
        "precession_phase_after_lr04_co2_LR_statistic": phase_lr_stat,
        "precession_phase_after_lr04_co2_LR_p_value": phase_p_value,
        "precession_phase_after_lr04_co2_delta_AICc": full.aicc - climate.aicc,
        "full_vs_stationary_LR_statistic": full_lr_stat,
        "full_vs_stationary_LR_p_value": full_p_value,
        "full_vs_stationary_delta_AICc": full.aicc - stationary.aicc,
        "full_model_converged": bool(full.converged),
        "full_model_log_likelihood": full.log_likelihood,
        "climate_model_log_likelihood": climate.log_likelihood,
        "stationary_log_likelihood": stationary.log_likelihood,
        "full_model_pre_phase_preferred_deg": float(np.degrees(preferred)),
        "full_model_pre_phase_rate_ratio_max_vs_min": float(np.exp(2.0 * amplitude)),
    }


def run_sensitivity_experiments(
    samples_long: pd.DataFrame,
    event_uncertainties: pd.DataFrame,
    *,
    n_realizations: int,
    n_lohmann_null: int,
    seed: int,
    window_ka: float,
    grid_step_ka: float,
    progress_every: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    lohmann_nulls = precompute_lohmann_nulls(
        event_uncertainties,
        n_null=n_lohmann_null,
        seed=seed,
        window_ka=window_ka,
        grid_step_ka=grid_step_ka,
    )
    pre_phase = rayleigh.build_phase_series("pre", rayleigh.DRIVER_SETTINGS["pre"])
    base_binned, bin_edges = build_hazard_base()

    baseline_rows = []
    for event_type, group in event_uncertainties.groupby("event_type", sort=False):
        ages = group.sort_values("event_index")["published_start_age_ka"].to_numpy(dtype=float)
        event_label = str(group["event_label"].iloc[0])
        row = {
            "realization_id": 0,
            "catalogue_kind": "published_catalogue",
            "event_type": event_type,
            "event_label": event_label,
            "n_events": int(len(ages)),
        }
        row.update(lohmann_metrics(ages, lohmann_nulls[event_type], window_ka))
        row.update(rayleigh_precession_metrics(ages, pre_phase))
        row.update(
            full_hazard_metrics(
                ages,
                event_type=event_type,
                event_label=event_label,
                base_binned=base_binned,
                bin_edges=bin_edges,
            )
        )
        baseline_rows.append(row)

    rows = []
    for realization_id, realization in samples_long.groupby("realization_id", sort=True):
        if progress_every > 0 and (int(realization_id) == 1 or int(realization_id) % progress_every == 0):
            print(f"Running realization {int(realization_id)}/{n_realizations}...")
        for event_type, group in realization.groupby("event_type", sort=False):
            group = group.sort_values("event_index")
            ages = group["sampled_age_ka"].to_numpy(dtype=float)
            event_label = str(group["event_label"].iloc[0])
            row = {
                "realization_id": int(realization_id),
                "catalogue_kind": "age_randomized",
                "event_type": event_type,
                "event_label": event_label,
                "n_events": int(len(ages)),
            }
            row.update(lohmann_metrics(ages, lohmann_nulls[event_type], window_ka))
            row.update(rayleigh_precession_metrics(ages, pre_phase))
            row.update(
                full_hazard_metrics(
                    ages,
                    event_type=event_type,
                    event_label=event_label,
                    base_binned=base_binned,
                    bin_edges=bin_edges,
                )
            )
            rows.append(row)

    baseline = pd.DataFrame(baseline_rows)
    results = pd.concat([baseline, pd.DataFrame(rows)], ignore_index=True)
    return results, baseline


def build_summary(results: pd.DataFrame) -> pd.DataFrame:
    metric_map = {
        "lohmann_poisson_ES_p_value": "Lohmann Poisson null",
        "lohmann_fixed_n_ES_p_value": "Lohmann fixed-N null",
        "rayleigh_pre_p_value": "Rayleigh precession phase",
        "precession_phase_after_lr04_co2_LR_p_value": "Precession phase after LR04+CO2",
        "full_vs_stationary_LR_p_value": "Poisson full vs stationary",
    }
    rows = []
    randomized = results[results["catalogue_kind"].eq("age_randomized")]
    baseline = results[results["catalogue_kind"].eq("published_catalogue")]
    for event_type, group in randomized.groupby("event_type", sort=False):
        base_group = baseline[baseline["event_type"].eq(event_type)]
        for metric_id, label in metric_map.items():
            values = group[metric_id].to_numpy(dtype=float)
            values = values[np.isfinite(values)]
            base_value = float(base_group[metric_id].iloc[0])
            rows.append(
                {
                    "event_type": event_type,
                    "event_label": str(group["event_label"].iloc[0]),
                    "metric_id": metric_id,
                    "metric_label": label,
                    "n_realizations": int(values.size),
                    "published_catalogue_p_value": base_value,
                    "p_value_median": float(np.median(values)),
                    "p_value_q025": float(np.quantile(values, 0.025)),
                    "p_value_q05": float(np.quantile(values, 0.05)),
                    "p_value_q95": float(np.quantile(values, 0.95)),
                    "p_value_q975": float(np.quantile(values, 0.975)),
                    "p_value_min": float(np.min(values)),
                    "p_value_max": float(np.max(values)),
                    "significant_fraction_p_lt_0p05": float(np.mean(values < 0.05)),
                    "published_catalogue_significant_p_lt_0p05": bool(base_value < 0.05),
                }
            )
    return pd.DataFrame(rows)


def write_outputs(
    *,
    error_series: pd.DataFrame,
    best_match_uncertainty: pd.DataFrame,
    event_uncertainties: pd.DataFrame,
    samples_long: pd.DataFrame,
    sampling_diagnostics: pd.DataFrame,
    results: pd.DataFrame,
    summary: pd.DataFrame,
    parameters: dict,
) -> None:
    ensure_dir(OUT_DATA_DIR)
    error_series.to_csv(OUT_DATA_DIR / "composite_exponential_age_error_series.csv", index=False)
    best_match_uncertainty.to_csv(
        OUT_DATA_DIR / "best_match_age_uncertainty_points_for_fig01.csv",
        index=False,
    )
    event_uncertainties.to_csv(OUT_DATA_DIR / "event_age_uncertainties.csv", index=False)
    samples_long.to_csv(OUT_DATA_DIR / "randomized_event_catalogues.csv", index=False)
    sampling_diagnostics.to_csv(OUT_DATA_DIR / "random_sampling_diagnostics_by_event.csv", index=False)
    results.to_csv(OUT_DATA_DIR / "age_uncertainty_core_experiment_results.csv", index=False)
    summary.to_csv(OUT_DATA_DIR / "age_uncertainty_core_experiment_summary.csv", index=False)
    pd.DataFrame([parameters]).to_csv(OUT_DATA_DIR / "parameters.csv", index=False)


def plot_error_model(
    error_series: pd.DataFrame,
    best_match_uncertainty: pd.DataFrame,
    write_pdf: bool,
) -> None:
    fig, (ax1, ax2) = plt.subplots(
        2,
        1,
        figsize=(10.4, 7.0),
        sharex=True,
        gridspec_kw={"height_ratios": [1.25, 1.0]},
    )
    observed = error_series.dropna(subset=["d18o"])
    ax1.plot(observed["age_ka"], observed["d18o"], color="#777777", lw=0.8, alpha=0.75)
    ax1.set_ylabel("Cheng composite d18O")
    ax1.invert_yaxis()
    ax1.set_title("Cheng composite record and approximate age-uncertainty envelope", loc="left")

    ax2.scatter(
        best_match_uncertainty["age_ka"],
        best_match_uncertainty["age_uncertainty_2sigma_kyr"],
        s=8,
        color="#4d4d4d",
        alpha=0.22,
        linewidth=0,
        label="Best-match inferred 2-sigma uncertainty",
    )
    ax2.plot(
        error_series["age_ka"],
        error_series["age_error_2sigma_kyr"],
        color="#b2182b",
        lw=2.0,
        label="Exponential envelope used here",
    )
    ax2.set_xlabel("Age (ka BP)")
    ax2.set_ylabel("2-sigma age uncertainty (kyr)")
    ax2.set_ylim(bottom=0.0)
    ax2.legend(loc="upper left", frameon=False)
    fig.tight_layout()
    save_figure(fig, "fig01_composite_age_error_model", write_pdf)


def plot_event_uncertainties(event_uncertainties: pd.DataFrame, error_series: pd.DataFrame, write_pdf: bool) -> None:
    fig, axes = plt.subplots(2, 1, figsize=(10.2, 7.2), sharex=True)
    for ax in axes:
        ax.plot(
            error_series["age_ka"],
            error_series["age_error_2sigma_kyr"],
            color="#bdbdbd",
            lw=1.5,
            zorder=1,
            label="Composite envelope",
        )
    for ax, (event_type, group) in zip(axes, event_uncertainties.groupby("event_type", sort=False)):
        settings = EVENT_SETTINGS[event_type]
        ax.scatter(
            group["published_start_age_ka"],
            group["age_error_2sigma_kyr"],
            s=22,
            color=settings["color"],
            edgecolor="white",
            linewidth=0.35,
            zorder=3,
            label=settings["label"],
        )
        ax.set_ylabel("2-sigma range (kyr)")
        ax.set_title(f"{settings['label']}: interpolated event-age uncertainty", loc="left")
        ax.legend(frameon=False, loc="upper left")
    axes[-1].set_xlabel("Published start age (ka BP)")
    fig.tight_layout()
    save_figure(fig, "fig02_event_age_uncertainty_interpolation", write_pdf)


def plot_sampling_bounds(event_uncertainties: pd.DataFrame, write_pdf: bool) -> None:
    fig, axes = plt.subplots(2, 1, figsize=(10.4, 7.2), sharex=False)
    for ax, (event_type, group) in zip(axes, event_uncertainties.groupby("event_type", sort=False)):
        settings = EVENT_SETTINGS[event_type]
        x = group["event_index"].to_numpy(dtype=float)
        published = group["published_start_age_ka"].to_numpy(dtype=float)
        lower = group["sampling_lower_bound_ka"].to_numpy(dtype=float)
        upper = group["sampling_upper_bound_ka"].to_numpy(dtype=float)
        ax.fill_between(x, lower, upper, color=settings["color"], alpha=0.18, label="Sampling bounds")
        ax.plot(x, published, color=settings["color"], lw=1.4, label="Published starts")
        ax.set_ylabel("Age (ka BP)")
        ax.set_title(f"{settings['label']}: rank-preserving truncated Gaussian bounds", loc="left")
        ax.legend(frameon=False, loc="upper left")
    axes[-1].set_xlabel("Event index")
    fig.tight_layout()
    save_figure(fig, "fig03_rank_preserving_sampling_bounds", write_pdf)

    fig, ax = plt.subplots(figsize=(9.4, 4.2))
    for event_type, group in event_uncertainties.groupby("event_type", sort=False):
        settings = EVENT_SETTINGS[event_type]
        ax.scatter(
            group["published_start_age_ka"],
            group["truncated_gaussian_mass_kept"],
            color=settings["color"],
            s=24,
            alpha=0.85,
            label=settings["label"],
        )
    ax.axhline(0.9545, color="#555555", lw=1.0, ls="--", label="Untruncated +/-2 sigma mass")
    ax.set_xlabel("Published start age (ka BP)")
    ax.set_ylabel("Gaussian probability mass retained")
    ax.set_ylim(0.0, 1.03)
    ax.set_title("Sampling-bound diagnostic for each start", loc="left")
    ax.legend(frameon=False, loc="lower left", ncol=2)
    fig.tight_layout()
    save_figure(fig, "fig04_sampling_truncation_mass_check", write_pdf)


def plot_randomized_catalogue_checks(
    samples_long: pd.DataFrame,
    sampling_diagnostics: pd.DataFrame,
    write_pdf: bool,
) -> None:
    fig, axes = plt.subplots(2, 1, figsize=(10.4, 7.2), sharex=False)
    for ax, (event_type, group) in zip(axes, sampling_diagnostics.groupby("event_type", sort=False)):
        settings = EVENT_SETTINGS[event_type]
        x = group["event_index"].to_numpy(dtype=float)
        ax.fill_between(
            x,
            group["sample_q025_age_ka"],
            group["sample_q975_age_ka"],
            color=settings["color"],
            alpha=0.18,
            label="Sample 95% interval",
        )
        ax.plot(x, group["sample_q500_age_ka"], color=settings["color"], lw=1.2, label="Sample median")
        ax.plot(
            x,
            group["published_start_age_ka"],
            color="black",
            lw=0.9,
            alpha=0.7,
            label="Published catalogue",
        )
        ax.set_ylabel("Age (ka BP)")
        ax.set_title(f"{settings['label']}: randomized starts by event index", loc="left")
        ax.legend(frameon=False, loc="upper left")
    axes[-1].set_xlabel("Event index")
    fig.tight_layout()
    save_figure(fig, "fig05_randomized_start_ensemble_check", write_pdf)


def draw_age_offset_histogram(ax: plt.Axes, samples_long: pd.DataFrame) -> None:
    lower = float(samples_long["sampled_minus_published_ka"].quantile(0.005))
    upper = float(samples_long["sampled_minus_published_ka"].quantile(0.995))
    max_abs = max(abs(lower), abs(upper))
    bins = np.linspace(-max_abs, max_abs, 72)
    for event_type, group in samples_long.groupby("event_type", sort=False):
        settings = EVENT_SETTINGS[event_type]
        values = group["sampled_minus_published_ka"].to_numpy(dtype=float)
        ax.hist(
            values,
            bins=bins,
            density=True,
            histtype="stepfilled",
            alpha=0.22,
            color=settings["color"],
            edgecolor=settings["color"],
            linewidth=0.8,
            label=settings["label"],
        )
        ax.hist(
            values,
            bins=bins,
            density=True,
            histtype="step",
            color=settings["color"],
            linewidth=1.5,
        )
    ax.axvline(0.0, color="#333333", lw=1.0, ls="--")
    ax.set_xlabel("Sampled age minus published start age (kyr)")
    ax.set_ylabel("Density")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(frameon=False, loc="upper left")


def significance_fraction_pivot(summary: pd.DataFrame) -> pd.DataFrame:
    pivot = summary.pivot(
        index="event_label",
        columns="metric_label",
        values="significant_fraction_p_lt_0p05",
    )
    pivot = pivot.loc[[EVENT_SETTINGS[event_type]["label"] for event_type in EVENT_SETTINGS]]
    ordered_columns = [
        "Lohmann Poisson null",
        "Lohmann fixed-N null",
        "Rayleigh precession phase",
        "Precession phase after LR04+CO2",
        "Poisson full vs stationary",
    ]
    return pivot[ordered_columns]


def soft_significance_cmap() -> LinearSegmentedColormap:
    return LinearSegmentedColormap.from_list(
        "soft_significance",
        ["#fffaf0", "#eaf4ff", "#cfe8f7", "#9fd3e6"],
    )


def draw_significance_fraction_heatmap(
    ax: plt.Axes,
    summary: pd.DataFrame,
    *,
    add_colorbar: bool,
    fig: plt.Figure | None = None,
    cbar_ax: plt.Axes | None = None,
) -> None:
    pivot = significance_fraction_pivot(summary)
    im = ax.imshow(
        pivot.to_numpy(dtype=float),
        cmap=soft_significance_cmap(),
        vmin=0.0,
        vmax=1.0,
        aspect="auto",
    )
    ax.set_xticks(np.arange(len(pivot.columns)))
    ax.set_xticklabels(pivot.columns, rotation=18, ha="right")
    ax.set_yticks(np.arange(len(pivot.index)))
    ax.set_yticklabels(pivot.index)
    for i in range(pivot.shape[0]):
        for j in range(pivot.shape[1]):
            value = float(pivot.iloc[i, j])
            ax.text(j, i, f"{value:.2f}", ha="center", va="center", color="#1f2933", fontsize=10)
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.tick_params(length=0)
    if add_colorbar:
        if fig is None:
            fig = ax.figure
        cbar = fig.colorbar(im, ax=ax, cax=cbar_ax, pad=0.02)
        cbar.set_label("Significant fraction")


def plot_age_offset_distribution(samples_long: pd.DataFrame, write_pdf: bool) -> None:
    fig, ax = plt.subplots(figsize=(9.4, 4.2))
    draw_age_offset_histogram(ax, samples_long)
    fig.tight_layout()
    save_figure(fig, "fig06_randomized_age_offset_distribution", write_pdf)


def plot_result_heatmap(summary: pd.DataFrame, write_pdf: bool) -> None:
    fig, ax = plt.subplots(figsize=(11.2, 3.0))
    draw_significance_fraction_heatmap(ax, summary, add_colorbar=True, fig=fig)
    fig.tight_layout()
    save_figure(fig, "fig08_significance_fraction_heatmap", write_pdf)


def plot_combined_age_offsets_and_significance(
    samples_long: pd.DataFrame,
    summary: pd.DataFrame,
    write_pdf: bool,
) -> None:
    fig = plt.figure(figsize=(11.6, 7.0))
    grid = fig.add_gridspec(
        2,
        2,
        width_ratios=[1.0, 0.045],
        height_ratios=[1.15, 1.0],
        hspace=0.58,
        wspace=0.10,
    )
    ax_hist = fig.add_subplot(grid[0, 0])
    ax_blank = fig.add_subplot(grid[0, 1])
    ax_heat = fig.add_subplot(grid[1, 0])
    cax = fig.add_subplot(grid[1, 1])
    ax_blank.axis("off")

    draw_age_offset_histogram(ax_hist, samples_long)
    draw_significance_fraction_heatmap(
        ax_heat,
        summary,
        add_colorbar=True,
        fig=fig,
        cbar_ax=cax,
    )
    ax_hist.text(
        -0.065,
        1.04,
        "a",
        transform=ax_hist.transAxes,
        fontsize=14,
        fontweight="bold",
        va="bottom",
        ha="left",
    )
    ax_heat.text(
        -0.065,
        1.04,
        "b",
        transform=ax_heat.transAxes,
        fontsize=14,
        fontweight="bold",
        va="bottom",
        ha="left",
    )
    ensure_dir(OUT_FIG_DIR)
    fig.savefig(
        OUT_FIG_DIR / "fig09_age_offsets_and_significance_fraction.png",
        dpi=300,
        bbox_inches="tight",
    )
    fig.savefig(
        OUT_FIG_DIR / "fig09_age_offsets_and_significance_fraction.pdf",
        bbox_inches="tight",
    )
    plt.close(fig)


def make_plots(
    *,
    error_series: pd.DataFrame,
    best_match_uncertainty: pd.DataFrame,
    event_uncertainties: pd.DataFrame,
    samples_long: pd.DataFrame,
    sampling_diagnostics: pd.DataFrame,
    summary: pd.DataFrame,
    write_pdf: bool,
) -> None:
    plot_error_model(error_series, best_match_uncertainty, write_pdf)
    plot_event_uncertainties(event_uncertainties, error_series, write_pdf)
    plot_sampling_bounds(event_uncertainties, write_pdf)
    plot_randomized_catalogue_checks(samples_long, sampling_diagnostics, write_pdf)
    plot_age_offset_distribution(samples_long, write_pdf)
    plot_result_heatmap(summary, write_pdf)
    plot_combined_age_offsets_and_significance(samples_long, summary, write_pdf)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Composite age-uncertainty sensitivity for Rousseau 0.4-4 kyr start catalogues."
    )
    parser.add_argument("--n-realizations", type=int, default=DEFAULT_N_REALIZATIONS)
    parser.add_argument("--n-lohmann-null", type=int, default=DEFAULT_N_LOHMANN_NULL)
    parser.add_argument("--seed", type=int, default=DEFAULT_RANDOM_SEED)
    parser.add_argument("--window-ka", type=float, default=lohmann.EVENT_COUNT_WINDOW_KA)
    parser.add_argument("--grid-step-ka", type=float, default=lohmann.TIME_GRID_STEP_KA)
    parser.add_argument("--progress-every", type=int, default=25)
    parser.add_argument("--no-pdf", action="store_true", help="Only write PNG figures.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    error_series = load_cheng_composite_with_error()
    best_match_uncertainty = load_best_match_uncertainty()
    event_uncertainties = load_event_catalogues(error_series)
    samples_long, sampling_diagnostics = sample_rank_preserving_catalogues(
        event_uncertainties,
        n_realizations=args.n_realizations,
        seed=args.seed,
    )
    results, _ = run_sensitivity_experiments(
        samples_long,
        event_uncertainties,
        n_realizations=args.n_realizations,
        n_lohmann_null=args.n_lohmann_null,
        seed=args.seed,
        window_ka=args.window_ka,
        grid_step_ka=args.grid_step_ka,
        progress_every=args.progress_every,
    )
    summary = build_summary(results)
    parameters = {
        "run_name": RUN_NAME,
        "cheng_xlsx": relative_path(CHENG_XLSX),
        "cheng_sheet": CHENG_COMPOSITE_SHEET,
        "best_match_uncertainty_csv": relative_path(BEST_MATCH_UNCERTAINTY_CSV),
        "strong_csv": relative_path(STRONG_CSV),
        "weak_csv": relative_path(WEAK_CSV),
        "analysis_start_ka": ANALYSIS_START_KA,
        "analysis_end_ka": ANALYSIS_END_KA,
        "error_2sigma_start_kyr": ERROR_2SIGMA_START_KA,
        "error_2sigma_end_kyr": ERROR_2SIGMA_END_KA,
        "n_realizations": int(args.n_realizations),
        "n_lohmann_null": int(args.n_lohmann_null),
        "seed": int(args.seed),
        "lohmann_window_ka": float(args.window_ka),
        "lohmann_grid_step_ka": float(args.grid_step_ka),
        "sampling_distribution": "truncated Gaussian with sigma = 2sigma_range / 2",
        "order_constraint": "event-specific bounds include adjacent-age midpoints",
    }
    write_outputs(
        error_series=error_series,
        best_match_uncertainty=best_match_uncertainty,
        event_uncertainties=event_uncertainties,
        samples_long=samples_long,
        sampling_diagnostics=sampling_diagnostics,
        results=results,
        summary=summary,
        parameters=parameters,
    )
    make_plots(
        error_series=error_series,
        best_match_uncertainty=best_match_uncertainty,
        event_uncertainties=event_uncertainties,
        samples_long=samples_long,
        sampling_diagnostics=sampling_diagnostics,
        summary=summary,
        write_pdf=not args.no_pdf,
    )

    print("\nAge-uncertainty core experiment summary:")
    print(
        summary[
            [
                "event_label",
                "metric_label",
                "published_catalogue_p_value",
                "p_value_median",
                "p_value_q025",
                "p_value_q975",
                "significant_fraction_p_lt_0p05",
            ]
        ].to_string(index=False)
    )
    print(f"\nWrote tables to: {OUT_DATA_DIR.relative_to(PROJECT_ROOT)}")
    print(f"Wrote figures to: {OUT_FIG_DIR.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    main()
