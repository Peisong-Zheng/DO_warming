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
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.signal import find_peaks
from paper_figure_export import save_paper_pdf
from toolbox.model_stats import information_criteria, likelihood_gain, likelihood_ratio_p_value
from toolbox.poisson import (
    binned_poisson_loglik,
    design_matrix as make_design_matrix,
    fitted_rate_and_mu,
    negative_binned_poisson_loglik,
)


PROJECT_ROOT = Path(__file__).resolve().parent
RUN_NAME = "Bin_hazard_phase_poisson"
OUT_DATA_DIR = PROJECT_ROOT / "data" / "processed" / RUN_NAME
OUT_FIG_DIR = PROJECT_ROOT / "figures" / RUN_NAME

STRONG_CSV = PROJECT_ROOT / "data/raw/rousseau_2023_ks_0p4_4kyr_strong_monsoon_start_times.csv"
WEAK_CSV = PROJECT_ROOT / "data/raw/rousseau_2023_ks_0p4_4kyr_weak_monsoon_start_times.csv"
LR04_XLSX = PROJECT_ROOT / "data/raw/lr04.xlsx"
CO2_XLSX = PROJECT_ROOT / "data/raw/composite_co2.xlsx"
PRE_TXT = PROJECT_ROOT / "data/raw/pre_800_inter100.txt"

ANALYSIS_START_KA = 0.0
ANALYSIS_END_KA = 640.0
BIN_WIDTH_KA = 0.2

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

DATASET_SETTINGS = {
    "strong_monsoon_start": {
        "label": "Strong monsoon starts",
        "path": STRONG_CSV,
        "color": "#d95f02",
    },
    "weak_monsoon_start": {
        "label": "Weak monsoon starts",
        "path": WEAK_CSV,
        "color": "#6a3d9a",
    },
}

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


@dataclass
class EventDataset:
    """One event catalogue: strong or weak monsoon starts."""

    dataset_id: str
    label: str
    color: str
    ages_ka: np.ndarray
    source: str


@dataclass
class FittedPoissonModel:
    """Container for one fitted binned Poisson hazard model."""

    dataset_id: str
    dataset_label: str
    model_id: str
    model_label: str
    terms: tuple[str, ...]
    beta: np.ndarray
    converged: bool
    optimizer_message: str
    log_likelihood: float
    aic: float
    aicc: float
    bic: float
    fitted_rate_per_kyr: np.ndarray
    fitted_mu_per_bin: np.ndarray


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def save_figure(fig: plt.Figure, stem: str, write_pdf: bool) -> None:
    ensure_dir(OUT_FIG_DIR)
    fig.savefig(OUT_FIG_DIR / f"{stem}.png", dpi=300, bbox_inches="tight")
    if write_pdf:
        fig.savefig(OUT_FIG_DIR / f"{stem}.pdf", bbox_inches="tight")
        save_paper_pdf(fig, PROJECT_ROOT, stem)
    plt.close(fig)


def find_column(columns: pd.Index, *needles: str) -> str:
    normalized = {str(column).strip().lower(): column for column in columns}
    for key, column in normalized.items():
        if all(needle.lower() in key for needle in needles):
            return str(column)
    raise ValueError(f"Cannot find a column containing {needles}.")


def clean_series(age_ka: np.ndarray, value: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    age_ka = np.asarray(age_ka, dtype=float)
    value = np.asarray(value, dtype=float)
    ok = np.isfinite(age_ka) & np.isfinite(value)
    df = pd.DataFrame({"age_ka": age_ka[ok], "value": value[ok]})
    df = df.groupby("age_ka", as_index=False)["value"].mean().sort_values("age_ka")
    return df["age_ka"].to_numpy(dtype=float), df["value"].to_numpy(dtype=float)


def scale_to_zero_mean_range_one(values: np.ndarray) -> tuple[np.ndarray, float, float, float, float]:
    values = np.asarray(values, dtype=float)
    mean = float(np.nanmean(values))
    vmin = float(np.nanmin(values))
    vmax = float(np.nanmax(values))
    value_range = vmax - vmin
    if not np.isfinite(value_range) or value_range <= 0.0:
        return np.zeros_like(values, dtype=float), mean, vmin, vmax, 1.0
    return (values - mean) / value_range, mean, vmin, vmax, value_range


def load_events(path: Path, dataset_id: str, label: str, color: str) -> EventDataset:
    df = pd.read_csv(path, encoding="utf-8-sig")
    if "start_time_ka_BP" not in df.columns:
        raise ValueError(f"{path} must contain start_time_ka_BP.")
    ages = pd.to_numeric(df["start_time_ka_BP"], errors="coerce").dropna().to_numpy(dtype=float)
    ages = np.sort(ages[(ages >= ANALYSIS_START_KA) & (ages <= ANALYSIS_END_KA)])
    return EventDataset(
        dataset_id=dataset_id,
        label=label,
        color=color,
        ages_ka=ages,
        source=str(path.relative_to(PROJECT_ROOT)),
    )


def load_all_events() -> list[EventDataset]:
    return [
        load_events(
            settings["path"],
            dataset_id,
            settings["label"],
            settings["color"],
        )
        for dataset_id, settings in DATASET_SETTINGS.items()
    ]


def make_bin_edges() -> np.ndarray:
    edges = np.arange(ANALYSIS_START_KA, ANALYSIS_END_KA + BIN_WIDTH_KA / 2.0, BIN_WIDTH_KA)
    edges[-1] = ANALYSIS_END_KA
    return np.round(edges, 10)


def load_lr04(centers_ka: np.ndarray) -> tuple[np.ndarray, dict]:
    raw = pd.read_excel(LR04_XLSX)
    age_col = find_column(raw.columns, "time")
    value_col = find_column(raw.columns, "d18o")
    age, value = clean_series(raw[age_col].to_numpy(), raw[value_col].to_numpy())
    interpolated = np.interp(centers_ka, age, value)
    scaled, mean, vmin, vmax, value_range = scale_to_zero_mean_range_one(interpolated)
    meta = {
        "forcing_id": "lr04",
        "forcing_label": "LR04 benthic d18O",
        "source": str(LR04_XLSX.relative_to(PROJECT_ROOT)),
        "mean": mean,
        "min": vmin,
        "max": vmax,
        "range": value_range,
    }
    return scaled, {"raw": interpolated, "meta": meta}


def load_co2(centers_ka: np.ndarray) -> tuple[np.ndarray, dict]:
    raw = pd.read_excel(CO2_XLSX, sheet_name="Sheet2")
    age_col = find_column(raw.columns, "gasage")
    value_col = find_column(raw.columns, "co2")
    age, value = clean_series(raw[age_col].to_numpy() / 1000.0, raw[value_col].to_numpy())
    interpolated = np.interp(centers_ka, age, value)
    scaled, mean, vmin, vmax, value_range = scale_to_zero_mean_range_one(interpolated)
    meta = {
        "forcing_id": "co2",
        "forcing_label": "CO2",
        "source": str(CO2_XLSX.relative_to(PROJECT_ROOT)),
        "mean": mean,
        "min": vmin,
        "max": vmax,
        "range": value_range,
    }
    return scaled, {"raw": interpolated, "meta": meta}


def load_precession_series() -> pd.DataFrame:
    raw = pd.read_csv(PRE_TXT, sep=r"\s+", header=None, names=["age_raw_ka", "value"])
    age, value = clean_series(np.abs(raw["age_raw_ka"].to_numpy()), raw["value"].to_numpy())
    return pd.DataFrame({"age_ka": age, "precession_index": value})


def enforce_alternating_extrema(extrema: pd.DataFrame) -> pd.DataFrame:
    """Keep a clean min/max/min/max sequence for phase anchoring.

    Peak detection can occasionally return adjacent extrema of the same type
    when the orbital series has small-scale structure. Phase construction needs
    alternating half-cycles, so adjacent duplicate maxima/minima are collapsed
    by retaining the more extreme member.
    """

    rows: list[pd.Series] = []
    for _, row in extrema.sort_values("age_ka").iterrows():
        if not rows:
            rows.append(row.copy())
            continue
        prev = rows[-1]
        if row["extremum_type"] != prev["extremum_type"]:
            rows.append(row.copy())
            continue
        if row["extremum_type"] == "maximum":
            if float(row["precession_index"]) > float(prev["precession_index"]):
                rows[-1] = row.copy()
        else:
            if float(row["precession_index"]) < float(prev["precession_index"]):
                rows[-1] = row.copy()
    out = pd.DataFrame(rows).reset_index(drop=True)
    out["half_cycle_index"] = np.arange(len(out))
    return out


def build_precession_phase(centers_ka: np.ndarray) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Convert the raw precession index into a circular phase at bin centers.

    The conversion is intentionally based on extrema rather than on calendar
    angle. Minima of the precession index are assigned phase 0 and maxima are
    assigned phase pi. Between two successive extrema, phase is linearly
    interpolated in unwrapped radians. The wrapped phase is then represented as
    sin(theta) and cos(theta) for regression.
    """

    pre = load_precession_series()
    values = pre["precession_index"].to_numpy(dtype=float)
    max_idx, _ = find_peaks(values)
    min_idx, _ = find_peaks(-values)

    maxima = pre.iloc[max_idx].copy()
    maxima["extremum_type"] = "maximum"
    minima = pre.iloc[min_idx].copy()
    minima["extremum_type"] = "minimum"
    extrema = pd.concat([minima, maxima], ignore_index=True).sort_values("age_ka")
    extrema = enforce_alternating_extrema(extrema)
    if len(extrema) < 3:
        raise ValueError("Too few precession extrema detected.")

    # Anchor the first detected extremum, then add pi for every half-cycle.
    # The unwrapped phase is needed for interpolation; wrapping before
    # interpolation would create artificial jumps at 2*pi -> 0.
    first_phase = 0.0 if extrema.loc[0, "extremum_type"] == "minimum" else np.pi
    extrema["anchor_phase_unwrapped_rad"] = first_phase + np.arange(len(extrema), dtype=float) * np.pi
    extrema["anchor_phase_rad"] = np.mod(extrema["anchor_phase_unwrapped_rad"], 2.0 * np.pi)
    extrema["anchor_phase_deg"] = np.degrees(extrema["anchor_phase_rad"])

    phase_unwrapped = np.interp(
        centers_ka,
        extrema["age_ka"].to_numpy(dtype=float),
        extrema["anchor_phase_unwrapped_rad"].to_numpy(dtype=float),
    )
    phase_rad = np.mod(phase_unwrapped, 2.0 * np.pi)
    pre_at_center = np.interp(
        centers_ka,
        pre["age_ka"].to_numpy(dtype=float),
        pre["precession_index"].to_numpy(dtype=float),
    )
    phase_table = pd.DataFrame(
        {
            "age_ka": centers_ka,
            "precession_index": pre_at_center,
            "pre_phase_unwrapped_rad": phase_unwrapped,
            "pre_phase_rad": phase_rad,
            "pre_phase_deg": np.degrees(phase_rad),
            "pre_phase_sin": np.sin(phase_rad),
            "pre_phase_cos": np.cos(phase_rad),
        }
    )
    return phase_table, extrema


def build_binned_inputs(events: list[EventDataset]) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Build the model matrix shared by all fitted hazard models.

    Each event catalogue is histogrammed onto the same 0.2 kyr grid. The
    climate/orbital predictors are interpolated to bin centers once, then
    copied for strong and weak event datasets. LR04 and CO2 are range-scaled so
    their coefficients are comparable as effects per full observed range.
    """

    edges = make_bin_edges()
    centers = 0.5 * (edges[:-1] + edges[1:])
    dt = np.diff(edges)

    lr04_scaled, lr04_info = load_lr04(centers)
    co2_scaled, co2_info = load_co2(centers)
    phase_table, phase_extrema = build_precession_phase(centers)

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

    rows = []
    for dataset in events:
        counts, _ = np.histogram(dataset.ages_ka, bins=edges)
        dataset_frame = base.copy()
        dataset_frame.insert(0, "dataset_id", dataset.dataset_id)
        dataset_frame.insert(1, "dataset_label", dataset.label)
        dataset_frame["event_count"] = counts.astype(int)
        dataset_frame["n_events_total"] = int(len(dataset.ages_ka))
        dataset_frame["source"] = dataset.source
        rows.append(dataset_frame)

    scale_summary = pd.DataFrame(
        [
            lr04_info["meta"],
            co2_info["meta"],
            {
                "forcing_id": "pre_phase_sin",
                "forcing_label": "sin(precession phase)",
                "source": str(PRE_TXT.relative_to(PROJECT_ROOT)),
                "mean": float(np.nanmean(base["pre_phase_sin"])),
                "min": float(np.nanmin(base["pre_phase_sin"])),
                "max": float(np.nanmax(base["pre_phase_sin"])),
                "range": float(np.nanmax(base["pre_phase_sin"]) - np.nanmin(base["pre_phase_sin"])),
            },
            {
                "forcing_id": "pre_phase_cos",
                "forcing_label": "cos(precession phase)",
                "source": str(PRE_TXT.relative_to(PROJECT_ROOT)),
                "mean": float(np.nanmean(base["pre_phase_cos"])),
                "min": float(np.nanmin(base["pre_phase_cos"])),
                "max": float(np.nanmax(base["pre_phase_cos"])),
                "range": float(np.nanmax(base["pre_phase_cos"]) - np.nanmin(base["pre_phase_cos"])),
            },
        ]
    )
    return pd.concat(rows, ignore_index=True), scale_summary, phase_extrema


def design_matrix(frame: pd.DataFrame, terms: tuple[str, ...]) -> np.ndarray:
    return make_design_matrix(frame, terms)


def poisson_loglik(beta: np.ndarray, x: np.ndarray, y: np.ndarray, dt: np.ndarray) -> float:
    """Poisson log-likelihood for binned event counts."""

    return binned_poisson_loglik(beta, x, y, dt)


def negative_loglik(beta: np.ndarray, x: np.ndarray, y: np.ndarray, dt: np.ndarray) -> float:
    """Objective passed to scipy.optimize.minimize."""

    return negative_binned_poisson_loglik(beta, x, y, dt)


def fit_poisson_model(
    dataset_frame: pd.DataFrame,
    model_id: str,
    terms: tuple[str, ...],
    model_label: str,
) -> FittedPoissonModel:
    """Fit one binned Poisson hazard GLM for one event dataset.

    The stationary model has only an intercept, so its MLE is analytic: total
    events divided by total duration. Models with predictors are optimized with
    L-BFGS-B. The returned fitted_rate_per_kyr is lambda_i; fitted_mu_per_bin is
    lambda_i * dt_i, the expected number of events in each bin.
    """

    x = design_matrix(dataset_frame, terms)
    y = dataset_frame["event_count"].to_numpy(dtype=float)
    dt = dataset_frame["dt_ka"].to_numpy(dtype=float)
    duration = float(dt.sum())
    n_events = float(y.sum())
    n_obs = len(y)
    beta0 = np.zeros(1 + len(terms), dtype=float)
    beta0[0] = np.log(max(n_events / duration, 1e-12))

    if not terms:
        beta = beta0
        converged = True
        message = "analytic stationary MLE"
    else:
        bounds = [(-20.0, 5.0)] + [(-20.0, 20.0)] * len(terms)
        result = minimize(
            negative_loglik,
            beta0,
            args=(x, y, dt),
            method="L-BFGS-B",
            bounds=bounds,
            options={"maxiter": 4000, "ftol": 1e-11},
        )
        beta = result.x
        converged = bool(result.success)
        message = str(result.message)

    # Convert fitted log-rates back to rates per kyr and expected bin counts.
    rate, mu = fitted_rate_and_mu(beta, x, dt)
    log_likelihood = poisson_loglik(beta, x, y, dt)
    k = len(beta)
    criteria = information_criteria(log_likelihood, k, n_obs)

    return FittedPoissonModel(
        dataset_id=str(dataset_frame["dataset_id"].iloc[0]),
        dataset_label=str(dataset_frame["dataset_label"].iloc[0]),
        model_id=model_id,
        model_label=model_label,
        terms=terms,
        beta=beta,
        converged=converged,
        optimizer_message=message,
        log_likelihood=log_likelihood,
        aic=criteria["AIC"],
        aicc=criteria["AICc"],
        bic=criteria["BIC"],
        fitted_rate_per_kyr=rate,
        fitted_mu_per_bin=mu,
    )


def fit_all_models(binned_inputs: pd.DataFrame) -> list[FittedPoissonModel]:
    models = []
    for _, group in binned_inputs.groupby("dataset_id", sort=False):
        for model_id, terms, label in MODEL_SPECS:
            models.append(fit_poisson_model(group, model_id, terms, label))
    return models


def build_model_summary(models: list[FittedPoissonModel], binned_inputs: pd.DataFrame) -> pd.DataFrame:
    """Summarize fitted models and derive phase-response quantities."""

    rows = []
    for model in models:
        subset = binned_inputs[binned_inputs["dataset_id"].eq(model.dataset_id)]
        y = subset["event_count"].to_numpy(dtype=float)
        row = {
            "dataset_id": model.dataset_id,
            "dataset_label": model.dataset_label,
            "model_id": model.model_id,
            "model_label": model.model_label,
            "terms": "+".join(model.terms) if model.terms else "none",
            "bin_width_ka": BIN_WIDTH_KA,
            "n_bins": int(len(subset)),
            "n_events": int(y.sum()),
            "n_parameters": int(len(model.beta)),
            "converged": model.converged,
            "optimizer_message": model.optimizer_message,
            "log_likelihood": model.log_likelihood,
            "AIC": model.aic,
            "AICc": model.aicc,
            "BIC": model.bic,
            "expected_events": float(np.sum(model.fitted_mu_per_bin)),
            "max_fitted_rate_per_kyr": float(np.max(model.fitted_rate_per_kyr)),
            "mean_fitted_rate_per_kyr": float(np.mean(model.fitted_rate_per_kyr)),
        }
        for term, beta in zip(("intercept",) + model.terms, model.beta):
            row[f"beta_{term}"] = float(beta)
        if "pre_phase_sin" in model.terms and "pre_phase_cos" in model.terms:
            # For beta_s sin(theta) + beta_c cos(theta), the maximum occurs at
            # theta = atan2(beta_s, beta_c). The amplitude is the vector length
            # sqrt(beta_s^2 + beta_c^2), and the max/min rate ratio over the
            # cycle is exp(2 * amplitude).
            beta_map = dict(zip(model.terms, model.beta[1:]))
            b_sin = float(beta_map["pre_phase_sin"])
            b_cos = float(beta_map["pre_phase_cos"])
            amplitude = float(np.hypot(b_sin, b_cos))
            preferred = float(np.mod(np.arctan2(b_sin, b_cos), 2.0 * np.pi))
            row["pre_phase_amplitude"] = amplitude
            row["pre_phase_preferred_rad"] = preferred
            row["pre_phase_preferred_deg"] = float(np.degrees(preferred))
            row["pre_phase_rate_ratio_max_vs_min"] = float(np.exp(2.0 * amplitude))
        rows.append(row)

    summary = pd.DataFrame(rows)
    summary["delta_AICc"] = summary["AICc"] - summary.groupby("dataset_id")["AICc"].transform("min")
    summary["rank_AICc"] = summary.groupby("dataset_id")["AICc"].rank(method="first")
    return summary.sort_values(["dataset_id", "AICc"]).reset_index(drop=True)


def build_coefficient_table(models: list[FittedPoissonModel]) -> pd.DataFrame:
    rows = []
    for model in models:
        names = ("intercept",) + model.terms
        for name, beta in zip(names, model.beta):
            rows.append(
                {
                    "dataset_id": model.dataset_id,
                    "dataset_label": model.dataset_label,
                    "model_id": model.model_id,
                    "term": name,
                    "beta": float(beta),
                    "rate_ratio_per_unit": float(np.exp(beta)),
                }
            )
    return pd.DataFrame(rows)


def model_lookup(models: list[FittedPoissonModel]) -> dict[tuple[str, str], FittedPoissonModel]:
    return {(model.dataset_id, model.model_id): model for model in models}


def build_likelihood_tests(models: list[FittedPoissonModel]) -> pd.DataFrame:
    """Compare nested models with likelihood-ratio tests.

    The most important comparison is ``phase_after_climate``:

        reduced = LR04 + CO2
        full    = LR04 + CO2 + sin(pre_phase) + cos(pre_phase)

    Its p value asks whether adding the two phase coefficients improves the
    likelihood more than expected from two extra parameters under the nested
    chi-square approximation.
    """

    lookup = model_lookup(models)
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


def build_fitted_rate_table(models: list[FittedPoissonModel], binned_inputs: pd.DataFrame) -> pd.DataFrame:
    rows = []
    keep_cols = [
        "dataset_id",
        "dataset_label",
        "bin_start_ka",
        "bin_end_ka",
        "bin_center_ka",
        "dt_ka",
        "event_count",
    ]
    for model in models:
        subset = binned_inputs[binned_inputs["dataset_id"].eq(model.dataset_id)].copy()
        for idx, (_, row) in enumerate(subset.iterrows()):
            out = {col: row[col] for col in keep_cols}
            out.update(
                {
                    "model_id": model.model_id,
                    "model_label": model.model_label,
                    "lambda_per_kyr": float(model.fitted_rate_per_kyr[idx]),
                    "expected_events_per_bin": float(model.fitted_mu_per_bin[idx]),
                }
            )
            rows.append(out)
    return pd.DataFrame(rows)


def plot_inputs_and_rates(
    binned_inputs: pd.DataFrame,
    fitted_rates: pd.DataFrame,
    likelihood_tests: pd.DataFrame,
    write_pdf: bool,
) -> None:
    fig, axes = plt.subplots(5, 1, figsize=(13, 10), sharex=True, height_ratios=[1.0, 1.0, 1.0, 1.4, 1.4])
    base = binned_inputs[binned_inputs["dataset_id"].eq("strong_monsoon_start")]
    axes[0].plot(base["bin_center_ka"], base["lr04"], color="#1b9e77", lw=1.0)
    axes[0].set_ylabel("LR04")
    axes[0].invert_yaxis()

    axes[1].plot(base["bin_center_ka"], base["co2"], color="#d95f02", lw=1.0)
    axes[1].set_ylabel("CO2")

    pre_raw = load_precession_series()
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
    return f"{p_value:.2e}"


def plot_fitted_hazards_only(
    binned_inputs: pd.DataFrame,
    fitted_rates: pd.DataFrame,
    likelihood_tests: pd.DataFrame,
    summary: pd.DataFrame,
    write_pdf: bool,
) -> None:
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
    models: list[FittedPoissonModel],
    summary: pd.DataFrame,
    coefficients: pd.DataFrame,
    likelihood_tests: pd.DataFrame,
    fitted_rates: pd.DataFrame,
) -> None:
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


def run_analysis(write_pdf: bool) -> tuple[pd.DataFrame, pd.DataFrame]:
    ensure_dir(OUT_DATA_DIR)
    ensure_dir(OUT_FIG_DIR)
    events = load_all_events()
    binned_inputs, scale_summary, phase_extrema = build_binned_inputs(events)
    models = fit_all_models(binned_inputs)
    summary = build_model_summary(models, binned_inputs)
    coefficients = build_coefficient_table(models)
    likelihood_tests = build_likelihood_tests(models)
    fitted_rates = build_fitted_rate_table(models, binned_inputs)
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
    parser = argparse.ArgumentParser(
        description="Fit 0.2 kyr bin Poisson hazard models for Rousseau monsoon starts."
    )
    parser.add_argument("--no-pdf", action="store_true", help="Only write PNG figures.")
    return parser.parse_args()


def main() -> None:
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
