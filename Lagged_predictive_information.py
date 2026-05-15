"""
Lagged model-based transfer-entropy style analysis for Rousseau et al. (2023)
strong/weak monsoon starts.

This script asks two predictive-information questions on 0.2 kyr event bins:

1. Which driver, at which lag, provides the largest information gain about
   future strong/weak monsoon event occurrence?
2. After controlling LR04 and CO2, does precession phase still provide
   additional lagged predictive information?

Because the event series is sparse, the scan uses nested Poisson event-rate
models to estimate conditional predictive information. The reported quantity
is the log-likelihood gain:

    I_like = logL(full model) - logL(reduced model)

in nats or bits, with likelihood-ratio p values and Benjamini-Hochberg q
values across the scanned lags. Age increases into the past, so a positive lag
means the driver is sampled at age t + lag for an event bin centered at age t.
All lag comparisons are made on a common bin support that remains valid up to
MAX_LAG_KA, so best-lag rankings are not driven by changing old-edge coverage.

Method sketch
-------------
Classical transfer entropy asks whether the past of X improves prediction of
the future of Y after conditioning on the past of Y. Here Y is the binned
monsoon-start count series, and X is one lagged driver. Because Y is sparse,
the script estimates this idea with nested Poisson hazard models:

    reduced: log(lambda_i) = beta0 + Y_history_i + resolution_i
    full:    log(lambda_i) = beta0 + Y_history_i + resolution_i + X_{i, lag}

The TE-like information is the gain in fitted log-likelihood:

    I_like = logL(full) - logL(reduced).

It is reported in nats and in bits per bin/event.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import to_rgb
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from paper_figure_export import save_paper_pdf
from toolbox.model_stats import (
    bits_from_loglik_gain,
    information_criteria,
    likelihood_gain,
    likelihood_ratio_p_value,
)
from toolbox.poisson import (
    binned_poisson_loglik,
    design_matrix,
    fitted_rate_and_mu,
    negative_binned_poisson_loglik,
)

from Bin_hazard_phase_poisson import (
    ANALYSIS_END_KA,
    ANALYSIS_START_KA,
    BIN_WIDTH_KA,
    DATASET_SETTINGS,
    PROJECT_ROOT,
    build_binned_inputs,
    load_all_events,
)
import Predictive_information_model as predictive


RUN_NAME = "Lagged_predictive_information"
OUT_DATA_DIR = PROJECT_ROOT / "data" / "processed" / RUN_NAME
OUT_FIG_DIR = PROJECT_ROOT / "figures" / RUN_NAME

MAX_LAG_KA = 10.0
LAG_STEP_KA = 0.2
Y_HISTORY_WINDOW_KA = predictive.MAIN_HISTORY_WINDOW_KA
MIN_EVENTS_FOR_MODEL = 5
PLOT_MAX_LAG_KA = 10.0
CORE_COMPONENT_MAX_LAG_KA = 10.0
FDR_Q_THRESHOLD = 0.05

HISTORY_TERMS = predictive.BASELINE_TERMS

DRIVER_TERM_SPECS = {
    "lr04": {
        "label": "LR04",
        "color": "#1b9e77",
        "source_terms": ("lr04_scaled",),
    },
    "co2": {
        "label": "CO2",
        "color": "#d95f02",
        "source_terms": ("co2_scaled",),
    },
    "pre_phase": {
        "label": "Precession phase",
        "color": "#7570b3",
        "source_terms": ("pre_phase_sin", "pre_phase_cos"),
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


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def save_figure(fig: plt.Figure, stem: str, write_pdf: bool) -> None:
    ensure_dir(OUT_FIG_DIR)
    fig.savefig(OUT_FIG_DIR / f"{stem}.png", dpi=300, bbox_inches="tight")
    if write_pdf:
        fig.savefig(OUT_FIG_DIR / f"{stem}.pdf", bbox_inches="tight")
        save_paper_pdf(fig, PROJECT_ROOT, stem)
    plt.close(fig)


def lag_grid(max_lag_ka: float | None = None) -> np.ndarray:
    """All lags scanned by the predictive-information analysis."""

    upper = MAX_LAG_KA if max_lag_ka is None else float(max_lag_ka)
    return np.round(np.arange(0.0, upper + LAG_STEP_KA / 2.0, LAG_STEP_KA), 10)


def add_event_history_feature(binned: pd.DataFrame, max_lag_ka: float = MAX_LAG_KA) -> pd.DataFrame:
    """Add the event-process baseline used in every reduced model.

    The reduced model now matches the event-process-baseline predictive framework:
    same-type event history in the older ``(t, t + W]`` interval, with
    ``W = 5 kyr``, plus local Cheng composite sampling resolution. The common
    lag-support flag keeps only bins for which all candidate lags up to
    ``max_lag_ka`` can be evaluated, preventing larger lags from being compared
    on a smaller old-edge sample.

    In the Poisson models the history and resolution terms are ordinary fitted
    covariates, not offsets:

        log(lambda_i) = beta0 + beta_h history_i + beta_r resolution_i + ...

    This makes the lag scan ask whether a driver adds information after local
    event-process memory and uneven speleothem sampling have been conditioned
    on.
    """

    binned, _, _ = predictive.add_resolution_control(binned)
    binned = predictive.add_same_type_history(binned, Y_HISTORY_WINDOW_KA)
    frames = []
    for _, group in binned.groupby("dataset_id", sort=False):
        group = group.sort_values("bin_center_ka").copy()
        centers = group["bin_center_ka"].to_numpy(dtype=float)
        group["common_lag_support"] = (centers + max_lag_ka) <= (centers[-1] + 1e-9)
        frames.append(group)
    return pd.concat(frames, ignore_index=True)


def load_binned_inputs_with_history(max_lag_ka: float = MAX_LAG_KA) -> pd.DataFrame:
    """Load binned covariates and append history/resolution controls."""

    events = load_all_events()
    binned, _, _ = build_binned_inputs(events)
    return add_event_history_feature(binned, max_lag_ka=max_lag_ka)


def add_lagged_columns(
    frame: pd.DataFrame,
    lag_specs: dict[str, tuple[str, float]],
) -> pd.DataFrame:
    """Create lagged driver columns on the age grid.

    A positive lag means "sample the driver at an older age":

        X_lagged(t) = X(t + lag).

    Linear interpolation keeps the lag scan on the same 0.2 kyr event-bin grid.
    Values beyond the available oldest driver age are set to NaN and are later
    dropped by the model-fitting functions.
    """

    out = frame.sort_values("bin_center_ka").copy()
    centers = out["bin_center_ka"].to_numpy(dtype=float)
    max_age = float(centers.max())
    for new_col, (source_col, lag_ka) in lag_specs.items():
        target_age = centers + float(lag_ka)
        values = np.interp(
            target_age,
            centers,
            out[source_col].to_numpy(dtype=float),
            left=np.nan,
            right=np.nan,
        )
        values[target_age > max_age] = np.nan
        out[new_col] = values
    return out


def poisson_loglik(beta: np.ndarray, x: np.ndarray, y: np.ndarray, dt: np.ndarray) -> float:
    """Poisson log likelihood for the model-based information calculations."""

    return binned_poisson_loglik(beta, x, y, dt)


def negative_loglik(beta: np.ndarray, x: np.ndarray, y: np.ndarray, dt: np.ndarray) -> float:
    """Numerically safe optimizer objective."""

    return negative_binned_poisson_loglik(beta, x, y, dt)


def fit_poisson(frame: pd.DataFrame, terms: tuple[str, ...]) -> dict:
    """Fit one reduced or full Poisson hazard model on common support.

    The input frame may contain lagged columns. Rows are kept only if all model
    terms are present, the 5 kyr same-type history window is complete, and the
    bin is in the common support for the full lag scan. This makes
    log-likelihood gains comparable across all tested lags.

    This is an ordinary Poisson regression fit by maximum likelihood. The
    design matrix ``x`` contains the requested terms, for example
    same-type history plus sampling resolution in the reduced model and those
    terms plus ``X_lag`` in the full model. After optimization, the fitted
    parameters are substituted back into ``poisson_loglik`` to get the model log
    likelihood used by the predictive-information score.
    """

    required = ["event_count", "dt_ka", *terms]
    data = frame.dropna(subset=required).copy()
    history_complete_col = (
        "same_type_history_complete"
        if "same_type_history_complete" in data.columns
        else "event_history_complete"
    )
    data = data[data[history_complete_col].astype(bool)].copy()
    if "common_lag_support" in data.columns:
        data = data[data["common_lag_support"].astype(bool)].copy()
    y = data["event_count"].to_numpy(dtype=float)
    if y.sum() < MIN_EVENTS_FOR_MODEL:
        return {
            "ok": False,
            "message": f"Too few events after filtering: {y.sum():.0f}",
            "n_bins": int(len(data)),
            "n_events": int(y.sum()),
        }

    dt = data["dt_ka"].to_numpy(dtype=float)
    # Each entry in ``terms`` is a fitted predictor column. In the single-driver
    # scan this means reduced_terms=(same-type history, sampling resolution) and
    # full_terms=reduced_terms + lagged_driver_terms.
    x = design_matrix(data, terms)
    duration = float(dt.sum())
    beta0 = np.zeros(1 + len(terms), dtype=float)
    beta0[0] = np.log(max(y.sum() / duration, 1e-12))

    if not terms:
        beta = beta0
        converged = True
        message = "analytic stationary MLE"
    else:
        bounds = [(-20.0, 5.0)] + [(-20.0, 20.0)] * len(terms)
        # Maximum-likelihood fit: minimize negative log likelihood to obtain
        # beta_hat for this specific reduced or full model.
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

    # Evaluate logL at beta_hat. The nested comparison later subtracts the
    # reduced-model value from the full-model value.
    log_likelihood = poisson_loglik(beta, x, y, dt)
    n_obs = len(data)
    k = len(beta)
    criteria = information_criteria(log_likelihood, k, n_obs)
    rate, _ = fitted_rate_and_mu(beta, x, dt)
    return {
        "ok": True,
        "terms": terms,
        "beta": beta,
        "converged": converged,
        "message": message,
        "n_bins": int(n_obs),
        "n_events": int(y.sum()),
        "log_likelihood": log_likelihood,
        "AIC": criteria["AIC"],
        "AICc": criteria["AICc"],
        "BIC": criteria["BIC"],
        "expected_events": float(np.sum(rate * dt)),
        "mean_rate_per_kyr": float(np.mean(rate)),
        "max_rate_per_kyr": float(np.max(rate)),
    }


def compare_nested_models(
    frame: pd.DataFrame,
    reduced_terms: tuple[str, ...],
    full_terms: tuple[str, ...],
) -> tuple[dict, dict, dict]:
    """Compare reduced and full models and convert logL gain to information.

    This is the central TE-like calculation. The reduced model represents the
    conditioning set, such as Y history alone or Y history plus climate
    controls. The full model adds the lagged driver being tested. Their
    log-likelihood difference is the model-based predictive-information gain:

        ll_gain = logL(full) - logL(reduced).

    Both logL values are evaluated after fitting each model separately by
    maximum likelihood. Therefore this is a model-based predictive-information
    score: it measures how much the additional lagged predictor improves the
    optimized Poisson hazard likelihood beyond the conditioning variables.

    The same gain is also used for a likelihood-ratio test:

        LR = 2 * ll_gain,

    with df equal to the number of added coefficients.
    """

    required = tuple(dict.fromkeys((*reduced_terms, *full_terms)))
    data = frame.dropna(subset=["event_count", "dt_ka", *required]).copy()
    history_complete_col = (
        "same_type_history_complete"
        if "same_type_history_complete" in data.columns
        else "event_history_complete"
    )
    data = data[data[history_complete_col].astype(bool)].copy()
    if "common_lag_support" in data.columns:
        data = data[data["common_lag_support"].astype(bool)].copy()

    # Fit the two nested Poisson regressions on the exact same rows. For the
    # single-driver scan this corresponds to:
    #   reduced: log(lambda_i) = beta0 + beta_h * history_i + beta_r * resolution_i
    #   full:    reduced + gamma * X_lag_i
    # For precession phase, X_lag contributes two fitted terms: sin(theta_lag)
    # and cos(theta_lag).
    reduced = fit_poisson(data, reduced_terms)
    full = fit_poisson(data, full_terms)
    if not reduced.get("ok") or not full.get("ok"):
        comparison = {
            "ok": False,
            "n_bins": min(reduced.get("n_bins", 0), full.get("n_bins", 0)),
            "n_events": min(reduced.get("n_events", 0), full.get("n_events", 0)),
            "message": f"reduced={reduced.get('message')}; full={full.get('message')}",
        }
        return comparison, reduced, full

    df = len(full["beta"]) - len(reduced["beta"])
    # TE-like information gain in nats. The same quantity is converted to
    # bits/bin and bits/event below, and 2*ll_gain gives the LR statistic.
    ll_gain = likelihood_gain(full["log_likelihood"], reduced["log_likelihood"])
    lr_stat, p_value = likelihood_ratio_p_value(ll_gain, df)
    bits_per_bin = bits_from_loglik_gain(ll_gain, full["n_bins"])
    bits_per_event = bits_from_loglik_gain(ll_gain, full["n_events"])
    comparison = {
        "ok": True,
        "n_bins": full["n_bins"],
        "n_events": full["n_events"],
        "df": int(df),
        "loglik_reduced": reduced["log_likelihood"],
        "loglik_full": full["log_likelihood"],
        "ll_gain_nats": ll_gain,
        "info_nats_per_bin": ll_gain / full["n_bins"],
        "info_bits_per_bin": bits_per_bin,
        "info_bits_per_event": bits_per_event,
        "LR_statistic": lr_stat,
        "LR_p_value": p_value,
        "delta_AICc_full_minus_reduced": full["AICc"] - reduced["AICc"],
        "expected_events_reduced": reduced["expected_events"],
        "expected_events_full": full["expected_events"],
        "max_rate_full_per_kyr": full["max_rate_per_kyr"],
        "converged_reduced": reduced["converged"],
        "converged_full": full["converged"],
    }
    return comparison, reduced, full


def bh_q_values(p_values: np.ndarray) -> np.ndarray:
    """Benjamini-Hochberg FDR correction for a family of lag tests."""

    p = np.asarray(p_values, dtype=float)
    q = np.full_like(p, np.nan, dtype=float)
    valid = np.isfinite(p)
    if valid.sum() == 0:
        return q
    pv = p[valid]
    order = np.argsort(pv)
    ranked = pv[order]
    m = len(ranked)
    adjusted = ranked * m / np.arange(1, m + 1)
    adjusted = np.minimum.accumulate(adjusted[::-1])[::-1]
    adjusted = np.clip(adjusted, 0.0, 1.0)
    q_valid = np.empty_like(adjusted)
    q_valid[order] = adjusted
    q[valid] = q_valid
    return q


def add_lagged_driver_terms(dataset_frame: pd.DataFrame, driver_id: str, lag_ka: float) -> tuple[pd.DataFrame, tuple[str, ...]]:
    """Create the lagged columns needed for one driver at one lag.

    Scalar drivers such as LR04 and CO2 contribute one term. Circular
    precession phase contributes two terms, sin(theta_lag) and cos(theta_lag),
    so the model can learn any preferred phase without a discontinuity at
    0/2*pi.
    """

    specs = {}
    lagged_terms = []
    for source_col in DRIVER_TERM_SPECS[driver_id]["source_terms"]:
        new_col = f"{driver_id}_{source_col}_lag"
        specs[new_col] = (source_col, lag_ka)
        lagged_terms.append(new_col)
    return add_lagged_columns(dataset_frame, specs), tuple(lagged_terms)


def run_single_driver_scans(binned: pd.DataFrame) -> pd.DataFrame:
    """Scan each driver/lag against the event-process baseline.

    For every event dataset, lag, and driver:

        reduced = same-type history + sampling resolution
        full    = reduced + lagged_driver

    The output table therefore ranks drivers by the predictive information they
    add beyond recent event clustering and local sampling resolution.
    """

    rows = []
    for dataset_id, dataset_frame in binned.groupby("dataset_id", sort=False):
        for lag_ka in lag_grid():
            for driver_id, spec in DRIVER_TERM_SPECS.items():
                lagged, driver_terms = add_lagged_driver_terms(dataset_frame, driver_id, lag_ka)
                reduced_terms = HISTORY_TERMS
                full_terms = (*HISTORY_TERMS, *driver_terms)
                comparison, _, _ = compare_nested_models(lagged, reduced_terms, full_terms)
                row = {
                    "analysis_type": "single_driver_given_y_history",
                    "dataset_id": dataset_id,
                    "dataset_label": dataset_frame["dataset_label"].iloc[0],
                    "driver_id": driver_id,
                    "driver_label": spec["label"],
                    "lag_ka": float(lag_ka),
                    "reduced_terms": "+".join(reduced_terms),
                    "full_terms": "+".join(full_terms),
                }
                row.update(comparison)
                rows.append(row)
    out = pd.DataFrame(rows)
    out["LR_q_value_BH"] = np.nan
    ok = out["ok"].astype(bool)
    for keys, group in out[ok].groupby(["analysis_type", "dataset_id", "driver_id"], sort=False):
        out.loc[group.index, "LR_q_value_BH"] = bh_q_values(group["LR_p_value"].to_numpy(dtype=float))
    return out


def best_single_driver_lags(single: pd.DataFrame) -> pd.DataFrame:
    """Pick the lag with maximum bits/event for each driver and dataset."""

    ok = single[single["ok"].astype(bool)].copy()
    idx = ok.groupby(["dataset_id", "driver_id"])["info_bits_per_event"].idxmax()
    best = ok.loc[idx].copy().reset_index(drop=True)
    best = best[
        [
            "dataset_id",
            "dataset_label",
            "driver_id",
            "driver_label",
            "lag_ka",
            "info_bits_per_event",
            "info_bits_per_bin",
            "LR_p_value",
            "LR_q_value_BH",
            "delta_AICc_full_minus_reduced",
        ]
    ]
    best = best.rename(
        columns={
            "lag_ka": "best_lag_ka",
            "info_bits_per_event": "best_info_bits_per_event",
            "info_bits_per_bin": "best_info_bits_per_bin",
            "LR_p_value": "best_LR_p_value",
            "LR_q_value_BH": "best_LR_q_value_BH",
            "delta_AICc_full_minus_reduced": "best_delta_AICc_full_minus_reduced",
        }
    )
    return best.sort_values(["dataset_id", "driver_id"]).reset_index(drop=True)


def run_pre_after_climate_same_lag(binned: pd.DataFrame) -> pd.DataFrame:
    """Test precession phase after LR04 and CO2 at the same lag.

    This asks whether phase adds information that is not already captured by
    LR04 and CO2 when all three predictors are sampled at the same look-back
    time.
    """

    rows = []
    for dataset_id, dataset_frame in binned.groupby("dataset_id", sort=False):
        for lag_ka in lag_grid():
            lagged = add_lagged_columns(
                dataset_frame,
                {
                    "lr04_lag": ("lr04_scaled", lag_ka),
                    "co2_lag": ("co2_scaled", lag_ka),
                    "pre_phase_sin_lag": ("pre_phase_sin", lag_ka),
                    "pre_phase_cos_lag": ("pre_phase_cos", lag_ka),
                },
            )
            reduced_terms = (*HISTORY_TERMS, "lr04_lag", "co2_lag")
            full_terms = (*reduced_terms, "pre_phase_sin_lag", "pre_phase_cos_lag")
            comparison, _, _ = compare_nested_models(lagged, reduced_terms, full_terms)
            row = {
                "analysis_type": "pre_phase_after_lr04_co2_same_lag",
                "dataset_id": dataset_id,
                "dataset_label": dataset_frame["dataset_label"].iloc[0],
                "driver_id": "pre_phase",
                "driver_label": "Precession phase",
                "lag_ka": float(lag_ka),
                "lr04_control_lag_ka": float(lag_ka),
                "co2_control_lag_ka": float(lag_ka),
                "reduced_terms": "+".join(reduced_terms),
                "full_terms": "+".join(full_terms),
            }
            row.update(comparison)
            rows.append(row)
    out = pd.DataFrame(rows)
    out["LR_q_value_BH"] = np.nan
    ok = out["ok"].astype(bool)
    for _, group in out[ok].groupby(["analysis_type", "dataset_id"], sort=False):
        out.loc[group.index, "LR_q_value_BH"] = bh_q_values(group["LR_p_value"].to_numpy(dtype=float))
    return out


def run_pre_after_best_climate_lags(binned: pd.DataFrame, best_lags: pd.DataFrame) -> pd.DataFrame:
    """Test precession phase after LR04/CO2 are fixed at their best lags.

    This is the stricter conditional test. First, LR04 and CO2 each get their
    best single-driver lag. Then precession phase is scanned across its own
    lags. If phase still improves the model, the result is less likely to be an
    artifact of choosing a poor lag for the climate controls.
    """

    rows = []
    for dataset_id, dataset_frame in binned.groupby("dataset_id", sort=False):
        best_for_dataset = best_lags[best_lags["dataset_id"].eq(dataset_id)]
        lr04_lag = float(best_for_dataset[best_for_dataset["driver_id"].eq("lr04")]["best_lag_ka"].iloc[0])
        co2_lag = float(best_for_dataset[best_for_dataset["driver_id"].eq("co2")]["best_lag_ka"].iloc[0])
        for pre_lag in lag_grid():
            lagged = add_lagged_columns(
                dataset_frame,
                {
                    "lr04_best_lag": ("lr04_scaled", lr04_lag),
                    "co2_best_lag": ("co2_scaled", co2_lag),
                    "pre_phase_sin_lag": ("pre_phase_sin", pre_lag),
                    "pre_phase_cos_lag": ("pre_phase_cos", pre_lag),
                },
            )
            reduced_terms = (*HISTORY_TERMS, "lr04_best_lag", "co2_best_lag")
            full_terms = (*reduced_terms, "pre_phase_sin_lag", "pre_phase_cos_lag")
            comparison, _, _ = compare_nested_models(lagged, reduced_terms, full_terms)
            row = {
                "analysis_type": "pre_phase_after_best_lr04_co2_lags",
                "dataset_id": dataset_id,
                "dataset_label": dataset_frame["dataset_label"].iloc[0],
                "driver_id": "pre_phase",
                "driver_label": "Precession phase",
                "lag_ka": float(pre_lag),
                "lr04_control_lag_ka": lr04_lag,
                "co2_control_lag_ka": co2_lag,
                "reduced_terms": "+".join(reduced_terms),
                "full_terms": "+".join(full_terms),
            }
            row.update(comparison)
            rows.append(row)
    out = pd.DataFrame(rows)
    out["LR_q_value_BH"] = np.nan
    ok = out["ok"].astype(bool)
    for _, group in out[ok].groupby(["analysis_type", "dataset_id"], sort=False):
        out.loc[group.index, "LR_q_value_BH"] = bh_q_values(group["LR_p_value"].to_numpy(dtype=float))
    return out


def build_best_summary(single: pd.DataFrame, same_lag: pd.DataFrame, best_climate: pd.DataFrame) -> pd.DataFrame:
    """Collect best-lag rows from all scan families into one summary table."""

    pieces = [best_single_driver_lags(single)]
    for table in [same_lag, best_climate]:
        ok = table[table["ok"].astype(bool)].copy()
        idx = ok.groupby(["analysis_type", "dataset_id"])["info_bits_per_event"].idxmax()
        best = ok.loc[idx].copy()
        pieces.append(
            best[
                [
                    "analysis_type",
                    "dataset_id",
                    "dataset_label",
                    "driver_id",
                    "driver_label",
                    "lag_ka",
                    "lr04_control_lag_ka",
                    "co2_control_lag_ka",
                    "info_bits_per_event",
                    "info_bits_per_bin",
                    "LR_p_value",
                    "LR_q_value_BH",
                    "delta_AICc_full_minus_reduced",
                ]
            ].rename(
                columns={
                    "lag_ka": "best_lag_ka",
                    "info_bits_per_event": "best_info_bits_per_event",
                    "info_bits_per_bin": "best_info_bits_per_bin",
                    "LR_p_value": "best_LR_p_value",
                    "LR_q_value_BH": "best_LR_q_value_BH",
                    "delta_AICc_full_minus_reduced": "best_delta_AICc_full_minus_reduced",
                }
            )
        )
    out = pd.concat(pieces, ignore_index=True, sort=False)
    out["analysis_type"] = out["analysis_type"].fillna("single_driver_given_y_history")
    return out.sort_values(["dataset_id", "analysis_type", "driver_id"]).reset_index(drop=True)


def run_core_component_diagnostic_scan(
    binned: pd.DataFrame,
    max_lag_ka: float = CORE_COMPONENT_MAX_LAG_KA,
) -> pd.DataFrame:
    """Scan the compact three-curve lagged-information diagnostic.

    The first two curves are the single-driver information gains used in
    Figure 4, restricted to lagged LR04 and CO2:

        EP baseline + LR04_lag  vs  EP baseline
        EP baseline + CO2_lag   vs  EP baseline

    The third curve asks whether lagged precession phase adds information
    beyond the unlagged climate-state model:

        EP baseline + LR04 + CO2 + pre_phase_lag
        vs
        EP baseline + LR04 + CO2

    Thus the third curve isolates the lag of precession phase while holding
    the slow climate-state controls fixed at the event-bin age.
    """

    rows = []
    for dataset_id, dataset_frame in binned.groupby("dataset_id", sort=False):
        for lag_ka in lag_grid(max_lag_ka):
            for curve_id, source_col, label in (
                ("lr04_after_ep_baseline", "lr04_scaled", "Lagged LR04+EP baseline vs EP baseline"),
                ("co2_after_ep_baseline", "co2_scaled", "Lagged CO2+EP baseline vs EP baseline"),
            ):
                lagged = add_lagged_columns(dataset_frame, {f"{curve_id}_lag": (source_col, lag_ka)})
                reduced_terms = HISTORY_TERMS
                full_terms = (*HISTORY_TERMS, f"{curve_id}_lag")
                comparison, _, _ = compare_nested_models(lagged, reduced_terms, full_terms)
                row = {
                    "analysis_type": "core_component_diagnostic",
                    "dataset_id": dataset_id,
                    "dataset_label": dataset_frame["dataset_label"].iloc[0],
                    "curve_id": curve_id,
                    "curve_label": label,
                    "lag_ka": float(lag_ka),
                    "reduced_terms": "+".join(reduced_terms),
                    "full_terms": "+".join(full_terms),
                }
                row.update(comparison)
                rows.append(row)

            lagged = add_lagged_columns(
                dataset_frame,
                {
                    "pre_phase_sin_lag": ("pre_phase_sin", lag_ka),
                    "pre_phase_cos_lag": ("pre_phase_cos", lag_ka),
                },
            )
            reduced_terms = (*HISTORY_TERMS, "lr04_scaled", "co2_scaled")
            full_terms = (*reduced_terms, "pre_phase_sin_lag", "pre_phase_cos_lag")
            comparison, _, _ = compare_nested_models(lagged, reduced_terms, full_terms)
            row = {
                "analysis_type": "core_component_diagnostic",
                "dataset_id": dataset_id,
                "dataset_label": dataset_frame["dataset_label"].iloc[0],
                "curve_id": "full_after_climate_state",
                "curve_label": "Lagged precession phase+climate-state model vs climate-state model",
                "lag_ka": float(lag_ka),
                "reduced_terms": "+".join(reduced_terms),
                "full_terms": "+".join(full_terms),
            }
            row.update(comparison)
            rows.append(row)

    out = pd.DataFrame(rows)
    out["LR_q_value_BH"] = np.nan
    ok = out["ok"].astype(bool)
    for _, group in out[ok].groupby(["dataset_id", "curve_id"], sort=False):
        out.loc[group.index, "LR_q_value_BH"] = bh_q_values(group["LR_p_value"].to_numpy(dtype=float))
    return out


def plot_single_driver_curves(single: pd.DataFrame, best_summary: pd.DataFrame, write_pdf: bool) -> None:
    def darken(color: str, factor: float = 0.48) -> tuple[float, float, float]:
        rgb = np.array(to_rgb(color))
        return tuple(np.clip(rgb * factor, 0.0, 1.0))

    fig, axes = plt.subplots(2, 1, figsize=(5.4, 8.8), sharex=True)
    legend_handles = None
    legend_labels = None
    for ax_idx, (ax, dataset_id) in enumerate(zip(axes, DATASET_SETTINGS)):
        sub = single[
            single["dataset_id"].eq(dataset_id)
            & single["ok"].astype(bool)
        ]
        for driver_id, spec in DRIVER_TERM_SPECS.items():
            g = sub[sub["driver_id"].eq(driver_id)]
            ax.plot(
                g["lag_ka"],
                g["info_bits_per_event"],
                color=spec["color"],
                lw=1.7,
                label=spec["label"],
            )
            sig = g["LR_q_value_BH"].le(FDR_Q_THRESHOLD).to_numpy(dtype=bool)
            if sig.any():
                sig_y = g["info_bits_per_event"].to_numpy(dtype=float).copy()
                sig_y[~sig] = np.nan
                sig_color = darken(spec["color"])
                ax.plot(
                    g["lag_ka"].to_numpy(dtype=float),
                    sig_y,
                    color=sig_color,
                    lw=4.2,
                    alpha=0.52,
                    solid_capstyle="round",
                    zorder=2.6,
                    label="_nolegend_",
                )
                singletons = sig & ~np.r_[False, sig[:-1]] & ~np.r_[sig[1:], False]
                if singletons.any():
                    ax.scatter(
                        g.loc[singletons, "lag_ka"],
                        g.loc[singletons, "info_bits_per_event"],
                        color=sig_color,
                        alpha=0.52,
                        s=18,
                        zorder=2.7,
                        label="_nolegend_",
                    )
            best = best_summary[
                best_summary["dataset_id"].eq(dataset_id)
                & best_summary["analysis_type"].eq("single_driver_given_y_history")
                & best_summary["driver_id"].eq(driver_id)
            ].iloc[0]
            if best["best_lag_ka"] <= PLOT_MAX_LAG_KA:
                ax.scatter(
                    best["best_lag_ka"],
                    best["best_info_bits_per_event"],
                    color=spec["color"],
                    s=72,
                    edgecolor="white",
                    linewidth=0.6,
                    zorder=3,
                )
        ax.set_ylabel(f"{DATASET_SETTINGS[dataset_id]['label']}\nInformation gain\n(bits/event)")
        ax.set_xlim(-0.3, PLOT_MAX_LAG_KA)
        ax.text(
            -0.075,
            1.02,
            chr(ord("a") + ax_idx),
            transform=ax.transAxes,
            ha="right",
            va="bottom",
            fontsize=11,
            fontweight="bold",
            clip_on=False,
        )
        if ax_idx == 0:
            legend_handles, legend_labels = ax.get_legend_handles_labels()
    if legend_handles is not None:
        fig.legend(
            legend_handles,
            legend_labels,
            frameon=False,
            loc="upper center",
            bbox_to_anchor=(0.56, 0.925),
            ncol=3,
        )
    axes[-1].set_xlabel("Driver lag (kyr; positive = older driver state)")
    fig.subplots_adjust(left=0.21, right=0.98, top=0.875, bottom=0.12, hspace=0.23)
    save_figure(fig, "fig01_single_driver_lagged_information", write_pdf)


def plot_single_driver_delta_aicc(single: pd.DataFrame, write_pdf: bool) -> None:
    fig, axes = plt.subplots(2, 1, figsize=(12.8, 7.2), sharex=True)
    for ax, dataset_id in zip(axes, DATASET_SETTINGS):
        sub = single[
            single["dataset_id"].eq(dataset_id)
            & single["ok"].astype(bool)
        ]
        for driver_id, spec in DRIVER_TERM_SPECS.items():
            g = sub[sub["driver_id"].eq(driver_id)]
            ax.plot(
                g["lag_ka"],
                g["delta_AICc_full_minus_reduced"],
                color=spec["color"],
                lw=1.5,
                label=spec["label"],
            )
        ax.axhline(0.0, color="#111111", lw=0.8, ls="--")
        ax.set_ylabel("Delta AICc\nfull - reduced")
        ax.set_title(f"{DATASET_SETTINGS[dataset_id]['label']}: penalized support", loc="left")
        ax.grid(True, color="#e6e6e6", lw=0.6)
        ax.legend(frameon=False, loc="lower right", ncol=3)
    axes[-1].set_xlabel("Driver lag (ka; positive = older driver state)")
    fig.suptitle(
        "Negative Delta AICc means the lagged driver improves the event-process baseline",
        y=0.995,
        fontsize=14,
    )
    fig.subplots_adjust(top=0.90, hspace=0.22)
    save_figure(fig, "fig02_single_driver_delta_aicc", write_pdf)


def plot_conditional_pre_phase(same_lag: pd.DataFrame, best_climate: pd.DataFrame, write_pdf: bool) -> None:
    fig, axes = plt.subplots(2, 1, figsize=(5.4, 8.8), sharex=True)
    control_styles = [
        (same_lag, "LR04+CO$_2$ at same lag", "#0072B2"),
        (best_climate, "LR04+CO$_2$ at best lags", "#D55E00"),
    ]
    for ax_idx, (ax, dataset_id) in enumerate(zip(axes, DATASET_SETTINGS)):
        for table, label, color in control_styles:
            sub = table[
                table["dataset_id"].eq(dataset_id)
                & table["ok"].astype(bool)
            ]
            ax.plot(
                sub["lag_ka"],
                sub["info_bits_per_event"],
                color=color,
                lw=1.7,
                ls="-",
                label=label,
            )
            significant = sub[sub["LR_q_value_BH"] <= 0.05]
            if not significant.empty:
                ax.scatter(
                    significant["lag_ka"],
                    significant["info_bits_per_event"],
                    facecolors="white",
                    edgecolors=color,
                    linewidths=0.7,
                    s=14,
                    alpha=0.78,
                    zorder=3,
                    label="_nolegend_",
                )
        # ax.axhline(0.0, color="#666666", lw=0.8, ls=":")
        ax.set_ylabel("Additional precession-phase\ninformation (bits/event)")
        ax.set_title(DATASET_SETTINGS[dataset_id]["label"], loc="left")
        # ax.grid(True, color="#e6e6e6", lw=0.6)
        ax.set_xlim(0.0, PLOT_MAX_LAG_KA)
        ax.text(
            -0.075,
            1.02,
            chr(ord("a") + ax_idx),
            transform=ax.transAxes,
            ha="right",
            va="bottom",
            fontsize=11,
            fontweight="bold",
            clip_on=False,
        )
        if ax_idx == 0:
            ax.legend(frameon=False, loc="upper right")
    axes[-1].set_xlabel("Precession phase lag (kyr; positive = older phase state)")
    fig.subplots_adjust(left=0.12, right=0.98, top=0.96, bottom=0.12, hspace=0.26)
    save_figure(fig, "fig03_conditional_pre_phase_after_climate", write_pdf)


def plot_core_component_diagnostic(scan: pd.DataFrame, write_pdf: bool) -> None:
    """Plot the compact core-component lagged-information diagnostic."""

    def darken(color: str, factor: float = 0.48) -> tuple[float, float, float]:
        rgb = np.array(to_rgb(color))
        return tuple(np.clip(rgb * factor, 0.0, 1.0))

    curve_specs = {
        "lr04_after_ep_baseline": {
            "label": "Lagged LR04+EP baseline vs EP baseline",
            "color": DRIVER_TERM_SPECS["lr04"]["color"],
        },
        "co2_after_ep_baseline": {
            "label": "Lagged CO$_2$+EP baseline vs EP baseline",
            "color": DRIVER_TERM_SPECS["co2"]["color"],
        },
        "full_after_climate_state": {
            "label": "Lagged precession phase+climate-state model\nvs climate-state model",
            "color": DRIVER_TERM_SPECS["pre_phase"]["color"],
        },
    }
    fig, axes = plt.subplots(2, 1, figsize=(4.7, 8.8), sharex=True)
    legend_handles = None
    legend_labels = None
    for ax_idx, (ax, dataset_id) in enumerate(zip(axes, DATASET_SETTINGS)):
        sub = scan[scan["dataset_id"].eq(dataset_id) & scan["ok"].astype(bool)]
        for curve_id, spec in curve_specs.items():
            g = sub[sub["curve_id"].eq(curve_id)].sort_values("lag_ka")
            ax.plot(
                g["lag_ka"],
                g["info_bits_per_event"],
                color=spec["color"],
                lw=1.7,
                label=spec["label"],
            )
            sig = g["LR_q_value_BH"].le(FDR_Q_THRESHOLD).to_numpy(dtype=bool)
            if sig.any():
                sig_y = g["info_bits_per_event"].to_numpy(dtype=float).copy()
                sig_y[~sig] = np.nan
                sig_color = darken(spec["color"])
                ax.plot(
                    g["lag_ka"].to_numpy(dtype=float),
                    sig_y,
                    color=sig_color,
                    lw=4.2,
                    alpha=0.52,
                    solid_capstyle="round",
                    zorder=2.6,
                    label="_nolegend_",
                )
            best = g.loc[g["info_bits_per_event"].idxmax()]
            ax.scatter(
                best["lag_ka"],
                best["info_bits_per_event"],
                color=spec["color"],
                s=72,
                edgecolor="white",
                linewidth=0.6,
                zorder=3,
                label="_nolegend_",
            )
        ax.set_ylabel(f"{DATASET_SETTINGS[dataset_id]['label']}\nInformation gain\n(bits/event)")
        ax.set_xlim(-0.2, CORE_COMPONENT_MAX_LAG_KA)
        ax.text(
            -0.065,
            1.02,
            chr(ord("a") + ax_idx),
            transform=ax.transAxes,
            ha="right",
            va="bottom",
            fontsize=11,
            fontweight="bold",
            clip_on=False,
        )
        if ax_idx == 0:
            legend_handles, legend_labels = ax.get_legend_handles_labels()
    if legend_handles is not None:
        fig.legend(
            legend_handles,
            legend_labels,
            frameon=False,
            loc="upper center",
            bbox_to_anchor=(0.55, 0.88),
            ncol=1,
        )
    axes[-1].set_xlabel("Driver lag (kyr; positive = older driver state)")
    fig.subplots_adjust(left=0.21, right=0.98, top=0.80, bottom=0.11, hspace=0.23)
    save_figure(fig, "fig05_core_component_lagged_information_0_10kyr", write_pdf)


def plot_best_lag_summary(best_summary: pd.DataFrame, write_pdf: bool) -> None:
    plot_data = best_summary[
        best_summary["analysis_type"].isin(
            [
                "single_driver_given_y_history",
                "pre_phase_after_best_lr04_co2_lags",
            ]
        )
    ].copy()
    plot_data["label"] = np.where(
        plot_data["analysis_type"].eq("single_driver_given_y_history"),
        plot_data["driver_label"],
        "Precession phase | best climate controls",
    )
    fig, axes = plt.subplots(1, 2, figsize=(12.8, 4.8), sharey=True)
    for ax, dataset_id in zip(axes, DATASET_SETTINGS):
        sub = plot_data[plot_data["dataset_id"].eq(dataset_id)].copy()
        sub = sub.sort_values("best_info_bits_per_event", ascending=True)
        colors = [
            DRIVER_TERM_SPECS.get(driver, {"color": "#444444"})["color"]
            if analysis == "single_driver_given_y_history"
            else "#111111"
            for driver, analysis in zip(sub["driver_id"], sub["analysis_type"])
        ]
        ax.barh(sub["label"], sub["best_info_bits_per_event"], color=colors, alpha=0.82)
        for y, (_, row) in enumerate(sub.iterrows()):
            ax.text(
                row["best_info_bits_per_event"],
                y,
                f" lag={row['best_lag_ka']:.1f} ka",
                va="center",
                ha="left",
                fontsize=8,
            )
        ax.set_xlabel("Best information gain (bits/event)")
        ax.set_title(DATASET_SETTINGS[dataset_id]["label"], loc="left")
        ax.grid(True, axis="x", color="#e6e6e6", lw=0.6)
    fig.suptitle("Best lag summary", y=0.995, fontsize=14)
    fig.subplots_adjust(top=0.84, wspace=0.35)
    save_figure(fig, "fig04_best_lag_summary", write_pdf)


def write_outputs(
    binned: pd.DataFrame,
    single: pd.DataFrame,
    same_lag: pd.DataFrame,
    best_climate: pd.DataFrame,
    best_summary: pd.DataFrame,
) -> None:
    ensure_dir(OUT_DATA_DIR)
    binned.to_csv(
        OUT_DATA_DIR / "binned_event_inputs_with_history_resolution_0p2kyr.csv",
        index=False,
    )
    single.to_csv(OUT_DATA_DIR / "single_driver_lagged_predictive_information.csv", index=False)
    same_lag.to_csv(OUT_DATA_DIR / "pre_phase_after_lr04_co2_same_lag.csv", index=False)
    best_climate.to_csv(OUT_DATA_DIR / "pre_phase_after_best_lr04_co2_lags.csv", index=False)
    best_summary.to_csv(OUT_DATA_DIR / "best_lag_summary.csv", index=False)
    params = pd.DataFrame(
        [
            {
                "analysis_start_ka": ANALYSIS_START_KA,
                "analysis_end_ka": ANALYSIS_END_KA,
                "bin_width_ka": BIN_WIDTH_KA,
                "max_lag_ka": MAX_LAG_KA,
                "lag_step_ka": LAG_STEP_KA,
                "y_history_window_ka": Y_HISTORY_WINDOW_KA,
                "resolution_control": "log local Cheng composite age spacing",
                "reduced_single_driver_model": "same-type event history + Cheng sampling resolution",
                "common_lag_support": f"only bins with center_age + {MAX_LAG_KA:g} ka <= oldest bin center are used in every lag comparison",
                "time_direction": "positive lag samples driver at event_age + lag, i.e. older/past climate state",
                "information_measure": "logL_full_minus_logL_reduced from nested Poisson hazard models",
                "multiple_testing": "Benjamini-Hochberg q values across scanned lags within each dataset/analysis/driver family",
            }
        ]
    )
    params.to_csv(OUT_DATA_DIR / "parameters.csv", index=False)


def run_analysis(write_pdf: bool) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Run the full lagged predictive-information workflow."""

    ensure_dir(OUT_DATA_DIR)
    ensure_dir(OUT_FIG_DIR)
    binned = load_binned_inputs_with_history()
    single = run_single_driver_scans(binned)
    best_lags = best_single_driver_lags(single)
    same_lag = run_pre_after_climate_same_lag(binned)
    best_climate = run_pre_after_best_climate_lags(binned, best_lags)
    best_summary = build_best_summary(single, same_lag, best_climate)
    diagnostic_binned = load_binned_inputs_with_history(max_lag_ka=CORE_COMPONENT_MAX_LAG_KA)
    core_diagnostic = run_core_component_diagnostic_scan(
        diagnostic_binned,
        max_lag_ka=CORE_COMPONENT_MAX_LAG_KA,
    )
    write_outputs(binned, single, same_lag, best_climate, best_summary)
    core_diagnostic.to_csv(
        OUT_DATA_DIR / "core_component_lagged_information_0_10kyr.csv",
        index=False,
    )
    plot_single_driver_curves(single, best_summary, write_pdf)
    plot_single_driver_delta_aicc(single, write_pdf)
    plot_conditional_pre_phase(same_lag, best_climate, write_pdf)
    plot_core_component_diagnostic(core_diagnostic, write_pdf)
    plot_best_lag_summary(best_summary, write_pdf)
    return single, same_lag, best_climate, best_summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Lagged model-based TE-like predictive information for Rousseau monsoon starts."
    )
    parser.add_argument("--no-pdf", action="store_true", help="Only write PNG figures.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    _, same_lag, best_climate, best_summary = run_analysis(write_pdf=not args.no_pdf)
    print("Best lag summary:")
    print(
        best_summary[
            [
                "analysis_type",
                "dataset_id",
                "driver_id",
                "best_lag_ka",
                "best_info_bits_per_event",
                "best_LR_p_value",
                "best_LR_q_value_BH",
                "best_delta_AICc_full_minus_reduced",
                "lr04_control_lag_ka",
                "co2_control_lag_ka",
            ]
        ].to_string(index=False)
    )
    print("\nBest conditional pre-phase rows:")
    combined = pd.concat([same_lag, best_climate], ignore_index=True)
    idx = combined[combined["ok"].astype(bool)].groupby(["analysis_type", "dataset_id"])["info_bits_per_event"].idxmax()
    print(
        combined.loc[idx, [
            "analysis_type",
            "dataset_id",
            "lag_ka",
            "info_bits_per_event",
            "LR_p_value",
            "LR_q_value_BH",
            "delta_AICc_full_minus_reduced",
            "lr04_control_lag_ka",
            "co2_control_lag_ka",
        ]].sort_values(["dataset_id", "analysis_type"]).to_string(index=False)
    )
    print(f"\nWrote tables to: {OUT_DATA_DIR.relative_to(PROJECT_ROOT)}")
    print(f"Wrote figures to: {OUT_FIG_DIR.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    main()
