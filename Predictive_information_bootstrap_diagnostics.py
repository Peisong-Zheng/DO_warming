"""
Sparse-bin diagnostics and parametric-bootstrap LR tests for the adjusted
predictive Poisson hazard models.

The main script fits the models used in the paper. This diagnostic script asks
whether the two main likelihood-ratio tests still look significant when the
reference distribution is generated directly from the fitted reduced model
rather than from the asymptotic chi-square approximation.

The two bootstrap tests are:

1. LR04 + CO2 after the event-process baseline:

       reduced = same-type history + Cheng sampling resolution
       full    = reduced + LR04 + CO2

2. Precession phase after the climate-state model:

       reduced = same-type history + Cheng sampling resolution + LR04 + CO2
       full    = reduced + sin(precession phase) + cos(precession phase)

By default, the bootstrap is dynamic with respect to the same-type history
term. Each simulated catalogue is generated from oldest to youngest bins under
the fitted reduced model, recomputing the history term from the simulated
events. The reduced and full models are then refitted to the simulated
catalogue. This keeps the bootstrap close to the model actually used in the
paper, where history is derived from the event sequence itself.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
import time

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.stats import chi2

import Bin_hazard_phase_poisson as base
import Predictive_information_model as predictive
from toolbox.model_stats import information_criteria


RUN_NAME = "Predictive_information_bootstrap_diagnostics"
OUT_DATA_DIR = base.PROJECT_ROOT / "data" / "processed" / RUN_NAME
OUT_FIG_DIR = base.PROJECT_ROOT / "figures" / RUN_NAME

DEFAULT_N_BOOTSTRAP = 2000
DEFAULT_SEED = 20260512
DEFAULT_BOOTSTRAP_MAXITER = 800

BOOTSTRAP_TESTS: tuple[tuple[str, str, str, str], ...] = (
    (
        "climate_lr04_co2_after_baseline",
        "history_resolution_baseline",
        "baseline_climate_lr04_co2",
        "LR04 + CO2 after history + resolution",
    ),
    (
        "phase_after_adjusted_climate",
        "baseline_climate_lr04_co2",
        "baseline_climate_lr04_co2_pre_phase",
        "Precession phase after history + resolution + LR04 + CO2",
    ),
)


@dataclass(frozen=True)
class BootstrapComparison:
    comparison_id: str
    reduced_model_id: str
    full_model_id: str
    label: str


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def model_spec(model_id: str) -> tuple[tuple[str, ...], str]:
    for candidate_id, terms, label in predictive.MODEL_SPECS:
        if candidate_id == model_id:
            return terms, label
    raise KeyError(f"Unknown adjusted model id: {model_id}")


def comparison_specs() -> list[BootstrapComparison]:
    return [BootstrapComparison(*values) for values in BOOTSTRAP_TESTS]


def fit_model_by_id(frame: pd.DataFrame, model_id: str) -> base.FittedPoissonModel:
    terms, label = model_spec(model_id)
    return base.fit_poisson_model(frame, model_id, terms, label)


def fit_poisson_model_fast(
    dataset_frame: pd.DataFrame,
    model_id: str,
    terms: tuple[str, ...],
    model_label: str,
    *,
    initial_beta: np.ndarray | None = None,
    maxiter: int = DEFAULT_BOOTSTRAP_MAXITER,
) -> base.FittedPoissonModel:
    """Bootstrap-oriented version of the Poisson fitter.

    The paper-facing fit uses a generous optimizer budget. Bootstrap refits are
    repeated thousands of times, so this version warm-starts from the observed
    fit and uses a smaller iteration cap. The returned object has the same
    structure as the main fitter.
    """

    x = base.design_matrix(dataset_frame, terms)
    y = dataset_frame["event_count"].to_numpy(dtype=float)
    dt = dataset_frame["dt_ka"].to_numpy(dtype=float)
    duration = float(dt.sum())
    n_events = float(y.sum())
    n_obs = len(y)

    beta0 = np.zeros(1 + len(terms), dtype=float)
    beta0[0] = np.log(max(n_events / duration, 1e-12))
    if initial_beta is not None and len(initial_beta) == len(beta0):
        beta0 = np.asarray(initial_beta, dtype=float).copy()
        # Keep the warm-start slope coefficients but recenter the intercept to
        # the simulated event count scale. This avoids slow searches when a
        # bootstrap catalogue has more or fewer events than the observed one.
        beta0[0] = np.log(max(n_events / duration, 1e-12))

    if not terms:
        beta = beta0
        converged = True
        message = "analytic stationary MLE"
    else:
        bounds = [(-20.0, 5.0)] + [(-20.0, 20.0)] * len(terms)
        result = minimize(
            base.negative_loglik,
            beta0,
            args=(x, y, dt),
            method="L-BFGS-B",
            bounds=bounds,
            options={"maxiter": int(maxiter), "ftol": 1e-9, "gtol": 1e-6, "maxls": 30},
        )
        beta = result.x
        converged = bool(result.success)
        message = str(result.message)

    eta = np.clip(beta[0] + x @ beta[1:], -50.0, 20.0)
    rate = np.exp(eta)
    mu = rate * dt
    log_likelihood = base.poisson_loglik(beta, x, y, dt)
    k = len(beta)
    criteria = information_criteria(log_likelihood, k, n_obs)

    return base.FittedPoissonModel(
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


def refit_comparison(
    frame: pd.DataFrame,
    comparison: BootstrapComparison,
    *,
    initial_reduced: base.FittedPoissonModel | None = None,
    initial_full: base.FittedPoissonModel | None = None,
    maxiter: int = DEFAULT_BOOTSTRAP_MAXITER,
) -> tuple[base.FittedPoissonModel, base.FittedPoissonModel, float]:
    reduced_terms, reduced_label = model_spec(comparison.reduced_model_id)
    full_terms, full_label = model_spec(comparison.full_model_id)
    reduced = fit_poisson_model_fast(
        frame,
        comparison.reduced_model_id,
        reduced_terms,
        reduced_label,
        initial_beta=None if initial_reduced is None else initial_reduced.beta,
        maxiter=maxiter,
    )
    full = fit_poisson_model_fast(
        frame,
        comparison.full_model_id,
        full_terms,
        full_label,
        initial_beta=None if initial_full is None else initial_full.beta,
        maxiter=maxiter,
    )
    lr_stat = 2.0 * max(full.log_likelihood - reduced.log_likelihood, 0.0)
    return reduced, full, lr_stat


def poisson_deviance(y: np.ndarray, mu: np.ndarray) -> float:
    y = np.asarray(y, dtype=float)
    mu = np.clip(np.asarray(mu, dtype=float), 1e-15, None)
    positive = y > 0
    out = np.empty_like(y, dtype=float)
    out[positive] = y[positive] * np.log(y[positive] / mu[positive]) - (y[positive] - mu[positive])
    out[~positive] = mu[~positive]
    return float(2.0 * np.sum(out))


def sparse_bin_diagnostics(
    fit_frame: pd.DataFrame,
    models: list[base.FittedPoissonModel],
) -> pd.DataFrame:
    """Compare observed sparsity with fitted Poisson expectations."""

    rows = []
    for model in models:
        subset = fit_frame[fit_frame["dataset_id"].eq(model.dataset_id)]
        y = subset["event_count"].to_numpy(dtype=float)
        mu = np.clip(model.fitted_mu_per_bin, 1e-15, None)
        k = len(model.beta)
        n = len(y)
        pearson_chi2 = float(np.sum((y - mu) ** 2 / mu))
        deviance = poisson_deviance(y, mu)
        expected_zero_count = float(np.sum(np.exp(-mu)))
        expected_multi_count = float(np.sum(1.0 - np.exp(-mu) * (1.0 + mu)))
        rows.append(
            {
                "dataset_id": model.dataset_id,
                "dataset_label": model.dataset_label,
                "model_id": model.model_id,
                "model_label": model.model_label,
                "n_bins": int(n),
                "n_events": int(y.sum()),
                "n_parameters": int(k),
                "observed_zero_count": int(np.sum(y == 0.0)),
                "expected_zero_count": expected_zero_count,
                "observed_zero_fraction": float(np.mean(y == 0.0)),
                "expected_zero_fraction": expected_zero_count / n,
                "observed_nonzero_count": int(np.sum(y > 0.0)),
                "observed_multi_event_bin_count": int(np.sum(y > 1.0)),
                "expected_multi_event_bin_count": expected_multi_count,
                "max_observed_event_count_per_bin": int(np.max(y)),
                "max_fitted_expected_events_per_bin": float(np.max(mu)),
                "mean_fitted_expected_events_per_bin": float(np.mean(mu)),
                "pearson_chi2": pearson_chi2,
                "pearson_dispersion": pearson_chi2 / max(n - k, 1),
                "deviance": deviance,
                "deviance_dispersion": deviance / max(n - k, 1),
            }
        )
    return pd.DataFrame(rows)


def compute_history_from_counts(centers: np.ndarray, counts: np.ndarray, window_ka: float) -> np.ndarray:
    """Same history definition as predictive.add_same_type_history, but array-only."""

    cumulative = np.concatenate([[0.0], np.cumsum(counts)])
    left_idx = np.searchsorted(centers, centers + 1e-9, side="right")
    right_idx = np.searchsorted(centers, centers + window_ka, side="right")
    return cumulative[right_idx] - cumulative[left_idx]


def simulate_dynamic_history_catalogue(
    full_support_frame: pd.DataFrame,
    reduced_model: base.FittedPoissonModel,
    rng: np.random.Generator,
    history_window_ka: float,
) -> pd.DataFrame:
    """Simulate one catalogue under the reduced model, recomputing history.

    Age increases into the past. The fitted history term counts events in the
    older interval (t, t + W], so simulation proceeds from the oldest bin toward
    the youngest bin. This makes the simulated history available before the
    event count in the current bin is drawn.
    """

    frame = full_support_frame.sort_values("bin_center_ka").copy().reset_index(drop=True)
    centers = frame["bin_center_ka"].to_numpy(dtype=float)
    dt = frame["dt_ka"].to_numpy(dtype=float)
    simulated = np.zeros(len(frame), dtype=float)
    beta_by_term = dict(zip(reduced_model.terms, reduced_model.beta[1:]))

    for idx in range(len(frame) - 1, -1, -1):
        eta = float(reduced_model.beta[0])
        for term, beta in beta_by_term.items():
            if term == predictive.HISTORY_TERM:
                right_idx = np.searchsorted(
                    centers,
                    centers[idx] + history_window_ka,
                    side="right",
                )
                value = float(np.sum(simulated[idx + 1 : right_idx]))
            else:
                value = float(frame.at[idx, term])
            eta += float(beta) * value
        mu = float(dt[idx] * np.exp(np.clip(eta, -50.0, 20.0)))
        simulated[idx] = rng.poisson(mu)

    frame["event_count"] = simulated.astype(int)
    frame = predictive.add_same_type_history(frame, history_window_ka)
    return predictive.model_frame(frame)


def simulate_conditional_catalogue(
    fit_frame: pd.DataFrame,
    reduced_model: base.FittedPoissonModel,
    rng: np.random.Generator,
) -> pd.DataFrame:
    """Simulate counts while keeping the fitted design matrix fixed."""

    out = fit_frame.copy()
    out["event_count"] = rng.poisson(np.clip(reduced_model.fitted_mu_per_bin, 1e-15, None)).astype(int)
    return out


def bootstrap_lrt(
    *,
    binned_full_support: pd.DataFrame,
    fit_frame: pd.DataFrame,
    observed_models: list[base.FittedPoissonModel],
    observed_lrt: pd.DataFrame,
    comparison: BootstrapComparison,
    n_bootstrap: int,
    rng: np.random.Generator,
    mode: str,
    maxiter: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    lookup = base.model_lookup(observed_models)
    rows = []
    summary_rows = []

    for dataset_id, dataset_frame in fit_frame.groupby("dataset_id", sort=False):
        full_support = binned_full_support[binned_full_support["dataset_id"].eq(dataset_id)].copy()
        reduced_obs = lookup[(dataset_id, comparison.reduced_model_id)]
        full_obs = lookup[(dataset_id, comparison.full_model_id)]
        obs_row = observed_lrt[
            observed_lrt["dataset_id"].eq(dataset_id)
            & observed_lrt["comparison_id"].eq(comparison.comparison_id)
        ].iloc[0]
        observed_lr = float(obs_row["LR_statistic"])

        tic = time.time()
        for bootstrap_id in range(1, n_bootstrap + 1):
            if mode == "dynamic_history":
                boot_frame = simulate_dynamic_history_catalogue(
                    full_support,
                    reduced_obs,
                    rng,
                    predictive.MAIN_HISTORY_WINDOW_KA,
                )
            elif mode == "conditional_fixed_design":
                boot_frame = simulate_conditional_catalogue(dataset_frame, reduced_obs, rng)
            else:
                raise ValueError(f"Unknown bootstrap mode: {mode}")

            reduced_boot, full_boot, boot_lr = refit_comparison(
                boot_frame,
                comparison,
                initial_reduced=reduced_obs,
                initial_full=full_obs,
                maxiter=maxiter,
            )
            rows.append(
                {
                    "bootstrap_id": bootstrap_id,
                    "bootstrap_mode": mode,
                    "dataset_id": dataset_id,
                    "dataset_label": reduced_obs.dataset_label,
                    "comparison_id": comparison.comparison_id,
                    "comparison_label": comparison.label,
                    "reduced_model_id": comparison.reduced_model_id,
                    "full_model_id": comparison.full_model_id,
                    "n_bins": int(len(boot_frame)),
                    "n_events": int(boot_frame["event_count"].sum()),
                    "nonzero_bin_count": int(np.sum(boot_frame["event_count"].to_numpy() > 0)),
                    "zero_fraction": float(np.mean(boot_frame["event_count"].to_numpy() == 0)),
                    "LR_statistic_bootstrap": boot_lr,
                    "ll_gain_nats_bootstrap": 0.5 * boot_lr,
                    "reduced_converged": bool(reduced_boot.converged),
                    "full_converged": bool(full_boot.converged),
                }
            )

            if bootstrap_id % max(n_bootstrap // 10, 1) == 0:
                elapsed = time.time() - tic
                print(
                    f"{mode} | {dataset_id} | {comparison.comparison_id}: "
                    f"{bootstrap_id}/{n_bootstrap} replicates in {elapsed:.1f}s",
                    flush=True,
                )

        sub = pd.DataFrame([row for row in rows if row["dataset_id"] == dataset_id and row["comparison_id"] == comparison.comparison_id and row["bootstrap_mode"] == mode])
        exceed = int(np.sum(sub["LR_statistic_bootstrap"].to_numpy() >= observed_lr))
        p_boot = (exceed + 1.0) / (len(sub) + 1.0)
        df = len(full_obs.beta) - len(reduced_obs.beta)
        summary_rows.append(
            {
                "bootstrap_mode": mode,
                "dataset_id": dataset_id,
                "dataset_label": reduced_obs.dataset_label,
                "comparison_id": comparison.comparison_id,
                "comparison_label": comparison.label,
                "reduced_model_id": comparison.reduced_model_id,
                "full_model_id": comparison.full_model_id,
                "df": int(df),
                "n_bootstrap": int(len(sub)),
                "n_bootstrap_exceeding_or_equal_observed": exceed,
                "observed_LR_statistic": observed_lr,
                "chi_square_p_value": float(obs_row["LR_p_value"]),
                "bootstrap_p_value": p_boot,
                "bootstrap_LR_mean": float(sub["LR_statistic_bootstrap"].mean()),
                "bootstrap_LR_median": float(sub["LR_statistic_bootstrap"].median()),
                "bootstrap_LR_95": float(sub["LR_statistic_bootstrap"].quantile(0.95)),
                "bootstrap_LR_99": float(sub["LR_statistic_bootstrap"].quantile(0.99)),
                "observed_ll_gain_nats": float(obs_row["ll_gain_nats"]),
                "observed_info_bits_per_event": float(obs_row["info_bits_per_event"]),
                "observed_delta_AICc_full_minus_reduced": float(
                    obs_row["delta_AICc_full_minus_reduced"]
                ),
            }
        )

    return pd.DataFrame(rows), pd.DataFrame(summary_rows)


def plot_bootstrap_nulls(
    replicates: pd.DataFrame,
    summary: pd.DataFrame,
    write_pdf: bool,
) -> None:
    comparisons = list(comparison_specs())
    datasets = list(base.DATASET_SETTINGS)
    modes = list(replicates["bootstrap_mode"].drop_duplicates())
    for mode in modes:
        fig, axes = plt.subplots(2, 2, figsize=(10.5, 7.4), sharex=False, sharey=False)
        for row_idx, dataset_id in enumerate(datasets):
            for col_idx, comparison in enumerate(comparisons):
                ax = axes[row_idx, col_idx]
                sub = replicates[
                    replicates["bootstrap_mode"].eq(mode)
                    & replicates["dataset_id"].eq(dataset_id)
                    & replicates["comparison_id"].eq(comparison.comparison_id)
                ]
                srow = summary[
                    summary["bootstrap_mode"].eq(mode)
                    & summary["dataset_id"].eq(dataset_id)
                    & summary["comparison_id"].eq(comparison.comparison_id)
                ].iloc[0]
                values = sub["LR_statistic_bootstrap"].to_numpy(dtype=float)
                ax.hist(values, bins=36, density=True, color="#bdbdbd", edgecolor="white")
                x_max = max(float(np.nanmax(values)), float(srow["observed_LR_statistic"])) * 1.08
                x = np.linspace(0.0, x_max, 400)
                ax.plot(
                    x,
                    chi2.pdf(x, int(srow["df"])),
                    color="#2b6cb0",
                    lw=1.2,
                    label=rf"$\chi^2_{{{int(srow['df'])}}}$",
                )
                ax.axvline(
                    float(srow["observed_LR_statistic"]),
                    color="#c51b7d",
                    lw=1.6,
                    label="observed LR",
                )
                ax.set_title(
                    f"{base.DATASET_SETTINGS[dataset_id]['label']}\n{comparison.label}",
                    loc="left",
                    fontsize=9,
                )
                ax.text(
                    0.98,
                    0.94,
                    f"LR={srow['observed_LR_statistic']:.2f}\n"
                    f"chi p={srow['chi_square_p_value']:.3g}\n"
                    f"boot p={srow['bootstrap_p_value']:.3g}",
                    transform=ax.transAxes,
                    ha="right",
                    va="top",
                    fontsize=8,
                    bbox={
                        "boxstyle": "round,pad=0.25",
                        "facecolor": "white",
                        "edgecolor": "#bbbbbb",
                        "alpha": 0.9,
                    },
                )
                ax.set_xlabel("LR statistic")
                ax.set_ylabel("density")
                ax.grid(False)
                if row_idx == 0 and col_idx == 0:
                    ax.legend(frameon=False, loc="upper right", fontsize=8)
        fig.subplots_adjust(left=0.08, right=0.98, top=0.94, bottom=0.08, hspace=0.46, wspace=0.28)
        stem = f"fig01_bootstrap_lrt_null_distributions_{mode}"
        ensure_dir(OUT_FIG_DIR)
        fig.savefig(OUT_FIG_DIR / f"{stem}.png", dpi=300, bbox_inches="tight")
        if write_pdf:
            fig.savefig(OUT_FIG_DIR / f"{stem}.pdf", bbox_inches="tight")
        plt.close(fig)


def run_analysis(
    *,
    n_bootstrap: int,
    seed: int,
    mode: str,
    maxiter: int,
    write_pdf: bool,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    ensure_dir(OUT_DATA_DIR)
    ensure_dir(OUT_FIG_DIR)
    rng = np.random.default_rng(seed)

    binned, _, _, _ = predictive.build_adjusted_inputs(predictive.MAIN_HISTORY_WINDOW_KA)
    fit_frame = predictive.model_frame(binned)
    observed_models = predictive.fit_adjusted_models(binned)
    observed_lrt = predictive.build_adjusted_likelihood_tests(
        observed_models,
        fit_frame,
        predictive.MAIN_HISTORY_WINDOW_KA,
    )

    diagnostics = sparse_bin_diagnostics(fit_frame, observed_models)
    diagnostics.to_csv(OUT_DATA_DIR / "poisson_sparse_bin_diagnostics.csv", index=False)

    replicate_tables = []
    summary_tables = []
    modes = ["dynamic_history"] if mode != "both" else ["dynamic_history", "conditional_fixed_design"]
    if mode == "conditional_fixed_design":
        modes = ["conditional_fixed_design"]

    for bootstrap_mode in modes:
        for comparison in comparison_specs():
            reps, summary = bootstrap_lrt(
                binned_full_support=binned,
                fit_frame=fit_frame,
                observed_models=observed_models,
                observed_lrt=observed_lrt,
                comparison=comparison,
                n_bootstrap=n_bootstrap,
                rng=rng,
                mode=bootstrap_mode,
                maxiter=maxiter,
            )
            replicate_tables.append(reps)
            summary_tables.append(summary)

    replicates = pd.concat(replicate_tables, ignore_index=True)
    summary = pd.concat(summary_tables, ignore_index=True)
    replicates.to_csv(OUT_DATA_DIR / "bootstrap_lrt_replicates.csv", index=False)
    summary.to_csv(OUT_DATA_DIR / "bootstrap_lrt_summary.csv", index=False)
    pd.DataFrame(
        [
            {
                "run_name": RUN_NAME,
                "n_bootstrap": int(n_bootstrap),
                "seed": int(seed),
                "bootstrap_mode": mode,
                "bootstrap_maxiter": int(maxiter),
                "bin_width_ka": base.BIN_WIDTH_KA,
                "history_window_ka": predictive.MAIN_HISTORY_WINDOW_KA,
                "bootstrap_null": "simulate event counts from fitted reduced model",
                "dynamic_history": "same-type history is recomputed from simulated event counts",
                "p_value_rule": "(1 + number of bootstrap LR >= observed LR) / (1 + n_bootstrap)",
            }
        ]
    ).to_csv(OUT_DATA_DIR / "parameters.csv", index=False)

    plot_bootstrap_nulls(replicates, summary, write_pdf)
    return diagnostics, summary, replicates


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run sparse-bin diagnostics and reduced-model parametric-bootstrap LR tests."
    )
    parser.add_argument(
        "--n-bootstrap",
        type=int,
        default=DEFAULT_N_BOOTSTRAP,
        help=f"Number of bootstrap replicates per dataset/comparison (default: {DEFAULT_N_BOOTSTRAP}).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=DEFAULT_SEED,
        help=f"Random seed (default: {DEFAULT_SEED}).",
    )
    parser.add_argument(
        "--mode",
        choices=["dynamic_history", "conditional_fixed_design", "both"],
        default="dynamic_history",
        help="Bootstrap mode. Dynamic history recomputes event history from each simulated catalogue.",
    )
    parser.add_argument(
        "--maxiter",
        type=int,
        default=DEFAULT_BOOTSTRAP_MAXITER,
        help=f"Maximum L-BFGS-B iterations for each bootstrap refit (default: {DEFAULT_BOOTSTRAP_MAXITER}).",
    )
    parser.add_argument("--no-pdf", action="store_true", help="Only write PNG figures.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    diagnostics, summary, _ = run_analysis(
        n_bootstrap=args.n_bootstrap,
        seed=args.seed,
        mode=args.mode,
        maxiter=args.maxiter,
        write_pdf=not args.no_pdf,
    )

    keep_diag = [
        "dataset_id",
        "model_id",
        "n_events",
        "observed_zero_fraction",
        "expected_zero_fraction",
        "pearson_dispersion",
        "deviance_dispersion",
    ]
    print("\nSparse-bin diagnostics:")
    print(
        diagnostics[
            diagnostics["model_id"].isin(
                [
                    "stationary",
                    "history_resolution_baseline",
                    "baseline_climate_lr04_co2",
                    "baseline_climate_lr04_co2_pre_phase",
                ]
            )
        ][keep_diag].to_string(index=False)
    )

    print("\nReduced-model bootstrap LR tests:")
    print(
        summary[
            [
                "bootstrap_mode",
                "dataset_id",
                "comparison_id",
                "observed_LR_statistic",
                "df",
                "chi_square_p_value",
                "bootstrap_p_value",
                "bootstrap_LR_95",
                "bootstrap_LR_99",
                "n_bootstrap_exceeding_or_equal_observed",
            ]
        ].to_string(index=False)
    )
    print(f"\nWrote tables to: {OUT_DATA_DIR.relative_to(base.PROJECT_ROOT)}")
    print(f"Wrote figures to: {OUT_FIG_DIR.relative_to(base.PROJECT_ROOT)}")


if __name__ == "__main__":
    main()
