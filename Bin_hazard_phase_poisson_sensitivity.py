"""
Sensitivity experiment for the Rousseau et al. (2023) 0.2 kyr bin Poisson
hazard models.

The baseline script
``analyze_rousseau2023_monsoon_bin_hazard_phase_poisson.py`` fits a compact
hazard GLM:

    Y_i ~ Poisson(lambda_i * dt)
    log(lambda_i) = beta0 + LR04_i + CO2_i + sin(pre_phase_i) + cos(pre_phase_i)

This script keeps the same binning, likelihood, and precession-phase
convention, then asks whether additional forcings contribute beyond that
baseline:

    AT, obliquity, eccentricity, and 65N summer-solstice insolation.

The main sensitivity checks are nested likelihood-ratio tests:

1. Add each extra forcing to the baseline model one at a time.
2. Add all extra forcings jointly to the baseline model.
3. In the full model, drop each extra forcing one at a time to estimate its
   unique contribution after the other extra forcings are also present.

The third check is useful because orbital terms and insolation can be strongly
correlated, so a variable can look helpful alone but add little once related
variables are already in the model.

Method sketch
-------------
This script does not introduce a new likelihood. It imports the baseline
Poisson hazard fitter and changes only the candidate model list. The baseline
model is treated as the scientific reference model:

    LR04 + CO2 + sin(pre_phase) + cos(pre_phase).

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
import numpy as np
import pandas as pd
import xarray as xr
from scipy.stats import chi2

from Bin_hazard_phase_poisson import (
    ANALYSIS_END_KA,
    ANALYSIS_START_KA,
    BIN_WIDTH_KA,
    DATASET_SETTINGS,
    PROJECT_ROOT,
    build_binned_inputs,
    build_coefficient_table,
    build_fitted_rate_table,
    build_model_summary,
    clean_series,
    fit_poisson_model,
    load_all_events,
    scale_to_zero_mean_range_one,
)


RUN_NAME = "rousseau2023_monsoon_bin_hazard_phase_poisson_sensitivity"
OUT_DATA_DIR = PROJECT_ROOT / "data" / "processed" / RUN_NAME
OUT_FIG_DIR = PROJECT_ROOT / "figures" / RUN_NAME

AT_CSV = PROJECT_ROOT / "data/raw/AT.csv"
OBL_TXT = PROJECT_ROOT / "data/raw/obl_800_inter100.txt"
ECC_TXT = PROJECT_ROOT / "data/raw/ecc_1000_inter100.txt"
INSOLATION_NC = PROJECT_ROOT / "data/raw/solstice_insolation_NH.nc"
INSOLATION_LATITUDE_DEG_N = 65.0

BASE_TERMS = ("lr04_scaled", "co2_scaled", "pre_phase_sin", "pre_phase_cos")
EXTRA_TERMS = ("AT_scaled", "obl_scaled", "ecc_scaled", "insol65N_scaled")
EXTRA_TERM_LABELS = {
    "AT_scaled": "AT",
    "obl_scaled": "obliquity",
    "ecc_scaled": "eccentricity",
    "insol65N_scaled": "65N solstice insolation",
}

MODEL_SPECS: list[tuple[str, tuple[str, ...], str]] = [
    ("stationary", (), "Stationary"),
    ("climate_lr04_co2", ("lr04_scaled", "co2_scaled"), "LR04 + CO2"),
    ("pre_phase", ("pre_phase_sin", "pre_phase_cos"), "Precession phase"),
    ("base_climate_phase", BASE_TERMS, "LR04 + CO2 + pre phase"),
    ("base_plus_AT", BASE_TERMS + ("AT_scaled",), "Base + AT"),
    ("base_plus_obl", BASE_TERMS + ("obl_scaled",), "Base + obliquity"),
    ("base_plus_ecc", BASE_TERMS + ("ecc_scaled",), "Base + eccentricity"),
    ("base_plus_insol65N", BASE_TERMS + ("insol65N_scaled",), "Base + 65N insolation"),
    ("extended_all", BASE_TERMS + EXTRA_TERMS, "Base + all extra forcings"),
    (
        "all_without_AT",
        BASE_TERMS + ("obl_scaled", "ecc_scaled", "insol65N_scaled"),
        "All extra except AT",
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
        "All extra except 65N insolation",
    ),
]

MODEL_COLORS = {
    "stationary": "#777777",
    "climate_lr04_co2": "#1b9e77",
    "pre_phase": "#7570b3",
    "base_climate_phase": "#d95f02",
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

PREDICTOR_TERMS = BASE_TERMS + EXTRA_TERMS


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

    age, cleaned = clean_series(age_ka, value)
    interpolated = np.interp(centers_ka, age, cleaned)
    scaled, mean, vmin, vmax, value_range = scale_to_zero_mean_range_one(interpolated)
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
    raw = pd.read_csv(path, sep=r"\s+", header=None, names=["age_raw_ka", "value"])
    return scale_interpolated_forcing(
        forcing_id,
        forcing_label,
        np.abs(raw["age_raw_ka"].to_numpy(dtype=float)),
        raw["value"].to_numpy(dtype=float),
        centers_ka,
        path,
    )


def load_insolation_65n(centers_ka: np.ndarray) -> tuple[np.ndarray, np.ndarray, dict]:
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

    events = load_all_events()
    binned_inputs, base_scale_summary, phase_extrema = build_binned_inputs(events)
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

    The actual optimizer and Poisson likelihood are imported from the baseline
    script. This function only loops over the expanded model specification
    table, so it is deliberately a thin wrapper.
    """

    models = []
    for _, group in binned_inputs.groupby("dataset_id", sort=False):
        for model_id, terms, label in MODEL_SPECS:
            models.append(fit_poisson_model(group, model_id, terms, label))
    return models


def model_lookup(models: list) -> dict[tuple[str, str], object]:
    return {(model.dataset_id, model.model_id): model for model in models}


def build_sensitivity_likelihood_tests(models: list) -> pd.DataFrame:
    """Nested LR tests for extra-forcing contribution.

    There are three logically different comparison families:

    1. add_to_base: add one extra forcing to the baseline model. This answers
       whether that one predictor helps when LR04, CO2, and precession phase
       are already present.
    2. joint_add_to_base: add all extra forcings together. This asks whether
       the extra block has any collective information.
    3. drop_one_from_full: compare the full model against a model missing one
       extra forcing. This estimates the unique contribution of that forcing
       after the other extras are also included.

    In every row, the reported LR p value comes from:

        LR = 2 * (logL_full - logL_reduced)
        df = number of added coefficients.
    """

    lookup = model_lookup(models)
    comparisons = [
        (
            "base_vs_stationary",
            "main_context",
            "stationary",
            "base_climate_phase",
            "LR04+CO2+phase vs stationary",
        ),
        (
            "AT_after_base",
            "add_to_base",
            "base_climate_phase",
            "base_plus_AT",
            "Add AT to base",
        ),
        (
            "obl_after_base",
            "add_to_base",
            "base_climate_phase",
            "base_plus_obl",
            "Add obliquity to base",
        ),
        (
            "ecc_after_base",
            "add_to_base",
            "base_climate_phase",
            "base_plus_ecc",
            "Add eccentricity to base",
        ),
        (
            "insol65N_after_base",
            "add_to_base",
            "base_climate_phase",
            "base_plus_insol65N",
            "Add 65N insolation to base",
        ),
        (
            "all_extras_after_base",
            "joint_add_to_base",
            "base_climate_phase",
            "extended_all",
            "Add all extra forcings to base",
        ),
        (
            "AT_unique_in_full",
            "drop_one_from_full",
            "all_without_AT",
            "extended_all",
            "AT unique in full",
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
            "65N insolation unique in full",
        ),
    ]
    rows = []
    for dataset_id in DATASET_SETTINGS:
        for comparison_id, family, reduced_id, full_id, label in comparisons:
            reduced = lookup[(dataset_id, reduced_id)]
            full = lookup[(dataset_id, full_id)]
            # All comparisons are nested by construction, so the added terms in
            # the full model define the chi-square degrees of freedom.
            lr_stat = 2.0 * (full.log_likelihood - reduced.log_likelihood)
            df = len(full.beta) - len(reduced.beta)
            p_value = float(chi2.sf(max(lr_stat, 0.0), df))
            rows.append(
                {
                    "dataset_id": dataset_id,
                    "dataset_label": full.dataset_label,
                    "comparison_id": comparison_id,
                    "comparison_family": family,
                    "comparison_label": label,
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


def plot_model_delta_aicc(summary: pd.DataFrame, write_pdf: bool) -> None:
    model_order = [model_id for model_id, _, _ in MODEL_SPECS]
    fig, axes = plt.subplots(1, 2, figsize=(13.2, 6.4), sharey=True)
    for ax, dataset_id in zip(axes, DATASET_SETTINGS):
        sub = summary[summary["dataset_id"].eq(dataset_id)].set_index("model_id").reindex(model_order)
        colors = [MODEL_COLORS.get(model_id, "#999999") for model_id in sub.index]
        y = np.arange(len(sub))
        ax.barh(y, sub["delta_AICc"], color=colors, alpha=0.86)
        ax.set_yticks(y)
        ax.set_yticklabels(sub["model_label"])
        ax.invert_yaxis()
        ax.set_xlabel("Delta AICc")
        ax.set_title(DATASET_SETTINGS[dataset_id]["label"], loc="left")
        ax.grid(True, axis="x", color="#e6e6e6", lw=0.6)
    fig.suptitle("Sensitivity Poisson hazard models: AICc comparison", y=0.995, fontsize=14)
    fig.subplots_adjust(top=0.90, wspace=0.22)
    save_figure(fig, "fig01_sensitivity_delta_AICc", write_pdf)


def plot_likelihood_tests(likelihood_tests: pd.DataFrame, write_pdf: bool) -> None:
    families = [
        ("add_to_base", "Single extra forcing added to base"),
        ("joint_add_to_base", "All extra forcings added to base"),
        ("drop_one_from_full", "Unique contribution inside full model"),
    ]
    fig, axes = plt.subplots(len(families), 2, figsize=(13.4, 8.2), sharex=True)
    threshold = -np.log10(0.05)
    for row_idx, (family, _) in enumerate(families):
        for col_idx, dataset_id in enumerate(DATASET_SETTINGS):
            ax = axes[row_idx, col_idx]
            sub = likelihood_tests[
                likelihood_tests["dataset_id"].eq(dataset_id)
                & likelihood_tests["comparison_family"].eq(family)
            ].copy()
            sub["minus_log10_p"] = -np.log10(np.maximum(sub["LR_p_value"], 1e-300))
            y = np.arange(len(sub))
            colors = ["#3182bd" if p < 0.05 else "#bdbdbd" for p in sub["LR_p_value"]]
            ax.barh(y, sub["minus_log10_p"], color=colors, alpha=0.86)
            ax.axvline(threshold, color="#222222", ls="--", lw=0.9)
            ax.set_yticks(y)
            ax.set_yticklabels(sub["comparison_label"])
            ax.invert_yaxis()
            ax.grid(True, axis="x", color="#e6e6e6", lw=0.6)
            if row_idx == 0:
                ax.set_title(DATASET_SETTINGS[dataset_id]["label"], loc="left")
            if row_idx == len(families) - 1:
                ax.set_xlabel("-log10 LR p value")
            for yi, (_, test) in zip(y, sub.iterrows()):
                ax.text(
                    test["minus_log10_p"] + 0.04,
                    yi,
                    f"p={test['LR_p_value']:.3g}, dAICc={test['delta_AICc_full_minus_reduced']:.2f}",
                    va="center",
                    fontsize=7,
                )
    fig.suptitle("Nested likelihood-ratio tests for added forcings", y=0.995, fontsize=14)
    fig.subplots_adjust(left=0.20, top=0.92, hspace=0.34, wspace=0.35)
    save_figure(fig, "fig02_sensitivity_likelihood_tests", write_pdf)


def plot_extended_coefficients(coefficients: pd.DataFrame, write_pdf: bool) -> None:
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
    fig, axes = plt.subplots(2, 1, figsize=(13.2, 5.8), sharex=True)
    for ax, dataset_id in zip(axes, DATASET_SETTINGS):
        settings = DATASET_SETTINGS[dataset_id]
        data = binned_inputs[binned_inputs["dataset_id"].eq(dataset_id)]
        events = data[data["event_count"] > 0]
        ax.vlines(
            events["bin_center_ka"],
            0.0,
            events["event_count"],
            color=settings["color"],
            lw=0.8,
            alpha=0.45,
            label="observed event bins",
        )
        for model_id, color, label in [
            ("base_climate_phase", "#d95f02", "base: LR04+CO2+phase"),
            ("extended_all", "#202020", "base + AT+obl+ecc+insol65N"),
        ]:
            rates = fitted_rates[
                fitted_rates["dataset_id"].eq(dataset_id)
                & fitted_rates["model_id"].eq(model_id)
            ]
            ax.plot(rates["bin_center_ka"], rates["lambda_per_kyr"], color=color, lw=1.2, label=label)
        test = likelihood_tests[
            likelihood_tests["dataset_id"].eq(dataset_id)
            & likelihood_tests["comparison_id"].eq("all_extras_after_base")
        ].iloc[0]
        ax.text(
            0.01,
            0.94,
            f"{settings['label']}: N={int(data['event_count'].sum())}, "
            f"all extras after base p={test['LR_p_value']:.3g}",
            transform=ax.transAxes,
            ha="left",
            va="top",
            bbox={"boxstyle": "round,pad=0.25", "facecolor": "white", "edgecolor": "#bbbbbb", "alpha": 0.86},
        )
        ax.set_ylabel("rate / kyr")
        ax.grid(True, color="#e6e6e6", lw=0.6)
        ax.legend(frameon=False, loc="upper right", ncol=3)
    axes[-1].set_xlim(ANALYSIS_END_KA, ANALYSIS_START_KA)
    axes[-1].set_xlabel("Age (ka BP)")
    fig.suptitle("Base versus extended fitted hazards", y=0.995, fontsize=14)
    fig.subplots_adjust(top=0.90, hspace=0.22)
    save_figure(fig, "fig04_base_vs_extended_hazards", write_pdf)


def plot_predictor_correlation(correlation_table: pd.DataFrame, write_pdf: bool) -> None:
    corr = correlation_table.pivot(index="term_1", columns="term_2", values="correlation")
    corr = corr.loc[list(PREDICTOR_TERMS), list(PREDICTOR_TERMS)]
    fig, ax = plt.subplots(figsize=(7.4, 6.2))
    image = ax.imshow(corr.to_numpy(dtype=float), vmin=-1.0, vmax=1.0, cmap="coolwarm")
    labels = [
        "LR04",
        "CO2",
        "sin phase",
        "cos phase",
        "AT",
        "obl",
        "ecc",
        "65N insol",
    ]
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
                "response": "event_count_per_0p2kyr_bin",
                "baseline_model": "LR04_scaled + CO2_scaled + sin(pre_phase) + cos(pre_phase)",
                "extra_forcings": "AT_scaled + obl_scaled + ecc_scaled + insol65N_scaled",
                "main_question": "Do extra forcings improve the baseline binned Poisson hazard model?",
            }
        ]
    )
    params.to_csv(OUT_DATA_DIR / "parameters.csv", index=False)


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
    models = fit_sensitivity_models(binned_inputs)
    summary = build_model_summary(models, binned_inputs)
    coefficients = build_coefficient_table(models)
    likelihood_tests = build_sensitivity_likelihood_tests(models)
    fitted_rates = build_fitted_rate_table(models, binned_inputs)
    predictor_correlation = build_predictor_correlation_table(binned_inputs)

    write_outputs(
        binned_inputs,
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
    plot_extended_coefficients(coefficients, write_pdf)
    plot_base_vs_extended_rates(binned_inputs, fitted_rates, likelihood_tests, write_pdf)
    plot_predictor_correlation(predictor_correlation, write_pdf)
    return summary, likelihood_tests


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Sensitivity GLM for extra forcings in Rousseau 2023 binned monsoon-start hazards."
    )
    parser.add_argument("--no-pdf", action="store_true", help="Only write PNG figures.")
    return parser.parse_args()


def main() -> None:
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
