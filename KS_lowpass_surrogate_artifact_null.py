"""
Surrogate null test for precession-band detection artifacts.

Reviewer concern
----------------
The Rousseau et al. (2023) transition catalogue was detected from the Cheng
et al. (2016) composite speleothem d18O record. Because that composite contains
strong orbital-scale variance, a KS detector might flag the steep flanks of the
smooth precession-band waveform and thereby create an apparent relation between
event timing and precession phase.

This script tests that mechanism directly. It keeps the low-frequency
component of the Cheng composite, estimated with a 10 kyr low-pass filter, and
adds synthetic high-frequency residuals generated from a Gaussian AR(p) model
fitted to the observed <10 kyr residual. The same TiPES/Rousseau-style KS
detector is then applied to the observed record, the low-pass-only record, and
many surrogate records. If the phase signal were mainly a smooth-waveform
detection artifact, the surrogate records should often generate event counts
and Rayleigh phase concentrations comparable to the observed raw record.

No published Rousseau transition ages are used by this script.
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
from scipy.signal import butter, sosfiltfilt

import KS_precession_band_removal_sensitivity as ksdet
import Orbital_phase_rayleigh as rayleigh


PROJECT_ROOT = Path(__file__).resolve().parent
RUN_NAME = "rousseau2023_monsoon_ks_lowpass_surrogate_artifact_null"
OUT_DATA_DIR = PROJECT_ROOT / "data" / "processed" / RUN_NAME
OUT_FIG_DIR = PROJECT_ROOT / "figures" / RUN_NAME

LOWPASS_CUTOFF_PERIOD_KA = 10.0
FILTER_ORDER = 4
RESAMPLE_STEP_KA = ksdet.RESAMPLE_STEP_KA
EDGE_TRIM_KA = 20.0
N_SURROGATES = 50
RANDOM_SEED = 20260512
MAX_AR_ORDER = 40

DETECTOR_PARAMS = ksdet.DetectorParams(
    n_windows=ksdet.DETECTOR_N_WINDOWS,
    d_cut=ksdet.DETECTOR_D_CUT,
    n_cut=ksdet.DETECTOR_N_CUT,
    sigma_cut=ksdet.DETECTOR_SIGMA_CUT,
    change_cut=ksdet.DETECTOR_CHANGE_CUT,
)

EVENT_SETTINGS = ksdet.EVENT_SETTINGS


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


@dataclass(frozen=True)
class ResidualModel:
    method: str
    mean: float
    variance: float
    std: float
    ar_order: int
    ar_coefficients: np.ndarray
    innovation_std: float
    aicc: float


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def save_figure(fig: plt.Figure, stem: str, write_pdf: bool, out_fig_dir: Path) -> None:
    ensure_dir(out_fig_dir)
    fig.savefig(out_fig_dir / f"{stem}.png", dpi=300, bbox_inches="tight")
    if write_pdf:
        fig.savefig(out_fig_dir / f"{stem}.pdf", bbox_inches="tight")
    plt.close(fig)


def lowpass_component(values: np.ndarray) -> np.ndarray:
    sample_rate_per_kyr = 1.0 / RESAMPLE_STEP_KA
    nyquist = 0.5 * sample_rate_per_kyr
    cutoff_frequency = 1.0 / LOWPASS_CUTOFF_PERIOD_KA
    sos = butter(
        FILTER_ORDER,
        cutoff_frequency / nyquist,
        btype="lowpass",
        output="sos",
    )
    return sosfiltfilt(sos, values)


def build_lowpass_decomposition() -> tuple[pd.DataFrame, pd.DataFrame]:
    composite = ksdet.load_cheng_composite()
    regular = ksdet.regularize_cheng(composite)
    raw = regular["d18o_raw"].to_numpy(dtype=float)
    low = lowpass_component(raw)
    residual = raw - low
    out = regular.copy()
    out["d18o_lowpass_10kyr"] = low
    out["d18o_residual_lt10kyr"] = residual

    on_original = composite.copy()
    on_original["d18o_lowpass_10kyr"] = np.interp(
        on_original["age_ka"].to_numpy(dtype=float),
        out["age_ka"].to_numpy(dtype=float),
        out["d18o_lowpass_10kyr"].to_numpy(dtype=float),
    )
    on_original["d18o_residual_lt10kyr"] = (
        on_original["d18o"] - on_original["d18o_lowpass_10kyr"]
    )
    return out, on_original


def autocorrelation(values: np.ndarray, lag_steps: int) -> float:
    x = np.asarray(values, dtype=float)
    x = x[np.isfinite(x)]
    if lag_steps <= 0 or len(x) <= lag_steps:
        return np.nan
    a = x[:-lag_steps] - np.mean(x[:-lag_steps])
    b = x[lag_steps:] - np.mean(x[lag_steps:])
    denom = np.sqrt(np.sum(a**2) * np.sum(b**2))
    if denom == 0.0:
        return np.nan
    return float(np.sum(a * b) / denom)


def ar_is_stationary(coefficients: np.ndarray) -> bool:
    coeff = np.asarray(coefficients, dtype=float)
    p = len(coeff)
    if p == 0:
        return False
    companion = np.zeros((p, p), dtype=float)
    companion[0, :] = coeff
    if p > 1:
        companion[1:, :-1] = np.eye(p - 1)
    return bool(np.max(np.abs(np.linalg.eigvals(companion))) < 0.999)


def fit_ar_model(centered: np.ndarray, max_order: int = MAX_AR_ORDER) -> ResidualModel:
    """Fit a Gaussian AR(p) residual model using conditional AICc.

    AR(1) is too restrictive for the de-orbital residual because its ACF must
    decay monotonically. Allowing p > 1 keeps a single Gaussian AR null while
    matching the observed short-scale autocorrelation more faithfully.
    """

    x = np.asarray(centered, dtype=float)
    x = x[np.isfinite(x)]
    x = x - np.mean(x)
    variance = float(np.var(x, ddof=1))
    std = float(np.sqrt(variance))
    best: dict | None = None
    max_order = min(int(max_order), max(1, len(x) // 10))
    for order in range(1, max_order + 1):
        y = x[order:]
        predictors = np.column_stack(
            [x[order - lag : len(x) - lag] for lag in range(1, order + 1)]
        )
        coefficients, *_ = np.linalg.lstsq(predictors, y, rcond=None)
        if not ar_is_stationary(coefficients):
            continue
        residual = y - predictors @ coefficients
        n_eff = len(y)
        rss = float(np.sum(residual**2))
        sigma2 = max(rss / n_eff, 1e-12)
        k = order + 1
        aic = n_eff * np.log(sigma2) + 2.0 * k
        aicc = aic + (2.0 * k * (k + 1.0)) / max(n_eff - k - 1.0, 1.0)
        candidate = {
            "order": order,
            "coefficients": coefficients,
            "innovation_std": float(np.sqrt(sigma2)),
            "aicc": float(aicc),
        }
        if best is None or candidate["aicc"] < best["aicc"]:
            best = candidate

    if best is None:
        phi = autocorrelation(x, 1)
        phi = float(np.clip(phi if np.isfinite(phi) else 0.0, -0.98, 0.98))
        best = {
            "order": 1,
            "coefficients": np.array([phi], dtype=float),
            "innovation_std": float(std * np.sqrt(max(1.0 - phi**2, 1e-12))),
            "aicc": np.nan,
        }

    return ResidualModel(
        method="gaussian_arp",
        mean=0.0,
        variance=variance,
        std=std,
        ar_order=int(best["order"]),
        ar_coefficients=np.asarray(best["coefficients"], dtype=float),
        innovation_std=float(best["innovation_std"]),
        aicc=float(best["aicc"]),
    )


def ar_model_acf(coefficients: np.ndarray, max_lag_steps: int) -> np.ndarray:
    coeff = np.asarray(coefficients, dtype=float)
    p = len(coeff)
    acf = np.full(max_lag_steps + 1, np.nan, dtype=float)
    if p == 0:
        return acf
    acf[0] = 1.0
    system = np.eye(p)
    rhs = np.zeros(p, dtype=float)
    for k in range(1, p + 1):
        row = k - 1
        for i, phi in enumerate(coeff, start=1):
            j = abs(k - i)
            if j == 0:
                rhs[row] += phi
            elif j <= p:
                system[row, j - 1] -= phi
    try:
        initial = np.linalg.solve(system, rhs)
    except np.linalg.LinAlgError:
        return acf
    acf[1 : p + 1] = initial
    for lag in range(p + 1, max_lag_steps + 1):
        acf[lag] = float(np.dot(coeff, acf[lag - np.arange(1, p + 1)]))
    return acf


def residual_statistics(decomposition: pd.DataFrame) -> tuple[pd.DataFrame, ResidualModel]:
    age = decomposition["age_ka"].to_numpy(dtype=float)
    residual = decomposition["d18o_residual_lt10kyr"].to_numpy(dtype=float)
    keep = (age >= age.min() + EDGE_TRIM_KA) & (age <= age.max() - EDGE_TRIM_KA)
    x = residual[keep]
    mean = float(np.mean(x))
    centered = x - mean
    variance = float(np.var(centered, ddof=1))
    std = float(np.sqrt(variance))
    model = fit_ar_model(centered, MAX_AR_ORDER)
    model = ResidualModel(
        method=model.method,
        mean=mean,
        variance=variance,
        std=std,
        ar_order=model.ar_order,
        ar_coefficients=model.ar_coefficients,
        innovation_std=model.innovation_std,
        aicc=model.aicc,
    )
    q = np.quantile(centered, [0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99])
    stats = [
        ("lowpass_cutoff_period_ka", LOWPASS_CUTOFF_PERIOD_KA),
        ("filter_order", FILTER_ORDER),
        ("resample_step_ka", RESAMPLE_STEP_KA),
        ("edge_trim_for_stats_ka", EDGE_TRIM_KA),
        ("n_regular_samples_total", int(len(residual))),
        ("n_regular_samples_used_for_stats", int(keep.sum())),
        ("raw_variance_used_interval", float(np.var(decomposition.loc[keep, "d18o_raw"], ddof=1))),
        ("lowpass_variance_used_interval", float(np.var(decomposition.loc[keep, "d18o_lowpass_10kyr"], ddof=1))),
        ("residual_mean", mean),
        ("residual_variance", variance),
        ("residual_std", std),
        ("residual_ar_method", model.method),
        ("residual_ar_max_order_considered", MAX_AR_ORDER),
        ("residual_ar_selected_order", model.ar_order),
        ("residual_ar_innovation_std", model.innovation_std),
        ("residual_ar_aicc", model.aicc),
        ("residual_ar_coefficients", ";".join(f"{value:.10g}" for value in model.ar_coefficients)),
        ("residual_acf_0p2kyr", autocorrelation(centered, 2)),
        ("residual_acf_0p5kyr", autocorrelation(centered, 5)),
        ("residual_acf_1kyr", autocorrelation(centered, int(round(1.0 / RESAMPLE_STEP_KA)))),
        ("residual_acf_2kyr", autocorrelation(centered, int(round(2.0 / RESAMPLE_STEP_KA)))),
        ("residual_q01_centered", float(q[0])),
        ("residual_q05_centered", float(q[1])),
        ("residual_q25_centered", float(q[2])),
        ("residual_q50_centered", float(q[3])),
        ("residual_q75_centered", float(q[4])),
        ("residual_q95_centered", float(q[5])),
        ("residual_q99_centered", float(q[6])),
    ]
    return pd.DataFrame(stats, columns=["statistic", "value"]), model


def generate_ar_residual(n: int, model: ResidualModel, rng: np.random.Generator) -> np.ndarray:
    order = model.ar_order
    coeff = model.ar_coefficients
    burnin = max(1000, 50 * order)
    total = n + burnin
    x = np.zeros(total, dtype=float)
    x[:order] = rng.normal(loc=0.0, scale=model.std, size=order)
    innovations = rng.normal(loc=0.0, scale=model.innovation_std, size=total)
    for idx in range(order, total):
        x[idx] = float(np.dot(coeff, x[idx - np.arange(1, order + 1)])) + innovations[idx]
    x = x[burnin:]
    x -= np.mean(x)
    current_std = np.std(x, ddof=1)
    if current_std > 0.0:
        x *= model.std / current_std
    return x + model.mean


def build_detection_record(
    original_grid: pd.DataFrame,
    regular: pd.DataFrame,
    value_regular: np.ndarray,
    value_col: str,
) -> pd.DataFrame:
    values = np.interp(
        original_grid["age_ka"].to_numpy(dtype=float),
        regular["age_ka"].to_numpy(dtype=float),
        np.asarray(value_regular, dtype=float),
    )
    return pd.DataFrame({"age_ka": original_grid["age_ka"].to_numpy(dtype=float), value_col: values})


def event_frame_from_detection(detected: pd.DataFrame, source: str) -> pd.DataFrame:
    out = detected.copy()
    if out.empty:
        return pd.DataFrame(
            columns=[
                "event_age_ka",
                "event_type",
                "event_label",
                "event_index",
                "source",
            ]
        )
    out["event_age_ka"] = out["age_ka"].astype(float)
    out["event_label"] = out["event_type"].map(
        {event_type: settings["label"] for event_type, settings in EVENT_SETTINGS.items()}
    )
    out["source"] = source
    out = out.sort_values(["event_type", "event_age_ka"]).reset_index(drop=True)
    out["event_index"] = out.groupby("event_type").cumcount() + 1
    return out


def build_precession_phase_product() -> dict[str, rayleigh.OrbitalPhase]:
    return {"pre": rayleigh.build_phase_series("pre", rayleigh.DRIVER_SETTINGS["pre"])}


def rayleigh_summary_for_detection(
    detected: pd.DataFrame,
    source: str,
    phase_products: dict[str, rayleigh.OrbitalPhase],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    events = event_frame_from_detection(detected, source)
    if events.empty:
        rows = []
        for event_type, settings in EVENT_SETTINGS.items():
            rows.append(
                {
                    "driver": "pre",
                    "driver_label": "Precession index",
                    "event_type": event_type,
                    "event_label": settings["label"],
                    "n_events_total": 0,
                    "n_phase_events_used": 0,
                    "n_extrapolated_phase_events": 0,
                    "mean_phase_rad": np.nan,
                    "mean_phase_deg": np.nan,
                    "mean_resultant_length": np.nan,
                    "rayleigh_R": np.nan,
                    "rayleigh_z": np.nan,
                    "rayleigh_p": np.nan,
                }
            )
        return pd.DataFrame(), pd.DataFrame(rows)
    event_phases = rayleigh.sample_event_phases(events, phase_products)
    results = rayleigh.build_rayleigh_results(event_phases)
    return event_phases, results[results["driver"].eq("pre")].reset_index(drop=True)


def run_detection_on_record(
    record: pd.DataFrame,
    value_col: str,
    source: str,
    phase_products: dict[str, rayleigh.OrbitalPhase],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    detected = ksdet.tipes_ks_detection(record, value_col, DETECTOR_PARAMS)
    event_phases, rayleigh_results = rayleigh_summary_for_detection(detected, source, phase_products)
    detected = detected.copy()
    detected["source"] = source
    return detected, event_phases, rayleigh_results


def summarize_null(observed: pd.DataFrame, surrogate: pd.DataFrame, lowpass_only: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for event_type, settings in EVENT_SETTINGS.items():
        obs = observed[observed["event_type"].eq(event_type)].iloc[0]
        sim = surrogate[surrogate["event_type"].eq(event_type)].copy()
        low = lowpass_only[lowpass_only["event_type"].eq(event_type)].iloc[0]
        obs_z = float(obs["rayleigh_z"])
        obs_rbar = float(obs["mean_resultant_length"])
        obs_n = int(obs["n_events_total"])
        sim_z = sim["rayleigh_z"].to_numpy(dtype=float)
        sim_rbar = sim["mean_resultant_length"].to_numpy(dtype=float)
        sim_n = sim["n_events_total"].to_numpy(dtype=float)
        sim_p = sim["rayleigh_p"].to_numpy(dtype=float)
        sim_neg_log10_p = -np.log10(np.maximum(sim_p, 1e-300))
        rows.append(
            {
                "event_type": event_type,
                "event_label": settings["label"],
                "observed_n_events": obs_n,
                "observed_mean_phase_deg": float(obs["mean_phase_deg"]),
                "observed_Rbar": obs_rbar,
                "observed_rayleigh_z": obs_z,
                "observed_rayleigh_p": float(obs["rayleigh_p"]),
                "lowpass_only_n_events": int(low["n_events_total"]),
                "lowpass_only_mean_phase_deg": float(low["mean_phase_deg"]),
                "lowpass_only_Rbar": float(low["mean_resultant_length"]),
                "lowpass_only_rayleigh_p": float(low["rayleigh_p"]),
                "surrogate_n": int(len(sim)),
                "surrogate_event_count_mean": float(np.nanmean(sim_n)),
                "surrogate_event_count_q025": float(np.nanquantile(sim_n, 0.025)),
                "surrogate_event_count_q975": float(np.nanquantile(sim_n, 0.975)),
                "surrogate_Rbar_mean": float(np.nanmean(sim_rbar)),
                "surrogate_Rbar_q95": float(np.nanquantile(sim_rbar, 0.95)),
                "surrogate_rayleigh_z_mean": float(np.nanmean(sim_z)),
                "surrogate_rayleigh_z_q95": float(np.nanquantile(sim_z, 0.95)),
                "surrogate_neg_log10_p_mean": float(np.nanmean(sim_neg_log10_p)),
                "surrogate_neg_log10_p_q95": float(np.nanquantile(sim_neg_log10_p, 0.95)),
                "empirical_p_z_ge_observed": float(
                    (np.sum(sim_z >= obs_z) + 1.0) / (np.sum(np.isfinite(sim_z)) + 1.0)
                ),
                "empirical_p_Rbar_ge_observed": float(
                    (np.sum(sim_rbar >= obs_rbar) + 1.0) / (np.sum(np.isfinite(sim_rbar)) + 1.0)
                ),
                "n_surrogate_rayleigh_p_lt_0p05": int(np.sum(sim_p < 0.05)),
                "fraction_surrogate_rayleigh_p_lt_0p05": float(np.nanmean(sim_p < 0.05)),
                "fraction_surrogate_event_count_ge_observed": float(np.nanmean(sim_n >= obs_n)),
            }
        )
    return pd.DataFrame(rows)


def plot_decomposition(
    decomposition: pd.DataFrame,
    residual_model: ResidualModel,
    write_pdf: bool,
    out_fig_dir: Path,
) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(11.2, 6.6))
    ax = axes[0, 0]
    ax.plot(decomposition["age_ka"], decomposition["d18o_raw"], color="#4a4a4a", lw=0.7, label="Cheng composite")
    ax.plot(
        decomposition["age_ka"],
        decomposition["d18o_lowpass_10kyr"],
        color="#0072B2",
        lw=1.2,
        label="10 kyr low-pass",
    )
    ax.set_ylabel("d18O")
    ax.set_title("Low-frequency component", loc="left")
    ax.legend(frameon=False, loc="upper right")

    ax = axes[0, 1]
    ax.plot(decomposition["age_ka"], decomposition["d18o_residual_lt10kyr"], color="#555555", lw=0.65)
    ax.axhline(0.0, color="#222222", lw=0.7)
    ax.set_ylabel("residual d18O")
    ax.set_title("<10 kyr residual", loc="left")

    ax = axes[1, 0]
    residual = decomposition["d18o_residual_lt10kyr"].to_numpy(dtype=float)
    ax.hist(residual, bins=45, color="#888888", alpha=0.78, density=True)
    ax.set_xlabel("residual d18O")
    ax.set_ylabel("density")
    ax.set_title("Residual distribution", loc="left")

    ax = axes[1, 1]
    max_lag_kyr = 5.0
    lags = np.arange(0, int(round(max_lag_kyr / RESAMPLE_STEP_KA)) + 1)
    acf = [1.0 if lag == 0 else autocorrelation(residual, lag) for lag in lags]
    ax.plot(lags * RESAMPLE_STEP_KA, acf, color="#222222", lw=1.2)
    model_acf = ar_model_acf(residual_model.ar_coefficients, int(lags[-1]))
    ax.plot(
        lags * RESAMPLE_STEP_KA,
        model_acf[lags],
        color="#D55E00",
        lw=1.0,
        ls="--",
        label=f"AR({residual_model.ar_order}) fit",
    )
    ax.axhline(0.0, color="#888888", lw=0.7)
    ax.set_xlabel("lag (kyr)")
    ax.set_ylabel("ACF")
    ax.set_title("Short-scale autocorrelation", loc="left")
    ax.legend(frameon=False, loc="upper right")

    for idx, ax in enumerate(axes.flat):
        ax.grid(False)
        ax.text(
            -0.12,
            1.04,
            chr(ord("a") + idx),
            transform=ax.transAxes,
            fontweight="bold",
            fontsize=12,
            va="bottom",
        )
    fig.subplots_adjust(left=0.08, right=0.98, bottom=0.10, top=0.93, hspace=0.36, wspace=0.28)
    save_figure(fig, "fig01_lowpass_residual_diagnostics", write_pdf, out_fig_dir)


def plot_surrogate_examples(
    original_grid: pd.DataFrame,
    regular: pd.DataFrame,
    example_records: list[tuple[str, np.ndarray, pd.DataFrame]],
    write_pdf: bool,
    out_fig_dir: Path,
) -> None:
    fig, axes = plt.subplots(len(example_records), 1, figsize=(11.5, 5.8), sharex=True)
    if len(example_records) == 1:
        axes = [axes]
    for ax, (label, values, detected) in zip(axes, example_records):
        ax.plot(regular["age_ka"], regular["d18o_raw"], color="#bdbdbd", lw=0.6, label="observed raw")
        ax.plot(regular["age_ka"], regular["d18o_lowpass_10kyr"], color="#0072B2", lw=1.0, label="10 kyr low-pass")
        ax.plot(regular["age_ka"], values, color="#222222", lw=0.65, label=label)
        ymin, ymax = ax.get_ylim()
        for event_type, settings in EVENT_SETTINGS.items():
            sub = detected[detected["event_type"].eq(event_type)]
            ax.vlines(sub["age_ka"], ymin, ymax, color=settings["color"], lw=0.45, alpha=0.5)
        ax.set_ylim(ymin, ymax)
        ax.set_ylabel("d18O")
        ax.grid(False)
    handles = [
        Line2D([0], [0], color="#bdbdbd", lw=0.8, label="observed raw"),
        Line2D([0], [0], color="#0072B2", lw=1.0, label="10 kyr low-pass"),
        Line2D([0], [0], color="#222222", lw=0.8, label="surrogate"),
        Line2D([0], [0], color=EVENT_SETTINGS["strong_monsoon_start"]["color"], lw=1.0, label="strong starts"),
        Line2D([0], [0], color=EVENT_SETTINGS["weak_monsoon_start"]["color"], lw=1.0, label="weak starts"),
    ]
    axes[0].legend(handles=handles, frameon=False, loc="upper right", ncol=5)
    axes[-1].set_xlabel("Age (kyr BP)")
    fig.subplots_adjust(left=0.07, right=0.98, bottom=0.11, top=0.94, hspace=0.16)
    save_figure(fig, "fig02_example_surrogate_records", write_pdf, out_fig_dir)


def plot_surrogate_null(
    observed: pd.DataFrame,
    surrogate: pd.DataFrame,
    lowpass_only: pd.DataFrame,
    write_pdf: bool,
    out_fig_dir: Path,
) -> None:
    fig, axes = plt.subplots(3, 2, figsize=(10.8, 8.4))
    event_order = ["strong_monsoon_start", "weak_monsoon_start"]
    observed_color = "#C51B7D"
    metrics = [
        ("n_events_total", "Detected events"),
        ("mean_resultant_length", "Rayleigh Rbar"),
        ("neg_log10_rayleigh_p", "-log10 Rayleigh p"),
    ]
    for row_idx, (metric, ylabel) in enumerate(metrics):
        for col_idx, event_type in enumerate(event_order):
            ax = axes[row_idx, col_idx]
            settings = EVENT_SETTINGS[event_type]
            sim = surrogate[surrogate["event_type"].eq(event_type)]
            obs = observed[observed["event_type"].eq(event_type)].iloc[0]
            low = lowpass_only[lowpass_only["event_type"].eq(event_type)].iloc[0]
            if metric == "neg_log10_rayleigh_p":
                values = -np.log10(np.maximum(sim["rayleigh_p"].to_numpy(dtype=float), 1e-300))
                obs_value = -np.log10(max(float(obs["rayleigh_p"]), 1e-300))
                low_value = -np.log10(max(float(low["rayleigh_p"]), 1e-300))
            else:
                values = sim[metric].to_numpy(dtype=float)
                obs_value = float(obs[metric])
                low_value = float(low[metric])
            values = values[np.isfinite(values)]
            ax.hist(values, bins=32, color="#a6bddb", edgecolor="white", alpha=0.9)
            ax.axvline(obs_value, color=observed_color, lw=1.8)
            if np.isfinite(low_value):
                ax.axvline(low_value, color="#222222", lw=1.2, ls="--")
            if metric == "neg_log10_rayleigh_p":
                ax.axvline(-np.log10(0.05), color="#777777", lw=1.0, ls=":")
            ax.set_ylabel("surrogate count")
            ax.set_xlabel(ylabel)
            ax.set_title(settings["label"], loc="left")
            ax.grid(False)
            if row_idx == 0 and col_idx == 1:
                handles = [
                    Line2D([0], [0], color="#a6bddb", lw=5.0, alpha=0.9, label="AR(p) surrogates"),
                    Line2D([0], [0], color=observed_color, lw=1.8, label="observed raw"),
                    Line2D([0], [0], color="#222222", lw=1.2, ls="--", label="low-pass only"),
                    Line2D([0], [0], color="#777777", lw=1.0, ls=":", label="p=0.05"),
                ]
                ax.legend(handles=handles, frameon=False, loc="upper right")
            ax.text(
                -0.16,
                1.04,
                chr(ord("a") + row_idx * 2 + col_idx),
                transform=ax.transAxes,
                fontweight="bold",
                fontsize=12,
                va="bottom",
            )
    fig.subplots_adjust(left=0.08, right=0.98, bottom=0.08, top=0.95, hspace=0.44, wspace=0.28)
    save_figure(fig, "fig03_surrogate_null_distributions", write_pdf, out_fig_dir)


def write_outputs(
    decomposition: pd.DataFrame,
    residual_stats: pd.DataFrame,
    observed_detected: pd.DataFrame,
    lowpass_detected: pd.DataFrame,
    surrogate_detected: pd.DataFrame,
    observed_rayleigh: pd.DataFrame,
    lowpass_rayleigh: pd.DataFrame,
    surrogate_rayleigh: pd.DataFrame,
    null_summary: pd.DataFrame,
    n_surrogates: int,
    seed: int,
) -> None:
    ensure_dir(OUT_DATA_DIR)
    decomposition.to_csv(OUT_DATA_DIR / "cheng_10kyr_lowpass_residual_decomposition.csv", index=False)
    residual_stats.to_csv(OUT_DATA_DIR / "cheng_lt10kyr_residual_statistics.csv", index=False)
    observed_detected.to_csv(OUT_DATA_DIR / "observed_raw_tipes_detections.csv", index=False)
    lowpass_detected.to_csv(OUT_DATA_DIR / "lowpass_only_tipes_detections.csv", index=False)
    surrogate_detected.to_csv(OUT_DATA_DIR / "surrogate_tipes_detections.csv", index=False)
    observed_rayleigh.to_csv(OUT_DATA_DIR / "observed_raw_rayleigh_precession.csv", index=False)
    lowpass_rayleigh.to_csv(OUT_DATA_DIR / "lowpass_only_rayleigh_precession.csv", index=False)
    surrogate_rayleigh.to_csv(OUT_DATA_DIR / "surrogate_rayleigh_precession.csv", index=False)
    null_summary.to_csv(OUT_DATA_DIR / "surrogate_null_summary.csv", index=False)
    pd.DataFrame(
        [
            {
                "run_name": RUN_NAME,
                "lowpass_cutoff_period_ka": LOWPASS_CUTOFF_PERIOD_KA,
                "filter_order": FILTER_ORDER,
                "resample_step_ka": RESAMPLE_STEP_KA,
                "edge_trim_for_residual_stats_ka": EDGE_TRIM_KA,
                "n_surrogates": n_surrogates,
                "random_seed": seed,
                "surrogate_method": "gaussian_arp",
                "max_ar_order": MAX_AR_ORDER,
                "detector_n_windows": DETECTOR_PARAMS.n_windows,
                "detector_d_cut": DETECTOR_PARAMS.d_cut,
                "detector_n_cut": DETECTOR_PARAMS.n_cut,
                "detector_sigma_cut": DETECTOR_PARAMS.sigma_cut,
                "detector_change_cut": DETECTOR_PARAMS.change_cut,
                "information_leakage_control": (
                    "No published Rousseau transition ages are used. The observed reference "
                    "is the raw Cheng composite processed with the same TiPES-style detector."
                ),
            }
        ]
    ).to_csv(OUT_DATA_DIR / "parameters.csv", index=False)


def run_analysis(
    n_surrogates: int = N_SURROGATES,
    seed: int = RANDOM_SEED,
    write_pdf: bool = True,
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    regular, original_grid = build_lowpass_decomposition()
    residual_stats, residual_model = residual_statistics(regular)
    phase_products = build_precession_phase_product()

    observed_record = original_grid[["age_ka", "d18o"]].rename(columns={"d18o": "d18o_observed"})
    lowpass_record = original_grid[["age_ka", "d18o_lowpass_10kyr"]].rename(
        columns={"d18o_lowpass_10kyr": "d18o_lowpass"}
    )
    observed_detected, _, observed_rayleigh = run_detection_on_record(
        observed_record,
        "d18o_observed",
        "observed_raw",
        phase_products,
    )
    lowpass_detected, _, lowpass_rayleigh = run_detection_on_record(
        lowpass_record,
        "d18o_lowpass",
        "lowpass_only",
        phase_products,
    )
    observed_rayleigh = observed_rayleigh.copy()
    observed_rayleigh.insert(0, "source", "observed_raw")
    lowpass_rayleigh = lowpass_rayleigh.copy()
    lowpass_rayleigh.insert(0, "source", "lowpass_only")

    surrogate_detection_rows = []
    surrogate_rayleigh_rows = []
    example_records: list[tuple[str, np.ndarray, pd.DataFrame]] = []
    low = regular["d18o_lowpass_10kyr"].to_numpy(dtype=float)

    for sim_id in range(1, n_surrogates + 1):
        synthetic_residual = generate_ar_residual(len(regular), residual_model, rng)
        surrogate_values = low + synthetic_residual
        source = f"surrogate_{sim_id:04d}"
        record = build_detection_record(original_grid, regular, surrogate_values, "d18o_surrogate")
        detected, _, rayleigh_results = run_detection_on_record(
            record,
            "d18o_surrogate",
            source,
            phase_products,
        )
        detected = detected.copy()
        detected.insert(0, "surrogate_id", sim_id)
        surrogate_detection_rows.append(detected)
        rayleigh_results = rayleigh_results.copy()
        rayleigh_results.insert(0, "surrogate_id", sim_id)
        rayleigh_results.insert(1, "source", source)
        surrogate_rayleigh_rows.append(rayleigh_results)
        if len(example_records) < 2:
            example_records.append((source, surrogate_values, detected))

    surrogate_detected = (
        pd.concat(surrogate_detection_rows, ignore_index=True)
        if surrogate_detection_rows
        else pd.DataFrame()
    )
    surrogate_rayleigh = pd.concat(surrogate_rayleigh_rows, ignore_index=True)
    null_summary = summarize_null(observed_rayleigh, surrogate_rayleigh, lowpass_rayleigh)

    write_outputs(
        regular,
        residual_stats,
        observed_detected,
        lowpass_detected,
        surrogate_detected,
        observed_rayleigh,
        lowpass_rayleigh,
        surrogate_rayleigh,
        null_summary,
        n_surrogates,
        seed,
    )
    plot_decomposition(regular, residual_model, write_pdf, OUT_FIG_DIR)
    plot_surrogate_examples(original_grid, regular, example_records, write_pdf, OUT_FIG_DIR)
    plot_surrogate_null(observed_rayleigh, surrogate_rayleigh, lowpass_rayleigh, write_pdf, OUT_FIG_DIR)
    return null_summary


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Build low-pass-plus-surrogate Cheng records and test whether the "
            "TiPES KS detector creates precession-phase structure by itself."
        )
    )
    parser.add_argument("--n-surrogates", type=int, default=N_SURROGATES)
    parser.add_argument("--seed", type=int, default=RANDOM_SEED)
    parser.add_argument("--no-pdf", action="store_true", help="Skip PDF figure output.")
    args = parser.parse_args()

    summary = run_analysis(
        n_surrogates=args.n_surrogates,
        seed=args.seed,
        write_pdf=not args.no_pdf,
    )
    cols = [
        "event_type",
        "observed_n_events",
        "observed_mean_phase_deg",
        "observed_Rbar",
        "observed_rayleigh_p",
        "lowpass_only_n_events",
        "lowpass_only_Rbar",
        "lowpass_only_rayleigh_p",
        "surrogate_event_count_mean",
        "surrogate_Rbar_q95",
        "surrogate_neg_log10_p_q95",
        "empirical_p_z_ge_observed",
        "fraction_surrogate_rayleigh_p_lt_0p05",
    ]
    print("Low-pass surrogate artifact-null summary:")
    print(summary[cols].to_string(index=False))
    print(f"\nWrote tables to: {OUT_DATA_DIR.relative_to(PROJECT_ROOT)}")
    print(f"Wrote figures to: {OUT_FIG_DIR.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    main()
