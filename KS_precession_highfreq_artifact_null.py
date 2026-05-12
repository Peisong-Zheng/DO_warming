"""
Simplified precession-band artifact null for the Cheng composite.

This script asks a narrow diagnostic question:

    Can a record made only from the precession-band component of the Cheng
    composite plus sub-kyr high-frequency residuals produce KS-detected
    transitions with apparent precession-phase organization?

Unlike the broader low-pass surrogate experiment, this null deliberately
removes the complex multi-orbital and glacial-scale background. It is intended
as a visual and statistical sanity check for the reviewer concern that the KS
algorithm might detect the smooth precession-band waveform itself.

No published Rousseau transition ages are used.
"""

from __future__ import annotations

import argparse
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
RUN_NAME = "rousseau2023_monsoon_ks_precession_highfreq_artifact_null"
OUT_DATA_DIR = PROJECT_ROOT / "data" / "processed" / RUN_NAME
OUT_FIG_DIR = PROJECT_ROOT / "figures" / RUN_NAME

RESAMPLE_STEP_KA = ksdet.RESAMPLE_STEP_KA
PRECESSION_BAND_KYR = (17.0, 25.0)
HIGHFREQ_CUTOFF_PERIOD_KA = 1.0
FILTER_ORDER = 4

DETECTOR_PARAMS = ksdet.DetectorParams(
    n_windows=ksdet.DETECTOR_N_WINDOWS,
    d_cut=ksdet.DETECTOR_D_CUT,
    n_cut=ksdet.DETECTOR_N_CUT,
    sigma_cut=ksdet.DETECTOR_SIGMA_CUT,
    change_cut=ksdet.DETECTOR_CHANGE_CUT,
)

COMPONENT_SETTINGS = {
    "observed_raw": {
        "label": "Observed raw composite",
        "value_col": "d18o_raw",
        "color": "#4a4a4a",
    },
    "precession_band": {
        "label": "17-25 kyr precession band",
        "value_col": "d18o_precession_band",
        "color": "#0072B2",
    },
    "highfreq_lt1kyr": {
        "label": "<1 kyr high-frequency component",
        "value_col": "d18o_highfreq_lt1kyr",
        "color": "#009E73",
    },
    "precession_plus_highfreq": {
        "label": "Precession band + <1 kyr component",
        "value_col": "d18o_precession_plus_highfreq",
        "color": "#D55E00",
    },
}

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


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def save_figure(fig: plt.Figure, stem: str, write_pdf: bool) -> None:
    ensure_dir(OUT_FIG_DIR)
    fig.savefig(OUT_FIG_DIR / f"{stem}.png", dpi=300, bbox_inches="tight")
    if write_pdf:
        fig.savefig(OUT_FIG_DIR / f"{stem}.pdf", bbox_inches="tight")
    plt.close(fig)


def butter_filter(values: np.ndarray, kind: str, cutoff_periods_kyr: float | tuple[float, float]) -> np.ndarray:
    sample_rate_per_kyr = 1.0 / RESAMPLE_STEP_KA
    nyquist = 0.5 * sample_rate_per_kyr
    if kind == "highpass":
        cutoff_frequency = 1.0 / float(cutoff_periods_kyr)
        wn = cutoff_frequency / nyquist
    elif kind == "bandpass":
        low_period, high_period = cutoff_periods_kyr
        if low_period <= 0.0 or high_period <= low_period:
            raise ValueError("Bandpass periods must be ordered as (short_period, long_period).")
        low_frequency = 1.0 / high_period
        high_frequency = 1.0 / low_period
        wn = [low_frequency / nyquist, high_frequency / nyquist]
    else:
        raise ValueError("kind must be 'highpass' or 'bandpass'.")
    sos = butter(FILTER_ORDER, wn, btype=kind, output="sos")
    return sosfiltfilt(sos, values)


def build_component_series() -> tuple[pd.DataFrame, pd.DataFrame]:
    composite = ksdet.load_cheng_composite()
    regular = ksdet.regularize_cheng(composite)
    raw = regular["d18o_raw"].to_numpy(dtype=float)
    mean_raw = float(np.mean(raw))
    precession = butter_filter(raw, "bandpass", PRECESSION_BAND_KYR)
    highfreq = butter_filter(raw, "highpass", HIGHFREQ_CUTOFF_PERIOD_KA)
    combined_centered = precession + highfreq

    out = regular.copy()
    out["d18o_precession_band"] = precession
    out["d18o_highfreq_lt1kyr"] = highfreq
    # Add the raw mean only for visual comparability. KS detections are
    # invariant to this constant shift because they compare local distributions.
    out["d18o_precession_plus_highfreq"] = combined_centered + mean_raw
    out["d18o_precession_band_plus_mean"] = precession + mean_raw
    out["d18o_highfreq_lt1kyr_plus_mean"] = highfreq + mean_raw

    on_original = composite.rename(columns={"d18o": "d18o_raw"}).copy()
    for component_id, settings in COMPONENT_SETTINGS.items():
        col = settings["value_col"]
        if col == "d18o_raw":
            continue
        on_original[col] = np.interp(
            on_original["age_ka"].to_numpy(dtype=float),
            out["age_ka"].to_numpy(dtype=float),
            out[col].to_numpy(dtype=float),
        )
    return out, on_original


def event_frame_from_detection(detected: pd.DataFrame, source: str) -> pd.DataFrame:
    out = detected.copy()
    if out.empty:
        return pd.DataFrame(
            columns=["event_age_ka", "event_type", "event_label", "event_index", "source"]
        )
    out["event_age_ka"] = out["age_ka"].astype(float)
    out["event_label"] = out["event_type"].map(
        {event_type: settings["label"] for event_type, settings in EVENT_SETTINGS.items()}
    )
    out["source"] = source
    out = out.sort_values(["event_type", "event_age_ka"]).reset_index(drop=True)
    out["event_index"] = out.groupby("event_type").cumcount() + 1
    return out


def run_rayleigh_for_detection(detected: pd.DataFrame, component_id: str) -> pd.DataFrame:
    events = event_frame_from_detection(detected, component_id)
    rows = []
    if events.empty:
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
        results = pd.DataFrame(rows)
    else:
        phase_products = {"pre": rayleigh.build_phase_series("pre", rayleigh.DRIVER_SETTINGS["pre"])}
        event_phases = rayleigh.sample_event_phases(events, phase_products)
        results = rayleigh.build_rayleigh_results(event_phases)
        results = results[results["driver"].eq("pre")].reset_index(drop=True)
    results.insert(0, "component_id", component_id)
    results.insert(1, "component_label", COMPONENT_SETTINGS[component_id]["label"])
    return results


def detect_components(component_original: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    detections = []
    rayleigh_results = []
    summaries = []
    for component_id, settings in COMPONENT_SETTINGS.items():
        record = component_original[["age_ka", settings["value_col"]]].rename(
            columns={settings["value_col"]: "value"}
        )
        detected = ksdet.tipes_ks_detection(record, "value", DETECTOR_PARAMS)
        detected = detected.copy()
        detected.insert(0, "component_id", component_id)
        detected.insert(1, "component_label", settings["label"])
        detections.append(detected)

        ray = run_rayleigh_for_detection(detected, component_id)
        rayleigh_results.append(ray)
        for event_type, event_settings in EVENT_SETTINGS.items():
            sub_det = detected[detected["event_type"].eq(event_type)]
            sub_ray = ray[ray["event_type"].eq(event_type)]
            row = {
                "component_id": component_id,
                "component_label": settings["label"],
                "event_type": event_type,
                "event_label": event_settings["label"],
                "n_detected_events": int(len(sub_det)),
            }
            if not sub_ray.empty:
                row.update(
                    {
                        "mean_phase_deg": float(sub_ray["mean_phase_deg"].iloc[0]),
                        "Rbar": float(sub_ray["mean_resultant_length"].iloc[0]),
                        "rayleigh_z": float(sub_ray["rayleigh_z"].iloc[0]),
                        "rayleigh_p": float(sub_ray["rayleigh_p"].iloc[0]),
                    }
                )
            summaries.append(row)
    return (
        pd.concat(detections, ignore_index=True),
        pd.concat(rayleigh_results, ignore_index=True),
        pd.DataFrame(summaries),
    )


def plot_components(component_regular: pd.DataFrame, write_pdf: bool) -> None:
    fig, axes = plt.subplots(4, 1, figsize=(11.5, 8.0), sharex=True)
    plot_specs = [
        ("d18o_raw", "Observed raw composite", "#4a4a4a"),
        ("d18o_precession_band_plus_mean", "17-25 kyr precession band + mean", "#0072B2"),
        ("d18o_highfreq_lt1kyr_plus_mean", "<1 kyr high-frequency component + mean", "#009E73"),
        ("d18o_precession_plus_highfreq", "Precession band + <1 kyr component + mean", "#D55E00"),
    ]
    for idx, (col, label, color) in enumerate(plot_specs):
        ax = axes[idx]
        ax.plot(component_regular["age_ka"], component_regular[col], color=color, lw=0.8)
        ax.set_ylabel("d18O")
        ax.set_title(label, loc="left")
        ax.grid(False)
        ax.text(
            -0.055,
            1.03,
            chr(ord("a") + idx),
            transform=ax.transAxes,
            fontweight="bold",
            fontsize=12,
            va="bottom",
        )
    axes[-1].set_xlabel("Age (kyr BP)")
    fig.subplots_adjust(left=0.08, right=0.99, bottom=0.08, top=0.96, hspace=0.24)
    save_figure(fig, "fig01_filtered_components", write_pdf)


def plot_component_detections(
    component_regular: pd.DataFrame,
    detections: pd.DataFrame,
    write_pdf: bool,
) -> None:
    fig, axes = plt.subplots(3, 1, figsize=(11.5, 7.0), sharex=True)
    plot_specs = [
        ("precession_band", "d18o_precession_band_plus_mean", "17-25 kyr precession band"),
        ("highfreq_lt1kyr", "d18o_highfreq_lt1kyr_plus_mean", "<1 kyr high-frequency component"),
        (
            "precession_plus_highfreq",
            "d18o_precession_plus_highfreq",
            "Precession band + <1 kyr component",
        ),
    ]
    for idx, (component_id, value_col, title) in enumerate(plot_specs):
        ax = axes[idx]
        color = COMPONENT_SETTINGS[component_id]["color"]
        ax.plot(component_regular["age_ka"], component_regular[value_col], color=color, lw=0.8)
        ymin, ymax = ax.get_ylim()
        for event_type, event_settings in EVENT_SETTINGS.items():
            sub = detections[
                detections["component_id"].eq(component_id) & detections["event_type"].eq(event_type)
            ]
            ax.vlines(
                sub["age_ka"],
                ymin,
                ymax,
                color=event_settings["color"],
                lw=0.5,
                alpha=0.65,
            )
        ax.set_ylim(ymin, ymax)
        ax.set_ylabel("d18O")
        ax.set_title(title, loc="left")
        ax.grid(False)
        ax.text(
            -0.055,
            1.03,
            chr(ord("a") + idx),
            transform=ax.transAxes,
            fontweight="bold",
            fontsize=12,
            va="bottom",
        )
    handles = [
        Line2D([0], [0], color=EVENT_SETTINGS["strong_monsoon_start"]["color"], lw=1.0, label="Strong starts"),
        Line2D([0], [0], color=EVENT_SETTINGS["weak_monsoon_start"]["color"], lw=1.0, label="Weak starts"),
    ]
    axes[0].legend(handles=handles, frameon=False, loc="upper right", ncol=2)
    axes[-1].set_xlabel("Age (kyr BP)")
    fig.subplots_adjust(left=0.08, right=0.99, bottom=0.09, top=0.95, hspace=0.22)
    save_figure(fig, "fig02_component_ks_detections", write_pdf)


def plot_summary(summary: pd.DataFrame, write_pdf: bool) -> None:
    component_order = ["observed_raw", "precession_band", "highfreq_lt1kyr", "precession_plus_highfreq"]
    event_order = ["strong_monsoon_start", "weak_monsoon_start"]
    fig, axes = plt.subplots(1, 3, figsize=(12.0, 3.7))
    x = np.arange(len(component_order))
    width = 0.34
    for event_idx, event_type in enumerate(event_order):
        offset = (-0.5 + event_idx) * width
        sub = summary[summary["event_type"].eq(event_type)].set_index("component_id").reindex(component_order)
        color = EVENT_SETTINGS[event_type]["color"]
        label = EVENT_SETTINGS[event_type]["label"]
        axes[0].bar(x + offset, sub["n_detected_events"], width=width, color=color, alpha=0.82, label=label)
        axes[1].bar(x + offset, sub["Rbar"], width=width, color=color, alpha=0.82)
        axes[2].bar(
            x + offset,
            -np.log10(np.maximum(sub["rayleigh_p"].to_numpy(dtype=float), 1e-300)),
            width=width,
            color=color,
            alpha=0.82,
        )
    axes[0].set_ylabel("Detected events")
    axes[1].set_ylabel("Rayleigh Rbar")
    axes[2].set_ylabel("-log10 Rayleigh p")
    axes[2].axhline(-np.log10(0.05), color="#777777", lw=0.9, ls=":")
    labels = ["Raw", "Pre band", "<1 kyr", "Pre + <1 kyr"]
    for idx, ax in enumerate(axes):
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=25, ha="right")
        ax.grid(False)
        ax.text(
            -0.16,
            1.04,
            chr(ord("a") + idx),
            transform=ax.transAxes,
            fontweight="bold",
            fontsize=12,
            va="bottom",
        )
    axes[0].legend(frameon=False, loc="upper right")
    fig.subplots_adjust(left=0.08, right=0.99, bottom=0.23, top=0.90, wspace=0.34)
    save_figure(fig, "fig03_component_rayleigh_summary", write_pdf)


def write_outputs(
    component_regular: pd.DataFrame,
    component_original: pd.DataFrame,
    detections: pd.DataFrame,
    rayleigh_results: pd.DataFrame,
    summary: pd.DataFrame,
) -> None:
    ensure_dir(OUT_DATA_DIR)
    component_regular.to_csv(OUT_DATA_DIR / "component_series_regular_0p1kyr.csv", index=False)
    component_original.to_csv(OUT_DATA_DIR / "component_series_original_age_grid.csv", index=False)
    detections.to_csv(OUT_DATA_DIR / "component_tipes_detections.csv", index=False)
    rayleigh_results.to_csv(OUT_DATA_DIR / "component_rayleigh_precession.csv", index=False)
    summary.to_csv(OUT_DATA_DIR / "component_summary.csv", index=False)
    pd.DataFrame(
        [
            {
                "run_name": RUN_NAME,
                "precession_band_short_period_ka": PRECESSION_BAND_KYR[0],
                "precession_band_long_period_ka": PRECESSION_BAND_KYR[1],
                "highfreq_cutoff_period_ka": HIGHFREQ_CUTOFF_PERIOD_KA,
                "filter_order": FILTER_ORDER,
                "resample_step_ka": RESAMPLE_STEP_KA,
                "detector_n_windows": DETECTOR_PARAMS.n_windows,
                "detector_d_cut": DETECTOR_PARAMS.d_cut,
                "detector_n_cut": DETECTOR_PARAMS.n_cut,
                "detector_sigma_cut": DETECTOR_PARAMS.sigma_cut,
                "detector_change_cut": DETECTOR_PARAMS.change_cut,
                "information_leakage_control": "No published Rousseau transition ages are used.",
            }
        ]
    ).to_csv(OUT_DATA_DIR / "parameters.csv", index=False)


def run_analysis(write_pdf: bool = True) -> pd.DataFrame:
    component_regular, component_original = build_component_series()
    detections, rayleigh_results, summary = detect_components(component_original)
    write_outputs(component_regular, component_original, detections, rayleigh_results, summary)
    plot_components(component_regular, write_pdf)
    plot_component_detections(component_regular, detections, write_pdf)
    plot_summary(summary, write_pdf)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run a precession-band plus sub-kyr high-frequency KS artifact null."
    )
    parser.add_argument("--no-pdf", action="store_true", help="Skip PDF figure output.")
    args = parser.parse_args()

    summary = run_analysis(write_pdf=not args.no_pdf)
    cols = ["component_id", "event_type", "n_detected_events", "mean_phase_deg", "Rbar", "rayleigh_p"]
    print("Precession-band plus high-frequency artifact-null summary:")
    print(summary[cols].to_string(index=False))
    print(f"\nWrote tables to: {OUT_DATA_DIR.relative_to(PROJECT_ROOT)}")
    print(f"Wrote figures to: {OUT_FIG_DIR.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    main()
