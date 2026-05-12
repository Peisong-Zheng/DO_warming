"""
Test whether Rousseau et al. (2023) strong/weak monsoon starts have preferred
precession or obliquity phases.

Phase convention:
- local minima of an orbital series are phase 0;
- local maxima are phase pi;
- phase is linearly interpolated between successive extrema and wrapped to
  [0, 2*pi).

This convention intentionally avoids assigning a climatic meaning to the sign
of the raw orbital index. The figures make the extrema and interpolation easy
to inspect before interpreting the Rayleigh tests.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.signal import find_peaks


PROJECT_ROOT = Path(__file__).resolve().parent
RUN_NAME = "rousseau2023_monsoon_orbital_phase_rayleigh"
OUT_DATA_DIR = PROJECT_ROOT / "data" / "processed" / RUN_NAME
OUT_FIG_DIR = PROJECT_ROOT / "figures" / RUN_NAME

STRONG_CSV = PROJECT_ROOT / "data/raw/rousseau_2023_ks_0p4_4kyr_strong_monsoon_start_times.csv"
WEAK_CSV = PROJECT_ROOT / "data/raw/rousseau_2023_ks_0p4_4kyr_weak_monsoon_start_times.csv"
PRE_TXT = PROJECT_ROOT / "data/raw/pre_800_inter100.txt"
OBL_TXT = PROJECT_ROOT / "data/raw/obl_800_inter100.txt"

ANALYSIS_START_KA = 0.0
ANALYSIS_END_KA = 640.0
RAYLEIGH_ALPHA = 0.05

DRIVER_SETTINGS = {
    "pre": {
        "label": "Precession index",
        "path": PRE_TXT,
        "color": "#0072B2",
    },
    "obl": {
        "label": "Obliquity",
        "path": OBL_TXT,
        "color": "#009E73",
    },
}

EVENT_COLORS = {
    "strong_monsoon_start": "#d95f02",
    "weak_monsoon_start": "#6a3d9a",
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
class OrbitalPhase:
    driver: str
    label: str
    series: pd.DataFrame
    extrema: pd.DataFrame


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def load_rousseau_events(path: Path, event_type: str, label: str) -> pd.DataFrame:
    df = pd.read_csv(path, encoding="utf-8-sig")
    if "start_time_ka_BP" not in df.columns:
        raise ValueError(f"{path} must contain start_time_ka_BP.")
    out = df.copy()
    out["event_age_ka"] = pd.to_numeric(out["start_time_ka_BP"], errors="coerce")
    out = out.dropna(subset=["event_age_ka"])
    out = out[
        (out["event_age_ka"] >= ANALYSIS_START_KA)
        & (out["event_age_ka"] <= ANALYSIS_END_KA)
    ].copy()
    out = out.sort_values("event_age_ka").reset_index(drop=True)
    out["event_index"] = np.arange(1, len(out) + 1)
    out["event_type"] = event_type
    out["event_label"] = label
    out["source"] = str(path.relative_to(PROJECT_ROOT))
    return out


def load_all_events() -> pd.DataFrame:
    return pd.concat(
        [
            load_rousseau_events(
                STRONG_CSV,
                "strong_monsoon_start",
                "Strong monsoon starts",
            ),
            load_rousseau_events(
                WEAK_CSV,
                "weak_monsoon_start",
                "Weak monsoon starts",
            ),
        ],
        ignore_index=True,
    )


def load_orbital_series(path: Path, driver: str, label: str) -> pd.DataFrame:
    raw = pd.read_csv(path, sep=r"\s+", header=None, names=["age_raw_ka", "value"])
    out = pd.DataFrame(
        {
            "driver": driver,
            "driver_label": label,
            "age_ka": pd.to_numeric(raw["age_raw_ka"], errors="coerce").abs(),
            "value": pd.to_numeric(raw["value"], errors="coerce"),
            "source": str(path.relative_to(PROJECT_ROOT)),
        }
    )
    out = out.dropna(subset=["age_ka", "value"])
    out = out.groupby(["driver", "driver_label", "age_ka", "source"], as_index=False)["value"].mean()
    out = out.sort_values("age_ka").reset_index(drop=True)
    return out


def enforce_alternating_extrema(extrema: pd.DataFrame) -> pd.DataFrame:
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
            if float(row["value"]) > float(prev["value"]):
                rows[-1] = row.copy()
        else:
            if float(row["value"]) < float(prev["value"]):
                rows[-1] = row.copy()
    out = pd.DataFrame(rows).reset_index(drop=True)
    out["half_cycle_index"] = np.arange(len(out))
    return out


def detect_extrema(orbital: pd.DataFrame) -> pd.DataFrame:
    value = orbital["value"].to_numpy(dtype=float)
    max_idx, _ = find_peaks(value)
    min_idx, _ = find_peaks(-value)

    maxima = orbital.iloc[max_idx].copy()
    maxima["extremum_type"] = "maximum"
    minima = orbital.iloc[min_idx].copy()
    minima["extremum_type"] = "minimum"

    extrema = pd.concat([minima, maxima], ignore_index=True).sort_values("age_ka")
    extrema = enforce_alternating_extrema(extrema)
    if len(extrema) < 3:
        raise ValueError(f"Too few extrema detected for {orbital['driver'].iloc[0]}.")
    return extrema.reset_index(drop=True)


def assign_anchor_phases(extrema: pd.DataFrame) -> pd.DataFrame:
    out = extrema.copy().reset_index(drop=True)
    first_phase = 0.0 if out.loc[0, "extremum_type"] == "minimum" else np.pi
    out["anchor_phase_unwrapped_rad"] = first_phase + np.arange(len(out), dtype=float) * np.pi
    out["anchor_phase_rad"] = np.mod(out["anchor_phase_unwrapped_rad"], 2.0 * np.pi)
    out["anchor_phase_deg"] = np.degrees(out["anchor_phase_rad"])
    return out


def interpolate_unwrapped_phase(
    ages_ka: np.ndarray,
    extrema_age_ka: np.ndarray,
    extrema_phase_unwrapped_rad: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    ages = np.asarray(ages_ka, dtype=float)
    x = np.asarray(extrema_age_ka, dtype=float)
    y = np.asarray(extrema_phase_unwrapped_rad, dtype=float)
    phase = np.interp(ages, x, y)
    extrapolated = (ages < x[0]) | (ages > x[-1])

    if len(x) >= 2:
        left = ages < x[0]
        if np.any(left):
            slope = (y[1] - y[0]) / (x[1] - x[0])
            phase[left] = y[0] + slope * (ages[left] - x[0])
        right = ages > x[-1]
        if np.any(right):
            slope = (y[-1] - y[-2]) / (x[-1] - x[-2])
            phase[right] = y[-1] + slope * (ages[right] - x[-1])

    return phase, extrapolated


def build_phase_series(driver: str, settings: dict) -> OrbitalPhase:
    orbital = load_orbital_series(settings["path"], driver, settings["label"])
    extrema = detect_extrema(orbital)
    extrema = assign_anchor_phases(extrema)
    phase_unwrapped, phase_extrapolated = interpolate_unwrapped_phase(
        orbital["age_ka"].to_numpy(dtype=float),
        extrema["age_ka"].to_numpy(dtype=float),
        extrema["anchor_phase_unwrapped_rad"].to_numpy(dtype=float),
    )
    orbital = orbital.copy()
    orbital["phase_unwrapped_rad"] = phase_unwrapped
    orbital["phase_rad"] = np.mod(phase_unwrapped, 2.0 * np.pi)
    orbital["phase_deg"] = np.degrees(orbital["phase_rad"])
    orbital["phase_extrapolated"] = phase_extrapolated
    return OrbitalPhase(driver=driver, label=settings["label"], series=orbital, extrema=extrema)


def circular_distance_to_phase(theta: np.ndarray, target: float) -> np.ndarray:
    return np.angle(np.exp(1j * (theta - target)))


def sample_event_phases(events: pd.DataFrame, phase_products: dict[str, OrbitalPhase]) -> pd.DataFrame:
    rows = []
    for phase_product in phase_products.values():
        series = phase_product.series
        ext = phase_product.extrema
        event_phase_unwrapped, event_extrapolated = interpolate_unwrapped_phase(
            events["event_age_ka"].to_numpy(dtype=float),
            ext["age_ka"].to_numpy(dtype=float),
            ext["anchor_phase_unwrapped_rad"].to_numpy(dtype=float),
        )
        event_phase = np.mod(event_phase_unwrapped, 2.0 * np.pi)
        value_at_event = np.interp(
            events["event_age_ka"].to_numpy(dtype=float),
            series["age_ka"].to_numpy(dtype=float),
            series["value"].to_numpy(dtype=float),
        )
        for idx, (_, event) in enumerate(events.iterrows()):
            rows.append(
                {
                    "driver": phase_product.driver,
                    "driver_label": phase_product.label,
                    "event_type": event["event_type"],
                    "event_label": event["event_label"],
                    "event_index": int(event["event_index"]),
                    "event_age_ka": float(event["event_age_ka"]),
                    "orbital_value_at_event": float(value_at_event[idx]),
                    "phase_unwrapped_rad": float(event_phase_unwrapped[idx]),
                    "phase_rad": float(event_phase[idx]),
                    "phase_deg": float(np.degrees(event_phase[idx])),
                    "phase_fraction": float(event_phase[idx] / (2.0 * np.pi)),
                    "phase_extrapolated": bool(event_extrapolated[idx]),
                    "signed_distance_to_min_rad": float(circular_distance_to_phase(event_phase[idx], 0.0)),
                    "signed_distance_to_max_rad": float(circular_distance_to_phase(event_phase[idx], np.pi)),
                    "source_event_order": event.get("order", np.nan),
                }
            )
    return pd.DataFrame(rows)


def rayleigh_p_value_from_z(z: float, n: int) -> float:
    """Rayleigh p value using the finite-n correction used in this script.

    In the Rayleigh test, the null hypothesis is circular uniformity. The test
    statistic used below is z = n * Rbar^2, where Rbar is the mean resultant
    length of the event-phase vectors. For large n, the upper-tail probability
    is approximately exp(-z). The extra terms in the parentheses are a standard
    finite-sample correction to that asymptotic p value.
    """
    if n <= 0 or not np.isfinite(z):
        return np.nan

    # Large-sample Rayleigh approximation:
    #     p ≈ exp(-z)
    #
    # This is the probability, under uniform phases, of obtaining a resultant
    # vector at least as concentrated as the observed one. It is accurate when n
    # is large, but our event catalogues have only ~100 events, so we apply the
    # usual expansion in powers of 1/n:
    #
    #     p ≈ exp(-z) * [1 + O(1/n) + O(1/n^2)]
    #
    # The first correction term is (2z - z^2)/(4n), and the second correction
    # term is the polynomial divided by 288 n^2. These terms slightly adjust the
    # p value for finite event counts without changing the Rayleigh statistic z.
    # Reference: 
    # Fisher, N. I. (1993). Statistical Analysis of Circular Data. Cambridge University Press.
    # https://metricgate.com/docs/rayleigh-uniformity-test/
    p = np.exp(-z) * (
        1.0
        + (2.0 * z - z**2) / (4.0 * n)
        - (24.0 * z - 132.0 * z**2 + 76.0 * z**3 - 9.0 * z**4) / (288.0 * n**2)
    )
    # Numerical approximations can very rarely fall just outside [0, 1]; clip to
    # keep the returned value interpretable as a probability.
    return float(np.clip(p, 0.0, 1.0))


def rayleigh_rbar_threshold(n: int, alpha: float = RAYLEIGH_ALPHA) -> float:
    """Critical mean resultant length for a Rayleigh test at p <= alpha."""
    if n <= 0:
        return np.nan

    low = 0.0
    high = 1.0
    if rayleigh_p_value_from_z(n * high**2, n) > alpha:
        return np.nan

    for _ in range(80):
        mid = 0.5 * (low + high)
        p_mid = rayleigh_p_value_from_z(n * mid**2, n)
        if p_mid <= alpha:
            high = mid
        else:
            low = mid
    return float(high)


def rayleigh_test(phases_rad: np.ndarray) -> dict[str, float]:
    theta = np.asarray(phases_rad, dtype=float)
    theta = theta[np.isfinite(theta)]
    n = theta.size
    if n == 0:
        return {
            "n_phase_events_used": 0,
            "mean_phase_rad": np.nan,
            "mean_phase_deg": np.nan,
            "mean_resultant_length": np.nan,
            "rayleigh_R": np.nan,
            "rayleigh_z": np.nan,
            "rayleigh_p": np.nan,
        }

    c = float(np.sum(np.cos(theta)))
    s = float(np.sum(np.sin(theta)))
    rayleigh_R = float(np.hypot(c, s))
    mean_phase = float(np.mod(np.arctan2(s, c), 2.0 * np.pi))
    mean_resultant_length = rayleigh_R / n
    z = n * mean_resultant_length**2

    # Standard large-sample Rayleigh p with finite-n correction.
    p = rayleigh_p_value_from_z(z, n)
    return {
        "n_phase_events_used": int(n),
        "mean_phase_rad": mean_phase,
        "mean_phase_deg": float(np.degrees(mean_phase)),
        "mean_resultant_length": float(mean_resultant_length),
        "rayleigh_R": rayleigh_R,
        "rayleigh_z": float(z),
        "rayleigh_p": p,
    }


def build_rayleigh_results(event_phases: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (driver, driver_label, event_type, event_label), group in event_phases.groupby(
        ["driver", "driver_label", "event_type", "event_label"],
        sort=False,
    ):
        used = group[~group["phase_extrapolated"].astype(bool)].copy()
        result = rayleigh_test(used["phase_rad"].to_numpy(dtype=float))
        result.update(
            {
                "driver": driver,
                "driver_label": driver_label,
                "event_type": event_type,
                "event_label": event_label,
                "n_events_total": int(len(group)),
                "n_extrapolated_phase_events": int(group["phase_extrapolated"].sum()),
            }
        )
        rows.append(result)
    out = pd.DataFrame(rows)
    return out[
        [
            "driver",
            "driver_label",
            "event_type",
            "event_label",
            "n_events_total",
            "n_phase_events_used",
            "n_extrapolated_phase_events",
            "mean_phase_rad",
            "mean_phase_deg",
            "mean_resultant_length",
            "rayleigh_R",
            "rayleigh_z",
            "rayleigh_p",
        ]
    ].sort_values(["driver", "event_type"]).reset_index(drop=True)


def save_figure(fig: plt.Figure, stem: str, write_pdf: bool) -> None:
    ensure_dir(OUT_FIG_DIR)
    fig.savefig(OUT_FIG_DIR / f"{stem}.png", dpi=300, bbox_inches="tight")
    if write_pdf:
        fig.savefig(OUT_FIG_DIR / f"{stem}.pdf", bbox_inches="tight")
    plt.close(fig)


def plot_extrema_phase_check(phase_products: dict[str, OrbitalPhase], write_pdf: bool) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(13, 6.8), sharex="col")
    for row_idx, phase_product in enumerate(phase_products.values()):
        series = phase_product.series
        extrema = phase_product.extrema
        sub = series[
            (series["age_ka"] >= ANALYSIS_START_KA)
            & (series["age_ka"] <= ANALYSIS_END_KA)
        ]
        ext_sub = extrema[
            (extrema["age_ka"] >= ANALYSIS_START_KA)
            & (extrema["age_ka"] <= ANALYSIS_END_KA)
        ]

        ax_value = axes[row_idx, 0]
        ax_phase = axes[row_idx, 1]
        color = DRIVER_SETTINGS[phase_product.driver]["color"]
        ax_value.plot(sub["age_ka"], sub["value"], color=color, lw=1.2)
        for kind, marker, face, label in [
            ("minimum", "v", "white", "min = phase 0"),
            ("maximum", "^", color, "max = phase pi"),
        ]:
            points = ext_sub[ext_sub["extremum_type"].eq(kind)]
            ax_value.scatter(
                points["age_ka"],
                points["value"],
                s=28,
                marker=marker,
                facecolors=face,
                edgecolors=color,
                linewidths=0.9,
                label=label,
                zorder=3,
            )
        ax_value.set_title(f"{phase_product.label}: detected extrema", loc="left")
        ax_value.set_ylabel("Raw orbital value")
        ax_value.grid(True, color="#e6e6e6", lw=0.7)
        ax_value.legend(frameon=False, loc="best")

        ax_phase.plot(sub["age_ka"], sub["phase_deg"], color=color, lw=1.1)
        ax_phase.scatter(
            ext_sub["age_ka"],
            ext_sub["anchor_phase_deg"],
            s=14,
            color="#202020",
            alpha=0.8,
            label="extremum anchors",
        )
        ax_phase.set_yticks([0, 90, 180, 270, 360])
        ax_phase.set_ylim(-8, 368)
        ax_phase.set_title(f"{phase_product.label}: wrapped phase", loc="left")
        ax_phase.set_ylabel("Phase (deg)")
        ax_phase.grid(True, color="#e6e6e6", lw=0.7)
        ax_phase.legend(frameon=False, loc="best")

    for ax in axes[-1]:
        ax.set_xlabel("Age (ka BP)")
    fig.suptitle("Orbital extrema and phase-conversion check", y=0.995, fontsize=14)
    fig.subplots_adjust(top=0.90, hspace=0.30, wspace=0.22)
    save_figure(fig, "fig01_orbital_extrema_phase_check", write_pdf)


def plot_event_phase_sampling(
    phase_products: dict[str, OrbitalPhase],
    event_phases: pd.DataFrame,
    write_pdf: bool,
) -> None:
    fig, axes = plt.subplots(len(phase_products), 1, figsize=(13, 5.8), sharex=True)
    axes = np.atleast_1d(axes)
    for ax, phase_product in zip(axes, phase_products.values()):
        series = phase_product.series
        sub = series[
            (series["age_ka"] >= ANALYSIS_START_KA)
            & (series["age_ka"] <= ANALYSIS_END_KA)
        ]
        ax.plot(
            sub["age_ka"],
            sub["phase_deg"],
            color=DRIVER_SETTINGS[phase_product.driver]["color"],
            lw=0.95,
            alpha=0.78,
        )
        phase_sub = event_phases[event_phases["driver"].eq(phase_product.driver)]
        for event_type, group in phase_sub.groupby("event_type", sort=False):
            used = group[~group["phase_extrapolated"].astype(bool)]
            extrapolated = group[group["phase_extrapolated"].astype(bool)]
            ax.scatter(
                used["event_age_ka"],
                used["phase_deg"],
                s=20,
                color=EVENT_COLORS[event_type],
                alpha=0.82,
                label=group["event_label"].iloc[0],
                zorder=3,
            )
            if not extrapolated.empty:
                ax.scatter(
                    extrapolated["event_age_ka"],
                    extrapolated["phase_deg"],
                    s=34,
                    facecolors="none",
                    edgecolors=EVENT_COLORS[event_type],
                    linewidths=1.2,
                    label=f"{group['event_label'].iloc[0]} extrapolated",
                    zorder=4,
                )
        ax.set_ylabel("Phase (deg)")
        ax.set_yticks([0, 90, 180, 270, 360])
        ax.set_ylim(-8, 368)
        ax.set_title(f"{phase_product.label}: event phase sampling", loc="left")
        ax.grid(True, color="#e6e6e6", lw=0.7)
        ax.legend(frameon=False, loc="upper right", ncol=2)
    axes[-1].set_xlabel("Age (ka BP)")
    fig.suptitle("Strong/weak monsoon starts sampled on orbital phase curves", y=0.995, fontsize=14)
    fig.subplots_adjust(top=0.90, hspace=0.28)
    save_figure(fig, "fig02_event_phase_sampling_check", write_pdf)


def plot_polar_rayleigh(
    event_phases: pd.DataFrame,
    rayleigh: pd.DataFrame,
    write_pdf: bool,
) -> None:
    drivers = list(DRIVER_SETTINGS)
    event_types = ["strong_monsoon_start", "weak_monsoon_start"]
    fig, axes = plt.subplots(
        len(drivers),
        len(event_types),
        figsize=(10.8, 9.8),
        subplot_kw={"projection": "polar"},
    )
    bins = np.linspace(0.0, 2.0 * np.pi, 19)
    width = bins[1] - bins[0]
    theta_grid = np.linspace(0.0, 2.0 * np.pi, 361)
    for i, driver in enumerate(drivers):
        for j, event_type in enumerate(event_types):
            ax = axes[i, j]
            panel_label = chr(ord("a") + i * len(event_types) + j)
            sub = event_phases[
                event_phases["driver"].eq(driver)
                & event_phases["event_type"].eq(event_type)
                & ~event_phases["phase_extrapolated"].astype(bool)
            ].copy()
            res = rayleigh[
                rayleigh["driver"].eq(driver)
                & rayleigh["event_type"].eq(event_type)
            ].iloc[0]
            counts, _ = np.histogram(sub["phase_rad"].to_numpy(dtype=float), bins=bins)
            ax.bar(
                bins[:-1],
                counts,
                width=width,
                align="edge",
                color=EVENT_COLORS[event_type],
                alpha=0.55,
                edgecolor="white",
                linewidth=0.8,
            )
            mean_phase = float(res["mean_phase_rad"])
            mean_r = float(res["mean_resultant_length"])
            max_count = max(counts.max(), 1)
            n_used = int(res["n_phase_events_used"])
            rbar_threshold = rayleigh_rbar_threshold(n_used)
            threshold_radius = rbar_threshold * max_count
            if np.isfinite(threshold_radius):
                ax.plot(
                    theta_grid,
                    np.full_like(theta_grid, threshold_radius),
                    color="#303030",
                    linestyle=(0, (4, 2)),
                    lw=1.15,
                    alpha=0.9,
                    zorder=4,
                )
            ax.annotate(
                "",
                xy=(mean_phase, mean_r * max_count),
                xytext=(mean_phase, 0),
                arrowprops={"arrowstyle": "->", "lw": 2.0, "color": "#202020"},
            )
            ax.set_theta_zero_location("E")
            ax.set_theta_direction(1)
            ax.set_xticks([0, np.pi / 2, np.pi, 3 * np.pi / 2])
            ax.set_xticklabels(["min", "90", "max", "270"])
            ax.set_rlabel_position(35)
            ax.tick_params(axis="x", pad=4, labelsize=9)
            ax.tick_params(axis="y", labelsize=8)
            ax.text(
                0.02,
                0.98,
                panel_label,
                transform=ax.transAxes,
                ha="left",
                va="top",
                fontsize=11,
                fontweight="bold",
                bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.8, "pad": 1.6},
            )
            ax.text(
                0.02,
                0.06,
                rf"$\bar{{R}}_{{0.05}}$={rbar_threshold:.2f}",
                transform=ax.transAxes,
                ha="left",
                va="bottom",
                fontsize=8,
                color="#303030",
                bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.72, "pad": 1.4},
            )
            ax.set_title(
                f"{sub['event_label'].iloc[0]}\n"
                rf"N={n_used}, $\bar{{R}}$={mean_r:.2f}, p={float(res['rayleigh_p']):.3f}",
                va="bottom",
                fontsize=9.5,
                pad=30,
            )
    fig.subplots_adjust(top=0.92, bottom=0.06, hspace=0.72, wspace=0.30)
    save_figure(fig, "fig03_rayleigh_phase_polar", write_pdf)


def plot_polar_rayleigh_single_driver(
    event_phases: pd.DataFrame,
    rayleigh: pd.DataFrame,
    driver: str,
    stem: str,
    write_pdf: bool,
) -> None:
    """Plot one driver from Fig. 3 as a separate strong/weak polar figure."""

    event_types = ["strong_monsoon_start", "weak_monsoon_start"]
    fig, axes = plt.subplots(
        1,
        len(event_types),
        figsize=(7.2, 3.4),
        subplot_kw={"projection": "polar"},
    )
    axes = np.atleast_1d(axes)
    bins = np.linspace(0.0, 2.0 * np.pi, 19)
    width = bins[1] - bins[0]
    theta_grid = np.linspace(0.0, 2.0 * np.pi, 361)
    panel_label_start = 2 if driver == "pre" else 0

    for j, event_type in enumerate(event_types):
        ax = axes[j]
        panel_label = chr(ord("a") + panel_label_start + j)
        sub = event_phases[
            event_phases["driver"].eq(driver)
            & event_phases["event_type"].eq(event_type)
            & ~event_phases["phase_extrapolated"].astype(bool)
        ].copy()
        res = rayleigh[
            rayleigh["driver"].eq(driver)
            & rayleigh["event_type"].eq(event_type)
        ].iloc[0]
        counts, _ = np.histogram(sub["phase_rad"].to_numpy(dtype=float), bins=bins)
        ax.bar(
            bins[:-1],
            counts,
            width=width,
            align="edge",
            color=EVENT_COLORS[event_type],
            alpha=0.55,
            edgecolor="white",
            linewidth=0.8,
        )
        mean_phase = float(res["mean_phase_rad"])
        mean_r = float(res["mean_resultant_length"])
        max_count = max(counts.max(), 1)
        n_used = int(res["n_phase_events_used"])
        rbar_threshold = rayleigh_rbar_threshold(n_used)
        threshold_radius = rbar_threshold * max_count
        if np.isfinite(threshold_radius):
            ax.plot(
                theta_grid,
                np.full_like(theta_grid, threshold_radius),
                color="#303030",
                linestyle=(0, (4, 2)),
                lw=1.15,
                alpha=0.9,
                zorder=4,
            )
            if driver == "pre":
                threshold_label_angle = np.deg2rad(135.0 if event_type == "strong_monsoon_start" else 135.0)
                ax.text(
                    threshold_label_angle,
                    threshold_radius * 1.98,
                    r"$\bar{R}_{0.05}$",
                    ha="center",
                    va="center",
                    fontsize=8,
                    color="#303030",
                    bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.72, "pad": 1.2},
                    zorder=5,
                )
        ax.annotate(
            "",
            xy=(mean_phase, mean_r * max_count),
            xytext=(mean_phase, 0),
            arrowprops={"arrowstyle": "->", "lw": 2.0, "color": "#202020"},
        )
        if driver == "pre":
            ax.text(
                mean_phase,
                mean_r * max_count + 1.05,
                f"{float(res['mean_phase_deg']):.1f}°",
                ha="left",
                va="center",
                fontsize=8.5,
                color="#202020",
                bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.72, "pad": 1.0},
                zorder=5,
            )
        ax.set_theta_zero_location("E")
        ax.set_theta_direction(1)
        ax.set_xticks([0, np.pi / 2, np.pi, 3 * np.pi / 2])
        ax.set_xticklabels(["min", "90", "max", "270"])
        ax.set_rlabel_position(35)
        ax.tick_params(axis="x", pad=4, labelsize=9)
        ax.tick_params(axis="y", labelsize=8)
        if driver == "pre":
            ax.set_ylim(0.0, 11.75)
            ax.set_yticks([2, 4, 6, 8, 10])
            ax.grid(True, color="#b8b8b8", lw=0.7, alpha=0.42)
        ax.text(
            0.03,
            0.98,
            panel_label,
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=11,
            fontweight="bold",
            bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.8, "pad": 1.6},
        )
        if driver != "pre":
            ax.text(
                0.02,
                0.06,
                rf"$\bar{{R}}_{{0.05}}$={rbar_threshold:.2f}",
                transform=ax.transAxes,
                ha="left",
                va="bottom",
                fontsize=8,
                color="#303030",
                bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.72, "pad": 1.4},
            )
        ax.set_title(
            f"{sub['event_label'].iloc[0]}\n"
            rf"N={n_used}, $\bar{{R}}$={mean_r:.2f}, p={float(res['rayleigh_p']):.3f}",
            va="bottom",
            fontsize=9.5,
            pad=30,
        )

    fig.subplots_adjust(top=0.84, bottom=0.08, left=0.05, right=0.95, wspace=0.28)
    save_figure(fig, stem, write_pdf)


def plot_split_polar_rayleigh(
    event_phases: pd.DataFrame,
    rayleigh: pd.DataFrame,
    write_pdf: bool,
) -> None:
    plot_polar_rayleigh_single_driver(
        event_phases,
        rayleigh,
        "pre",
        "fig03a_rayleigh_phase_polar_precession",
        write_pdf,
    )
    plot_polar_rayleigh_single_driver(
        event_phases,
        rayleigh,
        "obl",
        "fig03b_rayleigh_phase_polar_obliquity",
        write_pdf,
    )


def plot_phase_ecdf(event_phases: pd.DataFrame, write_pdf: bool) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.2), sharey=True)
    for ax_idx, (ax, driver) in enumerate(zip(axes, DRIVER_SETTINGS)):
        driver_events = event_phases[
            event_phases["driver"].eq(driver)
            & ~event_phases["phase_extrapolated"].astype(bool)
        ]
        for event_type, group in driver_events.groupby(
            "event_type",
            sort=False,
        ):
            phase_fraction = np.sort(group["phase_fraction"].to_numpy(dtype=float))
            y = np.arange(1, len(phase_fraction) + 1, dtype=float) / len(phase_fraction)
            ax.step(
                phase_fraction,
                y,
                where="post",
                color=EVENT_COLORS[event_type],
                lw=1.5,
                label=group["event_label"].iloc[0],
            )
        ax.plot([0, 1], [0, 1], color="#666666", ls="--", lw=1.0, label="uniform")
        ax.set_title(f"{DRIVER_SETTINGS[driver]['label']} phase ECDF", loc="left")
        ax.set_xlabel("Phase fraction (0=min, 0.5=max)")
        ax.grid(True, color="#e6e6e6", lw=0.7)
        ax.text(
            -0.04,
            1.04,
            chr(ord("a") + ax_idx),
            transform=ax.transAxes,
            ha="right",
            va="bottom",
            fontsize=12,
            fontweight="bold",
            clip_on=False,
        )
    axes[0].set_ylabel("Empirical CDF")
    axes[-1].legend(frameon=False, loc="lower right")
    fig.subplots_adjust(top=0.88, wspace=0.18)
    save_figure(fig, "fig04_phase_ecdf_uniform_check", write_pdf)


def write_outputs(
    phase_products: dict[str, OrbitalPhase],
    events: pd.DataFrame,
    event_phases: pd.DataFrame,
    rayleigh: pd.DataFrame,
) -> None:
    ensure_dir(OUT_DATA_DIR)
    pd.concat([product.series for product in phase_products.values()], ignore_index=True).to_csv(
        OUT_DATA_DIR / "orbital_phase_series_pre_obl.csv",
        index=False,
    )
    pd.concat([product.extrema for product in phase_products.values()], ignore_index=True).to_csv(
        OUT_DATA_DIR / "orbital_phase_extrema_pre_obl.csv",
        index=False,
    )
    events.to_csv(OUT_DATA_DIR / "rousseau2023_monsoon_events_used.csv", index=False)
    event_phases.to_csv(OUT_DATA_DIR / "rousseau2023_monsoon_event_orbital_phases.csv", index=False)
    rayleigh.to_csv(OUT_DATA_DIR / "rayleigh_phase_preference_results.csv", index=False)


def run_analysis(write_pdf: bool) -> pd.DataFrame:
    ensure_dir(OUT_DATA_DIR)
    ensure_dir(OUT_FIG_DIR)
    events = load_all_events()
    phase_products = {
        driver: build_phase_series(driver, settings)
        for driver, settings in DRIVER_SETTINGS.items()
    }
    event_phases = sample_event_phases(events, phase_products)
    rayleigh = build_rayleigh_results(event_phases)
    write_outputs(phase_products, events, event_phases, rayleigh)
    plot_extrema_phase_check(phase_products, write_pdf)
    plot_event_phase_sampling(phase_products, event_phases, write_pdf)
    plot_polar_rayleigh(event_phases, rayleigh, write_pdf)
    plot_split_polar_rayleigh(event_phases, rayleigh, write_pdf)
    plot_phase_ecdf(event_phases, write_pdf)
    return rayleigh


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Rayleigh tests of Rousseau 2023 monsoon-start phases relative to precession and obliquity."
    )
    parser.add_argument("--no-pdf", action="store_true", help="Only write PNG figures.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rayleigh = run_analysis(write_pdf=not args.no_pdf)
    print(
        rayleigh[
            [
                "driver",
                "event_type",
                "n_events_total",
                "n_phase_events_used",
                "n_extrapolated_phase_events",
                "mean_phase_deg",
                "mean_resultant_length",
                "rayleigh_z",
                "rayleigh_p",
            ]
        ].to_string(index=False)
    )
    print(f"\nWrote tables to: {OUT_DATA_DIR.relative_to(PROJECT_ROOT)}")
    print(f"Wrote figures to: {OUT_FIG_DIR.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    main()
