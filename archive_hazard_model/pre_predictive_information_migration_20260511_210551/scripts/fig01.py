"""
Figure 1: data and phase framework for Rousseau et al. (2023) monsoon starts.

The figure combines the core data streams used in the event-timing analysis:

1. Cheng et al. (2016) composite speleothem d18O with Rousseau strong/weak
   monsoon-start timings.
2. LR04 benthic d18O background state.
3. Atmospheric CO2 background state.
4. Precession phase, with the raw precession index on a secondary axis.

The precession phase convention is inherited from
analyze_rousseau2023_monsoon_bin_hazard_phase_poisson.py:
precession-index minima are phase 0 deg and maxima are phase 180 deg.
"""

from __future__ import annotations

import argparse
import warnings
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.transforms import Bbox
import numpy as np
import pandas as pd

import archive_hazard_model.pre_predictive_information_migration_20260511_210551.scripts.Bin_hazard_phase_poisson as hazard


PROJECT_ROOT = Path(__file__).resolve().parent
RUN_NAME = "fig01"
OUT_FIG_DIR = PROJECT_ROOT / "figures" / RUN_NAME

CHENG_CANDIDATES = (
    PROJECT_ROOT / "data/raw/Cheng_2016.xlsx",
    PROJECT_ROOT / "old_five_proxy/data/raw/Cheng_2016.xlsx",
)

# Strong monsoon starts correspond to Greenland interstadial/warm onsets, so
# use the warm event color for strong starts and the cool event color for weak
# starts.
STRONG_COLOR = hazard.DATASET_SETTINGS["weak_monsoon_start"]["color"]
WEAK_COLOR = hazard.DATASET_SETTINGS["strong_monsoon_start"]["color"]
COMPOSITE_COLOR = "#2b2b2b"
LR04_COLOR = "#1b9e77"
CO2_COLOR = "#d95f02"
PRE_PHASE_COLOR = "#7570b3"
PRE_RAW_COLOR = "#5f5f5f"


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


def find_existing_cheng_file() -> Path:
    for path in CHENG_CANDIDATES:
        if path.exists():
            return path
    candidates = "\n".join(str(path.relative_to(PROJECT_ROOT)) for path in CHENG_CANDIDATES)
    raise FileNotFoundError(f"Cannot find Cheng_2016.xlsx. Checked:\n{candidates}")


def load_cheng_composite(path: Path) -> pd.DataFrame:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        raw = pd.read_excel(path, sheet_name="Composite record")
    age_col = hazard.find_column(raw.columns, "age", "ka")
    d18o_col = hazard.find_column(raw.columns, "18o")
    out = pd.DataFrame(
        {
            "age_ka": pd.to_numeric(raw[age_col], errors="coerce"),
            "d18o": pd.to_numeric(raw[d18o_col], errors="coerce"),
        }
    )
    out = out.dropna().groupby("age_ka", as_index=False)["d18o"].mean()
    out = out.sort_values("age_ka")
    out = out[
        (out["age_ka"] >= hazard.ANALYSIS_START_KA)
        & (out["age_ka"] <= hazard.ANALYSIS_END_KA)
    ].copy()
    return out.reset_index(drop=True)


def make_plot_grid() -> pd.DataFrame:
    edges = hazard.make_bin_edges()
    centers = 0.5 * (edges[:-1] + edges[1:])

    _, lr04_info = hazard.load_lr04(centers)
    _, co2_info = hazard.load_co2(centers)
    phase_table, _ = hazard.build_precession_phase(centers)

    return pd.DataFrame(
        {
            "age_ka": centers,
            "lr04": lr04_info["raw"],
            "co2": co2_info["raw"],
            "pre_phase_deg": phase_table["pre_phase_deg"],
            "precession_index": phase_table["precession_index"],
        }
    )


def load_event_ages() -> tuple[np.ndarray, np.ndarray]:
    events = {event.dataset_id: event for event in hazard.load_all_events()}
    strong = events["strong_monsoon_start"].ages_ka
    weak = events["weak_monsoon_start"].ages_ka
    return strong, weak


def add_panel_labels(axes: list[plt.Axes]) -> None:
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


def axes_group_bbox(axes: list[plt.Axes]) -> Bbox:
    return Bbox.union([ax.get_position() for ax in axes])


def add_event_header(
    fig: plt.Figure,
    axes: list[plt.Axes],
    strong_ages: np.ndarray,
    weak_ages: np.ndarray,
) -> None:
    """Place event-count labels above the shared plotting frame."""

    bbox = axes_group_bbox(axes)
    y = bbox.y1 + 0.014
    fig.text(
        bbox.x0,
        y,
        f"Strong starts, N={len(strong_ages)}",
        ha="left",
        va="bottom",
        color=STRONG_COLOR,
        fontsize=9,
        fontweight="bold",
    )
    fig.text(
        bbox.x0 + 0.18,
        y,
        f"Weak starts, N={len(weak_ages)}",
        ha="left",
        va="bottom",
        color=WEAK_COLOR,
        fontsize=9,
        fontweight="bold",
    )


def add_event_overlay(
    fig: plt.Figure,
    axes: list[plt.Axes],
    strong_ages: np.ndarray,
    weak_ages: np.ndarray,
) -> None:
    """Draw top event ticks and one shared outer frame."""

    bbox = axes_group_bbox(axes)

    overlay = fig.add_axes([bbox.x0, bbox.y0, bbox.width, bbox.height], zorder=10)
    overlay.patch.set_alpha(0.0)
    overlay.set_xlim(hazard.ANALYSIS_START_KA, hazard.ANALYSIS_END_KA)
    overlay.set_ylim(0.0, 1.0)
    overlay.vlines(strong_ages, 0.980, 0.998, color=STRONG_COLOR, lw=0.70, alpha=1.0, zorder=2)
    overlay.vlines(weak_ages, 0.956, 0.974, color=WEAK_COLOR, lw=0.70, alpha=1.0, zorder=1)
    overlay.set_xticks([])
    overlay.set_yticks([])
    overlay.tick_params(left=False, right=False, bottom=False, top=False)
    for spine in overlay.spines.values():
        spine.set_visible(True)
        spine.set_color("#202020")
        spine.set_linewidth(0.9)


def save_figure(fig: plt.Figure, stem: str, write_pdf: bool = True) -> None:
    ensure_dir(OUT_FIG_DIR)
    png_path = OUT_FIG_DIR / f"{stem}.png"
    pdf_path = OUT_FIG_DIR / f"{stem}.pdf"
    fig.savefig(png_path, dpi=300, bbox_inches="tight")
    if write_pdf:
        fig.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)


def plot_data_phase_framework(write_pdf: bool = True) -> None:
    composite = load_cheng_composite(find_existing_cheng_file())
    plot_grid = make_plot_grid()
    pre_raw = hazard.load_precession_series()
    pre_raw = pre_raw[
        (pre_raw["age_ka"] >= hazard.ANALYSIS_START_KA)
        & (pre_raw["age_ka"] <= hazard.ANALYSIS_END_KA)
    ].copy()
    strong_ages, weak_ages = load_event_ages()

    fig, axes_arr = plt.subplots(
        4,
        1,
        figsize=(13.0, 8.4),
        sharex=True,
        height_ratios=[1.45, 0.85, 0.85, 1.05],
    )
    axes = list(axes_arr)

    axes[0].plot(
        composite["age_ka"],
        composite["d18o"],
        color=COMPOSITE_COLOR,
        lw=0.55,
        label=r"Cheng composite $\delta^{18}\mathrm{O}$",
        zorder=2,
    )
    axes[0].invert_yaxis()
    axes[0].set_ylim(-5.0, -12.8)
    axes[0].set_yticks([-12, -10, -8, -6])
    axes[0].set_ylabel(r"Speleothem $\delta^{18}\mathrm{O}$" "\n(per mil)")

    axes[1].plot(plot_grid["age_ka"], plot_grid["lr04"], color=LR04_COLOR, lw=1.0)
    axes[1].set_ylabel(r"LR04 $\delta^{18}\mathrm{O}$" "\n(per mil)")
    axes[1].invert_yaxis()

    axes[2].plot(plot_grid["age_ka"], plot_grid["co2"], color=CO2_COLOR, lw=1.0)
    axes[2].set_ylabel("CO$_2$\n(ppm)")

    pre_ax = axes[3].twinx()
    axes[3].set_zorder(2)
    pre_ax.set_zorder(1)
    axes[3].patch.set_visible(False)
    pre_ax.plot(
        pre_raw["age_ka"],
        pre_raw["precession_index"],
        color=PRE_RAW_COLOR,
        lw=0.75,
        alpha=0.38,
        zorder=1,
        label="Precession index",
    )
    pre_ax.set_ylabel("Precession index", color=PRE_RAW_COLOR)
    pre_ax.tick_params(axis="y", colors=PRE_RAW_COLOR, labelsize=8)
    pre_ax.spines["right"].set_color("#8a8a8a")

    axes[3].plot(
        plot_grid["age_ka"],
        plot_grid["pre_phase_deg"],
        color=PRE_PHASE_COLOR,
        lw=0.9,
        zorder=3,
        label="Precession phase",
    )
    axes[3].set_ylabel("Precession\nphase (deg)")
    axes[3].set_ylim(-8, 368)
    axes[3].set_yticks([0, 90, 180, 270, 360])
    axes[3].tick_params(axis="y", colors=PRE_PHASE_COLOR)
    axes[3].spines["left"].set_color(PRE_PHASE_COLOR)

    for ax in axes:
        ax.grid(False)
        ax.set_xlim(hazard.ANALYSIS_START_KA, hazard.ANALYSIS_END_KA)
        ax.patch.set_alpha(0.0)
        for spine in ax.spines.values():
            spine.set_visible(False)
    for ax in axes[:-1]:
        ax.tick_params(axis="x", which="both", bottom=False, top=False, labelbottom=False)
    axes[-1].tick_params(axis="x", which="both", bottom=True, top=False, direction="out")
    for spine in pre_ax.spines.values():
        spine.set_visible(False)

    axes[-1].set_xlabel("Age (kyr BP)")
    axes[-1].set_xticks(np.arange(0, hazard.ANALYSIS_END_KA + 1, 100))
    add_panel_labels(axes)

    fig.subplots_adjust(left=0.10, right=0.90, top=0.95, bottom=0.09, hspace=0.08)
    add_event_header(fig, axes, strong_ages, weak_ages)
    add_event_overlay(fig, axes, strong_ages, weak_ages)
    save_figure(fig, "fig01", write_pdf=write_pdf)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot Figure 1 data and phase framework.")
    parser.add_argument("--no-pdf", action="store_true", help="Only write PNG output.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    plot_data_phase_framework(write_pdf=not args.no_pdf)
    print(f"Wrote figures to: {OUT_FIG_DIR.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    main()
