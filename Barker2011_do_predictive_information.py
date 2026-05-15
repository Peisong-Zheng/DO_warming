"""
Rayleigh and predictive-information checks for Barker et al. (2011) predicted
D-O warming events.

The Barker et al. supplementary workbook lists predicted D-O event occurrence
from the synthetic Greenland record. The variable-threshold picks are treated
here as an independent analogue of strong-monsoon starts: transitions toward
warmer Greenland/interstadial-like conditions.

Three variable-threshold age variants are analysed:

1. variable_threshold_edc3_0_640:
   Event ages from the EDC3 column, restricted to 0-640 ka.
2. variable_threshold_edc3_0_800:
   Event ages from the EDC3 column, restricted to 0-800 ka.
3. variable_threshold_speleo_0_400:
   Event ages from the Barker SpeleoAge column, where available, restricted to
   0-400 ka. This keeps the age scale aligned with the Chinese speleothem-tuned
   part of Barker et al. (2011), but necessarily uses fewer events.

For each variant this script runs:
- Rayleigh tests of precession and obliquity phase preference;
- binned Poisson predictive-information models with same-type history and EDC
  sampling resolution as the event/detection baseline, then LR04, CO2, and
  precession phase as predictors.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle
import numpy as np
import pandas as pd
from scipy.stats import chi2
from paper_figure_export import save_paper_pdf

import Bin_hazard_phase_poisson as base
import Orbital_phase_rayleigh as rayleigh
import Predictive_information_model as predictive


RUN_NAME = "Barker2011_do_predictive_information"
OUT_DATA_DIR = base.PROJECT_ROOT / "data" / "processed" / RUN_NAME
OUT_FIG_DIR = base.PROJECT_ROOT / "figures" / RUN_NAME

BARKER_XLS = base.PROJECT_ROOT / "data/raw/Barker et al-2011-SOM.xls"
JOUZEL_TXT = (
    base.PROJECT_ROOT
    / "data/raw/Jouzel-etal-2007-Science-Orbital and Millennial Antarctic Climate Variability over the Past 800,000 Years.txt"
)

MAIN_HISTORY_WINDOW_KA = 5.0
RAYLEIGH_ALPHA = 0.05

HISTORY_TERM = predictive.HISTORY_TERM
RESOLUTION_TERM = "edc_log_resolution_scaled"
BASELINE_TERMS = (HISTORY_TERM, RESOLUTION_TERM)
CLIMATE_TERMS = predictive.CLIMATE_TERMS
PHASE_TERMS = predictive.PHASE_TERMS

MODEL_SPECS: list[tuple[str, tuple[str, ...], str]] = [
    ("stationary", (), "Stationary"),
    ("history_resolution_baseline", BASELINE_TERMS, "Event-process baseline"),
    (
        "baseline_climate_lr04_co2",
        BASELINE_TERMS + CLIMATE_TERMS,
        "Climate-state model",
    ),
    (
        "baseline_pre_phase",
        BASELINE_TERMS + PHASE_TERMS,
        "Event-process baseline + precession phase",
    ),
    (
        "baseline_climate_lr04_co2_pre_phase",
        BASELINE_TERMS + CLIMATE_TERMS + PHASE_TERMS,
        "Full predictive model",
    ),
]

LR_TEST_SPECS: list[tuple[str, str, str, str]] = [
    (
        "baseline_vs_stationary",
        "stationary",
        "history_resolution_baseline",
        "Do same-type event history and EDC sampling resolution improve over a constant rate?",
    ),
    (
        "climate_lr04_co2_after_baseline",
        "history_resolution_baseline",
        "baseline_climate_lr04_co2",
        "Does LR04+CO2 improve over the EDC event-process baseline?",
    ),
    (
        "pre_phase_after_baseline",
        "history_resolution_baseline",
        "baseline_pre_phase",
        "Does precession phase improve over the EDC event-process baseline?",
    ),
    (
        "phase_after_baseline_climate",
        "baseline_climate_lr04_co2",
        "baseline_climate_lr04_co2_pre_phase",
        "Does precession phase add information after the Barker climate-state model?",
    ),
    (
        "full_after_baseline",
        "history_resolution_baseline",
        "baseline_climate_lr04_co2_pre_phase",
        "Does the full predictive model improve over the EDC event-process baseline?",
    ),
]

CATALOGUE_VARIANTS = [
    {
        "dataset_id": "barker_variable_threshold_edc3_0_640",
        "event_type": "barker_do_warming_variable_threshold_edc3",
        "label": "Barker variable-threshold D-O warmings, EDC3 0-640 ka",
        "age_column": "Age kyr (EDC3)",
        "pick_column": "DO pick variable threshold",
        "analysis_start_ka": 0.0,
        "analysis_end_ka": 640.0,
        "resolution_age_axis": "edc3",
        "color": "#d95f02",
        "primary": True,
    },
    {
        "dataset_id": "barker_variable_threshold_edc3_0_800",
        "event_type": "barker_do_warming_variable_threshold_edc3_0_800",
        "label": "Barker variable-threshold D-O warmings, EDC3 0-800 ka",
        "age_column": "Age kyr (EDC3)",
        "pick_column": "DO pick variable threshold",
        "analysis_start_ka": 0.0,
        "analysis_end_ka": 800.0,
        "resolution_age_axis": "edc3",
        "color": "#117733",
        "primary": False,
    },
    {
        "dataset_id": "barker_variable_threshold_speleo_0_400",
        "event_type": "barker_do_warming_variable_threshold_speleo",
        "label": "Barker variable-threshold D-O warmings, SpeleoAge 0-400 ka",
        "age_column": "SpeloAge (kyr).1",
        "pick_column": "DO pick variable threshold",
        "analysis_start_ka": 0.0,
        "analysis_end_ka": 400.0,
        "resolution_age_axis": "speleo_to_edc3",
        "color": "#cc6677",
        "primary": False,
    },
]


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


def save_figure(fig: plt.Figure, stem: str, write_pdf: bool = True) -> None:
    ensure_dir(OUT_FIG_DIR)
    fig.savefig(OUT_FIG_DIR / f"{stem}.png", dpi=300, bbox_inches="tight")
    if write_pdf:
        fig.savefig(OUT_FIG_DIR / f"{stem}.pdf", bbox_inches="tight")
        save_paper_pdf(fig, base.PROJECT_ROOT, stem)
    plt.close(fig)


def format_p(value: float) -> str:
    if not np.isfinite(value):
        return "NA"
    if value < 1e-4:
        return f"{value:.1e}"
    if value < 0.01:
        return f"{value:.4f}"
    return f"{value:.3f}"


def short_variant_label(settings: dict) -> str:
    """Compact label for Barker catalogue variants used in figure panels."""

    if "edc3_0_800" in settings["dataset_id"]:
        return "EDC3 0-800 kyr"
    if "edc3_0_640" in settings["dataset_id"]:
        return "EDC3 0-640 kyr"
    return "SpeleoAge 0-400 kyr"


def load_barker_table_s3() -> pd.DataFrame:
    """Load Barker et al. Table S3 from Sheet1 of the supplementary workbook."""

    raw = pd.read_excel(BARKER_XLS, sheet_name="Sheet1", header=8)
    required = [
        "Age kyr (EDC3)",
        "SpeloAge (kyr).1",
        "DO pick",
        "DO pick variable threshold",
    ]
    missing = [column for column in required if column not in raw.columns]
    if missing:
        raise ValueError(f"Missing expected Barker columns: {missing}")
    out = raw[required].copy()
    for column in required:
        out[column] = pd.to_numeric(out[column], errors="coerce")
    out = out.dropna(subset=["Age kyr (EDC3)"], how="all")
    return out.reset_index(drop=True)


def load_barker_speleo_age_mapping() -> pd.DataFrame:
    """Load Barker's SpeleoAge-to-EDC3 mapping from Table S2."""

    raw = pd.read_excel(BARKER_XLS, sheet_name="Sheet1", header=8)
    mapping = raw[["EDC3 Age (kyr)", "SpeloAge (kyr)"]].copy()
    mapping["edc3_age_ka"] = pd.to_numeric(mapping["EDC3 Age (kyr)"], errors="coerce")
    mapping["speleo_age_ka"] = pd.to_numeric(mapping["SpeloAge (kyr)"], errors="coerce")
    mapping = mapping.dropna(subset=["edc3_age_ka", "speleo_age_ka"])
    mapping = mapping.groupby("speleo_age_ka", as_index=False)["edc3_age_ka"].mean()
    return mapping.sort_values("speleo_age_ka").reset_index(drop=True)


def interpolate_with_linear_extrapolation(x_new: np.ndarray, x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """One-dimensional interpolation with linear extrapolation at both ends."""

    x_new = np.asarray(x_new, dtype=float)
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    order = np.argsort(x)
    x = x[order]
    y = y[order]
    out = np.interp(x_new, x, y)
    if len(x) >= 2:
        left = x_new < x[0]
        if np.any(left):
            slope = (y[1] - y[0]) / (x[1] - x[0])
            out[left] = y[0] + slope * (x_new[left] - x[0])
        right = x_new > x[-1]
        if np.any(right):
            slope = (y[-1] - y[-2]) / (x[-1] - x[-2])
            out[right] = y[-1] + slope * (x_new[right] - x[-1])
    return out


def load_jouzel_edc_resolution_source() -> pd.DataFrame:
    """Estimate local EDC3 sample spacing from the Jouzel et al. bag record.

    The definition mirrors ``Predictive_information_model.py``:
    local resolution is the median of the previous and next age gaps at each
    observed point. The model later uses log(local spacing), range-scaled over
    the fitted bins.
    """

    lines = JOUZEL_TXT.read_text(encoding="latin1").splitlines()
    start = None
    for idx, line in enumerate(lines):
        if line.strip().startswith("Bag") and "ztop" in line and "Age" in line:
            start = idx + 1
            break
    if start is None:
        raise ValueError(f"Cannot locate data table in {JOUZEL_TXT}.")

    raw = pd.read_csv(
        JOUZEL_TXT,
        sep=r"\s+",
        skiprows=start,
        header=None,
        names=["bag", "ztop_m", "age_yr_bp", "deuterium", "temperature"],
        encoding="latin1",
    )
    frame = pd.DataFrame(
        {
            "edc3_age_ka": pd.to_numeric(raw["age_yr_bp"], errors="coerce") / 1000.0,
            "deuterium": pd.to_numeric(raw["deuterium"], errors="coerce"),
            "temperature": pd.to_numeric(raw["temperature"], errors="coerce"),
        }
    )
    frame = frame.dropna(subset=["edc3_age_ka"])
    frame = frame.groupby("edc3_age_ka", as_index=False)[["deuterium", "temperature"]].mean()
    frame = frame.sort_values("edc3_age_ka").reset_index(drop=True)
    age = frame["edc3_age_ka"].to_numpy(dtype=float)
    previous_gap = np.r_[np.nan, np.diff(age)]
    next_gap = np.r_[np.diff(age), np.nan]
    local_spacing = np.nanmedian(np.vstack([previous_gap, next_gap]), axis=0)
    fallback = np.nanmedian(np.diff(age))
    local_spacing = np.where(np.isfinite(local_spacing) & (local_spacing > 0.0), local_spacing, fallback)
    frame["edc_local_resolution_ka"] = local_spacing
    frame["edc_local_sampling_density_per_kyr"] = 1.0 / local_spacing
    return frame


def add_edc_resolution_control(binned: pd.DataFrame, resolution_age_axis: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Append the EDC local-resolution term to a binned Barker design table."""

    centers = binned["bin_center_ka"].to_numpy(dtype=float)
    if resolution_age_axis == "edc3":
        edc3_centers = centers
    elif resolution_age_axis == "speleo_to_edc3":
        mapping = load_barker_speleo_age_mapping()
        edc3_centers = interpolate_with_linear_extrapolation(
            centers,
            mapping["speleo_age_ka"].to_numpy(dtype=float),
            mapping["edc3_age_ka"].to_numpy(dtype=float),
        )
    else:
        raise ValueError(f"Unknown resolution age axis: {resolution_age_axis}")

    source = load_jouzel_edc_resolution_source()
    spacing = interpolate_with_linear_extrapolation(
        edc3_centers,
        source["edc3_age_ka"].to_numpy(dtype=float),
        source["edc_local_resolution_ka"].to_numpy(dtype=float),
    )
    spacing = np.clip(spacing, 1e-6, None)
    log_spacing = np.log(spacing)
    scaled, mean, vmin, vmax, value_range = base.scale_to_zero_mean_range_one(log_spacing)

    out = binned.copy()
    out["resolution_edc3_age_ka"] = edc3_centers
    out["edc_local_resolution_ka"] = spacing
    out["edc_log_resolution"] = log_spacing
    out[RESOLUTION_TERM] = scaled

    meta = pd.DataFrame(
        [
            {
                "dataset_id": str(out["dataset_id"].iloc[0]),
                "forcing_id": RESOLUTION_TERM,
                "forcing_label": "log EDC local age spacing",
                "source": str(JOUZEL_TXT.relative_to(base.PROJECT_ROOT)),
                "resolution_age_axis": resolution_age_axis,
                "mean": mean,
                "min": vmin,
                "max": vmax,
                "range": value_range,
            }
        ]
    )
    return out, meta


def build_event_catalogues() -> tuple[pd.DataFrame, list[base.EventDataset]]:
    table = load_barker_table_s3()
    event_frames = []
    datasets: list[base.EventDataset] = []
    for settings in CATALOGUE_VARIANTS:
        age = pd.to_numeric(table[settings["age_column"]], errors="coerce")
        pick = pd.to_numeric(table[settings["pick_column"]], errors="coerce")
        frame = pd.DataFrame(
            {
                "event_age_ka": age,
                "pick_value": pick,
            }
        )
        frame = frame.dropna(subset=["event_age_ka", "pick_value"])
        frame = frame[frame["pick_value"].eq(1.0)].copy()
        frame = frame[
            frame["event_age_ka"].between(
                settings["analysis_start_ka"],
                settings["analysis_end_ka"],
                inclusive="both",
            )
        ].copy()
        frame = frame.sort_values("event_age_ka").reset_index(drop=True)
        frame["event_index"] = np.arange(1, len(frame) + 1)
        frame["dataset_id"] = settings["dataset_id"]
        frame["event_type"] = settings["event_type"]
        frame["event_label"] = settings["label"]
        frame["age_column"] = settings["age_column"]
        frame["pick_column"] = settings["pick_column"]
        frame["analysis_start_ka"] = settings["analysis_start_ka"]
        frame["analysis_end_ka"] = settings["analysis_end_ka"]
        frame["source"] = str(BARKER_XLS.relative_to(base.PROJECT_ROOT))
        event_frames.append(frame)

        datasets.append(
            base.EventDataset(
                dataset_id=settings["dataset_id"],
                label=settings["label"],
                color=settings["color"],
                ages_ka=frame["event_age_ka"].to_numpy(dtype=float),
                source=str(BARKER_XLS.relative_to(base.PROJECT_ROOT)),
            )
        )

    return pd.concat(event_frames, ignore_index=True), datasets


def build_rayleigh_tables(events: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    phase_products = {
        driver: rayleigh.build_phase_series(driver, settings)
        for driver, settings in rayleigh.DRIVER_SETTINGS.items()
    }
    event_phases = rayleigh.sample_event_phases(events, phase_products)
    results = rayleigh.build_rayleigh_results(event_phases)
    return event_phases, results


def build_variant_binned_inputs(
    dataset: base.EventDataset,
    analysis_end_ka: float,
    resolution_age_axis: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build base predictor bins and append history/resolution controls."""

    old_start = base.ANALYSIS_START_KA
    old_end = base.ANALYSIS_END_KA
    try:
        base.ANALYSIS_START_KA = 0.0
        base.ANALYSIS_END_KA = float(analysis_end_ka)
        binned, _, _ = base.build_binned_inputs([dataset])
    finally:
        base.ANALYSIS_START_KA = old_start
        base.ANALYSIS_END_KA = old_end

    binned = binned[
        binned["bin_center_ka"].between(0.0, analysis_end_ka, inclusive="both")
    ].copy()
    binned, resolution_meta = add_edc_resolution_control(binned, resolution_age_axis)
    binned = predictive.add_same_type_history(binned, MAIN_HISTORY_WINDOW_KA)
    return binned, resolution_meta


def fit_predictive_models(binned: pd.DataFrame) -> tuple[list[base.FittedPoissonModel], pd.DataFrame]:
    fit_frame = predictive.model_frame(binned)
    models: list[base.FittedPoissonModel] = []
    for _, group in fit_frame.groupby("dataset_id", sort=False):
        for model_id, terms, label in MODEL_SPECS:
            models.append(base.fit_poisson_model(group, model_id, terms, label))
    return models, fit_frame


def build_likelihood_tests(
    models: list[base.FittedPoissonModel],
    fit_frame: pd.DataFrame,
) -> pd.DataFrame:
    lookup = base.model_lookup(models)
    rows = []
    for dataset_id, group in fit_frame.groupby("dataset_id", sort=False):
        n_events = int(group["event_count"].sum())
        n_bins = int(len(group))
        for comparison_id, reduced_id, full_id, question in LR_TEST_SPECS:
            reduced = lookup[(dataset_id, reduced_id)]
            full = lookup[(dataset_id, full_id)]
            ll_gain = full.log_likelihood - reduced.log_likelihood
            lr_stat = 2.0 * ll_gain
            df = len(full.beta) - len(reduced.beta)
            p_value = float(chi2.sf(max(lr_stat, 0.0), df))
            rows.append(
                {
                    "dataset_id": dataset_id,
                    "dataset_label": full.dataset_label,
                    "history_window_ka": MAIN_HISTORY_WINDOW_KA,
                    "comparison_id": comparison_id,
                    "question": question,
                    "reduced_model_id": reduced_id,
                    "full_model_id": full_id,
                    "df": int(df),
                    "n_bins": n_bins,
                    "n_events": n_events,
                    "loglik_reduced": reduced.log_likelihood,
                    "loglik_full": full.log_likelihood,
                    "ll_gain_nats": ll_gain,
                    "info_bits_per_event": ll_gain / np.log(2.0) / max(n_events, 1),
                    "info_bits_per_bin": ll_gain / np.log(2.0) / max(n_bins, 1),
                    "LR_statistic": lr_stat,
                    "LR_p_value": p_value,
                    "reject_LR_at_0p05": p_value < 0.05,
                    "delta_AICc_full_minus_reduced": full.aicc - reduced.aicc,
                }
            )
    return pd.DataFrame(rows)


def build_predictive_tables(
    datasets: list[base.EventDataset],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    binned_frames = []
    summary_frames = []
    lrt_frames = []
    fitted_frames = []
    resolution_meta_frames = []

    end_lookup = {settings["dataset_id"]: settings["analysis_end_ka"] for settings in CATALOGUE_VARIANTS}
    resolution_lookup = {
        settings["dataset_id"]: settings["resolution_age_axis"] for settings in CATALOGUE_VARIANTS
    }
    for dataset in datasets:
        binned, resolution_meta = build_variant_binned_inputs(
            dataset,
            end_lookup[dataset.dataset_id],
            resolution_lookup[dataset.dataset_id],
        )
        models, fit_frame = fit_predictive_models(binned)
        summary = base.build_model_summary(models, fit_frame)
        lrt = build_likelihood_tests(models, fit_frame)
        fitted = base.build_fitted_rate_table(models, fit_frame)

        binned_frames.append(fit_frame)
        summary_frames.append(summary)
        lrt_frames.append(lrt)
        fitted_frames.append(fitted)
        resolution_meta_frames.append(resolution_meta)

    return (
        pd.concat(binned_frames, ignore_index=True),
        pd.concat(summary_frames, ignore_index=True),
        pd.concat(lrt_frames, ignore_index=True),
        pd.concat(fitted_frames, ignore_index=True),
        pd.concat(resolution_meta_frames, ignore_index=True),
    )


def draw_rayleigh_precession_panel(
    ax: plt.Axes,
    event_phases: pd.DataFrame,
    results: pd.DataFrame,
    settings: dict,
    panel_label: str,
    title_pad: float = 29.0,
) -> None:
    event_type = settings["event_type"]
    phases = event_phases[
        event_phases["driver"].eq("pre")
        & event_phases["event_type"].eq(event_type)
        & ~event_phases["phase_extrapolated"].astype(bool)
    ]["phase_rad"].to_numpy(dtype=float)
    bins = np.linspace(0.0, 2.0 * np.pi, 19)
    theta_grid = np.linspace(0.0, 2.0 * np.pi, 361)
    counts, edges = np.histogram(phases, bins=bins)
    widths = np.diff(edges)
    ax.bar(
        edges[:-1],
        counts,
        width=widths,
        align="edge",
        color=settings["color"],
        alpha=0.55,
        edgecolor="white",
        linewidth=0.6,
    )
    row = results[results["driver"].eq("pre") & results["event_type"].eq(event_type)].iloc[0]
    mean_phase = float(row["mean_phase_rad"])
    rbar = float(row["mean_resultant_length"])
    n = int(row["n_phase_events_used"])
    max_count = max(int(counts.max()), 1)
    rbar_threshold = rayleigh.rayleigh_rbar_threshold(n, RAYLEIGH_ALPHA)
    threshold_radius = rbar_threshold * max_count
    y_max = max(max_count + 1.2, threshold_radius + 1.3)
    if np.isfinite(threshold_radius):
        ax.plot(
            theta_grid,
            np.full_like(theta_grid, threshold_radius),
            color="#303030",
            linestyle=(0, (4, 2)),
            lw=1.1,
            alpha=0.9,
            zorder=4,
        )
        ax.text(
            np.deg2rad(126.0),
            min(threshold_radius + 0.72, y_max - 0.18),
            rf"$\bar{{R}}_{{0.05}}$={rbar_threshold:.2f}",
            ha="center",
            va="bottom",
            fontsize=10.2,
            color="#303030",
            bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.78, "pad": 1.0},
            zorder=5,
        )
    ax.annotate(
        "",
        xy=(mean_phase, max_count * rbar),
        xytext=(mean_phase, 0.0),
        arrowprops={"arrowstyle": "-|>", "lw": 1.6, "color": "#222222"},
    )
    ax.set_ylim(0, y_max)
    ax.set_theta_zero_location("E")
    ax.set_theta_direction(1)
    ax.set_xticks([0, np.pi / 2, np.pi, 3 * np.pi / 2])
    ax.set_xticklabels(["min", "90", "max", "270"])
    ax.tick_params(axis="x", pad=2, labelsize=9.8)
    ax.set_yticklabels([])
    ax.grid(color="0.82", alpha=0.45, lw=0.6)
    ax.text(
        -0.08,
        1.08,
        panel_label,
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=14,
        fontweight="bold",
    )
    ax.set_title(
        f"{short_variant_label(settings)}\n"
        rf"N={n}, $\bar{{R}}$={rbar:.2f}, p={format_p(row['rayleigh_p'])}" "\n"
        rf"mean={row['mean_phase_deg']:.1f} deg",
        fontsize=9.8,
        pad=title_pad,
    )


def plot_rayleigh_precession(event_phases: pd.DataFrame, results: pd.DataFrame) -> None:
    variants = [settings for settings in CATALOGUE_VARIANTS if settings["pick_column"] == "DO pick variable threshold"]
    fig, axes = plt.subplots(
        1,
        len(variants),
        figsize=(10.8, 4.3),
        subplot_kw={"projection": "polar"},
    )
    if len(variants) == 1:
        axes = [axes]
    for panel_label, ax, settings in zip("abc", axes, variants):
        draw_rayleigh_precession_panel(ax, event_phases, results, settings, panel_label, title_pad=32.0)
    fig.subplots_adjust(left=0.04, right=0.98, bottom=0.08, top=0.76, wspace=0.34)
    save_figure(fig, "fig01_barker_precession_rayleigh_polar")


def plot_predictive_summary(
    event_phases: pd.DataFrame,
    rayleigh_results: pd.DataFrame,
    lrt: pd.DataFrame,
) -> None:
    wanted = [
        "climate_lr04_co2_after_baseline",
        "phase_after_baseline_climate",
        "full_after_baseline",
    ]
    labels = {
        "climate_lr04_co2_after_baseline": "Climate-state model\nvs EP baseline",
        "phase_after_baseline_climate": "Full predictive model\nvs climate-state model",
        "full_after_baseline": "Full predictive model\nvs EP baseline",
    }
    variants = [settings["dataset_id"] for settings in CATALOGUE_VARIANTS]
    metric_specs = [
        ("minus_log10_p", r"$-\log_{10}(p)$", "d"),
        ("delta_aicc", r"$\Delta$AICc", "e"),
        ("bits_per_event", "bits per event", "f"),
    ]
    fig = plt.figure(figsize=(10.2, 8.9), constrained_layout=False)
    gs = fig.add_gridspec(2, 3, height_ratios=[1.08, 1.0])
    for idx, settings in enumerate(CATALOGUE_VARIANTS):
        ax = fig.add_subplot(gs[0, idx], projection="polar")
        draw_rayleigh_precession_panel(
            ax,
            event_phases,
            rayleigh_results,
            settings,
            panel_label=chr(ord("a") + idx),
            title_pad=20.0,
        )

    axes = [fig.add_subplot(gs[1, idx]) for idx in range(3)]
    height = 0.25
    y = np.arange(len(wanted))
    handles = []
    handle_labels = []
    for ax, (metric, ylabel, panel_label) in zip(axes, metric_specs):
        for idx, dataset_id in enumerate(variants):
            sub = lrt[lrt["dataset_id"].eq(dataset_id)].set_index("comparison_id")
            if metric == "minus_log10_p":
                values = [-np.log10(max(float(sub.loc[item, "LR_p_value"]), 1e-300)) for item in wanted]
            elif metric == "delta_aicc":
                values = [float(sub.loc[item, "delta_AICc_full_minus_reduced"]) for item in wanted]
            else:
                values = [float(sub.loc[item, "info_bits_per_event"]) for item in wanted]
            label = next(s["label"] for s in CATALOGUE_VARIANTS if s["dataset_id"] == dataset_id)
            short = (
                label.replace("Barker variable-threshold D-O warmings, ", "")
                .replace(" ka", " ka")
            )
            color = next(s["color"] for s in CATALOGUE_VARIANTS if s["dataset_id"] == dataset_id)
            bars = ax.barh(
                y + (idx - 1) * height,
                values,
                height=height,
                label=short,
                color=color,
                alpha=0.82,
            )
            if ax is axes[0]:
                handles.append(bars[0])
                handle_labels.append(short)
        if metric == "minus_log10_p":
            ax.axvline(-np.log10(0.05), color="0.25", lw=0.8, ls="--")
        if metric == "delta_aicc":
            ax.axvline(0.0, color="0.25", lw=0.8)
        ax.set_yticks(y)
        ax.set_yticklabels([labels[item] for item in wanted] if ax is axes[0] else [])
        ax.invert_yaxis()
        ax.set_xlabel(ylabel)
        ax.tick_params(axis="both", labelsize=12.0)
        ax.xaxis.label.set_size(13.0)
        ax.grid(False)
        ax.text(
            -0.08,
            1.04,
            panel_label,
            transform=ax.transAxes,
            ha="left",
            va="bottom",
            fontsize=15,
            fontweight="bold",
        )
    fig.legend(
        handles,
        handle_labels,
        frameon=False,
        loc="upper center",
        bbox_to_anchor=(0.61, 0.485),
        ncol=3,
        fontsize=10.6,
    )
    fig.subplots_adjust(left=0.19, right=0.985, bottom=0.09, top=0.86, hspace=0.74, wspace=0.55)
    save_figure(fig, "fig02_barker_predictive_likelihood_tests")


def plot_inputs_and_rates(binned: pd.DataFrame, fitted: pd.DataFrame, lrt: pd.DataFrame) -> None:
    """Plot Barker primary predictors and fitted event rates for EDC3 0-640 kyr."""

    settings = CATALOGUE_VARIANTS[0]
    dataset_id = settings["dataset_id"]
    frame = binned[binned["dataset_id"].eq(dataset_id)].copy()
    rates = fitted[fitted["dataset_id"].eq(dataset_id)].copy()
    x = frame["bin_center_ka"].to_numpy(dtype=float)

    fig = plt.figure(figsize=(9.4, 8.2))
    outer = fig.add_gridspec(
        2,
        1,
        height_ratios=[3.55, 1.65],
        hspace=0.25,
    )
    top_grid = outer[0].subgridspec(
        5,
        1,
        height_ratios=[0.95, 0.72, 0.72, 0.82, 0.62],
        hspace=0.05,
    )
    top_axes: list[plt.Axes] = []
    for idx in range(5):
        ax = fig.add_subplot(top_grid[idx, 0], sharex=top_axes[0] if top_axes else None)
        top_axes.append(ax)
    rate_ax = fig.add_subplot(outer[1, 0], sharex=top_axes[0])

    pre_ax = top_axes[0]
    pre_ax.set_zorder(2)
    pre_ax.patch.set_visible(False)
    pre_ax.plot(x, frame["pre_phase_deg"], color="#6a3d9a", lw=1.1, label="precession phase")
    raw_ax = pre_ax.twinx()
    raw_ax.set_zorder(1)
    raw_ax.plot(x, frame["precession_index"], color="#808080", lw=0.8, alpha=0.55, label="precession index")
    pre_ax.set_ylabel("precession\nphase")
    pre_ax.set_yticks([0, 180, 360])
    raw_ax.set_ylabel("precession\nindex", color="#666666")
    raw_ax.tick_params(axis="y", colors="#666666", labelsize=9.5)

    top_axes[1].plot(x, frame["lr04"], color="#3f7f93", lw=1.0)
    top_axes[1].set_ylabel(r"LR04 $\delta^{18}$O")
    top_axes[1].invert_yaxis()

    top_axes[2].plot(x, frame["co2"], color="#8a5a44", lw=1.0)
    top_axes[2].set_ylabel("CO$_2$")

    edc = load_jouzel_edc_resolution_source()
    edc = edc[
        edc["edc3_age_ka"].between(settings["analysis_start_ka"], settings["analysis_end_ka"])
    ]
    edc_plot = edc[edc["deuterium"].between(-650.0, -250.0)]
    top_axes[3].plot(
        edc_plot["edc3_age_ka"],
        edc_plot["deuterium"],
        color="#202020",
        lw=0.72,
    )
    top_axes[3].set_ylabel(r"EDC $\delta$D")

    density = 1.0 / frame["edc_local_resolution_ka"].to_numpy(dtype=float)
    top_axes[4].plot(x, density, color="#444444", lw=0.9)
    top_axes[4].fill_between(x, 0.0, density, color="#444444", alpha=0.14)
    top_axes[4].set_ylabel("Density")
    top_axes[4].set_ylim(0.0, np.nanmax(density) * 1.05)

    rate_models = [
        ("history_resolution_baseline", "event-process baseline", "#8c8c8c", "-", 0.72, 1.0),
        ("baseline_climate_lr04_co2", "climate-state model", "#2b6cb0", "-", 1.0, 1.15),
        ("baseline_climate_lr04_co2_pre_phase", "full predictive model", "#c51b7d", "-", 1.0, 1.15),
    ]
    max_rate = 0.0
    for model_id, label, color, ls, alpha, lw in rate_models:
        sub = rates[rates["model_id"].eq(model_id)]
        max_rate = max(max_rate, float(sub["lambda_per_kyr"].max()))
        rate_ax.plot(
            sub["bin_center_ka"],
            sub["lambda_per_kyr"],
            color=color,
            lw=lw,
            ls=ls,
            alpha=alpha,
            label=label,
        )
    y_max = max_rate * 1.18
    event_x = rates[
        rates["model_id"].eq("stationary") & rates["event_count"].gt(0)
    ]["bin_center_ka"].to_numpy(dtype=float)
    rate_ax.vlines(
        event_x,
        ymin=0.0,
        ymax=y_max,
        color=settings["color"],
        alpha=0.78,
        lw=0.65,
        label="Barker warming events",
    )
    rate_ax.set_ylim(0.0, y_max)
    rate_ax.set_ylabel("rate / kyr")
    phase_test = lrt[
        lrt["dataset_id"].eq(dataset_id)
        & lrt["comparison_id"].eq("phase_after_baseline_climate")
    ].iloc[0]
    rate_ax.text(
        0.99,
        0.94,
        "Precession phase after climate-state model: "
        f"LR={phase_test['LR_statistic']:.2f}, p={format_p(phase_test['LR_p_value'])}",
        transform=rate_ax.transAxes,
        ha="right",
        va="top",
        fontsize=9.6,
        bbox={"boxstyle": "round,pad=0.25", "facecolor": "white", "edgecolor": "#bbbbbb", "alpha": 0.86},
    )
    rate_ax.legend(
        handles=[
            Line2D(
                [0],
                [0],
                color="#8c8c8c",
                lw=1.2,
                ls="-",
                alpha=0.78,
                label="event-process baseline",
            ),
            Line2D([0], [0], color="#2b6cb0", lw=1.3, label="climate-state model"),
            Line2D([0], [0], color="#c51b7d", lw=1.3, label="full predictive model"),
            Line2D(
                [0],
                [0],
                marker="|",
                linestyle="None",
                markersize=13,
                markeredgewidth=1.5,
                color=settings["color"],
                label="Barker warming events",
            ),
        ],
        frameon=False,
        loc="lower right",
        bbox_to_anchor=(1.0, 1.04),
        ncol=2,
        fontsize=11.0,
        borderaxespad=0.0,
    )

    for ax in top_axes:
        ax.grid(False)
        ax.set_xlim(settings["analysis_start_ka"], settings["analysis_end_ka"])
        ax.tick_params(axis="x", labelbottom=False, length=0)
        ax.tick_params(axis="y", length=2.5, pad=2, labelsize=9.5)
        ax.yaxis.label.set_size(10.5)
        for spine in ax.spines.values():
            spine.set_visible(False)
    top_axes[-1].tick_params(
        axis="x",
        labelbottom=False,
        bottom=True,
        top=False,
        length=3.0,
        width=0.8,
        direction="out",
    )
    for spine in raw_ax.spines.values():
        spine.set_visible(False)
    raw_ax.grid(False)
    raw_ax.set_xlim(settings["analysis_start_ka"], settings["analysis_end_ka"])

    rate_ax.grid(False)
    rate_ax.set_xlim(settings["analysis_start_ka"], settings["analysis_end_ka"])
    rate_ax.set_xlabel("Age (kyr BP, EDC3)")
    rate_ax.tick_params(axis="both", labelsize=10.5)
    rate_ax.xaxis.label.set_size(11.5)
    rate_ax.yaxis.label.set_size(11.5)
    for label, ax, y_offset in (("a", top_axes[0], 1.08), ("b", rate_ax, 1.02)):
        ax.text(
            -0.045,
            y_offset,
            label,
            transform=ax.transAxes,
            ha="right",
            va="bottom",
            fontsize=13,
            fontweight="bold",
            clip_on=False,
        )

    fig.subplots_adjust(left=0.11, right=0.90, top=0.985, bottom=0.08)
    fig.canvas.draw()
    top_positions = [ax.get_position() for ax in top_axes]
    x0 = min(pos.x0 for pos in top_positions)
    y0 = min(pos.y0 for pos in top_positions)
    x1 = max(pos.x1 for pos in top_positions)
    y1 = max(pos.y1 for pos in top_positions)
    fig.add_artist(
        Rectangle(
            (x0, y0),
            x1 - x0,
            y1 - y0,
            transform=fig.transFigure,
            fill=False,
            edgecolor="#1f1f1f",
            linewidth=0.9,
            zorder=20,
        )
    )
    save_figure(fig, "fig03_barker_inputs_and_fitted_rates")


def write_outputs(
    events: pd.DataFrame,
    event_phases: pd.DataFrame,
    rayleigh_results: pd.DataFrame,
    binned: pd.DataFrame,
    summary: pd.DataFrame,
    lrt: pd.DataFrame,
    fitted: pd.DataFrame,
    resolution_meta: pd.DataFrame,
) -> None:
    ensure_dir(OUT_DATA_DIR)
    events.to_csv(OUT_DATA_DIR / "barker2011_event_catalogues_used.csv", index=False)
    event_phases.to_csv(OUT_DATA_DIR / "barker2011_event_orbital_phases.csv", index=False)
    rayleigh_results.to_csv(OUT_DATA_DIR / "barker2011_rayleigh_phase_results.csv", index=False)
    binned.to_csv(OUT_DATA_DIR / "barker2011_binned_predictive_inputs.csv", index=False)
    summary.to_csv(OUT_DATA_DIR / "barker2011_predictive_model_summary.csv", index=False)
    lrt.to_csv(OUT_DATA_DIR / "barker2011_predictive_likelihood_tests.csv", index=False)
    fitted.to_csv(OUT_DATA_DIR / "barker2011_fitted_rates.csv", index=False)
    resolution_meta.to_csv(OUT_DATA_DIR / "barker2011_edc_resolution_scale_summary.csv", index=False)
    pd.DataFrame(
        [
            {
                "run_name": RUN_NAME,
                "source": str(BARKER_XLS.relative_to(base.PROJECT_ROOT)),
                "resolution_source": str(JOUZEL_TXT.relative_to(base.PROJECT_ROOT)),
                "bin_width_ka": base.BIN_WIDTH_KA,
                "history_window_ka": MAIN_HISTORY_WINDOW_KA,
                "note": "Baseline includes same-type history and EDC local age-spacing resolution.",
            }
        ]
    ).to_csv(OUT_DATA_DIR / "parameters.csv", index=False)


def print_summary(rayleigh_results: pd.DataFrame, summary: pd.DataFrame, lrt: pd.DataFrame) -> None:
    print("\nBarker et al. (2011) Rayleigh results:")
    ray = rayleigh_results[rayleigh_results["driver"].eq("pre")].copy()
    print(
        ray[
            [
                "event_type",
                "n_phase_events_used",
                "mean_phase_deg",
                "mean_resultant_length",
                "rayleigh_z",
                "rayleigh_p",
            ]
        ].to_string(index=False)
    )

    print("\nPredictive-information likelihood tests:")
    key_tests = lrt[
        lrt["comparison_id"].isin(
            [
                "climate_lr04_co2_after_baseline",
                "phase_after_baseline_climate",
                "full_after_baseline",
            ]
        )
    ].copy()
    print(
        key_tests[
            [
                "dataset_id",
                "comparison_id",
                "n_events",
                "LR_statistic",
                "df",
                "LR_p_value",
                "delta_AICc_full_minus_reduced",
                "info_bits_per_event",
            ]
        ].to_string(index=False)
    )

    full = summary[summary["model_id"].eq("baseline_climate_lr04_co2_pre_phase")].copy()
    cols = [
        "dataset_id",
        "n_events",
        "delta_AICc",
        "pre_phase_preferred_deg",
        "pre_phase_rate_ratio_max_vs_min",
    ]
    print("\nFull-model precession-phase estimates:")
    print(full[cols].to_string(index=False))
    print(f"\nWrote outputs to {OUT_DATA_DIR.relative_to(base.PROJECT_ROOT)}")
    print(f"Wrote figures to {OUT_FIG_DIR.relative_to(base.PROJECT_ROOT)}")


def main() -> None:
    events, datasets = build_event_catalogues()
    event_phases, rayleigh_results = build_rayleigh_tables(events)
    binned, model_summary, lrt, fitted, resolution_meta = build_predictive_tables(datasets)
    write_outputs(
        events,
        event_phases,
        rayleigh_results,
        binned,
        model_summary,
        lrt,
        fitted,
        resolution_meta,
    )
    plot_rayleigh_precession(event_phases, rayleigh_results)
    plot_predictive_summary(event_phases, rayleigh_results, lrt)
    plot_inputs_and_rates(binned, fitted, lrt)
    print_summary(rayleigh_results, model_summary, lrt)


if __name__ == "__main__":
    main()
