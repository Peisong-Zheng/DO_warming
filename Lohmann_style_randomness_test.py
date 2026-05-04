"""
Test whether Rousseau et al. (2023) Chinese-speleo abrupt monsoon-start
chronologies are consistent with stationary random occurrence.

The primary test follows the one-process event-count analysis of
Lohmann and Ditlevsen (2018): events are modeled as a fixed-rate Poisson
process, the number of events in a running window E(t) is calculated, and
the scalar statistic E_S is the RMS departure of E(t) from its constant
expectation. Monte Carlo realizations with the same observation length are
used to obtain a one-sided p value.

This Rousseau-specific script uses the strong- and weak-monsoon transition
start times only. It does not implement the two-process stadial/interstadial
model from Lohmann and Ditlevsen because the strong/weak start lists are
tested here as separate event sequences.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parent
RUN_NAME = "rousseau2023_monsoon_randomness_lohmann"
OUT_DATA_DIR = PROJECT_ROOT / "data" / "processed" / RUN_NAME
OUT_FIG_DIR = PROJECT_ROOT / "figures" / RUN_NAME

ROUSSEAU_STRONG_CSV = PROJECT_ROOT / "data/raw/rousseau_2023_strong_monsoon_start_times.csv"
ROUSSEAU_WEAK_CSV = PROJECT_ROOT / "data/raw/rousseau_2023_weak_monsoon_start_times.csv"
ROUSSEAU_MAX_AGE_KA = 640.0

RANDOM_SEED = 20260428
N_MONTE_CARLO = 5000
EVENT_COUNT_WINDOW_KA = 20.0
TIME_GRID_STEP_KA = 0.2


plt.rcParams.update(
    {
        "font.family": "DejaVu Sans",
        "font.size": 10,
        "axes.titlesize": 10,
        "axes.labelsize": 10,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "legend.fontsize": 8,
        "axes.linewidth": 0.8,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": False,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    }
)


@dataclass
class EventDataset:
    dataset_id: str
    label: str
    ages_ka: np.ndarray
    analysis_start_ka: float
    analysis_end_ka: float
    source: str
    events: pd.DataFrame


@dataclass
class RandomnessResult:
    dataset: EventDataset
    centers_ka: np.ndarray
    observed_counts: np.ndarray
    expected_count: float
    observed_es: float
    conditional_es_samples: np.ndarray
    poisson_es_samples: np.ndarray
    conditional_count_q025: np.ndarray
    conditional_count_q500: np.ndarray
    conditional_count_q975: np.ndarray
    conditional_es_p_value: float
    poisson_es_p_value: float


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def as_numeric_age(df: pd.DataFrame, age_col: str, min_age: float, max_age: float) -> pd.Series:
    ages = pd.to_numeric(df[age_col], errors="coerce")
    ages = ages[np.isfinite(ages)]
    ages = ages[(ages >= min_age) & (ages <= max_age)]
    return ages


def unique_sorted(values: np.ndarray, decimals: int = 6) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    values = np.round(values, decimals=decimals)
    return np.unique(np.sort(values))


def load_rousseau_monsoon_events(path: Path, dataset_id: str, label: str) -> EventDataset:
    df = pd.read_csv(path, encoding="utf-8-sig")
    if "start_time_ka_BP" not in df.columns:
        raise ValueError(f"{path} must contain a start_time_ka_BP column.")
    events = df.copy()
    events["event_age_ka"] = as_numeric_age(events, "start_time_ka_BP", 0.0, ROUSSEAU_MAX_AGE_KA)
    events = events.dropna(subset=["event_age_ka"]).sort_values("event_age_ka").reset_index(drop=True)
    events["source_event_id"] = events.get("order", events.index + 1)
    return EventDataset(
        dataset_id=dataset_id,
        label=label,
        ages_ka=unique_sorted(events["event_age_ka"].to_numpy()),
        analysis_start_ka=0.0,
        analysis_end_ka=ROUSSEAU_MAX_AGE_KA,
        source=str(path.relative_to(PROJECT_ROOT)),
        events=events,
    )


def load_all_event_datasets() -> list[EventDataset]:
    return [
        load_rousseau_monsoon_events(
            ROUSSEAU_STRONG_CSV,
            "rousseau2023_strong_monsoon_starts",
            "Rousseau 2023 strong monsoon starts",
        ),
        load_rousseau_monsoon_events(
            ROUSSEAU_WEAK_CSV,
            "rousseau2023_weak_monsoon_starts",
            "Rousseau 2023 weak monsoon starts",
        ),
    ]


def moving_event_counts(
    event_ages_ka: np.ndarray,
    centers_ka: np.ndarray,
    window_ka: float,
) -> np.ndarray:
    event_ages_ka = np.sort(np.asarray(event_ages_ka, dtype=float))
    left = centers_ka - 0.5 * window_ka
    right = centers_ka + 0.5 * window_ka
    return (
        np.searchsorted(event_ages_ka, right, side="right")
        - np.searchsorted(event_ages_ka, left, side="left")
    ).astype(float)


def centered_time_grid(start_ka: float, end_ka: float, window_ka: float, step_ka: float) -> np.ndarray:
    if end_ka - start_ka <= window_ka:
        raise ValueError("The analysis interval must be longer than the moving window.")
    first = start_ka + 0.5 * window_ka
    last = end_ka - 0.5 * window_ka
    return np.arange(first, last + 0.5 * step_ka, step_ka)


def rms_deviation(values: np.ndarray, expected: float) -> float:
    return float(np.sqrt(np.mean((np.asarray(values, dtype=float) - expected) ** 2)))


def p_value_greater_or_equal(samples: np.ndarray, observed: float) -> float:
    samples = np.asarray(samples, dtype=float)
    samples = samples[np.isfinite(samples)]
    return float((np.count_nonzero(samples >= observed) + 1.0) / (samples.size + 1.0))


def simulate_lohmann_one_process(
    dataset: EventDataset,
    window_ka: float,
    grid_step_ka: float,
    n_monte_carlo: int,
    rng: np.random.Generator,
) -> RandomnessResult:
    ages = dataset.ages_ka[
        (dataset.ages_ka >= dataset.analysis_start_ka)
        & (dataset.ages_ka <= dataset.analysis_end_ka)
    ]
    ages = unique_sorted(ages)
    if ages.size < 3:
        raise ValueError(f"{dataset.dataset_id} has too few events for this test.")

    duration_ka = float(dataset.analysis_end_ka - dataset.analysis_start_ka)
    rate_per_kyr = ages.size / duration_ka
    expected_count = rate_per_kyr * window_ka
    centers = centered_time_grid(
        dataset.analysis_start_ka,
        dataset.analysis_end_ka,
        window_ka,
        grid_step_ka,
    )
    observed_counts = moving_event_counts(ages, centers, window_ka)
    observed_es = rms_deviation(observed_counts, expected_count)

    conditional_counts = np.empty((n_monte_carlo, centers.size), dtype=np.float32)
    conditional_es = np.empty(n_monte_carlo, dtype=float)
    poisson_es = np.empty(n_monte_carlo, dtype=float)
    for i in range(n_monte_carlo):
        conditional_events = np.sort(
            rng.uniform(dataset.analysis_start_ka, dataset.analysis_end_ka, size=ages.size)
        )
        counts = moving_event_counts(conditional_events, centers, window_ka)
        conditional_counts[i] = counts
        conditional_es[i] = rms_deviation(counts, expected_count)

        n_events = int(rng.poisson(rate_per_kyr * duration_ka))
        poisson_events = np.sort(
            rng.uniform(dataset.analysis_start_ka, dataset.analysis_end_ka, size=n_events)
        )
        poisson_counts = moving_event_counts(poisson_events, centers, window_ka)
        poisson_es[i] = rms_deviation(poisson_counts, expected_count)

    return RandomnessResult(
        dataset=dataset,
        centers_ka=centers,
        observed_counts=observed_counts,
        expected_count=expected_count,
        observed_es=observed_es,
        conditional_es_samples=conditional_es,
        poisson_es_samples=poisson_es,
        conditional_count_q025=np.quantile(conditional_counts, 0.025, axis=0),
        conditional_count_q500=np.quantile(conditional_counts, 0.500, axis=0),
        conditional_count_q975=np.quantile(conditional_counts, 0.975, axis=0),
        conditional_es_p_value=p_value_greater_or_equal(conditional_es, observed_es),
        poisson_es_p_value=p_value_greater_or_equal(poisson_es, observed_es),
    )


def build_summary_table(results: list[RandomnessResult], window_ka: float) -> pd.DataFrame:
    rows = []
    for result in results:
        ds = result.dataset
        duration = ds.analysis_end_ka - ds.analysis_start_ka
        ages = ds.ages_ka[
            (ds.ages_ka >= ds.analysis_start_ka)
            & (ds.ages_ka <= ds.analysis_end_ka)
        ]
        rows.append(
            {
                "dataset_id": ds.dataset_id,
                "dataset_label": ds.label,
                "source": ds.source,
                "analysis_start_ka": ds.analysis_start_ka,
                "analysis_end_ka": ds.analysis_end_ka,
                "duration_ka": duration,
                "n_events": len(ages),
                "rate_per_kyr_from_event_count": len(ages) / duration,
                f"expected_events_per_{window_ka:g}kyr": result.expected_count,
                "lohmann_event_count_ES": result.observed_es,
                "lohmann_poisson_ES_p_value": result.poisson_es_p_value,
                "conditional_fixed_n_ES_p_value": result.conditional_es_p_value,
                "reject_poisson_ES_at_0p05": result.poisson_es_p_value < 0.05,
                "reject_fixed_n_ES_at_0p05": result.conditional_es_p_value < 0.05,
            }
        )
    return pd.DataFrame(rows)


def build_event_sequence_table(datasets: list[EventDataset]) -> pd.DataFrame:
    rows = []
    for ds in datasets:
        for idx, event in ds.events.sort_values("event_age_ka").reset_index(drop=True).iterrows():
            age = float(event["event_age_ka"])
            if age < ds.analysis_start_ka or age > ds.analysis_end_ka:
                continue
            rows.append(
                {
                    "dataset_id": ds.dataset_id,
                    "dataset_label": ds.label,
                    "event_index": idx + 1,
                    "event_age_ka": age,
                    "analysis_start_ka": ds.analysis_start_ka,
                    "analysis_end_ka": ds.analysis_end_ka,
                    "source_event_id": event.get("source_event_id", idx + 1),
                    "cheng_event_age_ka": event.get("cheng_event_age_ka", np.nan),
                    "ch4_event_age_ka": event.get("ch4_event_age_ka", np.nan),
                    "event_label": event.get("event_label", np.nan),
                    "source": ds.source,
                }
            )
    return pd.DataFrame(rows)


def build_moving_window_table(results: list[RandomnessResult]) -> pd.DataFrame:
    rows = []
    for result in results:
        for center, observed, q025, q500, q975 in zip(
            result.centers_ka,
            result.observed_counts,
            result.conditional_count_q025,
            result.conditional_count_q500,
            result.conditional_count_q975,
        ):
            rows.append(
                {
                    "dataset_id": result.dataset.dataset_id,
                    "window_center_ka": center,
                    "observed_event_count": observed,
                    "expected_event_count": result.expected_count,
                    "fixed_n_mc_q025": q025,
                    "fixed_n_mc_q500": q500,
                    "fixed_n_mc_q975": q975,
                }
            )
    return pd.DataFrame(rows)


def panel_grid(n_panels: int) -> tuple[int, int]:
    ncols = 2
    nrows = int(np.ceil(n_panels / ncols))
    return nrows, ncols


def save_figure(fig: plt.Figure, stem: str, write_pdf: bool) -> None:
    ensure_dir(OUT_FIG_DIR)
    fig.savefig(OUT_FIG_DIR / f"{stem}.png", dpi=300, bbox_inches="tight")
    if write_pdf:
        fig.savefig(OUT_FIG_DIR / f"{stem}.pdf", bbox_inches="tight")
    plt.close(fig)


def short_label(label: str) -> str:
    return label.replace("Rousseau 2023 ", "")


def format_p_value(value: float) -> str:
    if value < 0.001:
        return "<0.001"
    return f"{value:.3f}"


def add_panel_label(ax: plt.Axes, label: str, x: float = -0.12, y: float = 1.04) -> None:
    ax.text(
        x,
        y,
        label,
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=11,
        fontweight="bold",
        clip_on=False,
    )


def add_inside_legend(
    ax: plt.Axes,
    handles: list,
    labels: list[str],
    title: str,
    loc: str = "upper right",
    ncol: int = 1,
    bbox_to_anchor: tuple[float, float] | None = None,
) -> None:
    legend = ax.legend(
        handles,
        labels,
        title=title,
        loc=loc,
        ncol=ncol,
        bbox_to_anchor=bbox_to_anchor,
        frameon=True,
        fancybox=False,
        framealpha=0.88,
        facecolor="white",
        edgecolor="#bbbbbb",
        fontsize=5.8,
        title_fontsize=6.4,
        handlelength=1.7,
        handletextpad=0.45,
        borderpad=0.32,
        labelspacing=0.24,
        columnspacing=0.75,
    )
    legend._legend_box.align = "left"


def plot_event_count_windows(results: list[RandomnessResult], window_ka: float, write_pdf: bool) -> None:
    nrows, ncols = panel_grid(len(results))
    fig, axes = plt.subplots(nrows, ncols, figsize=(7.2, 3.2 * nrows), sharey=False)
    axes = np.atleast_1d(axes).ravel()
    for panel, (ax, result) in enumerate(zip(axes, results)):
        n_events = len(result.dataset.ages_ka)
        add_panel_label(ax, chr(ord("a") + panel))
        ax.fill_between(
            result.centers_ka,
            result.conditional_count_q025,
            result.conditional_count_q975,
            color="#d0d0d0",
            alpha=0.75,
            linewidth=0.0,
        )
        ax.plot(
            result.centers_ka,
            result.conditional_count_q500,
            color="#777777",
            linewidth=1.0,
            linestyle=":",
        )
        ax.plot(result.centers_ka, result.observed_counts, color="#202020", linewidth=1.6)
        ax.axhline(result.expected_count, color="#d95f02", linestyle="--", linewidth=1.1)
        ax.set_xlabel("Age (ka)")
        ax.set_ylabel("Events per window")
        handles = [
            Patch(facecolor="#d0d0d0", edgecolor="none", alpha=0.75),
            Line2D([0], [0], color="#777777", linewidth=1.0, linestyle=":"),
            Line2D([0], [0], color="#202020", linewidth=1.6),
            Line2D([0], [0], color="#d95f02", linewidth=1.1, linestyle="--"),
        ]
        labels = [
            rf"fixed-N 95%: $p_N$={format_p_value(result.conditional_es_p_value)}",
            "fixed-N null median",
            rf"observed $E(t)$: $E_S$={result.observed_es:.2f}",
            rf"Poisson mean: $p_P$={format_p_value(result.poisson_es_p_value)}",
        ]
        add_inside_legend(
            ax,
            handles,
            labels,
            title=f"{short_label(result.dataset.label)}\nN={n_events}",
            loc="lower center",
            ncol=2,
            bbox_to_anchor=(0.5, 1.03),
        )
    for ax in axes[len(results) :]:
        ax.axis("off")
    fig.subplots_adjust(left=0.10, right=0.99, bottom=0.16, top=0.68, hspace=0.60, wspace=0.30)
    save_figure(fig, "lohmann_moving_window_event_counts", write_pdf)


def plot_es_distributions(results: list[RandomnessResult], write_pdf: bool) -> None:
    nrows, ncols = panel_grid(len(results))
    fig, axes = plt.subplots(nrows, ncols, figsize=(7.2, 3.2 * nrows))
    axes = np.atleast_1d(axes).ravel()
    for panel, (ax, result) in enumerate(zip(axes, results)):
        add_panel_label(ax, chr(ord("a") + panel))
        bins = np.histogram_bin_edges(
            np.concatenate([result.conditional_es_samples, result.poisson_es_samples, [result.observed_es]]),
            bins=40,
        )
        ax.hist(
            result.poisson_es_samples,
            bins=bins,
            color="#9ecae1",
            alpha=0.55,
            density=True,
            edgecolor="#4d8fbd",
            linewidth=0.25,
        )
        ax.hist(
            result.conditional_es_samples,
            bins=bins,
            histtype="step",
            color="#222222",
            linewidth=1.7,
            density=True,
            linestyle="-",
        )
        ax.axvline(result.observed_es, color="#d95f02", linewidth=1.8)
        ax.set_xlabel(r"$E_S$")
        ax.set_ylabel("Density")
        handles = [
            Patch(facecolor="#9ecae1", edgecolor="#4d8fbd", alpha=0.55),
            Line2D([0], [0], color="#222222", linewidth=1.7),
            Line2D([0], [0], color="#d95f02", linewidth=1.8),
        ]
        labels = [
            rf"Poisson null: $p_P$={format_p_value(result.poisson_es_p_value)}",
            rf"fixed-N null: $p_N$={format_p_value(result.conditional_es_p_value)}",
            rf"observed $E_S$={result.observed_es:.2f}",
        ]
        add_inside_legend(
            ax,
            handles,
            labels,
            title=f"{short_label(result.dataset.label)}\nN={len(result.dataset.ages_ka)}",
            loc="lower center",
            bbox_to_anchor=(0.5, 1.03),
        )
    for ax in axes[len(results) :]:
        ax.axis("off")
    fig.subplots_adjust(left=0.10, right=0.99, bottom=0.16, top=0.68, hspace=0.60, wspace=0.30)
    save_figure(fig, "lohmann_ES_null_distributions", write_pdf)


def plot_summary_p_values(summary: pd.DataFrame, write_pdf: bool) -> None:
    labels = [
        "Strong\nmonsoon" if "strong" in item else "Weak\nmonsoon"
        for item in summary["dataset_id"]
    ]
    x = np.arange(len(summary))
    width = 0.28
    fig, ax = plt.subplots(figsize=(4.8, 2.8))
    add_panel_label(ax, "a", x=-0.08)
    poisson_bars = ax.bar(
        x - width / 2,
        summary["lohmann_poisson_ES_p_value"],
        width,
        color="#3182bd",
        label="Poisson-rate null",
    )
    fixed_n_bars = ax.bar(
        x + width / 2,
        summary["conditional_fixed_n_ES_p_value"],
        width,
        color="#969696",
        label="fixed-N null",
    )
    ax.axhline(0.05, color="#222222", linestyle="--", linewidth=1.0)
    ax.text(
        0.99,
        0.052,
        "p=0.05",
        transform=ax.get_yaxis_transform(),
        ha="right",
        va="bottom",
        fontsize=7,
    )
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=0, ha="center", fontsize=8)
    ax.set_ylabel("p value")
    max_p = float(
        summary[["lohmann_poisson_ES_p_value", "conditional_fixed_n_ES_p_value"]]
        .to_numpy()
        .max()
    )
    ax.set_ylim(0.0, min(1.02, max(0.10, 0.065, max_p * 1.35)))
    ax.legend(frameon=False, ncol=1, loc="upper right", fontsize=7, handlelength=1.8)
    for bars, values in [
        (poisson_bars, summary["lohmann_poisson_ES_p_value"]),
        (fixed_n_bars, summary["conditional_fixed_n_ES_p_value"]),
    ]:
        for bar, value in zip(bars, values):
            value = float(value)
            x_text = bar.get_x() + bar.get_width() / 2
            if abs(value - 0.05) < 0.008:
                y_text = max(value - 0.004, 0.003)
                va = "top"
            else:
                y_text = value + 0.002
                va = "bottom"
            ax.text(
                x_text,
                y_text,
                format_p_value(value),
                ha="center",
                va=va,
                fontsize=7,
                bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.70, "pad": 0.4},
            )
    ax.margins(x=0.18)
    fig.subplots_adjust(left=0.15, right=0.98, bottom=0.30, top=0.92)
    save_figure(fig, "randomness_test_p_value_summary", write_pdf)


def run_analysis(
    n_monte_carlo: int,
    seed: int,
    window_ka: float,
    grid_step_ka: float,
    write_pdf: bool,
) -> pd.DataFrame:
    ensure_dir(OUT_DATA_DIR)
    ensure_dir(OUT_FIG_DIR)
    rng = np.random.default_rng(seed)
    datasets = load_all_event_datasets()
    results = [
        simulate_lohmann_one_process(
            dataset=dataset,
            window_ka=window_ka,
            grid_step_ka=grid_step_ka,
            n_monte_carlo=n_monte_carlo,
            rng=rng,
        )
        for dataset in datasets
    ]

    summary = build_summary_table(results, window_ka)
    events = build_event_sequence_table(datasets)
    moving = build_moving_window_table(results)

    summary.to_csv(OUT_DATA_DIR / "rousseau2023_monsoon_randomness_summary.csv", index=False)
    events.to_csv(OUT_DATA_DIR / "rousseau2023_monsoon_sequences_used.csv", index=False)
    moving.to_csv(OUT_DATA_DIR / "lohmann_moving_window_monsoon_start_counts.csv", index=False)

    plot_event_count_windows(results, window_ka, write_pdf)
    plot_es_distributions(results, write_pdf)
    plot_summary_p_values(summary, write_pdf)
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Lohmann-style random occurrence tests for Rousseau 2023 monsoon-start sequences."
    )
    parser.add_argument("--n-monte-carlo", type=int, default=N_MONTE_CARLO)
    parser.add_argument("--seed", type=int, default=RANDOM_SEED)
    parser.add_argument("--window-ka", type=float, default=EVENT_COUNT_WINDOW_KA)
    parser.add_argument("--grid-step-ka", type=float, default=TIME_GRID_STEP_KA)
    parser.add_argument("--no-pdf", action="store_true", help="Only write PNG figures.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary = run_analysis(
        n_monte_carlo=args.n_monte_carlo,
        seed=args.seed,
        window_ka=args.window_ka,
        grid_step_ka=args.grid_step_ka,
        write_pdf=not args.no_pdf,
    )
    display_cols = [
        "dataset_id",
        "n_events",
        "lohmann_event_count_ES",
        "lohmann_poisson_ES_p_value",
        "conditional_fixed_n_ES_p_value",
    ]
    print(summary[display_cols].to_string(index=False))
    print(f"\nWrote tables to: {OUT_DATA_DIR.relative_to(PROJECT_ROOT)}")
    print(f"Wrote figures to: {OUT_FIG_DIR.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    main()
