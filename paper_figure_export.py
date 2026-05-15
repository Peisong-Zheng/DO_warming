"""Export only paper-referenced PDFs with manuscript-style names.

The analysis scripts keep their normal outputs under the project-level
``figures/`` tree.  For Overleaf, the manuscript uses a flat
``monsoon_paper/figures/`` directory whose filenames match the figure order:
``Fig01.pdf``--``Fig04.pdf`` for the main text and ``FigS01.pdf``--``FigS09.pdf``
for Supporting Information.

This module serves two purposes:

1. plotting scripts can call ``save_paper_pdf(fig, project_root, stem)`` or
   ``copy_pdf_to_paper(...)`` and only paper-referenced figures are exported;
2. running ``python paper_figure_export.py`` refreshes the whole Overleaf-ready
   figure directory from the canonical project-level figure outputs.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import shutil


PROJECT_ROOT = Path(__file__).resolve().parent


@dataclass(frozen=True)
class PaperFigure:
    stem: str
    paper_name: str
    source_relpath: str
    tex_file: str
    figure_label: str


PAPER_FIGURES: tuple[PaperFigure, ...] = (
    PaperFigure(
        stem="fig01_adjusted_inputs_and_fitted_hazards",
        paper_name="Fig01.pdf",
        source_relpath="figures/Predictive_information_model/fig01_adjusted_inputs_and_fitted_hazards.pdf",
        tex_file="main.tex",
        figure_label="fig:data_phase_framework",
    ),
    PaperFigure(
        stem="fig03a_rayleigh_phase_polar_precession",
        paper_name="Fig02.pdf",
        source_relpath="figures/Orbital_phase_rayleigh/fig03a_rayleigh_phase_polar_precession.pdf",
        tex_file="main.tex",
        figure_label="fig:precession_rayleigh",
    ),
    PaperFigure(
        stem="fig02_adjusted_model_comparison_delta_aicc",
        paper_name="Fig03.pdf",
        source_relpath="figures/Predictive_information_model/fig02_adjusted_model_comparison_delta_aicc.pdf",
        tex_file="main.tex",
        figure_label="fig:predictive_model_comparison",
    ),
    PaperFigure(
        stem="fig05_core_component_lagged_information_0_10kyr",
        paper_name="Fig04.pdf",
        source_relpath="figures/Lagged_predictive_information/fig05_core_component_lagged_information_0_10kyr.pdf",
        tex_file="main.tex",
        figure_label="fig:lagged-single-driver-info",
    ),
    PaperFigure(
        stem="fig03b_rayleigh_phase_polar_obliquity",
        paper_name="FigS01.pdf",
        source_relpath="figures/Orbital_phase_rayleigh/fig03b_rayleigh_phase_polar_obliquity.pdf",
        tex_file="SI.tex",
        figure_label="fig:obliquity_rayleigh",
    ),
    PaperFigure(
        stem="fig04_phase_ecdf_uniform_check",
        paper_name="FigS02.pdf",
        source_relpath="figures/Orbital_phase_rayleigh/fig04_phase_ecdf_uniform_check.pdf",
        tex_file="SI.tex",
        figure_label="fig:phase_ecdf_uniform_check",
    ),
    PaperFigure(
        stem="fig03_fitted_hazards_from_fig01de",
        paper_name="FigS03.pdf",
        source_relpath="figures/Bin_hazard_phase_poisson/fig03_fitted_hazards_from_fig01de.pdf",
        tex_file="SI.tex",
        figure_label="fig:climate_phase_hazards_no_event_baseline",
    ),
    PaperFigure(
        stem="fig07_extended_baseline_predictor_correlation",
        paper_name="FigS04.pdf",
        source_relpath="figures/Bin_hazard_phase_poisson_sensitivity/fig07_extended_baseline_predictor_correlation.pdf",
        tex_file="SI.tex",
        figure_label="fig:extended_baseline_correlation",
    ),
    PaperFigure(
        stem="fig06_sensitivity_aicc_and_likelihood_tests",
        paper_name="FigS05.pdf",
        source_relpath="figures/Bin_hazard_phase_poisson_sensitivity/fig06_sensitivity_aicc_and_likelihood_tests.pdf",
        tex_file="SI.tex",
        figure_label="fig:sensitivity_aicc_lrt",
    ),
    PaperFigure(
        stem="fig04_base_vs_extended_hazards",
        paper_name="FigS06.pdf",
        source_relpath="figures/Bin_hazard_phase_poisson_sensitivity/fig04_base_vs_extended_hazards.pdf",
        tex_file="SI.tex",
        figure_label="fig:base_vs_extended_hazards",
    ),
    PaperFigure(
        stem="fig09_age_offsets_and_significance_fraction",
        paper_name="FigS07.pdf",
        source_relpath="figures/Composite_age_uncertainty_core_sensitivity/fig09_age_offsets_and_significance_fraction.pdf",
        tex_file="SI.tex",
        figure_label="fig:age_uncertainty_sensitivity",
    ),
    PaperFigure(
        stem="fig02_barker_predictive_likelihood_tests",
        paper_name="FigS08.pdf",
        source_relpath="figures/Barker2011_do_predictive_information/fig02_barker_predictive_likelihood_tests.pdf",
        tex_file="SI.tex",
        figure_label="fig:barker_predictive_sensitivity",
    ),
    PaperFigure(
        stem="fig03_barker_inputs_and_fitted_rates",
        paper_name="FigS09.pdf",
        source_relpath="figures/Barker2011_do_predictive_information/fig03_barker_inputs_and_fitted_rates.pdf",
        tex_file="SI.tex",
        figure_label="fig:barker_inputs_rates",
    ),
)

PAPER_FIGURE_STEMS = {entry.stem for entry in PAPER_FIGURES}
PAPER_FIGURE_BY_STEM = {entry.stem: entry for entry in PAPER_FIGURES}
PAPER_FIGURE_NAMES = {entry.paper_name for entry in PAPER_FIGURES}


def paper_figures_dir(project_root: Path = PROJECT_ROOT) -> Path:
    out_dir = Path(project_root) / "monsoon_paper" / "figures"
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir


def paper_path_for_stem(project_root: Path, stem: str) -> Path | None:
    entry = PAPER_FIGURE_BY_STEM.get(stem)
    if entry is None:
        return None
    return paper_figures_dir(project_root) / entry.paper_name


def save_paper_pdf(fig, project_root: Path, stem: str) -> Path | None:
    out_path = paper_path_for_stem(project_root, stem)
    if out_path is None:
        return None
    fig.savefig(out_path, bbox_inches="tight")
    return out_path


def copy_pdf_to_paper(project_root: Path, source_pdf: Path, stem: str | None = None) -> Path | None:
    out_stem = stem if stem is not None else Path(source_pdf).stem
    out_path = paper_path_for_stem(project_root, out_stem)
    if out_path is None:
        return None
    shutil.copyfile(source_pdf, out_path)
    return out_path


def export_all(project_root: Path = PROJECT_ROOT, *, clean: bool = True) -> list[Path]:
    out_dir = paper_figures_dir(project_root)
    written: list[Path] = []
    for entry in PAPER_FIGURES:
        source = project_root / entry.source_relpath
        if not source.exists():
            raise FileNotFoundError(f"Missing source for {entry.paper_name}: {source}")
        target = out_dir / entry.paper_name
        shutil.copyfile(source, target)
        written.append(target)
    if clean:
        for pdf in out_dir.glob("*.pdf"):
            if pdf.name not in PAPER_FIGURE_NAMES:
                pdf.unlink()
    return written


def main() -> None:
    written = export_all(PROJECT_ROOT, clean=True)
    print("Exported paper figures:")
    for path in written:
        print(f"  {path.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    main()
