"""
Build Figure 2 by vertically stacking two existing PDF figure panels.

Output:
    figures/fig02/fig02.pdf
"""

from __future__ import annotations

from pathlib import Path

from pypdf import PdfReader, PdfWriter, Transformation
from pypdf._page import PageObject

from paper_figure_export import copy_pdf_to_paper


PROJECT_ROOT = Path(__file__).resolve().parent
TOP_PDF = (
    PROJECT_ROOT
    / "figures/rousseau2023_monsoon_randomness_lohmann/lohmann_ES_null_distributions.pdf"
)
BOTTOM_PDF = (
    PROJECT_ROOT
    / "figures/rousseau2023_monsoon_orbital_phase_rayleigh/fig03a_rayleigh_phase_polar_precession.pdf"
)
OUT_DIR = PROJECT_ROOT / "figures/fig02"
OUT_PDF = OUT_DIR / "fig02.pdf"

GAP_PT = 10.0


def first_page(path: Path):
    if not path.exists():
        raise FileNotFoundError(f"Missing input PDF: {path}")
    reader = PdfReader(str(path))
    if not reader.pages:
        raise ValueError(f"Input PDF has no pages: {path}")
    return reader.pages[0]


def page_size(page) -> tuple[float, float]:
    return float(page.mediabox.width), float(page.mediabox.height)


def add_centered_page(canvas, page, x_offset: float, y_offset: float, scale: float) -> None:
    transform = Transformation().scale(scale).translate(tx=x_offset, ty=y_offset)
    canvas.merge_transformed_page(page, transform, over=True)


def build_fig02() -> Path:
    top_page = first_page(TOP_PDF)
    bottom_page = first_page(BOTTOM_PDF)

    top_w, top_h = page_size(top_page)
    bottom_w, bottom_h = page_size(bottom_page)
    target_w = max(top_w, bottom_w)
    top_scale = target_w / top_w
    bottom_scale = target_w / bottom_w
    top_h_scaled = top_h * top_scale
    bottom_h_scaled = bottom_h * bottom_scale
    total_h = top_h_scaled + GAP_PT + bottom_h_scaled

    canvas = PageObject.create_blank_page(width=target_w, height=total_h)
    add_centered_page(
        canvas,
        bottom_page,
        x_offset=(target_w - bottom_w * bottom_scale) / 2.0,
        y_offset=0.0,
        scale=bottom_scale,
    )
    add_centered_page(
        canvas,
        top_page,
        x_offset=(target_w - top_w * top_scale) / 2.0,
        y_offset=bottom_h_scaled + GAP_PT,
        scale=top_scale,
    )

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    writer = PdfWriter()
    writer.add_page(canvas)
    with OUT_PDF.open("wb") as handle:
        writer.write(handle)
    copy_pdf_to_paper(PROJECT_ROOT, OUT_PDF, "fig02")
    return OUT_PDF


def main() -> None:
    output = build_fig02()
    print(f"Wrote {output.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    main()
