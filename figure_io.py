"""figure_io.py — figure output helper and shared display labels.

Reduced to what this project actually uses: `SaveFigure`, plus the two label
dicts other modules import (`METRIC_LABELS`, `BEHAVIOR_DISPLAY`). The module
previously carried ~20 panel-composition and styling helpers inherited from
fc_methods_comparison_cml (metric_color, pack_panels, panel_subfigures,
highlight_significant_clusters, ...). That pipeline is independent of this one
and none of them had a caller here, so they were removed rather than shipped as
834 lines of dead weight; they remain in that repo if ever needed.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Union

from matplotlib.figure import Figure


METRIC_LABELS: dict[str, str] = {
    "coh":   "Coh",
    "plv":   "PLV",
    "ppc":   "PPC",
    "ciplv": "ciPLV",
    "pli":   "PLI",
    "wpli":  "wPLI",
    "aec":   "AEC",
    "aec_c": "AEC-c",
    "pac":   "PAC",
    "dpli":  "dPLI",
    "gc":    "GC",
    "gc_tr": "GC-TR",
}

# Canonical metric -> colour map, used everywhere a figure colours by FC
# measure, so a measure keeps the same colour across all figures (instead of
# each figure sampling a colormap by position). Keyed by metric name; fixed
# regardless of how many metrics a given figure shows or their order.


BEHAVIOR_DISPLAY: dict[str, dict[str, str]] = {
    "en":      {"title": "Encoding",
                "contrast_title": "Encoding\nRecalled vs. Not Recalled"},
    "rm":      {"title": "Retrieval",
                "contrast_title": "Retrieval vs. Silent"},
    "word_on": {"title": "Word on Screen",
                "contrast_title": "Word Presentation\nvs. Pre-Word"},
    "voc":     {"title": "Vocalization",
                "contrast_title": "Vocalization\nvs. Pre-Vocalization"},
    "en_all":  {"title": "All Encoding", "contrast_title": "All Encoding"},
    "rm_all":  {"title": "All Retrieval", "contrast_title": "All Retrieval"},
}


def SaveFigure(
    fig: Figure,
    name: str,
    results_subdir: Union[str, Path],
    dpi: int = 400,
) -> tuple[Path, Path]:
    """Save `fig` as PNG + PDF at `dpi` into `results_subdir`.

    Parameters
    ----------
    fig : matplotlib.figure.Figure
        Figure to save.
    name : str
        Output basename (without extension).
    results_subdir : str | pathlib.Path
        Directory to write into. Created if missing.
    dpi : int
        DPI for both raster (PNG) and vector (PDF affects rasterized
        sub-elements only).

    Returns
    -------
    (png_path, pdf_path) : tuple[Path, Path]
        Absolute paths of the two emitted files.
    """
    out_dir = Path(results_subdir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Strip any extension the caller accidentally included.
    stem = os.path.splitext(name)[0]

    png_path = out_dir / f"{stem}.png"
    pdf_path = out_dir / f"{stem}.pdf"

    fig.savefig(png_path, dpi=dpi, bbox_inches="tight")  # pyright: ignore[reportUnknownMemberType]
    fig.savefig(pdf_path, dpi=dpi, bbox_inches="tight")  # pyright: ignore[reportUnknownMemberType]

    return png_path, pdf_path
