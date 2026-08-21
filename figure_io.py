"""
figure_io.py

Single entry point for figure saving across the project. Always emits PNG
(for slides) + PDF (for LaTeX) at high DPI, matching the user's global
preference for figures (400 DPI).
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Sequence, Union

import numpy as np
import matplotlib.style as mpl_style  # registers mpl.style submodule
from matplotlib.axes import Axes
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure, FigureBase, SubFigure
from matplotlib.patches import Rectangle
from matplotlib.text import Text
from matplotlib.transforms import Bbox

# Apply the project's figure style at module import. Anything that imports
# SaveFigure picks up these defaults automatically (axes titlesize, label
# sizes, tick label sizes, top/right spines off, pdf.fonttype=42 for
# Illustrator round-tripping). See figure_style.mplstyle at repo root.
_STYLE_PATH = Path(__file__).resolve().parent / "figure_style.mplstyle"
if _STYLE_PATH.exists():
    mpl_style.use(str(_STYLE_PATH))


# ───────────────────────────────────────────────────────────────────────────
# Shared plot-tuning constants
# ───────────────────────────────────────────────────────────────────────────
# Structurally identical heatmap figures pull from the same dict here, then
# each plot script spreads it into its local FIG_PARAMS and overrides only
# what's unique (figsize, cmap, subplot/suptitle/etc.). Edit one number
# below and every figure in the family inherits it.
#
# Sizes were tuned for square 11–12-metric label sets at ~7-inch panel width:
# 14 pt ticks rotated 40° leave ~1 char of margin between adjacent labels;
# 11 pt cell annotations fit a 2-decimal value at this density without
# clipping; 16 pt colorbar label is legible at slide-resolution downscaling
# while still smaller than the variant suptitle (typically 18–20 pt).

METRIC_METRIC_HEATMAP_PARAMS = {
    "tick_fontsize": 21,   # +50% (measure-corr / beh-corr heatmap tick labels)
    "x_tick_rotation": 40,
    "y_tick_rotation": 40,
    "annot_fontsize": 11,
    "cbar_label_fontsize": 18,
    # cbar tick fontsize is inherited from mplstyle ytick.labelsize (16).
    "cmap_bad_color": "lightgrey",
    # FDR level for the bootstrap reverse-percentile significance outlines.
    "sig_alpha": 0.05,
}

# Region × region (86×86) heatmaps in a multi-metric grid. Per-cell region
# labels are not drawn by default (set show_region_names=True to enable);
# the group ("frontal", "temporal", …) + hemisphere (L/R) bars carry the
# regional context. Sizes match the 28-inch panel-grid output.
REGION_REGION_HEATMAP_PARAMS = {
    "region_label_fontsize": 6,
    "group_label_fontsize": 20,
    "group_label_rotation": 45,    # fallback for both axes
    "group_label_rotation_x": 90,  # x (bottom) vertical
    "group_label_rotation_y": 0,   # y (left) horizontal (no overlap, easiest read)
    "hemisphere_label_fontsize": 24,
    "hemisphere_lw": 2,
    # Region-division tick marks (at group boundaries) + optional dividing lines.
    "division_tick_length": 5.0,
    "division_tick_width": 1.0,
    "divider_lw": 0.6,
    "subplot_title_fontsize": 26,
    "suptitle_fontsize": 32,
    "cbar_label_fontsize": 24,
    "cbar_tick_fontsize": 27,
    "cmap_bad_color": "lightgrey",
}


# ───────────────────────────────────────────────────────────────────────────
# Display strings — metric short-names → camera-ready labels, behavior keys
# → camera-ready titles. Single source consumed across every plot script
# (replaces 17 duplicate CORRECT_LABELS dicts + per-script BEH_CONFIG titles).
# Pure presentation; data parameters live in config.yaml + project_paths.
# ───────────────────────────────────────────────────────────────────────────
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
METRIC_COLORS: dict[str, str] = {
    "coh":   "#1f77b4",  # blue
    "plv":   "#ff7f0e",  # orange
    "ppc":   "#2ca02c",  # green
    "ciplv": "#d62728",  # red
    "pli":   "#9467bd",  # purple
    "wpli":  "#8c564b",  # brown
    "aec":   "#e377c2",  # pink
    "aec_c": "#bcbd22",  # olive
    "pac":   "#17becf",  # cyan
    "dpli":  "#7f7f7f",  # grey (exploratory)
    "gc":    "#aec7e8",  # light blue (excluded from thesis figures)
    "gc_tr": "#ffbb78",  # light orange (excluded from thesis figures)
}


def metric_color(metric: str, default: str = "#333333") -> str:
    """Canonical colour for an FC measure (see `METRIC_COLORS`)."""
    return METRIC_COLORS.get(metric, default)


def ticklabel_clearance_axesfrac(
    ax: Axes, axis: str, *, renderer: Any = None,
) -> tuple[float, float]:
    """Where an `axis`'s tick labels end, in axes-fraction perpendicular coords.

    Measures the tick labels' rendered bounding box (the way matplotlib's own
    axis-label placement does) and returns ``(edge, pt)`` where `edge` is the
    axes-fraction coordinate perpendicular to `axis` of the labels' outer edge
    (``≤ 0``: for ``axis="y"`` the x just left of the labels, for ``axis="x"``
    the y just below them) and `pt` is the size of one typographic point in
    that same axes-fraction unit. Together they let a caller place decorations a
    fixed number of *points* beyond the labels regardless of axes size. Needs a
    drawn figure (a renderer); pass `renderer` to reuse one.

    Returns ``(0.0, pt)`` if the axis has no non-empty tick labels.
    """
    fig = ax.get_figure()
    assert fig is not None
    if renderer is None:
        fig.canvas.draw()  # pyright: ignore[reportUnknownMemberType]
        renderer = fig.canvas.get_renderer()  # pyright: ignore[reportAttributeAccessIssue, reportUnknownMemberType, reportUnknownVariableType]
    bb = ax.get_window_extent(renderer)  # pyright: ignore[reportUnknownArgumentType]
    perp_px = bb.width if axis == "y" else bb.height
    pt = (fig.dpi / 72.0) / perp_px  # one point, in axes-fraction along perp
    labels = (ax.get_yticklabels() if axis == "y" else ax.get_xticklabels())
    inv = ax.transAxes.inverted()
    boxes = [t.get_window_extent(renderer) for t in labels  # pyright: ignore[reportUnknownArgumentType]
             if t.get_text().strip()]
    if not boxes:
        return 0.0, pt
    union = Bbox.union(boxes)
    if axis == "y":
        return float(inv.transform((union.x0, union.y0))[0]), pt  # pyright: ignore[reportUnknownMemberType]
    return float(inv.transform((union.x0, union.y0))[1]), pt  # pyright: ignore[reportUnknownMemberType]


def draw_metric_band_bars(
    ax: Axes,
    bands: "Sequence[tuple[str, int, int]]",
    *,
    fontsize: float = 15,
    linewidth: float = 4.0,
    color: str = "black",
    pad_pts: float = 8.0,
    gap_pts: float | None = None,
    label_pad_pts: float = 3.0,
    span_pad: float = 0.12,
    axis: str = "x",
    stack: bool = True,
    renderer: Any = None,
) -> float:
    """Draw labelled bracket bars just beyond a heatmap axis's tick labels.

    Each band is ``(label, start, end)`` with `end` exclusive (in heatmap index
    units along `axis`). The bars are auto-positioned a fixed `pad_pts` *past
    the tick labels* — measured from the rendered label bounding box, the same
    way matplotlib offsets an axis label by `labelpad` — so they never overlap
    the labels and never sit too far out, independent of axes size. Overlapping
    bands (e.g. a measure that is both theta and gamma) stack outward `gap_pts`
    apart, each with its label just beyond it.

    `axis="x"` draws horizontal bars under the columns (labels below); `axis="y"`
    draws vertical bars left of the rows (labels rotated 90°, to the left).

    `stack=True` (default) puts each band on its own row, `gap_pts` apart — for
    overlapping spans. `stack=False` puts every band on the *same* row, side by
    side — for disjoint spans that read as one bracket (e.g. L / R hemispheres).

    Requires a drawn figure (uses the renderer). **Call after the layout is
    final** (e.g. after `fig.canvas.draw()` + `set_layout_engine("none")`), else
    later reflow shifts the tick labels out from under the bars.

    Returns the bars' outward extent in *points* beyond the tick labels — set
    ``ax.{x,y}axis.labelpad`` to this (plus a small margin) to push the axis
    label past the bars (matplotlib measures `labelpad` from the tick labels).
    """
    edge, pt = ticklabel_clearance_axesfrac(ax, axis, renderer=renderer)
    if gap_pts is None:
        gap_pts = fontsize * 1.5  # room for a bar + its label between stack rows
    trans = ax.get_yaxis_transform() if axis == "y" else ax.get_xaxis_transform()
    base = edge - pad_pts * pt  # first bar, just past the labels (more negative)
    for i, (label, start, end) in enumerate(bands):
        p = base - (i if stack else 0) * gap_pts * pt
        lab_p = p - label_pad_pts * pt
        if axis == "y":
            ax.plot([p, p], [start + span_pad, end - span_pad], transform=trans,  # pyright: ignore[reportUnknownMemberType]
                    color=color, lw=linewidth, clip_on=False, solid_capstyle="butt")
            ax.text(lab_p, (start + end) / 2.0, label, transform=trans,  # pyright: ignore[reportUnknownMemberType]
                    ha="right", va="center", rotation=90, fontsize=fontsize,
                    color=color, clip_on=False)
        else:
            ax.plot([start + span_pad, end - span_pad], [p, p], transform=trans,  # pyright: ignore[reportUnknownMemberType]
                    color=color, lw=linewidth, clip_on=False, solid_capstyle="butt")
            ax.text((start + end) / 2.0, lab_p, label, transform=trans,  # pyright: ignore[reportUnknownMemberType]
                    ha="center", va="top", fontsize=fontsize, color=color,
                    clip_on=False)
    # Outward extent in points BEYOND the tick labels (so it maps straight to
    # labelpad): pad to the first bar + the stack + the outermost band's label.
    n = len(bands)
    return pad_pts + (n - 1) * gap_pts + label_pad_pts + fontsize * 1.3

# `title` is the short behaviour name; `contrast_title` is the two-line
# "<behaviour>\n<contrast>" label used as the panel/figure title wherever a
# figure shows one contrast. Single source — every figure pulls its contrast
# title from here (via `contrast_title()`), so wording is edited in one place.
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


def contrast_title(beh: str) -> str:
    """Canonical two-line contrast title for a behaviour (see BEHAVIOR_DISPLAY).
    Falls back to the short title, then the raw key."""
    d = BEHAVIOR_DISPLAY.get(beh, {})
    return d.get("contrast_title") or d.get("title") or beh


# Canonical behaviour/contrast -> colour map, for figures that overlay multiple
# contrasts in one axes (distinct from the metric palette; this colours by
# behaviour). Dark2-style, chosen to be distinguishable from METRIC_COLORS.
CONTRAST_COLORS: dict[str, str] = {
    "en":      "#1b9e77",  # teal
    "rm":      "#d95f02",  # orange
    "word_on": "#7570b3",  # purple
    "voc":     "#e7298a",  # magenta
    "en_all":  "#66a61e",  # green
    "rm_all":  "#e6ab02",  # gold
}


def contrast_color(beh: str, default: str = "#333333") -> str:
    """Canonical colour for a behaviour/contrast (see CONTRAST_COLORS)."""
    return CONTRAST_COLORS.get(beh, default)


# ───────────────────────────────────────────────────────────────────────────
# Multi-panel figure layout
# ───────────────────────────────────────────────────────────────────────────
# Helpers for composing several single-analysis figures into one labelled
# multi-panel figure. Each analysis plotting function takes an optional
# `ax` (single-axes plots) or host `SubFigure` (grid plots) argument; the
# orchestrator builds the panel layout, then drops each analysis into its
# slot and tags it with `panel_label`.

# Default styling for the bold "A." panel tags. Large enough to read at a
# glance against 25–32 pt subplot/sup-titles.
PANEL_LABEL_FONTSIZE = 50
# Offset, in typographic points, of the tag from the panel's top-left corner
# (Axes targets). Sits just ABOVE the axes top-left, left-aligned to the axes
# left edge, so it extends up-and-right into the title margin and never
# overhangs the panel to its left (which would collide with a tightly-packed
# left neighbour).
PANEL_LABEL_OFFSET_PTS = (0.0, 4.0)


def panel_label(
    target: Axes | FigureBase,
    letter: str,
    *,
    fontsize: float = PANEL_LABEL_FONTSIZE,
    fontweight: str = "bold",
    offset_pts: tuple[float, float] = PANEL_LABEL_OFFSET_PTS,
    in_layout: bool = True,
    **text_kwargs: Any,
) -> Text:
    """Place a bold panel tag (e.g. ``"A."``) at a panel's upper-left corner.

    For an ``Axes`` (the usual panel, including the top-left subplot of a grid)
    the tag is annotated just above-left of the axes — in the title / y-label
    margin that already exists — so it adds no extra whitespace. ``in_layout``
    stays True so the tag is kept inside ``savefig(bbox_inches="tight")``'s
    crop. Design the panel title and y-label to not overlap it; tune inter-
    panel spacing via `panel_subfigures`' wspace/hspace.

    For a ``SubFigure``/``Figure`` target the tag is placed at the interior
    top-left corner instead (use this only when there is no single panel axes
    to anchor to).

    Parameters
    ----------
    target : matplotlib.axes.Axes | matplotlib.figure.FigureBase
        The panel to tag.
    letter : str
        Panel letter; a trailing period is appended if absent (``"A"`` → ``"A."``).
    fontsize, fontweight : float, str
        Text styling for the tag.
    offset_pts : tuple[float, float]
        ``(dx, dy)`` offset in points from the axes' top-left corner (Axes only).
    in_layout : bool
        Whether the tag participates in layout / tight-bbox. Default True so it
        is not cropped out by ``bbox_inches="tight"``.

    Returns
    -------
    matplotlib.text.Text
        The created text artist.
    """
    label = letter if letter.endswith(".") else f"{letter}."
    if isinstance(target, Axes):
        art: Text = target.annotate(  # pyright: ignore[reportUnknownMemberType]
            label, xy=(0.0, 1.0), xycoords="axes fraction",
            xytext=offset_pts, textcoords="offset points",
            ha="left", va="bottom", annotation_clip=False,
            fontsize=fontsize, fontweight=fontweight, **text_kwargs,
        )
    else:
        art = target.text(  # pyright: ignore[reportUnknownMemberType]
            0.0, 1.0, label, ha="left", va="top", clip_on=False,
            fontsize=fontsize, fontweight=fontweight, **text_kwargs,
        )
    art.set_in_layout(in_layout)
    return art


SIG_MARKER_HEIGHT_FRAC = 0.90  # default: near the top of the axes


def place_significance_markers(
    ax: Axes,
    x: Sequence[float],
    significant: Sequence[bool],
    *,
    height_frac: float | Sequence[float] = SIG_MARKER_HEIGHT_FRAC,
    marker: str = "*",
    fontsize: float = 14,
    color: str = "black",
) -> None:
    """Place significance markers at a fixed fraction of the axes height.

    By convention a marker sits at `height_frac` of the axes height measured from
    the bottom (0 = bottom edge, 1 = top edge), so the default 0.90 is near the
    top and every marker is horizontally aligned regardless of the bar / point it
    annotates (rather than tracking individual heights). `height_frac` may be a
    scalar (applied to all) or a per-marker sequence (individual overrides). `x`
    is in data coordinates; the y position uses the x-axis blended transform
    (data x, axes-fraction y), so it is unaffected by later y-limit changes.
    """
    trans = ax.get_xaxis_transform()
    xs = list(x)
    if isinstance(height_frac, (int, float)):
        fracs = [float(height_frac)] * len(xs)
    else:
        fracs = [float(f) for f in height_frac]
    for xi, sig, fr in zip(xs, significant, fracs):
        if sig:
            ax.text(xi, fr, marker, transform=trans, ha="center", va="center",
                    fontsize=fontsize, color=color)


def drop_axis_marks(ax: Axes, *, x: bool = False, y: bool = False) -> Axes:
    """Blank an axis's label + tick labels on a composite panel, keeping the
    tick marks themselves.

    Use from multi-panel layout code (NOT inside the analysis plotters) to
    remove redundant marks: drop the x-axis (``x=True``) on the upper of two
    vertically-adjacent panels that share an x-axis, and the y-axis
    (``y=True``) on the right of two horizontally-adjacent panels that share a
    y-axis. The tick *marks* are left in place so the shared grid stays
    visible for cross-panel comparison; only the axis title and tick-label
    text are blanked. The axis limits/scale are untouched, so panels align.
    """
    if x:
        ax.set_xlabel("")  # pyright: ignore[reportUnknownMemberType]
        ax.tick_params(axis="x", which="both",  # pyright: ignore[reportUnknownMemberType]
                       labelbottom=False, labeltop=False)
    if y:
        ax.set_ylabel("")  # pyright: ignore[reportUnknownMemberType]
        ax.tick_params(axis="y", which="both",  # pyright: ignore[reportUnknownMemberType]
                       labelleft=False, labelright=False)
    return ax


def check_data_parameter(
    data_parameter: dict[str, Any], required: set[str], *, fn: str = "figure",
    optional: set[str] | None = None,
) -> dict[str, Any]:
    """Validate a figure function's ``data_parameter`` dict.

    Uniform dataset-selection API across all single-figure functions: each
    function declares the exact keys it consumes (e.g. ``{"beh", "band",
    "cond"}`` plus family-specific ``N`` / ``estimator`` / ``region`` /
    ``region_pair`` / ``behavior_pair``). Raises ``ValueError`` if any required
    key is missing OR any unexpected key is present, so a mis-routed parameter
    fails loudly rather than being silently ignored. Keys in ``optional`` are
    permitted but not required (e.g. a pooled case-study ``case`` selector).
    Returns the dict for convenient chaining.
    """
    keys = set(data_parameter)
    missing = required - keys
    extra = keys - required - (optional or set())
    if missing or extra:
        raise ValueError(
            f"{fn}: data_parameter key mismatch "
            f"(missing={sorted(missing)}, unexpected={sorted(extra)}); "
            f"required exactly {sorted(required)}"
        )
    return data_parameter


def drop_legend(ax: Axes) -> Axes:
    """Remove the legend from `ax` if present (for panels that share one
    legend across a composite). Uses matplotlib's ``Legend.remove``."""
    leg = ax.get_legend()
    if leg is not None:
        leg.remove()
    return ax


def drop_colorbar(ax: Axes, *, keep_space: bool = False) -> Axes:
    """Remove the colorbar(s) attached to mappables on `ax` (e.g. the
    redundant per-panel colorbar when heatmap panels share one).

    By default uses matplotlib's ``Colorbar.remove``, which *restores* the
    host axes to its pre-colorbar (larger) size — so a heatmap whose
    colorbar is dropped grows and shifts relative to sibling panels that
    keep theirs. Pass ``keep_space=True`` to instead hide the colorbar
    (``set_visible(False)``) while leaving its reserved space intact, so
    the host axes keeps the same size/position as panels that retain a
    visible colorbar (preserving alignment across a panel grid)."""
    for artist in [*ax.collections, *ax.images]:
        cb = getattr(artist, "colorbar", None)
        if cb is not None:
            if keep_space:
                cb.ax.set_visible(False)
            else:
                cb.remove()
    return ax


def highlight_significant_clusters(
    significance_mask: "np.ndarray[Any, Any]", *,
    ax: Axes | None = None,
    border_color: str = "black",
    border_linewidth: float = 2.5,
    offset: float = 0.5,
) -> Axes:
    """Outline clusters of significant heatmap cells, merging neighbours.

    `significance_mask` is a boolean ``(rows, cols)`` array (cell ``[i, j]``
    significant when True). A border segment is drawn on each cell edge that
    faces a non-significant (or out-of-bounds) neighbour, so adjacent
    significant cells share no internal line — a contiguous block reads as one
    merged outline.

    `offset` shifts the edge coordinates to match the heatmap's cell-corner
    convention: ``0.5`` for seaborn ``heatmap``/``pcolormesh`` (cell ``[i, j]``
    spans ``[j, j+1] x [i, i+1]``); pass ``0.0`` for ``imshow`` (cell centred
    on integer coordinates).
    """
    if ax is None:
        import matplotlib.pyplot as plt
        ax = plt.gca()
    mask = np.asarray(significance_mask, dtype=bool)
    rows, cols = mask.shape
    padded = np.pad(mask, pad_width=1, constant_values=False)
    lw, ec = border_linewidth, border_color
    for i in range(rows):
        for j in range(cols):
            if not mask[i, j]:
                continue
            x0, y0 = j - 0.5 + offset, i - 0.5 + offset
            if not padded[i, j + 1]:        # top edge
                ax.add_patch(Rectangle((x0, y0), 1, 0, linewidth=lw,
                                       edgecolor=ec, facecolor="none"))
            if not padded[i + 2, j + 1]:    # bottom edge
                ax.add_patch(Rectangle((x0, y0 + 1), 1, 0, linewidth=lw,
                                       edgecolor=ec, facecolor="none"))
            if not padded[i + 1, j]:        # left edge
                ax.add_patch(Rectangle((x0, y0), 0, 1, linewidth=lw,
                                       edgecolor=ec, facecolor="none"))
            if not padded[i + 1, j + 2]:    # right edge
                ax.add_patch(Rectangle((x0 + 1, y0), 0, 1, linewidth=lw,
                                       edgecolor=ec, facecolor="none"))
    return ax


def set_axis_range(
    axes: Axes | Sequence[Axes],
    *,
    xlim: tuple[float, float] | str | None = None,
    ylim: tuple[float, float] | str | None = None,
) -> list[Axes]:
    """Apply common x/y limits across composite panels so they're comparable.

    Pass an explicit ``(lo, hi)`` for an axis, or ``"auto"`` to use the union
    of the panels' current limits on that axis. ``None`` leaves it untouched.
    """
    ax_list = [axes] if isinstance(axes, Axes) else list(axes)
    if xlim == "auto":
        xlim = (min(a.get_xlim()[0] for a in ax_list),
                max(a.get_xlim()[1] for a in ax_list))
    if ylim == "auto":
        ylim = (min(a.get_ylim()[0] for a in ax_list),
                max(a.get_ylim()[1] for a in ax_list))
    for a in ax_list:
        if isinstance(xlim, tuple):
            a.set_xlim(xlim)
        if isinstance(ylim, tuple):
            a.set_ylim(ylim)
    return ax_list


def set_color_range(
    panels: Axes | Sequence[Axes],
    *,
    vmin: float | str | None = None,
    vmax: float | str | None = None,
) -> list[Any]:
    """Unify the color scale across heatmap/image panels (one shared scale).

    Collects the mappable on each Axes (seaborn/imshow), then applies common
    color limits. ``vmin``/``vmax`` may be explicit numbers or ``"auto"`` to
    use the union of the mappables' data ranges. Pair with a single shared
    colorbar (see `drop_colorbar` to remove redundant per-panel bars).
    """
    ax_list = [panels] if isinstance(panels, Axes) else list(panels)
    maps: list[Any] = []
    for a in ax_list:
        for art in [*a.collections, *a.images]:
            if hasattr(art, "set_clim"):
                maps.append(art)
    if vmin == "auto" or vmax == "auto":
        arrs = [np.asarray(m.get_array()) for m in maps if m.get_array() is not None]
        if arrs:
            flat = np.concatenate([np.asarray(x).ravel() for x in arrs])
            if vmin == "auto":
                vmin = float(np.nanmin(flat))  # pyright: ignore[reportUnknownMemberType, reportUnknownArgumentType]
            if vmax == "auto":
                vmax = float(np.nanmax(flat))  # pyright: ignore[reportUnknownMemberType, reportUnknownArgumentType]
    for m in maps:
        m.set_clim(
            vmin if isinstance(vmin, (int, float)) else None,
            vmax if isinstance(vmax, (int, float)) else None,
        )
    return maps


def panel_subfigures(
    nrows: int,
    ncols: int,
    *,
    figsize: tuple[float, float],
    width_ratios: Sequence[float] | None = None,
    height_ratios: Sequence[float] | None = None,
    label: bool = True,
    wspace: float | None = None,
    hspace: float | None = None,
) -> tuple[Figure, list[SubFigure]]:
    """Create a multi-panel figure as a grid of `SubFigure` hosts.

    A `SubFigure` is the universal panel slot: a single-axes analysis renders
    into ``subfig.subplots()`` (pass that Axes as the analysis's ``ax``); a
    grid-style analysis renders into the subfigure itself (pass ``subfig`` as
    its ``host``). `width_ratios`/`height_ratios` give the per-panel share of
    the figure's width/height, exactly like ``subplot_mosaic``/``GridSpec``.

    Returns the parent `Figure` and the subfigures flattened in row-major
    order. With ``label=True`` each panel is tagged ``"A."``, ``"B."``, … via
    `panel_label`.
    """
    fig = Figure(figsize=figsize)
    subfigs = fig.subfigures(  # pyright: ignore[reportUnknownMemberType, reportUnknownVariableType]
        nrows,
        ncols,
        width_ratios=width_ratios,
        height_ratios=height_ratios,
        wspace=wspace,
        hspace=hspace,
        squeeze=False,
    )
    flat: list[SubFigure] = [subfigs[r][c] for r in range(nrows) for c in range(ncols)]
    if label:
        label_panels(flat)
    return fig, flat


def label_panels(
    panels: Sequence[Axes | FigureBase],
    *,
    start: str = "A",
    **label_kwargs: Any,
) -> list[Text]:
    """Tag a sequence of panels with consecutive letters (``A.``, ``B.``, …).

    Each entry may be an `Axes` or a `SubFigure`/`Figure`; `panel_label`
    dispatches on type. Extra keyword arguments are forwarded to it.
    """
    first = ord(start.upper())
    return [
        panel_label(panel, chr(first + i), **label_kwargs)
        for i, panel in enumerate(panels)
    ]


def _subfig_content_bbox(
    sf: SubFigure, renderer: Any, inv: Any
) -> Bbox:
    """Tight bbox (figure fraction) of a subfigure's axes, including the
    panel-letter tag (it sits above the axes top-left, so it only inflates the
    top margin — exactly the space that must be reserved so a tag never
    overlaps the panel above).

    Axes hidden via ``set_visible(False)`` but kept for spacing (e.g. a
    ``drop_colorbar(keep_space=True)`` colorbar) are reserved by their box
    extent, so a panel with a hidden colorbar reserves the same width as a
    sibling whose colorbar is still drawn — keeping a square-heatmap grid
    aligned."""
    boxes: list[Bbox] = []
    for a in sf.axes:
        bb = a.get_tightbbox(renderer) if a.get_visible() \
            else a.get_window_extent(renderer)
        if bb is not None:
            boxes.append(bb.transformed(inv))
    return Bbox.union(boxes) if boxes else sf.bbox.transformed(inv)


def pack_panels(fig: Figure, *, gap_in: float = 0.5) -> Figure:
    """Pack a `panel_subfigures` grid so adjacent panels are separated by a
    fixed absolute gap (``gap_in`` inches, ~one panel-letter width) instead of
    matplotlib's default per-subfigure margins (which leave 10–25 % of the
    figure empty between panels).

    Measurement-driven and general: after a draw, each panel's per-side
    decoration extent (axis labels, ticks, attached colorbar) is measured from
    its content tight bbox; the axes are then repositioned to reserve exactly
    that decoration on outer sides and ``decoration + gap_in/2`` on sides that
    face a neighbour, so the *content* gap between neighbours equals ``gap_in``.
    Aspect-locked (square) heatmaps are squared and anchored to their inner
    side so the square fills its cell rather than centring with slack. Every
    axes in a subfigure (e.g. a heatmap + its colorbar) is moved by the same
    transform so attachments stay put. Panel letters move with their axes; any
    outer slack is removed by the ``bbox_inches="tight"`` crop on save.

    No-op for figures without a multi-panel subfigure grid.
    """
    subs: list[SubFigure] = list(getattr(fig, "subfigs", []) or [])
    if len(subs) < 2:
        return fig
    inv = fig.transFigure.inverted()
    FigureCanvasAgg(fig).draw()
    renderer = fig.canvas.get_renderer()  # pyright: ignore[reportAttributeAccessIssue, reportUnknownMemberType, reportUnknownVariableType]
    fw, fh = (float(v) for v in fig.get_size_inches())  # pyright: ignore[reportUnknownMemberType, reportUnknownArgumentType]
    gx, gy = gap_in / fw, gap_in / fh

    sfbbs = [sf.bbox.transformed(inv) for sf in subs]
    xs = sorted({round(b.x0, 3) for b in sfbbs})
    ys = sorted({round(b.y0, 3) for b in sfbbs})  # bottom-up
    ncols, nrows = len(xs), len(ys)

    # Pass 1: per-panel target box reserving its decorations + half-gap on
    # neighbour-facing sides (content gap between neighbours == gap_in).
    plans: list[dict[str, Any]] = []
    for sf, sfbb in zip(subs, sfbbs):
        if not sf.axes:
            continue
        prim = sf.axes[0]
        old = prim.get_position()
        ocx0, ocx1 = sfbb.x0 + old.x0 * sfbb.width, sfbb.x0 + old.x1 * sfbb.width
        ocy0, ocy1 = sfbb.y0 + old.y0 * sfbb.height, sfbb.y0 + old.y1 * sfbb.height
        if ocx1 <= ocx0 or ocy1 <= ocy0:
            continue
        col = xs.index(round(sfbb.x0, 3))
        row = nrows - 1 - ys.index(round(sfbb.y0, 3))  # 0 = top row
        tb = _subfig_content_bbox(sf, renderer, inv)
        left = (ocx0 - tb.x0) + (0.0 if col == 0 else gx / 2)
        right = (tb.x1 - ocx1) + (0.0 if col == ncols - 1 else gx / 2)
        bot = (ocy0 - tb.y0) + (0.0 if row == nrows - 1 else gy / 2)
        top = (tb.y1 - ocy1) + (0.0 if row == 0 else gy / 2)
        plans.append({
            "sf": sf, "sfbb": sfbb, "prim": prim, "col": col, "row": row,
            "oc": (ocx0, ocy0, ocx1, ocy1), "square": prim.get_aspect() == 1.0,
            "nx0": sfbb.x0 + left, "nx1": sfbb.x1 - right,
            "ny0": sfbb.y0 + bot, "ny1": sfbb.y1 - top,
        })

    # Square panels (aspect-locked heatmaps) must share one size and a common
    # per-column left / per-row top, else asymmetric decorations (e.g. a
    # colorbar on one panel, x-labels only on the bottom row) give each panel a
    # different square and the grid stops aligning. Size to the smallest box.
    sq = [p for p in plans if p["square"]]
    if sq:
        s_in = min(min((p["nx1"] - p["nx0"]) * fw, (p["ny1"] - p["ny0"]) * fh)
                   for p in sq)
        sw, sh = s_in / fw, s_in / fh
        col_left: dict[int, float] = {}
        row_top: dict[int, float] = {}
        for p in sq:
            col_left[p["col"]] = max(col_left.get(p["col"], p["nx0"]), p["nx0"])
            row_top[p["row"]] = min(row_top.get(p["row"], p["ny1"]), p["ny1"])
        for p in sq:
            p["nx0"] = col_left[p["col"]]
            p["nx1"] = p["nx0"] + sw
            p["ny1"] = row_top[p["row"]]
            p["ny0"] = p["ny1"] - sh

    # Pass 2: move every axes in each subfigure by the transform that maps its
    # primary axes' old box to the planned box (so attached colorbars follow).
    for p in plans:
        sf, sfbb = p["sf"], p["sfbb"]
        ocx0, ocy0, ocx1, ocy1 = p["oc"]
        nx0, ny0, nx1, ny1 = p["nx0"], p["ny0"], p["nx1"], p["ny1"]
        sx, sy = (nx1 - nx0) / (ocx1 - ocx0), (ny1 - ny0) / (ocy1 - ocy0)
        for a in sf.axes:
            ap = a.get_position()
            apx0, apy0 = sfbb.x0 + ap.x0 * sfbb.width, sfbb.y0 + ap.y0 * sfbb.height
            apx1, apy1 = sfbb.x0 + ap.x1 * sfbb.width, sfbb.y0 + ap.y1 * sfbb.height
            fx0, fy0 = nx0 + (apx0 - ocx0) * sx, ny0 + (apy0 - ocy0) * sy
            fx1, fy1 = nx0 + (apx1 - ocx0) * sx, ny0 + (apy1 - ocy0) * sy
            a.set_position(((fx0 - sfbb.x0) / sfbb.width,
                            (fy0 - sfbb.y0) / sfbb.height,
                            (fx1 - fx0) / sfbb.width,
                            (fy1 - fy0) / sfbb.height))
    return fig


# ───────────────────────────────────────────────────────────────────────────
# In-memory pixel comparison (figure regression checks)
# ───────────────────────────────────────────────────────────────────────────
# Render a figure straight to an RGBA pixel array — no file, no metadata. This
# sidesteps the byte-hash pitfalls entirely: PNGs carry an mpl-version
# `Software` chunk and PDFs carry a wall-clock `/CreationDate` + `/ModDate`, so
# two *identical* figures hash differently across runs/versions. Comparing
# decoded pixels is inherently metadata-agnostic. Pair with
# `scripts/compare_figures.py` (which does the same for on-disk PNGs).


def figure_to_rgba(
    fig: Figure, dpi: float | None = None
) -> np.ndarray[Any, np.dtype[np.uint8]]:
    """Render `fig` to an ``(H, W, 4)`` uint8 RGBA array via the Agg backend.

    No file is written, so no metadata is involved. `dpi` defaults to the
    figure's own dpi; pass an explicit value to compare two figures at a
    common resolution.
    """
    canvas = FigureCanvasAgg(fig)
    canvas.draw()
    if dpi is not None:
        fig.set_dpi(dpi)
        canvas.draw()
    return np.asarray(canvas.buffer_rgba(), dtype=np.uint8)


def figures_max_abs_diff(
    fig_a: Figure, fig_b: Figure, dpi: float | None = None
) -> float:
    """Max absolute per-pixel RGBA difference between two figures.

    Returns ``inf`` if the rendered arrays differ in shape (e.g. different
    figsize/dpi). ``0.0`` means pixel-identical.
    """
    a = figure_to_rgba(fig_a, dpi=dpi).astype(np.float64)
    b = figure_to_rgba(fig_b, dpi=dpi).astype(np.float64)
    if a.shape != b.shape:
        return float("inf")
    diff: np.float64 = np.abs(a - b).max()  # pyright: ignore[reportUnknownMemberType]
    return float(diff)


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
