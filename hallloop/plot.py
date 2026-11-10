"""Render the selected field sweep and projection diagnostics."""

from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure


UP_COLOR = "#1c5c9c"
DOWN_COLOR = "#aa4a21"
PAIRED_COLOR = "#087e73"
RAW_COLOR = "#929eae"
INK = "#182b3f"
SECONDARY_INK = "#40546b"
GRID = "#e1e8f1"


def _style_axis(ax, title, field_label):
    ax.set_title(title, loc="left", color=INK, fontsize=13, fontweight="bold", pad=10)
    ax.set_xlabel(field_label, color=SECONDARY_INK, parse_math=False)
    ax.tick_params(axis="both", colors=SECONDARY_INK, labelsize=9, length=3)
    ax.set_axisbelow(True)
    ax.grid(color=GRID, linewidth=0.8)
    ax.spines[["top", "right"]].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color("#acbbcb")


def _legend(ax, **kwargs):
    ax.legend(fontsize=8.5, labelcolor=SECONDARY_INK, framealpha=0.95,
              facecolor="white", edgecolor=GRID, **kwargs)


def plot(result, output_path, *, field_label, longitudinal_label, transverse_label, raw):
    """Save a PNG with raw signals, projected signals, and matched field coverage."""
    field, longitudinal, transverse = raw
    paired_field = result.field
    lo, hi = min(paired_field), max(paired_field)

    figure = Figure(figsize=(12, 8.1), facecolor="#f5f8fc", layout="constrained")
    FigureCanvasAgg(figure)
    figure.get_layout_engine().set(w_pad=0.16, h_pad=0.12, wspace=0.05, hspace=0.04)
    figure.suptitle("HallLoop diagnostics", color=INK, fontsize=22, fontweight="bold")
    (raw_ax, longitudinal_ax), (odd_ax, coverage_ax) = figure.subplots(2, 2)

    _style_axis(raw_ax, "Raw transverse signal", field_label)
    raw_ax.set_ylabel(transverse_label, color=SECONDARY_INK, parse_math=False)
    raw_ax.plot(field, transverse, color=RAW_COLOR, linewidth=3, label="Full acquisition")
    raw_ax.plot(paired_field, result.transverse_up, color=UP_COLOR, label="Up on paired grid")
    raw_ax.plot(paired_field, result.transverse_down, color=DOWN_COLOR, linestyle="--",
                label="Down on paired grid")
    _legend(raw_ax, loc="best")

    _style_axis(longitudinal_ax, "Longitudinal even projection", field_label)
    longitudinal_ax.set_ylabel(longitudinal_label, color=SECONDARY_INK, parse_math=False)
    longitudinal_ax.plot(field, longitudinal, color=RAW_COLOR, linewidth=3, label="Full acquisition")
    longitudinal_ax.plot(paired_field, result.longitudinal_even_up, color=UP_COLOR,
                         label="Paired up sweep")
    longitudinal_ax.plot(paired_field, result.longitudinal_even_down, color=DOWN_COLOR,
                         linestyle="--", label="Paired down sweep")
    _legend(longitudinal_ax, loc="best")

    _style_axis(odd_ax, "Transverse odd projection", field_label)
    odd_ax.set_ylabel(transverse_label, color=SECONDARY_INK, parse_math=False)
    odd_ax.axhline(0, color=RAW_COLOR, linewidth=0.8)
    odd_ax.plot(paired_field, result.transverse_odd_up, color=PAIRED_COLOR, linewidth=2.7,
                label="Opposite-branch pairing")
    odd_ax.plot(paired_field, result.transverse_odd_down, color=PAIRED_COLOR, linewidth=2.7)
    odd_ax.plot(paired_field, result.naive_odd_up, color=UP_COLOR, linestyle="--",
                label="Same-branch up")
    odd_ax.plot(paired_field, result.naive_odd_down, color=DOWN_COLOR, linestyle="--",
                label="Same-branch down")
    _legend(odd_ax, loc="best")

    _style_axis(coverage_ax, "Selected sweep coverage", field_label)
    coverage_ax.axvspan(lo, hi, color=PAIRED_COLOR, alpha=0.1, linewidth=0)
    coverage_ax.axvline(0, color=RAW_COLOR, linewidth=0.8)
    coverage_ax.set_yticks([2, 1, 0], ["Up sweep", "Down sweep", "Paired span"])
    coverage_ax.set_ylim(-0.5, 2.5)
    coverage_ax.tick_params(axis="y", length=0)
    for branch in result.branches:
        start, end = branch["field_range"]
        if branch["direction"] == "up":
            y, color = 2, UP_COLOR
        else:
            y, color = 1, DOWN_COLOR
            start, end = end, start
        coverage_ax.annotate("", xy=(end, y), xytext=(start, y),
                             arrowprops={"arrowstyle": "-|>", "color": color, "lw": 2.5,
                                         "shrinkA": 0, "shrinkB": 0, "mutation_scale": 13})
    coverage_ax.hlines(0, lo, hi, color=PAIRED_COLOR, linewidth=3)
    coverage_ax.plot([lo, hi], [0, 0], "o", color=PAIRED_COLOR, markersize=5)

    field_min, field_max = min(field), max(field)
    margin = 0.035 * (field_max - field_min)
    for ax in (raw_ax, longitudinal_ax, odd_ax, coverage_ax):
        ax.set_xlim(field_min - margin, field_max + margin)

    figure.savefig(output_path, dpi=160, format="png")
