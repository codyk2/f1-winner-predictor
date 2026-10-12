"""Figures for the README and notebooks. One style, one palette.

Colours are a validated colour-blind-safe set (blue, orange) on a light
surface; green and red are reserved for hit and miss and always come with a
distinct marker shape, never colour alone. Every figure has a table twin in
the README.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.patches import PathPatch  # noqa: E402
from matplotlib.path import Path as MplPath  # noqa: E402
from matplotlib.ticker import MaxNLocator, MultipleLocator  # noqa: E402

from f1pred.config import FIGURES  # noqa: E402

SURFACE, INK, INK2, MUTED = "#fcfcfb", "#0b0b0b", "#52514e", "#898781"
GRID, AXIS, CONTEXT = "#e1e0d9", "#c3c2b7", "#c3c2b7"
BLUE, ORANGE = "#2a78d6", "#eb6834"
GOOD, CRITICAL = "#0ca30c", "#d03b3b"
DPI = 200

plt.rcParams.update({
    "font.family": ["Helvetica Neue", "Helvetica", "Arial", "DejaVu Sans"],
    "font.size": 10,
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "text.color": INK, "axes.labelcolor": INK2, "axes.edgecolor": AXIS,
    "xtick.color": MUTED, "ytick.color": MUTED,
    "xtick.labelcolor": INK2, "ytick.labelcolor": INK2,
    "axes.spines.top": False, "axes.spines.right": False, "axes.spines.left": False,
    "axes.grid": True, "axes.grid.axis": "y", "grid.color": GRID, "grid.linewidth": 0.8,
    "axes.axisbelow": True, "axes.linewidth": 0.8,
    "xtick.major.size": 0, "ytick.major.size": 0,
    "legend.frameon": False, "lines.linewidth": 2.0, "lines.solid_capstyle": "round",
})


def new_figure(width: float, height: float, ncols: int = 1, left: float = 0.9,
               right: float = 0.3, top: float = 0.85, bottom: float = 0.7,
               wspace: float = 0.9) -> tuple[plt.Figure, list[plt.Axes]]:
    """A figure with fixed margins in inches, so bar geometry is predictable."""
    fig, axes = plt.subplots(1, ncols, figsize=(width, height))
    axes = [axes] if ncols == 1 else list(axes)
    plot_w = (width - left - right - wspace * (ncols - 1)) / ncols
    fig.subplots_adjust(left=left / width, right=1 - right / width, top=1 - top / height,
                        bottom=bottom / height, wspace=wspace / plot_w)
    return fig, axes


def titles(fig: plt.Figure, title: str, subtitle: str = "", left: float = 0.9) -> None:
    """Left-aligned title and subtitle above the plot area."""
    w, h = fig.get_size_inches()
    fig.text(left / w, 1 - 0.32 / h, title, fontsize=12.5, fontweight="bold", color=INK,
             ha="left", va="center")
    if subtitle:
        fig.text(left / w, 1 - 0.58 / h, subtitle, fontsize=9.5, color=INK2,
                 ha="left", va="center")


def rounded_bars(ax: plt.Axes, positions: np.ndarray, lengths: np.ndarray, color: str,
                 thickness_in: float = 0.2, radius_in: float = 0.04,
                 horizontal: bool = False) -> None:
    """Thin bars from a zero baseline: rounded at the data end, square at the baseline.

    Call after the axis limits are set, since the corner radius is in inches.
    """
    fig = ax.figure
    box = ax.get_position()
    width_in = box.width * fig.get_size_inches()[0]
    height_in = box.height * fig.get_size_inches()[1]
    (x0, x1), (y0, y1) = ax.get_xlim(), ax.get_ylim()
    per_in_x, per_in_y = abs(x1 - x0) / width_in, abs(y1 - y0) / height_in
    pos_scale, len_scale = (per_in_y, per_in_x) if horizontal else (per_in_x, per_in_y)
    half = thickness_in * pos_scale / 2
    for pos, length in zip(positions, lengths, strict=True):
        if not np.isfinite(length) or length <= 0:
            continue
        r_len = min(radius_in * len_scale, length)
        r_pos = min(radius_in * pos_scale, half)
        # (position, length) outline, clockwise from the baseline
        pts = [(pos - half, 0), (pos - half, length - r_len), (pos - half, length),
               (pos - half + r_pos, length), (pos + half - r_pos, length), (pos + half, length),
               (pos + half, length - r_len), (pos + half, 0), (pos - half, 0)]
        codes = [MplPath.MOVETO, MplPath.LINETO, MplPath.CURVE3, MplPath.CURVE3,
                 MplPath.LINETO, MplPath.CURVE3, MplPath.CURVE3, MplPath.LINETO,
                 MplPath.CLOSEPOLY]
        verts = [(ln, p) for p, ln in pts] if horizontal else pts
        ax.add_patch(PathPatch(MplPath(verts, codes), facecolor=color, edgecolor="none", zorder=3))


def pct_axis(ax: plt.Axes, axis: str = "y", upper: float = 1.0, step: float = 0.25) -> None:
    ticks = np.arange(0, upper + 1e-9, step)
    getattr(ax, f"set_{axis}ticks")(ticks)
    getattr(ax, f"set_{axis}ticklabels")([f"{t:.0%}" for t in ticks])
    getattr(ax, f"set_{axis}lim")(0, upper)


def save(fig: plt.Figure, name: str, directory: Path = FIGURES) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / name
    fig.savefig(path, dpi=DPI)
    plt.close(fig)
    return path


# ----------------------------------------------------------------- the figures

def pole_conversion_by_season(table: pd.DataFrame) -> plt.Figure:
    """Columns: share of races won from pole, one bar per season.

    ``table`` has columns season, races, pole_wins.
    """
    fig, (ax,) = new_figure(7.2, 3.9)
    rate = table["pole_wins"] / table["races"]
    x = np.arange(len(table))
    ax.set_xlim(-0.6, len(table) - 0.4)
    pct_axis(ax)
    rounded_bars(ax, x, rate.to_numpy(), BLUE, thickness_in=0.24)
    for xi, r, wins, races in zip(x, rate, table["pole_wins"], table["races"], strict=True):
        ax.text(xi, r + 0.025, f"{r:.0%}", ha="center", va="bottom", fontsize=9.5, color=INK)
        ax.text(xi, -0.105, f"{wins} of {races}", ha="center", va="top", fontsize=8.5,
                color=MUTED)
    ax.set_xticks(x)
    ax.set_xticklabels(table["season"].astype(str))
    ax.spines["bottom"].set_color(AXIS)
    titles(fig, "The pole sitter wins about half the time",
           "Share of races won by the driver starting first, by season")
    return fig


def calibration(tables: dict[str, pd.DataFrame], subtitle: str) -> plt.Figure:
    """Predicted against observed win rate per probability bin, with 95% Wilson intervals."""
    fig, (ax,) = new_figure(6.4, 5.6, top=0.95)
    ax.grid(True, axis="both")
    ax.plot([0, 1], [0, 1], color=AXIS, linewidth=1.0, zorder=1)
    ax.text(0.9, 0.87, "perfect calibration", rotation=40, color=MUTED, fontsize=8.5,
            ha="center", va="center", rotation_mode="anchor")
    for (name, table), color, marker in zip(tables.items(), (BLUE, ORANGE), ("o", "s"),
                                             strict=False):
        n, k = table["rows"].to_numpy(), table["wins"].to_numpy()
        p_hat = k / n
        z = 1.96
        centre = (p_hat + z * z / (2 * n)) / (1 + z * z / n)
        half = z * np.sqrt(p_hat * (1 - p_hat) / n + z * z / (4 * n * n)) / (1 + z * z / n)
        ax.vlines(table["predicted"], centre - half, centre + half, color=color,
                  linewidth=1.2, alpha=0.55, zorder=2)
        ax.plot(table["predicted"], table["observed"], color=color, marker=marker,
                markersize=7.5, markeredgecolor=SURFACE, markeredgewidth=1.6,
                label=name, zorder=3, clip_on=False)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ticks = np.arange(0, 1.01, 0.2)
    ax.set_xticks(ticks)
    ax.set_yticks(ticks)
    ax.set_xticklabels([f"{t:.0%}" for t in ticks])
    ax.set_yticklabels([f"{t:.0%}" for t in ticks])
    ax.set_xlabel("Predicted win probability (bin mean)")
    ax.set_ylabel("Share of drivers in the bin who won")
    ax.set_aspect("equal")
    ax.legend(loc="upper left", fontsize=9.5, handlelength=1.6)
    titles(fig, "When the model says 40%, does the driver win 40% of the time?", subtitle)
    return fig


def ablation(table: pd.DataFrame, pole_hit: float, logreg_loss: float,
             chosen: int, subtitle: str) -> plt.Figure:
    """Two panels on separate axes: hit rate and log loss as feature groups are added."""
    fig, (a, b) = new_figure(9.6, 4.5, ncols=2, top=1.3, bottom=0.95, wspace=1.0)
    x = np.arange(len(table))
    panels = [
        (a, "winner_hit", "Winner hit rate", pole_hit, "pole sitter", True),
        (b, "log_loss", "Log loss (lower is better)", logreg_loss, "grid-only logistic", False),
    ]
    for ax, col, label, ref, ref_label, as_pct in panels:
        y = table[col].to_numpy()
        lo, hi = min(y.min(), ref), max(y.max(), ref)
        pad = (hi - lo) * 0.35 or 0.01
        ax.set_ylim(lo - pad, hi + pad)
        ax.set_xlim(-0.4, len(table) - 0.6)
        ax.axhline(ref, color=CONTEXT, linewidth=1.4, zorder=1)
        ax.text(len(table) - 0.65, ref, f" {ref_label} ", color=INK2, fontsize=8.5,
                ha="right", va="bottom")
        ax.plot(x, y, color=BLUE, marker="o", markersize=7.5, markeredgecolor=SURFACE,
                markeredgewidth=1.6, zorder=3)
        ax.plot(x[chosen], y[chosen], marker="o", markersize=12, markerfacecolor="none",
                markeredgecolor=INK, markeredgewidth=1.3, linestyle="none", zorder=4)
        fmt = (lambda v: f"{v:.0%}") if as_pct else (lambda v: f"{v:.4f}")
        left_half = chosen < len(table) / 2
        ax.annotate("frozen: " + fmt(y[chosen]), (x[chosen], y[chosen]),
                    xytext=(11 if left_half else -11, 9), textcoords="offset points",
                    ha="left" if left_half else "right", fontsize=8.5, color=INK)
        ax.set_xticks(x)
        ax.set_xticklabels(table["step"], rotation=30, ha="right", rotation_mode="anchor")
        ax.set_title(label, loc="left", fontsize=10, color=INK2, pad=8)
        if as_pct:
            ax.yaxis.set_major_locator(MultipleLocator(0.05 if hi - lo > 0.1 else 0.02))
            ax.yaxis.set_major_formatter(lambda v, _: f"{v:.0%}")
        else:
            ax.yaxis.set_major_locator(MaxNLocator(5))
            ax.yaxis.set_major_formatter(lambda v, _: f"{v:.3f}")
        ax.spines["bottom"].set_color(AXIS)
    titles(fig, "What each feature group adds", subtitle)
    return fig


def feature_importance(importance: pd.Series, subtitle: str) -> plt.Figure:
    """Horizontal bars: mean absolute SHAP contribution per feature, largest first."""
    importance = importance.sort_values()
    fig, (ax,) = new_figure(7.2, 0.27 * len(importance) + 1.5, left=1.9, bottom=0.6)
    ax.grid(True, axis="x")
    ax.grid(False, axis="y")
    y = np.arange(len(importance))
    ax.set_ylim(-0.7, len(importance) - 0.3)
    ax.set_xlim(0, importance.max() * 1.12)
    rounded_bars(ax, y, importance.to_numpy(), BLUE, thickness_in=0.13, horizontal=True)
    ax.set_yticks(y)
    ax.set_yticklabels(importance.index)
    ax.text(importance.iloc[-1], y[-1], f"  {importance.iloc[-1]:.2f}", va="center",
            fontsize=9, color=INK)
    ax.set_xlabel("Mean |SHAP| (log-odds of winning)")
    titles(fig, "Starting position and qualifying gap carry most of the signal", subtitle,
           left=0.3)
    return fig


def race_strip(per_race: pd.DataFrame) -> plt.Figure:
    """One mark per 2025 race: the pick's probability, hits and misses marked."""
    fig, (ax,) = new_figure(10.5, 4.3, bottom=0.95)
    x = np.arange(len(per_race))
    ax.set_xlim(-0.7, len(per_race) - 0.3)
    pct_axis(ax)
    p = per_race["pick_p"].to_numpy()
    hit = per_race["hit"].to_numpy(dtype=bool)
    ax.vlines(x, 0, p, color=CONTEXT, linewidth=1.6, zorder=2)
    ax.scatter(x[hit], p[hit], s=95, marker="o", color=GOOD, edgecolor=SURFACE,
               linewidth=1.6, zorder=3, label=f"pick won ({hit.sum()})")
    ax.scatter(x[~hit], p[~hit], s=80, marker="X", color=CRITICAL, edgecolor=SURFACE,
               linewidth=1.2, zorder=3, label=f"pick lost ({(~hit).sum()})")
    for xi, pi, code in zip(x, p, per_race["pick"], strict=True):
        ax.text(xi, pi + 0.05, code, ha="center", va="bottom", fontsize=8, color=INK2)
    ax.set_xticks(x)
    ax.set_xticklabels(per_race["label"], rotation=45, ha="right", rotation_mode="anchor",
                       fontsize=8.5)
    ax.set_ylabel("Win probability given to the pick")
    ax.spines["bottom"].set_color(AXIS)
    ax.legend(loc="upper right", ncols=2, fontsize=9, handletextpad=0.3, columnspacing=1.4,
              bbox_to_anchor=(1.0, 1.13))
    titles(fig, "The model's pick for each 2025 race",
           "Driver code above each mark; height is the win probability the model gave them")
    return fig


# ------------------------------------------------------- notebook-only helpers

def show(fig: plt.Figure):
    """Render a figure as an inline PNG in a notebook (the backend here is Agg)."""
    import io

    from IPython.display import Image

    buf = io.BytesIO()
    fig.savefig(buf, dpi=DPI, format="png")
    plt.close(fig)
    return Image(data=buf.getvalue(), width=int(fig.get_size_inches()[0] * 100))


def share_bars(labels: list[str], shares: list[float], notes: list[str], title: str,
               subtitle: str) -> plt.Figure:
    """A few thin columns of shares, each with its value on the cap and a note below."""
    fig, (ax,) = new_figure(1.2 * len(labels) + 2.2, 3.9)
    x = np.arange(len(labels))
    ax.set_xlim(-0.6, len(labels) - 0.4)
    pct_axis(ax)
    rounded_bars(ax, x, np.asarray(shares, dtype=float), BLUE, thickness_in=0.24)
    for xi, share, note in zip(x, shares, notes, strict=True):
        ax.text(xi, share + 0.025, f"{share:.0%}", ha="center", va="bottom", fontsize=9.5,
                color=INK)
        ax.text(xi, -0.105, note, ha="center", va="top", fontsize=8.5, color=MUTED)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.spines["bottom"].set_color(AXIS)
    titles(fig, title, subtitle)
    return fig


def circuit_bars(table: pd.DataFrame, title: str, subtitle: str) -> plt.Figure:
    """Horizontal bars: pole conversion per circuit, with wins of races at the bar end."""
    table = table.sort_values(["rate", "races"]).reset_index(drop=True)
    fig, (ax,) = new_figure(7.2, 0.27 * len(table) + 1.5, left=1.7, bottom=0.6)
    ax.grid(True, axis="x")
    ax.grid(False, axis="y")
    y = np.arange(len(table))
    ax.set_ylim(-0.7, len(table) - 0.3)
    pct_axis(ax, axis="x", upper=1.15)
    ax.set_xticks(np.arange(0, 1.01, 0.25))
    ax.set_xticklabels([f"{t:.0%}" for t in np.arange(0, 1.01, 0.25)])
    rounded_bars(ax, y, table["rate"].to_numpy(), BLUE, thickness_in=0.13, horizontal=True)
    for yi, rate, wins, races in zip(y, table["rate"], table["pole_wins"], table["races"],
                                     strict=True):
        ax.text(rate + 0.02, yi, f"{int(wins)} of {int(races)}", va="center", fontsize=8.5,
                color=INK2)
    ax.set_yticks(y)
    ax.set_yticklabels(table["circuit"].str.replace("_", " ").str.title())
    titles(fig, title, subtitle, left=0.3)
    return fig


def coverage_map(matrix: pd.DataFrame, title: str, subtitle: str) -> plt.Figure:
    """Season-by-round grid: share of drivers with a long run (one hue, light to dark)."""
    from matplotlib.colors import LinearSegmentedColormap

    ramp = LinearSegmentedColormap.from_list("blue", ["#cde2fb", "#3987e5", "#0d366b"])
    fig, (ax,) = new_figure(9.6, 0.36 * len(matrix) + 2.1, left=0.8, bottom=1.0, right=0.4)
    ax.grid(False)
    values = matrix.to_numpy(dtype=float)
    for i in range(values.shape[0]):
        for j in range(values.shape[1]):
            v = values[i, j]
            if np.isnan(v):
                continue
            ax.add_patch(plt.Rectangle((j + 0.04, i + 0.06), 0.92, 0.88,
                                       facecolor=ramp(v) if v > 0 else SURFACE,
                                       edgecolor=AXIS if v == 0 else "none", linewidth=0.8))
            if v == 0:
                ax.text(j + 0.5, i + 0.5, "0", ha="center", va="center", fontsize=8,
                        color=INK2)
    ax.set_xlim(0, values.shape[1])
    ax.set_ylim(values.shape[0], 0)
    ax.set_xticks(np.arange(values.shape[1]) + 0.5)
    ax.set_xticklabels(matrix.columns)
    ax.set_yticks(np.arange(values.shape[0]) + 0.5)
    ax.set_yticklabels(matrix.index)
    ax.set_xlabel("Round")
    for spine in ax.spines.values():
        spine.set_visible(False)
    w, h = fig.get_size_inches()
    for k, (v, label) in enumerate([(0.25, "25%"), (0.5, "50%"), (0.75, "75%"), (1.0, "100%")]):
        x0 = 0.8 / w + k * 0.085
        fig.patches.append(plt.Rectangle((x0, 0.2 / h), 0.022, 0.16 / h, transform=fig.transFigure,
                                         facecolor=ramp(v), edgecolor="none"))
        fig.text(x0 + 0.027, 0.28 / h, label, fontsize=8.5, color=INK2, va="center")
    fig.text(0.8 / w + 4 * 0.085, 0.28 / h, "of drivers with a long run; 0 = none",
             fontsize=8.5, color=INK2, va="center")
    titles(fig, title, subtitle, left=0.8)
    return fig
