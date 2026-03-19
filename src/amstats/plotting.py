"""Consistent matplotlib style defaults for course notebooks."""

import matplotlib.pyplot as plt
import numpy as np

# ---------------------------------------------------------------------------
# ggplot2 default discrete palette — matches manim_utils.COLORS order
#
# Same hex values as the Manim palette so animations and static plots
# feel visually coherent.
# ---------------------------------------------------------------------------
SALMON = "#F8766D"
GOLD = "#B79F00"
EMERALD = "#00BA38"
CYAN = "#00BFC4"
PERIWINKLE = "#619CFF"
ORCHID = "#F564E3"

PALETTE = [SALMON, GOLD, EMERALD, CYAN, PERIWINKLE, ORCHID]


def apply_style():
    """Apply the course matplotlib style globally."""
    plt.rcParams.update(
        {
            "figure.figsize": (8, 5),
            "figure.dpi": 120,
            "axes.prop_cycle": plt.cycler("color", PALETTE),
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.alpha": 0.3,
            "font.size": 12,
            "axes.titlesize": 14,
            "axes.labelsize": 12,
            "legend.fontsize": 11,
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
        }
    )


def hist_with_pdf(data, pdf_func, bins=30, x_label="x", title="", ax=None):
    """Plot a histogram of data overlaid with a theoretical PDF curve."""
    if ax is None:
        _, ax = plt.subplots()

    ax.hist(data, bins=bins, density=True, alpha=0.6, color=PALETTE[0], edgecolor="white")

    x_min, x_max = data.min(), data.max()
    margin = (x_max - x_min) * 0.1
    xs = np.linspace(x_min - margin, x_max + margin, 300)
    ax.plot(xs, pdf_func(xs), color=PALETTE[1], linewidth=2)

    ax.set_xlabel(x_label)
    ax.set_ylabel("Density")
    if title:
        ax.set_title(title)
    return ax
