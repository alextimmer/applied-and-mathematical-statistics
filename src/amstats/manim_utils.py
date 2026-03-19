"""Shared Manim helpers: color palette, axes constructors, common animations."""

from manim import (
    Axes,
    ManimColor,
    MathTex,
    Text,
    VGroup,
    config,
)

# ---------------------------------------------------------------------------
# Color palette — custom, colorblind-friendly, vibrant on black backgrounds
#
# Inspired by ggplot2 / grammar-of-graphics aesthetics, tuned for Manim's
# dark canvas.  Every color should be distinguishable in both full-color
# and the three most common forms of color-vision deficiency.
#
# To use:  from amstats.manim_utils import C, COLORS, PALETTE
#          dot = Dot(color=C.CORAL)
#          bar = Rectangle(fill_color=COLORS[2])
# ---------------------------------------------------------------------------

class C:
    """Course color constants — use these instead of Manim built-ins.

    Core palette is the ggplot2 default discrete scale (scales::hue_pal()),
    slightly brightened to pop on Manim's black background.
    """

    # ── ggplot2 hue scale (6-color) — the iconic grammar-of-graphics look ──
    SALMON    = ManimColor("#F8766D")   # ggplot2 hue 1 — warm red/salmon
    GOLD      = ManimColor("#B79F00")   # ggplot2 hue 2 — olive gold
    EMERALD   = ManimColor("#00BA38")   # ggplot2 hue 3 — vivid green
    CYAN      = ManimColor("#00BFC4")   # ggplot2 hue 4 — bright cyan
    PERIWINKLE = ManimColor("#619CFF")  # ggplot2 hue 5 — periwinkle blue
    ORCHID    = ManimColor("#F564E3")   # ggplot2 hue 6 — pink/orchid

    # Semantic aliases — map roles to the palette above
    HIGHLIGHT = SALMON                   # call-outs, important results
    THEORY    = PERIWINKLE               # theoretical curves, reference lines
    DATA      = GOLD                     # observed data points, bars
    NEUTRAL   = ManimColor("#A8A8A8")    # cool grey — axes, grids
    LABEL     = ManimColor("#E0E0E0")    # near-white — readable on black


# Named dict for role-based lookup
PALETTE = {
    "primary":   C.PERIWINKLE,
    "secondary": C.SALMON,
    "accent":    C.EMERALD,
    "highlight": C.HIGHLIGHT,
    "neutral":   C.NEUTRAL,
    "warn":      C.GOLD,
    "extra":     C.ORCHID,
    "theory":    C.THEORY,
    "data":      C.DATA,
    "label":     C.LABEL,
}

# Ordered list for cycling through categories / distributions
COLORS = [C.SALMON, C.GOLD, C.EMERALD, C.CYAN, C.PERIWINKLE, C.ORCHID]


def get_color(index: int):
    """Return a color from the palette, cycling if index exceeds length."""
    return COLORS[index % len(COLORS)]


# ---------------------------------------------------------------------------
# Axes helpers
# ---------------------------------------------------------------------------
def stats_axes(
    x_range: tuple = (0, 10, 1),
    y_range: tuple = (0, 1, 0.2),
    x_length: float = 8,
    y_length: float = 5,
    x_label: str = "x",
    y_label: str = "P(x)",
    **kwargs,
) -> Axes:
    """Create a consistently styled Axes object for statistical plots."""
    axes = Axes(
        x_range=list(x_range),
        y_range=list(y_range),
        x_length=x_length,
        y_length=y_length,
        axis_config={"include_numbers": True, "font_size": 24},
        **kwargs,
    )
    labels = axes.get_axis_labels(
        x_label=MathTex(x_label, font_size=28),
        y_label=MathTex(y_label, font_size=28),
    )
    return VGroup(axes, labels)


# ---------------------------------------------------------------------------
# Common text helpers
# ---------------------------------------------------------------------------
def section_title(text: str, font_size: int = 40) -> Text:
    """Create a styled section title for scene introductions."""
    return Text(text, font_size=font_size, color=PALETTE["primary"])


# ---------------------------------------------------------------------------
# Quality presets
# ---------------------------------------------------------------------------
def set_quality(quality: str = "medium"):
    """Set Manim rendering quality. Use in notebooks before %%manim cells."""
    presets = {
        "low": {"pixel_height": 480, "pixel_width": 854, "frame_rate": 15},
        "medium": {"pixel_height": 720, "pixel_width": 1280, "frame_rate": 30},
        "high": {"pixel_height": 1080, "pixel_width": 1920, "frame_rate": 60},
    }
    preset = presets.get(quality, presets["medium"])
    config.pixel_height = preset["pixel_height"]
    config.pixel_width = preset["pixel_width"]
    config.frame_rate = preset["frame_rate"]
