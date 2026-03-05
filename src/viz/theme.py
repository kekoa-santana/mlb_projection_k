"""
The Data Diamond — cream/navy matplotlib styling for Twitter/X content cards.

Provides brand constants, rcParams setup, watermark/footer helpers, and a
save utility for consistent 16:9 or 1:1 social-media PNGs.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib as mpl
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.offsetbox import AnnotationBbox, OffsetImage

# ---------------------------------------------------------------------------
# Brand palette
# ---------------------------------------------------------------------------
GOLD = "#C8A96E"
TEAL = "#4FC3C8"
SLATE = "#7B8FA6"
CREAM = "#F5F2EE"
DARK_BG = "#0F1117"
WHITE = "#FFFFFF"

PROJECT_ROOT = Path(__file__).resolve().parents[2]
OUTPUTS_DIR = PROJECT_ROOT / "outputs" / "content"
LOGO_PATH = PROJECT_ROOT / "iconTransparent.png"

# Aspect-ratio presets (width, height in inches at 300 DPI)
ASPECT_SIZES: dict[str, tuple[float, float]] = {
    "16:9": (16, 9),
    "1:1": (10, 10),
}


def apply_theme() -> None:
    """Set matplotlib rcParams for cream/navy Twitter card style."""
    mpl.rcParams.update({
        "figure.facecolor": CREAM,
        "axes.facecolor": CREAM,
        "savefig.facecolor": CREAM,
        "text.color": DARK_BG,
        "axes.labelcolor": DARK_BG,
        "xtick.color": DARK_BG,
        "ytick.color": DARK_BG,
        "axes.grid": False,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.spines.left": False,
        "axes.spines.bottom": False,
        "font.size": 14,
        "font.family": "sans-serif",
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.3,
    })


# Backward-compat alias
apply_dark_theme = apply_theme


def add_watermark(fig: Figure) -> None:
    """Add a barely-visible diagonal 'TheDataDiamond' watermark."""
    fig.text(
        0.5, 0.5,
        "TheDataDiamond",
        fontsize=60,
        color=SLATE,
        alpha=0.03,
        ha="center",
        va="center",
        rotation=30,
        transform=fig.transFigure,
        zorder=0,
    )


def add_brand_footer(
    fig: Figure,
    subtitle: str = "Bayesian projection model",
) -> None:
    """Add logo + 'TheDataDiamond' footer left, subtitle right."""
    # --- Logo ---
    if LOGO_PATH.exists():
        logo_img = mpimg.imread(str(LOGO_PATH))
        imagebox = OffsetImage(logo_img, zoom=0.025)
        imagebox.image.axes = fig.axes[0] if fig.axes else None
        ab = AnnotationBbox(
            imagebox, (0.03, 0.025),
            xycoords="figure fraction",
            frameon=False,
            box_alignment=(0.0, 0.0),
        )
        fig.add_artist(ab)
        text_x = 0.078  # shift text right of logo
    else:
        text_x = 0.04

    fig.text(
        text_x, 0.02,
        "TheDataDiamond",
        fontsize=12,
        color=GOLD,
        ha="left",
        va="bottom",
        transform=fig.transFigure,
        fontweight="bold",
    )
    fig.text(
        0.96, 0.02,
        subtitle,
        fontsize=10,
        color=SLATE,
        ha="right",
        va="bottom",
        transform=fig.transFigure,
    )


def save_card(
    fig: Figure,
    name: str,
    output_dir: Path | str | None = None,
    aspect: str = "16:9",
) -> Path:
    """Save a figure as a PNG content card.

    Parameters
    ----------
    fig : Figure
        Matplotlib figure to save.
    name : str
        Filename stem (without extension).
    output_dir : Path | str | None
        Directory to save into. Defaults to ``outputs/content/``.
    aspect : str
        Aspect ratio key (``"16:9"`` or ``"1:1"``).

    Returns
    -------
    Path
        Path to the saved PNG.
    """
    out = Path(output_dir) if output_dir else OUTPUTS_DIR
    out.mkdir(parents=True, exist_ok=True)
    w, h = ASPECT_SIZES.get(aspect, ASPECT_SIZES["16:9"])
    fig.set_size_inches(w, h)
    path = out / f"{name}.png"
    fig.savefig(path)
    plt.close(fig)
    return path


def format_pct(value: float, decimals: int = 1) -> str:
    """Format a rate (0-1) as a percentage string.

    Parameters
    ----------
    value : float
        Value between 0 and 1.
    decimals : int
        Number of decimal places.

    Returns
    -------
    str
        e.g. ``"23.5%"``
    """
    return f"{value * 100:.{decimals}f}%"
