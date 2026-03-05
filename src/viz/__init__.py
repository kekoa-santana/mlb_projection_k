"""Visualization utilities for The Data Diamond."""
from src.viz.theme import (
    CREAM,
    DARK_BG,
    GOLD,
    SLATE,
    TEAL,
    WHITE,
    add_brand_footer,
    add_watermark,
    apply_dark_theme,
    apply_theme,
    format_pct,
    save_card,
)
from src.viz.projections import (
    enrich_with_team_info,
    find_movers,
    plot_hitter_k_movers,
    plot_pitcher_card,
    plot_pitcher_k_movers,
    project_posteriors_forward,
)

__all__ = [
    "DARK_BG",
    "GOLD",
    "SLATE",
    "TEAL",
    "WHITE",
    "add_brand_footer",
    "add_watermark",
    "apply_dark_theme",
    "apply_theme",
    "CREAM",
    "enrich_with_team_info",
    "find_movers",
    "format_pct",
    "plot_hitter_k_movers",
    "plot_pitcher_card",
    "plot_pitcher_k_movers",
    "project_posteriors_forward",
    "save_card",
]
