#!/usr/bin/env python
"""
Generate composite breakout/regression content cards for hitters and pitchers.

Usage
-----
    # Quick (fewer draws, faster)
    python scripts/generate_composite_cards.py --quick

    # Full quality
    python scripts/generate_composite_cards.py

    # Hitters only
    python scripts/generate_composite_cards.py --quick --hitters-only

    # Pitchers only
    python scripts/generate_composite_cards.py --quick --pitchers-only
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.viz.theme import apply_theme, save_card
from src.viz.composite_cards import plot_hitter_composite, plot_pitcher_composite

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(name)-25s %(levelname)-7s %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("composite_cards")

SEASONS = list(range(2018, 2026))
FROM_SEASON = 2025
SEASON_LABEL = "2026"
N_TOP = 10


def generate_hitter_cards(
    draws: int, tune: int, chains: int, n_top: int = N_TOP,
) -> None:
    """Fit hitter models and generate breakout + regression cards."""
    from src.models.hitter_projections import (
        fit_all_models,
        project_forward,
        find_breakouts_and_regressions,
    )

    logger.info("Fitting hitter models...")
    results = fit_all_models(
        seasons=SEASONS, min_pa=100,
        draws=draws, tune=tune, chains=chains, random_seed=42,
    )

    logger.info("Projecting hitters forward from %d...", FROM_SEASON)
    proj = project_forward(results, from_season=FROM_SEASON, min_pa=200)
    logger.info("Projected %d hitters", len(proj))

    breakouts, regressions = find_breakouts_and_regressions(proj, n_top=n_top)

    apply_theme()

    fig = plot_hitter_composite(breakouts, card_type="breakout",
                                season_label=SEASON_LABEL)
    path = save_card(fig, "hitter_composite_breakouts")
    logger.info("Saved: %s", path)

    fig = plot_hitter_composite(regressions, card_type="regression",
                                season_label=SEASON_LABEL)
    path = save_card(fig, "hitter_composite_regressions")
    logger.info("Saved: %s", path)


def generate_pitcher_cards(
    draws: int, tune: int, chains: int, n_top: int = N_TOP,
) -> None:
    """Fit pitcher models and generate breakout + regression cards."""
    from src.models.pitcher_projections import (
        fit_all_models,
        project_forward,
        find_breakouts_and_regressions,
    )

    logger.info("Fitting pitcher models...")
    results = fit_all_models(
        seasons=SEASONS, min_bf=100,
        draws=draws, tune=tune, chains=chains, random_seed=42,
    )

    logger.info("Projecting pitchers forward from %d...", FROM_SEASON)
    proj = project_forward(results, from_season=FROM_SEASON, min_bf=200)
    logger.info("Projected %d pitchers", len(proj))

    breakouts, regressions = find_breakouts_and_regressions(proj, n_top=n_top)

    apply_theme()

    fig = plot_pitcher_composite(breakouts, card_type="breakout",
                                 season_label=SEASON_LABEL)
    path = save_card(fig, "pitcher_composite_breakouts")
    logger.info("Saved: %s", path)

    fig = plot_pitcher_composite(regressions, card_type="regression",
                                  season_label=SEASON_LABEL)
    path = save_card(fig, "pitcher_composite_regressions")
    logger.info("Saved: %s", path)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate composite breakout/regression content cards",
    )
    parser.add_argument("--quick", action="store_true",
                        help="Fewer MCMC draws (faster but noisier)")
    parser.add_argument("--hitters-only", action="store_true")
    parser.add_argument("--pitchers-only", action="store_true")
    parser.add_argument("--n-top", type=int, default=N_TOP,
                        help="Number of players per card")
    args = parser.parse_args()

    if args.quick:
        draws, tune, chains = 500, 250, 2
        logger.info("QUICK mode: draws=%d, tune=%d, chains=%d",
                     draws, tune, chains)
    else:
        draws, tune, chains = 2000, 1000, 4
        logger.info("FULL mode: draws=%d, tune=%d, chains=%d",
                     draws, tune, chains)

    do_hitters = not args.pitchers_only
    do_pitchers = not args.hitters_only

    if do_hitters:
        generate_hitter_cards(draws, tune, chains, n_top=args.n_top)

    if do_pitchers:
        generate_pitcher_cards(draws, tune, chains, n_top=args.n_top)

    logger.info("Done! Cards saved to outputs/content/")


if __name__ == "__main__":
    main()
