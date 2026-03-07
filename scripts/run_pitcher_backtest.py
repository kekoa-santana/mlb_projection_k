#!/usr/bin/env python
"""
Run walk-forward backtest for all pitcher stats (K%, BB%, HR/BF).

Usage
-----
    # Quick (fewer draws, faster but noisier)
    python scripts/run_pitcher_backtest.py --quick

    # Single stat only
    python scripts/run_pitcher_backtest.py --quick --stat k_rate

    # Full quality
    python scripts/run_pitcher_backtest.py
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.evaluation.pitcher_backtest import run_pitcher_backtest

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(name)-25s %(levelname)-7s %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("pitcher_backtest")

FOLDS = [
    {"train_seasons": [2018, 2019, 2020, 2021, 2022], "test_season": 2023},
    {"train_seasons": [2018, 2019, 2020, 2021, 2022, 2023], "test_season": 2024},
    {"train_seasons": [2018, 2019, 2020, 2021, 2022, 2023, 2024], "test_season": 2025},
]


def main() -> None:
    parser = argparse.ArgumentParser(description="Pitcher multi-stat backtest")
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--stat", type=str, default=None,
                        help="Single stat to backtest (e.g. k_rate, bb_rate)")
    args = parser.parse_args()

    if args.quick:
        sampling = dict(draws=500, tune=250, chains=2, random_seed=42)
        logger.info("QUICK mode: %s", sampling)
    else:
        sampling = dict(draws=2000, tune=1000, chains=4, random_seed=42)
        logger.info("FULL mode: %s", sampling)

    stats = [args.stat] if args.stat else None

    summary = run_pitcher_backtest(stats=stats, folds=FOLDS, **sampling)

    print("\n=== PITCHER BACKTEST RESULTS ===")
    print(summary.to_string(index=False))
    print()

    print("=== PER-STAT VERDICT ===")
    for stat, group in summary.groupby("stat"):
        avg_mae_imp = group["mae_improvement_pct"].mean()
        avg_cov = group["coverage_95"].mean()
        beats = avg_mae_imp > 0
        print(f"  {stat}: Bayes {'BEATS' if beats else 'LOSES TO'} Marcel "
              f"(MAE improvement: {avg_mae_imp:+.1f}%, "
              f"95% coverage: {avg_cov:.1%})")

    out_dir = PROJECT_ROOT / "outputs"
    out_dir.mkdir(exist_ok=True)
    summary.to_csv(out_dir / "pitcher_multi_stat_backtest.csv", index=False)
    logger.info("Results saved to outputs/pitcher_multi_stat_backtest.csv")


if __name__ == "__main__":
    main()
