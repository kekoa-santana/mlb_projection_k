"""
Walk-forward backtesting for the generalized pitcher projection model.

Supports all stats in PITCHER_STAT_CONFIGS (K%, BB%, HR/BF).
Evaluates Bayesian projections vs Marcel baseline across folds.
"""
from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import beta as beta_dist
from sklearn.metrics import brier_score_loss

from src.data.feature_eng import build_multi_season_pitcher_data
from src.models.pitcher_model import (
    PITCHER_STAT_CONFIGS,
    check_convergence,
    extract_rate_samples,
    fit_pitcher_model,
    prepare_pitcher_data,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Marcel baseline (generalized for pitchers)
# ---------------------------------------------------------------------------
def marcel_pitcher(
    df_history: pd.DataFrame,
    stat: str,
    league_avg: float | None = None,
) -> pd.DataFrame:
    """Marcel projection for any pitcher rate stat.

    Weights recent seasons 5/4/3, regresses toward league mean.

    Parameters
    ----------
    df_history : pd.DataFrame
        Multi-season data with pitcher_id, season, batters_faced, and
        the stat's count and rate columns.
    stat : str
        Stat key from PITCHER_STAT_CONFIGS.
    league_avg : float | None
        League-average rate.

    Returns
    -------
    pd.DataFrame
        Columns: pitcher_id, marcel_{stat}, reliability, weighted_bf.
    """
    cfg = PITCHER_STAT_CONFIGS[stat]
    weights = {0: 5, 1: 4, 2: 3}
    available_seasons = sorted(df_history["season"].unique(), reverse=True)

    if league_avg is None:
        total_count = df_history[cfg.count_col].sum()
        total_bf = df_history[cfg.trials_col].sum()
        league_avg = total_count / total_bf if total_bf > 0 else cfg.league_avg

    # Pitchers are less stable — smaller regression constant
    reg_bf = {"k_rate": 800, "bb_rate": 800, "hr_per_bf": 1200}
    reg = reg_bf.get(stat, 800)

    records = []
    for pitcher_id, group in df_history.groupby("pitcher_id"):
        weighted_count = 0.0
        weighted_bf = 0.0

        for offset, season in enumerate(available_seasons):
            if offset > 2:
                break
            w = weights.get(offset, 0)
            row = group[group["season"] == season]
            if len(row) == 0:
                continue
            r = row.iloc[0]
            weighted_count += w * float(r[cfg.count_col])
            weighted_bf += w * float(r[cfg.trials_col])

        if weighted_bf == 0:
            continue

        raw_rate = weighted_count / weighted_bf
        reliability = weighted_bf / (weighted_bf + reg)
        marcel_rate = reliability * raw_rate + (1 - reliability) * league_avg

        records.append({
            "pitcher_id": pitcher_id,
            f"marcel_{stat}": marcel_rate,
            "reliability": reliability,
            "weighted_bf": weighted_bf,
        })

    return pd.DataFrame(records)


# ---------------------------------------------------------------------------
# Walk-forward for a single stat
# ---------------------------------------------------------------------------
def walk_forward_pitcher_stat_backtest(
    train_seasons: list[int],
    test_season: int,
    stat: str,
    min_bf_train: int = 50,
    min_bf_test: int = 100,
    draws: int = 1000,
    tune: int = 500,
    chains: int = 2,
    random_seed: int = 42,
) -> dict[str, Any]:
    """Train on historical seasons, project forward, evaluate on test season.

    Parameters
    ----------
    train_seasons : list[int]
        Seasons to train on.
    test_season : int
        Season to evaluate against.
    stat : str
        Stat key from PITCHER_STAT_CONFIGS.
    min_bf_train, min_bf_test
        Minimum BF thresholds.
    draws, tune, chains, random_seed
        MCMC sampling parameters.

    Returns
    -------
    dict
        Evaluation metrics, comparison DataFrame, convergence info.
    """
    cfg = PITCHER_STAT_CONFIGS[stat]
    logger.info("Pitcher walk-forward %s: train=%s, test=%d",
                stat, train_seasons, test_season)

    # Load data
    all_seasons = sorted(set(train_seasons) | {test_season})
    df_all = build_multi_season_pitcher_data(all_seasons, min_bf=min_bf_train)

    df_train = df_all[df_all["season"].isin(train_seasons)]
    df_test = df_all[
        (df_all["season"] == test_season) & (df_all["batters_faced"] >= min_bf_test)
    ]

    common = set(df_train["pitcher_id"]) & set(df_test["pitcher_id"])
    logger.info("Common pitchers: %d", len(common))
    if len(common) < 10:
        logger.warning("Very few common pitchers")

    # Fit model on train data only
    df_model = df_train[df_train["pitcher_id"].isin(common)].copy()
    data = prepare_pitcher_data(df_model, stat)
    model, trace = fit_pitcher_model(
        data, draws=draws, tune=tune, chains=chains, random_seed=random_seed,
    )

    conv = check_convergence(trace, stat)

    # Forward-project from last training season
    last_train = max(train_seasons)
    proj_means = {}
    proj_sds = {}
    proj_lo = {}
    proj_hi = {}

    for pitcher_id in common:
        try:
            samples = extract_rate_samples(
                trace, data, pitcher_id, last_train,
                project_forward=True, random_seed=random_seed,
            )
            proj_means[pitcher_id] = float(np.mean(samples))
            proj_sds[pitcher_id] = float(np.std(samples))
            proj_lo[pitcher_id] = float(np.percentile(samples, 2.5))
            proj_hi[pitcher_id] = float(np.percentile(samples, 97.5))
        except ValueError:
            continue

    # Marcel baseline
    marcel = marcel_pitcher(
        df_train[df_train["pitcher_id"].isin(common)], stat,
    )

    # Build comparison
    test_actuals = df_test[df_test["pitcher_id"].isin(common)][
        ["pitcher_id", "pitcher_name", "batters_faced", cfg.rate_col]
    ].rename(columns={cfg.rate_col: f"actual_{stat}", "batters_faced": "actual_bf"})

    comp = test_actuals.copy()
    comp[f"bayes_{stat}"] = comp["pitcher_id"].map(proj_means)
    comp[f"bayes_{stat}_sd"] = comp["pitcher_id"].map(proj_sds)
    comp["ci_95_lo"] = comp["pitcher_id"].map(proj_lo)
    comp["ci_95_hi"] = comp["pitcher_id"].map(proj_hi)
    comp = comp.merge(marcel[["pitcher_id", f"marcel_{stat}", "weighted_bf"]],
                      on="pitcher_id", how="inner")
    comp = comp.dropna(subset=[f"bayes_{stat}", f"actual_{stat}"])

    # Metrics
    actual = comp[f"actual_{stat}"].values
    bayes = comp[f"bayes_{stat}"].values
    marcel_pred = comp[f"marcel_{stat}"].values

    bayes_mae = float(np.mean(np.abs(actual - bayes)))
    marcel_mae = float(np.mean(np.abs(actual - marcel_pred)))
    bayes_rmse = float(np.sqrt(np.mean((actual - bayes) ** 2)))
    marcel_rmse = float(np.sqrt(np.mean((actual - marcel_pred) ** 2)))

    ci_lo = comp["ci_95_lo"].values
    ci_hi = comp["ci_95_hi"].values
    coverage_95 = float(np.mean((actual >= ci_lo) & (actual <= ci_hi)))

    mae_imp = (marcel_mae - bayes_mae) / marcel_mae * 100 if marcel_mae > 0 else 0
    rmse_imp = (marcel_rmse - bayes_rmse) / marcel_rmse * 100 if marcel_rmse > 0 else 0

    # Brier score
    league_avg = cfg.league_avg
    actual_above = (actual > league_avg).astype(float)

    bayes_prob_above = np.array([float(np.mean(
        extract_rate_samples(trace, data, pid, last_train,
                             project_forward=True, random_seed=random_seed)
        > league_avg
    )) if pid in proj_means else 0.5 for pid in comp["pitcher_id"]])

    strength = np.maximum(comp["weighted_bf"].values.astype(float), 1.0)
    alpha = 1.0 + (marcel_pred * strength)
    beta_param = 1.0 + ((1.0 - marcel_pred) * strength)
    marcel_prob_above = beta_dist.sf(league_avg, alpha, beta_param)

    bayes_brier = float(brier_score_loss(actual_above, bayes_prob_above))
    marcel_brier = float(brier_score_loss(actual_above, marcel_prob_above))

    logger.info(
        "pitcher %s: Bayes MAE=%.4f, Marcel MAE=%.4f (%.1f%%)",
        stat, bayes_mae, marcel_mae, mae_imp,
    )
    logger.info("pitcher %s: 95%% CI coverage: %.1f%%", stat, coverage_95 * 100)

    return {
        "stat": stat,
        "test_season": test_season,
        "n_players": len(comp),
        "bayes_mae": bayes_mae,
        "marcel_mae": marcel_mae,
        "mae_improvement_pct": mae_imp,
        "bayes_rmse": bayes_rmse,
        "marcel_rmse": marcel_rmse,
        "rmse_improvement_pct": rmse_imp,
        "coverage_95": coverage_95,
        "bayes_brier": bayes_brier,
        "marcel_brier": marcel_brier,
        "convergence": conv,
        "comparison_df": comp,
    }


# ---------------------------------------------------------------------------
# Multi-fold, multi-stat backtest
# ---------------------------------------------------------------------------
def run_pitcher_backtest(
    stats: list[str] | None = None,
    folds: list[dict[str, Any]] | None = None,
    **sampling_kwargs: Any,
) -> pd.DataFrame:
    """Run walk-forward backtesting across stats and folds.

    Parameters
    ----------
    stats : list[str] | None
        Stats to backtest. Defaults to all in PITCHER_STAT_CONFIGS.
    folds : list[dict]
        Each dict has 'train_seasons' and 'test_season'.
    **sampling_kwargs
        Passed to ``walk_forward_pitcher_stat_backtest``.

    Returns
    -------
    pd.DataFrame
        Summary metrics across all stats and folds.
    """
    if stats is None:
        stats = list(PITCHER_STAT_CONFIGS.keys())
    if folds is None:
        folds = [
            {"train_seasons": [2018, 2019, 2020, 2021, 2022], "test_season": 2023},
            {"train_seasons": [2018, 2019, 2020, 2021, 2022, 2023], "test_season": 2024},
            {"train_seasons": [2018, 2019, 2020, 2021, 2022, 2023, 2024], "test_season": 2025},
        ]

    results = []
    for stat in stats:
        for fold in folds:
            logger.info(
                "=== pitcher %s: train=%s -> test=%d ===",
                stat, fold["train_seasons"], fold["test_season"],
            )
            metrics = walk_forward_pitcher_stat_backtest(
                train_seasons=fold["train_seasons"],
                test_season=fold["test_season"],
                stat=stat,
                **sampling_kwargs,
            )
            results.append({
                "stat": stat,
                "test_season": metrics["test_season"],
                "n_players": metrics["n_players"],
                "bayes_mae": metrics["bayes_mae"],
                "marcel_mae": metrics["marcel_mae"],
                "mae_improvement_pct": metrics["mae_improvement_pct"],
                "bayes_rmse": metrics["bayes_rmse"],
                "marcel_rmse": metrics["marcel_rmse"],
                "rmse_improvement_pct": metrics["rmse_improvement_pct"],
                "coverage_95": metrics["coverage_95"],
                "bayes_brier": metrics["bayes_brier"],
                "marcel_brier": metrics["marcel_brier"],
                "converged": metrics["convergence"]["converged"],
            })

    summary = pd.DataFrame(results)
    logger.info("\n=== Pitcher Backtest Summary ===\n%s", summary.to_string())
    return summary
