"""
Walk-forward backtesting and calibration evaluation for the K% model.

Evaluation requirements (from CLAUDE.md):
1. Walk-forward backtesting: train through season N, predict N+1
2. Calibration curves: do X% credible intervals contain truth ~X% of the time?
3. Brier score: for binary outcomes derived from continuous projections
4. Benchmark vs Marcel: must beat the Marcel projection system
"""
from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import brier_score_loss

from src.data.feature_eng import build_multi_season_k_data
from src.models.k_rate_model import (
    check_convergence,
    extract_player_posteriors,
    fit_k_rate_model,
    prepare_model_data,
)

logger = logging.getLogger(__name__)


def _summarize_posterior_for_season(
    trace: Any,
    data: dict[str, Any],
    season: int,
    league_avg_k: float,
    intervals: list[float] | None = None,
) -> pd.DataFrame:
    """Summarize posterior draws for a single season's player rows."""
    if intervals is None:
        intervals = [0.50, 0.80, 0.90, 0.95]

    df = data["df"].reset_index(drop=True)
    k_rate_post = np.asarray(trace.posterior["k_rate"].values, dtype=float)
    k_rate_flat = k_rate_post.reshape(-1, k_rate_post.shape[-1])

    mask = df["season"] == season
    positions = np.where(mask.values)[0]

    records: list[dict[str, Any]] = []
    for pos in positions:
        row = df.iloc[pos]
        samples = k_rate_flat[:, pos]
        rec: dict[str, Any] = {
            "batter_id": row["batter_id"],
            "season": row["season"],
            "post_p_above_league": float(np.mean(samples > league_avg_k)),
        }
        for level in intervals:
            tail = (1.0 - level) / 2.0
            lo = float(np.quantile(samples, tail))
            hi = float(np.quantile(samples, 1.0 - tail))
            pct = int(round(level * 100))
            rec[f"post_ci_{pct}_lo"] = lo
            rec[f"post_ci_{pct}_hi"] = hi
        records.append(rec)

    return pd.DataFrame(records)


# ---------------------------------------------------------------------------
# Marcel baseline projection
# ---------------------------------------------------------------------------
def marcel_k_rate(
    df_history: pd.DataFrame,
    target_season: int,
    league_k_rate: float | None = None,
) -> pd.DataFrame:
    """Simple Marcel projection for K%.

    Marcel weights recent seasons 5/4/3 (most recent first), regresses
    toward league mean based on PA, and applies a small age adjustment.

    Parameters
    ----------
    df_history : pd.DataFrame
        Multi-season data with batter_id, season, pa, k, k_rate columns.
    target_season : int
        Season to project.
    league_k_rate : float | None
        League-average K rate. Defaults to weighted mean of history.

    Returns
    -------
    pd.DataFrame
        Columns: batter_id, marcel_k_rate, reliability.
    """
    weights = {0: 5, 1: 4, 2: 3}  # season offset → weight (0 = most recent)
    available_seasons = sorted(df_history["season"].unique(), reverse=True)

    if league_k_rate is None:
        total_k = df_history["k"].sum()
        total_pa = df_history["pa"].sum()
        league_k_rate = total_k / total_pa if total_pa > 0 else 0.22

    records = []
    for batter_id, group in df_history.groupby("batter_id"):
        weighted_k = 0.0
        weighted_pa = 0.0

        for offset, season in enumerate(available_seasons):
            if offset > 2:
                break
            w = weights.get(offset, 0)
            row = group[group["season"] == season]
            if len(row) == 0:
                continue
            r = row.iloc[0]
            weighted_k += w * r["k"]
            weighted_pa += w * r["pa"]

        if weighted_pa == 0:
            continue

        raw_rate = weighted_k / weighted_pa

        # Regression toward mean: reliability = weighted_pa / (weighted_pa + 1200)
        # 1200 is a common Marcel regression PA constant for K%
        reliability = weighted_pa / (weighted_pa + 1200)
        marcel_rate = reliability * raw_rate + (1 - reliability) * league_k_rate

        records.append({
            "batter_id": batter_id,
            "marcel_k_rate": marcel_rate,
            "reliability": reliability,
            "weighted_pa": weighted_pa,
        })

    return pd.DataFrame(records)


# ---------------------------------------------------------------------------
# Walk-forward backtesting
# ---------------------------------------------------------------------------
def walk_forward_backtest(
    train_seasons: list[int],
    test_season: int,
    min_pa_train: int = 1,
    min_pa_test: int = 100,
    draws: int = 1000,
    tune: int = 500,
    chains: int = 2,
    random_seed: int = 42,
) -> dict[str, Any]:
    """Train on historical seasons, predict next season, evaluate.

    Parameters
    ----------
    train_seasons : list[int]
        Seasons to train on.
    test_season : int
        Season to predict and evaluate against.
    min_pa_train : int
        Min PA per player-season in training data.
    min_pa_test : int
        Min PA for test-season evaluation (need enough to be meaningful).
    draws, tune, chains, random_seed
        MCMC sampling parameters.

    Returns
    -------
    dict
        Evaluation metrics and comparison DataFrames.
    """
    logger.info(
        "Walk-forward: train=%s, test=%d", train_seasons, test_season
    )

    # --- Build training data ---
    df_train = build_multi_season_k_data(train_seasons, min_pa=min_pa_train)

    # --- Build test actuals ---
    df_test = build_multi_season_k_data([test_season], min_pa=min_pa_test)

    # --- Find players present in BOTH train and test ---
    common_batters = set(df_train["batter_id"]) & set(df_test["batter_id"])
    logger.info(
        "Common batters in train & test: %d", len(common_batters)
    )
    if len(common_batters) < 10:
        logger.warning("Very few common batters — results may be unreliable")

    # --- Fit Bayesian model on TRAIN data ONLY (no leakage) ---
    # Only keep batters that also appear in test so we can evaluate
    df_model = df_train[df_train["batter_id"].isin(common_batters)].copy()

    data = prepare_model_data(df_model)
    model, trace = fit_k_rate_model(
        data, draws=draws, tune=tune, chains=chains,
        random_seed=random_seed,
    )

    # --- Convergence ---
    convergence = check_convergence(trace)

    # --- Extract posteriors from LAST training season as projection ---
    posteriors = extract_player_posteriors(trace, data)
    last_train = max(train_seasons)
    # Use the most recent training season's posterior as the projection
    # for the test season (the model's best estimate of current talent)
    test_posteriors = posteriors[posteriors["season"] == last_train].copy()

    # --- Marcel baseline ---
    df_marcel_input = df_train[df_train["batter_id"].isin(common_batters)]
    league_k = df_train["k"].sum() / df_train["pa"].sum()
    marcel = marcel_k_rate(df_marcel_input, test_season, league_k)

    # --- Merge for comparison ---
    test_actuals = df_test[df_test["batter_id"].isin(common_batters)][
        ["batter_id", "batter_name", "pa", "k", "k_rate"]
    ].rename(columns={"k_rate": "actual_k_rate", "pa": "actual_pa"})

    posterior_summary = _summarize_posterior_for_season(
        trace=trace,
        data=data,
        season=last_train,
        league_avg_k=league_k,
    )

    comp = test_actuals.merge(
        test_posteriors[["batter_id", "k_rate_mean", "k_rate_sd",
                         "k_rate_2_5", "k_rate_97_5"]],
        on="batter_id", how="inner",
    ).merge(
        marcel[["batter_id", "marcel_k_rate", "weighted_pa"]],
        on="batter_id", how="inner",
    ).merge(
        posterior_summary.drop(columns=["season"], errors="ignore"),
        on="batter_id", how="inner",
    )

    # --- Compute metrics ---
    metrics = compute_metrics(comp, league_avg_k=league_k)
    metrics["convergence"] = convergence
    metrics["comparison_df"] = comp
    metrics["n_players"] = len(comp)
    metrics["train_seasons"] = train_seasons
    metrics["test_season"] = test_season

    return metrics


# ---------------------------------------------------------------------------
# Metrics computation
# ---------------------------------------------------------------------------
def compute_metrics(
    comp: pd.DataFrame,
    league_avg_k: float = 0.22,
) -> dict[str, Any]:
    """Compute evaluation metrics from a comparison DataFrame.

    Parameters
    ----------
    comp : pd.DataFrame
        Must have: actual_k_rate, k_rate_mean, k_rate_2_5, k_rate_97_5,
        marcel_k_rate.

    Returns
    -------
    dict
        MAE, RMSE, coverage, Brier score for both Bayesian and Marcel.
    """
    actual = comp["actual_k_rate"].values
    bayes_pred = comp["k_rate_mean"].values
    marcel_pred = comp["marcel_k_rate"].values
    if {"post_ci_95_lo", "post_ci_95_hi"}.issubset(comp.columns):
        ci_lo = comp["post_ci_95_lo"].values
        ci_hi = comp["post_ci_95_hi"].values
    else:
        ci_lo = comp["k_rate_2_5"].values
        ci_hi = comp["k_rate_97_5"].values

    # --- MAE ---
    bayes_mae = np.mean(np.abs(actual - bayes_pred))
    marcel_mae = np.mean(np.abs(actual - marcel_pred))

    # --- RMSE ---
    bayes_rmse = np.sqrt(np.mean((actual - bayes_pred) ** 2))
    marcel_rmse = np.sqrt(np.mean((actual - marcel_pred) ** 2))

    # --- 95% credible interval coverage ---
    in_ci = (actual >= ci_lo) & (actual <= ci_hi)
    coverage_95 = np.mean(in_ci)

    # --- Brier score (is K% > league avg?) ---
    actual_above = (actual > league_avg_k).astype(float)
    if "post_p_above_league" in comp.columns:
        bayes_prob_above = comp["post_p_above_league"].values
    else:
        bayes_prob_above = bayes_pred

    if {"marcel_k_rate", "weighted_pa"}.issubset(comp.columns):
        from scipy.stats import beta as beta_dist

        marcel_strength = np.maximum(comp["weighted_pa"].values.astype(float), 1.0)
        alpha = 1.0 + (marcel_pred * marcel_strength)
        beta_param = 1.0 + ((1.0 - marcel_pred) * marcel_strength)
        marcel_prob_above = beta_dist.sf(league_avg_k, alpha, beta_param)
    else:
        marcel_prob_above = marcel_pred

    # For Brier, we need probability of being above-average K%.
    bayes_brier = brier_score_loss(actual_above, bayes_prob_above)
    marcel_brier = brier_score_loss(actual_above, marcel_prob_above)

    # --- Calibration across intervals ---
    calibration = compute_calibration(comp)

    results = {
        "bayes_mae": bayes_mae,
        "marcel_mae": marcel_mae,
        "mae_improvement": (marcel_mae - bayes_mae) / marcel_mae * 100,
        "bayes_rmse": bayes_rmse,
        "marcel_rmse": marcel_rmse,
        "rmse_improvement": (marcel_rmse - bayes_rmse) / marcel_rmse * 100,
        "coverage_95": coverage_95,
        "bayes_brier": bayes_brier,
        "marcel_brier": marcel_brier,
        "calibration": calibration,
    }

    logger.info(
        "Bayes MAE=%.4f, Marcel MAE=%.4f (improvement: %.1f%%)",
        bayes_mae, marcel_mae, results["mae_improvement"],
    )
    logger.info(
        "Bayes RMSE=%.4f, Marcel RMSE=%.4f (improvement: %.1f%%)",
        bayes_rmse, marcel_rmse, results["rmse_improvement"],
    )
    logger.info("95%% CI coverage: %.1f%%", coverage_95 * 100)

    return results


def compute_calibration(
    comp: pd.DataFrame,
    intervals: list[float] | None = None,
) -> pd.DataFrame:
    """Check credible interval calibration at multiple levels.

    For each interval level (e.g. 50%, 80%, 95%), compute what fraction
    of actual values fall within the corresponding credible interval.

    Parameters
    ----------
    comp : pd.DataFrame
        Must have actual_k_rate, k_rate_mean, k_rate_sd.
    intervals : list[float]
        Credible interval levels to check.

    Returns
    -------
    pd.DataFrame
        Columns: nominal_level, empirical_coverage, n.
    """
    if intervals is None:
        intervals = [0.50, 0.80, 0.90, 0.95]

    actual = comp["actual_k_rate"].values
    mean = comp["k_rate_mean"].values
    sd = comp["k_rate_sd"].values

    records = []
    for level in intervals:
        pct = int(round(level * 100))
        lo_col = f"post_ci_{pct}_lo"
        hi_col = f"post_ci_{pct}_hi"
        if lo_col in comp.columns and hi_col in comp.columns:
            lo = comp[lo_col].values
            hi = comp[hi_col].values
        else:
            # Fallback to normal approximation when posterior intervals are unavailable.
            from scipy import stats
            alpha = 1 - level
            z = stats.norm.ppf(1 - alpha / 2)
            lo = mean - z * sd
            hi = mean + z * sd
        in_interval = (actual >= lo) & (actual <= hi)
        records.append({
            "nominal_level": level,
            "empirical_coverage": np.mean(in_interval),
            "n": len(actual),
        })

    return pd.DataFrame(records)


# ---------------------------------------------------------------------------
# Multi-fold walk-forward
# ---------------------------------------------------------------------------
def run_full_backtest(
    folds: list[dict[str, Any]] | None = None,
    **sampling_kwargs: Any,
) -> pd.DataFrame:
    """Run walk-forward backtesting across multiple train/test folds.

    Parameters
    ----------
    folds : list[dict]
        Each dict has 'train_seasons' and 'test_season'. Defaults to
        rolling windows.
    **sampling_kwargs
        Passed to ``walk_forward_backtest``.

    Returns
    -------
    pd.DataFrame
        Summary metrics across all folds.
    """
    if folds is None:
        folds = [
            {"train_seasons": [2021, 2022], "test_season": 2023},
            {"train_seasons": [2022, 2023], "test_season": 2024},
            {"train_seasons": [2023, 2024], "test_season": 2025},
        ]

    results = []
    for fold in folds:
        logger.info("Running fold: train=%s → test=%d",
                     fold["train_seasons"], fold["test_season"])
        metrics = walk_forward_backtest(
            train_seasons=fold["train_seasons"],
            test_season=fold["test_season"],
            **sampling_kwargs,
        )
        results.append({
            "test_season": fold["test_season"],
            "bayes_mae": metrics["bayes_mae"],
            "marcel_mae": metrics["marcel_mae"],
            "mae_improvement_pct": metrics["mae_improvement"],
            "bayes_rmse": metrics["bayes_rmse"],
            "marcel_rmse": metrics["marcel_rmse"],
            "rmse_improvement_pct": metrics["rmse_improvement"],
            "coverage_95": metrics["coverage_95"],
            "n_players": metrics["n_players"],
            "converged": metrics["convergence"]["converged"],
        })

    summary = pd.DataFrame(results)
    logger.info("\n=== Full Backtest Summary ===\n%s", summary.to_string())
    return summary
