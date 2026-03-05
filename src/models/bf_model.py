"""
Step 13: Batters-Faced distribution model with empirical Bayes partial pooling.

BF for starters is very stable (mean~22, std~4.5). A shrinkage estimator
is appropriate — no MCMC needed.

Shrinkage formula:
    reliability = n / (n + k)       where k = sigma^2 / tau^2
    mu_pitcher  = rel * obs_mean + (1 - rel) * pop_mean
    sigma_pitcher = rel * obs_std + (1 - rel) * pop_within_std
"""
from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats

logger = logging.getLogger(__name__)

# Population defaults (validated against 2022-2025 starter data)
DEFAULT_POP_BF_MU = 22.0
DEFAULT_POP_BF_SIGMA = 4.5
DEFAULT_POP_WITHIN_STD = 3.4
DEFAULT_SHRINKAGE_K = 2.4  # sigma^2 / tau^2 ≈ 11.56 / 4.84


def compute_pitcher_bf_priors(
    game_logs: pd.DataFrame,
    pop_mu: float = DEFAULT_POP_BF_MU,
    pop_within_std: float = DEFAULT_POP_WITHIN_STD,
    shrinkage_k: float = DEFAULT_SHRINKAGE_K,
    min_starts: int = 5,
) -> pd.DataFrame:
    """Compute shrinkage-estimated BF priors per pitcher-season.

    Parameters
    ----------
    game_logs : pd.DataFrame
        Stacked game logs across seasons. Must have columns:
        pitcher_id, season, batters_faced, is_starter.
    pop_mu : float
        Population mean BF for starters.
    pop_within_std : float
        Population within-pitcher game-to-game std.
    shrinkage_k : float
        Shrinkage constant k = sigma^2 / tau^2.
    min_starts : int
        Pitchers below this get pure population prior.

    Returns
    -------
    pd.DataFrame
        One row per (pitcher_id, season) with columns:
        pitcher_id, season, n_starts, raw_mean_bf, raw_std_bf,
        mu_bf, sigma_bf, reliability.
    """
    # Filter to starters with BF >= 3 (minimum real appearance)
    starters = game_logs[
        (game_logs["is_starter"] == True)  # noqa: E712
        & (game_logs["batters_faced"] >= 3)
    ].copy()

    if starters.empty:
        logger.warning("No starter game logs found")
        return pd.DataFrame(columns=[
            "pitcher_id", "season", "n_starts", "raw_mean_bf", "raw_std_bf",
            "mu_bf", "sigma_bf", "reliability",
        ])

    # Per-pitcher-season aggregation
    agg = starters.groupby(["pitcher_id", "season"]).agg(
        n_starts=("batters_faced", "count"),
        raw_mean_bf=("batters_faced", "mean"),
        raw_std_bf=("batters_faced", "std"),
    ).reset_index()

    # Fill NaN std (single-start pitchers) with pop within-pitcher std
    agg["raw_std_bf"] = agg["raw_std_bf"].fillna(pop_within_std)

    # Shrinkage
    agg["reliability"] = agg["n_starts"] / (agg["n_starts"] + shrinkage_k)

    # Below min_starts: reliability = 0 (pure population prior)
    below_min = agg["n_starts"] < min_starts
    agg.loc[below_min, "reliability"] = 0.0

    agg["mu_bf"] = (
        agg["reliability"] * agg["raw_mean_bf"]
        + (1 - agg["reliability"]) * pop_mu
    )
    agg["sigma_bf"] = (
        agg["reliability"] * agg["raw_std_bf"]
        + (1 - agg["reliability"]) * pop_within_std
    )

    logger.info(
        "BF priors: %d pitcher-seasons, mean reliability=%.3f",
        len(agg), agg["reliability"].mean(),
    )
    return agg


def get_bf_distribution(
    pitcher_id: int,
    season: int,
    bf_priors: pd.DataFrame,
    pop_mu: float = DEFAULT_POP_BF_MU,
    pop_within_std: float = DEFAULT_POP_WITHIN_STD,
) -> dict[str, Any]:
    """Look up BF distribution parameters for a pitcher.

    Parameters
    ----------
    pitcher_id : int
        MLB pitcher ID.
    season : int
        Season to look up.
    bf_priors : pd.DataFrame
        Output of ``compute_pitcher_bf_priors``.
    pop_mu : float
        Fallback population mean.
    pop_within_std : float
        Fallback population std.

    Returns
    -------
    dict
        Keys: mu_bf, sigma_bf, reliability, dist_type.
    """
    mask = (bf_priors["pitcher_id"] == pitcher_id) & (bf_priors["season"] == season)
    rows = bf_priors[mask]

    if rows.empty:
        return {
            "mu_bf": pop_mu,
            "sigma_bf": pop_within_std,
            "reliability": 0.0,
            "dist_type": "population_fallback",
        }

    row = rows.iloc[0]
    return {
        "mu_bf": float(row["mu_bf"]),
        "sigma_bf": float(row["sigma_bf"]),
        "reliability": float(row["reliability"]),
        "dist_type": "shrinkage",
    }


def draw_bf_samples(
    mu_bf: float,
    sigma_bf: float,
    n_draws: int,
    bf_min: int = 3,
    bf_max: int = 35,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """Draw integer BF samples from a truncated normal.

    Parameters
    ----------
    mu_bf : float
        Mean BF.
    sigma_bf : float
        Std BF.
    n_draws : int
        Number of samples to draw.
    bf_min : int
        Minimum BF (inclusive).
    bf_max : int
        Maximum BF (inclusive).
    rng : numpy Generator, optional
        For reproducibility. If None, uses default.

    Returns
    -------
    np.ndarray
        Integer BF values, shape (n_draws,).
    """
    if sigma_bf <= 0:
        return np.full(n_draws, int(np.clip(np.round(mu_bf), bf_min, bf_max)), dtype=int)

    a = (bf_min - mu_bf) / sigma_bf
    b = (bf_max - mu_bf) / sigma_bf

    if rng is None:
        rng = np.random.default_rng()

    # Use scipy truncnorm with numpy rng
    samples = stats.truncnorm.rvs(
        a, b, loc=mu_bf, scale=sigma_bf, size=n_draws,
        random_state=rng.integers(0, 2**31),
    )

    return np.clip(np.round(samples), bf_min, bf_max).astype(int)
