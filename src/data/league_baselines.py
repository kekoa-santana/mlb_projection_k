"""
League-average baselines v2 — per (pitch_type, batter_stand) and
per (pitch_archetype, batter_stand).

Replaces the simple pitch_type-only baselines in feature_eng.py with
platoon-split-aware rates. Supports the Layer 2 matchup model and
improved Bayesian priors.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

from src.data.pitch_archetypes import get_pitch_archetype_offerings
from src.data.queries import get_pitcher_outcomes_by_stand

logger = logging.getLogger(__name__)

CACHE_DIR = Path(__file__).resolve().parents[2] / "data" / "cached"
CONFIG_DIR = Path(__file__).resolve().parents[2] / "config"

# Rate columns produced by _aggregate_baselines
RATE_COLS = (
    "whiff_rate",
    "chase_rate",
    "csw_pct",
    "barrel_rate",
    "hard_hit_rate",
    "xwoba_contact",
)

# Raw count columns that get summed before computing rates
COUNT_COLS = (
    "pitches",
    "swings",
    "whiffs",
    "out_of_zone_pitches",
    "chase_swings",
    "called_strikes",
    "csw",
    "bip",
    "hard_hits",
    "barrels_proxy",
    "xwoba_contact_sum",
    "xwoba_contact_n",
)


# ---------------------------------------------------------------------------
# Config helper
# ---------------------------------------------------------------------------
def _get_train_seasons() -> list[int]:
    """Read ``seasons.train`` from ``config/model.yaml``."""
    path = CONFIG_DIR / "model.yaml"
    with open(path, "r") as f:
        cfg = yaml.safe_load(f)
    return cfg["seasons"]["train"]


# ---------------------------------------------------------------------------
# Cache helpers
# ---------------------------------------------------------------------------
def _load_or_build(
    cache_path: Path,
    builder: callable,
    force_rebuild: bool = False,
) -> pd.DataFrame:
    """Load from Parquet cache if available, otherwise build and cache."""
    if cache_path.exists() and not force_rebuild:
        logger.info("Loading cached %s", cache_path.name)
        return pd.read_parquet(cache_path)

    logger.info("Building %s", cache_path.name)
    df = builder()
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    df.to_parquet(cache_path, index=False)
    logger.info("Cached %s (%d rows)", cache_path.name, len(df))
    return df


# ---------------------------------------------------------------------------
# Generic aggregation
# ---------------------------------------------------------------------------
def _aggregate_baselines(df: pd.DataFrame, group_cols: list[str]) -> pd.DataFrame:
    """Group by ``group_cols``, sum raw counts, then compute rate columns.

    All divisions use ``.replace(0, np.nan)`` to avoid division-by-zero.
    """
    agg = df.groupby(group_cols, dropna=False)[list(COUNT_COLS)].sum().reset_index()

    agg["whiff_rate"] = agg["whiffs"] / agg["swings"].replace(0, np.nan)
    agg["chase_rate"] = agg["chase_swings"] / agg["out_of_zone_pitches"].replace(0, np.nan)
    agg["csw_pct"] = agg["csw"] / agg["pitches"].replace(0, np.nan)
    agg["barrel_rate"] = agg["barrels_proxy"] / agg["bip"].replace(0, np.nan)
    agg["hard_hit_rate"] = agg["hard_hits"] / agg["bip"].replace(0, np.nan)
    agg["xwoba_contact"] = agg["xwoba_contact_sum"] / agg["xwoba_contact_n"].replace(0, np.nan)

    return agg


# ---------------------------------------------------------------------------
# Intermediate: outcomes joined with archetypes
# ---------------------------------------------------------------------------
def _build_outcomes_with_archetypes(
    season: int,
    n_clusters: int = 8,
    force_rebuild: bool = False,
) -> pd.DataFrame:
    """Join SQL pitcher outcomes with archetype offerings.

    LEFT join so un-matched rows (missing shape data) still contribute
    to pitch_type baselines — they just have pitch_archetype = NaN.
    """
    cache_path = CACHE_DIR / f"outcomes_arch_{season}_k{n_clusters}.parquet"

    def _build() -> pd.DataFrame:
        outcomes = get_pitcher_outcomes_by_stand(season)
        archetypes = get_pitch_archetype_offerings(season, n_clusters=n_clusters)

        # Keep only the columns we need from archetypes
        arch_cols = archetypes[["pitcher_id", "pitch_type", "pitch_archetype"]].copy()
        # De-duplicate in case archetypes has multiple rows per (pitcher_id, pitch_type)
        arch_cols = arch_cols.drop_duplicates(subset=["pitcher_id", "pitch_type"])

        merged = outcomes.merge(
            arch_cols,
            on=["pitcher_id", "pitch_type"],
            how="left",
        )
        merged["season"] = season
        return merged

    return _load_or_build(cache_path, _build, force_rebuild)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------
def get_baselines_by_pitch_type_stand(
    season: int,
    n_clusters: int = 8,
    force_rebuild: bool = False,
) -> pd.DataFrame:
    """League baselines per (pitch_type, batter_stand) for one season.

    Returns ~24 rows (12 pitch types x 2 stands).
    """
    cache_path = CACHE_DIR / f"league_baselines_pt_stand_{season}.parquet"

    def _build() -> pd.DataFrame:
        df = _build_outcomes_with_archetypes(season, n_clusters, force_rebuild)
        agg = _aggregate_baselines(df, ["pitch_type", "batter_stand"])
        agg["season"] = season
        return agg

    return _load_or_build(cache_path, _build, force_rebuild)


def get_baselines_by_archetype_stand(
    season: int,
    n_clusters: int = 8,
    force_rebuild: bool = False,
) -> pd.DataFrame:
    """League baselines per (pitch_archetype, batter_stand) for one season.

    Returns ~16 rows (8 archetypes x 2 stands).
    """
    cache_path = CACHE_DIR / f"league_baselines_arch_stand_{season}_k{n_clusters}.parquet"

    def _build() -> pd.DataFrame:
        df = _build_outcomes_with_archetypes(season, n_clusters, force_rebuild)
        # Drop rows without an archetype assignment for this grouping
        df_arch = df.dropna(subset=["pitch_archetype"])
        agg = _aggregate_baselines(df_arch, ["pitch_archetype", "batter_stand"])
        agg["season"] = season
        return agg

    return _load_or_build(cache_path, _build, force_rebuild)


def get_pooled_baselines(
    seasons: list[int] | None = None,
    grouping: str = "archetype_stand",
    n_clusters: int = 8,
    force_rebuild: bool = False,
) -> pd.DataFrame:
    """Pool raw counts across seasons THEN divide to compute rates.

    This correctly weights short seasons (e.g. 2020) by volume rather
    than giving each season equal weight.

    Parameters
    ----------
    seasons : list[int] | None
        Seasons to pool.  Defaults to ``model.yaml`` ``seasons.train``.
    grouping : str
        One of: "pitch_type_stand", "archetype_stand", "pitch_type",
        "archetype", "overall".
    n_clusters : int
        Number of archetype clusters (only used for archetype groupings).
    force_rebuild : bool
        If True, rebuild from scratch.
    """
    valid_groupings = {
        "pitch_type_stand",
        "archetype_stand",
        "pitch_type",
        "archetype",
        "overall",
    }
    if grouping not in valid_groupings:
        raise ValueError(f"grouping must be one of {valid_groupings}, got '{grouping}'")

    if seasons is None:
        seasons = _get_train_seasons()

    season_min, season_max = min(seasons), max(seasons)
    cache_path = CACHE_DIR / f"league_baselines_pooled_{grouping}_{season_min}_{season_max}.parquet"

    def _build() -> pd.DataFrame:
        frames = []
        for s in seasons:
            df = _build_outcomes_with_archetypes(s, n_clusters, force_rebuild=False)
            frames.append(df)
        pooled = pd.concat(frames, ignore_index=True)

        group_map = {
            "pitch_type_stand": ["pitch_type", "batter_stand"],
            "archetype_stand": ["pitch_archetype", "batter_stand"],
            "pitch_type": ["pitch_type"],
            "archetype": ["pitch_archetype"],
            "overall": [],
        }
        group_cols = group_map[grouping]

        if "archetype" in grouping:
            pooled = pooled.dropna(subset=["pitch_archetype"])

        if group_cols:
            agg = _aggregate_baselines(pooled, group_cols)
        else:
            # Overall: single-row aggregate
            sums = pooled[list(COUNT_COLS)].sum()
            agg = pd.DataFrame([sums])
            agg["whiff_rate"] = agg["whiffs"] / agg["swings"].replace(0, np.nan)
            agg["chase_rate"] = agg["chase_swings"] / agg["out_of_zone_pitches"].replace(0, np.nan)
            agg["csw_pct"] = agg["csw"] / agg["pitches"].replace(0, np.nan)
            agg["barrel_rate"] = agg["barrels_proxy"] / agg["bip"].replace(0, np.nan)
            agg["hard_hit_rate"] = agg["hard_hits"] / agg["bip"].replace(0, np.nan)
            agg["xwoba_contact"] = agg["xwoba_contact_sum"] / agg["xwoba_contact_n"].replace(0, np.nan)

        agg["season_min"] = season_min
        agg["season_max"] = season_max
        return agg

    return _load_or_build(cache_path, _build, force_rebuild)


def get_prior_baselines(
    pitch_type: str | None = None,
    pitch_archetype: int | None = None,
    batter_stand: str | None = None,
    seasons: list[int] | None = None,
    n_clusters: int = 8,
) -> dict[str, float]:
    """Convenience lookup returning a dict of baseline rates.

    Fallback chain: archetype+stand -> pitch_type+stand -> pitch_type -> overall.

    Returns
    -------
    dict
        Keys: whiff_rate, chase_rate, csw_pct, barrel_rate, hard_hit_rate,
        xwoba_contact.
    """
    def _row_to_dict(row: pd.Series) -> dict[str, float]:
        return {col: row[col] for col in RATE_COLS}

    # Try archetype + stand
    if pitch_archetype is not None and batter_stand is not None:
        df = get_pooled_baselines(seasons, "archetype_stand", n_clusters)
        match = df[
            (df["pitch_archetype"] == pitch_archetype)
            & (df["batter_stand"] == batter_stand)
        ]
        if len(match) == 1:
            return _row_to_dict(match.iloc[0])

    # Try pitch_type + stand
    if pitch_type is not None and batter_stand is not None:
        df = get_pooled_baselines(seasons, "pitch_type_stand", n_clusters)
        match = df[
            (df["pitch_type"] == pitch_type)
            & (df["batter_stand"] == batter_stand)
        ]
        if len(match) == 1:
            return _row_to_dict(match.iloc[0])

    # Try pitch_type only
    if pitch_type is not None:
        df = get_pooled_baselines(seasons, "pitch_type", n_clusters)
        match = df[df["pitch_type"] == pitch_type]
        if len(match) == 1:
            return _row_to_dict(match.iloc[0])

    # Fallback: overall
    df = get_pooled_baselines(seasons, "overall", n_clusters)
    return _row_to_dict(df.iloc[0])
