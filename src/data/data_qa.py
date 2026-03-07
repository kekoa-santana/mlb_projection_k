"""
Data QA sanity reports for the MLB projection database.

Runs a suite of checks against the production schema and returns structured
DataFrames summarising findings.  All checks are read-only SQL queries.

Checks
------
1. row_counts       – pitches / PAs / batted-balls / games per season
2. denominators     – pitches-per-PA and PA-per-game consistency
3. nan_prevalence   – IEEE NaN float counts in fact_pitch and sat_batted_balls
4. bb_coverage      – sat_batted_balls completeness vs is_bip in fact_pitch
5. player_orphans   – batters in fact_pitch with no dim_player record
6. pitch_type_dist  – pitch type volumes and NULL share
7. anomaly_flags    – extreme velocity / movement values per season

Usage
-----
    from src.data.data_qa import run_all_qa
    report = run_all_qa()          # returns dict of DataFrames
    report["bb_coverage"]          # inspect a specific check
"""
from __future__ import annotations

import logging
from typing import Any

import pandas as pd

from src.data.db import read_sql

logger = logging.getLogger(__name__)

# Thresholds used to flag anomalies
_WARN_THRESHOLDS: dict[str, float] = {
    "null_pitch_type_pct": 1.0,      # > 1% null pitch_type is suspicious
    "nan_speed_pct": 1.0,
    "nan_spin_pct": 2.0,             # spin has more legitimate NaN
    "nan_plate_z_pct": 3.0,
    "bb_coverage_min_pct": 50.0,     # seasons below 50% batted-ball coverage
    "pitches_per_pa_lo": 3.5,
    "pitches_per_pa_hi": 4.5,
    "pa_per_game_lo": 68.0,
    "pa_per_game_hi": 82.0,
}


# ---------------------------------------------------------------------------
# 1. Row counts per season
# ---------------------------------------------------------------------------
def check_row_counts() -> pd.DataFrame:
    """Row counts for main tables per regular-season year.

    Uses separate sub-queries per table to avoid expensive multi-table
    cross-joins on the large fact tables.

    Returns
    -------
    pd.DataFrame
        Columns: season, games, plate_appearances, pitches, batted_balls.
    """
    query = """
    WITH game_counts AS (
        SELECT season, COUNT(*) AS games
        FROM production.dim_game
        WHERE game_type = 'R'
        GROUP BY season
    ),
    pitch_counts AS (
        SELECT dg.season, COUNT(*) AS pitches
        FROM production.fact_pitch fp
        JOIN production.dim_game dg ON fp.game_pk = dg.game_pk
        WHERE dg.game_type = 'R'
        GROUP BY dg.season
    ),
    pa_counts AS (
        SELECT dg.season, COUNT(*) AS plate_appearances
        FROM production.fact_pa fpa
        JOIN production.dim_game dg ON fpa.game_pk = dg.game_pk
        WHERE dg.game_type = 'R'
        GROUP BY dg.season
    ),
    bb_counts AS (
        SELECT dg.season, COUNT(*) AS batted_balls
        FROM production.sat_batted_balls sbb
        JOIN production.fact_pitch fp ON sbb.pitch_id = fp.pitch_id
        JOIN production.dim_game dg  ON fp.game_pk   = dg.game_pk
        WHERE dg.game_type = 'R'
        GROUP BY dg.season
    )
    SELECT
        gc.season,
        gc.games,
        COALESCE(pac.plate_appearances, 0) AS plate_appearances,
        COALESCE(pc.pitches, 0)            AS pitches,
        COALESCE(bc.batted_balls, 0)       AS batted_balls
    FROM game_counts gc
    LEFT JOIN pitch_counts  pc  ON gc.season = pc.season
    LEFT JOIN pa_counts     pac ON gc.season = pac.season
    LEFT JOIN bb_counts     bc  ON gc.season = bc.season
    ORDER BY gc.season
    """
    df = read_sql(query)
    logger.info("Row counts:\n%s", df.to_string(index=False))
    return df


# ---------------------------------------------------------------------------
# 2. Denominator consistency
# ---------------------------------------------------------------------------
def check_denominators() -> pd.DataFrame:
    """Pitches-per-PA and PA-per-game ratios by season.

    Both metrics should be stable across seasons (3.85–4.10 and 73–80
    respectively).  Outliers indicate data loading issues.

    Returns
    -------
    pd.DataFrame
        Columns: season, total_pa, total_pitches, pitches_per_pa,
        pa_per_game, pitches_per_pa_ok, pa_per_game_ok.
    """
    query = """
    SELECT
        dg.season,
        COUNT(DISTINCT fpa.pa_id)   AS total_pa,
        COUNT(fp.pitch_id)          AS total_pitches,
        ROUND(COUNT(fp.pitch_id)::numeric
              / NULLIF(COUNT(DISTINCT fpa.pa_id), 0), 2) AS pitches_per_pa,
        ROUND(COUNT(DISTINCT fpa.pa_id)::numeric
              / NULLIF(COUNT(DISTINCT fpa.game_pk), 0), 1) AS pa_per_game
    FROM production.dim_game dg
    JOIN production.fact_pa fpa   ON dg.game_pk = fpa.game_pk
    JOIN production.fact_pitch fp ON fpa.pa_id  = fp.pa_id
    WHERE dg.game_type = 'R'
      AND fpa.events IS NOT NULL
    GROUP BY dg.season
    ORDER BY dg.season
    """
    df = read_sql(query)

    lo_p = _WARN_THRESHOLDS["pitches_per_pa_lo"]
    hi_p = _WARN_THRESHOLDS["pitches_per_pa_hi"]
    lo_g = _WARN_THRESHOLDS["pa_per_game_lo"]
    hi_g = _WARN_THRESHOLDS["pa_per_game_hi"]

    df["pitches_per_pa_ok"] = df["pitches_per_pa"].between(lo_p, hi_p)
    df["pa_per_game_ok"]    = df["pa_per_game"].between(lo_g, hi_g)

    bad = df[~(df["pitches_per_pa_ok"] & df["pa_per_game_ok"])]
    if len(bad):
        logger.warning("Denominator anomalies detected:\n%s", bad.to_string(index=False))
    else:
        logger.info("Denominators: all seasons within expected ranges")

    return df


# ---------------------------------------------------------------------------
# 3. IEEE NaN prevalence in fact_pitch
# ---------------------------------------------------------------------------
def check_nan_prevalence() -> pd.DataFrame:
    """Count of IEEE NaN float values in key numeric columns of fact_pitch.

    PostgreSQL stores IEEE NaN as a valid float literal ('NaN') distinct
    from NULL.  Standard AVG / MIN / MAX propagate NaN incorrectly —
    always use CASE WHEN col != 'NaN' guards.

    Returns
    -------
    pd.DataFrame
        One row per column with nan_count, total, nan_pct, flag.
    """
    query = """
    SELECT
        COUNT(*) AS total,
        SUM(CASE WHEN release_speed     = 'NaN' THEN 1 ELSE 0 END) AS nan_speed,
        SUM(CASE WHEN pfx_x             = 'NaN' THEN 1 ELSE 0 END) AS nan_pfx_x,
        SUM(CASE WHEN pfx_z             = 'NaN' THEN 1 ELSE 0 END) AS nan_pfx_z,
        SUM(CASE WHEN release_spin_rate = 'NaN' THEN 1 ELSE 0 END) AS nan_spin,
        SUM(CASE WHEN plate_x           = 'NaN' THEN 1 ELSE 0 END) AS nan_plate_x,
        SUM(CASE WHEN plate_z           = 'NaN' THEN 1 ELSE 0 END) AS nan_plate_z,
        SUM(CASE WHEN pitch_type IS NULL         THEN 1 ELSE 0 END) AS null_pitch_type,
        SUM(CASE WHEN zone   IS NULL             THEN 1 ELSE 0 END) AS null_zone
    FROM production.fact_pitch
    """
    raw = read_sql(query).iloc[0]
    total = int(raw["total"])

    columns = [
        ("release_speed",     "nan_speed",       _WARN_THRESHOLDS["nan_speed_pct"]),
        ("pfx_x",             "nan_pfx_x",       _WARN_THRESHOLDS["nan_speed_pct"]),
        ("pfx_z",             "nan_pfx_z",       _WARN_THRESHOLDS["nan_speed_pct"]),
        ("release_spin_rate", "nan_spin",        _WARN_THRESHOLDS["nan_spin_pct"]),
        ("plate_x",           "nan_plate_x",     _WARN_THRESHOLDS["nan_speed_pct"]),
        ("plate_z",           "nan_plate_z",     _WARN_THRESHOLDS["nan_plate_z_pct"]),
        ("pitch_type (NULL)", "null_pitch_type", _WARN_THRESHOLDS["null_pitch_type_pct"]),
        ("zone (NULL)",       "null_zone",       _WARN_THRESHOLDS["null_pitch_type_pct"]),
    ]

    records = []
    for col_name, key, threshold in columns:
        count = int(raw[key])
        pct   = count / total * 100
        records.append({
            "column":    col_name,
            "nan_count": count,
            "total":     total,
            "nan_pct":   round(pct, 3),
            "flag":      pct > threshold,
        })

    df = pd.DataFrame(records)
    flagged = df[df["flag"]]
    if len(flagged):
        logger.warning("NaN flags:\n%s", flagged[["column", "nan_pct"]].to_string(index=False))
    else:
        logger.info("NaN prevalence: no columns exceed thresholds")

    return df


# ---------------------------------------------------------------------------
# 4. Batted ball coverage per season
# ---------------------------------------------------------------------------
def check_bb_coverage() -> pd.DataFrame:
    """What fraction of is_bip pitches have a sat_batted_balls record?

    Coverage increases from ~21 %% (2018) to ~89 %% (2025) due to gradual
    ETL back-fill.  xwOBA / barrel metrics are unreliable for pre-2022
    seasons.

    Returns
    -------
    pd.DataFrame
        Columns: season, bip_count, bb_rows, coverage_pct, coverage_ok.
    """
    query = """
    SELECT
        dg.season,
        SUM(fp.is_bip::int)            AS bip_count,
        COUNT(sbb.pitch_id)            AS bb_rows,
        ROUND(COUNT(sbb.pitch_id)::numeric
              / NULLIF(SUM(fp.is_bip::int), 0) * 100, 1) AS coverage_pct
    FROM production.dim_game dg
    JOIN production.fact_pitch fp
          ON dg.game_pk = fp.game_pk
    LEFT JOIN production.sat_batted_balls sbb
          ON fp.pitch_id = sbb.pitch_id
    WHERE dg.game_type = 'R'
    GROUP BY dg.season
    ORDER BY dg.season
    """
    df = read_sql(query)
    threshold = _WARN_THRESHOLDS["bb_coverage_min_pct"]
    df["coverage_ok"] = df["coverage_pct"] >= threshold

    low = df[~df["coverage_ok"]]
    if len(low):
        logger.warning(
            "Low batted-ball coverage (< %.0f%%) in seasons: %s",
            threshold, low["season"].tolist(),
        )
    return df


# ---------------------------------------------------------------------------
# 5. Player orphans
# ---------------------------------------------------------------------------
def check_player_orphans() -> pd.DataFrame:
    """Batters in fact_pitch with no matching dim_player record, by season.

    These are mostly pitchers batting (pre-universal DH) and minor-league
    call-ups whose records weren't captured in dim_player.  They are
    excluded from season_totals via the JOIN to dim_player but remain in
    pitch-level profiles.  The pitch volume is negligible for model
    purposes.

    Returns
    -------
    pd.DataFrame
        Columns: season, orphaned_batters, orphaned_pitches, pct_of_pitches.
    """
    query = """
    WITH totals AS (
        SELECT dg.season, COUNT(*) AS total_pitches
        FROM production.fact_pitch fp
        JOIN production.dim_game dg ON fp.game_pk = dg.game_pk
        WHERE dg.game_type = 'R'
        GROUP BY dg.season
    ),
    orphans AS (
        SELECT dg.season,
               COUNT(DISTINCT fp.batter_id) AS orphaned_batters,
               COUNT(*)                     AS orphaned_pitches
        FROM production.fact_pitch fp
        JOIN production.dim_game dg ON fp.game_pk = dg.game_pk
        LEFT JOIN production.dim_player dp ON fp.batter_id = dp.player_id
        WHERE dp.player_id IS NULL
          AND dg.game_type  = 'R'
        GROUP BY dg.season
    )
    SELECT
        t.season,
        COALESCE(o.orphaned_batters, 0) AS orphaned_batters,
        COALESCE(o.orphaned_pitches, 0) AS orphaned_pitches,
        ROUND(COALESCE(o.orphaned_pitches, 0)::numeric
              / NULLIF(t.total_pitches, 0) * 100, 2)  AS pct_of_pitches
    FROM totals t
    LEFT JOIN orphans o ON t.season = o.season
    ORDER BY t.season
    """
    df = read_sql(query)
    total_orphans = int(df["orphaned_batters"].sum())
    logger.info(
        "Player orphans: %d distinct batters across all seasons "
        "(mostly pitchers batting / pre-DH era)",
        total_orphans,
    )
    return df


# ---------------------------------------------------------------------------
# 6. Pitch type distribution
# ---------------------------------------------------------------------------
def check_pitch_type_dist() -> pd.DataFrame:
    """Pitch type volumes (all seasons, regular season only).

    Returns
    -------
    pd.DataFrame
        Columns: pitch_type, pitches, pct, included (whether it passes
        the EXCLUDED_PITCH_TYPES filter in constants.py).
    """
    from src.utils.constants import EXCLUDED_PITCH_TYPES  # avoid circular import

    query = """
    SELECT
        COALESCE(fp.pitch_type, '(NULL)') AS pitch_type,
        COUNT(*) AS pitches
    FROM production.fact_pitch fp
    JOIN production.dim_game dg ON fp.game_pk = dg.game_pk
    WHERE dg.game_type = 'R'
    GROUP BY fp.pitch_type
    ORDER BY pitches DESC
    """
    df = read_sql(query)
    total = df["pitches"].sum()
    df["pct"] = (df["pitches"] / total * 100).round(2)
    df["included"] = ~df["pitch_type"].isin(EXCLUDED_PITCH_TYPES | {"(NULL)"})
    return df


# ---------------------------------------------------------------------------
# 7. Anomaly flags (pitch shape extremes)
# ---------------------------------------------------------------------------
def check_anomaly_flags() -> pd.DataFrame:
    """Flag extreme velocity values per season.

    Pitches < 40 mph or > 105 mph are almost certainly data errors.

    Returns
    -------
    pd.DataFrame
        Columns: season, pitches_typed, avg_velo, min_velo, max_velo,
        under_40, over_105.
    """
    query = """
    SELECT
        dg.season,
        COUNT(*)                           AS pitches_typed,
        ROUND(AVG(CASE WHEN fp.release_speed != 'NaN'
                       THEN fp.release_speed END)::numeric, 1) AS avg_velo,
        MIN(CASE WHEN fp.release_speed != 'NaN'
                 THEN fp.release_speed END)                    AS min_velo,
        MAX(CASE WHEN fp.release_speed != 'NaN'
                 THEN fp.release_speed END)                    AS max_velo,
        SUM(CASE WHEN fp.release_speed != 'NaN'
                      AND fp.release_speed < 40
                 THEN 1 ELSE 0 END)        AS under_40,
        SUM(CASE WHEN fp.release_speed != 'NaN'
                      AND fp.release_speed > 105
                 THEN 1 ELSE 0 END)        AS over_105
    FROM production.fact_pitch fp
    JOIN production.dim_game dg ON fp.game_pk = dg.game_pk
    WHERE dg.game_type = 'R'
      AND fp.pitch_type IS NOT NULL
    GROUP BY dg.season
    ORDER BY dg.season
    """
    df = read_sql(query)
    extreme = df[(df["under_40"] > 0) | (df["over_105"] > 0)]
    if len(extreme):
        logger.warning(
            "Extreme velocity rows found in seasons: %s",
            extreme["season"].tolist(),
        )
    return df


# ---------------------------------------------------------------------------
# Master runner
# ---------------------------------------------------------------------------
def run_all_qa(verbose: bool = True) -> dict[str, Any]:
    """Run all QA checks and return a dict of DataFrames.

    Parameters
    ----------
    verbose : bool
        If True, print a human-readable summary to stdout.

    Returns
    -------
    dict
        Keys: "row_counts", "denominators", "nan_prevalence",
              "bb_coverage", "player_orphans", "pitch_type_dist",
              "anomaly_flags", "summary".
    """
    logger.info("=== Running full Data QA suite ===")

    results: dict[str, Any] = {
        "row_counts":    check_row_counts(),
        "denominators":  check_denominators(),
        "nan_prevalence": check_nan_prevalence(),
        "bb_coverage":   check_bb_coverage(),
        "player_orphans": check_player_orphans(),
        "pitch_type_dist": check_pitch_type_dist(),
        "anomaly_flags": check_anomaly_flags(),
    }

    # --- Summary table ---
    bb = results["bb_coverage"]
    nan = results["nan_prevalence"]
    denom = results["denominators"]

    summary_rows = []

    # NaN flags
    for _, row in nan[nan["flag"]].iterrows():
        summary_rows.append({
            "check": "nan_prevalence",
            "severity": "WARNING",
            "detail": f"{row['column']}: {row['nan_pct']:.2f}% NaN/NULL",
        })

    # BB coverage flags
    for _, row in bb[~bb["coverage_ok"]].iterrows():
        summary_rows.append({
            "check": "bb_coverage",
            "severity": "WARNING",
            "detail": (
                f"Season {row['season']}: only {row['coverage_pct']:.1f}% "
                "batted-ball coverage — xwOBA/barrel metrics unreliable"
            ),
        })

    # Denominator flags
    bad_denom = denom[~(denom["pitches_per_pa_ok"] & denom["pa_per_game_ok"])]
    for _, row in bad_denom.iterrows():
        summary_rows.append({
            "check": "denominators",
            "severity": "WARNING",
            "detail": (
                f"Season {row['season']}: pitches/PA={row['pitches_per_pa']}, "
                f"PA/game={row['pa_per_game']}"
            ),
        })

    if not summary_rows:
        summary_rows.append({
            "check": "overall",
            "severity": "OK",
            "detail": "No critical issues detected beyond expected coverage ramp-up",
        })

    results["summary"] = pd.DataFrame(summary_rows)

    if verbose:
        _print_report(results)

    logger.info("=== Data QA complete ===")
    return results


def _print_report(results: dict[str, Any]) -> None:
    """Print a human-readable QA report."""
    sep = "=" * 60

    print(f"\n{sep}")
    print("  DATA QA REPORT")
    print(sep)

    print("\n[1] Row counts per season")
    print(results["row_counts"].to_string(index=False))

    print("\n[2] Denominator consistency")
    print(results["denominators"].to_string(index=False))

    print("\n[3] IEEE NaN prevalence (fact_pitch)")
    print(results["nan_prevalence"].to_string(index=False))

    print("\n[4] Batted-ball coverage (sat_batted_balls / is_bip)")
    print(results["bb_coverage"].to_string(index=False))

    print("\n[5] Player orphans (batter in fact_pitch, missing from dim_player)")
    print(results["player_orphans"].to_string(index=False))

    print("\n[6] Pitch type distribution")
    print(results["pitch_type_dist"].to_string(index=False))

    print("\n[7] Anomaly flags (extreme velocities)")
    print(results["anomaly_flags"].to_string(index=False))

    print(f"\n{sep}")
    print("  SUMMARY")
    print(sep)
    print(results["summary"].to_string(index=False))
    print()
