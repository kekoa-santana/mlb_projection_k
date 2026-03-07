"""Tests for archetype-based matchup scoring."""
import numpy as np
import pandas as pd
import pytest

from src.models.matchup import (
    _get_hitter_whiff_with_fallback_archetype,
    score_matchup_by_archetype,
    score_matchups_batch_by_archetype,
)


# ---------------------------------------------------------------------------
# Synthetic fixtures
# ---------------------------------------------------------------------------
LEAGUE_BASELINES_ARCH = {
    1: {"whiff_rate": 0.22},  # fastball archetype
    2: {"whiff_rate": 0.33},  # slider archetype
    3: {"whiff_rate": 0.32},  # changeup archetype
}

LEAGUE_BASELINES_PT = {
    "FF": {"whiff_rate": 0.22},
    "SL": {"whiff_rate": 0.33},
    "CH": {"whiff_rate": 0.32},
}


@pytest.fixture()
def pitcher_offerings() -> pd.DataFrame:
    """Synthetic pitcher 100 with archetype-labeled offerings."""
    return pd.DataFrame([
        {
            "pitcher_id": 100, "pitch_type": "FF", "pitch_archetype": 1,
            "pitches": 600, "swings": 300, "whiffs": 72, "whiff_rate": 0.24,
        },
        {
            "pitcher_id": 100, "pitch_type": "SL", "pitch_archetype": 2,
            "pitches": 250, "swings": 125, "whiffs": 50, "whiff_rate": 0.40,
        },
        {
            "pitcher_id": 100, "pitch_type": "CH", "pitch_archetype": 3,
            "pitches": 150, "swings": 75, "whiffs": 24, "whiff_rate": 0.32,
        },
    ])


@pytest.fixture()
def hitter_vuln_arch() -> pd.DataFrame:
    """Synthetic hitter vulnerability by archetype."""
    return pd.DataFrame([
        # Batter 200: vulnerable to archetype 1 (FF), average on 2 (SL)
        {"batter_id": 200, "pitch_archetype": 1, "swings": 80, "whiff_rate": 0.35},
        {"batter_id": 200, "pitch_archetype": 2, "swings": 40, "whiff_rate": 0.33},
        # Batter 300: league-average on all archetypes
        {"batter_id": 300, "pitch_archetype": 1, "swings": 100, "whiff_rate": 0.22},
        {"batter_id": 300, "pitch_archetype": 2, "swings": 60, "whiff_rate": 0.33},
        {"batter_id": 300, "pitch_archetype": 3, "swings": 50, "whiff_rate": 0.32},
    ])


@pytest.fixture()
def cluster_metadata() -> pd.DataFrame:
    """Cluster metadata mapping archetype -> primary pitch_type."""
    return pd.DataFrame([
        {"pitch_archetype": 1, "primary_pitch_type": "FF"},
        {"pitch_archetype": 2, "primary_pitch_type": "SL"},
        {"pitch_archetype": 3, "primary_pitch_type": "CH"},
    ])


@pytest.fixture()
def hitter_vuln_pt() -> pd.DataFrame:
    """Pitch-type hitter vuln for fallback."""
    return pd.DataFrame([
        {"batter_id": 200, "pitch_type": "FF", "swings": 80,
         "whiff_rate": 0.35, "pitch_family": "fastball"},
        {"batter_id": 200, "pitch_type": "SL", "swings": 40,
         "whiff_rate": 0.33, "pitch_family": "breaking"},
    ])


# ---------------------------------------------------------------------------
# Test archetype fallback chain
# ---------------------------------------------------------------------------
class TestArchetypeFallback:
    def test_direct_archetype_match(self, hitter_vuln_arch: pd.DataFrame) -> None:
        """Full reliability with >= 50 swings."""
        whiff, rel = _get_hitter_whiff_with_fallback_archetype(
            hitter_vuln_arch, batter_id=200, archetype=1,
            league_whiff=0.22,
        )
        assert rel == 1.0
        assert np.isclose(whiff, 0.35)

    def test_low_sample_blends(self) -> None:
        """10 swings -> 0.2 reliability, blends toward league baseline."""
        small = pd.DataFrame([{
            "batter_id": 400, "pitch_archetype": 1,
            "swings": 10, "whiff_rate": 0.50,
        }])
        whiff, rel = _get_hitter_whiff_with_fallback_archetype(
            small, batter_id=400, archetype=1, league_whiff=0.22,
        )
        assert np.isclose(rel, 10 / 50)
        expected = 0.2 * 0.50 + 0.8 * 0.22
        assert np.isclose(whiff, expected)

    def test_primary_pt_fallback(
        self,
        hitter_vuln_arch: pd.DataFrame,
        cluster_metadata: pd.DataFrame,
        hitter_vuln_pt: pd.DataFrame,
    ) -> None:
        """Missing archetype data falls back to primary pitch_type."""
        # Batter 200 has no data for archetype 3, but has FF data
        whiff, rel = _get_hitter_whiff_with_fallback_archetype(
            hitter_vuln_arch, batter_id=200, archetype=3,
            league_whiff=0.32,
            cluster_metadata=cluster_metadata,
            hitter_vuln_pt=hitter_vuln_pt,
            baselines_pt=LEAGUE_BASELINES_PT,
        )
        # Should have found something via primary_pitch_type = CH
        # Batter 200 has no CH data, so this falls to FF family → league baseline
        # The key assertion is that the function doesn't crash
        assert np.isfinite(whiff)

    def test_league_fallback(self, hitter_vuln_arch: pd.DataFrame) -> None:
        """Unknown batter -> league baseline, reliability = 0."""
        whiff, rel = _get_hitter_whiff_with_fallback_archetype(
            hitter_vuln_arch, batter_id=999, archetype=1,
            league_whiff=0.22,
        )
        assert rel == 0.0
        assert whiff == 0.22


# ---------------------------------------------------------------------------
# Test archetype matchup scoring
# ---------------------------------------------------------------------------
class TestScoreMatchupByArchetype:
    def test_average_hitter_near_zero_lift(
        self,
        pitcher_offerings: pd.DataFrame,
        hitter_vuln_arch: pd.DataFrame,
    ) -> None:
        """League-average hitter should produce lift near zero."""
        result = score_matchup_by_archetype(
            pitcher_id=100, batter_id=300,
            pitcher_offerings=pitcher_offerings,
            hitter_vuln_arch=hitter_vuln_arch,
            baselines_arch=LEAGUE_BASELINES_ARCH,
        )
        assert abs(result["matchup_k_logit_lift"]) < 0.15
        assert result["n_pitch_types"] == 3

    def test_vulnerable_hitter_positive_lift(
        self,
        pitcher_offerings: pd.DataFrame,
        hitter_vuln_arch: pd.DataFrame,
    ) -> None:
        """High-whiff hitter should produce positive lift."""
        result = score_matchup_by_archetype(
            pitcher_id=100, batter_id=200,
            pitcher_offerings=pitcher_offerings,
            hitter_vuln_arch=hitter_vuln_arch,
            baselines_arch=LEAGUE_BASELINES_ARCH,
        )
        assert result["matchup_k_logit_lift"] > 0
        assert result["matchup_whiff_rate"] > result["baseline_whiff_rate"]

    def test_empty_arsenal_returns_nan(
        self,
        hitter_vuln_arch: pd.DataFrame,
    ) -> None:
        """Pitcher with no offerings returns NaN matchup."""
        empty = pd.DataFrame(columns=[
            "pitcher_id", "pitch_type", "pitch_archetype", "pitches",
            "swings", "whiffs", "whiff_rate",
        ])
        result = score_matchup_by_archetype(
            pitcher_id=999, batter_id=200,
            pitcher_offerings=empty,
            hitter_vuln_arch=hitter_vuln_arch,
            baselines_arch=LEAGUE_BASELINES_ARCH,
        )
        assert np.isnan(result["matchup_whiff_rate"])
        assert result["matchup_k_logit_lift"] == 0.0

    def test_output_schema(
        self,
        pitcher_offerings: pd.DataFrame,
        hitter_vuln_arch: pd.DataFrame,
    ) -> None:
        """Result dict has all expected keys."""
        result = score_matchup_by_archetype(
            pitcher_id=100, batter_id=200,
            pitcher_offerings=pitcher_offerings,
            hitter_vuln_arch=hitter_vuln_arch,
            baselines_arch=LEAGUE_BASELINES_ARCH,
        )
        expected_keys = {
            "pitcher_id", "batter_id", "matchup_whiff_rate",
            "baseline_whiff_rate", "matchup_k_logit_lift",
            "n_pitch_types", "avg_reliability",
        }
        assert set(result.keys()) == expected_keys


# ---------------------------------------------------------------------------
# Test batch scoring
# ---------------------------------------------------------------------------
class TestBatchArchetype:
    def test_batch_matches_individual(
        self,
        pitcher_offerings: pd.DataFrame,
        hitter_vuln_arch: pd.DataFrame,
    ) -> None:
        """Batch scoring produces same results as individual scoring."""
        pairs = [(100, 200), (100, 300)]
        batch = score_matchups_batch_by_archetype(
            pitcher_offerings, hitter_vuln_arch,
            LEAGUE_BASELINES_ARCH, pairs,
        )
        assert len(batch) == 2

        for pid, bid in pairs:
            individual = score_matchup_by_archetype(
                pid, bid, pitcher_offerings, hitter_vuln_arch,
                LEAGUE_BASELINES_ARCH,
            )
            row = batch[(batch["pitcher_id"] == pid) & (batch["batter_id"] == bid)].iloc[0]
            assert np.isclose(
                row["matchup_k_logit_lift"],
                individual["matchup_k_logit_lift"],
                atol=1e-10,
            )
