"""Tests for the game K posterior Monte Carlo engine (Step 14)."""
import numpy as np
import pandas as pd
import pytest

from src.models.game_k_model import (
    compute_k_over_probs,
    simulate_game_ks,
)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------
class TestSimulateGameKsOutputShape:
    def test_returns_correct_shape(self) -> None:
        """Returns (n_draws,) array of non-negative ints."""
        k_rate_samples = np.full(4000, 0.25)
        result = simulate_game_ks(
            pitcher_k_rate_samples=k_rate_samples,
            bf_mu=23.0,
            bf_sigma=3.5,
            n_draws=4000,
            random_seed=42,
        )
        assert result.shape == (4000,)
        assert result.dtype in (np.int32, np.int64, int)
        assert result.min() >= 0


class TestSimulateGameKsExpectedValue:
    def test_expected_k_reasonable(self) -> None:
        """K%=0.25, BF~23 → expected K ≈ 5.75 ± 0.5."""
        k_rate_samples = np.full(10000, 0.25)
        result = simulate_game_ks(
            pitcher_k_rate_samples=k_rate_samples,
            bf_mu=23.0,
            bf_sigma=3.5,
            n_draws=10000,
            random_seed=42,
        )
        expected = result.mean()
        # 0.25 * 23 = 5.75, allow ±0.5 for MC noise
        assert abs(expected - 5.75) < 0.5


class TestSimulateGameKsNoMatchup:
    def test_no_matchup_works(self) -> None:
        """lineup_matchup_lifts=None works and returns baseline K."""
        k_rate_samples = np.full(2000, 0.22)
        result = simulate_game_ks(
            pitcher_k_rate_samples=k_rate_samples,
            bf_mu=22.0,
            bf_sigma=4.0,
            lineup_matchup_lifts=None,
            n_draws=2000,
            random_seed=42,
        )
        assert result.shape == (2000,)
        assert result.mean() > 0


class TestSimulateGameKsPositiveLiftIncreasesKs:
    def test_positive_lifts(self) -> None:
        """All positive lifts → higher expected K than baseline."""
        k_rate_samples = np.full(5000, 0.22)

        baseline = simulate_game_ks(
            pitcher_k_rate_samples=k_rate_samples,
            bf_mu=22.0, bf_sigma=4.0,
            lineup_matchup_lifts=None,
            n_draws=5000, random_seed=42,
        )
        boosted = simulate_game_ks(
            pitcher_k_rate_samples=k_rate_samples,
            bf_mu=22.0, bf_sigma=4.0,
            lineup_matchup_lifts=np.full(9, 0.5),  # +0.5 logit lift
            n_draws=5000, random_seed=42,
        )
        assert boosted.mean() > baseline.mean()


class TestComputeKOverProbsMonotonic:
    def test_p_over_decreases(self) -> None:
        """P(over X.5) decreases as X increases."""
        rng = np.random.default_rng(42)
        k_samples = rng.poisson(6.0, size=10000)
        result = compute_k_over_probs(k_samples)

        p_overs = result["p_over"].values
        # Each successive line should have lower or equal P(over)
        for i in range(len(p_overs) - 1):
            assert p_overs[i] >= p_overs[i + 1]

    def test_columns(self) -> None:
        """Output has expected columns."""
        k_samples = np.array([3, 4, 5, 6, 7, 5, 6, 4, 8, 5])
        result = compute_k_over_probs(k_samples)
        expected_cols = {"line", "p_over", "p_under", "expected_k", "std_k"}
        assert expected_cols.issubset(set(result.columns))
        # Default 13 lines (0.5 through 12.5)
        assert len(result) == 13

    def test_custom_lines(self) -> None:
        """Custom lines produce correct number of rows."""
        k_samples = np.array([3, 4, 5, 6, 7])
        result = compute_k_over_probs(k_samples, lines=[4.5, 5.5, 6.5])
        assert len(result) == 3
