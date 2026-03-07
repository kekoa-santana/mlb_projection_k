"""Tests for the generalized hitter projection model."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.models.hitter_model import (
    STAT_CONFIGS,
    build_hitter_model,
    check_convergence,
    extract_posteriors,
    extract_rate_samples,
    fit_hitter_model,
    prepare_hitter_data,
)


def _make_synthetic_data(
    n_players: int = 30,
    n_seasons: int = 3,
    pa_per_season: int = 400,
    seed: int = 42,
) -> pd.DataFrame:
    """Generate synthetic hitter data for testing."""
    rng = np.random.default_rng(seed)
    records = []
    for p in range(n_players):
        true_k_rate = rng.beta(5, 18)   # ~0.22
        true_bb_rate = rng.beta(3, 33)  # ~0.08
        true_hr_rate = rng.beta(1, 29)  # ~0.03
        true_xwoba = rng.normal(0.315, 0.035)
        age_base = rng.integers(22, 36)
        for s in range(n_seasons):
            pa = pa_per_season + rng.integers(-50, 50)
            k = rng.binomial(pa, true_k_rate)
            bb = rng.binomial(pa, true_bb_rate)
            hr = rng.binomial(pa, true_hr_rate)
            age = age_base + s
            if age <= 25:
                age_bucket = 0
            elif age <= 30:
                age_bucket = 1
            else:
                age_bucket = 2
            records.append({
                "batter_id": 100000 + p,
                "batter_name": f"Player, Test{p}",
                "batter_stand": rng.choice(["L", "R"]),
                "season": 2022 + s,
                "age": age,
                "age_bucket": age_bucket,
                "pa": pa,
                "k": k,
                "bb": bb,
                "hits": int(pa * 0.25),
                "hr": hr,
                "xwoba_avg": float(np.clip(true_xwoba + rng.normal(0, 0.02), 0.1, 0.5)),
                "barrel_pct": float(rng.beta(2, 30)),
                "hard_hit_pct": float(rng.beta(8, 16)),
                "k_rate": round(k / pa, 4),
                "bb_rate": round(bb / pa, 4),
                "hr_rate": round(hr / pa, 4),
            })
    return pd.DataFrame(records)


class TestPrepareData:
    """Test data preparation for all stat types."""

    def test_prepare_binomial_stat(self):
        df = _make_synthetic_data(n_players=10, n_seasons=2)
        data = prepare_hitter_data(df, "k_rate")
        assert data["n_players"] == 10
        assert data["n_seasons"] == 2
        assert "trials" in data
        assert "counts" in data
        assert len(data["player_age_bucket"]) == 10
        assert all(b in (0, 1, 2) for b in data["player_age_bucket"])

    def test_prepare_normal_stat(self):
        df = _make_synthetic_data(n_players=10, n_seasons=2)
        data = prepare_hitter_data(df, "xwoba")
        assert "y_obs" in data
        assert "pa_weight" in data
        assert data["stat"] == "xwoba"

    def test_prepare_all_stats(self):
        df = _make_synthetic_data(n_players=10, n_seasons=2)
        for stat in STAT_CONFIGS:
            data = prepare_hitter_data(df, stat)
            assert data["stat"] == stat
            assert data["n_players"] == 10


class TestBuildModel:
    """Test model construction (no sampling)."""

    def test_build_binomial(self):
        df = _make_synthetic_data(n_players=10, n_seasons=2)
        data = prepare_hitter_data(df, "k_rate")
        model = build_hitter_model(data)
        assert "mu_pop" in model.named_vars
        assert "sigma_player" in model.named_vars
        assert "sigma_season" in model.named_vars
        assert "rate" in model.named_vars
        # mu_pop should have shape (3,) for age buckets
        assert model.named_vars["mu_pop"].eval().shape == (3,)

    def test_build_normal(self):
        df = _make_synthetic_data(n_players=10, n_seasons=2)
        data = prepare_hitter_data(df, "xwoba")
        model = build_hitter_model(data)
        assert "sigma_obs" in model.named_vars


class TestFitAndExtract:
    """Test model fitting with minimal draws on synthetic data."""

    @pytest.fixture(scope="class")
    def fitted_k_rate(self):
        df = _make_synthetic_data(n_players=15, n_seasons=2)
        data = prepare_hitter_data(df, "k_rate")
        model, trace = fit_hitter_model(
            data, draws=200, tune=100, chains=2, random_seed=42,
        )
        return data, trace

    @pytest.fixture(scope="class")
    def fitted_bb_rate(self):
        df = _make_synthetic_data(n_players=15, n_seasons=2)
        data = prepare_hitter_data(df, "bb_rate")
        model, trace = fit_hitter_model(
            data, draws=200, tune=100, chains=2, random_seed=42,
        )
        return data, trace

    @pytest.fixture(scope="class")
    def fitted_xwoba(self):
        df = _make_synthetic_data(n_players=15, n_seasons=2)
        data = prepare_hitter_data(df, "xwoba")
        model, trace = fit_hitter_model(
            data, draws=200, tune=100, chains=2, random_seed=42,
        )
        return data, trace

    def test_k_rate_posteriors(self, fitted_k_rate):
        data, trace = fitted_k_rate
        posteriors = extract_posteriors(trace, data)
        assert len(posteriors) == len(data["df"])
        assert "k_rate_mean" in posteriors.columns
        assert "age_bucket" in posteriors.columns
        # K rates should be in (0, 1)
        assert posteriors["k_rate_mean"].between(0, 1).all()

    def test_bb_rate_posteriors(self, fitted_bb_rate):
        data, trace = fitted_bb_rate
        posteriors = extract_posteriors(trace, data)
        assert "bb_rate_mean" in posteriors.columns
        assert posteriors["bb_rate_mean"].between(0, 1).all()

    def test_xwoba_posteriors(self, fitted_xwoba):
        data, trace = fitted_xwoba
        posteriors = extract_posteriors(trace, data)
        assert "xwoba_mean" in posteriors.columns
        # xwOBA should be roughly in (0.1, 0.6)
        assert posteriors["xwoba_mean"].between(0.05, 0.7).all()

    def test_extract_samples(self, fitted_k_rate):
        data, trace = fitted_k_rate
        pid = data["player_ids"][0]
        season = data["df"]["season"].max()
        samples = extract_rate_samples(trace, data, pid, season,
                                       project_forward=False)
        assert len(samples) > 0
        assert all(0 < s < 1 for s in samples)

    def test_extract_samples_forward(self, fitted_k_rate):
        data, trace = fitted_k_rate
        pid = data["player_ids"][0]
        season = data["df"]["season"].max()
        base = extract_rate_samples(trace, data, pid, season,
                                    project_forward=False)
        fwd = extract_rate_samples(trace, data, pid, season,
                                   project_forward=True)
        # Forward-projected should have wider spread
        assert np.std(fwd) >= np.std(base) * 0.9  # allow small noise

    def test_convergence(self, fitted_k_rate):
        data, trace = fitted_k_rate
        result = check_convergence(trace, "k_rate")
        assert "converged" in result
        assert "max_rhat" in result
        assert "min_ess_bulk" in result

    def test_age_bucket_mu_pop(self, fitted_k_rate):
        """mu_pop should have 3 age-bucket values."""
        _, trace = fitted_k_rate
        mu_pop = trace.posterior["mu_pop"].values
        assert mu_pop.shape[-1] == 3
