import numpy as np
import pandas as pd
import pytest

from src.models.pitcher_k_rate_model import (
    build_pitcher_k_rate_model,
    prepare_pitcher_model_data,
)


def _make_pitcher_df() -> pd.DataFrame:
    """Synthetic 3-row DataFrame: pitcher 100 in 2023+2024, pitcher 101 in 2024."""
    return pd.DataFrame([
        {
            "pitcher_id": 100,
            "pitcher_name": "Ace",
            "pitch_hand": "R",
            "season": 2023,
            "batters_faced": 600,
            "k": 150,
            "k_rate": 0.250,
            "bb": 50,
            "bb_rate": 0.083,
            "ip": 180.0,
            "games": 30,
            "whiff_rate": 0.28,
            "barrel_rate_against": 0.06,
            "is_starter": 1,
        },
        {
            "pitcher_id": 100,
            "pitcher_name": "Ace",
            "pitch_hand": "R",
            "season": 2024,
            "batters_faced": 620,
            "k": 160,
            "k_rate": 0.258,
            "bb": 48,
            "bb_rate": 0.077,
            "ip": 190.0,
            "games": 32,
            "whiff_rate": 0.30,
            "barrel_rate_against": 0.05,
            "is_starter": 1,
        },
        {
            "pitcher_id": 101,
            "pitcher_name": "Bullpen",
            "pitch_hand": "L",
            "season": 2024,
            "batters_faced": 250,
            "k": 70,
            "k_rate": 0.280,
            "bb": 30,
            "bb_rate": 0.120,
            "ip": 65.0,
            "games": 55,
            "whiff_rate": 0.32,
            "barrel_rate_against": 0.04,
            "is_starter": 0,
        },
    ])


def test_prepare_pitcher_model_data_basic() -> None:
    """Shapes, dtypes, n_players=2, n_seasons=2, bf/k arrays, is_starter."""
    df = _make_pitcher_df()
    data = prepare_pitcher_model_data(df)

    assert data["n_players"] == 2
    assert data["n_seasons"] == 2
    assert data["bf"].shape == (3,)
    assert data["k"].shape == (3,)
    assert data["bf"].dtype == int
    assert data["k"].dtype == int
    # Player-level covariates always present
    assert "whiff_player_mean" in data
    assert "barrel_player_mean" in data
    assert data["whiff_player_mean"].shape == (2,)
    assert data["barrel_player_mean"].shape == (2,)
    assert np.all(np.isfinite(data["whiff_player_mean"]))
    assert np.all(np.isfinite(data["barrel_player_mean"]))
    # is_starter passthrough
    assert "is_starter" in data
    assert data["is_starter"].shape == (3,)
    assert list(data["is_starter"]) == [1, 1, 0]


def test_prepare_pitcher_model_data_player_level_z_scores() -> None:
    """Player-level means are z-scored across players."""
    df = _make_pitcher_df()
    data = prepare_pitcher_model_data(df)

    # With 2 players having different whiff/barrel means,
    # z-scores should have mean ~0 and std ~1
    wpm = data["whiff_player_mean"]
    bpm = data["barrel_player_mean"]
    assert np.isclose(wpm.mean(), 0.0, atol=0.01)
    assert np.isclose(bpm.mean(), 0.0, atol=0.01)


def test_prepare_pitcher_model_data_constant_covariates() -> None:
    """Constant whiff_rate/barrel_rate_against -> z-scores = 0."""
    df = _make_pitcher_df()
    df["whiff_rate"] = 0.25
    df["barrel_rate_against"] = 0.05
    data = prepare_pitcher_model_data(df)

    assert np.allclose(data["whiff_player_mean"], 0.0)
    assert np.allclose(data["barrel_player_mean"], 0.0)


def test_build_pitcher_k_rate_model_has_expected_variables() -> None:
    """Model has core hierarchical vars + beta_starter but no covariate betas."""
    df = _make_pitcher_df()
    data = prepare_pitcher_model_data(df)
    model = build_pitcher_k_rate_model(data)

    rv_names = {rv.name for rv in model.free_RVs}
    det_names = {d.name for d in model.deterministics}
    all_names = rv_names | det_names

    # Core variables present
    assert "mu_pop" in rv_names
    assert "sigma_player" in rv_names
    assert "beta_starter" in rv_names
    assert "alpha" in det_names
    assert "k_rate" in det_names

    # Covariate betas intentionally excluded (whiff r=0.71 collapses sigma_player)
    assert "beta_whiff" not in all_names
    assert "beta_barrel_against" not in all_names

    # Hitter-specific variables absent
    assert "beta_barrel" not in all_names
    assert "beta_hard_hit" not in all_names
    assert "gamma_pop" not in all_names


def test_build_pitcher_k_rate_model_uses_bf_not_pa() -> None:
    """k_obs uses bf (batters_faced); 'pa' not in data dict."""
    df = _make_pitcher_df()
    data = prepare_pitcher_model_data(df)

    assert "bf" in data
    assert "pa" not in data

    model = build_pitcher_k_rate_model(data)
    assert model is not None


def test_build_pitcher_k_rate_model_no_role_no_beta_starter() -> None:
    """Without is_starter column, model has no beta_starter variable."""
    df = _make_pitcher_df().drop(columns=["is_starter"])
    data = prepare_pitcher_model_data(df)

    assert "is_starter" not in data

    model = build_pitcher_k_rate_model(data)
    rv_names = {rv.name for rv in model.free_RVs}
    assert "beta_starter" not in rv_names


def test_prepare_pitcher_model_data_missing_covariates_fills_zeros() -> None:
    """Missing whiff_rate/barrel_rate_against columns -> NaN filled, z=0."""
    df = _make_pitcher_df().drop(columns=["whiff_rate", "barrel_rate_against"])
    data = prepare_pitcher_model_data(df)

    assert np.allclose(data["whiff_player_mean"], 0.0)
    assert np.allclose(data["barrel_player_mean"], 0.0)
