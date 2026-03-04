import numpy as np
import pandas as pd

from src.models.k_rate_model import prepare_model_data


def test_prepare_model_data_handles_constant_covariates() -> None:
    df = pd.DataFrame(
        [
            {
                "batter_id": 10,
                "batter_name": "A",
                "season": 2023,
                "pa": 100,
                "k": 20,
                "k_rate": 0.20,
                "barrel_pct": 0.08,
                "hard_hit_pct": 0.35,
            },
            {
                "batter_id": 10,
                "batter_name": "A",
                "season": 2024,
                "pa": 120,
                "k": 26,
                "k_rate": 0.2167,
                "barrel_pct": 0.08,
                "hard_hit_pct": 0.35,
            },
            {
                "batter_id": 11,
                "batter_name": "B",
                "season": 2024,
                "pa": 90,
                "k": 18,
                "k_rate": 0.20,
                "barrel_pct": 0.08,
                "hard_hit_pct": 0.35,
            },
        ]
    )

    data = prepare_model_data(df)

    assert data["n_players"] == 2
    assert data["n_seasons"] == 2
    assert data["pa"].shape[0] == 3
    assert np.all(np.isfinite(data["barrel_z"]))
    assert np.all(np.isfinite(data["hard_hit_z"]))
    assert np.allclose(data["barrel_z"], 0.0)
    assert np.allclose(data["hard_hit_z"], 0.0)
