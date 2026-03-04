import numpy as np
import pandas as pd

from src.data import feature_eng


def test_build_hitter_vulnerability_computes_rates_and_excludes_pitches(
    monkeypatch,
) -> None:
    raw = pd.DataFrame(
        [
            {
                "batter_id": 1,
                "batter_stand": "R",
                "pitch_type": "FF",
                "pitches": 10,
                "swings": 5,
                "whiffs": 2,
                "out_of_zone_pitches": 4,
                "chase_swings": 1,
                "called_strikes": 2,
                "csw": 4,
                "bip": 3,
                "hard_hits": 1,
                "xwoba_contact": 0.33,
                "barrels_proxy": 1,
            },
            {
                "batter_id": 1,
                "batter_stand": "R",
                "pitch_type": "FA",  # excluded in constants
                "pitches": 8,
                "swings": 4,
                "whiffs": 1,
                "out_of_zone_pitches": 2,
                "chase_swings": 1,
                "called_strikes": 1,
                "csw": 2,
                "bip": 2,
                "hard_hits": 1,
                "xwoba_contact": 0.30,
                "barrels_proxy": 0,
            },
        ]
    )
    monkeypatch.setattr(
        feature_eng,
        "get_hitter_pitch_type_profile",
        lambda season: raw.copy(),
    )

    out = feature_eng._build_hitter_vulnerability(2024)

    assert set(out["pitch_type"]) == {"FF"}
    row = out.iloc[0]
    assert np.isclose(row["whiff_rate"], 2 / 5)
    assert np.isclose(row["chase_rate"], 1 / 4)
    assert np.isclose(row["csw_pct"], 4 / 10)
    assert row["season"] == 2024
