"""Tests for the pitch archetype clustering module.

All tests monkeypatch ``get_pitch_shape_offerings`` to avoid DB dependency.
"""
import numpy as np
import pandas as pd
import pytest

from src.data import pitch_archetypes


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _make_offerings(
    rows: list[dict],
    season: int = 2024,
) -> pd.DataFrame:
    """Build a minimal offerings DataFrame from a list of row dicts."""
    defaults = {
        "season": season,
        "pitch_hand": "R",
        "pitches": 300,
        "release_pos_x": -1.5,
        "release_pos_z": 5.7,
        "pitch_family": "fastball",
    }
    records = [{**defaults, **r} for r in rows]
    return pd.DataFrame(records)


def _rhp_fastball(**overrides) -> dict:
    return {
        "pitcher_id": 1,
        "pitch_hand": "R",
        "pitch_type": "FF",
        "pitch_name": "4-Seam Fastball",
        "release_speed": 96.0,
        "pfx_x": -0.5,
        "pfx_z": 1.6,
        "release_spin_rate": 2400.0,
        "release_extension": 6.4,
        "pitch_family": "fastball",
        **overrides,
    }


def _rhp_slider(**overrides) -> dict:
    return {
        "pitcher_id": 2,
        "pitch_hand": "R",
        "pitch_type": "SL",
        "pitch_name": "Slider",
        "release_speed": 84.0,
        "pfx_x": 0.3,
        "pfx_z": -0.3,
        "release_spin_rate": 2700.0,
        "release_extension": 6.0,
        "pitch_family": "breaking",
        **overrides,
    }


def _rhp_changeup(**overrides) -> dict:
    return {
        "pitcher_id": 3,
        "pitch_hand": "R",
        "pitch_type": "CH",
        "pitch_name": "Changeup",
        "release_speed": 86.0,
        "pfx_x": -1.1,
        "pfx_z": 0.8,
        "release_spin_rate": 1800.0,
        "release_extension": 5.9,
        "pitch_family": "offspeed",
        **overrides,
    }


def _rhp_sinker(**overrides) -> dict:
    return {
        "pitcher_id": 4,
        "pitch_hand": "R",
        "pitch_type": "SI",
        "pitch_name": "Sinker",
        "release_speed": 95.5,
        "pfx_x": 1.4,
        "pfx_z": 0.5,
        "release_spin_rate": 2200.0,
        "release_extension": 6.4,
        "pitch_family": "fastball",
        **overrides,
    }


def _diverse_offerings(season: int = 2024) -> pd.DataFrame:
    """4 distinct pitch types — enough for k=2..4."""
    return _make_offerings(
        [
            _rhp_fastball(pitcher_id=1),
            _rhp_slider(pitcher_id=2),
            _rhp_changeup(pitcher_id=3),
            _rhp_sinker(pitcher_id=4),
        ],
        season=season,
    )


def _monkeypatch_offerings(monkeypatch, offerings_by_season: dict):
    """Replace get_pitch_shape_offerings with a dict-lookup stub."""
    def _fake(season):
        return offerings_by_season.get(season, pd.DataFrame()).copy()

    monkeypatch.setattr(pitch_archetypes, "get_pitch_shape_offerings", _fake)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------
class TestNormalizeHandedness:
    def test_flips_lhp_pfx_x(self):
        df = _make_offerings([
            _rhp_fastball(pitcher_id=1, pitch_hand="R", pfx_x=-0.5),
            _rhp_fastball(pitcher_id=2, pitch_hand="L", pfx_x=-0.5),
        ])
        out = pitch_archetypes._normalize_handedness(df)

        rhp_row = out[out["pitcher_id"] == 1].iloc[0]
        lhp_row = out[out["pitcher_id"] == 2].iloc[0]

        assert rhp_row["pfx_x"] == pytest.approx(-0.5)
        assert rhp_row["pfx_x_flipped"] == False  # noqa: E712
        assert lhp_row["pfx_x"] == pytest.approx(0.5)
        assert lhp_row["pfx_x_flipped"] == True  # noqa: E712


class TestShapeFeatures:
    def test_has_five_elements(self):
        assert len(pitch_archetypes.SHAPE_FEATURES) == 5

    def test_no_release_pos(self):
        for feat in pitch_archetypes.SHAPE_FEATURES:
            assert "release_pos" not in feat


class TestFitDeterministic:
    def test_same_input_same_centroids(self, monkeypatch, tmp_path):
        monkeypatch.setattr(pitch_archetypes, "CACHE_DIR", tmp_path)
        data = _diverse_offerings(2024)
        _monkeypatch_offerings(monkeypatch, {2024: data})
        monkeypatch.setattr(
            pitch_archetypes, "_get_clustering_seasons", lambda: [2024]
        )

        a1 = pitch_archetypes.fit_pitch_archetypes(
            seasons=[2024], n_clusters=2, random_state=42, force_rebuild=True
        )
        a2 = pitch_archetypes.fit_pitch_archetypes(
            seasons=[2024], n_clusters=2, random_state=42, force_rebuild=True
        )

        np.testing.assert_array_almost_equal(
            a1["kmeans"].cluster_centers_,
            a2["kmeans"].cluster_centers_,
        )


class TestLhpRhpMirror:
    def test_mirror_pitches_cluster_together(self, monkeypatch, tmp_path):
        """A RHP fastball (pfx_x=-0.5) and a LHP fastball (pfx_x=+0.5)
        should get the same archetype after normalization flips the LHP."""
        monkeypatch.setattr(pitch_archetypes, "CACHE_DIR", tmp_path)

        rhp_fb = _rhp_fastball(pitcher_id=1, pitch_hand="R", pfx_x=-0.5)
        lhp_fb = _rhp_fastball(pitcher_id=2, pitch_hand="L", pfx_x=0.5)
        slider = _rhp_slider(pitcher_id=3, pitch_hand="R")
        changeup = _rhp_changeup(pitcher_id=4, pitch_hand="R")

        data = _make_offerings([rhp_fb, lhp_fb, slider, changeup])
        _monkeypatch_offerings(monkeypatch, {2024: data})
        monkeypatch.setattr(
            pitch_archetypes, "_get_clustering_seasons", lambda: [2024]
        )

        arts = pitch_archetypes.fit_pitch_archetypes(
            seasons=[2024], n_clusters=2, random_state=42, force_rebuild=True
        )
        assigned = pitch_archetypes.assign_pitch_archetypes(
            season=2024, artifacts=arts, n_clusters=2, force_rebuild=True
        )

        rhp_label = assigned.loc[
            assigned["pitcher_id"] == 1, "pitch_archetype"
        ].iloc[0]
        lhp_label = assigned.loc[
            assigned["pitcher_id"] == 2, "pitch_archetype"
        ].iloc[0]
        assert rhp_label == lhp_label


class TestCrossSeasonStability:
    def test_same_pitch_different_seasons_same_label(
        self, monkeypatch, tmp_path
    ):
        """Identical pitch offerings in different seasons → same archetype."""
        monkeypatch.setattr(pitch_archetypes, "CACHE_DIR", tmp_path)

        base = [
            _rhp_fastball(pitcher_id=1),
            _rhp_slider(pitcher_id=2),
            _rhp_changeup(pitcher_id=3),
            _rhp_sinker(pitcher_id=4),
        ]
        data_2023 = _make_offerings(base, season=2023)
        data_2024 = _make_offerings(base, season=2024)
        data_2025 = _make_offerings(base, season=2025)

        _monkeypatch_offerings(
            monkeypatch, {2023: data_2023, 2024: data_2024, 2025: data_2025}
        )
        monkeypatch.setattr(
            pitch_archetypes, "_get_clustering_seasons", lambda: [2023, 2024]
        )

        arts = pitch_archetypes.fit_pitch_archetypes(
            seasons=[2023, 2024],
            n_clusters=2,
            random_state=42,
            force_rebuild=True,
        )

        a24 = pitch_archetypes.assign_pitch_archetypes(
            season=2024, artifacts=arts, n_clusters=2, force_rebuild=True
        )
        a25 = pitch_archetypes.assign_pitch_archetypes(
            season=2025, artifacts=arts, n_clusters=2, force_rebuild=True
        )

        labels_24 = a24.sort_values("pitcher_id")["pitch_archetype"].tolist()
        labels_25 = a25.sort_values("pitcher_id")["pitch_archetype"].tolist()
        assert labels_24 == labels_25


class TestFullPipeline:
    def test_fit_assign_correct_shapes_and_columns(
        self, monkeypatch, tmp_path
    ):
        monkeypatch.setattr(pitch_archetypes, "CACHE_DIR", tmp_path)

        data = _diverse_offerings(2024)
        _monkeypatch_offerings(monkeypatch, {2024: data})
        monkeypatch.setattr(
            pitch_archetypes, "_get_clustering_seasons", lambda: [2024]
        )

        arts = pitch_archetypes.fit_pitch_archetypes(
            seasons=[2024], n_clusters=2, random_state=42, force_rebuild=True
        )

        # Cluster metadata
        clusters = arts["clusters"]
        assert len(clusters) == 2
        assert "pitch_archetype" in clusters.columns
        assert "primary_pitch_type" in clusters.columns
        assert "archetype_name" in clusters.columns

        # Assignments
        assigned = pitch_archetypes.assign_pitch_archetypes(
            season=2024, artifacts=arts, n_clusters=2, force_rebuild=True
        )
        assert len(assigned) == 4
        assert set(assigned["pitch_archetype"]) == {1, 2}
        assert "pfx_x_flipped" in assigned.columns
        assert "release_pos_x" in assigned.columns

        # Public API wrappers
        offerings = pitch_archetypes.get_pitch_archetype_offerings(
            season=2024, n_clusters=2, force_rebuild=True
        )
        assert len(offerings) == 4

        cluster_df = pitch_archetypes.get_pitch_archetype_clusters(
            n_clusters=2, force_rebuild=True
        )
        assert len(cluster_df) == 2


class TestEvaluateClusterCounts:
    def test_output_columns_and_row_count(self, monkeypatch, tmp_path):
        monkeypatch.setattr(pitch_archetypes, "CACHE_DIR", tmp_path)

        # Need n_samples > max(k) for silhouette_score to work.
        # k_range goes up to 4, so we need at least 5 distinct rows.
        rows = [
            _rhp_fastball(pitcher_id=1),
            _rhp_slider(pitcher_id=2),
            _rhp_changeup(pitcher_id=3),
            _rhp_sinker(pitcher_id=4),
            _rhp_fastball(pitcher_id=5, release_speed=92.0, pfx_x=0.8,
                          release_spin_rate=2100.0, pitch_type="FC",
                          pitch_name="Cutter", pitch_family="fastball"),
            _rhp_slider(pitcher_id=6, release_speed=78.0, pfx_x=-0.1,
                        release_spin_rate=2900.0, pitch_type="CU",
                        pitch_name="Curveball", pitch_family="breaking"),
        ]
        data = _make_offerings(rows, season=2024)
        _monkeypatch_offerings(monkeypatch, {2024: data})
        monkeypatch.setattr(
            pitch_archetypes, "_get_clustering_seasons", lambda: [2024]
        )

        result = pitch_archetypes.evaluate_cluster_counts(
            seasons=[2024], k_range=range(2, 5), random_state=42
        )

        assert list(result.columns) == ["k", "inertia", "silhouette_score"]
        assert len(result) == 3
        assert result["k"].tolist() == [2, 3, 4]
        assert all(result["inertia"] > 0)
        assert all((-1 <= result["silhouette_score"]) & (result["silhouette_score"] <= 1))


class TestValidation:
    def test_fit_rejects_n_clusters_below_2(self, monkeypatch, tmp_path):
        monkeypatch.setattr(pitch_archetypes, "CACHE_DIR", tmp_path)
        with pytest.raises(ValueError, match="n_clusters must be >= 2"):
            pitch_archetypes.fit_pitch_archetypes(
                seasons=[2024], n_clusters=1, force_rebuild=True
            )
