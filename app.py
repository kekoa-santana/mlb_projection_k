"""
The Data Diamond — MLB Bayesian Projection Dashboard.

Interactive Streamlit app for exploring hierarchical Bayesian player
projections, posterior distributions, and game-level K predictions.

Run:
    streamlit run app.py
"""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
from scipy.stats import gaussian_kde

# ---------------------------------------------------------------------------
# Config & paths
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent
DASHBOARD_DIR = PROJECT_ROOT / "data" / "dashboard"

# Brand colors
GOLD = "#C8A96E"
TEAL = "#4FC3C8"
SLATE = "#7B8FA6"
CREAM = "#F5F2EE"
DARK = "#0F1117"

st.set_page_config(
    page_title="The Data Diamond | MLB Projections",
    page_icon="",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Custom CSS
# ---------------------------------------------------------------------------
st.markdown(f"""
<style>
    /* Header bar */
    .brand-header {{
        background: {DARK};
        padding: 1rem 2rem;
        border-radius: 8px;
        margin-bottom: 1.5rem;
        display: flex;
        align-items: center;
        justify-content: space-between;
    }}
    .brand-title {{
        color: {GOLD};
        font-size: 1.8rem;
        font-weight: 700;
        letter-spacing: 1px;
    }}
    .brand-subtitle {{
        color: {SLATE};
        font-size: 0.9rem;
    }}
    /* Metric cards */
    .metric-card {{
        background: {DARK};
        border-radius: 8px;
        padding: 1rem 1.2rem;
        text-align: center;
    }}
    .metric-value {{
        color: {GOLD};
        font-size: 1.6rem;
        font-weight: 700;
    }}
    .metric-label {{
        color: {SLATE};
        font-size: 0.75rem;
        text-transform: uppercase;
        letter-spacing: 1px;
    }}
    /* Positive/negative delta styling */
    .delta-pos {{ color: {TEAL}; font-weight: 600; }}
    .delta-neg {{ color: #E57373; font-weight: 600; }}
    /* Table styling */
    .stDataFrame {{ font-size: 0.85rem; }}
</style>
""", unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Data loading (cached)
# ---------------------------------------------------------------------------
@st.cache_data
def load_projections(player_type: str) -> pd.DataFrame:
    """Load pre-computed projection parquet."""
    path = DASHBOARD_DIR / f"{player_type}_projections.parquet"
    if not path.exists():
        return pd.DataFrame()
    return pd.read_parquet(path)


@st.cache_data
def load_k_samples() -> dict[str, np.ndarray]:
    """Load pitcher K% posterior samples."""
    path = DASHBOARD_DIR / "pitcher_k_samples.npz"
    if not path.exists():
        return {}
    data = np.load(path)
    return {k: data[k] for k in data.files}


@st.cache_data
def load_bf_priors() -> pd.DataFrame:
    """Load BF priors."""
    path = DASHBOARD_DIR / "bf_priors.parquet"
    if not path.exists():
        return pd.DataFrame()
    return pd.read_parquet(path)


def _check_data_exists() -> bool:
    """Check if pre-computed data exists."""
    required = [
        DASHBOARD_DIR / "hitter_projections.parquet",
        DASHBOARD_DIR / "pitcher_projections.parquet",
    ]
    return all(p.exists() for p in required)


def _fmt_pct(val: float, decimals: int = 1) -> str:
    """Format a 0-1 rate as percentage string."""
    return f"{val * 100:.{decimals}f}%"


def _delta_html(val: float, higher_is_better: bool = True) -> str:
    """Format a delta as colored HTML."""
    pct = val * 100
    if (pct > 0 and higher_is_better) or (pct < 0 and not higher_is_better):
        return f'<span class="delta-pos">+{pct:.1f}pp</span>'
    elif pct == 0:
        return f'<span style="color:{SLATE}">0.0pp</span>'
    else:
        return f'<span class="delta-neg">{pct:.1f}pp</span>'


def _metric_card(label: str, value: str) -> str:
    """Render a styled metric card."""
    return f"""
    <div class="metric-card">
        <div class="metric-value">{value}</div>
        <div class="metric-label">{label}</div>
    </div>
    """


# ---------------------------------------------------------------------------
# Matplotlib helpers
# ---------------------------------------------------------------------------
def _create_posterior_fig(
    samples: np.ndarray,
    observed: float | None = None,
    stat_label: str = "K%",
    color: str = TEAL,
) -> plt.Figure:
    """Create a posterior KDE plot with brand styling."""
    fig, ax = plt.subplots(figsize=(7, 3.5))
    fig.patch.set_facecolor(DARK)
    ax.set_facecolor(DARK)

    # KDE
    pct_samples = samples * 100
    kde = gaussian_kde(pct_samples, bw_method=0.3)
    x = np.linspace(pct_samples.min() - 2, pct_samples.max() + 2, 300)
    y = kde(x)

    ax.fill_between(x, y, alpha=0.3, color=color)
    ax.plot(x, y, color=color, linewidth=2)

    # Credible interval
    ci_lo, ci_hi = np.percentile(pct_samples, [2.5, 97.5])
    ci_mask = (x >= ci_lo) & (x <= ci_hi)
    ax.fill_between(x[ci_mask], y[ci_mask], alpha=0.15, color=color)

    # Mean line
    mean_val = np.mean(pct_samples)
    ax.axvline(mean_val, color=GOLD, linewidth=2, linestyle="--", alpha=0.9)
    ax.text(
        mean_val, ax.get_ylim()[1] * 0.92,
        f" {mean_val:.1f}%",
        color=GOLD, fontsize=11, fontweight="bold", va="top",
    )

    # Observed line
    if observed is not None:
        obs_pct = observed * 100
        ax.axvline(obs_pct, color=SLATE, linewidth=1.5, linestyle=":", alpha=0.8)
        ax.text(
            obs_pct, ax.get_ylim()[1] * 0.75,
            f" {obs_pct:.1f}% (obs)",
            color=SLATE, fontsize=9, va="top",
        )

    # CI annotation
    ax.text(
        0.98, 0.95,
        f"95% CI: [{ci_lo:.1f}%, {ci_hi:.1f}%]",
        transform=ax.transAxes,
        color=SLATE, fontsize=9, ha="right", va="top",
    )

    ax.set_xlabel(stat_label, color=SLATE, fontsize=10)
    ax.set_ylabel("Density", color=SLATE, fontsize=10)
    ax.tick_params(colors=SLATE, labelsize=9)
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.set_yticks([])

    fig.tight_layout()
    return fig


def _create_game_k_fig(
    k_samples: np.ndarray,
    pitcher_name: str,
) -> plt.Figure:
    """Create a game K distribution histogram."""
    fig, ax = plt.subplots(figsize=(7, 3.5))
    fig.patch.set_facecolor(DARK)
    ax.set_facecolor(DARK)

    max_k = int(k_samples.max()) + 1
    bins = np.arange(-0.5, max_k + 1.5, 1)
    counts, _, bars = ax.hist(
        k_samples, bins=bins, density=True,
        color=TEAL, alpha=0.7, edgecolor=DARK, linewidth=0.5,
    )

    # Color the mode bar gold
    mode_k = int(np.median(k_samples))
    for bar in bars:
        if abs(bar.get_x() + 0.5 - mode_k) < 0.5:
            bar.set_facecolor(GOLD)
            bar.set_alpha(0.9)

    mean_k = np.mean(k_samples)
    ax.axvline(mean_k, color=GOLD, linewidth=2, linestyle="--", alpha=0.9)
    ax.text(
        mean_k + 0.3, ax.get_ylim()[1] * 0.9,
        f"E[K] = {mean_k:.1f}",
        color=GOLD, fontsize=11, fontweight="bold", va="top",
    )

    ax.set_xlabel("Strikeouts", color=SLATE, fontsize=11)
    ax.set_ylabel("Probability", color=SLATE, fontsize=10)
    ax.set_title(
        f"{pitcher_name} — Projected K Distribution (2026)",
        color=CREAM, fontsize=13, fontweight="bold", pad=12,
    )
    ax.tick_params(colors=SLATE, labelsize=9)
    for spine in ax.spines.values():
        spine.set_visible(False)

    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Page: Projections
# ---------------------------------------------------------------------------
def page_projections() -> None:
    """Sortable projection tables for pitchers and hitters."""
    player_type = st.radio(
        "Select player type",
        ["Pitcher", "Hitter"],
        horizontal=True,
    )

    df = load_projections(player_type.lower())
    if df.empty:
        st.warning("No projection data found. Run `scripts/precompute_dashboard_data.py` first.")
        return

    # Determine columns based on player type
    if player_type == "Pitcher":
        id_col = "pitcher_id"
        name_col = "pitcher_name"
        hand_col = "pitch_hand"
        stat_configs = [
            ("K%", "k_rate", True),
            ("BB%", "bb_rate", False),
            ("HR/BF", "hr_per_bf", False),
        ]
        role_options = ["All", "Starters", "Relievers"]
        role = st.selectbox("Role filter", role_options)
        if role == "Starters":
            df = df[df["is_starter"] == 1]
        elif role == "Relievers":
            df = df[df["is_starter"] == 0]
    else:
        id_col = "batter_id"
        name_col = "batter_name"
        hand_col = "batter_stand"
        stat_configs = [
            ("K%", "k_rate", False),
            ("BB%", "bb_rate", True),
            ("HR/PA", "hr_rate", True),
            ("xwOBA", "xwoba", True),
        ]

    # Sort options
    sort_options = ["Composite Score"] + [s[0] for s in stat_configs]
    sort_by = st.selectbox("Sort by", sort_options)

    if sort_by == "Composite Score":
        sort_col = "composite_score"
        ascending = False
    else:
        stat_key = next(s[1] for s in stat_configs if s[0] == sort_by)
        sort_col = f"delta_{stat_key}"
        higher_is_better = next(s[2] for s in stat_configs if s[0] == sort_by)
        ascending = not higher_is_better

    df_sorted = df.sort_values(sort_col, ascending=ascending).reset_index(drop=True)

    # Build display table
    display_rows = []
    for _, row in df_sorted.iterrows():
        r = {
            "Rank": len(display_rows) + 1,
            "Name": row[name_col],
            "Hand": row.get(hand_col, ""),
            "Age": int(row["age"]) if pd.notna(row.get("age")) else "",
            "Score": round(row["composite_score"], 2),
        }
        for label, key, _ in stat_configs:
            obs_col = f"observed_{key}"
            proj_col = f"projected_{key}"
            delta_col = f"delta_{key}"
            if obs_col in row.index and pd.notna(row.get(obs_col)):
                r[f"{label} (2025)"] = _fmt_pct(row[obs_col])
                r[f"{label} (2026)"] = _fmt_pct(row[proj_col])
                r[f"{label} delta"] = f"{row[delta_col] * 100:+.1f}pp"
            else:
                r[f"{label} (2025)"] = "—"
                r[f"{label} (2026)"] = "—"
                r[f"{label} delta"] = "—"
        display_rows.append(r)

    display_df = pd.DataFrame(display_rows)
    st.dataframe(
        display_df,
        use_container_width=True,
        hide_index=True,
        height=600,
    )

    st.caption(
        f"Showing {len(display_df)} players. "
        "Composite score weights stat deltas (normalized, direction-aware). "
        "Positive = projected improvement."
    )


# ---------------------------------------------------------------------------
# Page: Player Profile
# ---------------------------------------------------------------------------
def page_player_profile() -> None:
    """Deep dive into a single player's projections."""
    player_type = st.radio(
        "Player type",
        ["Pitcher", "Hitter"],
        horizontal=True,
        key="profile_type",
    )

    df = load_projections(player_type.lower())
    if df.empty:
        st.warning("No projection data found. Run `scripts/precompute_dashboard_data.py` first.")
        return

    if player_type == "Pitcher":
        name_col = "pitcher_name"
        id_col = "pitcher_id"
        hand_col = "pitch_hand"
        stat_configs = [
            ("K%", "k_rate", True, "Higher K% = more strikeout stuff"),
            ("BB%", "bb_rate", False, "Lower BB% = better control"),
            ("HR/BF", "hr_per_bf", False, "Lower HR/BF = fewer dingers allowed"),
        ]
    else:
        name_col = "batter_name"
        id_col = "batter_id"
        hand_col = "batter_stand"
        stat_configs = [
            ("K%", "k_rate", False, "Lower K% = better contact ability"),
            ("BB%", "bb_rate", True, "Higher BB% = better plate discipline"),
            ("HR/PA", "hr_rate", True, "Higher HR/PA = more power"),
            ("xwOBA", "xwoba", True, "Higher xwOBA = better expected production"),
        ]

    # Player selector
    names = df.sort_values(name_col)[name_col].unique().tolist()
    selected_name = st.selectbox("Select player", names, key="profile_player")

    player_row = df[df[name_col] == selected_name].iloc[0]
    player_id = int(player_row[id_col])

    # Header
    hand = player_row.get(hand_col, "")
    age = int(player_row["age"]) if pd.notna(player_row.get("age")) else "?"
    role = ""
    if player_type == "Pitcher" and "is_starter" in player_row.index:
        role = "SP" if player_row["is_starter"] else "RP"

    header_parts = [f"Age {age}"]
    if hand:
        header_parts.append(f"{'LHP' if hand == 'L' else 'RHP'}" if player_type == "Pitcher"
                           else f"Bats {'L' if hand == 'L' else 'R'}")
    if role:
        header_parts.append(role)

    st.markdown(f"""
    <div class="brand-header">
        <div>
            <div class="brand-title">{selected_name}</div>
            <div class="brand-subtitle">{' | '.join(header_parts)} | 2026 Projection</div>
        </div>
        <div style="color:{GOLD}; font-size:1.2rem; font-weight:600;">
            Composite: {player_row['composite_score']:+.2f}
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Stat metrics row
    cols = st.columns(len(stat_configs))
    for col, (label, key, higher_better, _) in zip(cols, stat_configs):
        obs_col = f"observed_{key}"
        proj_col = f"projected_{key}"
        if obs_col in player_row.index and pd.notna(player_row.get(obs_col)):
            with col:
                st.markdown(
                    _metric_card(f"Proj. {label}", _fmt_pct(player_row[proj_col])),
                    unsafe_allow_html=True,
                )
                delta = player_row[f"delta_{key}"]
                st.markdown(
                    f"<div style='text-align:center; margin-top:4px;'>"
                    f"2025: {_fmt_pct(player_row[obs_col])} "
                    f"({_delta_html(delta, higher_better)})</div>",
                    unsafe_allow_html=True,
                )
        else:
            with col:
                st.markdown(
                    _metric_card(f"Proj. {label}", "—"),
                    unsafe_allow_html=True,
                )

    # Posterior KDE (for pitchers with K% samples)
    k_samples = load_k_samples()
    sample_key = str(player_id)

    if player_type == "Pitcher" and sample_key in k_samples:
        st.markdown("---")
        st.subheader("K% Posterior Distribution")
        samples = k_samples[sample_key]
        obs_k = player_row.get("observed_k_rate")
        fig = _create_posterior_fig(
            samples,
            observed=obs_k if pd.notna(obs_k) else None,
            stat_label="Projected K% (2026)",
        )
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)

        ci_lo, ci_hi = np.percentile(samples * 100, [2.5, 97.5])
        st.caption(
            f"Dashed gold = projected mean | Dotted gray = 2025 observed | "
            f"Shaded = 95% credible interval [{ci_lo:.1f}%, {ci_hi:.1f}%]"
        )

    # Stat detail table
    st.markdown("---")
    st.subheader("Stat Breakdown")
    detail_rows = []
    for label, key, higher_better, desc in stat_configs:
        obs_col = f"observed_{key}"
        proj_col = f"projected_{key}"
        sd_col = f"projected_{key}_sd"
        lo_col = f"projected_{key}_2_5"
        hi_col = f"projected_{key}_97_5"
        if obs_col in player_row.index and pd.notna(player_row.get(obs_col)):
            detail_rows.append({
                "Stat": label,
                "2025 Observed": _fmt_pct(player_row[obs_col]),
                "2026 Projected": _fmt_pct(player_row[proj_col]),
                "Delta": f"{player_row[f'delta_{key}'] * 100:+.1f}pp",
                "95% CI": (
                    f"[{_fmt_pct(player_row[lo_col])}, {_fmt_pct(player_row[hi_col])}]"
                    if lo_col in player_row.index and pd.notna(player_row.get(lo_col))
                    else "—"
                ),
                "Uncertainty (SD)": (
                    _fmt_pct(player_row[sd_col])
                    if sd_col in player_row.index and pd.notna(player_row.get(sd_col))
                    else "—"
                ),
                "Description": desc,
            })
    if detail_rows:
        st.dataframe(
            pd.DataFrame(detail_rows),
            use_container_width=True,
            hide_index=True,
        )


# ---------------------------------------------------------------------------
# Page: Game K Simulator
# ---------------------------------------------------------------------------
def page_game_k_sim() -> None:
    """Simulate game K totals for a selected pitcher."""
    import sys
    sys.path.insert(0, str(PROJECT_ROOT))
    from src.models.game_k_model import compute_k_over_probs, simulate_game_ks
    from src.models.bf_model import get_bf_distribution

    k_samples_dict = load_k_samples()
    bf_priors = load_bf_priors()

    if not k_samples_dict:
        st.warning(
            "No K% posterior samples found. "
            "Run `scripts/precompute_dashboard_data.py` first."
        )
        return

    pitcher_proj = load_projections("pitcher")
    if pitcher_proj.empty:
        st.warning("No pitcher projections found.")
        return

    # Filter to pitchers with K% samples
    available_ids = set(k_samples_dict.keys())
    pitchers_with_samples = pitcher_proj[
        pitcher_proj["pitcher_id"].astype(str).isin(available_ids)
    ].sort_values("pitcher_name")

    if pitchers_with_samples.empty:
        st.warning("No pitchers with K% samples found.")
        return

    # Pitcher selector
    name_to_id = dict(zip(
        pitchers_with_samples["pitcher_name"],
        pitchers_with_samples["pitcher_id"],
    ))
    selected_name = st.selectbox(
        "Select pitcher",
        sorted(name_to_id.keys()),
        key="gamek_pitcher",
    )
    pitcher_id = int(name_to_id[selected_name])
    k_rate_samples = k_samples_dict[str(pitcher_id)]

    # BF parameters
    bf_info = get_bf_distribution(pitcher_id, 2025, bf_priors)
    bf_mu = bf_info["mu_bf"]
    bf_sigma = bf_info["sigma_bf"]

    col1, col2 = st.columns(2)
    with col1:
        bf_mu_adj = st.slider(
            "Expected batters faced",
            min_value=10, max_value=35, value=int(round(bf_mu)),
            help="Adjust based on expected workload",
        )
    with col2:
        st.markdown(
            _metric_card("Projected K%", _fmt_pct(np.mean(k_rate_samples))),
            unsafe_allow_html=True,
        )

    # Simulate
    game_ks = simulate_game_ks(
        pitcher_k_rate_samples=k_rate_samples,
        bf_mu=float(bf_mu_adj),
        bf_sigma=bf_sigma,
        n_draws=10000,
        random_seed=42,
    )

    # K distribution chart
    fig = _create_game_k_fig(game_ks, selected_name)
    st.pyplot(fig, use_container_width=True)
    plt.close(fig)

    # P(over X.5) table
    st.subheader("K Prop Lines")
    k_over = compute_k_over_probs(game_ks)
    # Filter to relevant lines
    k_over = k_over[(k_over["line"] >= 2.5) & (k_over["line"] <= 10.5)].copy()

    # Style the probabilities
    display_lines = []
    for _, row in k_over.iterrows():
        line = row["line"]
        p_over = row["p_over"]
        display_lines.append({
            "Line": f"Over {line:.1f}",
            "P(Over)": f"{p_over:.1%}",
            "P(Under)": f"{1 - p_over:.1%}",
            "Edge Signal": (
                "Strong Over" if p_over > 0.65
                else "Lean Over" if p_over > 0.55
                else "Lean Under" if p_over < 0.45
                else "Strong Under" if p_over < 0.35
                else "Toss-up"
            ),
        })

    st.dataframe(
        pd.DataFrame(display_lines),
        use_container_width=True,
        hide_index=True,
    )

    # Summary stats
    st.markdown("---")
    cols = st.columns(4)
    stats = [
        ("Expected K", f"{np.mean(game_ks):.1f}"),
        ("Std Dev", f"{np.std(game_ks):.1f}"),
        ("Median K", f"{np.median(game_ks):.0f}"),
        ("90th Pctile", f"{np.percentile(game_ks, 90):.0f}"),
    ]
    for col, (label, val) in zip(cols, stats):
        with col:
            st.markdown(_metric_card(label, val), unsafe_allow_html=True)

    st.caption(
        f"Based on {len(game_ks):,} Monte Carlo simulations. "
        f"K% posterior from hierarchical Bayesian model trained on 2018-2025. "
        f"BF distribution: mu={bf_mu_adj}, sigma={bf_sigma:.1f}."
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    # Brand header
    st.markdown(f"""
    <div class="brand-header">
        <div>
            <div class="brand-title">The Data Diamond</div>
            <div class="brand-subtitle">
                Hierarchical Bayesian MLB Projections | 2026 Season
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    if not _check_data_exists():
        st.error(
            "Dashboard data not found. Run the pre-computation first:\n\n"
            "```bash\n"
            "python scripts/precompute_dashboard_data.py\n"
            "```"
        )
        return

    tab1, tab2, tab3 = st.tabs([
        "Projections",
        "Player Profile",
        "Game K Simulator",
    ])

    with tab1:
        page_projections()
    with tab2:
        page_player_profile()
    with tab3:
        page_game_k_sim()


if __name__ == "__main__":
    main()
