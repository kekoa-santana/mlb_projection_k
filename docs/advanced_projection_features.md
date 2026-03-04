# Advanced Feature Engineering for MLB Projection Systems
### High-Signal Features Used in Professional Baseball Modeling

These features represent some of the most powerful signals used in professional projection systems. Many public models rely heavily on season-level statistics, but front offices emphasize **underlying skill stability, trajectory, and contextual skill interactions**.

The features below are designed to integrate cleanly with the **Data Diamond projection architecture** and can be implemented using Statcast pitch-level and batted-ball data.

---

# 1. Stabilized Contact Quality

Raw xwOBA fluctuates heavily over short windows. What matters more for projections is **how stable a hitter's contact quality is over time**.

### Key Idea
Instead of relying on a single average xwOBA value, track **rolling convergence windows** and measure variance.

### Example Features

- `rolling_200_bip_xwoba`
- `rolling_400_bip_xwoba`
- `rolling_800_bip_xwoba`
- `xwoba_trend`
- `xwoba_variance`

### Why It Matters

Two hitters may have identical xwOBA:

| Player | xwOBA | Variance |
|------|------|------|
| Player A | .340 | low |
| Player B | .340 | high |

Player A is far more predictable and easier to project.

Variance itself becomes a predictive feature.

---

# 2. Aging Curve Delta

Most projection models include age as a feature. This is insufficient.

Instead measure **performance relative to expected aging curves**.

### Example
expected_wRC_plus(age)
actual_wRC_plus
aging_delta = actual - expected


### Features

- `aging_delta`
- `aging_delta_trend`

### Why It Matters

This identifies players who are:

- Late bloomers
- Early decliners
- Aging faster or slower than expected

---

# 3. Pitch Arsenal Shape Clustering

Pitch types alone are a poor representation of pitcher skill.

Instead cluster pitchers by **pitch shape characteristics**.

### Inputs

- `release_speed`
- `pfx_x`
- `pfx_z`
- `spin_rate`
- `release_height`
- `release_side`
- `extension`

### Method

Apply clustering such as:

- K-Means
- Gaussian Mixture Models

### Example Archetypes

| Cluster | Description |
|-------|------------|
| 1 | High-ride 4-seamers |
| 2 | Sinker/slider groundball profiles |
| 3 | Sweeper-heavy arsenals |
| 4 | Changeup specialists |

### Features

- `pitch_archetype`
- `cluster_avg_whiff_rate`
- `cluster_avg_run_value`

### Why It Matters

Pitchers tend to perform similarly within **arsenal archetypes**.

This allows models to borrow strength from similar pitchers.

---

# 4. Plate Discipline Stability

Plate discipline metrics stabilize faster than most hitting statistics.

Important metrics:

- `zone_swing_rate`
- `chase_rate`
- `contact_rate`
- `zone_contact_rate`

### Stability Features

- `chase_rate_rolling`
- `chase_rate_delta`
- `contact_rate_variance`

### Why It Matters

Hitters often show decline signals through discipline metrics **before production drops**.

---

# 5. Pitch Sequencing Entropy

Pitch predictability is an underused feature.

Measure how predictable a pitcher's sequencing patterns are.

### Concept

Compute conditional pitch probabilities:
P(pitch_type_t | pitch_type_t-1)


Then compute **entropy**.

### Feature

`pitch_sequence_entropy`

### Why It Matters

Low entropy pitchers are predictable and often degrade faster once hitters adjust.

---

# 6. Velocity Trend Acceleration

Velocity itself matters, but the **rate of change in velocity** is even more predictive.

### Example Features

- `velo_last_30_days`
- `velo_last_90_days`
- `velo_last_365_days`

Derived metrics:

- `velo_trend`
- `velo_acceleration`

### Why It Matters

Velocity loss can signal:

- injury risk
- mechanical breakdown
- aging effects

The **second derivative** (acceleration) often predicts performance drops before ERA rises.

---

# 7. Contact Suppression vs Whiff Skill

Pitchers suppress offense through two different skill pathways:

| Skill Type | Description |
|-----------|-------------|
| Whiff Skill | Missing bats |
| Contact Suppression | Weak contact |

### Example Features

Whiff-based:

- `whiff_rate`
- `swinging_strike_rate`

Contact-based:

- `xwoba_on_contact`
- `hard_hit_rate`
- `barrel_rate`

### Why It Matters

Pitchers who rely on contact suppression often show **higher year-to-year stability** than pure strikeout pitchers.

Separating these skill types improves projections.

---

# 8. Role Adjustment (Starter vs Reliever)

Relievers often appear more dominant because of:

- shorter outings
- higher effort
- favorable matchups

### Features

- `starter_ip_percent`
- `reliever_ip_percent`
- `times_through_order_penalty`

### Why It Matters

Without role adjustments, projection models often **overestimate reliever talent**.

---

# 9. Park and Environment Normalization

Ballpark environments influence batted-ball outcomes.

Adjust features using environmental context.

### Variables

- `park_factor`
- `altitude`
- `temperature`
- `roof_open`

### Why It Matters

Pitchers and hitters should be evaluated relative to environment-adjusted baselines.

---

# Key Insight

Projection performance is driven primarily by **feature engineering**, not algorithm complexity.

Typical impact distribution:
70% feature engineering
20% data quality and cleaning
10% modeling algorithm


A well-engineered feature set combined with a simple model such as **XGBoost or hierarchical Bayesian regression** often outperforms complex models built on weak features.

---

# Future Expansion: Player Similarity Embeddings

Many professional systems use similarity-based projections.

### Concept

Embed players into a feature space using:

- pitch shape
- contact quality
- discipline metrics
- aging trajectories

Then identify:
nearest historical player comps


Future performance can be estimated from the trajectory of similar players.

This approach can complement Bayesian projections.

---

# Integration with Data Diamond Projection System

These features integrate naturally into the three-layer architecture:

| Layer | Feature Types |
|-----|------|
| Season Talent Model | Aging delta, discipline stability |
| Matchup Model | Arsenal clustering, sequencing entropy |
| Game-Level Prediction | Velocity trends, role adjustments |

---

# Implementation Notes

- Compute features in `feature_eng.py`
- Cache intermediate datasets as Parquet files
- Validate features with walk-forward backtesting
- Ensure features use **only historical information** to prevent leakage