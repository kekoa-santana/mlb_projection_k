# The Data Diamond — Visual Style Guide

Reference for maintaining consistent branding across all Data Diamond projects and content.

---

## Color Palette

| Name | Hex | Usage |
|------|-----|-------|
| **Gold** | `#C8A96E` | Primary accent. Titles, projected values, CI bars, KDE fills, delta labels, footer brand name. |
| **Teal** | `#4FC3C8` | Secondary accent. Positive/improvement bars, "increase" direction. |
| **Slate** | `#7B8FA6` | Neutral. Secondary text, observed values, axes, reference lines, footer subtitle, watermark, decline bars. |
| **Cream** | `#F5F2EE` | Background. Figure, axes, and save facecolor. |
| **Dark** | `#0F1117` | Primary text on cream background. Player names, bold labels, axis ticks. |
| **White** | `#FFFFFF` | Rarely used. Available for contrast on dark elements. |

### When to use what
- **Gold** = the thing you want people to look at (projections, headlines, brand name)
- **Teal** = good/positive direction (K% increase for pitchers, K% decrease for hitters)
- **Slate** = context/secondary info (observed values, baselines, supporting text)
- **Dark** = readable body text and bold labels

---

## Typography

| Element | Font | Size | Weight | Color |
|---------|------|------|--------|-------|
| Chart title | sans-serif | 22pt | bold | Gold |
| Chart subtitle | sans-serif | 13pt | normal | Slate |
| Card title | sans-serif | 18pt | bold | Gold |
| Player name (card) | sans-serif | 26pt | bold | Dark |
| Team/hand label | sans-serif | 16pt | normal | Slate |
| Big stat number | sans-serif | 32pt | bold | Dark (observed) / Gold (projected) |
| Stat label | sans-serif | 13pt | normal | Slate |
| Bar chart name | sans-serif | 11pt | bold | Dark |
| Bar chart value | sans-serif | 11pt | bold | Dark |
| Delta label (pp) | sans-serif | 11pt | bold | Gold |
| Observed reference | sans-serif | 9pt | normal | Slate |
| Footer brand | sans-serif | 12pt | bold | Gold |
| Footer subtitle | sans-serif | 10pt | normal | Slate |
| Base rcParam | sans-serif | 14pt | normal | Dark |

**Font family**: Always `sans-serif` (system default). No custom fonts required.

---

## Figure Formats

### 16:9 Landscape (Twitter/X feed cards)
- **Size**: 16 x 9 inches
- **Use for**: Mover charts, comparison tables, leaderboards
- **Layout**: Typically 1x2 subplot grid (left/right panels)

### 1:1 Square (Instagram, individual cards)
- **Size**: 10 x 10 inches
- **Use for**: Individual player cards, single-stat features
- **Layout**: Vertical stack via GridSpec

### Output
- **Format**: PNG
- **DPI**: 300
- **Padding**: 0.3 inches (`savefig.pad_inches`)
- **Save directory**: `outputs/content/`

---

## Chart Styling Rules

### Background & Frame
- Background: Cream (`#F5F2EE`) on figure, axes, and saved file
- **No spines** (top, right, left, bottom all hidden by default)
- **No grid lines**
- Exception: KDE density plots show bottom spine in Slate

### Bars
- **Height**: 0.55 units (horizontal bars)
- **Alpha**: 0.85
- **Color**: Teal for positive direction, Slate for negative direction
- **X-axis range**: 0.05 to 0.45 for K% charts

### Reference Lines
- **Style**: Dashed (`--`)
- **Color**: Slate
- **Width**: 1.5pt
- Marks observed/baseline values

### KDE Density Plots
- **Fill**: Gold at alpha 0.3
- **Outline**: Gold at linewidth 2pt
- **Observed line**: Slate dashed, 2pt
- **Projected line**: Gold solid, 2.5pt
- **Bandwidth**: 0.25 (Scott's Rule variant)

### Credible Interval Bars
- **Line**: Gold, 8pt width, round cap
- **End caps**: Gold `|` marker, size 20
- **Mean marker**: Dark diamond (`D`), size 8
- **Endpoint labels**: 13pt Slate, placed below bar

### Legends
- Background: Cream
- Border: Slate
- Text: Dark
- Font size: 10pt
- Position: upper right (default)

---

## Watermark

- **Text**: "TheDataDiamond"
- **Size**: 60pt
- **Color**: Slate at **alpha 0.03** (barely visible)
- **Rotation**: 30 degrees
- **Position**: Dead center of figure (0.5, 0.5)
- **Z-order**: 0 (behind everything)

The watermark should be invisible at a glance but detectable on close inspection or when image contrast is adjusted.

---

## Brand Footer

Every chart gets a footer with three elements:

```
[Logo]  TheDataDiamond                          Bayesian projection model
^left                                                              right^
```

| Element | Position | Details |
|---------|----------|---------|
| Logo icon | x=0.03, y=0.025 | `iconTransparent.png`, zoom=0.025, no frame |
| Brand name | x=0.078, y=0.02 | "TheDataDiamond", 12pt bold Gold, left-aligned |
| Subtitle | x=0.96, y=0.02 | Default "Bayesian projection model", 10pt Slate, right-aligned |

If the logo file is missing, the brand name shifts left to x=0.04.

---

## Layout Patterns

### Two-Column Mover Chart (16:9)

```
+----------------------------------------------------------+
|          PITCHER K% PROJECTIONS - 2026          (Gold 22pt bold)
|     Bayesian model: 2025 observed vs 2026 projected (Slate 13pt)
|                                                          |
|  PROJECTED K% INCREASE    |    PROJECTED K% DECREASE     |
|  (14pt bold Dark)         |    (14pt bold Dark)           |
|                           |                               |
|  Name (TEAM) HAND  XX.X%  +X.Xpp | Name (TEAM) HAND XX.X% -X.Xpp |
|  2025: XX.X%       [===TEAL===]   | 2025: XX.X%    [===SLATE===]   |
|  ...                      |    ...                        |
|                                                          |
|  [logo] TheDataDiamond              Bayesian projection model |
+----------------------------------------------------------+
```

- Subplot spacing: top=0.80, bottom=0.10, left=0.04, right=0.96, wspace=0.15
- Title at y=0.93, subtitle at y=0.885
- Row spacing: 1.4 units between players

### Individual Player Card (1:1)

```
+---------------------------+
|   K% PROJECTION - 2026    |  (Gold 18pt bold, y=0.97)
|                           |
|       PLAYER NAME         |  (Dark 26pt bold)
|       TEAM  |  HAND       |  (Slate 16pt)
|                           |
|  2025 K%    2026 PROJECTED|
|   XX.X%       XX.X%       |  (Dark 32pt / Gold 32pt)
|                           |
|    [KDE density curve]    |  (Gold fill, Slate observed line)
|          K%               |
|                           |
|   95% CREDIBLE INTERVAL   |  (Dark 13pt bold)
|   lo% [====<>=====] hi%   |  (Gold bar, Dark diamond mean)
|                           |
| [logo] TheDataDiamond  Bayesian projection model |
+---------------------------+
```

- GridSpec: 4 rows, height_ratios=[0.8, 0.8, 3, 1.2]
- Spacing: top=0.92, bottom=0.08, left=0.1, right=0.9, hspace=0.35

---

## Name Formatting

Player names follow this pattern:

```
First Last (TEAM) XHP
```

- **First Last**: Full name (not abbreviated on mover charts)
- **TEAM**: 2-3 letter abbreviation in parentheses. Omit parentheses if no team data.
- **XHP**: `RHP` or `LHP` for pitchers (throwing hand), `RHP` or `LHP` for hitters (batting side)

On individual cards, the name is split:
- Line 1: Full name in all caps (26pt bold Dark)
- Line 2: `TEAM  |  XHP` (16pt Slate)

---

## Implementation

The canonical implementation lives in:
- **`src/viz/theme.py`** — palette constants, `apply_theme()`, `add_watermark()`, `add_brand_footer()`, `save_card()`
- **`src/viz/projections.py`** — chart-specific layouts (mover columns, player cards)

To use in a new project, copy `theme.py` and call:
```python
from theme import apply_theme, add_watermark, add_brand_footer, save_card

apply_theme()           # set rcParams
fig, ax = plt.subplots()
# ... build your chart ...
add_watermark(fig)
add_brand_footer(fig)
save_card(fig, "my_chart", aspect="16:9")
```
