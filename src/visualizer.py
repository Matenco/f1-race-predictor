"""
Generate a standalone interactive HTML report with the prediction.

Self-contained single file: open in any browser with no server, no dependencies.
Plotly charts are embedded inline so the file works offline once generated.

Sections rendered:
1. Hero card: race name, date, circuit, mode (blind / informed)
2. Top-5 prediction with per-position confidence bars
3. Probability heatmap: P(position j | driver i) for the top 12 drivers
4. Monte Carlo bar chart: P(win) / P(podium) / P(top 5)
5. Most likely top-5 combinations table
6. Model validation summary (so the reader knows the historical accuracy)
"""

from __future__ import annotations

import logging
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.io as pio

from . import config
from .predict import RacePrediction

logger = logging.getLogger(__name__)


# F1-inspired colour palette
COLOURS = {
    "background":     "#0f1419",
    "card":           "#1a1f2e",
    "border":         "#2a3142",
    "primary":        "#e10600",   # F1 red
    "secondary":      "#3671c6",   # blue
    "accent":         "#27f4d2",   # teal
    "text":           "#e8e8e8",
    "text_muted":     "#9aa0a6",
    "win":            "#e10600",
    "podium":         "#ff8000",
    "top5":           "#3671c6",
}

TEAM_COLOURS = {
    "Red Bull Racing":    "#3671C6",
    "Ferrari":            "#E8002D",
    "Mercedes":           "#27F4D2",
    "McLaren":            "#FF8000",
    "Aston Martin":       "#229971",
    "Alpine":             "#FF87BC",
    "Williams":           "#64C4FF",
    "Racing Bulls":       "#6692FF",
    "RB":                 "#6692FF",
    "AlphaTauri":         "#5E8FAA",
    "Kick Sauber":        "#52E252",
    "Alfa Romeo":         "#C92D4B",
    "Haas F1 Team":       "#B6BABD",
    "Audi":               "#00D2BE",
    "Cadillac":           "#C9B037",
}

PLOTLY_TEMPLATE = {
    "layout": {
        "paper_bgcolor": COLOURS["card"],
        "plot_bgcolor": COLOURS["card"],
        "font": {"family": "Inter, system-ui, sans-serif", "color": COLOURS["text"], "size": 13},
        "colorway": [COLOURS["primary"], COLOURS["secondary"], COLOURS["accent"]],
        "xaxis": {"gridcolor": COLOURS["border"], "zerolinecolor": COLOURS["border"]},
        "yaxis": {"gridcolor": COLOURS["border"], "zerolinecolor": COLOURS["border"]},
        "margin": {"l": 60, "r": 40, "t": 60, "b": 60},
    }
}


def _team_colour(team: str) -> str:
    return TEAM_COLOURS.get(team, COLOURS["text_muted"])


# =============================================================================
#  Individual chart builders
# =============================================================================
def _build_validation_history_chart(pred: RacePrediction) -> go.Figure | None:
    """Per-race scoring across the walk-forward validation window.

    Shows model vs grid baseline vs form baseline for every CV race, plus a
    rolling 5-race mean. Lets the reader see whether the model consistently
    beats the baseline (rather than just on average).
    """
    if not pred.validation_history:
        return None
    vh = pd.DataFrame(pred.validation_history)
    if len(vh) == 0:
        return None
    vh["label"] = vh.apply(lambda r: f"{int(r['Year'])} R{int(r['Round']):02d}", axis=1)
    x = list(range(len(vh)))

    def _roll(s: pd.Series, w: int = 5) -> pd.Series:
        return s.rolling(window=w, min_periods=1).mean()

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=x, y=vh["XGBoost_Hungarian"],
        mode="lines+markers",
        name="XGBoost + Hungarian",
        line={"color": COLOURS["accent"], "width": 1},
        marker={"size": 6},
        opacity=0.55,
        hovertemplate="<b>%{customdata}</b><br>XGBoost: %{y}/10<extra></extra>",
        customdata=vh["label"],
    ))
    fig.add_trace(go.Scatter(
        x=x, y=_roll(vh["XGBoost_Hungarian"]),
        mode="lines",
        name="XGBoost (5-race avg)",
        line={"color": COLOURS["accent"], "width": 3},
        hoverinfo="skip",
    ))
    fig.add_trace(go.Scatter(
        x=x, y=_roll(vh["Baseline_Grid"]),
        mode="lines",
        name="Grid baseline (5-race avg)",
        line={"color": COLOURS["primary"], "width": 2, "dash": "dash"},
        hoverinfo="skip",
    ))
    fig.add_trace(go.Scatter(
        x=x, y=_roll(vh["Baseline_Form"]),
        mode="lines",
        name="Form baseline (5-race avg)",
        line={"color": COLOURS["podium"], "width": 2, "dash": "dot"},
        hoverinfo="skip",
    ))

    # X-axis ticks: show race labels every ~5 entries to avoid overcrowding
    step = max(1, len(vh) // 10)
    tick_idx = list(range(0, len(vh), step))
    fig.update_layout(
        template=PLOTLY_TEMPLATE,
        title={
            "text": f"Per-race validation score across {len(vh)} held-out races",
            "x": 0.5, "xanchor": "center",
        },
        xaxis={
            "title": "Race (chronological, 2025 → 2026)",
            "tickmode": "array",
            "tickvals": tick_idx,
            "ticktext": [vh["label"].iloc[i] for i in tick_idx],
            "tickangle": -45,
        },
        yaxis={"title": "Points (0–10)", "range": [-0.5, 10.5]},
        height=440,
        legend={"orientation": "h", "yanchor": "bottom", "y": 1.02,
                "xanchor": "right", "x": 1},
    )
    return fig


def _build_top5_chart(pred: RacePrediction) -> go.Figure:
    """Hero chart: bars for the predicted top-5 with team colours and P(top5)."""
    by_abbr = {d.abbreviation: d for d in pred.drivers}
    positions = [p for p, _ in pred.predicted_top5]
    abbrs = [d for _, d in pred.predicted_top5]
    p_top5 = [by_abbr[a].p_top5 * 100 for a in abbrs]
    teams = [by_abbr[a].team for a in abbrs]
    full_names = [by_abbr[a].full_name for a in abbrs]
    colours = [_team_colour(t) for t in teams]

    text_labels = [
        f"<b>P{pos}</b><br>{abbr}<br>{p:.0f}%"
        for pos, abbr, p in zip(positions, abbrs, p_top5, strict=False)
    ]

    fig = go.Figure(go.Bar(
        x=[f"P{p}" for p in positions],
        y=p_top5,
        text=text_labels,
        textposition="inside",
        textfont={"size": 15, "color": "white"},
        marker={"color": colours, "line": {"color": "white", "width": 1}},
        customdata=list(zip(full_names, teams, strict=False)),
        hovertemplate=(
            "<b>%{customdata[0]}</b><br>"
            "Team: %{customdata[1]}<br>"
            "P(top 5): %{y:.1f}%<extra></extra>"
        ),
    ))
    fig.update_layout(
        template=PLOTLY_TEMPLATE,
        title={"text": "Predicted Top 5 — confidence per position", "x": 0.5, "xanchor": "center"},
        xaxis_title=None,
        yaxis_title="P(driver in top 5)",
        yaxis={"ticksuffix": "%", "range": [0, 105]},
        showlegend=False,
        height=380,
    )
    return fig


def _build_probability_heatmap(pred: RacePrediction) -> go.Figure:
    """Heatmap of P(finish position j | driver i) for the top 12 drivers."""
    drivers_sorted = sorted(pred.drivers, key=lambda d: d.p_top5, reverse=True)[:12]
    z = np.array([d.prob_per_position for d in drivers_sorted]) * 100
    y_labels = [f"{d.abbreviation}  ({d.p_top5:.0%})" for d in drivers_sorted]
    x_labels = ["P1", "P2", "P3", "P4", "P5"]

    text = [[f"{v:.0f}%" for v in row] for row in z]

    fig = go.Figure(go.Heatmap(
        z=z,
        x=x_labels,
        y=y_labels,
        text=text,
        texttemplate="%{text}",
        textfont={"size": 12},
        colorscale=[
            [0.0, COLOURS["card"]],
            [0.3, "#3a2030"],
            [0.6, "#a02038"],
            [1.0, COLOURS["primary"]],
        ],
        hovertemplate="<b>%{y}</b><br>%{x}: %{z:.1f}%<extra></extra>",
        showscale=True,
        colorbar={"title": {"text": "P(%)"}, "tickfont": {"color": COLOURS["text"]}},
    ))
    fig.update_layout(
        template=PLOTLY_TEMPLATE,
        title={"text": "Probability matrix — P(finishing position) per driver",
                   "x": 0.5, "xanchor": "center"},
        xaxis_title="Predicted position",
        yaxis_title=None,
        yaxis={"autorange": "reversed"},
        height=520,
    )
    return fig


def _build_monte_carlo_chart(pred: RacePrediction) -> go.Figure:
    """Grouped bars: P(win), P(podium), P(top 5) for the top 12 drivers."""
    sorted_drivers = sorted(pred.drivers, key=lambda d: d.p_top5, reverse=True)[:12]
    abbrs = [d.abbreviation for d in sorted_drivers]
    p_wins = [d.p_win * 100 for d in sorted_drivers]
    p_podiums = [d.p_podium * 100 for d in sorted_drivers]
    p_top5s = [d.p_top5 * 100 for d in sorted_drivers]

    fig = go.Figure()
    fig.add_trace(go.Bar(
        name="P(win)", x=abbrs, y=p_wins,
        marker_color=COLOURS["win"],
        hovertemplate="<b>%{x}</b><br>P(win): %{y:.1f}%<extra></extra>",
    ))
    fig.add_trace(go.Bar(
        name="P(podium)", x=abbrs, y=p_podiums,
        marker_color=COLOURS["podium"],
        hovertemplate="<b>%{x}</b><br>P(podium): %{y:.1f}%<extra></extra>",
    ))
    fig.add_trace(go.Bar(
        name="P(top 5)", x=abbrs, y=p_top5s,
        marker_color=COLOURS["top5"],
        hovertemplate="<b>%{x}</b><br>P(top 5): %{y:.1f}%<extra></extra>",
    ))
    fig.update_layout(
        template=PLOTLY_TEMPLATE,
        title={
            "text": f"Monte Carlo simulation ({pred.n_simulations:,} runs)",
            "x": 0.5, "xanchor": "center",
        },
        barmode="group",
        xaxis_title=None,
        yaxis_title="Probability",
        yaxis={"ticksuffix": "%"},
        height=420,
        legend={"orientation": "h", "yanchor": "bottom", "y": 1.02, "xanchor": "right", "x": 1},
    )
    return fig


# =============================================================================
#  HTML template
# =============================================================================
_HTML_TEMPLATE = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>{race_name} — F1 Prediction</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap" rel="stylesheet">
<style>
  * {{ margin: 0; padding: 0; box-sizing: border-box; }}
  body {{
    font-family: 'Inter', system-ui, -apple-system, sans-serif;
    background: {bg};
    color: {text};
    line-height: 1.6;
    padding: 24px 16px 60px;
  }}
  .container {{ max-width: 1200px; margin: 0 auto; }}

  /* Hero */
  .hero {{
    background: linear-gradient(135deg, {primary} 0%, #7a0300 100%);
    border-radius: 16px;
    padding: 36px 32px;
    margin-bottom: 24px;
    box-shadow: 0 8px 32px rgba(225, 6, 0, 0.25);
  }}
  .hero h1 {{
    font-size: 2.4rem;
    font-weight: 800;
    letter-spacing: -0.02em;
    margin-bottom: 8px;
    color: white;
  }}
  .hero .subtitle {{
    font-size: 1.1rem;
    opacity: 0.92;
    color: white;
  }}
  .hero .badges {{ margin-top: 16px; display: flex; gap: 10px; flex-wrap: wrap; }}
  .badge {{
    background: rgba(255,255,255,0.15);
    padding: 5px 14px;
    border-radius: 20px;
    font-size: 0.85rem;
    font-weight: 500;
    color: white;
    backdrop-filter: blur(8px);
  }}

  /* Cards */
  .card {{
    background: {card};
    border: 1px solid {border};
    border-radius: 14px;
    padding: 24px;
    margin-bottom: 20px;
  }}
  .card h2 {{
    font-size: 1.15rem;
    font-weight: 600;
    margin-bottom: 4px;
    color: {text};
  }}
  .card .card-subtitle {{
    color: {muted};
    font-size: 0.88rem;
    margin-bottom: 16px;
  }}

  .grid-2 {{ display: grid; grid-template-columns: 1fr 1fr; gap: 20px; }}
  @media (max-width: 768px) {{ .grid-2 {{ grid-template-columns: 1fr; }} }}

  /* Top-5 list summary */
  .top5-list {{ display: flex; flex-direction: column; gap: 8px; }}
  .top5-row {{
    display: flex; align-items: center; gap: 14px;
    padding: 12px 16px;
    background: {bg};
    border-radius: 8px;
    border-left: 4px solid;
  }}
  .top5-row .pos {{ font-weight: 800; font-size: 1.3rem; min-width: 36px; }}
  .top5-row .driver {{ flex: 1; }}
  .top5-row .driver-abbr {{ font-weight: 700; font-size: 1.05rem; }}
  .top5-row .driver-name {{ color: {muted}; font-size: 0.85rem; }}
  .top5-row .team-name {{ font-size: 0.85rem; color: {muted}; }}
  .top5-row .prob {{
    font-weight: 600; font-size: 1rem; color: {accent};
    min-width: 70px; text-align: right;
  }}

  /* Combos table */
  table {{
    width: 100%; border-collapse: collapse; font-size: 0.92rem;
  }}
  table th, table td {{
    padding: 10px 12px; text-align: left;
    border-bottom: 1px solid {border};
  }}
  table th {{ font-weight: 600; color: {muted}; font-size: 0.82rem; text-transform: uppercase; letter-spacing: 0.04em; }}
  table tr:hover td {{ background: rgba(255,255,255,0.02); }}
  .combo-cell {{ font-family: 'JetBrains Mono', monospace; font-size: 0.88rem; }}

  /* Stats grid */
  .stats {{
    display: grid; grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
    gap: 16px;
  }}
  .stat {{
    background: {bg};
    padding: 16px 18px;
    border-radius: 10px;
    border: 1px solid {border};
  }}
  .stat .stat-value {{
    font-size: 1.6rem; font-weight: 700; color: {accent};
    margin-bottom: 4px;
  }}
  .stat .stat-label {{ font-size: 0.82rem; color: {muted}; }}

  /* Footer */
  .footer {{
    text-align: center;
    color: {muted};
    font-size: 0.82rem;
    margin-top: 40px;
    padding-top: 20px;
    border-top: 1px solid {border};
  }}
  .footer code {{
    background: {card};
    padding: 2px 6px;
    border-radius: 4px;
    font-size: 0.85em;
  }}
  a {{ color: {secondary}; text-decoration: none; }}
  a:hover {{ text-decoration: underline; }}

  /* Last race recap */
  .recap-card {{
    background: {card};
    border: 1px solid {border};
    border-radius: 14px;
    padding: 24px;
    margin-bottom: 20px;
  }}
  .recap-header {{
    display: flex; align-items: center; justify-content: space-between;
    flex-wrap: wrap; gap: 12px; margin-bottom: 16px;
  }}
  .recap-title h2 {{ font-size: 1.15rem; font-weight: 600; }}
  .recap-title .recap-sub {{ color: {muted}; font-size: 0.88rem; margin-top: 2px; }}
  .recap-score {{
    display: flex; align-items: center; gap: 16px;
  }}
  .recap-score .score-big {{
    background: {accent};
    color: #0a1018;
    padding: 8px 16px; border-radius: 10px;
    font-weight: 700; font-size: 1.4rem;
  }}
  .recap-score .score-vs {{ color: {muted}; font-size: 0.85rem; }}
  .recap-cols {{
    display: grid; grid-template-columns: 1fr 1fr; gap: 18px;
    margin-bottom: 14px;
  }}
  @media (max-width: 700px) {{ .recap-cols {{ grid-template-columns: 1fr; }} }}
  .recap-col h3 {{
    font-size: 0.78rem; text-transform: uppercase; letter-spacing: 0.06em;
    color: {muted}; margin-bottom: 10px; font-weight: 600;
  }}
  .recap-row {{
    display: flex; align-items: center; gap: 10px;
    padding: 8px 12px; margin-bottom: 4px;
    background: {bg}; border-radius: 6px; font-size: 0.92rem;
  }}
  .recap-row .pos {{
    background: {border}; color: {text};
    width: 28px; height: 28px; border-radius: 4px;
    display: flex; align-items: center; justify-content: center;
    font-weight: 700; font-size: 0.85rem;
  }}
  .recap-row .driver {{ font-weight: 600; flex: 1; }}
  .recap-row.exact-hit {{ background: rgba(39, 244, 210, 0.12); border-left: 3px solid {accent}; }}
  .recap-row.exact-hit .pos {{ background: {accent}; color: #0a1018; }}
  .recap-row.partial-hit {{ background: rgba(255, 128, 0, 0.10); border-left: 3px solid {podium}; }}
  .recap-row.miss {{ opacity: 0.55; }}
  .recap-legend {{
    margin-top: 6px; font-size: 0.82rem; color: {muted};
    display: flex; gap: 16px; flex-wrap: wrap;
  }}
  .recap-legend .leg-dot {{
    display: inline-block; width: 10px; height: 10px; border-radius: 2px;
    margin-right: 6px; vertical-align: middle;
  }}

  /* Validation prose */
  .val-prose {{
    background: {bg}; border-radius: 8px; padding: 14px 18px;
    margin: 14px 0 18px; font-size: 0.95rem; line-height: 1.65;
    border-left: 3px solid {accent};
  }}
  .val-prose .lift-positive {{ color: {accent}; font-weight: 700; }}
  .val-prose .lift-negative {{ color: {primary}; font-weight: 700; }}
</style>
</head>
<body>
<div class="container">

  <!-- HERO -->
  <div class="hero">
    <h1>{race_name}</h1>
    <div class="subtitle">{country} &middot; {circuit} &middot; {race_date}</div>
    <div class="badges">
      <span class="badge">Round {round_number} of {year}</span>
      <span class="badge">{mode_label}</span>
      <span class="badge">XGBoost + Hungarian assignment</span>
      <span class="badge">{n_sim:,} Monte Carlo runs</span>
    </div>
  </div>

  <!-- LAST RACE RECAP (only rendered when prior prediction has actual results) -->
  {recap_html}

  <!-- TOP 5 SUMMARY + CHART -->
  <div class="grid-2">
    <div class="card">
      <h2>Predicted Top 5</h2>
      <div class="card-subtitle">Optimal assignment via Hungarian algorithm</div>
      <div class="top5-list">{top5_html}</div>
      {sixth_html}
    </div>
    <div class="card">
      {top5_chart}
    </div>
  </div>

  <!-- HEATMAP -->
  <div class="card">
    <h2>Position-by-position probability matrix</h2>
    <div class="card-subtitle">For each driver, the model's estimated probability of finishing at each position. Brighter = more likely.</div>
    {heatmap_chart}
  </div>

  <!-- MONTE CARLO -->
  <div class="card">
    <h2>Monte Carlo outcomes</h2>
    <div class="card-subtitle">Win / podium / top-5 frequencies over {n_sim:,} simulated races.</div>
    {monte_carlo_chart}
  </div>

  <!-- COMBOS -->
  <div class="card">
    <h2>Most likely Top-5 combinations</h2>
    <div class="card-subtitle">Most frequent permutations across simulations.</div>
    <table>
      <thead><tr><th>#</th><th>Frequency</th><th>Combination (P1 → P5)</th></tr></thead>
      <tbody>{combos_html}</tbody>
    </table>
  </div>

  <!-- VALIDATION — expanded -->
  <div class="card">
    <h2>How well does this model actually work?</h2>
    <div class="card-subtitle">Walk-forward time-series cross-validation: for each race in 2025–2026, the model was trained on every earlier race and predicted the held-out one. Numbers below are out of sample.</div>
    {validation_prose}
    <div class="stats">{stats_html}</div>
    {validation_chart}
  </div>

  <div class="footer">
    Generated {generated_at} &middot;
    <code>python pipeline.py</code> &middot;
    Data: <a href="https://github.com/theOehrly/Fast-F1" target="_blank">FastF1</a>
  </div>
</div>
</body>
</html>
"""


# =============================================================================
#  Top-5 list rendering
# =============================================================================
def _render_top5_list(pred: RacePrediction) -> str:
    by_abbr = {d.abbreviation: d for d in pred.drivers}
    rows = []
    for pos, drv in pred.predicted_top5:
        d = by_abbr[drv]
        rows.append(f"""
        <div class="top5-row" style="border-left-color: {_team_colour(d.team)};">
          <div class="pos">P{pos}</div>
          <div class="driver">
            <div class="driver-abbr">{drv}</div>
            <div class="team-name">{d.team}</div>
          </div>
          <div class="prob">{d.p_top5*100:.0f}%</div>
        </div>""")
    return "".join(rows)


def _render_sixth(pred: RacePrediction) -> str:
    if not pred.sixth_driver:
        return ""
    sixth = next(d for d in pred.drivers if d.abbreviation == pred.sixth_driver)
    return f"""
    <div style="margin-top: 12px; padding: 10px 14px; background: {COLOURS['bg']
    if 'bg' in COLOURS else COLOURS['background']}; border-radius: 8px;
    border-left: 3px solid {COLOURS['text_muted']}; font-size: 0.88rem;">
      <span style="color: {COLOURS['text_muted']};">First reserve:</span>
      <strong>{sixth.abbreviation}</strong>
      <span style="color: {COLOURS['text_muted']};">({sixth.team}) &middot;
      P(top 5) = {sixth.p_top5*100:.0f}%</span>
    </div>"""


def _render_combos_table(pred: RacePrediction) -> str:
    rows = []
    for i, (combo, count) in enumerate(pred.most_likely_combos, 1):
        freq = count / pred.n_simulations * 100
        rows.append(
            f"<tr><td>{i}</td><td>{freq:.2f}%</td>"
            f"<td class='combo-cell'>{' &rarr; '.join(combo)}</td></tr>"
        )
    return "".join(rows)


def _render_validation_stats(pred: RacePrediction) -> str:
    summary = pred.model_validation_summary
    if not summary:
        return (
            "<div class='stat'><div class='stat-value'>—</div>"
            "<div class='stat-label'>No validation data saved</div></div>"
        )
    items = [
        ("XGBoost + Hungarian", f"{summary.get('xgboost_hungarian_mean', 0):.2f}",
         f"± {summary.get('xgboost_hungarian_std', 0):.2f} pts/race"),
        ("Grid baseline", f"{summary.get('baseline_grid_mean', 0):.2f}",
         "pts/race"),
        ("Form baseline", f"{summary.get('baseline_form_mean', 0):.2f}",
         "pts/race"),
        ("Position RMSE", f"{summary.get('rmse_mean', 0):.2f}",
         "predicted vs actual"),
        ("Validation races", f"{summary.get('n_races', 0)}",
         "walk-forward CV"),
    ]
    return "".join(
        f"<div class='stat'><div class='stat-value'>{v}</div>"
        f"<div class='stat-label'>{label}<br><span style='font-size:0.75rem;'>{detail}</span></div></div>"
        for label, v, detail in items
    )


def _render_validation_prose(pred: RacePrediction) -> str:
    """Plain-language interpretation of the validation numbers."""
    summary = pred.model_validation_summary
    if not summary:
        return ""
    xgb_mean = summary.get("xgboost_hungarian_mean", 0)
    grid_mean = summary.get("baseline_grid_mean", 0)
    form_mean = summary.get("baseline_form_mean", 0)
    n_races = int(summary.get("n_races", 0))

    n_beat_grid = n_beat_form = 0
    if pred.validation_history:
        vh = pd.DataFrame(pred.validation_history)
        n_beat_grid = int((vh["XGBoost_Hungarian"] > vh["Baseline_Grid"]).sum())
        n_beat_form = int((vh["XGBoost_Hungarian"] > vh["Baseline_Form"]).sum())

    lift_grid = xgb_mean - grid_mean
    lift_form = xgb_mean - form_mean
    lift_grid_class = "lift-positive" if lift_grid > 0 else "lift-negative"
    lift_form_class = "lift-positive" if lift_form > 0 else "lift-negative"

    parts = [
        f"<p>Across <b>{n_races} held-out races</b>, the model averaged "
        f"<b>{xgb_mean:.2f}</b> points out of 10 per race. "
        f"The starting grid as-is would have scored <b>{grid_mean:.2f}</b>, "
        f"and a recent-form ranking would have scored <b>{form_mean:.2f}</b>.</p>",
        f"<p>That's a <span class='{lift_grid_class}'>"
        f"{'+' if lift_grid >= 0 else ''}{lift_grid:.2f} point lift over the grid baseline</span> "
        f"and a <span class='{lift_form_class}'>"
        f"{'+' if lift_form >= 0 else ''}{lift_form:.2f} point lift over the form baseline</span>. "
        f"The grid baseline is hard to beat &mdash; qualifying position correlates ~0.7 with "
        f"finish position, so any model has to add real signal beyond &quot;the fastest qualifier "
        f"wins&quot;.</p>",
    ]
    if pred.validation_history:
        parts.append(
            f"<p>Race-by-race, the model beat the grid baseline in "
            f"<b>{n_beat_grid}/{n_races} races ({n_beat_grid/n_races:.0%})</b> "
            f"and the form baseline in <b>{n_beat_form}/{n_races} races "
            f"({n_beat_form/n_races:.0%})</b>.</p>"
        )
    return f"<div class='val-prose'>{''.join(parts)}</div>"


def _render_recap(pred: RacePrediction) -> str:
    """Render the 'Last race scorecard' card showing predicted vs actual."""
    recap = pred.last_race_recap
    if recap is None:
        return ""

    exact_set = set(recap.exact_hits)
    partial_set = set(recap.in_top5_hits)

    pred_rows = []
    for pos, drv in recap.predicted_top5:
        if drv in exact_set:
            cls, marker = "exact-hit", "&#10003;&nbsp;exact"
        elif drv in partial_set:
            actual_pos = next(
                (p for p, d in recap.actual_top5 if d == drv), None,
            )
            cls = "partial-hit"
            marker = f"in top 5 (actual P{actual_pos})" if actual_pos else "in top 5"
        else:
            cls = "miss"
            marker = "missed"
        pred_rows.append(
            f"<div class='recap-row {cls}'>"
            f"<div class='pos'>P{pos}</div>"
            f"<div class='driver'>{drv}</div>"
            f"<div style='font-size:0.82rem; color:{COLOURS['text_muted']};'>{marker}</div>"
            f"</div>"
        )

    actual_rows = [
        f"<div class='recap-row'>"
        f"<div class='pos'>P{pos}</div><div class='driver'>{drv}</div></div>"
        for pos, drv in recap.actual_top5
    ]

    score_diff = recap.score - recap.grid_baseline_score
    if score_diff > 0:
        diff_label = f"+{score_diff} vs grid"
    elif score_diff < 0:
        diff_label = f"{score_diff} vs grid"
    else:
        diff_label = "tied with grid"

    return f"""
  <div class="recap-card">
    <div class="recap-header">
      <div class="recap-title">
        <h2>Last race scorecard &mdash; {recap.race_name}</h2>
        <div class="recap-sub">{recap.race_date} &middot; predicted vs actual</div>
      </div>
      <div class="recap-score">
        <div class="score-big">{recap.score} / 10</div>
        <div class="score-vs">grid baseline: {recap.grid_baseline_score}/10<br>{diff_label}</div>
      </div>
    </div>
    <div class="recap-cols">
      <div class="recap-col">
        <h3>What we predicted</h3>
        {''.join(pred_rows)}
      </div>
      <div class="recap-col">
        <h3>What actually happened</h3>
        {''.join(actual_rows)}
      </div>
    </div>
    <div class="recap-legend">
      <span><span class="leg-dot" style="background:{COLOURS['accent']};"></span>Exact-position hit (+2 pts)</span>
      <span><span class="leg-dot" style="background:{COLOURS['podium']};"></span>In top 5, wrong position (+1 pt)</span>
      <span><span class="leg-dot" style="background:{COLOURS['border']};"></span>Missed top 5 (0 pts)</span>
    </div>
  </div>"""


# =============================================================================
#  Public API
# =============================================================================
def render_html(pred: RacePrediction, output_path: Path | None = None) -> Path:
    """Render the prediction as a standalone HTML file and return its path."""
    if output_path is None:
        slug = pred.next_race.name.lower().replace(" ", "_").replace("'", "")
        date_str = pred.next_race.date.strftime("%Y%m%d")
        output_path = config.PREDICTIONS_DIR / f"{date_str}_{slug}.html"

    pio.templates["f1_dark"] = PLOTLY_TEMPLATE
    chart_kwargs = {"include_plotlyjs": "cdn", "full_html": False,
                    "config": {"displayModeBar": False}}

    top5_chart_html = _build_top5_chart(pred).to_html(**chart_kwargs)
    heatmap_html = _build_probability_heatmap(pred).to_html(**chart_kwargs)
    mc_html = _build_monte_carlo_chart(pred).to_html(**chart_kwargs)
    val_chart_fig = _build_validation_history_chart(pred)
    val_chart_html = val_chart_fig.to_html(**chart_kwargs) if val_chart_fig else ""

    mode_label = "AFTER QUALIFYING (informed)" if pred.mode == "informed" else "BEFORE QUALIFYING (blind)"

    html = _HTML_TEMPLATE.format(
        race_name=pred.next_race.name,
        country=pred.next_race.country,
        circuit=pred.next_race.circuit,
        race_date=pred.next_race.date.strftime("%A, %B %d, %Y"),
        round_number=pred.next_race.round_number,
        year=pred.next_race.year,
        mode_label=mode_label,
        n_sim=pred.n_simulations,
        recap_html=_render_recap(pred),
        top5_html=_render_top5_list(pred),
        sixth_html=_render_sixth(pred),
        top5_chart=top5_chart_html,
        heatmap_chart=heatmap_html,
        monte_carlo_chart=mc_html,
        combos_html=_render_combos_table(pred),
        validation_prose=_render_validation_prose(pred),
        stats_html=_render_validation_stats(pred),
        validation_chart=val_chart_html,
        generated_at=datetime.now().strftime("%Y-%m-%d %H:%M"),
        bg=COLOURS["background"],
        card=COLOURS["card"],
        border=COLOURS["border"],
        primary=COLOURS["primary"],
        secondary=COLOURS["secondary"],
        accent=COLOURS["accent"],
        text=COLOURS["text"],
        muted=COLOURS["text_muted"],
        podium=COLOURS["podium"],
    )

    output_path.write_text(html, encoding="utf-8")
    logger.info("HTML report written to %s", output_path)
    return output_path
