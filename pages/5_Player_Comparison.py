"""
pages/5_Player_Comparison.py – SUPRR Selector | Player Comparison
Interactive side-by-side analysis for 2 or 3 batters or bowlers.
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Player Comparison | SUPRR Selector",
    page_icon="🥊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Global CSS ────────────────────────────────────────────────────────────────
st.markdown(
    """
    <style>
    .stApp { background-color: #0A0F1E; }
    .stApp * { color: #FFFFFF; }
    [data-testid="stSidebar"] { background-color: #0D1527; }

    .metric-card {
        background: #141929;
        border: 1px solid #1E2F4A;
        border-radius: 12px;
        padding: 20px;
        text-align: center;
        height: 100%;
    }
    .metric-value { font-size: 2.2rem; font-weight: 700; color: #00D4A1; }
    .metric-label { font-size: 0.85rem; color: #A0ADB8; margin-top: 4px; }

    .player-card {
        background: linear-gradient(135deg, #141929, #1a2a4a);
        border: 1px solid #1E2F4A;
        border-top: 3px solid #00D4A1;
        border-radius: 12px;
        padding: 20px 24px;
        text-align: center;
        height: 100%;
    }
    .player-name { font-size: 1.4rem; font-weight: 800; color: #FFFFFF; }
    .player-team { font-size: 0.88rem; color: #A0ADB8; margin-top: 4px; }
    .format-badge {
        display: inline-block;
        background: #1E2F4A;
        color: #4A9EFF;
        font-size: 0.78rem;
        font-weight: 700;
        padding: 3px 10px;
        border-radius: 20px;
        margin-top: 8px;
    }
    .rank-badge {
        display: inline-block;
        background: linear-gradient(135deg, #00D4A1, #4A9EFF);
        color: #0A0F1E;
        font-size: 0.82rem;
        font-weight: 800;
        padding: 4px 14px;
        border-radius: 20px;
        margin-top: 6px;
    }
    .kpi-row {
        background: #141929;
        border: 1px solid #1E2F4A;
        border-radius: 8px;
        padding: 10px 14px;
        margin-bottom: 6px;
    }
    .kpi-label { font-size: 0.82rem; color: #A0ADB8; }
    .kpi-value { font-size: 1.1rem; font-weight: 700; color: #FFFFFF; }
    .kpi-winner { font-size: 1.1rem; font-weight: 700; color: #00D4A1; }
    .kpi-loser  { font-size: 1.1rem; font-weight: 700; color: #A0ADB8; }

    .section-header {
        font-size: 1.3rem; font-weight: 700; color: #FFFFFF;
        margin-top: 32px; margin-bottom: 16px;
        padding-bottom: 8px; border-bottom: 1px solid #1E2F4A;
    }
    .page-header {
        background: linear-gradient(135deg, #141929 0%, #1a2540 100%);
        border: 1px solid #1E2F4A;
        border-left: 4px solid #00D4A1;
        border-radius: 12px;
        padding: 24px 28px;
        margin-bottom: 24px;
    }
    .page-title { font-size: 2rem; font-weight: 800; color: #FFFFFF; margin: 0; }
    .page-desc  { font-size: 0.92rem; color: #A0ADB8; margin-top: 8px; line-height: 1.7; }
    </style>
    """,
    unsafe_allow_html=True,
)

# ── Chart theme ───────────────────────────────────────────────────────────────
_CHART_LAYOUT = dict(
    paper_bgcolor="#1E2538",
    plot_bgcolor="#1E2538",
    font=dict(color="#FFFFFF", size=12),
    margin=dict(l=40, r=20, t=50, b=40),
)
_PLAYER_COLORS = ["#00D4A1", "#4A9EFF", "#FF9F40"]

# ── Pipeline ──────────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner="Loading analytics pipeline...")
def load_pipeline():
    import warnings
    warnings.filterwarnings("ignore")
    from ranking_engine import run_full_pipeline
    return run_full_pipeline()


pipeline = load_pipeline()
batter_rankings = pipeline["batter_rankings"]
bowler_rankings = pipeline["bowler_rankings"]
batter_full = pipeline["batter_full"]
bowler_full = pipeline["bowler_full"]
full_df = pipeline["full_df"]
data_summary = pipeline["data_summary"]


# ── Helpers ───────────────────────────────────────────────────────────────────
def _fmt(val, decimals: int = 1, suffix: str = "") -> str:
    """Format a numeric value safely, returning 'N/A' for NaN."""
    try:
        if val is None or (isinstance(val, float) and np.isnan(val)):
            return "N/A"
        return f"{float(val):.{decimals}f}{suffix}"
    except Exception:
        return "N/A"


def _safe_float(val, default: float = np.nan) -> float:
    try:
        f = float(val)
        return default if np.isnan(f) else f
    except Exception:
        return default


def _get_rank1_name(rankings: pd.DataFrame, full: pd.DataFrame, rank: int = 1) -> str:
    """Return the player name at a given rank, with fallback."""
    try:
        if rankings.empty or full.empty:
            names = full["player_name"].dropna().unique()
            return names[rank - 1] if len(names) >= rank else ""
        row = rankings[rankings["rank"] == rank]
        if not row.empty and "player_name" in row.columns:
            return str(row.iloc[0]["player_name"])
        # Fallback: sorted full frame
        sorted_full = full.sort_values("adjusted_score", ascending=False)
        return str(sorted_full.iloc[rank - 1]["player_name"])
    except Exception:
        return ""


def _lookup(full: pd.DataFrame, name: str, col: str):
    """Return cell value for player `name` in `col`; NaN on miss."""
    try:
        rows = full[full["player_name"] == name]
        if rows.empty or col not in rows.columns:
            return np.nan
        return rows.iloc[0][col]
    except Exception:
        return np.nan


def _normalize_series(values: list) -> list:
    """Min-max scale a list of floats to [0, 100], NaN → 0."""
    cleaned = [v if (v is not None and not np.isnan(v)) else 0.0 for v in values]
    mn, mx = min(cleaned), max(cleaned)
    if mx == mn:
        return [50.0] * len(cleaned)
    return [(v - mn) / (mx - mn) * 100.0 for v in cleaned]


# ─────────────────────────────────────────────────────────────────────────────
# Sidebar
# ─────────────────────────────────────────────────────────────────────────────
st.sidebar.markdown("## 🥊 Comparison Controls")

comp_type = st.sidebar.radio(
    "Comparison type",
    options=["Batters", "Bowlers"],
    index=0,
    horizontal=True,
)

n_players = st.sidebar.radio(
    "Number of players",
    options=[2, 3],
    index=0,
    horizontal=True,
)

# Player selector depends on type
if comp_type == "Batters":
    if batter_full.empty:
        st.error("No batter data available. Run the pipeline first.")
        st.stop()

    all_names = sorted(batter_full["player_name"].dropna().unique().tolist())
    defaults = [_get_rank1_name(batter_rankings, batter_full, r) for r in range(1, n_players + 1)]
    defaults = [d for d in defaults if d in all_names]
    while len(defaults) < n_players and all_names:
        for n in all_names:
            if n not in defaults:
                defaults.append(n)
                break

    selected = []
    for i in range(n_players):
        def_val = defaults[i] if i < len(defaults) else all_names[0]
        label = f"Player {i + 1}"
        chosen = st.sidebar.selectbox(
            label,
            options=all_names,
            index=all_names.index(def_val) if def_val in all_names else 0,
            key=f"bat_sel_{i}",
        )
        selected.append(chosen)
else:
    if bowler_full.empty:
        st.error("No bowler data available. Run the pipeline first.")
        st.stop()

    all_names = sorted(bowler_full["player_name"].dropna().unique().tolist())
    defaults = [_get_rank1_name(bowler_rankings, bowler_full, r) for r in range(1, n_players + 1)]
    defaults = [d for d in defaults if d in all_names]
    while len(defaults) < n_players and all_names:
        for n in all_names:
            if n not in defaults:
                defaults.append(n)
                break

    selected = []
    for i in range(n_players):
        def_val = defaults[i] if i < len(defaults) else all_names[0]
        label = f"Player {i + 1}"
        chosen = st.sidebar.selectbox(
            label,
            options=all_names,
            index=all_names.index(def_val) if def_val in all_names else 0,
            key=f"bowl_sel_{i}",
        )
        selected.append(chosen)

# Deduplicate while preserving order
seen = set()
selected_unique = []
for s in selected:
    if s not in seen:
        seen.add(s)
        selected_unique.append(s)
selected = selected_unique

# Format filter (cosmetic – shown in sidebar)
st.sidebar.markdown("---")
fmt_filter = st.sidebar.selectbox(
    "Format context",
    options=["All", "IPL", "T20I"],
    index=0,
)

# ─────────────────────────────────────────────────────────────────────────────
# Page header
# ─────────────────────────────────────────────────────────────────────────────
st.markdown(
    """
    <div class="page-header">
        <div class="page-title">🥊 Player Comparison</div>
        <div class="page-desc">
            Head-to-head pressure performance analysis. Compare up to three players
            across pressure metrics, format consistency, sample context, and composite ratings.
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

# ─────────────────────────────────────────────────────────────────────────────
# BATTER COMPARISON
# ─────────────────────────────────────────────────────────────────────────────
if comp_type == "Batters":
    df_ref = batter_full

    # ── 1. Player Cards ───────────────────────────────────────────────────────
    st.markdown('<div class="section-header">Player Overview</div>', unsafe_allow_html=True)
    cols = st.columns(len(selected))
    for i, name in enumerate(selected):
        team = _lookup(df_ref, name, "main_team_name") or _lookup(df_ref, name, "main_team_id")
        fmt_cov = _lookup(df_ref, name, "format_coverage") or "—"
        rank_val = _lookup(df_ref, name, "rank")
        rank_str = f"Rank #{int(rank_val)}" if not (isinstance(rank_val, float) and np.isnan(rank_val)) else "—"
        color = _PLAYER_COLORS[i % len(_PLAYER_COLORS)]
        with cols[i]:
            st.markdown(
                f"""
                <div class="player-card" style="border-top-color:{color}">
                    <div class="player-name">{name}</div>
                    <div class="player-team">{team if pd.notna(team) else '—'}</div>
                    <div><span class="format-badge">{fmt_cov}</span></div>
                    <div><span class="rank-badge">{rank_str}</span></div>
                </div>
                """,
                unsafe_allow_html=True,
            )

    # ── 2. KPI Comparison ─────────────────────────────────────────────────────
    st.markdown('<div class="section-header">Head-to-Head Metrics</div>', unsafe_allow_html=True)

    batter_kpis = [
        ("Pressure Rating",   "adjusted_score",           "higher", 1),
        ("Pressure SR",       "pressure_strike_rate",     "higher", 1),
        ("Baseline SR",       "baseline_sr",              "higher", 1),
        ("SR Improvement",    "sr_improvement",           "higher", 1),
        ("Pressure Balls",    "pressure_balls",           "higher", 0),
        ("Pressure Matches",  "pressure_matches",         "higher", 0),
        ("Dismissal Rate %",  "pressure_dismissal_rate",  "lower",  2),
        ("Boundary %",        "pressure_boundary_pct",   "higher", 1),
        ("Dot Ball %",        "pressure_dot_pct",         "lower",  1),
        ("IPL Pressure SR",   "ipl_pressure_sr",          "higher", 1),
        ("T20I Pressure SR",  "t20i_pressure_sr",         "higher", 1),
    ]

    # Format filter: override SR column display
    if fmt_filter == "IPL":
        batter_kpis[1] = ("IPL Pressure SR", "ipl_pressure_sr", "higher", 1)
    elif fmt_filter == "T20I":
        batter_kpis[1] = ("T20I Pressure SR", "t20i_pressure_sr", "higher", 1)

    kpi_cols = st.columns([2] + [1] * len(selected))
    kpi_cols[0].markdown("**Metric**")
    for i, name in enumerate(selected):
        kpi_cols[i + 1].markdown(f"**{name.split()[-1]}**")

    for label, col_name, direction, decimals in batter_kpis:
        vals = [_safe_float(_lookup(df_ref, name, col_name)) for name in selected]
        finite_vals = [v for v in vals if not np.isnan(v)]
        if finite_vals:
            best_val = max(finite_vals) if direction == "higher" else min(finite_vals)
        else:
            best_val = None

        row_cols = st.columns([2] + [1] * len(selected))
        row_cols[0].markdown(f'<div class="kpi-label">{label}</div>', unsafe_allow_html=True)
        for i, v in enumerate(vals):
            is_winner = (best_val is not None) and (not np.isnan(v)) and (v == best_val)
            css_class = "kpi-winner" if is_winner else "kpi-value"
            suffix = " ✓" if is_winner else ""
            row_cols[i + 1].markdown(
                f'<div class="{css_class}">{_fmt(v, decimals)}{suffix}</div>',
                unsafe_allow_html=True,
            )

    # ── 3. Radar Chart ────────────────────────────────────────────────────────
    st.markdown('<div class="section-header">Radar: Pressure Profile</div>', unsafe_allow_html=True)

    radar_axes = [
        ("SR Improvement",        "sr_improvement"),
        ("Runs Above Expected",   "runs_above_expected_pressure"),
        ("Dismissal Avoidance",   "pressure_dismissal_rate"),   # will invert
        ("Boundary %",            "pressure_boundary_pct"),
        ("Dot Avoidance",         "pressure_dot_pct"),          # will invert
    ]
    # Collect raw values per axis
    axis_labels = [a[0] for a in radar_axes]
    all_axis_vals: dict[str, list] = {a[1]: [] for a in radar_axes}
    for name in selected:
        for _, col in radar_axes:
            all_axis_vals[col].append(_safe_float(_lookup(df_ref, name, col)))

    # Invert lower-is-better axes BEFORE normalization
    invert_cols = {"pressure_dismissal_rate", "pressure_dot_pct"}
    for col in invert_cols:
        raw = all_axis_vals[col]
        finite = [v for v in raw if not np.isnan(v)]
        if finite:
            mx = max(finite) + 1e-9
            all_axis_vals[col] = [mx - v if not np.isnan(v) else 0.0 for v in raw]

    # Normalize each axis 0-100
    norm_vals: dict[str, list] = {}
    for _, col in radar_axes:
        norm_vals[col] = _normalize_series(all_axis_vals[col])

    fig_radar = go.Figure()
    for i, name in enumerate(selected):
        r_vals = [norm_vals[col][i] for _, col in radar_axes]
        r_vals_closed = r_vals + [r_vals[0]]
        theta_closed = axis_labels + [axis_labels[0]]
        fig_radar.add_trace(go.Scatterpolar(
            r=r_vals_closed,
            theta=theta_closed,
            fill="toself",
            name=name,
            line=dict(color=_PLAYER_COLORS[i % len(_PLAYER_COLORS)], width=2),
            fillcolor=_PLAYER_COLORS[i % len(_PLAYER_COLORS)],
            opacity=0.25,
        ))

    fig_radar.update_layout(
        **_CHART_LAYOUT,
        polar=dict(
            bgcolor="#1E2538",
            radialaxis=dict(
                visible=True, range=[0, 100],
                tickfont=dict(color="#A0ADB8", size=10),
                gridcolor="#1E2F4A",
            ),
            angularaxis=dict(
                tickfont=dict(color="#FFFFFF", size=11),
                gridcolor="#1E2F4A",
            ),
        ),
        showlegend=True,
        legend=dict(
            bgcolor="#141929", bordercolor="#1E2F4A",
            font=dict(color="#FFFFFF"),
        ),
        title=dict(text="Pressure Component Profile (0-100 scale)", font=dict(color="#FFFFFF")),
        height=460,
    )
    st.plotly_chart(fig_radar, use_container_width=True)

    # ── 4. Bar Comparison ─────────────────────────────────────────────────────
    st.markdown('<div class="section-header">Key Metric Comparison</div>', unsafe_allow_html=True)

    bar_metrics = [
        ("Pressure SR",   "pressure_strike_rate",   False),
        ("Baseline SR",   "baseline_sr",             False),
        ("Boundary %",    "pressure_boundary_pct",   False),
        ("Dot % (lower=better)", "pressure_dot_pct", True),
    ]

    fig_bar = go.Figure()
    for i, name in enumerate(selected):
        x_vals = []
        y_vals = []
        for label, col, invert in bar_metrics:
            v = _safe_float(_lookup(df_ref, name, col))
            if not np.isnan(v):
                x_vals.append(100 - v if invert else v)
            else:
                x_vals.append(0)
            y_vals.append(label)

        fig_bar.add_trace(go.Bar(
            y=y_vals,
            x=x_vals,
            name=name,
            orientation="h",
            marker_color=_PLAYER_COLORS[i % len(_PLAYER_COLORS)],
            opacity=0.85,
        ))

    fig_bar.update_layout(
        **_CHART_LAYOUT,
        barmode="group",
        xaxis=dict(title="Value", gridcolor="#1E2F4A"),
        yaxis=dict(tickfont=dict(size=12)),
        legend=dict(bgcolor="#141929", bordercolor="#1E2F4A"),
        title=dict(text="Grouped Key Metrics", font=dict(color="#FFFFFF")),
        height=320,
    )
    st.plotly_chart(fig_bar, use_container_width=True)

    # ── 5. Consistency Analysis ───────────────────────────────────────────────
    st.markdown('<div class="section-header">Format Consistency</div>', unsafe_allow_html=True)

    has_both = [
        n for n in selected
        if (not np.isnan(_safe_float(_lookup(df_ref, n, "ipl_pressure_sr"))))
        and (not np.isnan(_safe_float(_lookup(df_ref, n, "t20i_pressure_sr"))))
    ]

    if has_both:
        ipl_avg = df_ref["ipl_pressure_sr"].dropna().mean() if "ipl_pressure_sr" in df_ref.columns else None
        t20i_avg = df_ref["t20i_pressure_sr"].dropna().mean() if "t20i_pressure_sr" in df_ref.columns else None

        fig_consist = go.Figure()
        for i, name in enumerate(has_both):
            ipl_sr = _safe_float(_lookup(df_ref, name, "ipl_pressure_sr"))
            t20i_sr = _safe_float(_lookup(df_ref, name, "t20i_pressure_sr"))
            fig_consist.add_trace(go.Bar(
                x=["IPL Pressure SR", "T20I Pressure SR"],
                y=[ipl_sr, t20i_sr],
                name=name,
                marker_color=_PLAYER_COLORS[has_both.index(name) % len(_PLAYER_COLORS)],
            ))

        # League average reference lines
        if ipl_avg and not np.isnan(ipl_avg):
            fig_consist.add_hline(y=ipl_avg, line_dash="dot", line_color="#FF4B6E",
                                   annotation_text=f"IPL Avg: {ipl_avg:.1f}", annotation_font_color="#FF4B6E")
        if t20i_avg and not np.isnan(t20i_avg):
            fig_consist.add_hline(y=t20i_avg, line_dash="dot", line_color="#4A9EFF",
                                   annotation_text=f"T20I Avg: {t20i_avg:.1f}", annotation_font_color="#4A9EFF")

        fig_consist.update_layout(
            **_CHART_LAYOUT,
            barmode="group",
            yaxis=dict(title="Pressure Strike Rate", gridcolor="#1E2F4A"),
            legend=dict(bgcolor="#141929", bordercolor="#1E2F4A"),
            title=dict(text="IPL vs T20I Pressure SR (players with both formats)", font=dict(color="#FFFFFF")),
            height=320,
        )
        st.plotly_chart(fig_consist, use_container_width=True)

        for name in has_both:
            ipl_sr = _safe_float(_lookup(df_ref, name, "ipl_pressure_sr"))
            t20i_sr = _safe_float(_lookup(df_ref, name, "t20i_pressure_sr"))
            diff = abs(ipl_sr - t20i_sr)
            consistency = _safe_float(_lookup(df_ref, name, "ipl_t20i_consistency"))
            note = f"**{name}**: IPL SR {ipl_sr:.1f} vs T20I SR {t20i_sr:.1f} — gap {diff:.1f}"
            if not np.isnan(consistency):
                note += f", consistency score {consistency:.2f}/1.0"
            st.caption(note)
    else:
        st.info("No selected players have pressure data in both IPL and T20I formats.")

    # ── 6. Sample Size Context ────────────────────────────────────────────────
    st.markdown('<div class="section-header">Sample Size Context</div>', unsafe_allow_html=True)

    try:
        from batting_analysis import ELIGIBILITY_THRESHOLDS as BAT_THRESH
        min_pb = BAT_THRESH.get("min_pressure_balls", 30)
    except Exception:
        min_pb = 30

    pop_avg_pb = df_ref["pressure_balls"].dropna().mean() if "pressure_balls" in df_ref.columns else 50

    fig_sample = go.Figure()
    for i, name in enumerate(selected):
        pb_val = _safe_float(_lookup(df_ref, name, "pressure_balls"), 0)
        fig_sample.add_trace(go.Bar(
            x=[name],
            y=[pb_val],
            name=name,
            marker_color=_PLAYER_COLORS[i % len(_PLAYER_COLORS)],
            showlegend=False,
        ))
        fig_sample.add_annotation(
            x=name, y=pb_val + 2,
            text=f"{int(pb_val)}",
            showarrow=False,
            font=dict(color=_PLAYER_COLORS[i % len(_PLAYER_COLORS)]),
        )

    fig_sample.add_hline(y=min_pb, line_dash="dash", line_color="#FF4B6E",
                          annotation_text=f"Eligibility threshold ({min_pb})",
                          annotation_font_color="#FF4B6E")
    fig_sample.add_hline(y=pop_avg_pb, line_dash="dot", line_color="#4A9EFF",
                          annotation_text=f"Population avg ({pop_avg_pb:.0f})",
                          annotation_font_color="#4A9EFF")

    fig_sample.update_layout(
        **_CHART_LAYOUT,
        yaxis=dict(title="Pressure Balls", gridcolor="#1E2F4A"),
        title=dict(text="Pressure Balls per Player", font=dict(color="#FFFFFF")),
        height=300,
    )
    st.plotly_chart(fig_sample, use_container_width=True)

    # ── 7. Summary Table + Download ───────────────────────────────────────────
    st.markdown('<div class="section-header">Statistical Summary</div>', unsafe_allow_html=True)

    summary_cols = [
        "player_name", "main_team_name", "rank", "adjusted_score",
        "pressure_strike_rate", "baseline_sr", "sr_improvement",
        "pressure_balls", "pressure_matches",
        "pressure_dismissal_rate", "pressure_boundary_pct", "pressure_dot_pct",
        "ipl_pressure_sr", "t20i_pressure_sr",
        "runs_above_expected_pressure", "format_coverage",
    ]
    rows = []
    for name in selected:
        player_rows = df_ref[df_ref["player_name"] == name]
        if not player_rows.empty:
            row_data = player_rows.iloc[0]
            row = {c: row_data.get(c, np.nan) for c in summary_cols if c in df_ref.columns}
            row["player_name"] = name
            rows.append(row)

    summary_df = pd.DataFrame(rows)
    if not summary_df.empty:
        display_cols = [c for c in summary_cols if c in summary_df.columns]
        st.dataframe(
            summary_df[display_cols].set_index("player_name").T,
            use_container_width=True,
        )

        csv_bytes = summary_df[display_cols].to_csv(index=False).encode("utf-8")
        st.download_button(
            label="⬇️ Download Comparison Table (CSV)",
            data=csv_bytes,
            file_name="player_comparison.csv",
            mime="text/csv",
        )


# ─────────────────────────────────────────────────────────────────────────────
# BOWLER COMPARISON
# ─────────────────────────────────────────────────────────────────────────────
else:
    df_ref = bowler_full

    # ── 1. Player Cards ───────────────────────────────────────────────────────
    st.markdown('<div class="section-header">Player Overview</div>', unsafe_allow_html=True)
    cols = st.columns(len(selected))
    for i, name in enumerate(selected):
        team = _lookup(df_ref, name, "main_team_name") or _lookup(df_ref, name, "main_team_id")
        fmt_cov = _lookup(df_ref, name, "format_coverage") or "—"
        rank_val = _lookup(df_ref, name, "rank")
        rank_str = f"Rank #{int(rank_val)}" if not (isinstance(rank_val, float) and np.isnan(rank_val)) else "—"
        color = _PLAYER_COLORS[i % len(_PLAYER_COLORS)]
        with cols[i]:
            st.markdown(
                f"""
                <div class="player-card" style="border-top-color:{color}">
                    <div class="player-name">{name}</div>
                    <div class="player-team">{team if pd.notna(team) else '—'}</div>
                    <div><span class="format-badge">{fmt_cov}</span></div>
                    <div><span class="rank-badge">{rank_str}</span></div>
                </div>
                """,
                unsafe_allow_html=True,
            )

    # ── 2. KPI Comparison ─────────────────────────────────────────────────────
    st.markdown('<div class="section-header">Head-to-Head Metrics</div>', unsafe_allow_html=True)

    bowler_kpis = [
        ("Pressure Rating",       "adjusted_score",             "higher", 1),
        ("Pressure Economy",      "pressure_economy",           "lower",  2),
        ("Baseline Economy",      "baseline_economy",           "lower",  2),
        ("Economy Improvement",   "economy_improvement",        "higher", 2),
        ("Pressure Wickets",      "pressure_wickets",           "higher", 0),
        ("Wickets Above Expected","wickets_above_expected",     "higher", 2),
        ("Dot Ball %",            "pressure_dot_pct",           "higher", 1),
        ("Boundary Conceded %",   "pressure_boundary_pct",      "lower",  1),
        ("Pressure Balls",        "pressure_balls",             "higher", 0),
        ("Pressure Matches",      "pressure_matches",           "higher", 0),
        ("IPL Pressure Economy",  "ipl_pressure_economy",       "lower",  2),
        ("T20I Pressure Economy", "t20i_pressure_economy",      "lower",  2),
    ]

    if fmt_filter == "IPL":
        bowler_kpis[1] = ("IPL Pressure Economy", "ipl_pressure_economy", "lower", 2)
    elif fmt_filter == "T20I":
        bowler_kpis[1] = ("T20I Pressure Economy", "t20i_pressure_economy", "lower", 2)

    kpi_cols = st.columns([2] + [1] * len(selected))
    kpi_cols[0].markdown("**Metric**")
    for i, name in enumerate(selected):
        kpi_cols[i + 1].markdown(f"**{name.split()[-1]}**")

    for label, col_name, direction, decimals in bowler_kpis:
        vals = [_safe_float(_lookup(df_ref, name, col_name)) for name in selected]
        finite_vals = [v for v in vals if not np.isnan(v)]
        if finite_vals:
            best_val = max(finite_vals) if direction == "higher" else min(finite_vals)
        else:
            best_val = None

        row_cols = st.columns([2] + [1] * len(selected))
        row_cols[0].markdown(f'<div class="kpi-label">{label}</div>', unsafe_allow_html=True)
        for i, v in enumerate(vals):
            is_winner = (best_val is not None) and (not np.isnan(v)) and (v == best_val)
            css_class = "kpi-winner" if is_winner else "kpi-value"
            suffix = " ✓" if is_winner else ""
            row_cols[i + 1].markdown(
                f'<div class="{css_class}">{_fmt(v, decimals)}{suffix}</div>',
                unsafe_allow_html=True,
            )

    # ── 3. Radar Chart ────────────────────────────────────────────────────────
    st.markdown('<div class="section-header">Radar: Pressure Profile</div>', unsafe_allow_html=True)

    bowl_radar_axes = [
        ("Economy Improvement",   "economy_improvement"),
        ("Runs Prevented",        "runs_prevented_pressure"),
        ("Wicket Taking",         "pressure_wickets"),
        ("Dot Generation %",      "pressure_dot_pct"),
        ("Boundary Prevention",   "pressure_boundary_pct"),   # will invert
    ]
    axis_labels = [a[0] for a in bowl_radar_axes]
    all_axis_vals = {a[1]: [] for a in bowl_radar_axes}
    for name in selected:
        for _, col in bowl_radar_axes:
            all_axis_vals[col].append(_safe_float(_lookup(df_ref, name, col)))

    bowl_invert_cols = {"pressure_boundary_pct"}
    for col in bowl_invert_cols:
        raw = all_axis_vals[col]
        finite = [v for v in raw if not np.isnan(v)]
        if finite:
            mx = max(finite) + 1e-9
            all_axis_vals[col] = [mx - v if not np.isnan(v) else 0.0 for v in raw]

    norm_vals = {}
    for _, col in bowl_radar_axes:
        norm_vals[col] = _normalize_series(all_axis_vals[col])

    fig_radar = go.Figure()
    for i, name in enumerate(selected):
        r_vals = [norm_vals[col][i] for _, col in bowl_radar_axes]
        r_vals_closed = r_vals + [r_vals[0]]
        theta_closed = axis_labels + [axis_labels[0]]
        fig_radar.add_trace(go.Scatterpolar(
            r=r_vals_closed,
            theta=theta_closed,
            fill="toself",
            name=name,
            line=dict(color=_PLAYER_COLORS[i % len(_PLAYER_COLORS)], width=2),
            fillcolor=_PLAYER_COLORS[i % len(_PLAYER_COLORS)],
            opacity=0.25,
        ))

    fig_radar.update_layout(
        **_CHART_LAYOUT,
        polar=dict(
            bgcolor="#1E2538",
            radialaxis=dict(visible=True, range=[0, 100], tickfont=dict(color="#A0ADB8"), gridcolor="#1E2F4A"),
            angularaxis=dict(tickfont=dict(color="#FFFFFF"), gridcolor="#1E2F4A"),
        ),
        showlegend=True,
        legend=dict(bgcolor="#141929", bordercolor="#1E2F4A"),
        title=dict(text="Bowling Pressure Component Profile (0-100 scale)", font=dict(color="#FFFFFF")),
        height=460,
    )
    st.plotly_chart(fig_radar, use_container_width=True)

    # ── 4. Bar Comparison ─────────────────────────────────────────────────────
    st.markdown('<div class="section-header">Key Metric Comparison</div>', unsafe_allow_html=True)

    bowl_bar_metrics = [
        ("Pressure Economy",    "pressure_economy",   True),   # lower=better, invert for display
        ("Baseline Economy",    "baseline_economy",   True),
        ("Dot Ball %",          "pressure_dot_pct",   False),
        ("Bdry Conceded % (inv)","pressure_boundary_pct", True),
    ]

    fig_bar = go.Figure()
    for i, name in enumerate(selected):
        x_vals = []
        y_vals = []
        for label, col, invert in bowl_bar_metrics:
            v = _safe_float(_lookup(df_ref, name, col))
            x_vals.append((12 - v) if (invert and not np.isnan(v)) else (v if not np.isnan(v) else 0))
            y_vals.append(label)

        fig_bar.add_trace(go.Bar(
            y=y_vals, x=x_vals, name=name,
            orientation="h",
            marker_color=_PLAYER_COLORS[i % len(_PLAYER_COLORS)],
            opacity=0.85,
        ))

    fig_bar.update_layout(
        **_CHART_LAYOUT,
        barmode="group",
        xaxis=dict(title="Value (inverted where lower=better)", gridcolor="#1E2F4A"),
        legend=dict(bgcolor="#141929", bordercolor="#1E2F4A"),
        title=dict(text="Grouped Key Metrics (inverted for lower-is-better)", font=dict(color="#FFFFFF")),
        height=320,
    )
    st.plotly_chart(fig_bar, use_container_width=True)

    # ── 5. Consistency Analysis ───────────────────────────────────────────────
    st.markdown('<div class="section-header">Format Consistency</div>', unsafe_allow_html=True)

    has_both = [
        n for n in selected
        if (not np.isnan(_safe_float(_lookup(df_ref, n, "ipl_pressure_economy"))))
        and (not np.isnan(_safe_float(_lookup(df_ref, n, "t20i_pressure_economy"))))
    ]

    if has_both:
        ipl_avg = df_ref["ipl_pressure_economy"].dropna().mean() if "ipl_pressure_economy" in df_ref.columns else None
        t20i_avg = df_ref["t20i_pressure_economy"].dropna().mean() if "t20i_pressure_economy" in df_ref.columns else None

        fig_consist = go.Figure()
        for i, name in enumerate(has_both):
            ipl_e = _safe_float(_lookup(df_ref, name, "ipl_pressure_economy"))
            t20i_e = _safe_float(_lookup(df_ref, name, "t20i_pressure_economy"))
            fig_consist.add_trace(go.Bar(
                x=["IPL Pressure Economy", "T20I Pressure Economy"],
                y=[ipl_e, t20i_e],
                name=name,
                marker_color=_PLAYER_COLORS[has_both.index(name) % len(_PLAYER_COLORS)],
            ))

        if ipl_avg and not np.isnan(ipl_avg):
            fig_consist.add_hline(y=ipl_avg, line_dash="dot", line_color="#FF4B6E",
                                   annotation_text=f"IPL Avg: {ipl_avg:.2f}", annotation_font_color="#FF4B6E")
        if t20i_avg and not np.isnan(t20i_avg):
            fig_consist.add_hline(y=t20i_avg, line_dash="dot", line_color="#4A9EFF",
                                   annotation_text=f"T20I Avg: {t20i_avg:.2f}", annotation_font_color="#4A9EFF")

        fig_consist.update_layout(
            **_CHART_LAYOUT,
            barmode="group",
            yaxis=dict(title="Pressure Economy", gridcolor="#1E2F4A"),
            legend=dict(bgcolor="#141929", bordercolor="#1E2F4A"),
            title=dict(text="IPL vs T20I Pressure Economy (lower is better)", font=dict(color="#FFFFFF")),
            height=320,
        )
        st.plotly_chart(fig_consist, use_container_width=True)

        for name in has_both:
            ipl_e = _safe_float(_lookup(df_ref, name, "ipl_pressure_economy"))
            t20i_e = _safe_float(_lookup(df_ref, name, "t20i_pressure_economy"))
            st.caption(f"**{name}**: IPL economy {ipl_e:.2f} vs T20I economy {t20i_e:.2f} — gap {abs(ipl_e - t20i_e):.2f}")
    else:
        st.info("No selected players have pressure economy data in both IPL and T20I formats.")

    # ── 6. Sample Size Context ────────────────────────────────────────────────
    st.markdown('<div class="section-header">Sample Size Context</div>', unsafe_allow_html=True)

    try:
        from bowling_analysis import ELIGIBILITY_THRESHOLDS as BOWL_THRESH
        min_pb = BOWL_THRESH.get("min_pressure_balls", 24)
    except Exception:
        min_pb = 24

    pop_avg_pb = df_ref["pressure_balls"].dropna().mean() if "pressure_balls" in df_ref.columns else 40

    fig_sample = go.Figure()
    for i, name in enumerate(selected):
        pb_val = _safe_float(_lookup(df_ref, name, "pressure_balls"), 0)
        fig_sample.add_trace(go.Bar(
            x=[name], y=[pb_val], name=name,
            marker_color=_PLAYER_COLORS[i % len(_PLAYER_COLORS)],
            showlegend=False,
        ))
        fig_sample.add_annotation(
            x=name, y=pb_val + 1, text=f"{int(pb_val)}",
            showarrow=False, font=dict(color=_PLAYER_COLORS[i % len(_PLAYER_COLORS)]),
        )

    fig_sample.add_hline(y=min_pb, line_dash="dash", line_color="#FF4B6E",
                          annotation_text=f"Eligibility threshold ({min_pb})", annotation_font_color="#FF4B6E")
    fig_sample.add_hline(y=pop_avg_pb, line_dash="dot", line_color="#4A9EFF",
                          annotation_text=f"Population avg ({pop_avg_pb:.0f})", annotation_font_color="#4A9EFF")

    fig_sample.update_layout(
        **_CHART_LAYOUT,
        yaxis=dict(title="Pressure Balls", gridcolor="#1E2F4A"),
        title=dict(text="Pressure Balls per Bowler", font=dict(color="#FFFFFF")),
        height=300,
    )
    st.plotly_chart(fig_sample, use_container_width=True)

    # ── 7. Summary Table + Download ───────────────────────────────────────────
    st.markdown('<div class="section-header">Statistical Summary</div>', unsafe_allow_html=True)

    summary_cols = [
        "player_name", "main_team_name", "rank", "adjusted_score",
        "pressure_economy", "baseline_economy", "economy_improvement",
        "pressure_wickets", "wickets_above_expected",
        "pressure_balls", "pressure_matches",
        "pressure_dot_pct", "pressure_boundary_pct",
        "ipl_pressure_economy", "t20i_pressure_economy", "format_coverage",
    ]
    rows = []
    for name in selected:
        player_rows = df_ref[df_ref["player_name"] == name]
        if not player_rows.empty:
            row_data = player_rows.iloc[0]
            row = {c: row_data.get(c, np.nan) for c in summary_cols if c in df_ref.columns}
            row["player_name"] = name
            rows.append(row)

    summary_df = pd.DataFrame(rows)
    if not summary_df.empty:
        display_cols = [c for c in summary_cols if c in summary_df.columns]
        st.dataframe(
            summary_df[display_cols].set_index("player_name").T,
            use_container_width=True,
        )
        csv_bytes = summary_df[display_cols].to_csv(index=False).encode("utf-8")
        st.download_button(
            label="⬇️ Download Comparison Table (CSV)",
            data=csv_bytes,
            file_name="bowler_comparison.csv",
            mime="text/csv",
        )
