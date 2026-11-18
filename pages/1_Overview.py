"""
pages/1_Overview.py – SUPRR Selector | Executive Overview
Full executive summary: KPIs, data distributions, top performers, methodology.
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Overview | SUPRR Selector",
    page_icon="📊",
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
    }
    .metric-value { font-size: 2.2rem; font-weight: 700; color: #00D4A1; }
    .metric-label { font-size: 0.85rem; color: #A0ADB8; margin-top: 4px; }
    .rank-badge {
        background: linear-gradient(135deg, #00D4A1, #4A9EFF);
        border-radius: 50%;
        width: 32px; height: 32px;
        display: flex; align-items: center; justify-content: center;
        font-weight: 700;
    }
    .section-header {
        font-size: 1.3rem;
        font-weight: 700;
        color: #FFFFFF;
        margin-top: 32px;
        margin-bottom: 16px;
        padding-bottom: 8px;
        border-bottom: 1px solid #1E2F4A;
    }
    .method-card {
        background: #141929;
        border: 1px solid #1E2F4A;
        border-top: 3px solid #00D4A1;
        border-radius: 12px;
        padding: 20px;
        height: 100%;
    }
    .method-step { font-size: 1.5rem; font-weight: 800; color: #00D4A1; }
    .method-title { font-size: 1rem; font-weight: 700; color: #FFFFFF; margin-top: 6px; }
    .method-desc { font-size: 0.82rem; color: #A0ADB8; margin-top: 8px; line-height: 1.6; }
    </style>
    """,
    unsafe_allow_html=True,
)

# ── Chart theme helper ────────────────────────────────────────────────────────
_CHART_LAYOUT = dict(
    paper_bgcolor="#1E2538",
    plot_bgcolor="#1E2538",
    font=dict(color="#FFFFFF", size=12),
    margin=dict(l=40, r=20, t=40, b=40),
)

_GREEN  = "#00D4A1"
_BLUE   = "#4A9EFF"
_RED    = "#FF4B6E"
_SUBTEXT = "#A0ADB8"

PRESSURE_COLORS = {
    "low":      "#4A9EFF",
    "moderate": "#00D4A1",
    "high":     "#FFB347",
    "extreme":  "#FF4B6E",
}

# ── Pipeline loading ──────────────────────────────────────────────────────────
@st.cache_resource(show_spinner="Loading analytics pipeline...")
def load_pipeline():
    import warnings
    warnings.filterwarnings("ignore")
    from ranking_engine import run_full_pipeline
    return run_full_pipeline()


pipeline        = load_pipeline()
batter_rankings = pipeline["batter_rankings"]
bowler_rankings = pipeline["bowler_rankings"]
full_df         = pipeline["full_df"]
data_summary    = pipeline["data_summary"]
batter_full     = pipeline["batter_full"]
bowler_full     = pipeline["bowler_full"]

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown(
        "<div style='text-align:center; padding:12px 0 16px 0;'>"
        "<span style='font-size:1.4rem; font-weight:900; color:#00D4A1;'>🏏 SUPRR</span>"
        "<div style='font-size:0.72rem; color:#A0ADB8; margin-top:2px;'>Pressure Analytics</div>"
        "</div>",
        unsafe_allow_html=True,
    )
    st.markdown("---")
    st.markdown(
        "<div style='font-size:0.8rem; color:#A0ADB8;'>Use the sidebar navigation to "
        "explore rankings, player profiles, and pressure analysis.</div>",
        unsafe_allow_html=True,
    )

# ═══════════════════════════════════════════════════════════════════════════════
# PAGE HEADER
# ═══════════════════════════════════════════════════════════════════════════════
st.markdown(
    "<div style='padding: 32px 0 8px 0;'>"
    "<h1 style='font-size:2rem; font-weight:800; color:#FFFFFF; margin-bottom:4px;'>"
    "📊 Executive Overview"
    "</h1>"
    "<div style='color:#A0ADB8; font-size:1rem;'>"
    "Complete data coverage, pressure landscape, and top performers across "
    "IPL &amp; T20I cricket (2022–2026)."
    "</div>"
    "</div>",
    unsafe_allow_html=True,
)
st.markdown("<hr style='border-color:#1E2F4A; margin: 16px 0 24px 0;'>", unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 1 — KPI ROW
# ═══════════════════════════════════════════════════════════════════════════════
st.markdown("<div class='section-header'>📈 Key Metrics</div>", unsafe_allow_html=True)

total_matches    = data_summary.get("total_matches", 0)
total_deliveries = data_summary.get("total_deliveries", 0)
batting_players  = data_summary.get("total_players_batting", 0)
bowling_players  = data_summary.get("total_players_bowling", 0)
total_players    = batting_players + bowling_players

# Pressure deliveries — from summary or recompute
high_pressure_bat = data_summary.get("high_pressure_bat_count", 0)
if high_pressure_bat == 0 and not full_df.empty and "is_high_pressure_bat" in full_df.columns:
    high_pressure_bat = int(full_df["is_high_pressure_bat"].sum())

eligible_batters = (
    len(batter_full[batter_full["is_eligible"]])
    if not batter_full.empty and "is_eligible" in batter_full.columns
    else len(batter_full)
)
eligible_bowlers = (
    len(bowler_full[bowler_full["is_eligible"]])
    if not bowler_full.empty and "is_eligible" in bowler_full.columns
    else len(bowler_full)
)

kpi_cols = st.columns(6)
kpis = [
    ("Total Matches",       f"{total_matches:,}"),
    ("Total Players",       f"{total_players:,}"),
    ("Total Deliveries",    f"{total_deliveries:,}"),
    ("Pressure Deliveries", f"{high_pressure_bat:,}"),
    ("Eligible Batters",    f"{eligible_batters:,}"),
    ("Eligible Bowlers",    f"{eligible_bowlers:,}"),
]
for col, (label, value) in zip(kpi_cols, kpis):
    with col:
        st.markdown(
            f"<div class='metric-card'>"
            f"<div class='metric-value'>{value}</div>"
            f"<div class='metric-label'>{label}</div>"
            f"</div>",
            unsafe_allow_html=True,
        )

st.markdown("<br>", unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 2 — DATA DISTRIBUTION
# ═══════════════════════════════════════════════════════════════════════════════
st.markdown("<div class='section-header'>🌍 Data Distribution</div>", unsafe_allow_html=True)

col_donut, col_bar = st.columns(2, gap="large")

with col_donut:
    # IPL vs T20I match split
    ipl_matches  = data_summary.get("ipl_matches", 0)
    t20i_matches = data_summary.get("t20i_matches", 0)

    if ipl_matches + t20i_matches > 0:
        fig_donut = go.Figure(
            go.Pie(
                labels=["IPL", "T20I"],
                values=[ipl_matches, t20i_matches],
                hole=0.55,
                marker=dict(colors=[_BLUE, _GREEN], line=dict(color="#0A0F1E", width=2)),
                textinfo="label+percent",
                textfont=dict(color="#FFFFFF", size=13),
                hovertemplate="<b>%{label}</b><br>Matches: %{value:,}<br>%{percent}<extra></extra>",
            )
        )
        fig_donut.update_layout(
            title=dict(text="IPL vs T20I Match Distribution", font=dict(color="#FFFFFF", size=14)),
            showlegend=True,
            legend=dict(font=dict(color="#FFFFFF")),
            annotations=[dict(
                text=f"<b>{ipl_matches + t20i_matches:,}</b><br><span style='font-size:10px'>Matches</span>",
                x=0.5, y=0.5, font_size=16, showarrow=False, font=dict(color="#FFFFFF"),
            )],
            **_CHART_LAYOUT,
        )
        st.plotly_chart(fig_donut, use_container_width=True)
    else:
        st.info("Match distribution data not available.")

with col_bar:
    # Deliveries by season — derive from full_df if possible
    if not full_df.empty and "season" in full_df.columns:
        season_counts = (
            full_df.groupby("season", observed=True)
            .size()
            .reset_index(name="deliveries")
            .sort_values("season")
        )
        season_counts["season"] = season_counts["season"].astype(str)
    elif not full_df.empty and "match_date" in full_df.columns:
        full_df["_year"] = pd.to_datetime(full_df["match_date"], errors="coerce").dt.year
        season_counts = (
            full_df.groupby("_year")
            .size()
            .reset_index(name="deliveries")
            .rename(columns={"_year": "season"})
            .dropna(subset=["season"])
            .sort_values("season")
        )
        season_counts["season"] = season_counts["season"].astype(int).astype(str)
    else:
        season_counts = pd.DataFrame({"season": [], "deliveries": []})

    if not season_counts.empty:
        fig_bar = go.Figure(
            go.Bar(
                x=season_counts["season"],
                y=season_counts["deliveries"],
                marker=dict(
                    color=season_counts["deliveries"],
                    colorscale=[[0, _BLUE], [1, _GREEN]],
                    showscale=False,
                    line=dict(color="#0A0F1E", width=0.5),
                ),
                hovertemplate="<b>Season %{x}</b><br>Deliveries: %{y:,}<extra></extra>",
                text=season_counts["deliveries"].apply(lambda x: f"{x:,}"),
                textposition="outside",
                textfont=dict(color="#FFFFFF", size=10),
            )
        )
        fig_bar.update_layout(
            title=dict(text="Deliveries by Season", font=dict(color="#FFFFFF", size=14)),
            xaxis=dict(title="Season", color="#A0ADB8", gridcolor="#1E2F4A"),
            yaxis=dict(title="Deliveries", color="#A0ADB8", gridcolor="#1E2F4A"),
            **_CHART_LAYOUT,
        )
        st.plotly_chart(fig_bar, use_container_width=True)
    else:
        # Fallback: use IPL/T20I delivery split
        ipl_del  = data_summary.get("ipl_deliveries", 0)
        t20i_del = data_summary.get("t20i_deliveries", 0)
        fig_bar = go.Figure(
            go.Bar(
                x=["IPL", "T20I"],
                y=[ipl_del, t20i_del],
                marker=dict(color=[_BLUE, _GREEN]),
                hovertemplate="<b>%{x}</b><br>Deliveries: %{y:,}<extra></extra>",
            )
        )
        fig_bar.update_layout(
            title=dict(text="Deliveries by Format", font=dict(color="#FFFFFF", size=14)),
            xaxis=dict(color="#A0ADB8"),
            yaxis=dict(title="Deliveries", color="#A0ADB8", gridcolor="#1E2F4A"),
            **_CHART_LAYOUT,
        )
        st.plotly_chart(fig_bar, use_container_width=True)

# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 3 — PRESSURE DISTRIBUTION
# ═══════════════════════════════════════════════════════════════════════════════
st.markdown("<div class='section-header'>🔥 Pressure Distribution</div>", unsafe_allow_html=True)

col_cat, col_phase = st.columns(2, gap="large")

with col_cat:
    # BPI category distribution
    if not full_df.empty and "pressure_category_bat" in full_df.columns:
        cat_counts = (
            full_df["pressure_category_bat"]
            .value_counts()
            .reindex(["low", "moderate", "high", "extreme"])
            .fillna(0)
            .reset_index()
        )
        cat_counts.columns = ["Category", "Count"]
        cat_counts["Category"] = cat_counts["Category"].str.capitalize()

        fig_cat = go.Figure(
            go.Pie(
                labels=cat_counts["Category"],
                values=cat_counts["Count"],
                hole=0.5,
                marker=dict(
                    colors=[PRESSURE_COLORS["low"], PRESSURE_COLORS["moderate"],
                            PRESSURE_COLORS["high"], PRESSURE_COLORS["extreme"]],
                    line=dict(color="#0A0F1E", width=2),
                ),
                textinfo="label+percent",
                textfont=dict(color="#FFFFFF", size=12),
                hovertemplate="<b>%{label}</b><br>Count: %{value:,}<br>%{percent}<extra></extra>",
            )
        )
        fig_cat.update_layout(
            title=dict(text="BPI Category Distribution", font=dict(color="#FFFFFF", size=14)),
            showlegend=True,
            legend=dict(font=dict(color="#FFFFFF")),
            **_CHART_LAYOUT,
        )
        st.plotly_chart(fig_cat, use_container_width=True)
    else:
        st.info("BPI category data not available.")

with col_phase:
    # Phase pressure means
    if not full_df.empty and "bpi" in full_df.columns and "innings_phase" in full_df.columns:
        phase_df = (
            full_df.groupby(
                full_df["innings_phase"].str.lower().str.strip(), observed=True
            )[["bpi", "bopi"]]
            .mean()
            .reindex(["powerplay", "middle", "death"])
            .dropna(how="all")
            .reset_index()
        )
        phase_df.columns = ["Phase", "Mean BPI", "Mean BoPI"]
        phase_df["Phase"] = phase_df["Phase"].str.capitalize()

        fig_phase = go.Figure()
        fig_phase.add_trace(go.Bar(
            name="Mean BPI",
            x=phase_df["Phase"],
            y=phase_df["Mean BPI"],
            marker_color=_GREEN,
            hovertemplate="<b>%{x}</b><br>Mean BPI: %{y:.1f}<extra></extra>",
        ))
        fig_phase.add_trace(go.Bar(
            name="Mean BoPI",
            x=phase_df["Phase"],
            y=phase_df["Mean BoPI"],
            marker_color=_BLUE,
            hovertemplate="<b>%{x}</b><br>Mean BoPI: %{y:.1f}<extra></extra>",
        ))
        fig_phase.update_layout(
            title=dict(text="Mean Pressure by Phase", font=dict(color="#FFFFFF", size=14)),
            barmode="group",
            xaxis=dict(color="#A0ADB8"),
            yaxis=dict(title="Mean Pressure Index", color="#A0ADB8", gridcolor="#1E2F4A"),
            legend=dict(font=dict(color="#FFFFFF")),
            **_CHART_LAYOUT,
        )
        st.plotly_chart(fig_phase, use_container_width=True)
    else:
        st.info("Phase pressure data not available.")

# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 4 — TOP PERFORMERS
# ═══════════════════════════════════════════════════════════════════════════════
st.markdown("<div class='section-header'>🏆 Top Performers Under Pressure</div>", unsafe_allow_html=True)

col_bat_tbl, col_bowl_tbl = st.columns(2, gap="large")

with col_bat_tbl:
    st.markdown(
        "<div style='font-size:1rem; font-weight:700; color:#00D4A1; margin-bottom:10px;'>"
        "🏏 Top 3 Batters"
        "</div>",
        unsafe_allow_html=True,
    )
    if not batter_rankings.empty:
        top3_bat = batter_rankings.head(3).copy()
        col_map = {
            "rank":                 "Rank",
            "player_name":          "Player",
            "main_team_name":       "Team",
            "pressure_strike_rate": "Pressure SR",
            "sr_improvement":       "SR Improvement",
            "pressure_balls":       "Pressure Balls",
            "format_coverage":      "Coverage",
        }
        available = {k: v for k, v in col_map.items() if k in top3_bat.columns}
        display = top3_bat[list(available.keys())].rename(columns=available).reset_index(drop=True)
        for c in ["Pressure SR", "SR Improvement"]:
            if c in display.columns:
                display[c] = display[c].apply(lambda x: f"{x:.1f}" if pd.notna(x) else "—")
        st.dataframe(display, use_container_width=True, hide_index=True)
    else:
        st.info("No batter ranking data available.")

with col_bowl_tbl:
    st.markdown(
        "<div style='font-size:1rem; font-weight:700; color:#4A9EFF; margin-bottom:10px;'>"
        "🎳 Top 3 Bowlers"
        "</div>",
        unsafe_allow_html=True,
    )
    if not bowler_rankings.empty:
        top3_bowl = bowler_rankings.head(3).copy()
        col_map = {
            "rank":                "Rank",
            "player_name":         "Player",
            "main_team_name":      "Team",
            "pressure_economy":    "Pressure Eco",
            "economy_improvement": "Eco Improvement",
            "pressure_balls":      "Pressure Balls",
            "format_coverage":     "Coverage",
        }
        available = {k: v for k, v in col_map.items() if k in top3_bowl.columns}
        display = top3_bowl[list(available.keys())].rename(columns=available).reset_index(drop=True)
        for c in ["Pressure Eco", "Eco Improvement"]:
            if c in display.columns:
                display[c] = display[c].apply(lambda x: f"{x:.2f}" if pd.notna(x) else "—")
        st.dataframe(display, use_container_width=True, hide_index=True)
    else:
        st.info("No bowler ranking data available.")

st.markdown("<br>", unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 5 — METHODOLOGY
# ═══════════════════════════════════════════════════════════════════════════════
st.markdown("<div class='section-header'>🔬 Methodology Overview</div>", unsafe_allow_html=True)

methods = [
    (
        "01",
        "Data & Match State",
        "867 IPL & T20I matches (2022–2026). Every delivery is enriched with "
        "live match state: required run rate, runs & wickets to date, balls "
        "remaining, innings phase (powerplay / middle / death), and 2nd-innings "
        "target. Data sourced from official ball-by-ball scorecards.",
    ),
    (
        "02",
        "Pressure Detection",
        "Two indices are calculated per delivery: BPI (Batter Pressure Index) "
        "and BoPI (Bowler Pressure Index), each scored 0–100. Five components — "
        "RRR difficulty, wickets lost, balls remaining, innings phase, and match "
        "leverage — are weighted and combined. Deliveries ≥ 50 are high-pressure.",
    ),
    (
        "03",
        "Expected Performance",
        "Gradient-boosted models trained on match context predict: expected runs "
        "per ball (batting), expected runs conceded per ball (bowling), and wicket "
        "probability. Actual outcomes are compared against these context-adjusted "
        "baselines, isolating genuine pressure performance from easy situations.",
    ),
    (
        "04",
        "Scoring & Rankings",
        "Five z-score components (SR improvement, runs above expected, dismissal "
        "avoidance, boundary contribution, dot avoidance for batters; and economy "
        "improvement, runs prevented, wicket-taking, dot generation, boundary "
        "prevention for bowlers) are weighted into a SUPRR composite score. "
        "Bayesian shrinkage adjusts for sample size. Bootstrap CIs bound every rank.",
    ),
]

m_cols = st.columns(4, gap="medium")
for col, (num, title, desc) in zip(m_cols, methods):
    with col:
        st.markdown(
            f"<div class='method-card'>"
            f"<div class='method-step'>{num}</div>"
            f"<div class='method-title'>{title}</div>"
            f"<div class='method-desc'>{desc}</div>"
            f"</div>",
            unsafe_allow_html=True,
        )

st.markdown("<br><br>", unsafe_allow_html=True)
st.markdown(
    "<div style='text-align:center; color:#A0ADB8; font-size:0.78rem; "
    "padding: 16px 0; border-top: 1px solid #1E2F4A;'>"
    "SUPRR Selector · Executive Overview · IPL &amp; T20I 2022–2026"
    "</div>",
    unsafe_allow_html=True,
)
