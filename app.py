"""
app.py – SUPRR Selector | Cricket Pressure Analytics
Main Streamlit entry point.
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

import streamlit as st

# ── Page config (must be first Streamlit call) ───────────────────────────────
st.set_page_config(
    page_title="SUPRR Selector | Cricket Pressure Analytics",
    page_icon="🏏",
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
        width: 32px;
        height: 32px;
        display: flex;
        align-items: center;
        justify-content: center;
        font-weight: 700;
    }
    .hero-title {
        font-size: 3.5rem;
        font-weight: 900;
        background: linear-gradient(135deg, #00D4A1, #4A9EFF);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        letter-spacing: 4px;
        margin-bottom: 0;
    }
    .hero-subtitle {
        font-size: 1.25rem;
        color: #A0ADB8;
        margin-top: 8px;
        margin-bottom: 4px;
    }
    .hero-tagline {
        font-size: 0.95rem;
        color: #4A9EFF;
        font-weight: 500;
        letter-spacing: 1px;
    }
    .hero-divider {
        border: none;
        border-top: 1px solid #1E2F4A;
        margin: 24px 0;
    }
    .step-card {
        background: #141929;
        border: 1px solid #1E2F4A;
        border-left: 4px solid #00D4A1;
        border-radius: 12px;
        padding: 18px 20px;
        height: 100%;
    }
    .step-number {
        font-size: 1.5rem;
        font-weight: 700;
        color: #00D4A1;
    }
    .step-title {
        font-size: 1rem;
        font-weight: 600;
        color: #FFFFFF;
        margin-top: 4px;
    }
    .step-desc {
        font-size: 0.82rem;
        color: #A0ADB8;
        margin-top: 6px;
        line-height: 1.5;
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
    .mini-table-header {
        background: #141929;
        border: 1px solid #1E2F4A;
        border-radius: 12px;
        padding: 16px 20px 8px 20px;
    }
    .mini-table-title {
        font-size: 1rem;
        font-weight: 700;
        color: #00D4A1;
        margin-bottom: 12px;
    }
    .nav-item {
        background: #141929;
        border: 1px solid #1E2F4A;
        border-radius: 8px;
        padding: 10px 14px;
        margin-bottom: 8px;
        cursor: pointer;
    }
    .nav-item:hover { border-color: #00D4A1; }
    .nav-item-title { font-size: 0.9rem; font-weight: 600; color: #FFFFFF; }
    .nav-item-desc { font-size: 0.78rem; color: #A0ADB8; margin-top: 2px; }
    </style>
    """,
    unsafe_allow_html=True,
)

# ── Pipeline loading ──────────────────────────────────────────────────────────
@st.cache_resource(show_spinner="Loading analytics pipeline...")
def load_pipeline():
    import warnings
    warnings.filterwarnings("ignore")
    from ranking_engine import run_full_pipeline
    return run_full_pipeline()


pipeline     = load_pipeline()
batter_rankings = pipeline["batter_rankings"]
bowler_rankings = pipeline["bowler_rankings"]
full_df         = pipeline["full_df"]
data_summary    = pipeline["data_summary"]
batter_full     = pipeline["batter_full"]
bowler_full     = pipeline["bowler_full"]

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown(
        "<div style='text-align:center; padding: 12px 0 20px 0;'>"
        "<span style='font-size:1.6rem; font-weight:900; color:#00D4A1; letter-spacing:2px;'>🏏 SUPRR</span>"
        "<div style='font-size:0.75rem; color:#A0ADB8; margin-top:4px;'>Pressure Analytics</div>"
        "</div>",
        unsafe_allow_html=True,
    )
    st.markdown("---")
    st.markdown("**Navigate**", help="Use the links below to explore each section.")

    nav_pages = [
        ("🏠", "Home",              "app",                    "Dashboard overview & KPIs"),
        ("📊", "Executive Overview","pages/1_Overview",        "Full data coverage & methodology"),
        ("🔥", "Pressure Analysis", "pages/2_Pressure_Analysis","BPI/BoPI engine & distributions"),
        ("🏏", "Batter Rankings",   "pages/3_Batter_Rankings", "Top batters under pressure"),
        ("🎳", "Bowler Rankings",   "pages/4_Bowler_Rankings", "Top bowlers under pressure"),
        ("👤", "Player Profile",    "pages/5_Player_Profile",  "Deep-dive individual profiles"),
        ("📈", "Trends",            "pages/6_Trends",          "Season & format trends"),
        ("📥", "Export",            "pages/7_Export",          "Download data & reports"),
    ]
    for icon, title, _, desc in nav_pages:
        st.markdown(
            f"<div class='nav-item'>"
            f"<div class='nav-item-title'>{icon} {title}</div>"
            f"<div class='nav-item-desc'>{desc}</div>"
            f"</div>",
            unsafe_allow_html=True,
        )

    st.markdown("---")
    st.markdown(
        "<div style='font-size:0.75rem; color:#A0ADB8; text-align:center;'>"
        "IPL &amp; T20I · 2022–2026<br>SUPRR v1.0"
        "</div>",
        unsafe_allow_html=True,
    )

# ── Hero Header ───────────────────────────────────────────────────────────────
st.markdown(
    "<div style='text-align:center; padding: 48px 0 32px 0;'>"
    "<div class='hero-title'>SUPRR SELECTOR</div>"
    "<div class='hero-subtitle'>Identifying the Most Reliable Players Under Pressure</div>"
    "<div class='hero-tagline'>IPL &amp; T20I &nbsp;|&nbsp; 2022–2026 &nbsp;|&nbsp; 867 Matches Analysed</div>"
    "</div>",
    unsafe_allow_html=True,
)
st.markdown("<hr class='hero-divider'>", unsafe_allow_html=True)

# ── KPI Cards ─────────────────────────────────────────────────────────────────
total_matches    = data_summary.get("total_matches", 867)
total_deliveries = data_summary.get("total_deliveries", 0)
ipl_matches      = data_summary.get("ipl_matches", 0)
t20i_matches     = data_summary.get("t20i_matches", 0)

batting_players  = data_summary.get("total_players_batting", 0)
bowling_players  = data_summary.get("total_players_bowling", 0)
total_players    = len(set()) if (batting_players == 0 and bowling_players == 0) else batting_players + bowling_players

high_pressure_bat  = data_summary.get("high_pressure_bat_count", data_summary.get("pressure_deliveries", 0))
# Compute from full_df if not in summary
if high_pressure_bat == 0 and not full_df.empty and "is_high_pressure_bat" in full_df.columns:
    high_pressure_bat = int(full_df["is_high_pressure_bat"].sum())

eligible_batters = len(batter_full[batter_full["is_eligible"]]) if (not batter_full.empty and "is_eligible" in batter_full.columns) else len(batter_full)
eligible_bowlers = len(bowler_full[bowler_full["is_eligible"]]) if (not bowler_full.empty and "is_eligible" in bowler_full.columns) else len(bowler_full)

st.markdown("<div class='section-header'>📊 Key Statistics</div>", unsafe_allow_html=True)

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

# ── Top 3 mini-tables ─────────────────────────────────────────────────────────
st.markdown("<div class='section-header'>🏆 Top Performers Under Pressure</div>", unsafe_allow_html=True)

col_bat, col_bowl = st.columns(2, gap="large")

with col_bat:
    st.markdown(
        "<div style='font-size:1rem; font-weight:700; color:#00D4A1; margin-bottom:10px;'>"
        "🏏 Top 3 Batters Under Pressure"
        "</div>",
        unsafe_allow_html=True,
    )
    if not batter_rankings.empty:
        top3_bat = batter_rankings.head(3).copy()
        display_cols_bat = {
            "rank":                  "Rank",
            "player_name":           "Player",
            "main_team_name":        "Team",
            "pressure_strike_rate":  "Pressure SR",
            "sr_improvement":        "SR Improvement",
            "pressure_balls":        "Pressure Balls",
            "format_coverage":       "Coverage",
        }
        # Keep only columns that exist
        available = {k: v for k, v in display_cols_bat.items() if k in top3_bat.columns}
        top3_bat_display = top3_bat[list(available.keys())].rename(columns=available)

        # Format numeric columns
        for c in ["Pressure SR", "SR Improvement"]:
            if c in top3_bat_display.columns:
                top3_bat_display[c] = top3_bat_display[c].apply(
                    lambda x: f"{x:.1f}" if isinstance(x, float) else x
                )

        st.dataframe(
            top3_bat_display,
            use_container_width=True,
            hide_index=True,
        )
    else:
        st.info("No batter ranking data available.")

with col_bowl:
    st.markdown(
        "<div style='font-size:1rem; font-weight:700; color:#4A9EFF; margin-bottom:10px;'>"
        "🎳 Top 3 Bowlers Under Pressure"
        "</div>",
        unsafe_allow_html=True,
    )
    if not bowler_rankings.empty:
        top3_bowl = bowler_rankings.head(3).copy()
        display_cols_bowl = {
            "rank":                "Rank",
            "player_name":         "Player",
            "main_team_name":      "Team",
            "pressure_economy":    "Pressure Eco",
            "economy_improvement": "Eco Improvement",
            "pressure_balls":      "Pressure Balls",
            "format_coverage":     "Coverage",
        }
        available = {k: v for k, v in display_cols_bowl.items() if k in top3_bowl.columns}
        top3_bowl_display = top3_bowl[list(available.keys())].rename(columns=available)

        for c in ["Pressure Eco", "Eco Improvement"]:
            if c in top3_bowl_display.columns:
                top3_bowl_display[c] = top3_bowl_display[c].apply(
                    lambda x: f"{x:.2f}" if isinstance(x, float) else x
                )

        st.dataframe(
            top3_bowl_display,
            use_container_width=True,
            hide_index=True,
        )
    else:
        st.info("No bowler ranking data available.")

st.markdown("<br>", unsafe_allow_html=True)

# ── How it works ─────────────────────────────────────────────────────────────
st.markdown("<div class='section-header'>⚙️ How It Works</div>", unsafe_allow_html=True)

steps = [
    (
        "01",
        "Data",
        "Ball-by-ball ingestion",
        "867 IPL & T20I matches (2022–2026). Every delivery enriched with "
        "match state: required run rate, wickets in hand, balls remaining, "
        "innings phase, and target context.",
    ),
    (
        "02",
        "Pressure Detection",
        "BPI & BoPI computation",
        "Batter Pressure Index (BPI) and Bowler Pressure Index (BoPI) score "
        "each delivery 0–100 using five situational components. Deliveries "
        "above 50 are classified as high-pressure.",
    ),
    (
        "03",
        "Expected Performance",
        "ML-based benchmarking",
        "Gradient-boosted models predict expected runs and wicket probability "
        "for each delivery given match context. Actual outcomes are compared "
        "against these baselines to measure true pressure value.",
    ),
    (
        "04",
        "Rankings",
        "SUPRR scoring & shrinkage",
        "Five component z-scores are weighted and combined into a composite "
        "SUPRR score. Bayesian shrinkage adjusts for sample size, and "
        "confidence intervals bound every ranking.",
    ),
]

step_cols = st.columns(4, gap="medium")
for col, (num, title, subtitle, desc) in zip(step_cols, steps):
    with col:
        st.markdown(
            f"<div class='step-card'>"
            f"<div class='step-number'>{num}</div>"
            f"<div class='step-title'>{title}</div>"
            f"<div style='font-size:0.78rem; color:#4A9EFF; margin-top:2px; font-weight:500;'>{subtitle}</div>"
            f"<div class='step-desc'>{desc}</div>"
            f"</div>",
            unsafe_allow_html=True,
        )

st.markdown("<br><br>", unsafe_allow_html=True)

# ── Footer ────────────────────────────────────────────────────────────────────
st.markdown(
    "<div style='text-align:center; color:#A0ADB8; font-size:0.78rem; "
    "padding: 20px 0; border-top: 1px solid #1E2F4A; margin-top: 16px;'>"
    "SUPRR Selector · IPL &amp; T20I Pressure Analytics · 2022–2026 · "
    "Data sourced from official scorecards"
    "</div>",
    unsafe_allow_html=True,
)
