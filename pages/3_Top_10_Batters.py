"""
pages/3_Top_10_Batters.py – SUPRR Selector | Top 10 Pressure Batters
Flagship page: interactive rankings, radar charts, player profiles,
and statistical justification for the best pressure batters.
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Top 10 Batters | SUPRR Selector",
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
        height: 100%;
    }
    .metric-value { font-size: 2.2rem; font-weight: 700; color: #00D4A1; }
    .metric-label { font-size: 0.85rem; color: #A0ADB8; margin-top: 4px; }
    .metric-sub   { font-size: 0.78rem; color: #4A9EFF; margin-top: 2px; }

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

    .profile-header {
        background: linear-gradient(135deg, #141929, #1a2a4a);
        border: 1px solid #1E2F4A;
        border-top: 3px solid #00D4A1;
        border-radius: 12px;
        padding: 20px 24px;
        margin-bottom: 16px;
    }
    .profile-name  { font-size: 1.6rem; font-weight: 800; color: #FFFFFF; }
    .profile-team  { font-size: 0.9rem; color: #A0ADB8; margin-top: 2px; }
    .rank-pill {
        display: inline-block;
        background: linear-gradient(135deg, #00D4A1, #4A9EFF);
        color: #0A0F1E;
        font-weight: 800;
        font-size: 0.9rem;
        padding: 4px 14px;
        border-radius: 20px;
        margin-top: 8px;
    }
    .stat-card {
        background: #141929;
        border: 1px solid #1E2F4A;
        border-radius: 10px;
        padding: 14px;
        text-align: center;
    }
    .stat-value { font-size: 1.5rem; font-weight: 700; color: #00D4A1; }
    .stat-label { font-size: 0.78rem; color: #A0ADB8; margin-top: 2px; }

    .justify-box {
        background: #141929;
        border: 1px solid #1E2F4A;
        border-left: 4px solid #4A9EFF;
        border-radius: 8px;
        padding: 16px 20px;
        font-size: 0.9rem;
        color: #A0ADB8;
        line-height: 1.8;
        margin-top: 12px;
    }
    .ci-box {
        background: #0d1527;
        border: 1px solid #1E2F4A;
        border-radius: 8px;
        padding: 10px 16px;
        font-size: 0.84rem;
        color: #A0ADB8;
        margin-top: 8px;
    }
    .format-mini {
        background: #141929;
        border: 1px solid #1E2F4A;
        border-radius: 8px;
        padding: 12px;
        text-align: center;
    }
    .format-title { font-size: 0.8rem; font-weight: 700; color: #4A9EFF; text-transform: uppercase; letter-spacing: 0.08em; }
    .format-val   { font-size: 1.3rem; font-weight: 700; color: #00D4A1; }
    .format-sub   { font-size: 0.75rem; color: #A0ADB8; }

    div[data-testid="stDataFrame"] { border-radius: 10px; overflow: hidden; }
    </style>
    """,
    unsafe_allow_html=True,
)

# ── Chart theme ───────────────────────────────────────────────────────────────
_CL = dict(
    paper_bgcolor="#1E2538",
    plot_bgcolor="#1E2538",
    font=dict(color="#FFFFFF", size=12),
    margin=dict(l=40, r=20, t=50, b=40),
)
_GREEN = "#00D4A1"
_BLUE  = "#4A9EFF"
_RED   = "#FF4B6E"

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
batter_full     = pipeline["batter_full"]
bowler_full     = pipeline["bowler_full"]
full_df         = pipeline["full_df"]

# ── Helper: safe value ────────────────────────────────────────────────────────
def _v(val, fmt=None, prefix="", suffix="", fallback="N/A"):
    """Return formatted string or fallback when val is NaN/None."""
    if val is None:
        return fallback
    try:
        f = float(val)
        if np.isnan(f):
            return fallback
        if fmt == "1f":
            s = f"{f:.1f}"
        elif fmt == "2f":
            s = f"{f:.2f}"
        elif fmt == "+1f":
            s = f"{f:+.1f}"
        elif fmt == "pct":
            s = f"{f:.1f}%"
        elif fmt == "int":
            s = str(int(round(f)))
        else:
            s = str(val)
        return f"{prefix}{s}{suffix}"
    except (TypeError, ValueError):
        return str(val) if val else fallback

# ── Sidebar filters ───────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🏏 Batter Filters")
    st.markdown("---")

    fmt_choice = st.selectbox(
        "Format",
        ["All Formats", "IPL Only", "T20I Only"],
        index=0,
    )

    min_balls = st.slider("Minimum Pressure Balls", 10, 150, 30, 5)
    bpi_thresh = st.slider("BPI Threshold (Pressure Cutoff)", 40, 80, 50, 5,
                           help="Minimum Ball Pressure Index score defining 'high pressure'.")
    top_n = st.selectbox("Show Top N", [5, 10, 15, 20], index=1)

    st.markdown("---")

    # Build download data from batter_full filtered
    @st.cache_data(show_spinner=False)
    def _download_df(_bf, _min_balls):
        df = _bf.copy()
        if "pressure_balls" in df.columns:
            df = df[df["pressure_balls"] >= _min_balls]
        return df.to_csv(index=False).encode("utf-8")

    dl_csv = _download_df(batter_full, min_balls)
    st.download_button(
        label="⬇ Download Rankings CSV",
        data=dl_csv,
        file_name="suprr_batter_rankings.csv",
        mime="text/csv",
        use_container_width=True,
    )

# ── Filter & sort data ────────────────────────────────────────────────────────
@st.cache_data(show_spinner=False)
def filter_batters(_df_full, fmt, min_b):
    df = _df_full.copy()
    if "pressure_balls" in df.columns:
        df = df[df["pressure_balls"] >= min_b]

    if fmt == "IPL Only":
        sort_col = "ipl_pressure_sr" if "ipl_pressure_sr" in df.columns else "adjusted_score"
        # keep only players with IPL data
        if "ipl_pressure_balls" in df.columns:
            df = df[pd.notna(df["ipl_pressure_balls"]) & (df["ipl_pressure_balls"] > 0)]
        df = df.sort_values(sort_col, ascending=False)
    elif fmt == "T20I Only":
        sort_col = "t20i_pressure_sr" if "t20i_pressure_sr" in df.columns else "adjusted_score"
        if "t20i_pressure_balls" in df.columns:
            df = df[pd.notna(df["t20i_pressure_balls"]) & (df["t20i_pressure_balls"] > 0)]
        df = df.sort_values(sort_col, ascending=False)
    else:
        df = df.sort_values("adjusted_score", ascending=False)

    df = df.reset_index(drop=True)
    df["rank"] = range(1, len(df) + 1)
    return df

filtered_full = filter_batters(batter_full, fmt_choice, min_balls)
top_df        = filtered_full.head(top_n).copy()

# ── Page header ───────────────────────────────────────────────────────────────
st.markdown(
    """
    <div class="page-header">
      <div class="page-title">🏏 Top Pressure Batters — SUPRR Rankings</div>
      <div class="page-desc">
        Rankings are built from the SUPRR (Situational Under-Pressure Performance Rating) model,
        measuring how batters perform when the Ball Pressure Index (BPI) crosses the high-pressure
        threshold. Each player's adjusted score weights strike-rate improvement, runs above
        expected output, dismissal avoidance, and boundary efficiency — all shrunk toward the
        population mean to guard against small-sample noise.
      </div>
    </div>
    """,
    unsafe_allow_html=True,
)

# ── Warnings ──────────────────────────────────────────────────────────────────
if len(filtered_full) == 0:
    st.warning("No batters meet the current filter criteria. Try reducing the minimum pressure balls.")
    st.stop()
elif len(filtered_full) < top_n:
    st.info(
        f"Only **{len(filtered_full)}** batters meet the current filters "
        f"(minimum {min_balls} pressure balls, {fmt_choice}). Showing all eligible players."
    )

# ── KPI row ───────────────────────────────────────────────────────────────────
st.markdown('<div class="section-header">📊 Summary</div>', unsafe_allow_html=True)

k1, k2, k3, k4 = st.columns(4)

eligible_count = len(filtered_full)
top10_slice    = filtered_full.head(10)

avg_psr  = top10_slice["pressure_strike_rate"].mean() if "pressure_strike_rate" in top10_slice else np.nan
avg_imp  = top10_slice["sr_improvement"].mean()       if "sr_improvement" in top10_slice else np.nan
avg_pb   = top10_slice["pressure_balls"].mean()       if "pressure_balls" in top10_slice else np.nan

for col, label, val, fmt, sub in [
    (k1, "Eligible Batters Analysed", eligible_count, "int", fmt_choice),
    (k2, "Avg Pressure Strike Rate",  avg_psr,        "1f",  "Top 10 players"),
    (k3, "Avg SR Improvement",        avg_imp,        "+1f", "vs. baseline"),
    (k4, "Avg Pressure Balls",        avg_pb,         "1f",  "per player (top 10)"),
]:
    with col:
        st.markdown(
            f"""
            <div class="metric-card">
              <div class="metric-value">{_v(val, fmt)}</div>
              <div class="metric-label">{label}</div>
              <div class="metric-sub">{sub}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

st.markdown("<br>", unsafe_allow_html=True)

# ── Rankings Table ────────────────────────────────────────────────────────────
st.markdown('<div class="section-header">🏆 Top N Rankings</div>', unsafe_allow_html=True)

def _build_display_table(df):
    cols_map = {
        "rank":                 "Rank",
        "player_name":          "Player",
        "main_team_name":       "Team",
        "adjusted_score":       "Pressure Rating",
        "pressure_strike_rate": "Pressure SR",
        "baseline_sr":          "Baseline SR",
        "sr_improvement":       "SR Improvement",
        "pressure_runs":        "Pressure Runs",
        "pressure_balls":       "Pressure Balls",
        "pressure_matches":     "Matches",
        "format_coverage":      "Format Coverage",
    }
    available = {k: v for k, v in cols_map.items() if k in df.columns}
    out = df[list(available.keys())].copy()
    out.columns = list(available.values())

    if "SR Improvement" in out.columns:
        out["SR Improvement"] = out["SR Improvement"].apply(
            lambda x: f"{x:+.1f}" if pd.notna(x) else "N/A"
        )
    for c in ["Pressure SR", "Baseline SR"]:
        if c in out.columns:
            out[c] = out[c].apply(lambda x: f"{x:.1f}" if pd.notna(x) else "N/A")
    if "Pressure Rating" in out.columns:
        out["Pressure Rating"] = out["Pressure Rating"].apply(
            lambda x: f"{x:.2f}" if pd.notna(x) else "N/A"
        )
    if "Pressure Runs" in out.columns:
        out["Pressure Runs"] = out["Pressure Runs"].apply(
            lambda x: str(int(round(x))) if pd.notna(x) else "N/A"
        )
    if "Pressure Balls" in out.columns:
        out["Pressure Balls"] = out["Pressure Balls"].apply(
            lambda x: str(int(round(x))) if pd.notna(x) else "N/A"
        )
    return out

display_tbl = _build_display_table(top_df)

col_config = {
    "Rank": st.column_config.NumberColumn("🏅 Rank", format="%d", width="small"),
    "Player": st.column_config.TextColumn("👤 Player", width="medium"),
    "Team": st.column_config.TextColumn("🏟 Team", width="small"),
    "Pressure Rating": st.column_config.TextColumn("⚡ Rating", width="small"),
    "Pressure SR": st.column_config.TextColumn("🔥 Press SR", width="small"),
    "Baseline SR": st.column_config.TextColumn("📏 Base SR", width="small"),
    "SR Improvement": st.column_config.TextColumn("📈 Improvement", width="small"),
    "Pressure Runs": st.column_config.TextColumn("🏃 Runs", width="small"),
    "Pressure Balls": st.column_config.TextColumn("🎯 Balls", width="small"),
    "Matches": st.column_config.TextColumn("🗓 Matches", width="small"),
    "Format Coverage": st.column_config.TextColumn("📡 Format", width="small"),
}

st.dataframe(
    display_tbl,
    column_config=col_config,
    use_container_width=True,
    hide_index=True,
)

# ── Charts row 1 ──────────────────────────────────────────────────────────────
st.markdown('<div class="section-header">📈 Performance Charts</div>', unsafe_allow_html=True)

c1, c2 = st.columns(2)

with c1:
    # Horizontal bar — adjusted_score, colored by pressure SR
    fig_bar = go.Figure()
    if not top_df.empty and "adjusted_score" in top_df.columns:
        plot_df = top_df.dropna(subset=["adjusted_score"]).sort_values("adjusted_score")
        psr_vals = plot_df.get("pressure_strike_rate", pd.Series(dtype=float)).fillna(0)

        fig_bar.add_trace(go.Bar(
            x=plot_df["adjusted_score"],
            y=plot_df["player_name"],
            orientation="h",
            marker=dict(
                color=psr_vals,
                colorscale=[[0, "#1a4a3a"], [0.5, "#00D4A1"], [1, "#00ffc3"]],
                showscale=True,
                colorbar=dict(title="Press SR", tickfont=dict(color="white"), title_font=dict(color="white")),
            ),
            text=[f"{v:.2f}" for v in plot_df["adjusted_score"]],
            textposition="outside",
            textfont=dict(color="white", size=11),
            hovertemplate="<b>%{y}</b><br>Rating: %{x:.3f}<extra></extra>",
        ))
        fig_bar.update_layout(
            **_CL,
            title=dict(text="Pressure Rating (Adjusted Score)", font=dict(size=14, color="white")),
            xaxis=dict(showgrid=True, gridcolor="#1E2F4A", title="Adjusted Score"),
            yaxis=dict(showgrid=False),
            height=400,
        )
    st.plotly_chart(fig_bar, use_container_width=True)

with c2:
    # Scatter — Pressure SR vs Baseline SR
    fig_sc = go.Figure()
    if not top_df.empty and "pressure_strike_rate" in top_df.columns and "baseline_sr" in top_df.columns:
        sc_df = top_df.dropna(subset=["pressure_strike_rate", "baseline_sr"])
        all_vals = pd.concat([sc_df["pressure_strike_rate"], sc_df["baseline_sr"]])
        mn, mx = all_vals.min() * 0.9, all_vals.max() * 1.1

        # 45-degree reference line
        fig_sc.add_trace(go.Scatter(
            x=[mn, mx], y=[mn, mx],
            mode="lines",
            line=dict(color="#A0ADB8", dash="dash", width=1.5),
            name="Baseline (y = x)",
            showlegend=True,
        ))
        bubble_sizes = sc_df.get("pressure_balls", pd.Series([20]*len(sc_df))).fillna(20)
        bubble_sizes = (bubble_sizes - bubble_sizes.min()) / (bubble_sizes.max() - bubble_sizes.min() + 1) * 30 + 8

        fig_sc.add_trace(go.Scatter(
            x=sc_df["baseline_sr"],
            y=sc_df["pressure_strike_rate"],
            mode="markers+text",
            text=sc_df["player_name"].apply(lambda x: x.split()[-1] if isinstance(x, str) else x),
            textposition="top center",
            textfont=dict(size=10, color="white"),
            marker=dict(
                size=bubble_sizes,
                color=_GREEN,
                opacity=0.8,
                line=dict(color="#0A0F1E", width=1),
            ),
            name="Players",
            hovertemplate="<b>%{text}</b><br>Baseline SR: %{x:.1f}<br>Pressure SR: %{y:.1f}<extra></extra>",
        ))
        fig_sc.update_layout(
            **_CL,
            title=dict(text="Pressure SR vs Baseline SR (above line = improved)", font=dict(size=14, color="white")),
            xaxis=dict(title="Baseline Strike Rate", range=[mn, mx], showgrid=True, gridcolor="#1E2F4A"),
            yaxis=dict(title="Pressure Strike Rate", range=[mn, mx], showgrid=True, gridcolor="#1E2F4A"),
            height=400,
        )
    st.plotly_chart(fig_sc, use_container_width=True)

# ── Charts row 2 ──────────────────────────────────────────────────────────────
c3, c4 = st.columns(2)

with c3:
    # Grouped bar — Actual vs Expected Pressure Runs
    fig_runs = go.Figure()
    if not top_df.empty and "pressure_runs" in top_df.columns:
        r_df = top_df.dropna(subset=["pressure_runs"])
        names = r_df["player_name"].tolist()

        fig_runs.add_trace(go.Bar(
            x=names,
            y=r_df["pressure_runs"],
            name="Actual Pressure Runs",
            marker_color=_GREEN,
            hovertemplate="<b>%{x}</b><br>Actual: %{y:.0f}<extra></extra>",
        ))
        if "expected_pressure_runs" in r_df.columns:
            exp_vals = r_df["expected_pressure_runs"]
            if exp_vals.notna().any():
                fig_runs.add_trace(go.Bar(
                    x=names,
                    y=exp_vals,
                    name="Expected Pressure Runs",
                    marker_color=_BLUE,
                    opacity=0.8,
                    hovertemplate="<b>%{x}</b><br>Expected: %{y:.0f}<extra></extra>",
                ))
        fig_runs.update_layout(
            **_CL,
            title=dict(text="Actual vs Expected Pressure Runs", font=dict(size=14, color="white")),
            barmode="group",
            xaxis=dict(tickangle=-35, showgrid=False),
            yaxis=dict(title="Runs", showgrid=True, gridcolor="#1E2F4A"),
            legend=dict(bgcolor="#141929", bordercolor="#1E2F4A"),
            height=400,
        )
    st.plotly_chart(fig_runs, use_container_width=True)

with c4:
    # IPL vs T20I pressure SR comparison
    fig_fmt = go.Figure()
    if not top_df.empty:
        ipl_col  = "ipl_pressure_sr"
        t20_col  = "t20i_pressure_sr"
        has_ipl  = ipl_col in top_df.columns
        has_t20  = t20_col in top_df.columns

        if has_ipl or has_t20:
            fmt_df = top_df[["player_name"]].copy()
            if has_ipl:
                fmt_df["IPL"] = top_df[ipl_col]
            if has_t20:
                fmt_df["T20I"] = top_df[t20_col]

            # only players with at least one format
            fmt_df = fmt_df.dropna(how="all", subset=[c for c in ["IPL", "T20I"] if c in fmt_df.columns])

            if has_ipl and "IPL" in fmt_df:
                fig_fmt.add_trace(go.Bar(
                    x=fmt_df["player_name"],
                    y=fmt_df["IPL"],
                    name="IPL",
                    marker_color=_BLUE,
                    hovertemplate="<b>%{x}</b><br>IPL SR: %{y:.1f}<extra></extra>",
                ))
            if has_t20 and "T20I" in fmt_df:
                fig_fmt.add_trace(go.Bar(
                    x=fmt_df["player_name"],
                    y=fmt_df["T20I"],
                    name="T20I",
                    marker_color=_GREEN,
                    hovertemplate="<b>%{x}</b><br>T20I SR: %{y:.1f}<extra></extra>",
                ))
        fig_fmt.update_layout(
            **_CL,
            title=dict(text="IPL vs T20I Pressure Strike Rate", font=dict(size=14, color="white")),
            barmode="group",
            xaxis=dict(tickangle=-35, showgrid=False),
            yaxis=dict(title="Pressure SR", showgrid=True, gridcolor="#1E2F4A"),
            legend=dict(bgcolor="#141929", bordercolor="#1E2F4A"),
            height=400,
        )
    st.plotly_chart(fig_fmt, use_container_width=True)

# ── Radar Chart ───────────────────────────────────────────────────────────────
st.markdown('<div class="section-header">🎯 Player Radar — Component Breakdown</div>', unsafe_allow_html=True)

radar_players = top_df["player_name"].dropna().tolist() if "player_name" in top_df.columns else []
sel_radar = st.selectbox("Select player for detailed radar chart", radar_players, key="radar_sel")

RADAR_AXES   = ["Runs Above Exp", "SR Improvement", "Dismissal Avoidance", "Boundary Contribution", "Dot Avoidance"]
RADAR_COLS   = ["component_runs_above_exp", "component_sr", "component_dismissal", "component_boundary", "component_dots"]

def _z_to_score(z):
    """Normalize z-score to 0-100."""
    if z is None or (isinstance(z, float) and np.isnan(z)):
        return 50.0
    return max(0.0, min(100.0, float(z) * 15 + 50))

if sel_radar and not top_df.empty:
    r_row = top_df[top_df["player_name"] == sel_radar]
    if not r_row.empty:
        r_row = r_row.iloc[0]
        radar_vals = [_z_to_score(r_row.get(c, np.nan)) for c in RADAR_COLS]
        radar_vals_closed = radar_vals + [radar_vals[0]]
        axes_closed       = RADAR_AXES + [RADAR_AXES[0]]

        fig_radar = go.Figure()
        fig_radar.add_trace(go.Scatterpolar(
            r=radar_vals_closed,
            theta=axes_closed,
            fill="toself",
            fillcolor=f"rgba(0,212,161,0.2)",
            line=dict(color=_GREEN, width=2),
            name=sel_radar,
        ))
        fig_radar.add_trace(go.Scatterpolar(
            r=[50] * (len(RADAR_AXES) + 1),
            theta=axes_closed,
            mode="lines",
            line=dict(color="#A0ADB8", width=1, dash="dash"),
            name="Average (50)",
            showlegend=True,
        ))
        fig_radar.update_layout(
            paper_bgcolor="#1E2538",
            plot_bgcolor="#1E2538",
            font=dict(color="white", size=12),
            polar=dict(
                bgcolor="#1E2538",
                radialaxis=dict(
                    visible=True, range=[0, 100],
                    gridcolor="#1E2F4A", tickfont=dict(color="#A0ADB8", size=10),
                    tickvals=[0, 25, 50, 75, 100],
                ),
                angularaxis=dict(gridcolor="#1E2F4A", tickfont=dict(color="white", size=12)),
            ),
            title=dict(text=f"Component Radar: {sel_radar}", font=dict(size=16, color="white"), x=0.5),
            showlegend=True,
            legend=dict(bgcolor="#141929", bordercolor="#1E2F4A"),
            height=500,
            margin=dict(l=60, r=60, t=80, b=60),
        )
        st.plotly_chart(fig_radar, use_container_width=True)

# ── Player Profile ────────────────────────────────────────────────────────────
st.markdown('<div class="section-header">👤 Player Profile</div>', unsafe_allow_html=True)

profile_players = top_df["player_name"].dropna().tolist() if "player_name" in top_df.columns else []
sel_player = st.selectbox("Select player for full profile", profile_players, key="profile_sel")

if sel_player and not top_df.empty:
    p = top_df[top_df["player_name"] == sel_player]
    if not p.empty:
        p = p.iloc[0]

        # Also check batter_full for the extra columns
        pf = pd.Series(dtype=object)
        if not batter_full.empty and "player_name" in batter_full.columns:
            pf_rows = batter_full[batter_full["player_name"] == sel_player]
            if not pf_rows.empty:
                pf = pf_rows.iloc[0]

        def _g(key):
            """Get from full row first, then top_df row."""
            v = pf.get(key, None) if not pf.empty else None
            if v is None or (isinstance(v, float) and np.isnan(v)):
                v = p.get(key, None)
            return v

        rank_val    = _v(_g("rank"), "int")
        team_val    = str(_g("main_team_name") or "Unknown")
        format_val  = str(_g("format_coverage") or "Unknown")

        with st.expander(f"📋 Full Profile: {sel_player}", expanded=True):
            # Header card
            st.markdown(
                f"""
                <div class="profile-header">
                  <div class="profile-name">{sel_player}</div>
                  <div class="profile-team">🏟 {team_val} &nbsp;|&nbsp; 📡 {format_val}</div>
                  <div class="rank-pill">🏅 Rank #{rank_val}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

            # Stat row 1
            s1, s2, s3, s4 = st.columns(4)
            for col_w, lbl, key, fmt in [
                (s1, "Pressure SR",   "pressure_strike_rate", "1f"),
                (s2, "Baseline SR",   "baseline_sr",          "1f"),
                (s3, "SR Improvement","sr_improvement",       "+1f"),
                (s4, "Pressure Balls","pressure_balls",       "int"),
            ]:
                with col_w:
                    st.markdown(
                        f'<div class="stat-card"><div class="stat-value">{_v(_g(key), fmt)}</div>'
                        f'<div class="stat-label">{lbl}</div></div>',
                        unsafe_allow_html=True,
                    )

            st.markdown("<br>", unsafe_allow_html=True)

            # Stat row 2
            s5, s6, s7, s8 = st.columns(4)
            for col_w, lbl, key, fmt in [
                (s5, "Pressure Runs",  "pressure_runs",              "int"),
                (s6, "Expected Runs",  "expected_pressure_runs",     "1f"),
                (s7, "Runs Above Exp", "runs_above_expected_pressure","1f"),
                (s8, "Matches",        "pressure_matches",            "int"),
            ]:
                with col_w:
                    st.markdown(
                        f'<div class="stat-card"><div class="stat-value">{_v(_g(key), fmt)}</div>'
                        f'<div class="stat-label">{lbl}</div></div>',
                        unsafe_allow_html=True,
                    )

            st.markdown("<br>", unsafe_allow_html=True)

            # IPL vs T20I comparison
            st.markdown("**IPL vs T20I Breakdown**")
            ipl_col_w, t20_col_w = st.columns(2)

            with ipl_col_w:
                st.markdown(
                    f"""
                    <div class="format-mini">
                      <div class="format-title">🟦 IPL</div>
                      <div class="format-val">{_v(_g('ipl_pressure_sr'), '1f')}</div>
                      <div class="format-sub">Pressure SR</div>
                      <div style="margin-top:8px;font-size:0.82rem;color:#A0ADB8;">
                        Runs: {_v(_g('ipl_pressure_runs'), 'int')} &nbsp;|&nbsp;
                        Balls: {_v(_g('ipl_pressure_balls'), 'int')} &nbsp;|&nbsp;
                        Matches: {_v(_g('ipl_pressure_matches'), 'int')}
                      </div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

            with t20_col_w:
                st.markdown(
                    f"""
                    <div class="format-mini">
                      <div class="format-title">🟩 T20I</div>
                      <div class="format-val">{_v(_g('t20i_pressure_sr'), '1f')}</div>
                      <div class="format-sub">Pressure SR</div>
                      <div style="margin-top:8px;font-size:0.82rem;color:#A0ADB8;">
                        Runs: {_v(_g('t20i_pressure_runs'), 'int')} &nbsp;|&nbsp;
                        Balls: {_v(_g('t20i_pressure_balls'), 'int')} &nbsp;|&nbsp;
                        Matches: {_v(_g('t20i_pressure_matches'), 'int')}
                      </div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

            st.markdown("<br>", unsafe_allow_html=True)

            # Statistical justification
            def _justify_batter(row_p, row_pf):
                name   = str(row_p.get("player_name", sel_player))
                rank_n = _v(row_p.get("rank"),                  "int")
                bpi    = _v(row_pf.get("avg_bpi") if not row_pf.empty else None, "1f")
                psr    = _v(row_p.get("pressure_strike_rate"),  "1f")
                bsr    = _v(row_p.get("baseline_sr"),           "1f")
                imp    = _v(row_p.get("sr_improvement"),        "+1f")
                p_runs = _v(row_p.get("pressure_runs"),         "int")
                rae    = _v(row_p.get("runs_above_expected_pressure") if "runs_above_expected_pressure" in row_p.index else None, "+1f")
                diss   = _v(row_pf.get("pressure_dismissal_rate") if not row_pf.empty else None, "pct")
                bound  = _v(row_pf.get("pressure_boundary_pct") if not row_pf.empty else None, "pct")
                dots   = _v(row_pf.get("pressure_dot_pct") if not row_pf.empty else None, "pct")
                pballs = _v(row_p.get("pressure_balls"), "int")
                pmatch = _v(row_p.get("pressure_matches"), "int")

                parts = [f"**Why {name} ranks #{rank_n}:**\n\n"]
                bpi_txt = f" (avg BPI: {bpi})" if bpi != "N/A" else ""
                parts.append(
                    f"Under high-pressure situations{bpi_txt}, {name} scores at a strike rate "
                    f"of **{psr}** compared to their overall baseline of **{bsr}** — "
                    f"an improvement of **{imp}** SR points. "
                )
                if rae != "N/A":
                    parts.append(f"They score **{rae}** runs above what the model predicts under pressure. ")
                if diss != "N/A":
                    parts.append(f"Their dismissal rate under pressure is **{diss}**, ")
                if bound != "N/A":
                    parts.append(f"while scoring **{bound}** of deliveries to the boundary. ")
                if dots != "N/A":
                    parts.append(f"Dot ball percentage: **{dots}**. ")
                parts.append(
                    f"This rating is based on **{p_runs}** pressure runs across **{pballs}** "
                    f"deliveries in **{pmatch}** matches."
                )
                return "".join(parts)

            justify_text = _justify_batter(p, pf)
            st.markdown(
                f'<div class="justify-box">{justify_text}</div>',
                unsafe_allow_html=True,
            )

            # CI and sample size
            ci_lo = _v(_g("ci_lower"), "1f")
            ci_hi = _v(_g("ci_upper"), "1f")
            pballs_n = _v(_g("pressure_balls"), "int")
            pmatch_n = _v(_g("pressure_matches"), "int")

            st.markdown(
                f"""
                <div class="ci-box">
                  📐 <strong>Pressure SR confidence interval:</strong> {ci_lo} – {ci_hi}
                  &nbsp;&nbsp;|&nbsp;&nbsp;
                  🗂 <strong>Sample:</strong> {pballs_n} pressure deliveries across {pmatch_n} matches
                </div>
                """,
                unsafe_allow_html=True,
            )

# ── Eligibility & Sample Size Chart ──────────────────────────────────────────
st.markdown('<div class="section-header">📏 Sample Size & Eligibility</div>', unsafe_allow_html=True)

if not filtered_full.empty and "pressure_balls" in filtered_full.columns:
    top20_samp = filtered_full.head(20).copy()
    top20_samp = top20_samp.dropna(subset=["pressure_balls"])

    fig_samp = go.Figure()
    fig_samp.add_trace(go.Bar(
        x=top20_samp["player_name"],
        y=top20_samp["pressure_balls"],
        marker_color=_GREEN,
        name="Pressure Balls",
        hovertemplate="<b>%{x}</b><br>Pressure Balls: %{y}<extra></extra>",
    ))
    fig_samp.add_hline(
        y=min_balls,
        line_dash="dash",
        line_color=_RED,
        annotation_text=f"Min threshold: {min_balls}",
        annotation_font_color=_RED,
    )
    fig_samp.update_layout(
        **_CL,
        title=dict(text="Pressure Balls per Player (Top 20) — Eligibility View", font=dict(size=14, color="white")),
        xaxis=dict(tickangle=-35, showgrid=False),
        yaxis=dict(title="Pressure Balls", showgrid=True, gridcolor="#1E2F4A"),
        height=380,
    )
    st.plotly_chart(fig_samp, use_container_width=True)
else:
    st.info("Sample size chart unavailable — 'pressure_balls' column not found in data.")
