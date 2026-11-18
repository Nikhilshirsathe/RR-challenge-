"""
player_baseline.py – Per-player baseline performance metrics for the SUPRR Selector.

Exports
-------
compute_batter_baselines(df) -> pd.DataFrame
    Overall and split (normal / IPL / T20I) batting metrics, indexed by striker_id.

compute_bowler_baselines(df) -> pd.DataFrame
    Overall and split (normal / IPL / T20I) bowling metrics, indexed by bowler_id.

Metric definitions
------------------
"Normal" deliveries are those where is_high_pressure_bat / is_high_pressure_bowl is
False, giving us a pressure-free baseline to compare against.

Bowler-credit wickets exclude: RunOut, Hit Wicket, Obstructing/Obstracting the field.
Only Caught, Bowled, LBW, Stumped (and any other mode not in the exclusion list) count
toward a bowler's wicket tally.
"""

from __future__ import annotations

import logging
from typing import Union

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Streamlit cache shim – degrades gracefully without Streamlit
# ---------------------------------------------------------------------------
try:
    import streamlit as st
    _cache = st.cache_data
    logger.debug("player_baseline: Streamlit detected – @st.cache_data available.")
except ImportError:
    def _cache(func=None, **_kwargs):   # type: ignore[misc]
        if func is not None:
            return func
        return lambda f: f
    logger.debug("player_baseline: Streamlit not available – using no-op cache decorator.")

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Dismissal types that do NOT credit the bowler
_NON_BOWLER_DISMISSALS = {
    "RunOut",
    "Run Out",
    "run out",
    "Hit Wicket",
    "hit wicket",
    "Obstracting the field",   # note typo from spec (intentionally kept)
    "Obstructing the field",
    "Obstructing The Field",
    "Handled the ball",
    "Retired hurt",
    "Retired out",
}

_MIN_NORMAL_BALLS = 10   # minimum balls to report normal_strike_rate

# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _most_common_name(series: pd.Series) -> str:
    """Return the most frequent non-null value in *series*, or empty string."""
    if series.empty:
        return ""
    counts = series.dropna().value_counts()
    return str(counts.index[0]) if len(counts) else ""


def _safe_sr(runs: Union[float, pd.Series],
             balls: Union[float, pd.Series],
             min_balls: int = 0) -> Union[float, pd.Series]:
    """Strike rate = runs / balls * 100, NaN when balls < min_balls or zero."""
    if isinstance(balls, pd.Series):
        result = np.where(
            balls >= max(min_balls, 1),
            runs / np.maximum(balls, 1) * 100.0,
            np.nan,
        )
        return pd.Series(result, index=balls.index)
    if balls < max(min_balls, 1):
        return float("nan")
    return runs / balls * 100.0


def _safe_economy(runs: float, balls: float) -> float:
    """Economy = runs / (balls / 6), NaN when balls == 0."""
    if balls == 0:
        return float("nan")
    return runs / (balls / 6.0)


def _is_bowler_credit(df: pd.DataFrame) -> pd.Series:
    """Return boolean Series: True when the wicket is credited to the bowler.

    If a 'dismissal_type' column is present, we exclude _NON_BOWLER_DISMISSALS.
    If the column is absent we fall back to crediting all is_wicket rows (the
    caller should log a warning in that case).
    """
    if "dismissal_type" not in df.columns:
        logger.warning(
            "_is_bowler_credit: 'dismissal_type' column not found — "
            "crediting all is_wicket rows to the bowler."
        )
        return df["is_wicket"].astype(bool)

    dismissal = df["dismissal_type"].fillna("").astype(str)
    is_non_bowler = dismissal.str.strip().isin(_NON_BOWLER_DISMISSALS)
    return df["is_wicket"].astype(bool) & ~is_non_bowler


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

@_cache
def compute_batter_baselines(df: pd.DataFrame) -> pd.DataFrame:
    """Compute per-batter baseline metrics from the enriched ball-by-ball frame.

    Parameters
    ----------
    df : pd.DataFrame
        Full enriched + pressure-annotated ball-by-ball dataframe (``ms``).

    Returns
    -------
    pd.DataFrame
        One row per striker_id.  Index = striker_id.
        Columns documented in module docstring.
    """
    if df is None or df.empty:
        logger.warning("compute_batter_baselines: received empty dataframe.")
        return pd.DataFrame()

    logger.info("compute_batter_baselines: processing %d rows", len(df))

    # ------------------------------------------------------------------ #
    # Validate required columns
    # ------------------------------------------------------------------ #
    required = {
        "striker_id", "striker_name", "match_id",
        "is_legal_ball", "runs_off_bat",
        "is_boundary_4", "is_boundary_6", "is_wicket",
        "is_high_pressure_bat",
    }
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"compute_batter_baselines: missing columns {missing}")

    # ------------------------------------------------------------------ #
    # Work on legal deliveries only for the per-ball stats
    # ------------------------------------------------------------------ #
    legal = df[df["is_legal_ball"].astype(bool)].copy()

    # ------------------------------------------------------------------ #
    # Overall stats  (all legal deliveries)
    # ------------------------------------------------------------------ #
    grp = legal.groupby("striker_id", sort=True)

    total_balls    = grp["is_legal_ball"].count().rename("total_balls_faced")
    total_runs     = grp["runs_off_bat"].sum().rename("total_runs")
    total_matches  = grp["match_id"].nunique().rename("total_matches")
    player_name    = grp["striker_name"].agg(_most_common_name).rename("player_name")

    # Boundaries
    total_boundaries = (
        grp["is_boundary_4"].sum() + grp["is_boundary_6"].sum()
    ).rename("_total_boundaries")

    # Dot balls: legal AND runs_off_bat == 0 AND NOT is_wicket
    dot_mask = (legal["runs_off_bat"] == 0) & (~legal["is_wicket"].astype(bool))
    legal_dots = legal.copy()
    legal_dots["_is_dot"] = dot_mask
    total_dots = legal_dots.groupby("striker_id")["_is_dot"].sum().rename("_total_dots")

    # Wickets (as striker)
    total_wickets = grp["is_wicket"].sum().rename("_total_wickets")

    # ------------------------------------------------------------------ #
    # Normal (non-pressure) stats
    # ------------------------------------------------------------------ #
    normal = legal[~legal["is_high_pressure_bat"].astype(bool)]
    ngrp   = normal.groupby("striker_id", sort=True)

    normal_balls   = ngrp["is_legal_ball"].count().rename("normal_balls_faced")
    normal_runs    = ngrp["runs_off_bat"].sum().rename("normal_runs")
    normal_bnd     = (ngrp["is_boundary_4"].sum() + ngrp["is_boundary_6"].sum()).rename("_normal_bnd")

    # ------------------------------------------------------------------ #
    # Format-split stats: IPL (match_format == 'T20') and T20I
    # ------------------------------------------------------------------ #
    def _format_batter_stats(fmt_label: str, prefix: str) -> pd.DataFrame:
        """Aggregate batter stats for one format label."""
        if "match_format" not in df.columns:
            logger.warning(
                "compute_batter_baselines: 'match_format' column missing — "
                "%s stats will be NaN.", fmt_label
            )
            return pd.DataFrame(columns=[f"{prefix}_balls", f"{prefix}_runs",
                                         f"{prefix}_sr", f"{prefix}_matches"])

        sub = legal[legal["match_format"].astype(str).str.upper() == fmt_label.upper()]
        if sub.empty:
            return pd.DataFrame(columns=[f"{prefix}_balls", f"{prefix}_runs",
                                         f"{prefix}_sr", f"{prefix}_matches"])

        sg = sub.groupby("striker_id", sort=True)
        balls   = sg["is_legal_ball"].count().rename(f"{prefix}_balls")
        runs    = sg["runs_off_bat"].sum().rename(f"{prefix}_runs")
        matches = sg["match_id"].nunique().rename(f"{prefix}_matches")
        return pd.concat([balls, runs, matches], axis=1)

    ipl_stats  = _format_batter_stats("T20",  "ipl")
    t20i_stats = _format_batter_stats("T20I", "t20i")

    # ------------------------------------------------------------------ #
    # Assemble base frame
    # ------------------------------------------------------------------ #
    base = pd.concat(
        [player_name, total_balls, total_runs, total_matches,
         total_boundaries, total_dots, total_wickets,
         normal_balls, normal_runs, normal_bnd],
        axis=1,
    )

    # Merge format splits (outer so we keep players even if they have no T20I
    # history, for instance)
    base = base.join(ipl_stats,  how="left")
    base = base.join(t20i_stats, how="left")

    # Fill NaN counts (players with no deliveries in that format)
    count_cols = [c for c in base.columns
                  if c.endswith(("_balls", "_runs", "_matches", "_matches"))]
    base[count_cols] = base[count_cols].fillna(0).astype(int)

    # ------------------------------------------------------------------ #
    # Derived rates
    # ------------------------------------------------------------------ #
    base["player_id"] = base.index   # duplicate index as column

    base["overall_strike_rate"] = (
        base["total_runs"] / np.maximum(base["total_balls_faced"], 1) * 100.0
    ).where(base["total_balls_faced"] > 0, other=np.nan)

    base["boundary_pct"] = (
        base["_total_boundaries"] / np.maximum(base["total_balls_faced"], 1) * 100.0
    ).where(base["total_balls_faced"] > 0, other=np.nan)

    base["dot_ball_pct"] = (
        base["_total_dots"] / np.maximum(base["total_balls_faced"], 1) * 100.0
    ).where(base["total_balls_faced"] > 0, other=np.nan)

    base["dismissal_rate"] = (
        base["_total_wickets"] / np.maximum(base["total_balls_faced"], 1) * 100.0
    ).where(base["total_balls_faced"] > 0, other=np.nan)

    # Normal SR: NaN if fewer than _MIN_NORMAL_BALLS
    base["normal_strike_rate"] = np.where(
        base["normal_balls_faced"] >= _MIN_NORMAL_BALLS,
        base["normal_runs"] / np.maximum(base["normal_balls_faced"], 1) * 100.0,
        np.nan,
    )

    base["normal_boundary_pct"] = (
        base["_normal_bnd"] / np.maximum(base["normal_balls_faced"], 1) * 100.0
    ).where(base["normal_balls_faced"] > 0, other=np.nan)

    # Format SRs
    if "ipl_balls" in base.columns and "ipl_runs" in base.columns:
        base["ipl_sr"] = np.where(
            base["ipl_balls"] > 0,
            base["ipl_runs"] / np.maximum(base["ipl_balls"], 1) * 100.0,
            np.nan,
        )
    else:
        base["ipl_sr"] = np.nan

    if "t20i_balls" in base.columns and "t20i_runs" in base.columns:
        base["t20i_sr"] = np.where(
            base["t20i_balls"] > 0,
            base["t20i_runs"] / np.maximum(base["t20i_balls"], 1) * 100.0,
            np.nan,
        )
    else:
        base["t20i_sr"] = np.nan

    # ------------------------------------------------------------------ #
    # Drop internal columns and reorder for clean export
    # ------------------------------------------------------------------ #
    base = base.drop(columns=["_total_boundaries", "_total_dots", "_total_wickets",
                               "_normal_bnd"], errors="ignore")

    output_cols = [
        "player_id", "player_name",
        "total_balls_faced", "total_runs", "total_matches",
        "overall_strike_rate", "boundary_pct", "dot_ball_pct", "dismissal_rate",
        "normal_balls_faced", "normal_runs", "normal_strike_rate", "normal_boundary_pct",
        "ipl_balls", "ipl_runs", "ipl_sr", "ipl_matches",
        "t20i_balls", "t20i_runs", "t20i_sr", "t20i_matches",
    ]
    # Keep only columns that exist (format splits may be absent in thin datasets)
    output_cols = [c for c in output_cols if c in base.columns]
    result = base[output_cols].copy()
    result.index.name = "striker_id"

    logger.info(
        "compute_batter_baselines: complete — %d batters, "
        "total balls=%d, total runs=%d",
        len(result),
        int(result["total_balls_faced"].sum()),
        int(result["total_runs"].sum()),
    )
    return result


@_cache
def compute_bowler_baselines(df: pd.DataFrame) -> pd.DataFrame:
    """Compute per-bowler baseline metrics from the enriched ball-by-ball frame.

    Parameters
    ----------
    df : pd.DataFrame
        Full enriched + pressure-annotated ball-by-ball dataframe (``ms``).

    Returns
    -------
    pd.DataFrame
        One row per bowler_id.  Index = bowler_id.
        Columns documented in module docstring.
    """
    if df is None or df.empty:
        logger.warning("compute_bowler_baselines: received empty dataframe.")
        return pd.DataFrame()

    logger.info("compute_bowler_baselines: processing %d rows", len(df))

    # ------------------------------------------------------------------ #
    # Validate required columns
    # ------------------------------------------------------------------ #
    required = {
        "bowler_id", "bowler_name", "match_id",
        "is_legal_ball", "runs_charged_to_bowler",
        "is_boundary_4", "is_boundary_6", "is_wicket",
        "is_high_pressure_bowl",
    }
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"compute_bowler_baselines: missing columns {missing}")

    # ------------------------------------------------------------------ #
    # Work on legal deliveries only
    # ------------------------------------------------------------------ #
    legal = df[df["is_legal_ball"].astype(bool)].copy()

    # Bowler-credit wicket column
    legal["_bowler_wicket"] = _is_bowler_credit(legal)

    # ------------------------------------------------------------------ #
    # Overall stats
    # ------------------------------------------------------------------ #
    grp = legal.groupby("bowler_id", sort=True)

    total_legal_balls  = grp["is_legal_ball"].count().rename("total_legal_balls")
    total_runs_conc    = grp["runs_charged_to_bowler"].sum().rename("total_runs_conceded")
    total_wickets      = grp["_bowler_wicket"].sum().astype(int).rename("total_wickets")
    total_matches      = grp["match_id"].nunique().rename("total_matches")
    player_name        = grp["bowler_name"].agg(_most_common_name).rename("player_name")

    # Dot balls: legal AND runs_charged_to_bowler == 0
    legal["_is_dot_bowl"] = legal["runs_charged_to_bowler"] == 0
    dot_balls = grp["_is_dot_bowl"].sum().rename("_dot_balls")

    # Boundaries conceded
    legal["_is_boundary_conc"] = (
        legal["is_boundary_4"].astype(bool) | legal["is_boundary_6"].astype(bool)
    )
    boundary_conc = grp["_is_boundary_conc"].sum().rename("_boundary_conc")

    # ------------------------------------------------------------------ #
    # Normal (non-pressure) stats
    # ------------------------------------------------------------------ #
    normal = legal[~legal["is_high_pressure_bowl"].astype(bool)]
    ngrp   = normal.groupby("bowler_id", sort=True)

    normal_balls      = ngrp["is_legal_ball"].count().rename("normal_balls")
    normal_runs_conc  = ngrp["runs_charged_to_bowler"].sum().rename("normal_runs_conceded")

    # ------------------------------------------------------------------ #
    # Format-split stats
    # ------------------------------------------------------------------ #
    def _format_bowler_stats(fmt_label: str, prefix: str) -> pd.DataFrame:
        """Aggregate bowler stats for one format label."""
        if "match_format" not in df.columns:
            logger.warning(
                "compute_bowler_baselines: 'match_format' column missing — "
                "%s stats will be NaN.", fmt_label
            )
            return pd.DataFrame(columns=[
                f"{prefix}_balls", f"{prefix}_runs_conceded",
                f"{prefix}_economy", f"{prefix}_matches", f"{prefix}_wickets",
            ])

        sub = legal[legal["match_format"].astype(str).str.upper() == fmt_label.upper()]
        if sub.empty:
            return pd.DataFrame(columns=[
                f"{prefix}_balls", f"{prefix}_runs_conceded",
                f"{prefix}_economy", f"{prefix}_matches", f"{prefix}_wickets",
            ])

        sg     = sub.groupby("bowler_id", sort=True)
        balls  = sg["is_legal_ball"].count().rename(f"{prefix}_balls")
        runs   = sg["runs_charged_to_bowler"].sum().rename(f"{prefix}_runs_conceded")
        wkts   = sg["_bowler_wicket"].sum().astype(int).rename(f"{prefix}_wickets")
        mats   = sg["match_id"].nunique().rename(f"{prefix}_matches")
        result = pd.concat([balls, runs, wkts, mats], axis=1)

        # Economy
        result[f"{prefix}_economy"] = np.where(
            result[f"{prefix}_balls"] > 0,
            result[f"{prefix}_runs_conceded"] / (result[f"{prefix}_balls"] / 6.0),
            np.nan,
        )
        return result

    ipl_stats  = _format_bowler_stats("T20",  "ipl")
    t20i_stats = _format_bowler_stats("T20I", "t20i")

    # ------------------------------------------------------------------ #
    # Assemble base frame
    # ------------------------------------------------------------------ #
    base = pd.concat(
        [player_name, total_legal_balls, total_runs_conc,
         total_wickets, total_matches,
         dot_balls, boundary_conc,
         normal_balls, normal_runs_conc],
        axis=1,
    )
    base = base.join(ipl_stats,  how="left")
    base = base.join(t20i_stats, how="left")

    # Fill NaN counts
    count_cols = [c for c in base.columns
                  if any(c.endswith(s) for s in ("_balls", "_runs_conceded",
                                                   "_matches", "_wickets"))]
    base[count_cols] = base[count_cols].fillna(0)
    for c in count_cols:
        if not c.endswith("_conceded"):   # leave float for runs columns
            try:
                base[c] = base[c].astype(int)
            except (ValueError, TypeError):
                pass

    # ------------------------------------------------------------------ #
    # Derived rates
    # ------------------------------------------------------------------ #
    base["player_id"] = base.index

    base["total_overs_bowled"] = base["total_legal_balls"] / 6.0

    base["overall_economy"] = np.where(
        base["total_legal_balls"] > 0,
        base["total_runs_conceded"] / np.maximum(base["total_overs_bowled"], 1e-9),
        np.nan,
    )

    base["dot_ball_pct"] = (
        base["_dot_balls"] / np.maximum(base["total_legal_balls"], 1) * 100.0
    ).where(base["total_legal_balls"] > 0, other=np.nan)

    base["boundary_conceded_pct"] = (
        base["_boundary_conc"] / np.maximum(base["total_legal_balls"], 1) * 100.0
    ).where(base["total_legal_balls"] > 0, other=np.nan)

    base["normal_economy"] = np.where(
        base["normal_balls"] > 0,
        base["normal_runs_conceded"] / (base["normal_balls"] / 6.0),
        np.nan,
    )

    # ------------------------------------------------------------------ #
    # Drop internal columns and reorder
    # ------------------------------------------------------------------ #
    base = base.drop(columns=["_dot_balls", "_boundary_conc"], errors="ignore")

    output_cols = [
        "player_id", "player_name",
        "total_legal_balls", "total_runs_conceded", "total_overs_bowled",
        "total_wickets", "overall_economy",
        "dot_ball_pct", "boundary_conceded_pct", "total_matches",
        "normal_balls", "normal_runs_conceded", "normal_economy",
        "ipl_balls", "ipl_runs_conceded", "ipl_economy", "ipl_matches", "ipl_wickets",
        "t20i_balls", "t20i_runs_conceded", "t20i_economy", "t20i_matches", "t20i_wickets",
    ]
    output_cols = [c for c in output_cols if c in base.columns]
    result = base[output_cols].copy()
    result.index.name = "bowler_id"

    logger.info(
        "compute_bowler_baselines: complete — %d bowlers, "
        "total legal balls=%d, total wickets=%d",
        len(result),
        int(result["total_legal_balls"].sum()),
        int(result["total_wickets"].sum()),
    )
    return result
