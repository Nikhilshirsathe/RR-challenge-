"""
scoring_engine.py – Composite pressure-performance ratings for the SUPRR Selector.

Exports
-------
DEFAULT_BATTER_WEIGHTS : dict
DEFAULT_BOWLER_WEIGHTS : dict

compute_batter_scores(batter_pressure_stats, weights=None) -> pd.DataFrame
    Adds raw_score, percentile_score, ipl_raw_score, t20i_raw_score,
    combined_score and component_* columns to the eligible-batter frame.

compute_bowler_scores(bowler_pressure_stats, weights=None) -> pd.DataFrame
    Same shape of output for eligible bowlers.

Both functions are decorated with @st.cache_data (shim degrades without
Streamlit), accept an optional weights dict merged with the defaults, and
handle zero-std metrics gracefully.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Streamlit cache shim
# ---------------------------------------------------------------------------
try:
    import streamlit as st  # type: ignore[import]
    _cache = st.cache_data
    logger.debug("scoring_engine: Streamlit detected – @st.cache_data active.")
except ImportError:
    def _cache(func=None, **_kwargs: Any):  # type: ignore[misc]
        if func is not None:
            return func
        return lambda f: f
    logger.debug("scoring_engine: Streamlit absent – using no-op cache.")

# ---------------------------------------------------------------------------
# Public constants
# ---------------------------------------------------------------------------

DEFAULT_BATTER_WEIGHTS: dict[str, float] = {
    "runs_above_expected": 0.25,
    "sr_improvement":      0.25,
    "dismissal_avoidance": 0.20,
    "boundary_contribution": 0.15,
    "dot_avoidance":       0.15,
}

DEFAULT_BOWLER_WEIGHTS: dict[str, float] = {
    "runs_prevented":      0.25,
    "economy_improvement": 0.25,
    "wicket_taking":       0.20,
    "dot_generation":      0.15,
    "boundary_prevention": 0.15,
}

# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _safe_col(df: pd.DataFrame, *candidates: str) -> pd.Series:
    """Return the first existing column, else a NaN series with the same index."""
    for col in candidates:
        if col in df.columns:
            return df[col].copy()
    logger.debug(
        "scoring_engine._safe_col: none of %s found – returning NaN series.",
        candidates,
    )
    return pd.Series(np.nan, index=df.index)


def _z_score(series: pd.Series, cap: float | None = None) -> pd.Series:
    """Compute z-scores; if std == 0, return 0.  Optionally cap to [-cap, cap]."""
    mean = series.mean(skipna=True)
    std  = series.std(skipna=True)
    if std == 0 or np.isnan(std):
        z = pd.Series(0.0, index=series.index)
    else:
        z = (series - mean) / std
    # fill any remaining NaN (players who had NaN raw value)
    z = z.fillna(0.0)
    if cap is not None:
        z = z.clip(-cap, cap)
    return z


def _percentile_score(raw_scores: pd.Series) -> pd.Series:
    """Convert a series of raw z-composite scores to 0-100 percentile ranks.

    Uses rank / n * 100 so the worst player gets > 0 and the best gets 100.
    Ties receive the average rank.
    """
    n = len(raw_scores)
    if n == 0:
        return raw_scores.copy()
    ranks = raw_scores.rank(method="average", na_option="bottom")
    return (ranks / n * 100.0).clip(0.0, 100.0)


def _normalize_0_100(series: pd.Series) -> pd.Series:
    """Min-max scale a series to [0, 100].  Returns 50 if all values identical."""
    mn, mx = series.min(skipna=True), series.max(skipna=True)
    if np.isnan(mn) or np.isnan(mx) or mx == mn:
        return pd.Series(50.0, index=series.index)
    return ((series - mn) / (mx - mn) * 100.0).clip(0.0, 100.0)


def _composite_z(df: pd.DataFrame, component_map: dict[str, tuple[pd.Series, float]]) -> pd.Series:
    """Weighted sum of pre-computed z-score series.

    Parameters
    ----------
    component_map : {col_name: (z_series, weight)}
    """
    total = pd.Series(0.0, index=df.index)
    for _name, (z_series, weight) in component_map.items():
        total = total + weight * z_series
    return total


def _format_raw_score(
    df: pd.DataFrame,
    fmt_col_balls: str,
    component_builders: "list[tuple[str, Any, float, float | None]]",
) -> pd.Series:
    """Compute a raw z-composite score restricted to players in a given format.

    Parameters
    ----------
    df : pd.DataFrame
        Full eligible-player frame.
    fmt_col_balls : str
        Column holding the number of pressure balls in this format (used to
        identify players who have data for the format).
    component_builders : list of (raw_series, weight, z_cap) tuples
        Each item is (raw_series, weight, cap_value).
    """
    # Only players with at least 1 pressure ball in this format contribute to
    # the format-specific z-scoring pool.  Others receive NaN, then 0 after fill.
    has_data = df[fmt_col_balls].fillna(0) > 0

    result = pd.Series(0.0, index=df.index)
    for raw_series, weight, cap in component_builders:
        # Restrict to players with format data to compute group statistics
        sub = raw_series.where(has_data)
        z   = _z_score(sub, cap=cap)
        result = result + weight * z

    # Players with no format data get NaN so the caller can decide how to blend
    result = result.where(has_data, other=np.nan)
    return result


# ---------------------------------------------------------------------------
# Public API – batters
# ---------------------------------------------------------------------------

@_cache
def compute_batter_scores(
    batter_pressure_stats: pd.DataFrame,
    weights: dict[str, float] | None = None,
) -> pd.DataFrame:
    """Add composite rating columns to the eligible-batter pressure-stats frame.

    Parameters
    ----------
    batter_pressure_stats : pd.DataFrame
        Output of ``batting_analysis.get_eligible_batters`` (or
        ``compute_batter_pressure_stats``).  Index = striker_id.
    weights : dict, optional
        Partial or full override of DEFAULT_BATTER_WEIGHTS.  Keys that are
        missing will fall back to the defaults.

    Returns
    -------
    pd.DataFrame
        Input frame with the following columns appended:

        component_runs_above_exp, component_sr, component_dismissal,
        component_boundary, component_dots
            Individual z-scores (capped to ±3 where noted).
        raw_score
            Weighted z-composite (unbounded).
        percentile_score
            0-100 percentile rank within the eligible group.
        ipl_raw_score, t20i_raw_score
            Format-specific weighted z-composites (NaN for formats with no data).
        combined_score
            Format-balanced 0-100 score.  50/50 blend when both formats present;
            single-format score otherwise.
    """
    if batter_pressure_stats is None or batter_pressure_stats.empty:
        logger.warning("compute_batter_scores: received empty DataFrame – returning as-is.")
        return batter_pressure_stats if batter_pressure_stats is not None else pd.DataFrame()

    df = batter_pressure_stats.copy()
    n  = len(df)
    logger.info("compute_batter_scores: scoring %d eligible batters.", n)

    # Merge weights
    w = {**DEFAULT_BATTER_WEIGHTS, **(weights or {})}
    logger.debug("compute_batter_scores: effective weights = %s", w)

    # ------------------------------------------------------------------ #
    # Component raw series
    # ------------------------------------------------------------------ #

    # 1. runs_above_expected  (fall back to sr_improvement if column absent)
    if "runs_above_expected_pressure" in df.columns:
        raw_rae = _safe_col(df, "runs_above_expected_pressure")
    else:
        logger.info(
            "compute_batter_scores: 'runs_above_expected_pressure' absent – "
            "using 'sr_improvement' as proxy."
        )
        raw_rae = _safe_col(df, "sr_improvement")

    # 2. sr_improvement
    raw_sr = _safe_col(df, "sr_improvement")

    # 3. dismissal_avoidance  (negate so lower dismissal rate → higher score)
    raw_dis = -_safe_col(df, "pressure_dismissal_rate")

    # 4. boundary_contribution
    raw_bnd = _safe_col(df, "pressure_boundary_pct")

    # 5. dot_avoidance  (negate so fewer dots → higher score)
    raw_dot = -_safe_col(df, "pressure_dot_pct")

    # ------------------------------------------------------------------ #
    # Overall z-scores (across the full eligible group)
    # ------------------------------------------------------------------ #
    z_rae = _z_score(raw_rae, cap=3.0)
    z_sr  = _z_score(raw_sr,  cap=3.0)
    z_dis = _z_score(raw_dis)
    z_bnd = _z_score(raw_bnd)
    z_dot = _z_score(raw_dot)

    df["component_runs_above_exp"] = z_rae
    df["component_sr"]             = z_sr
    df["component_dismissal"]      = z_dis
    df["component_boundary"]       = z_bnd
    df["component_dots"]           = z_dot

    # ------------------------------------------------------------------ #
    # Overall raw score
    # ------------------------------------------------------------------ #
    df["raw_score"] = (
        w["runs_above_expected"] * z_rae
        + w["sr_improvement"]    * z_sr
        + w["dismissal_avoidance"]  * z_dis
        + w["boundary_contribution"] * z_bnd
        + w["dot_avoidance"]     * z_dot
    )

    df["percentile_score"] = _percentile_score(df["raw_score"])

    # ------------------------------------------------------------------ #
    # Format-specific raw scores
    # ------------------------------------------------------------------ #
    # IPL (match_format == 'T20')
    # Component series are approximated from format-split columns where available.
    # For format splits, we use pressure SR as the primary dimension available.
    # For a full decomposition we'd need per-format dismissal/dot/boundary,
    # which are not in the upstream frame; we approximate using the available
    # format-level SR, then fall back to overall components.

    def _fmt_raw_score(prefix: str, balls_col: str) -> pd.Series:
        """Build a simplified z-composite for one format using available columns."""
        # Strike-rate proxy for runs_above_expected and sr_improvement
        sr_col    = f"{prefix}_pressure_sr"
        balls_col_ = f"{prefix}_pressure_balls"

        if sr_col in df.columns and balls_col_ in df.columns:
            # Only players who have pressure balls in this format contribute
            has_data = df[balls_col_].fillna(0) > 0

            # Format-specific raw components (use full z-score only for SR;
            # the other components fall back to the overall z if no format split)
            fmt_sr_raw = _safe_col(df, sr_col).where(has_data)
            z_fmt_sr   = _z_score(fmt_sr_raw)

            # Approximate format-specific runs_above_expected as format SR deviation
            # from full-group mean (best available proxy)
            fmt_composite = (
                w["runs_above_expected"] * z_fmt_sr          # proxy
                + w["sr_improvement"]    * z_fmt_sr          # same proxy
                + w["dismissal_avoidance"]  * z_dis.where(has_data, 0.0)
                + w["boundary_contribution"] * z_bnd.where(has_data, 0.0)
                + w["dot_avoidance"]     * z_dot.where(has_data, 0.0)
            )
            return fmt_composite.where(has_data, other=np.nan)
        else:
            logger.debug(
                "compute_batter_scores: format columns for '%s' absent – "
                "format score will be NaN.", prefix
            )
            return pd.Series(np.nan, index=df.index)

    df["ipl_raw_score"]  = _fmt_raw_score("ipl",  "ipl_pressure_balls")
    df["t20i_raw_score"] = _fmt_raw_score("t20i", "t20i_pressure_balls")

    # ------------------------------------------------------------------ #
    # Combined score (format-balanced 0-100)
    # ------------------------------------------------------------------ #
    ipl_norm  = _normalize_0_100(df["ipl_raw_score"].fillna(df["ipl_raw_score"].median()))
    t20i_norm = _normalize_0_100(df["t20i_raw_score"].fillna(df["t20i_raw_score"].median()))

    has_ipl  = df["ipl_pressure_balls"].fillna(0) > 0  if "ipl_pressure_balls"  in df.columns else pd.Series(False, index=df.index)
    has_t20i = df["t20i_pressure_balls"].fillna(0) > 0 if "t20i_pressure_balls" in df.columns else pd.Series(False, index=df.index)

    combined = pd.Series(np.nan, index=df.index)

    # Both formats
    both_mask = has_ipl & has_t20i
    combined.loc[both_mask] = (
        0.5 * ipl_norm.loc[both_mask] + 0.5 * t20i_norm.loc[both_mask]
    )

    # IPL only
    ipl_only_mask = has_ipl & ~has_t20i
    combined.loc[ipl_only_mask] = ipl_norm.loc[ipl_only_mask]

    # T20I only
    t20i_only_mask = ~has_ipl & has_t20i
    combined.loc[t20i_only_mask] = t20i_norm.loc[t20i_only_mask]

    # Neither (should not occur in eligible set, but guard anyway)
    neither_mask = ~has_ipl & ~has_t20i
    combined.loc[neither_mask] = df["percentile_score"].loc[neither_mask]

    df["combined_score"] = combined

    logger.info(
        "compute_batter_scores: complete – raw_score range [%.3f, %.3f], "
        "percentile_score range [%.1f, %.1f], combined_score range [%.1f, %.1f].",
        df["raw_score"].min(), df["raw_score"].max(),
        df["percentile_score"].min(), df["percentile_score"].max(),
        df["combined_score"].min(), df["combined_score"].max(),
    )
    return df


# ---------------------------------------------------------------------------
# Public API – bowlers
# ---------------------------------------------------------------------------

@_cache
def compute_bowler_scores(
    bowler_pressure_stats: pd.DataFrame,
    weights: dict[str, float] | None = None,
) -> pd.DataFrame:
    """Add composite rating columns to the eligible-bowler pressure-stats frame.

    Parameters
    ----------
    bowler_pressure_stats : pd.DataFrame
        Output of ``bowling_analysis.get_eligible_bowlers``.  Index = bowler_id.
    weights : dict, optional
        Partial or full override of DEFAULT_BOWLER_WEIGHTS.

    Returns
    -------
    pd.DataFrame
        Input frame with appended columns:

        component_runs_prevented, component_economy, component_wickets,
        component_dots, component_boundary_prev
            Individual z-scores.
        raw_score, percentile_score
            Overall composite and 0-100 rank.
        ipl_raw_score, t20i_raw_score
            Format-specific z-composites.
        combined_score
            Format-balanced 0-100 score.
    """
    if bowler_pressure_stats is None or bowler_pressure_stats.empty:
        logger.warning("compute_bowler_scores: received empty DataFrame – returning as-is.")
        return bowler_pressure_stats if bowler_pressure_stats is not None else pd.DataFrame()

    df = bowler_pressure_stats.copy()
    n  = len(df)
    logger.info("compute_bowler_scores: scoring %d eligible bowlers.", n)

    # Merge weights
    w = {**DEFAULT_BOWLER_WEIGHTS, **(weights or {})}
    logger.debug("compute_bowler_scores: effective weights = %s", w)

    # ------------------------------------------------------------------ #
    # Component raw series
    # ------------------------------------------------------------------ #

    # 1. runs_prevented  (positive = conceded fewer runs than expected)
    raw_rp = _safe_col(df, "runs_prevented_pressure")

    # 2. economy_improvement = baseline_economy - pressure_economy (positive = better)
    if "economy_improvement" in df.columns:
        raw_ei = _safe_col(df, "economy_improvement")
    else:
        # Compute on the fly if columns are available
        bl_econ = _safe_col(df, "baseline_economy")
        pr_econ = _safe_col(df, "pressure_economy")
        raw_ei  = bl_econ - pr_econ
        logger.info(
            "compute_bowler_scores: 'economy_improvement' absent – computed "
            "from baseline_economy - pressure_economy."
        )

    # 3. wicket_taking
    if "wickets_above_expected" in df.columns:
        raw_wk = _safe_col(df, "wickets_above_expected")
    else:
        # pressure_wickets / pressure_overs as a crude rate
        wk  = _safe_col(df, "pressure_wickets")
        ovs = _safe_col(df, "pressure_overs").replace(0, np.nan)
        raw_wk = wk / ovs
        logger.info(
            "compute_bowler_scores: 'wickets_above_expected' absent – using "
            "pressure_wickets / pressure_overs as proxy."
        )

    # 4. dot_generation
    raw_dg = _safe_col(df, "pressure_dot_pct")

    # 5. boundary_prevention  (negate: fewer boundaries → higher score)
    raw_bp = -_safe_col(df, "pressure_boundary_pct")

    # ------------------------------------------------------------------ #
    # Overall z-scores
    # ------------------------------------------------------------------ #
    z_rp  = _z_score(raw_rp)
    z_ei  = _z_score(raw_ei)
    z_wk  = _z_score(raw_wk)
    z_dg  = _z_score(raw_dg)
    z_bp  = _z_score(raw_bp)

    df["component_runs_prevented"]    = z_rp
    df["component_economy"]           = z_ei
    df["component_wickets"]           = z_wk
    df["component_dots"]              = z_dg
    df["component_boundary_prev"]     = z_bp

    # ------------------------------------------------------------------ #
    # Overall raw score
    # ------------------------------------------------------------------ #
    df["raw_score"] = (
        w["runs_prevented"]      * z_rp
        + w["economy_improvement"] * z_ei
        + w["wicket_taking"]       * z_wk
        + w["dot_generation"]      * z_dg
        + w["boundary_prevention"] * z_bp
    )

    df["percentile_score"] = _percentile_score(df["raw_score"])

    # ------------------------------------------------------------------ #
    # Format-specific raw scores
    # ------------------------------------------------------------------ #
    def _fmt_raw_score_bowl(prefix: str) -> pd.Series:
        balls_col = f"{prefix}_pressure_balls"
        econ_col  = f"{prefix}_pressure_economy"
        wkts_col  = f"{prefix}_pressure_wickets"

        if balls_col not in df.columns:
            logger.debug(
                "compute_bowler_scores: format columns for '%s' absent – "
                "format score will be NaN.", prefix
            )
            return pd.Series(np.nan, index=df.index)

        has_data = df[balls_col].fillna(0) > 0

        # Economy-based proxy for the format score
        if econ_col in df.columns:
            # Economy improvement proxy: how much better than group mean economy
            fmt_econ = _safe_col(df, econ_col).where(has_data)
            z_fmt_ei = _z_score(-fmt_econ)  # negate: lower economy = better
        else:
            z_fmt_ei = z_ei.where(has_data, 0.0)

        if wkts_col in df.columns:
            balls_denom = df[balls_col].replace(0, np.nan).where(has_data)
            fmt_wk_rate = _safe_col(df, wkts_col).where(has_data) / (balls_denom / 6.0)
            z_fmt_wk    = _z_score(fmt_wk_rate)
        else:
            z_fmt_wk = z_wk.where(has_data, 0.0)

        fmt_composite = (
            w["runs_prevented"]      * z_fmt_ei   # proxy
            + w["economy_improvement"] * z_fmt_ei
            + w["wicket_taking"]       * z_fmt_wk
            + w["dot_generation"]      * z_dg.where(has_data, 0.0)
            + w["boundary_prevention"] * z_bp.where(has_data, 0.0)
        )
        return fmt_composite.where(has_data, other=np.nan)

    df["ipl_raw_score"]  = _fmt_raw_score_bowl("ipl")
    df["t20i_raw_score"] = _fmt_raw_score_bowl("t20i")

    # ------------------------------------------------------------------ #
    # Combined score (format-balanced 0-100)
    # ------------------------------------------------------------------ #
    ipl_norm  = _normalize_0_100(df["ipl_raw_score"].fillna(df["ipl_raw_score"].median()))
    t20i_norm = _normalize_0_100(df["t20i_raw_score"].fillna(df["t20i_raw_score"].median()))

    has_ipl  = df["ipl_pressure_balls"].fillna(0)  > 0 if "ipl_pressure_balls"  in df.columns else pd.Series(False, index=df.index)
    has_t20i = df["t20i_pressure_balls"].fillna(0) > 0 if "t20i_pressure_balls" in df.columns else pd.Series(False, index=df.index)

    combined = pd.Series(np.nan, index=df.index)

    both_mask     = has_ipl & has_t20i
    ipl_only_mask = has_ipl & ~has_t20i
    t20i_only_mask = ~has_ipl & has_t20i
    neither_mask  = ~has_ipl & ~has_t20i

    combined.loc[both_mask]      = 0.5 * ipl_norm.loc[both_mask] + 0.5 * t20i_norm.loc[both_mask]
    combined.loc[ipl_only_mask]  = ipl_norm.loc[ipl_only_mask]
    combined.loc[t20i_only_mask] = t20i_norm.loc[t20i_only_mask]
    combined.loc[neither_mask]   = df["percentile_score"].loc[neither_mask]

    df["combined_score"] = combined

    logger.info(
        "compute_bowler_scores: complete – raw_score range [%.3f, %.3f], "
        "percentile_score range [%.1f, %.1f], combined_score range [%.1f, %.1f].",
        df["raw_score"].min(), df["raw_score"].max(),
        df["percentile_score"].min(), df["percentile_score"].max(),
        df["combined_score"].min(), df["combined_score"].max(),
    )
    return df
