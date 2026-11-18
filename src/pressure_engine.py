"""
pressure_engine.py – Compute per-delivery pressure indices for the SUPRR Selector.

Exports
-------
compute_pressure(df, bpi_weights=None, bopi_weights=None)
    Main entry point. Accepts the enriched ball-by-ball dataframe produced by
    ``match_state.compute_match_states`` and returns it with BPI/BoPI columns
    appended.

DEFAULT_BPI_WEIGHTS : dict
    Default component weights for the Batter Pressure Index.

DEFAULT_BOPI_WEIGHTS : dict
    Default component weights for the Bowler Pressure Index.

PRESSURE_THRESHOLDS : dict
    Boundary tuples (lo, hi) for 'low', 'moderate', 'high', 'extreme'.

get_pressure_summary(df) -> dict
    Aggregate pressure statistics over a pressure-annotated dataframe.

Index definitions
-----------------
BPI  – Batter Pressure Index (0–100).
       Reflects how much situational pressure the batting side is under on
       this delivery.  Higher = more pressure on the batter.

BoPI – Bowler Pressure Index (0–100).
       Reflects how much situational pressure the bowling side is under on
       this delivery.  Higher = more pressure on the bowler.

Intermediate component columns are exposed so the dashboard can break down
each index into its contributing factors:

BPI components
  bpi_rrr_component       – A: Required Run Rate Difficulty   (30 %)
  bpi_wickets_component   – B: Wickets Lost                   (20 %)
  bpi_balls_component     – C: Balls Remaining (bat view)     (15 %)
  bpi_phase_component     – D: Innings Phase                  (10 %)
  bpi_leverage_component  – E: Match Leverage                 (25 %)

BoPI components
  bopi_threat_component   – A: Opposition Scoring Threat      (25 %)
  bopi_wih_component      – B: Wickets in Hand                (20 %)
  bopi_balls_component    – C: Balls Remaining (bowl view)    (15 %)
  bopi_leverage_component – D: Match Leverage                 (25 %)
  bopi_phase_component    – E: Innings Phase                  (15 %)
"""

from __future__ import annotations

import logging
from typing import Dict, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Public constants
# ---------------------------------------------------------------------------

DEFAULT_BPI_WEIGHTS: Dict[str, float] = {
    "required_run_rate_difficulty": 0.30,
    "wickets_lost":                 0.20,
    "balls_remaining":              0.15,
    "innings_phase":                0.10,
    "match_leverage":               0.25,
}

DEFAULT_BOPI_WEIGHTS: Dict[str, float] = {
    "opposition_scoring_threat": 0.25,
    "wickets_in_hand":           0.20,
    "balls_remaining":           0.15,
    "match_leverage":            0.25,
    "innings_phase":             0.15,
}

PRESSURE_THRESHOLDS: Dict[str, tuple] = {
    "low":      (0,   25),
    "moderate": (25,  50),
    "high":     (50,  75),
    "extreme":  (75, 100),
}

# ---------------------------------------------------------------------------
# Streamlit cache decorator (optional – degrades gracefully without Streamlit)
# ---------------------------------------------------------------------------
try:
    import streamlit as st
    _cache = st.cache_data
    logger.debug("pressure_engine: Streamlit detected – @st.cache_data available.")
except ImportError:
    # Provide a no-op decorator so the module loads without Streamlit.
    def _cache(func=None, **_kwargs):  # type: ignore[misc]
        if func is not None:
            return func
        return lambda f: f

    logger.debug("pressure_engine: Streamlit not available – using no-op cache decorator.")


# ===========================================================================
# BPI component functions
# Each accepts a pd.DataFrame and returns a float64 Series (0–100).
# ===========================================================================

def _bpi_rrr_component(df: pd.DataFrame) -> pd.Series:
    """Component A – Required Run Rate Difficulty (0–100).

    Innings 2 uses pre_rrr; innings 1 uses scoring_difficulty_1st_inn
    (par_score_at_this_stage − pre_score).  Inf/NaN pre_rrr maps to 100.
    """
    logger.debug("Computing BPI component A: RRR difficulty")

    is_inn2 = df["innings"].astype(int) == 2

    # ---- Innings 2 -------------------------------------------------------
    rrr = pd.to_numeric(df["pre_rrr"], errors="coerce")

    # Replace inf with a sentinel that maps to score = 100
    rrr_capped = rrr.replace([np.inf, -np.inf], np.nan)

    inn2_score = np.where(
        rrr_capped.isna(),   # was inf or genuinely NaN (treat as 100)
        100.0,
        np.where(
            rrr_capped <= 6,
            0.0,
            np.where(
                rrr_capped <= 12,
                (rrr_capped - 6.0) / 6.0 * 60.0,
                np.where(
                    rrr_capped <= 18,
                    60.0 + (rrr_capped - 12.0) / 6.0 * 40.0,
                    100.0,
                ),
            ),
        ),
    )

    # ---- Innings 1 -------------------------------------------------------
    diff = pd.to_numeric(df["scoring_difficulty_1st_inn"], errors="coerce").fillna(0.0)

    inn1_score = np.where(
        diff <= -20,
        0.0,
        np.where(
            diff <= 0,
            (diff + 20.0) / 20.0 * 30.0,
            np.where(
                diff <= 30,
                30.0 + diff / 30.0 * 70.0,
                100.0,
            ),
        ),
    )

    score = np.where(is_inn2, inn2_score, inn1_score)
    return pd.Series(score, index=df.index, dtype="float64").clip(0, 100)


def _bpi_wickets_component(df: pd.DataFrame) -> pd.Series:
    """Component B – Wickets Lost (0–100).

    Maps pre_wickets (0–10) to pressure score via piecewise linear scale.
    """
    logger.debug("Computing BPI component B: wickets lost")

    w = pd.to_numeric(df["pre_wickets"], errors="coerce").fillna(0.0)

    score = np.where(
        w <= 0,
        0.0,
        np.where(
            w <= 3,
            w / 3.0 * 30.0,
            np.where(
                w <= 6,
                30.0 + (w - 3.0) / 3.0 * 40.0,
                np.where(
                    w <= 9,
                    70.0 + (w - 6.0) / 3.0 * 30.0,
                    100.0,
                ),
            ),
        ),
    )
    return pd.Series(score, index=df.index, dtype="float64").clip(0, 100)


def _bpi_balls_remaining_component(df: pd.DataFrame) -> pd.Series:
    """Component C – Balls Remaining, batter view (0–100).

    More balls remaining = less urgent = lower score.
    """
    logger.debug("Computing BPI component C: balls remaining (bat)")

    b = pd.to_numeric(df["pre_balls_remaining"], errors="coerce").fillna(0.0)

    score = np.where(
        b >= 60,
        0.0,
        np.where(
            b >= 30,
            (60.0 - b) / 30.0 * 40.0,
            np.where(
                b >= 10,
                40.0 + (30.0 - b) / 20.0 * 40.0,
                np.where(
                    b > 0,
                    80.0 + (10.0 - b) / 10.0 * 20.0,
                    100.0,
                ),
            ),
        ),
    )
    return pd.Series(score, index=df.index, dtype="float64").clip(0, 100)


def _bpi_phase_component(df: pd.DataFrame) -> pd.Series:
    """Component D – Innings Phase (0–100).

    powerplay → 20, middle → 40, death → 80.
    """
    logger.debug("Computing BPI component D: innings phase")

    phase = df["innings_phase"].astype(str).str.lower().str.strip()
    score = (
        phase
        .map({"powerplay": 20.0, "middle": 40.0, "death": 80.0})
        .fillna(40.0)   # default to middle if unknown
    )
    return score.astype("float64")


def _leverage_component(df: pd.DataFrame) -> pd.Series:
    """Shared leverage score used by BPI component E and BoPI component D.

    Innings 2: based on runs required vs balls remaining.
    Innings 1: phase-based lookup.
    Returns 0–100.
    """
    logger.debug("Computing leverage component")

    is_inn2 = df["innings"].astype(int) == 2

    runs_req   = pd.to_numeric(df["pre_runs_required"],  errors="coerce").fillna(0.0)
    balls_rem  = pd.to_numeric(df["pre_balls_remaining"], errors="coerce").fillna(0.0)
    target     = pd.to_numeric(df["target"],              errors="coerce").fillna(1.0)

    # Innings 2 leverage
    # Condition: runs_required > balls_remaining * 1.5
    massive_pressure = runs_req > balls_rem * 1.5

    # Condition: batting team has already won (runs_required <= 0)
    already_won = runs_req <= 0

    # Base case: 20 + (runs_required / max(1, target)) * 60
    safe_target  = np.maximum(target, 1.0)
    leverage_mid = 20.0 + (runs_req / safe_target) * 60.0

    inn2_leverage = np.where(
        already_won,
        10.0,
        np.where(massive_pressure, 80.0, leverage_mid),
    )

    # Innings 1 leverage (phase-based)
    phase = df["innings_phase"].astype(str).str.lower().str.strip()
    inn1_leverage = (
        phase
        .map({"powerplay": 30.0, "middle": 50.0, "death": 80.0})
        .fillna(50.0)
        .to_numpy()
    )

    leverage = np.where(is_inn2, inn2_leverage, inn1_leverage)
    return pd.Series(leverage, index=df.index, dtype="float64").clip(0, 100)


# ===========================================================================
# BoPI component functions
# ===========================================================================

def _bopi_wickets_in_hand_component(df: pd.DataFrame) -> pd.Series:
    """BoPI Component B – Wickets in Hand (0–100).

    More wickets remaining = harder for the bowler (higher score).
    """
    logger.debug("Computing BoPI component B: wickets in hand")

    wih = pd.to_numeric(df["pre_wickets_remaining"], errors="coerce").fillna(0.0)

    score = np.where(
        wih >= 10,
        100.0,
        np.where(
            wih >= 7,
            60.0 + (wih - 6.0) / 4.0 * 40.0,
            np.where(
                wih >= 4,
                30.0 + (wih - 3.0) / 3.0 * 30.0,
                np.where(
                    wih >= 1,
                    10.0 + (wih - 1.0) / 2.0 * 20.0,
                    0.0,
                ),
            ),
        ),
    )
    return pd.Series(score, index=df.index, dtype="float64").clip(0, 100)


def _bopi_balls_remaining_component(df: pd.DataFrame) -> pd.Series:
    """BoPI Component C – Balls Remaining, bowler view (0–100).

    More balls remaining = more onus on the bowler = higher score.
    """
    logger.debug("Computing BoPI component C: balls remaining (bowl)")

    b = pd.to_numeric(df["pre_balls_remaining"], errors="coerce").fillna(0.0)

    score = np.where(
        b >= 60,
        100.0,
        np.where(
            b >= 30,
            60.0 + (b - 30.0) / 30.0 * 40.0,
            np.where(
                b >= 10,
                20.0 + (b - 10.0) / 20.0 * 40.0,
                b / 10.0 * 20.0,
            ),
        ),
    )
    return pd.Series(score, index=df.index, dtype="float64").clip(0, 100)


def _bopi_phase_component(df: pd.DataFrame) -> pd.Series:
    """BoPI Component E – Innings Phase (0–100).

    powerplay → 60, middle → 40, death → 90.
    """
    logger.debug("Computing BoPI component E: innings phase")

    phase = df["innings_phase"].astype(str).str.lower().str.strip()
    score = (
        phase
        .map({"powerplay": 60.0, "middle": 40.0, "death": 90.0})
        .fillna(40.0)
    )
    return score.astype("float64")


# ===========================================================================
# Pressure classification helper
# ===========================================================================

def _classify_pressure(series: pd.Series) -> pd.Series:
    """Map a 0–100 numeric series to 'low'/'moderate'/'high'/'extreme' labels.

    Boundaries (inclusive on left, exclusive on right, except the top):
        [0, 25)  → 'low'
        [25, 50) → 'moderate'
        [50, 75) → 'high'
        [75, 100] → 'extreme'
    """
    return pd.cut(
        series,
        bins=[
            PRESSURE_THRESHOLDS["low"][0],
            PRESSURE_THRESHOLDS["moderate"][0],
            PRESSURE_THRESHOLDS["high"][0],
            PRESSURE_THRESHOLDS["extreme"][0],
            PRESSURE_THRESHOLDS["extreme"][1] + 1e-9,  # include 100
        ],
        labels=["low", "moderate", "high", "extreme"],
        right=False,
        include_lowest=True,
    ).astype(str)


# ===========================================================================
# Main public API
# ===========================================================================

@_cache
def compute_pressure(
    df: pd.DataFrame,
    bpi_weights:  Optional[Dict[str, float]] = None,
    bopi_weights: Optional[Dict[str, float]] = None,
) -> pd.DataFrame:
    """Compute BPI and BoPI for every delivery and append results to *df*.

    Parameters
    ----------
    df : pd.DataFrame
        Enriched ball-by-ball dataframe produced by
        ``match_state.compute_match_states``.  Must contain at minimum:
        innings, innings_phase, pre_wickets, pre_wickets_remaining,
        pre_balls_remaining, pre_rrr, pre_runs_required, target,
        scoring_difficulty_1st_inn, par_score_at_this_stage, pre_score.

    bpi_weights : dict, optional
        Override specific BPI component weights.  Any key not supplied falls
        back to ``DEFAULT_BPI_WEIGHTS``.  Weights need not sum to 1 — they
        are used as-is; it is the caller's responsibility to keep totals at 1
        for a meaningful 0–100 final index.

    bopi_weights : dict, optional
        Override specific BoPI component weights (same convention).

    Returns
    -------
    pd.DataFrame
        Input dataframe with the following extra columns:

        Intermediate BPI components (0–100 each):
            bpi_rrr_component, bpi_wickets_component, bpi_balls_component,
            bpi_phase_component, bpi_leverage_component

        Intermediate BoPI components (0–100 each):
            bopi_threat_component, bopi_wih_component, bopi_balls_component,
            bopi_leverage_component, bopi_phase_component

        Final indices and classifications:
            bpi, bopi,
            pressure_category_bat, pressure_category_bowl,
            is_high_pressure_bat, is_high_pressure_bowl
    """
    if df is None or df.empty:
        logger.warning("compute_pressure received empty dataframe — returning as-is.")
        return df

    logger.info("compute_pressure: processing %d deliveries", len(df))

    # ------------------------------------------------------------------
    # Merge supplied weights with defaults
    # ------------------------------------------------------------------
    bw = {**DEFAULT_BPI_WEIGHTS,  **(bpi_weights  or {})}
    ow = {**DEFAULT_BOPI_WEIGHTS, **(bopi_weights or {})}
    logger.debug("BPI weights:  %s", bw)
    logger.debug("BoPI weights: %s", ow)

    out = df.copy()

    # ------------------------------------------------------------------
    # BPI components (all 0–100)
    # ------------------------------------------------------------------
    logger.info("compute_pressure: computing BPI components")

    out["bpi_rrr_component"]      = _bpi_rrr_component(out)
    out["bpi_wickets_component"]  = _bpi_wickets_component(out)
    out["bpi_balls_component"]    = _bpi_balls_remaining_component(out)
    out["bpi_phase_component"]    = _bpi_phase_component(out)
    out["bpi_leverage_component"] = _leverage_component(out)

    # ------------------------------------------------------------------
    # BPI final score
    # ------------------------------------------------------------------
    out["bpi"] = (
        bw["required_run_rate_difficulty"] * out["bpi_rrr_component"]
        + bw["wickets_lost"]               * out["bpi_wickets_component"]
        + bw["balls_remaining"]            * out["bpi_balls_component"]
        + bw["innings_phase"]              * out["bpi_phase_component"]
        + bw["match_leverage"]             * out["bpi_leverage_component"]
    ).clip(0, 100)

    logger.info(
        "compute_pressure: BPI computed — mean=%.2f, min=%.2f, max=%.2f",
        out["bpi"].mean(), out["bpi"].min(), out["bpi"].max(),
    )

    # ------------------------------------------------------------------
    # BoPI components (all 0–100)
    # Component A re-uses BPI directly
    # Components D (leverage) mirrors BPI component E
    # ------------------------------------------------------------------
    logger.info("compute_pressure: computing BoPI components")

    out["bopi_threat_component"]   = out["bpi"]                           # A
    out["bopi_wih_component"]      = _bopi_wickets_in_hand_component(out)  # B
    out["bopi_balls_component"]    = _bopi_balls_remaining_component(out)  # C
    out["bopi_leverage_component"] = out["bpi_leverage_component"]         # D (shared)
    out["bopi_phase_component"]    = _bopi_phase_component(out)            # E

    # ------------------------------------------------------------------
    # BoPI final score
    # ------------------------------------------------------------------
    out["bopi"] = (
        ow["opposition_scoring_threat"] * out["bopi_threat_component"]
        + ow["wickets_in_hand"]         * out["bopi_wih_component"]
        + ow["balls_remaining"]         * out["bopi_balls_component"]
        + ow["match_leverage"]          * out["bopi_leverage_component"]
        + ow["innings_phase"]           * out["bopi_phase_component"]
    ).clip(0, 100)

    logger.info(
        "compute_pressure: BoPI computed — mean=%.2f, min=%.2f, max=%.2f",
        out["bopi"].mean(), out["bopi"].min(), out["bopi"].max(),
    )

    # ------------------------------------------------------------------
    # Classification labels
    # ------------------------------------------------------------------
    logger.info("compute_pressure: classifying pressure levels")

    out["pressure_category_bat"]  = _classify_pressure(out["bpi"])
    out["pressure_category_bowl"] = _classify_pressure(out["bopi"])

    # ------------------------------------------------------------------
    # Boolean high-pressure flags
    # ------------------------------------------------------------------
    out["is_high_pressure_bat"]  = out["bpi"]  >= 50.0
    out["is_high_pressure_bowl"] = out["bopi"] >= 50.0

    logger.info(
        "compute_pressure: complete. "
        "high-pressure deliveries: bat=%d (%.1f%%), bowl=%d (%.1f%%)",
        out["is_high_pressure_bat"].sum(),
        out["is_high_pressure_bat"].mean() * 100,
        out["is_high_pressure_bowl"].sum(),
        out["is_high_pressure_bowl"].mean() * 100,
    )

    return out


# ===========================================================================
# Summary helper
# ===========================================================================

def get_pressure_summary(df: pd.DataFrame) -> dict:
    """Return aggregate pressure statistics over a pressure-annotated dataframe.

    Expects ``df`` to already have the columns produced by
    ``compute_pressure``.  Call ``compute_pressure`` first if they are absent.

    Parameters
    ----------
    df : pd.DataFrame
        Dataframe with bpi, bopi, pressure_category_bat,
        pressure_category_bowl, is_high_pressure_bat,
        is_high_pressure_bowl, innings_phase columns.

    Returns
    -------
    dict with keys:
        total_deliveries        – int
        high_pressure_bat_count – int
        high_pressure_bowl_count– int
        pressure_distribution   – dict mapping category → count for bat and bowl
        phase_pressure_means    – dict mapping phase → mean BPI and mean BoPI
    """
    required_cols = {
        "bpi", "bopi", "pressure_category_bat", "pressure_category_bowl",
        "is_high_pressure_bat", "is_high_pressure_bowl", "innings_phase",
    }
    missing = required_cols - set(df.columns)
    if missing:
        logger.error(
            "get_pressure_summary: missing columns %s — run compute_pressure first.", missing
        )
        raise ValueError(
            f"get_pressure_summary: missing columns {missing}. "
            "Call compute_pressure(df) before get_pressure_summary(df)."
        )

    logger.info("get_pressure_summary: summarising %d rows", len(df))

    total = len(df)

    high_bat  = int(df["is_high_pressure_bat"].sum())
    high_bowl = int(df["is_high_pressure_bowl"].sum())

    # Category distribution for bat and bowl
    cat_order = ["low", "moderate", "high", "extreme"]

    bat_dist  = df["pressure_category_bat"].value_counts()
    bowl_dist = df["pressure_category_bowl"].value_counts()

    pressure_distribution = {
        "bat":  {cat: int(bat_dist.get(cat,  0)) for cat in cat_order},
        "bowl": {cat: int(bowl_dist.get(cat, 0)) for cat in cat_order},
    }

    # Phase-level pressure means
    phases = ["powerplay", "middle", "death"]
    phase_grp = df.groupby(
        df["innings_phase"].astype(str).str.lower().str.strip()
    )

    phase_pressure_means: Dict[str, dict] = {}
    for phase in phases:
        if phase in phase_grp.groups:
            grp = phase_grp.get_group(phase)
            phase_pressure_means[phase] = {
                "mean_bpi":  round(float(grp["bpi"].mean()),  2),
                "mean_bopi": round(float(grp["bopi"].mean()), 2),
                "count":     int(len(grp)),
            }
        else:
            phase_pressure_means[phase] = {"mean_bpi": None, "mean_bopi": None, "count": 0}

    summary = {
        "total_deliveries":         total,
        "high_pressure_bat_count":  high_bat,
        "high_pressure_bowl_count": high_bowl,
        "pressure_distribution":    pressure_distribution,
        "phase_pressure_means":     phase_pressure_means,
    }

    logger.info(
        "get_pressure_summary: total=%d | high-pressure bat=%d bowl=%d",
        total, high_bat, high_bowl,
    )
    return summary
