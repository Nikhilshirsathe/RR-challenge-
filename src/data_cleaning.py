"""
data_cleaning.py - Validate and clean the enriched ball-by-ball dataframe.

Cleaning pipeline applied by clean_bbb():
  1. Remove super overs (innings >= 3)
  2. Remove deliveries from matches with no competition data (comp_id NaN)
  3. Validate overid is 1-20
  4. Validate innings is 1 or 2
  5. Remove deliveries where BOTH striker_id and bowler_id are 0 / NaN

Each step logs a row-count before/after report.
"""

import logging
from typing import Any

import pandas as pd

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _log_step(step: str, before: int, after: int) -> None:
    removed = before - after
    pct = (removed / before * 100) if before else 0.0
    logger.info(
        "[clean] %s | before=%d  after=%d  removed=%d (%.2f%%)",
        step, before, after, removed, pct,
    )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def clean_bbb(df: pd.DataFrame) -> pd.DataFrame:
    """Apply all cleaning / validation steps to the enriched ball-by-ball df.

    Parameters
    ----------
    df : pd.DataFrame
        Output of ``data_loader.load_enriched_bbb()``.

    Returns
    -------
    pd.DataFrame
        Cleaned copy (input is not mutated).
    """
    if df is None or df.empty:
        logger.warning("clean_bbb received empty dataframe — returning as-is.")
        return df

    out = df.copy()
    initial_rows = len(out)
    logger.info("=== DATA CLEANING PIPELINE START | initial rows: %d ===", initial_rows)

    # ------------------------------------------------------------------
    # Step 1: Remove super overs (innings 3 or 4)
    # ------------------------------------------------------------------
    before = len(out)
    if "innings" in out.columns:
        out = out[out["innings"].astype(int).isin([1, 2])].copy()
    else:
        logger.warning("Column 'innings' not found — step 1 skipped.")
    _log_step("Step 1 — remove super overs (innings >= 3)", before, len(out))

    # ------------------------------------------------------------------
    # Step 2: Remove matches with no competition data (comp_id is NaN)
    # ------------------------------------------------------------------
    before = len(out)
    if "comp_id" in out.columns:
        out = out[out["comp_id"].notna()].copy()
    else:
        logger.warning("Column 'comp_id' not found — step 2 skipped.")
    _log_step("Step 2 — remove matches with no competition data", before, len(out))

    # ------------------------------------------------------------------
    # Step 3: Validate overid is 1-20
    # ------------------------------------------------------------------
    before = len(out)
    if "overid" in out.columns:
        over_numeric = pd.to_numeric(out["overid"], errors="coerce")
        valid_over = over_numeric.between(1, 20, inclusive="both")
        invalid_overs = (~valid_over).sum()
        if invalid_overs > 0:
            logger.warning(
                "Step 3 — found %d deliveries with overid outside 1-20; removing.",
                invalid_overs,
            )
        out = out[valid_over].copy()
    else:
        logger.warning("Column 'overid' not found — step 3 skipped.")
    _log_step("Step 3 — validate overid in [1, 20]", before, len(out))

    # ------------------------------------------------------------------
    # Step 4: Validate innings is 1 or 2  (redundant after step 1,
    #         but kept as an explicit quality gate for any edge cases)
    # ------------------------------------------------------------------
    before = len(out)
    if "innings" in out.columns:
        valid_inn = out["innings"].astype(int).isin([1, 2])
        invalid_inn = (~valid_inn).sum()
        if invalid_inn > 0:
            logger.warning(
                "Step 4 — found %d deliveries with innings not in [1,2]; removing.",
                invalid_inn,
            )
        out = out[valid_inn].copy()
    else:
        logger.warning("Column 'innings' not found — step 4 skipped.")
    _log_step("Step 4 — validate innings in [1, 2]", before, len(out))

    # ------------------------------------------------------------------
    # Step 5: Remove deliveries where both striker_id AND bowler_id are 0 / NaN
    # ------------------------------------------------------------------
    before = len(out)
    has_striker = "striker_id" in out.columns
    has_bowler  = "bowler_id"  in out.columns
    if has_striker and has_bowler:
        striker_zero = (
            pd.to_numeric(out["striker_id"], errors="coerce").fillna(0) == 0
        )
        bowler_zero = (
            pd.to_numeric(out["bowler_id"], errors="coerce").fillna(0) == 0
        )
        both_zero = striker_zero & bowler_zero
        invalid_ids = both_zero.sum()
        if invalid_ids > 0:
            logger.warning(
                "Step 5 — found %d deliveries where both striker_id and "
                "bowler_id are 0/NaN; removing.",
                invalid_ids,
            )
        out = out[~both_zero].copy()
    else:
        missing = [c for c in ("striker_id", "bowler_id") if c not in out.columns]
        logger.warning(
            "Columns %s not found — step 5 skipped.", missing
        )
    _log_step("Step 5 — remove rows where striker_id=0 AND bowler_id=0", before, len(out))

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    total_removed = initial_rows - len(out)
    total_pct = (total_removed / initial_rows * 100) if initial_rows else 0.0
    logger.info(
        "=== DATA CLEANING PIPELINE COMPLETE | final rows: %d  "
        "total removed: %d (%.2f%%) ===",
        len(out), total_removed, total_pct,
    )

    # Reset index after all filters
    out = out.reset_index(drop=True)
    return out


def get_data_summary(
    df: pd.DataFrame,
    matches_df: pd.DataFrame,
    comps_df: pd.DataFrame,
) -> dict[str, Any]:
    """Return a high-level data quality / coverage summary dict.

    Parameters
    ----------
    df : pd.DataFrame
        Cleaned enriched ball-by-ball dataframe.
    matches_df : pd.DataFrame
        Raw matches dataframe.
    comps_df : pd.DataFrame
        Raw competitions dataframe.

    Returns
    -------
    dict with keys:
        total_matches, total_deliveries, total_legal_balls,
        pressure_deliveries (placeholder 0),
        total_players_batting, total_players_bowling,
        ipl_matches, t20i_matches, ipl_deliveries, t20i_deliveries
    """
    if df is None or df.empty:
        logger.warning("get_data_summary received empty dataframe.")
        return {}

    # ------------------------------------------------------------------
    # Identify IPL vs T20I via match_format column (or via comp join)
    # ------------------------------------------------------------------
    # Prefer the match_format column already on df (added by data_loader);
    # fall back to joining comps_df if needed.
    if "match_format" in df.columns:
        fmt_series = df["match_format"].str.upper()
    elif "comp_id" in df.columns and comps_df is not None:
        comp_fmt = comps_df[["comp_id", "match_format"]].drop_duplicates()
        merged = df[["match_id", "comp_id"]].drop_duplicates().merge(
            comp_fmt, on="comp_id", how="left"
        )
        fmt_map = dict(zip(merged["match_id"], merged["match_format"].str.upper()))
        fmt_series = df["match_id"].map(fmt_map).str.upper()
    else:
        fmt_series = pd.Series(["UNKNOWN"] * len(df), index=df.index)

    is_t20     = fmt_series == "T20"     # domestic T20 (IPL, BBL, etc.)
    is_t20i    = fmt_series == "T20I"    # international T20

    # ------------------------------------------------------------------
    # Match counts  (count unique match_ids per format in BBB, not in
    # matches_df, so super-overs and uncleaned matches are already excluded)
    # ------------------------------------------------------------------
    ipl_match_ids  = df.loc[is_t20,  "match_id"].unique()
    t20i_match_ids = df.loc[is_t20i, "match_id"].unique()

    # Legal balls
    legal_col = "is_legal_ball"
    if legal_col in df.columns:
        legal_mask = df[legal_col].astype(bool)
    else:
        legal_mask = pd.Series(True, index=df.index)

    summary: dict[str, Any] = {
        "total_matches":          df["match_id"].nunique(),
        "total_deliveries":       len(df),
        "total_legal_balls":      int(legal_mask.sum()),
        "pressure_deliveries":    0,                         # filled by pressure engine
        "total_players_batting":  int(df["striker_id"].nunique()),
        "total_players_bowling":  int(df["bowler_id"].nunique()),
        "ipl_matches":            int(len(ipl_match_ids)),
        "t20i_matches":           int(len(t20i_match_ids)),
        "ipl_deliveries":         int(is_t20.sum()),
        "t20i_deliveries":        int(is_t20i.sum()),
    }

    logger.info(
        "Data summary: %d matches | %d deliveries | %d legal balls | "
        "%d IPL matches | %d T20I matches",
        summary["total_matches"],
        summary["total_deliveries"],
        summary["total_legal_balls"],
        summary["ipl_matches"],
        summary["t20i_matches"],
    )
    return summary
