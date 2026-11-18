"""
match_state.py - Reconstruct the full match state BEFORE every delivery.

This is the most critical module. For each row in the ball-by-ball data,
compute what the match situation was BEFORE that ball was bowled.

Key principle: use only information from PREVIOUS deliveries in the same
match+innings. All "pre_*" columns are computed via cumsum + shift so that
only information available before the delivery is used — no look-ahead.

Columns added by compute_match_states()
----------------------------------------
pre_score               – runs scored in this innings before this ball
pre_wickets             – wickets fallen before this ball
pre_legal_balls         – legal balls bowled before this ball in this innings
pre_overs_complete      – floor(pre_legal_balls / 6)
pre_balls_in_current_over – pre_legal_balls % 6
pre_balls_remaining     – max(0, total_overs*6 - pre_legal_balls)
pre_current_rr          – pre_score / (pre_legal_balls/6), NaN if < 6 balls
target                  – innings-2 only: innings-1 total + 1
pre_runs_required       – innings-2 only: target - pre_score
pre_rrr                 – innings-2 only: pre_runs_required / (pre_balls_remaining/6)
pre_wickets_remaining   – 10 - pre_wickets
innings_phase           – 'powerplay' / 'middle' / 'death'
par_score_at_this_stage – estimated par at the current ball count
scoring_difficulty_1st_inn – par_score_at_this_stage - pre_score (innings 1 only)
"""

import logging
import random

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Par-score rates (runs per over) by phase
# ---------------------------------------------------------------------------
_PAR_RATES = {
    "powerplay": 7.5,   # overs 1-6   (balls 1-36)
    "middle":    8.0,   # overs 7-15  (balls 37-90)
    "death":     9.5,   # overs 16-20 (balls 91-120)
}

# Phase boundary in ball counts (cumulative legal balls in an innings)
_PP_BALLS    = 36    # 6 overs × 6 balls
_MIDDLE_BALLS = 90   # 15 overs × 6 balls

_PP_PAR    = _PAR_RATES["powerplay"] * 6          # 45 runs in 36 balls
_MID_PAR   = _PAR_RATES["middle"]    * 9          # 72 runs in 54 balls
# death par is per ball beyond ball 90

_BALLS_PER_OVER = 6
_DEFAULT_OVERS  = 20


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _par_score_for_ball_count(ball_count: "pd.Series") -> "pd.Series":
    """Vectorised progressive par score at a given cumulative legal-ball count.

    For ball count b:
        if b <= 36  → (b/6) * 7.5
        if b <= 90  → PP_par + ((b-36)/6) * 8.0
        else        → PP_par + MID_par + ((b-90)/6) * 9.5
    """
    b = ball_count.astype(float)

    pp_phase    = (b / _BALLS_PER_OVER) * _PAR_RATES["powerplay"]
    mid_phase   = _PP_PAR + ((b - _PP_BALLS) / _BALLS_PER_OVER) * _PAR_RATES["middle"]
    death_phase = _PP_PAR + _MID_PAR + ((b - _MIDDLE_BALLS) / _BALLS_PER_OVER) * _PAR_RATES["death"]

    return np.where(
        b <= _PP_BALLS,
        pp_phase,
        np.where(b <= _MIDDLE_BALLS, mid_phase, death_phase),
    )


def _innings_phase_from_overid(overid: "pd.Series") -> "pd.Series":
    return pd.cut(
        overid.astype(int),
        bins=[0, 6, 15, 20],
        labels=["powerplay", "middle", "death"],
        right=True,
    ).astype(str)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def compute_match_states(df: pd.DataFrame) -> pd.DataFrame:
    """Add pre-delivery match-state columns to the ball-by-ball dataframe.

    Parameters
    ----------
    df : pd.DataFrame
        Cleaned enriched ball-by-ball dataframe (output of
        ``data_cleaning.clean_bbb``).

    Returns
    -------
    pd.DataFrame
        Input dataframe with additional match-state columns appended.
        Sorted by match_id, innings, innings_ball_num.
    """
    if df is None or df.empty:
        logger.warning("compute_match_states received empty dataframe — returning as-is.")
        return df

    logger.info("compute_match_states: starting on %d rows", len(df))

    # ------------------------------------------------------------------
    # 1. Sort — mandatory before cumsum/shift
    # ------------------------------------------------------------------
    out = df.sort_values(
        ["match_id", "innings", "innings_ball_num"],
        ascending=True,
        kind="mergesort",        # stable sort
    ).copy()

    # ------------------------------------------------------------------
    # 2. Ensure required columns are numeric
    # ------------------------------------------------------------------
    for col in ("total_runs_on_ball", "is_legal_ball", "is_wicket",
                "first_inn_revised_overs", "second_inn_revised_overs",
                "innings", "overid"):
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce")

    # Fill revised-overs columns: NaN → default 20
    out["first_inn_revised_overs"]  = out["first_inn_revised_overs"].fillna(_DEFAULT_OVERS)
    out["second_inn_revised_overs"] = out["second_inn_revised_overs"].fillna(_DEFAULT_OVERS)

    # Ensure is_legal_ball and is_wicket are bool-compatible integers
    out["is_legal_ball"] = out["is_legal_ball"].fillna(0).astype(int)
    out["is_wicket"]     = out["is_wicket"].fillna(0).astype(int)

    # total_runs_on_ball may need computing if column absent
    if "total_runs_on_ball" not in out.columns:
        logger.warning(
            "'total_runs_on_ball' not found; recomputing from runs + extras1_runs + extras_2_runs."
        )
        for c in ("runs", "extras1_runs", "extras_2_runs"):
            out[c] = pd.to_numeric(out[c], errors="coerce").fillna(0)
        out["total_runs_on_ball"] = out["runs"] + out["extras1_runs"] + out["extras_2_runs"]

    # ------------------------------------------------------------------
    # 3. Groupby (match_id, innings) — build cumulative pre-delivery stats.
    #    CRITICAL: use groupby transform with cumsum().shift(1) so the shift
    #    is scoped WITHIN each group. The pattern grp[col].cumsum().shift(1)
    #    shifts across the ENTIRE dataframe and leaks the last row of one
    #    innings into the first row of the next innings.
    # ------------------------------------------------------------------
    grp = out.groupby(["match_id", "innings"], sort=False)

    out["pre_score"]       = grp["total_runs_on_ball"].transform(
        lambda x: x.cumsum().shift(1)
    ).fillna(0)
    out["pre_wickets"]     = grp["is_wicket"].transform(
        lambda x: x.cumsum().shift(1)
    ).fillna(0)
    out["pre_legal_balls"] = grp["is_legal_ball"].transform(
        lambda x: x.cumsum().shift(1)
    ).fillna(0)

    # ------------------------------------------------------------------
    # 4. Over-based metrics
    # ------------------------------------------------------------------
    out["pre_overs_complete"]       = (out["pre_legal_balls"] // _BALLS_PER_OVER).astype(int)
    out["pre_balls_in_current_over"] = (out["pre_legal_balls"] % _BALLS_PER_OVER).astype(int)

    # Total overs for this innings (from match data)
    total_overs = np.where(
        out["innings"].astype(int) == 1,
        out["first_inn_revised_overs"],
        out["second_inn_revised_overs"],
    )
    out["_total_overs"] = total_overs

    total_balls_in_innings = out["_total_overs"] * _BALLS_PER_OVER
    out["pre_balls_remaining"] = np.maximum(
        0,
        total_balls_in_innings - out["pre_legal_balls"]
    ).astype(int)

    # ------------------------------------------------------------------
    # 5. Current run rate (NaN when < 6 legal balls have been bowled)
    # ------------------------------------------------------------------
    legal_overs = out["pre_legal_balls"] / _BALLS_PER_OVER
    out["pre_current_rr"] = np.where(
        out["pre_legal_balls"] >= _BALLS_PER_OVER,
        out["pre_score"] / legal_overs,
        np.nan,
    )

    # ------------------------------------------------------------------
    # 6. Target (innings 2 only)
    #    target = innings-1 total for that match + 1
    # ------------------------------------------------------------------
    # Compute innings-1 total for every match from the cumulative score
    # at the LAST ball of innings 1.
    inn1_totals = (
        out[out["innings"] == 1]
        .groupby("match_id")["total_runs_on_ball"]
        .sum()
        .rename("inn1_total")
        .reset_index()
    )
    inn1_totals["target"] = inn1_totals["inn1_total"] + 1

    out = out.merge(inn1_totals[["match_id", "target"]], on="match_id", how="left")

    # For innings 1 rows, target is NaN
    out.loc[out["innings"] == 1, "target"] = np.nan

    # ------------------------------------------------------------------
    # 7. Runs required & required run rate (innings 2 only)
    # ------------------------------------------------------------------
    out["pre_runs_required"] = np.where(
        out["innings"] == 2,
        out["target"] - out["pre_score"],
        np.nan,
    )

    with np.errstate(divide="ignore", invalid="ignore"):
        balls_rem_overs = out["pre_balls_remaining"] / _BALLS_PER_OVER
        rrr_raw = np.where(
            (out["innings"] == 2) & (out["pre_balls_remaining"] > 0),
            out["pre_runs_required"] / balls_rem_overs,
            np.where(out["innings"] == 2, np.inf, np.nan),
        )
    out["pre_rrr"] = rrr_raw

    # ------------------------------------------------------------------
    # 8. Wickets remaining
    # ------------------------------------------------------------------
    out["pre_wickets_remaining"] = (10 - out["pre_wickets"]).clip(lower=0)

    # ------------------------------------------------------------------
    # 9. Innings phase
    # ------------------------------------------------------------------
    if "innings_phase" not in out.columns:
        out["innings_phase"] = _innings_phase_from_overid(out["overid"])

    # ------------------------------------------------------------------
    # 10. Par score (innings 1) and scoring difficulty
    # ------------------------------------------------------------------
    out["par_score_at_this_stage"] = _par_score_for_ball_count(out["pre_legal_balls"])

    # scoring_difficulty: positive → batting team is behind par (harder situation)
    out["scoring_difficulty_1st_inn"] = np.where(
        out["innings"] == 1,
        out["par_score_at_this_stage"] - out["pre_score"],
        np.nan,
    )

    # ------------------------------------------------------------------
    # 11. Tidy up helper column
    # ------------------------------------------------------------------
    out = out.drop(columns=["_total_overs"], errors="ignore")

    # Cast integer columns
    for col in ("pre_score", "pre_wickets", "pre_legal_balls",
                "pre_overs_complete", "pre_balls_in_current_over",
                "pre_balls_remaining", "pre_wickets_remaining"):
        out[col] = out[col].fillna(0).astype(int)

    logger.info("compute_match_states: complete. %d rows, %d cols", *out.shape)
    return out


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

def validate_match_states(df: pd.DataFrame, n: int = 3) -> None:
    """Spot-check n random matches and print pre-delivery state summaries.

    Uses print (not logger) so results are always visible in interactive use.
    """
    if df is None or df.empty:
        print("validate_match_states: dataframe is empty.")
        return

    required = {
        "match_id", "innings", "innings_ball_num",
        "pre_score", "pre_wickets", "pre_legal_balls",
        "pre_balls_remaining", "target", "pre_rrr",
    }
    missing = required - set(df.columns)
    if missing:
        print(f"validate_match_states: missing columns {missing} — run compute_match_states first.")
        return

    all_match_ids = df["match_id"].unique().tolist()
    if len(all_match_ids) < n:
        n = len(all_match_ids)

    sample_ids = random.sample(all_match_ids, n)
    sep = "=" * 72

    for mid in sample_ids:
        mdf = df[df["match_id"] == mid].copy()
        print(f"\n{sep}")
        print(f"MATCH ID: {mid}")

        for inn in sorted(mdf["innings"].unique()):
            idf = mdf[mdf["innings"] == inn].sort_values("innings_ball_num")
            print(f"\n  Innings {inn}  ({len(idf)} deliveries)")
            print(f"  {'ball':>6}  {'pre_score':>9}  {'pre_wkt':>7}  {'pre_lb':>6}  "
                  f"{'balls_rem':>9}  {'target':>7}  {'pre_rrr':>8}")
            print(f"  {'-'*65}")

            # Print first 3 and last 3 deliveries
            show_rows = pd.concat([idf.head(3), idf.tail(3)]).drop_duplicates()
            for _, row in show_rows.iterrows():
                tgt   = f"{row['target']:.0f}" if pd.notna(row["target"]) else "  -"
                rrr   = f"{row['pre_rrr']:.2f}" if pd.notna(row["pre_rrr"]) and not np.isinf(row["pre_rrr"]) else ("inf" if np.isinf(row.get("pre_rrr", 0)) else "  -")
                print(
                    f"  {row['innings_ball_num']:>6}  "
                    f"{row['pre_score']:>9}  "
                    f"{row['pre_wickets']:>7}  "
                    f"{row['pre_legal_balls']:>6}  "
                    f"{row['pre_balls_remaining']:>9}  "
                    f"{tgt:>7}  "
                    f"{rrr:>8}"
                )

            # Basic sanity assertions (non-fatal)
            first_row = idf.iloc[0]
            last_row  = idf.iloc[-1]

            issues: list[str] = []
            if int(first_row["pre_score"]) != 0:
                issues.append(f"pre_score of first ball is {first_row['pre_score']} (expected 0)")
            if int(first_row["pre_wickets"]) != 0:
                issues.append(f"pre_wickets of first ball is {first_row['pre_wickets']} (expected 0)")
            if int(first_row["pre_legal_balls"]) != 0:
                issues.append(f"pre_legal_balls of first ball is {first_row['pre_legal_balls']} (expected 0)")

            # Check monotonicity of pre_score within innings
            scores = idf["pre_score"].values
            if not all(scores[i] <= scores[i + 1] for i in range(len(scores) - 1)):
                issues.append("pre_score is not monotonically non-decreasing!")

            if issues:
                print(f"\n  ⚠ ISSUES:")
                for issue in issues:
                    print(f"    • {issue}")
            else:
                print(f"\n  ✓ Sanity checks passed for innings {inn}.")

    print(f"\n{sep}\nValidation complete for {n} matches.\n")
