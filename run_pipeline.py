"""
run_pipeline.py – Execute the full SUPRR Selector ranking pipeline and save outputs.

Usage:
    python run_pipeline.py

Writes:
    C:/Users/Nikhi/suprr_selector/outputs/batter_rankings.csv
    C:/Users/Nikhi/suprr_selector/outputs/bowler_rankings.csv
    C:/Users/Nikhi/suprr_selector/outputs/analysis_summary.json
"""

import logging
import os
import sys
import time
import traceback
from pathlib import Path

# ---------------------------------------------------------------------------
# Path setup – must happen before any src imports
# ---------------------------------------------------------------------------
SRC_DIR    = Path(__file__).parent / "src"
OUTPUT_DIR = Path(__file__).parent / "outputs"
sys.path.insert(0, str(SRC_DIR))

# ---------------------------------------------------------------------------
# Logging setup
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)-35s  %(message)s",
    datefmt="%H:%M:%S",
    handlers=[
        logging.StreamHandler(sys.stdout),
    ],
)
logger = logging.getLogger("run_pipeline")

# ---------------------------------------------------------------------------
# Pandas display options
# ---------------------------------------------------------------------------
import pandas as pd
import numpy as np

pd.set_option("display.max_columns", 20)
pd.set_option("display.width", 160)
pd.set_option("display.float_format", "{:.3f}".format)
pd.set_option("display.max_rows", 20)

# ---------------------------------------------------------------------------
# Main execution
# ---------------------------------------------------------------------------

def print_separator(title: str = "") -> None:
    sep = "=" * 72
    if title:
        print(f"\n{sep}")
        print(f"  {title}")
        print(sep)
    else:
        print(sep)


def run() -> None:
    print_separator("SUPRR SELECTOR – Full Pipeline Run")
    print(f"  SRC directory : {SRC_DIR}")
    print(f"  Output directory: {OUTPUT_DIR}")
    print_separator()

    overall_start = time.perf_counter()

    # ------------------------------------------------------------------ #
    # Import ranking_engine
    # ------------------------------------------------------------------ #
    try:
        import ranking_engine as re_mod
        logger.info("ranking_engine imported successfully.")
    except Exception as exc:
        logger.error("FATAL: could not import ranking_engine: %s", exc)
        traceback.print_exc()
        sys.exit(1)

    # ------------------------------------------------------------------ #
    # Run the pipeline
    # ------------------------------------------------------------------ #
    logger.info("Starting run_full_pipeline() ...")
    try:
        results = re_mod.run_full_pipeline()
    except Exception as exc:
        logger.error("Pipeline raised an unexpected exception: %s", exc)
        traceback.print_exc()
        sys.exit(1)

    elapsed = time.perf_counter() - overall_start
    logger.info("Pipeline completed in %.2f seconds.", elapsed)

    # ------------------------------------------------------------------ #
    # Save outputs
    # ------------------------------------------------------------------ #
    print_separator("Saving outputs")
    try:
        saved = re_mod.save_outputs(results, str(OUTPUT_DIR))
        for name, path in saved.items():
            logger.info("  ✓ %s  →  %s", name, path)
    except Exception as exc:
        logger.error("save_outputs failed: %s", exc)
        traceback.print_exc()

    # ------------------------------------------------------------------ #
    # Print data summary
    # ------------------------------------------------------------------ #
    print_separator("Data Summary")
    ds = results.get("data_summary", {})
    for k, v in ds.items():
        print(f"  {k:<35}: {v}")

    # ------------------------------------------------------------------ #
    # Print Top 10 Batter Rankings
    # ------------------------------------------------------------------ #
    print_separator("Top 10 Batter Rankings")
    batter_rankings: pd.DataFrame = results.get("batter_rankings", pd.DataFrame())
    if batter_rankings.empty:
        print("  [No batter rankings generated – see log for errors]")
    else:
        display_cols = [c for c in [
            "rank", "player_name", "main_team_name",
            "adjusted_score", "pressure_runs", "pressure_strike_rate",
            "sr_improvement", "pressure_balls", "pressure_matches",
            "format_coverage",
        ] if c in batter_rankings.columns]
        print(batter_rankings[display_cols].to_string(index=False))

    # ------------------------------------------------------------------ #
    # Print Top 10 Bowler Rankings
    # ------------------------------------------------------------------ #
    print_separator("Top 10 Bowler Rankings")
    bowler_rankings: pd.DataFrame = results.get("bowler_rankings", pd.DataFrame())
    if bowler_rankings.empty:
        print("  [No bowler rankings generated – see log for errors]")
    else:
        display_cols = [c for c in [
            "rank", "player_name", "main_team_name",
            "adjusted_score", "pressure_economy", "economy_improvement",
            "pressure_wickets", "wickets_above_expected",
            "pressure_balls", "pressure_matches",
            "format_coverage",
        ] if c in bowler_rankings.columns]
        print(bowler_rankings[display_cols].to_string(index=False))

    # ------------------------------------------------------------------ #
    # Verify output files exist and print their contents
    # ------------------------------------------------------------------ #
    print_separator("Output File Verification")
    output_files = {
        "batter_rankings.csv":    OUTPUT_DIR / "batter_rankings.csv",
        "bowler_rankings.csv":    OUTPUT_DIR / "bowler_rankings.csv",
        "analysis_summary.json":  OUTPUT_DIR / "analysis_summary.json",
    }
    all_ok = True
    for fname, fpath in output_files.items():
        if fpath.exists():
            size_kb = fpath.stat().st_size / 1024
            print(f"  ✓ {fname:<35}  ({size_kb:.1f} KB)  → {fpath}")
        else:
            print(f"  ✗ {fname:<35}  MISSING")
            all_ok = False

    if all_ok:
        print("\n  All output files verified successfully.")
    else:
        print("\n  WARNING: some output files are missing (see above).")

    # ------------------------------------------------------------------ #
    # Print analysis_summary.json contents
    # ------------------------------------------------------------------ #
    summary_path = OUTPUT_DIR / "analysis_summary.json"
    if summary_path.exists():
        import json
        print_separator("analysis_summary.json")
        with open(summary_path, encoding="utf-8") as fh:
            summary_data = json.load(fh)
        print(json.dumps(summary_data, indent=2))

    print_separator()
    print(f"  Pipeline finished in {elapsed:.1f}s")
    print_separator()


if __name__ == "__main__":
    run()
