"""Notebook-friendly U1 review launcher.

Run after paid cells are stored:
    uv run utrack score u1
    uv run utrack report u1

Then execute this file with `uv run python notebooks/u1_review.py` or open it
interactively in an IDE supporting # %% cells.
"""
# %%
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
scores_path = ROOT / "artifacts/u1/u1_scores.parquet"
report_path = ROOT / "artifacts/u1/u1_report.md"

if not scores_path.exists():
    raise SystemExit("No scored U1 data yet. Run: uv run utrack score u1")

scores = pd.read_parquet(scores_path)
print(scores[["benchmark_id", "condition_id", "repeat", "scaled_crps_a3", "utility_b1_a3", "utility_b2_a3", "cost_usd"]].sort_values(["benchmark_id", "condition_id", "repeat"]).to_string(index=False))
print(f"\nTotal scored-cell cost: ${scores.cost_usd.sum():.6f}")
print(f"Review report: {report_path}")

# %%
# Primary interpretation:
# - headline score: scaled_crps_a3 (lower is better)
# - headline utility: utility_b1_a3 (positive means context helped)
# - use plots under artifacts/u1/review/ to inspect full 10–90% bands and medians.
# Do not select trajectories based on observed future values.
