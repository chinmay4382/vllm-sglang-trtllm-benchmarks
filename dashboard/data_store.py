"""
Persistent data store for benchmark and evaluation results.

Uses Parquet for columnar efficiency; CSV for human-readable export.
"""

import logging
import os
from pathlib import Path
from typing import List, Optional

import pandas as pd

logger = logging.getLogger(__name__)

_PERF_COLUMNS = [
    "model_name", "concurrency", "dataset_name", "timestamp",
    "rps", "tps", "avg_ttft", "p50_ttft", "p95_ttft", "p99_ttft",
    "avg_itl", "p95_itl", "avg_latency", "p95_latency",
    "total_requests", "successful_requests", "failed_requests",
    "total_tokens", "total_time",
    "accuracy", "exact_match", "pass_at_k",
]

_EVAL_COLUMNS = [
    "model_name", "dataset_name", "timestamp",
    "accuracy", "exact_match", "pass_at_k",
    "num_samples", "num_correct", "backend",
]


class DataStore:
    """Append-only store for benchmark runs; writes Parquet + CSV side-by-side."""

    def __init__(self, path: str):
        self.path = Path(path)
        self._eval_path = self.path.parent / "evaluation_results.parquet"
        self.path.parent.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Performance data
    # ------------------------------------------------------------------

    def append_performance(self, rows: List[dict]) -> None:
        """Append new performance metric rows to the store."""
        new_df = pd.DataFrame(rows, columns=_PERF_COLUMNS)
        existing = self.load_performance()
        frames = [f for f in [existing, new_df] if not f.empty]
        combined = pd.concat(frames, ignore_index=True) if frames else new_df
        combined.to_parquet(self.path, index=False)
        combined.to_csv(self.path.with_suffix(".csv"), index=False)
        logger.info("Saved %d performance rows → %s", len(rows), self.path)

    def load_performance(self) -> pd.DataFrame:
        """Load all stored performance results."""
        if not self.path.exists():
            return pd.DataFrame(columns=_PERF_COLUMNS)
        try:
            return pd.read_parquet(self.path)
        except Exception as exc:  # noqa: BLE001
            logger.warning("Could not read performance parquet: %s", exc)
            return pd.DataFrame(columns=_PERF_COLUMNS)

    # ------------------------------------------------------------------
    # Evaluation data
    # ------------------------------------------------------------------

    def append_evaluations(self, rows: List[dict]) -> None:
        """Append new evaluation result rows to the store."""
        new_df = pd.DataFrame(rows, columns=_EVAL_COLUMNS)
        existing = self.load_evaluations()
        frames = [f for f in [existing, new_df] if not f.empty]
        combined = pd.concat(frames, ignore_index=True) if frames else new_df
        combined.to_parquet(self._eval_path, index=False)
        combined.to_csv(self._eval_path.with_suffix(".csv"), index=False)
        logger.info("Saved %d evaluation rows → %s", len(rows), self._eval_path)

    def load_evaluations(self) -> pd.DataFrame:
        """Load all stored evaluation results."""
        if not self._eval_path.exists():
            return pd.DataFrame(columns=_EVAL_COLUMNS)
        try:
            return pd.read_parquet(self._eval_path)
        except Exception as exc:  # noqa: BLE001
            logger.warning("Could not read evaluation parquet: %s", exc)
            return pd.DataFrame(columns=_EVAL_COLUMNS)

    def clear(self) -> None:
        """Delete all stored results (useful for fresh test runs)."""
        for p in [self.path, self._eval_path]:
            if p.exists():
                p.unlink()
            csv = p.with_suffix(".csv")
            if csv.exists():
                csv.unlink()
        logger.info("DataStore cleared.")
