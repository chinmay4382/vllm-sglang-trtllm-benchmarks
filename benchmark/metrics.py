"""
Core metrics computation for LLM benchmarking.

Handles per-request timing data and aggregates into RPS, TPS, TTFT, ITL.
"""

import logging
import time
from dataclasses import dataclass, field
from typing import List, Optional

import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class RequestResult:
    """Timing and token data captured for a single inference request."""

    request_id: int
    prompt: str
    start_time: float
    first_token_time: Optional[float] = None
    end_time: Optional[float] = None
    total_tokens: int = 0
    prompt_tokens: int = 0
    error: Optional[str] = None

    @property
    def success(self) -> bool:
        return self.error is None and self.end_time is not None

    @property
    def latency(self) -> Optional[float]:
        if self.end_time and self.start_time:
            return self.end_time - self.start_time
        return None

    @property
    def ttft(self) -> Optional[float]:
        """Time to first token in seconds."""
        if self.first_token_time and self.start_time:
            return self.first_token_time - self.start_time
        return None

    @property
    def itl(self) -> Optional[float]:
        """Inter-token latency in seconds/token (excluding TTFT)."""
        if self.ttft is not None and self.latency is not None and self.total_tokens > 1:
            generation_time = self.latency - self.ttft
            return generation_time / (self.total_tokens - 1)
        return None


@dataclass
class BenchmarkMetrics:
    """Aggregated metrics for a complete benchmark run."""

    model_name: str
    concurrency: int
    dataset_name: str
    timestamp: float = field(default_factory=time.time)

    # Core performance metrics
    rps: float = 0.0
    tps: float = 0.0
    avg_ttft: float = 0.0
    p50_ttft: float = 0.0
    p95_ttft: float = 0.0
    p99_ttft: float = 0.0
    avg_itl: float = 0.0
    p95_itl: float = 0.0
    avg_latency: float = 0.0
    p95_latency: float = 0.0

    # Throughput stats
    total_requests: int = 0
    successful_requests: int = 0
    failed_requests: int = 0
    total_tokens: int = 0
    total_time: float = 0.0

    # Optional quality metrics (from GuideLLM)
    accuracy: Optional[float] = None
    exact_match: Optional[float] = None
    pass_at_k: Optional[float] = None

    def to_dict(self) -> dict:
        return {
            "model_name": self.model_name,
            "concurrency": self.concurrency,
            "dataset_name": self.dataset_name,
            "timestamp": self.timestamp,
            "rps": round(self.rps, 4),
            "tps": round(self.tps, 4),
            "avg_ttft": round(self.avg_ttft, 4),
            "p50_ttft": round(self.p50_ttft, 4),
            "p95_ttft": round(self.p95_ttft, 4),
            "p99_ttft": round(self.p99_ttft, 4),
            "avg_itl": round(self.avg_itl, 4),
            "p95_itl": round(self.p95_itl, 4),
            "avg_latency": round(self.avg_latency, 4),
            "p95_latency": round(self.p95_latency, 4),
            "total_requests": self.total_requests,
            "successful_requests": self.successful_requests,
            "failed_requests": self.failed_requests,
            "total_tokens": self.total_tokens,
            "total_time": round(self.total_time, 4),
            "accuracy": self.accuracy,
            "exact_match": self.exact_match,
            "pass_at_k": self.pass_at_k,
        }


def compute_metrics(
    results: List[RequestResult],
    model_name: str,
    concurrency: int,
    dataset_name: str,
    wall_time: float,
) -> BenchmarkMetrics:
    """
    Aggregate a list of RequestResult objects into a BenchmarkMetrics summary.

    Args:
        results: Per-request timing results.
        model_name: Name of the model under test.
        concurrency: Number of concurrent workers used.
        dataset_name: Name of the prompt dataset.
        wall_time: Elapsed wall-clock time for the entire run.

    Returns:
        Populated BenchmarkMetrics instance.
    """
    metrics = BenchmarkMetrics(
        model_name=model_name,
        concurrency=concurrency,
        dataset_name=dataset_name,
    )

    metrics.total_requests = len(results)
    successful = [r for r in results if r.success]
    metrics.successful_requests = len(successful)
    metrics.failed_requests = metrics.total_requests - metrics.successful_requests
    metrics.total_time = wall_time

    if not successful:
        logger.warning("No successful requests — cannot compute meaningful metrics.")
        return metrics

    metrics.total_tokens = sum(r.total_tokens for r in successful)
    metrics.rps = metrics.successful_requests / wall_time if wall_time > 0 else 0.0
    metrics.tps = metrics.total_tokens / wall_time if wall_time > 0 else 0.0

    ttfts = [r.ttft for r in successful if r.ttft is not None]
    if ttfts:
        metrics.avg_ttft = float(pd.Series(ttfts).mean())
        metrics.p50_ttft = float(pd.Series(ttfts).quantile(0.50))
        metrics.p95_ttft = float(pd.Series(ttfts).quantile(0.95))
        metrics.p99_ttft = float(pd.Series(ttfts).quantile(0.99))

    itls = [r.itl for r in successful if r.itl is not None]
    if itls:
        metrics.avg_itl = float(pd.Series(itls).mean())
        metrics.p95_itl = float(pd.Series(itls).quantile(0.95))

    latencies = [r.latency for r in successful if r.latency is not None]
    if latencies:
        metrics.avg_latency = float(pd.Series(latencies).mean())
        metrics.p95_latency = float(pd.Series(latencies).quantile(0.95))

    logger.info(
        "Run summary | model=%s concurrency=%d RPS=%.2f TPS=%.2f "
        "TTFT_avg=%.3fs ITL_avg=%.4fs",
        model_name,
        concurrency,
        metrics.rps,
        metrics.tps,
        metrics.avg_ttft,
        metrics.avg_itl,
    )
    return metrics


def results_to_dataframe(metrics_list: List[BenchmarkMetrics]) -> pd.DataFrame:
    """Convert a list of BenchmarkMetrics into a tidy DataFrame for storage/display."""
    return pd.DataFrame([m.to_dict() for m in metrics_list])
