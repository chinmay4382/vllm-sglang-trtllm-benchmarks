"""
Entry point for the LLM benchmarking pipeline.

Usage examples:
  # Full run (load test + eval) against a local vLLM server
  python main.py

  # Load test only
  python main.py --no-eval

  # Evaluation only (skips load test)
  python main.py --no-bench

  # Custom model and server
  python main.py --model meta-llama/Llama-3.1-8B-Instruct --base-url http://localhost:8000

  # Use demo data (no server required)
  python main.py --demo
"""

import argparse
import logging
import sys
import time
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("main")


# ---------------------------------------------------------------------------
# Demo data generation (no live server required)
# ---------------------------------------------------------------------------


def generate_demo_data(store, model: str) -> None:
    """Insert synthetic benchmark results so the dashboard works without a server."""
    import random
    import time as _time

    from benchmark.metrics import BenchmarkMetrics

    logger.info("Generating demo performance data …")
    concurrency_levels = [1, 5, 10, 20, 50]
    datasets = ["sharegpt", "code-alpaca"]
    metrics_list = []

    rng = random.Random(42)
    for ds in datasets:
        for c in concurrency_levels:
            # Realistic degradation curves
            base_tps = 350 * (1 - 0.007 * c) + rng.uniform(-10, 10)
            base_ttft = 0.04 + 0.003 * c + rng.uniform(-0.005, 0.005)
            m = BenchmarkMetrics(
                model_name=model,
                concurrency=c,
                dataset_name=ds,
                timestamp=_time.time() - rng.uniform(0, 3600),
            )
            m.rps = max(0.1, c * 1.8 - 0.03 * c**2 + rng.uniform(-0.5, 0.5))
            m.tps = max(10.0, base_tps)
            m.avg_ttft = max(0.01, base_ttft)
            m.p50_ttft = m.avg_ttft * 0.9
            m.p95_ttft = m.avg_ttft * 1.8
            m.p99_ttft = m.avg_ttft * 2.5
            m.avg_itl = 0.003 + 0.0001 * c
            m.p95_itl = m.avg_itl * 2.0
            m.avg_latency = m.avg_ttft + m.avg_itl * 200
            m.p95_latency = m.avg_latency * 1.7
            m.total_requests = 50
            m.successful_requests = rng.randint(46, 50)
            m.failed_requests = 50 - m.successful_requests
            m.total_tokens = int(m.tps * 30)
            m.total_time = 30.0
            metrics_list.append(m)

    store.append_performance([m.to_dict() for m in metrics_list])

    logger.info("Generating demo evaluation data …")
    from evaluation.guidellm_runner import EvalResult

    eval_results = [
        EvalResult(model_name=model, dataset_name="MMLU", accuracy=0.72, exact_match=0.72, num_samples=10, num_correct=7, backend="demo"),
        EvalResult(model_name=model, dataset_name="GSM8K", accuracy=0.625, exact_match=0.625, num_samples=8, num_correct=5, backend="demo"),
        EvalResult(model_name=model, dataset_name="HumanEval", accuracy=0.60, pass_at_k=0.60, num_samples=5, num_correct=3, backend="demo"),
    ]
    store.append_evaluations([r.to_dict() for r in eval_results])
    logger.info("Demo data written.")


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------


def run_benchmarks(args: argparse.Namespace) -> None:
    from benchmark.load_test import run_load_test
    from dashboard.data_store import DataStore

    store = DataStore(str(Path("data") / "benchmark_results.parquet"))

    if args.demo:
        generate_demo_data(store, args.model)
        logger.info("Demo data ready — launch the dashboard with: streamlit run dashboard/app.py")
        return

    concurrency_levels = [int(c) for c in args.concurrency.split(",")]

    # ---- Load / performance benchmarks ----
    if not args.no_bench:
        logger.info(
            "Starting load test | model=%s url=%s concurrency=%s requests=%d",
            args.model,
            args.base_url,
            concurrency_levels,
            args.num_requests,
        )
        try:
            metrics_list = run_load_test(
                base_url=args.base_url,
                model=args.model,
                concurrency_levels=concurrency_levels,
                num_requests=args.num_requests,
                max_tokens=args.max_tokens,
                dataset_name=args.dataset,
            )
            store.append_performance([m.to_dict() for m in metrics_list])
            logger.info("Load test complete. %d concurrency levels recorded.", len(metrics_list))
        except Exception as exc:
            logger.error("Load test failed: %s", exc)
            logger.info("Tip: ensure vLLM is running at %s", args.base_url)

    # ---- Quality evaluations ----
    if not args.no_eval:
        from evaluation.guidellm_runner import GuideLLMRunner

        logger.info("Starting quality evaluations …")
        runner = GuideLLMRunner(base_url=args.base_url, model=args.model)
        datasets = args.eval_datasets.split(",") if args.eval_datasets else None

        try:
            eval_results = runner.run_all(datasets=datasets)
            store.append_evaluations([r.to_dict() for r in eval_results])
            logger.info("Evaluations complete. %d datasets evaluated.", len(eval_results))
        except Exception as exc:
            logger.error("Evaluation run failed: %s", exc)

    logger.info("All done. Launch the dashboard: streamlit run dashboard/app.py")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="vLLM + GuideLLM benchmarking pipeline",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--base-url", default="http://localhost:8000", help="vLLM server base URL")
    p.add_argument(
        "--model",
        default="meta-llama/Llama-3.1-8B-Instruct",
        help="Model name as registered in vLLM",
    )
    p.add_argument(
        "--concurrency",
        default="1,5,10,20,50",
        help="Comma-separated list of concurrency levels",
    )
    p.add_argument("--num-requests", type=int, default=50, help="Requests per concurrency level")
    p.add_argument("--max-tokens", type=int, default=256, help="Max output tokens per request")
    p.add_argument("--dataset", default="default", help="Label for the prompt dataset used")
    p.add_argument(
        "--eval-datasets",
        default="mmlu,gsm8k,humaneval",
        help="Comma-separated evaluation datasets",
    )
    p.add_argument("--no-bench", action="store_true", help="Skip load / performance benchmarks")
    p.add_argument("--no-eval", action="store_true", help="Skip quality evaluations")
    p.add_argument("--demo", action="store_true", help="Populate with synthetic demo data (no server required)")
    p.add_argument("--clear", action="store_true", help="Clear existing stored results before running")
    return p


if __name__ == "__main__":
    parser = build_parser()
    args = parser.parse_args()

    if args.clear:
        from dashboard.data_store import DataStore
        DataStore(str(Path("data") / "benchmark_results.parquet")).clear()
        logger.info("Previous results cleared.")

    run_benchmarks(args)
