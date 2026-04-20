# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

End-to-end LLM benchmarking platform that measures serving performance (RPS, TPS, TTFT, ITL) and quality evaluation (MMLU, GSM8K, HumanEval) for vLLM and SGLang inference servers, with a Streamlit visualization dashboard.

## Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
# Optional GPU serving: pip install vllm>=0.5.0
# Optional quality eval: pip install guidellm>=0.1.0
```

## Common Commands

**Run full benchmark pipeline (requires running inference server):**
```bash
python main.py \
  --model Qwen/Qwen2.5-7B-Instruct \
  --base-url http://localhost:8000 \
  --concurrency 1,5,10,20,50 \
  --num-requests 50
```

**Demo mode (no server required — generates synthetic data):**
```bash
python main.py --demo
```

**Load test only / eval only:**
```bash
python main.py --no-eval
python main.py --no-bench --eval-datasets mmlu,gsm8k
```

**Clear previous results then re-run:**
```bash
python main.py --clear
```

**Launch Streamlit dashboard:**
```bash
streamlit run dashboard/app.py \
  --server.address 0.0.0.0 \
  --server.port 8501 \
  --server.fileWatcherType none \
  --server.enableCORS false \
  --server.enableXsrfProtection false
```

**Start vLLM server (examples):**
```bash
vllm serve Qwen/Qwen2.5-7B-Instruct --host 0.0.0.0 --port 8000 --attention-backend FLASHINFER
vllm serve meta-llama/Llama-3.1-70B-Instruct --tensor-parallel-size 4 --port 8000
```

## Architecture

```
main.py                  # CLI entry point; orchestrates full pipeline
benchmark/
  load_test.py           # Async concurrent load tester (asyncio + semaphore)
  metrics.py             # RequestResult, BenchmarkMetrics, compute_metrics()
evaluation/
  guidellm_runner.py     # GuideLLM wrapper with DirectEvaluator fallback
dashboard/
  app.py                 # Streamlit UI (KPIs, line charts, scatter, bar charts)
  data_store.py          # Append-only Parquet+CSV persistence
data/                    # Auto-created; stores .parquet + .csv results (gitignored)
```

### Data Flow

1. `main.py` parses CLI args, creates `DataStore`
2. **Load test**: `LoadTester` sweeps concurrency levels, streams OpenAI-compatible chat/completions, captures per-request timing (`RequestResult`), aggregates into `BenchmarkMetrics` via pandas quantiles, persists to DataStore
3. **Evaluation**: `GuideLLMRunner` tries GuideLLM library first; falls back to `DirectEvaluator` (curated sample questions) if unavailable; persists `EvalResult` to DataStore
4. **Dashboard**: Streamlit loads cached Parquet, filters by model/dataset, renders visualizations

### Key Metrics

- **TTFT**: `first_token_time - start_time` (captured from streaming response)
- **ITL**: `(end_time - first_token_time) / (total_tokens - 1)` (inter-token latency during generation)
- **RPS/TPS**: `successful_requests / wall_time`, `total_output_tokens / wall_time`
- Percentiles (p50/p95/p99) computed via pandas quantiles on per-request data

### Storage Schema

Performance data stored with columns: `model_name, concurrency, dataset_name, timestamp, rps, tps, avg_ttft, p50_ttft, p95_ttft, p99_ttft, avg_itl, p95_itl, avg_latency, p95_latency, total_requests, successful_requests, failed_requests, total_tokens, total_time, accuracy, exact_match, pass_at_k`

Evaluation data stored with columns: `model_name, dataset_name, timestamp, accuracy, exact_match, pass_at_k, num_samples, num_correct, backend`

## Testing

`pytest` and `pytest-asyncio` are in requirements.txt but no test suite exists yet. Async tests should use `--asyncio-mode=auto`.
