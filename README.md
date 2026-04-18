# vLLM + GuideLLM Benchmarking Platform

End-to-end LLM benchmarking: serving performance (RPS, TPS, TTFT, ITL) + quality
evaluation (MMLU, GSM8K, HumanEval) with a Streamlit dashboard.

---

## Dashboard

### Overview — KPIs & Performance Charts
![Dashboard Overview](docs/images/dashboard_overview.png)

### Throughput / Latency Tradeoff
![Tradeoff Scatter](docs/images/dashboard_charts.png)

### Quality Metrics
![Quality Metrics](docs/images/dashboard_scatter.png)

> **Benchmark results above** — `Qwen/Qwen2.5-7B-Instruct` on RTX 5090, concurrency 1→50.
> Peak throughput: **2,907 TPS** at concurrency 50. Quality: MMLU **100%**, HumanEval **100%**, GSM8K **50%**.

---

## Project structure

```
.
├── benchmark/
│   ├── metrics.py          # RequestResult, BenchmarkMetrics, compute_metrics()
│   └── load_test.py        # Async concurrent load tester (streaming)
├── evaluation/
│   └── guidellm_runner.py  # GuideLLM wrapper + direct-API fallback evaluator
├── dashboard/
│   ├── app.py              # Streamlit dashboard
│   └── data_store.py       # Parquet/CSV persistence layer
├── docs/images/            # Dashboard screenshots
├── data/                   # Auto-created; stores .parquet + .csv results
├── main.py                 # Pipeline entry point (CLI)
└── requirements.txt
```

---

## Quick start

### 1. Install dependencies

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

### 2a. Try with demo data (no GPU required)

```bash
python main.py --demo
streamlit run dashboard/app.py
```

### 2b. Run against a live vLLM server

**Start vLLM:**

```bash
# Qwen2.5-7B (RTX 5090 — use FlashInfer backend)
vllm serve Qwen/Qwen2.5-7B-Instruct \
  --host 0.0.0.0 --port 8000 \
  --attention-backend FLASHINFER

# Llama single GPU
vllm serve meta-llama/Llama-3.1-8B-Instruct \
  --port 8000

# Multi-GPU (tensor parallelism)
vllm serve meta-llama/Llama-3.1-70B-Instruct \
  --tensor-parallel-size 4 \
  --port 8000
```

**Run benchmarks:**

```bash
# Full pipeline (load test + quality eval)
python main.py \
  --model Qwen/Qwen2.5-7B-Instruct \
  --base-url http://localhost:8000 \
  --concurrency 1,5,10,20,50 \
  --num-requests 50

# Load test only
python main.py --no-eval

# Evaluation only
python main.py --no-bench --eval-datasets mmlu,gsm8k

# Clear previous results and re-run
python main.py --clear
```

**Launch dashboard:**

```bash
streamlit run dashboard/app.py \
  --server.address 0.0.0.0 \
  --server.port 8501 \
  --server.fileWatcherType none \
  --server.enableCORS false \
  --server.enableXsrfProtection false
# → open http://localhost:8501
```

---

## Benchmark results — Qwen/Qwen2.5-7B-Instruct (RTX 5090)

| Concurrency | RPS | TPS | TTFT avg | Latency p95 |
|---|---|---|---|---|
| 1 | 0.41 | 101 | 25 ms | — |
| 5 | 2.00 | 496 | 43 ms | — |
| 10 | 3.70 | 917 | 53 ms | — |
| 20 | 5.93 | 1,474 | 85 ms | — |
| 50 | 11.71 | 2,907 | 160 ms | 4,256 ms |

**Quality evaluation:**

| Dataset | Metric | Score |
|---|---|---|
| MMLU | Accuracy | 100% |
| GSM8K | Exact Match | 50% |
| HumanEval | pass@1 | 100% |

---

## Metrics reference

| Metric | Formula | Unit |
|--------|---------|------|
| **RPS** | successful_requests / wall_time | req/s |
| **TPS** | total_output_tokens / wall_time | tok/s |
| **TTFT** | mean(first_token_ts − start_ts) | seconds |
| **ITL** | mean((end_ts − first_token_ts) / (tokens − 1)) | s/token |

All metrics are computed per concurrency level. Percentile variants (p50/p95/p99)
are stored for TTFT and latency.

---

## GuideLLM

When `guidellm` is installed it is used as the primary evaluator:

```bash
pip install guidellm
```

Without it, the platform falls back to a built-in direct-API evaluator using
curated sample questions from MMLU, GSM8K, and HumanEval — sufficient for
relative model comparisons.

---

## Dashboard sections

1. **Overview KPIs** — RPS, TPS, TTFT avg/p95, ITL, latency p95
2. **Performance Graphs** — TPS/TTFT vs concurrency
3. **Tradeoff Scatter** — TPS vs TTFT (bubble = concurrency level)
4. **Quality Metrics** — accuracy/exact-match/pass@1 per dataset
5. **Raw Data** — filterable tables + CSV download

---

## Exported files

After each run, results are written to `data/`:

- `benchmark_results.parquet` / `.csv` — performance metrics
- `evaluation_results.parquet` / `.csv` — quality evaluation scores
