# vLLM + SGLang + TRT-LLM Benchmarking Platform

End-to-end LLM benchmarking: serving performance (RPS, TPS, TTFT, ITL) + quality
evaluation (MMLU, GSM8K, HumanEval) with a Streamlit dashboard. Supports **vLLM**,
**SGLang**, and **TensorRT-LLM** — any OpenAI-compatible endpoint works.

---

## Dashboard

### vLLM — Overview & KPIs
![vLLM Dashboard Overview](docs/images/dashboard_overview.png)

### vLLM — Throughput / Latency Tradeoff
![vLLM Tradeoff Scatter](docs/images/dashboard_charts.png)

### vLLM — Quality Metrics
![vLLM Quality Metrics](docs/images/dashboard_scatter.png)

> **vLLM results** — `Qwen/Qwen2.5-7B-Instruct` on RTX 5090.
> GPU ceiling: **~5,650 TPS** (queue-saturated). Practical saturation: **~2,917 TPS** at c=64. Quality: MMLU **100%**, HumanEval **100%**, GSM8K **50%**.

---

### SGLang — Overview & KPIs
![SGLang Dashboard Overview](docs/images/sglang_dashboard_overview.png)

### SGLang — Throughput / Latency Tradeoff
![SGLang Tradeoff Scatter](docs/images/sglang_dashboard_charts.png)

### SGLang — Quality Metrics
![SGLang Quality Metrics](docs/images/sglang_dashboard_scatter.png)

> **SGLang results** — `Qwen/Qwen2.5-7B-Instruct` on RTX 5090.
> GPU ceiling: **~5,640 TPS** (queue-saturated). Practical saturation: **~2,887 TPS** at c=64. Quality: MMLU **100%**, HumanEval **100%**, GSM8K **62.5%**.

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

### 2c. Run against a live SGLang server

**Start SGLang:**

```bash
python -m sglang.launch_server \
  --model-path Qwen/Qwen2.5-7B-Instruct \
  --host 0.0.0.0 --port 8000
```

### 2d. Run against TensorRT-LLM

**Prerequisites (one-time system setup):**

```bash
# 1. Install TRT-LLM and dependencies
pip install tensorrt-llm --extra-index-url https://pypi.nvidia.com
pip install hf_transfer

# 2. Install OpenMPI (required by mpi4py)
apt-get install -y libopenmpi-dev openmpi-bin

# 3. Install CUDA 13 cuBLAS (TRT-LLM 1.2.0 is built against CUDA 13 cuBLAS,
#    even if your CUDA toolkit is 12.x)
apt-get install -y libcublas-13-0

# 4. Restore the /usr/local/cuda symlink to your toolkit (apt may redirect it)
update-alternatives --set cuda /usr/local/cuda-12.8
```

**Create an extra options file to disable FlashInfer sampling** (required on
Blackwell / SM 12.x with CUDA < 12.9, since nvcc 12.8 cannot compile
`compute_120f` kernels):

```bash
cat > /tmp/trtllm_extra.yaml << 'EOF'
disable_flashinfer_sampling: true
EOF
```

**Start TRT-LLM server:**

```bash
export LD_LIBRARY_PATH=/usr/local/cuda-13.0/targets/x86_64-linux/lib:$LD_LIBRARY_PATH

trtllm-serve serve Qwen/Qwen2.5-7B-Instruct \
  --host 0.0.0.0 \
  --port 8000 \
  --max_batch_size 64 \
  --max_num_tokens 4096 \
  --extra_llm_api_options /tmp/trtllm_extra.yaml
```

On first run, TRT-LLM downloads the model, loads it via the PyTorch backend,
and runs CUDA graph warmup for batch sizes 1–64. Wait for:
`INFO: Application startup complete.`

> **Note:** Verify the model name TRT-LLM registered before running benchmarks:
> `curl http://localhost:8000/v1/models`

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

Three backends benchmarked: vLLM (`--attention-backend FLASHINFER`), SGLang, and TRT-LLM (PyTorch backend, FlashInfer sampling disabled due to CUDA 12.8 / Blackwell limitation).

### TRT-LLM baseline sweep (concurrency 1 → 50, 50 req each)

| Concurrency | RPS | TPS | TTFT avg | ITL avg |
|---|---|---|---|---|
| 1  | 0.40 | 99   | 38 ms  | 10.0 ms |
| 5  | 1.87 | 466  | 71 ms  | 10.5 ms |
| 10 | 3.40 | 847  | 96 ms  | 11.5 ms |
| 20 | 5.32 | 1,322 | 154 ms | 12.2 ms |
| 50 | 6.28 | 1,565 | 294 ms | 30.8 ms |

> TRT-LLM underperforms vLLM/SGLang at higher concurrency because it is running the **PyTorch backend** (not a compiled TRT engine) and FlashInfer sampling is disabled. GPU throughput ceiling: **~1,950 TPS**, saturation at **c=32**. Both limitations stem from CUDA 12.8 not supporting Blackwell (SM 12.x); upgrading to CUDA ≥ 12.9 would unlock the full TRT engine path and FlashInfer.

### Three-way comparison — baseline sweep

| Concurrency | vLLM TPS | SGLang TPS | TRT-LLM TPS | vLLM TTFT | SGLang TTFT | TRT-LLM TTFT |
|---|---|---|---|---|---|---|
| 1  | 101   | 102   | 99    | 25 ms  | 33 ms  | 38 ms  |
| 5  | 496   | 479   | 466   | 43 ms  | 118 ms | 71 ms  |
| 10 | 917   | 913   | 847   | 53 ms  | 52 ms  | 96 ms  |
| 20 | 1,474 | 1,479 | 1,322 | 85 ms  | 63 ms  | 154 ms |
| 50 | 2,907 | 2,576 | 1,565 | 160 ms | 186 ms | 294 ms |

See [`RESULTS.md`](RESULTS.md) for the full three-way comparison across all test suites.

---

### vLLM detailed results (three test suites)

### vLLM — Baseline sweep (concurrency 1 → 50, 50 req each)

| Concurrency | RPS | TPS | TTFT avg | TTFT p95 | ITL avg | Latency p95 |
|---|---|---|---|---|---|---|
| 1 | 0.41 | 101 | 25 ms | 31 ms | 9.9 ms | 2,468 ms |
| 5 | 2.00 | 496 | 43 ms | 53 ms | 10.0 ms | 2,512 ms |
| 10 | 3.70 | 917 | 53 ms | 64 ms | 10.7 ms | 2,710 ms |
| 20 | 5.93 | 1,474 | 85 ms | 109 ms | 11.1 ms | 2,881 ms |
| 50 | 11.71 | 2,907 | 160 ms | 170 ms | 16.6 ms | 4,263 ms |

### vLLM — Fine-grained sweep — finding the saturation point (concurrency 1 → 128, 50 req each)

| Concurrency | RPS | TPS | TTFT avg | TTFT p95 | Latency p95 |
|---|---|---|---|---|---|
| 1 | 0.41 | 101 | 24 ms | 30 ms | 2,467 ms |
| 2 | 0.81 | 201 | 30 ms | 32 ms | 2,475 ms |
| 4 | 1.54 | 383 | 39 ms | 44 ms | 2,495 ms |
| 8 | 2.85 | 708 | 48 ms | 65 ms | 2,530 ms |
| 16 | 4.71 | 1,169 | 72 ms | 82 ms | 2,732 ms |
| 32 | 8.70 | 2,158 | 103 ms | 121 ms | 2,909 ms |
| **64** | **11.74** | **2,917** | **146 ms** | **163 ms** | **4,255 ms** |
| 128 | 11.61 | 2,882 | 185 ms | 195 ms | 4,300 ms |

> **Saturation point: c=64 (~2,917 TPS).** Adding more users yields no throughput gain — the GPU is fully utilised. Marginal TPS actually regresses at c=128 as queue overhead grows.

### vLLM — Overload test (concurrency 100 → 500, 100 req each)

| Concurrency | RPS | TPS | TTFT avg | TTFT p95 | Latency p95 | Errors |
|---|---|---|---|---|---|---|
| 100 | 22.3 | 5,550 | 264 ms | 294 ms | 4,463 ms | 0 |
| 150 | 22.5 | 5,590 | 249 ms | 277 ms | 4,434 ms | 0 |
| 200 | 22.3 | 5,529 | 244 ms | 272 ms | 4,471 ms | 0 |
| 300 | 22.5 | 5,584 | 227 ms | 262 ms | 4,442 ms | 0 |
| 500 | 22.8 | 5,658 | 174 ms | 206 ms | 4,379 ms | 0 |

> **Zero errors at all concurrency levels.** vLLM's continuous batching absorbs extreme load gracefully. Higher TPS here (~5,650) vs. the fine-grained sweep (~2,917) is because a perpetually-full queue lets vLLM pack every decode step to maximum batch size, roughly doubling throughput compared to bursty low-request tests.

### vLLM — Extreme overload (concurrency 1,000 → 5,000, 200 req each)

| Concurrency | RPS | TPS | TTFT avg | TTFT p95 | Latency p95 | Errors |
|---|---|---|---|---|---|---|
| 1,000 | 22.2 | 5,512 | 2,538 ms | 4,788 ms | 8,970 ms | 0 |
| 2,000 | 22.1 | 5,485 | 2,542 ms | 4,778 ms | 8,956 ms | 0 |
| 3,000 | 22.1 | 5,491 | 2,555 ms | 4,799 ms | 8,995 ms | 0 |
| 5,000 | 22.1 | 5,491 | 2,548 ms | 4,800 ms | 9,013 ms | 0 |

> **vLLM never hard-fails — it queues everything.** The real breaking point is latency, not errors:
> - TTFT explodes from **~174 ms** (c=500) → **~2,538 ms** (c=1,000) — a **14× jump**
> - Latency p95 doubles from **~4.4 s** → **~9.0 s** between c=500 and c=1,000
> - TPS stays flat at ~5,500 regardless of queue depth — the GPU is the ceiling
>
> **Practical SLA boundary: c ≈ 500–1,000.** If your target is TTFT < 500 ms, the system degrades well before requests are dropped. To enforce hard limits, set `--max-num-seqs` on the vLLM server or use a client-side timeout.

### Quality evaluation

| Dataset | Metric | vLLM | SGLang | TRT-LLM |
|---|---|---|---|---|
| MMLU | Accuracy | 100% | 100% | 100% |
| GSM8K | Exact Match | 50% | 62.5% | **87.5%** |
| HumanEval | pass@1 | 100% | 100% | 100% |

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
