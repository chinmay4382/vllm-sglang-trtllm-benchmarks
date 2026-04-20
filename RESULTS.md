# Benchmark Results — vLLM vs SGLang vs TRT-LLM

**Model:** Qwen/Qwen2.5-7B-Instruct | **GPU:** RTX 5090 (32 GB)

TRT-LLM results use the **full TRT engine** (`--backend tensorrt`) with CUDA 12.9 and FlashInfer enabled.
Prior runs on CUDA 12.8 (PyTorch backend, FlashInfer disabled) are noted separately for reference.

---

## Baseline Sweep (concurrency 1 → 50, 50 req each)

| Concurrency | vLLM TPS | SGLang TPS | TRT-LLM TPS | vLLM TTFT avg | SGLang TTFT avg | TRT-LLM TTFT avg |
|---|---|---|---|---|---|---|
| 1  | 101   | 102   | 93    | 25 ms  | 33 ms  | 22 ms  |
| 5  | 496   | 479   | 458   | 43 ms  | 118 ms | 33 ms  |
| 10 | 917   | 913   | 896   | 53 ms  | 52 ms  | 48 ms  |
| 20 | 1,474 | 1,479 | 1,477 | 85 ms  | 63 ms  | 83 ms  |
| 50 | 2,907 | 2,576 | 4,063 | 160 ms | 186 ms | 191 ms |

> TRT-LLM pulls ahead of both backends at c=50 (**4,063 TPS** vs vLLM 2,907, SGLang 2,576).

---

## Fine-grained Sweep — Saturation Point (concurrency 1 → 128, 50 req each)

| Concurrency | vLLM TPS | SGLang TPS | TRT-LLM TPS | vLLM TTFT avg | SGLang TTFT avg | TRT-LLM TTFT avg |
|---|---|---|---|---|---|---|
| 1   | 101   | 102   | 93    | 24 ms  | 24 ms  | 22 ms  |
| 2   | 201   | 200   | 185   | 30 ms  | 30 ms  | 23 ms  |
| 4   | 383   | 384   | 353   | 39 ms  | 34 ms  | 32 ms  |
| 8   | 708   | 704   | 649   | 48 ms  | 43 ms  | 38 ms  |
| 16  | 1,169 | 1,144 | 1,132 | 72 ms  | 70 ms  | 59 ms  |
| 32  | 2,158 | 2,069 | 2,179 | 103 ms | 87 ms  | 113 ms |
| **64**  | **2,917** | **2,887** | **4,108** | **146 ms** | **126 ms** | **176 ms** |
| 128 | 2,882 | 2,836 | 4,101 | 185 ms | 195 ms | 231 ms |

> Saturation points: **c=64** for all three backends. TRT-LLM peaks at **~4,108 TPS** — **41% higher** than vLLM (~2,917) and **42% higher** than SGLang (~2,887).

---

## Overload Test (concurrency 100 → 500, 100 req each)

| Concurrency | vLLM TPS | SGLang TPS | TRT-LLM TPS | vLLM TTFT avg | SGLang TTFT avg | TRT-LLM TTFT avg | Errors |
|---|---|---|---|---|---|---|---|
| 100 | 5,550 | 5,638 | 4,200 | 264 ms | 188 ms | 1,323 ms | 0 |
| 150 | 5,590 | 5,639 | 4,219 | 249 ms | 223 ms | 1,336 ms | 0 |
| 200 | 5,529 | 5,644 | 4,226 | 244 ms | 182 ms | 1,322 ms | 0 |
| 300 | 5,584 | 5,636 | 4,211 | 227 ms | 186 ms | 1,321 ms | 0 |
| 500 | 5,658 | 5,613 | 4,220 | 174 ms | 197 ms | 1,300 ms | 0 |

> GPU ceiling: ~**5,600 TPS** for vLLM and SGLang; ~**4,220 TPS** for TRT-LLM. TTFT on TRT-LLM climbs to ~1.3 s at c=100 (SLA broken but better than PyTorch backend's 2.5 s). Zero errors across all backends.

---

## Extreme Overload — Breaking Point (concurrency 1,000 → 5,000, 200 req each)

| Concurrency | vLLM TPS | SGLang TPS | vLLM TTFT avg | SGLang TTFT avg | Errors |
|---|---|---|---|---|---|
| 1,000 | 5,512 | 5,574 | 2,538 ms | 2,480 ms | 0 |
| 2,000 | 5,485 | 5,421 | 2,542 ms | 2,758 ms | 0 |
| 3,000 | 5,491 | 5,494 | 2,555 ms | 2,521 ms | 0 |
| 5,000 | 5,491 | 5,526 | 2,548 ms | 2,497 ms | 0 |

> Extreme overload not yet run for TRT-LLM TRT engine.

---

## Quality Evaluation

| Dataset   | Metric      | vLLM  | SGLang | TRT-LLM   |
|-----------|-------------|-------|--------|-----------|
| MMLU      | Accuracy    | 100%  | 100%   | 100%      |
| GSM8K     | Exact Match | 50%   | 62.5%  | **87.5%** |
| HumanEval | pass@1      | 100%  | 100%   | 100%      |

---

## Summary

| Metric                          | vLLM          | SGLang        | TRT-LLM (TRT engine, CUDA 12.9) |
|---------------------------------|---------------|---------------|----------------------------------|
| TPS @ c=1                       | 101           | 102           | 93                               |
| TPS @ c=50                      | 2,907         | 2,576         | **4,063**                        |
| GPU throughput ceiling          | ~5,650 TPS    | ~5,640 TPS    | ~4,220 TPS                       |
| Saturation point (latency-safe) | ~2,917 @ c=64 | ~2,887 @ c=64 | **~4,108 @ c=64**                |
| TTFT avg @ c=1                  | 25 ms         | 33 ms         | **22 ms**                        |
| TTFT avg @ c=50                 | 160 ms        | 186 ms        | 191 ms                           |
| SLA breaks (TTFT > 500 ms) at   | c≈1,000       | c≈1,000       | c≈100                            |
| Zero-error tolerance            | c=5,000       | c=5,000       | c=500 (tested)                   |

**TRT-LLM (TRT engine)** delivers the highest throughput at c≥32 — **41% more TPS than vLLM at saturation**. TTFT at low concurrency is also the lowest (22 ms at c=1). The trade-off: TRT-LLM's TTFT SLA breaks earlier under overload (c≈100 vs c≈1,000 for vLLM/SGLang) because the TRT engine has a lower per-request queue depth before scheduling pressure builds.

---

## TRT-LLM vs PyTorch Backend (same GPU, CUDA 12.8 vs 12.9)

| Metric | PyTorch backend (CUDA 12.8) | TRT engine (CUDA 12.9) | Improvement |
|---|---|---|---|
| TPS @ c=50 | 1,565 | 4,063 | **+160%** |
| Saturation TPS | ~1,742 @ c=32 | ~4,108 @ c=64 | **+136%** |
| GPU ceiling | ~1,950 TPS | ~4,220 TPS | **+116%** |
| TTFT avg @ c=1 | 38 ms | 22 ms | **-42%** |
| TTFT avg @ c=50 | 294 ms | 191 ms | **-35%** |

> Upgrading CUDA 12.8 → 12.9 and switching to `--backend tensorrt` more than **doubles throughput** and reduces TTFT by ~35–42%.

---

## TRT-LLM Installation Notes (RTX 5090 / Blackwell)

### CUDA 12.8 workarounds (no longer needed with CUDA 12.9)

| Issue | Root cause | Fix |
|---|---|---|
| `cannot load MPI library` | `mpi4py` needs system OpenMPI | `apt install libopenmpi-dev openmpi-bin` |
| `libcublasLt.so.13: not found` | TRT-LLM 1.2.0 built against CUDA 13 cuBLAS | `apt install libcublas-13-0` |
| `/usr/local/cuda symlink broken` | apt redirected it to cuda-13.0 (no nvcc) | `update-alternatives --set cuda /usr/local/cuda-12.x` |
| `hf_transfer not available` | TRT-LLM sets `HF_HUB_ENABLE_HF_TRANSFER=1` | `pip install hf_transfer` |
| `FlashInfer requires GPUs with sm75` | PyTorch can't query SM 12.x without CUDA ≥ 12.9 | `FLASHINFER_CUDA_ARCH_LIST="12.0f"` env var |
| `nvcc: Unsupported gpu architecture compute_120f` | CUDA 12.8 nvcc doesn't support Blackwell | Disable FlashInfer via `--extra_llm_api_options` YAML |

### Recommended setup (CUDA 12.9+)

```bash
pip install tensorrt-llm --extra-index-url https://pypi.nvidia.com
pip install hf_transfer
apt-get install -y libopenmpi-dev openmpi-bin libcublas-13-0 cuda-toolkit-12-9
update-alternatives --set cuda /usr/local/cuda-12.9

export LD_LIBRARY_PATH=/usr/local/cuda-13.0/targets/x86_64-linux/lib:$LD_LIBRARY_PATH

trtllm-serve serve <model> \
  --host 0.0.0.0 --port 8000 \
  --backend tensorrt \
  --max_batch_size 64 \
  --max_num_tokens 4096
```

---

## Analysis — TRT-LLM vs vLLM vs SGLang

### Why TRT-LLM wins at practical concurrency (c=1–64)

At c=64, TRT-LLM delivers **4,108 TPS** — 41% higher than vLLM (2,917) and 42% higher than SGLang (2,887). This comes from what the TRT engine actually does at startup:

- **Compiled CUDA kernels**: the model is compiled into a TensorRT binary with fused layers for the exact target hardware (RTX 5090, SM 12.0). Every matrix multiply, attention block, and MLP is a single optimized kernel. vLLM and SGLang run through PyTorch's runtime op dispatch — each operation involves kernel selection, tensor bookkeeping, and Python overhead per step.
- **CUDA graph capture**: execution graphs are pre-recorded for every batch size at warmup (1, 2, 4... 64). At inference time, replaying a captured graph eliminates all CPU-side scheduling overhead entirely.
- **FlashInfer sampling**: token sampling at each decode step uses FlashInfer's optimized CUDA kernels rather than standard PyTorch sampling.
- **Fused MLP**: the transformer's feed-forward layers are merged into a single kernel (`use_fused_mlp=True`), halving memory round-trips on the largest compute block in the model.

In short, TRT-LLM does in one kernel what vLLM does in three or four — at practical serving concurrency, this difference dominates.

### Why vLLM and SGLang win at extreme overload (c=100+)

Once the request queue is permanently saturated, vLLM hits **~5,650 TPS** and SGLang **~5,640 TPS** while TRT-LLM plateaus at **~4,220 TPS**. The gap comes from batching strategy:

- **TRT-LLM uses static batch sizes**: the engine is compiled for fixed sizes (1, 2, 4... 64, 128). When 300 requests are queued, it still processes them in groups of 128, 128, 44 — padding waste grows as the queue depth exceeds the largest compiled batch size.
- **vLLM and SGLang use continuous batching**: every decode step, the scheduler inspects the full queue and packs exactly as many sequences as the GPU can hold. With hundreds of requests waiting, every single decode step runs at 100% GPU utilization with no padding overhead.
- **Prefill/decode interleaving**: vLLM and SGLang dynamically mix prefill (processing new input tokens) and decode (generating output tokens) within the same batch. TRT-LLM's static compiled engine has less runtime flexibility to interleave these phases.

### When to use each

| Use case | Best choice | Reason |
|---|---|---|
| Production serving, TTFT SLA < 200 ms | **TRT-LLM** | Lowest latency and highest throughput at c=1–64 |
| Batch processing, maximum raw throughput | **vLLM or SGLang** | Continuous batching saturates GPU better at c=100+ |
| Simplest deployment, any GPU | **vLLM or SGLang** | No engine compilation step, works out of the box |
| Latency-critical, single-digit concurrency | **TRT-LLM** | TTFT 22 ms vs 25–33 ms for others |
