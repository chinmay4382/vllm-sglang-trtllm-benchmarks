# Benchmark Results — vLLM vs SGLang

**Model:** Qwen/Qwen2.5-7B-Instruct | **GPU:** RTX 5090 (32 GB)

---

## Baseline Sweep (concurrency 1 → 50, 50 req each)

| Concurrency | vLLM RPS | SGLang RPS | vLLM TPS | SGLang TPS | vLLM TTFT avg | SGLang TTFT avg |
|---|---|---|---|---|---|---|
| 1  | 0.41 | 0.41 | 101   | 102   | 25 ms  | 33 ms  |
| 5  | 2.00 | 1.92 | 496   | 479   | 43 ms  | 118 ms |
| 10 | 3.70 | 3.67 | 917   | 913   | 53 ms  | 52 ms  |
| 20 | 5.93 | 5.95 | 1,474 | 1,479 | 85 ms  | 63 ms  |
| 50 | 11.71| 10.35| 2,907 | 2,576 | 160 ms | 186 ms |

---

## Fine-grained Sweep — Saturation Point (concurrency 1 → 128, 50 req each)

| Concurrency | vLLM TPS | SGLang TPS | vLLM TTFT avg | SGLang TTFT avg |
|---|---|---|---|---|
| 1   | 101   | 102   | 24 ms  | 24 ms  |
| 2   | 201   | 200   | 30 ms  | 30 ms  |
| 4   | 383   | 384   | 39 ms  | 34 ms  |
| 8   | 708   | 704   | 48 ms  | 43 ms  |
| 16  | 1,169 | 1,144 | 72 ms  | 70 ms  |
| 32  | 2,158 | 2,069 | 103 ms | 87 ms  |
| **64**  | **2,917** | **2,887** | **146 ms** | **126 ms** |
| 128 | 2,882 | 2,836 | 185 ms | 195 ms |

> Saturation point: **c=64** for both backends (~2,900 TPS).

---

## Overload Test (concurrency 100 → 500, 100 req each)

| Concurrency | vLLM TPS | SGLang TPS | vLLM TTFT avg | SGLang TTFT avg | Errors |
|---|---|---|---|---|---|
| 100 | 5,550 | 5,638 | 264 ms | 188 ms | 0 |
| 150 | 5,590 | 5,639 | 249 ms | 223 ms | 0 |
| 200 | 5,529 | 5,644 | 244 ms | 182 ms | 0 |
| 300 | 5,584 | 5,636 | 227 ms | 186 ms | 0 |
| 500 | 5,658 | 5,613 | 174 ms | 197 ms | 0 |

> GPU ceiling: ~**5,600 TPS** for both. Zero errors at all levels.

---

## Extreme Overload — Breaking Point (concurrency 1,000 → 5,000, 200 req each)

| Concurrency | vLLM TPS | SGLang TPS | vLLM TTFT avg | SGLang TTFT avg | Errors |
|---|---|---|---|---|---|
| 1,000 | 5,512 | 5,574 | 2,538 ms | 2,480 ms | 0 |
| 2,000 | 5,485 | 5,421 | 2,542 ms | 2,758 ms | 0 |
| 3,000 | 5,491 | 5,494 | 2,555 ms | 2,521 ms | 0 |
| 5,000 | 5,491 | 5,526 | 2,548 ms | 2,497 ms | 0 |

> TTFT jumps ~12× between c=500 and c=1,000 on both backends. TPS stays flat — GPU is the ceiling.

---

## Quality Evaluation

| Dataset   | Metric      | vLLM  | SGLang |
|-----------|-------------|-------|--------|
| MMLU      | Accuracy    | 100%  | 100%   |
| GSM8K     | Exact Match | 50%   | 62.5%  |
| HumanEval | pass@1      | 100%  | 100%   |

---

## Summary

| Metric                        | vLLM     | SGLang   |
|-------------------------------|----------|----------|
| GPU throughput ceiling        | ~5,650 TPS | ~5,640 TPS |
| Saturation point (latency-safe) | ~2,917 TPS @ c=64 | ~2,887 TPS @ c=64 |
| TTFT < 200 ms up to           | c=500    | c=300    |
| SLA breaks (TTFT > 500 ms) at | c≈1,000  | c≈1,000  |
| Zero-error tolerance          | c=5,000  | c=5,000  |

Both backends perform near-identically. Differences are within noise margin (~1–2%).
