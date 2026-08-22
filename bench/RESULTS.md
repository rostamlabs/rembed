# Benchmark log

One entry per ladder rung. All numbers from `bench/compare.py` (warm-up
discard, medians over 30 runs, both orders) unless noted. Laptop numbers are
relative-only/provisional — see DESIGN.md; a pinned cloud box arrives ~M3.

## M0 — naive fp32 baseline (2026-08-22)

Machine: linux/amd64 laptop (Linux 6.5), Go 1.26.1, onnxruntime CPU EP.
Input: "The quick brown fox jumps over the lazy dog." (seq=12),
model all-MiniLM-L6-v2.

| engine | median | p10 | p90 | spread |
|--------|--------|-----|-----|--------|
| rembed M0 | 127.4 ms | 125.3 ms | 130.4 ms | 4.0% |
| ONNX Runtime | 3.2 ms | 2.7 ms | 6.7 ms | 122% (unstable, FLAGged) |

**ours / onnx ≈ 40×.** Expected for three plain loops with per-call
allocations; this is the denominator every later rung is measured against.
Correctness: all 8 golden cases match the ONNX reference within 2.9e-7
(tolerance 1e-4); token ids match HF exactly.

Kernel micro-baseline (`go test -bench MatMulNaive ./internal/tensor/`,
FFN shape 128×384 · 384×1536): ~58–60 ms/op (three runs: 58.7, 59.9, 57.4).
The M1 cache-blocked body is measured against this.
