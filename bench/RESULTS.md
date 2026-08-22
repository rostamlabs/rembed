# Benchmark log

One entry per ladder rung. All numbers from `bench/compare.py` (warm-up
discard, medians over 30 runs, both orders) unless noted. Laptop numbers are
relative-only/provisional — see DESIGN.md; a pinned cloud box arrives ~M3.

## M0 — naive fp32 baseline (2026-08-22)

Machine: linux/amd64 laptop (Linux 6.5), Go 1.26.1, onnxruntime CPU EP.
Input: "The quick brown fox jumps over the lazy dog." (seq=12),
model all-MiniLM-L6-v2.

Two ONNX configurations, because the Go engine is single-threaded until M2:
the pinned 1-thread run is the like-for-like baseline; the default-pool run
is what a user actually gets from ORT out of the box.

| engine | threads | median | p10 | p90 | spread |
|--------|---------|--------|-----|-----|--------|
| rembed M0 | 1 | 122.1 ms | 121.7 ms | 126.4 ms | 3.8% |
| ONNX Runtime | 1 (pinned) | 6.1 ms | 5.9 ms | 6.8 ms | 14.9% |
| ONNX Runtime | default pool | 2.7 ms | 2.5 ms | 3.1 ms | 22.7% |

**ours / onnx ≈ 20× single-threaded (≈ 46× vs ORT's default all-core
pool).** Expected for three plain loops with per-call allocations. The 20×
single-thread figure is the denominator for M1 (kernel quality); the
default-pool figure is the one M2's parallelism must chase.
Correctness: all 11 golden cases (including accents, non-ASCII symbols, and
CJK) match the ONNX reference within 1.5e-7 (tolerance 1e-4); token ids
match HF exactly.

Kernel micro-baseline (`go test -bench MatMulNaive ./internal/tensor/`,
FFN shape 128×384 · 384×1536): ~58–60 ms/op (three runs: 58.7, 59.9, 57.4).
The M1 cache-blocked body is measured against this.
