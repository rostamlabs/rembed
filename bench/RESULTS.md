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

## M1 — alloc-free + cache-blocked matmul (2026-08-22)

Same machine/input as M0. Kernel (`go test -bench MatMul ./internal/tensor/`,
median of 5×20 runs):

| kernel | 128×384·384×1536 (FFN) | 12×384·384×384 (projection) |
|--------|------------------------|------------------------------|
| naive | 67.5 ms | 2.0 ms |
| blocked (1×4) | 35.0 ms | 1.0 ms |

**Kernel ≈ 1.9–2.0×.** A 2×4 micro-kernel was tried and measured ~7% slower
than 1×4 (register spills); noted in the kernel's comment, revisit at M3.

End-to-end (`bench/compare.py`, ORT pinned to 1 thread):

| engine | median | p10 | p90 | spread |
|--------|--------|-----|-----|--------|
| rembed M1 | 67.4 ms | 67.2 ms | 67.9 ms | 1.0% |
| ONNX Runtime (1 thread) | 6.1 ms | 5.9 ms | 7.9 ms | 31.8% |

**e2e 122.1 ms → 67.4 ms = 1.81× over M0; ours/onnx 20× → 11×
single-threaded.** Steady-state allocations: 29 allocs / 13 KB per embed
(tokenizer output + returned vector; M0 reallocated every intermediate
buffer each call — ~250 KB across 15 allocations at seq=12, ~10 MB at
seq=512).

**Honest note vs the plan:** DESIGN.md targeted 3–5× for this rung; measured
is 1.81×. The forward pass is ~90% matmul (≈256 MFLOP at seq=12, ≈4.3
GFLOPS achieved), so e2e can never beat the kernel's own 2×, and ~2× is
about the scalar-Go ceiling for this loop (no autovectorization; FMA latency
chains bound it). The remaining gap to ONNX is precisely what M2
(parallelism, ~11× → target /nCores) and M3 (AVX2 SIMD, the big lever) exist
to close. The 3–5× estimate assumed more headroom in scalar code than
exists.
