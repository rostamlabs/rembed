# Benchmarks

rembed is measured against ONNX Runtime as a constant baseline. The headline: the
optimization ladder takes the naive Go forward pass from **~45× slower** than
ONNX Runtime to **statistical parity** for fp32, and — with int8 — **consistently
ahead** of it, on the reference laptop.

!!! warning "Single-machine numbers are relative-only"
    Every figure here is same-session and same-machine. Absolute latencies drift
    between sessions and between CPUs; treat the **ratios** to the ONNX baseline
    as the result, not the milliseconds. Any delta within measurement noise is
    flagged as such below. Publishable magnitude claims wait on a pinned
    bare-metal box.

## Methodology

The harness (`bench/compare.py`, driving `rembed bench -json`) follows the same
discipline throughout:

- **Warm-up discard** — the first runs are thrown away.
- **Medians over N runs** (30 by default), with p10/p90 and spread reported —
  not a single sample.
- **Order-reversal control** — every comparison is run both us-then-ONNX *and*
  ONNX-then-us, so a warm-cache ordering effect can't masquerade as a win.
- **ONNX Runtime as the constant baseline**, in the same session.
- **Noise flagging** — any delta inside the measured spread is marked, not
  reported as a result.
- **`quantized` verification** — the JSON output records which quantization mode
  *actually* ran, because int8 fallback is silent by design; a benchmark can't
  accidentally credit int8 speed to an fp32 run.

## The fp32 ladder

Reference laptop (i9-13900H), all-MiniLM-L6-v2, vs ONNX Runtime's default
configuration — the per-rung ratio to ONNX:

```
M0 ~45×  →  M1 ~25×  →  M2 9.6×  →  M3 3.3×  →  M4 1.05×  →  int8 0.89×
```

- **M0** — naive, ~20× slower single-threaded (~46× vs ONNX's default thread
  pool); correctness `2.9e-7` vs the golden.
- **M1** — alloc-free + cache-blocked matmul: 1.81× end-to-end, the scalar-Go
  ceiling.
- **M2** — goroutine parallelism: 4.1× (≈9.6× vs ONNX's default pool).
- **M3** — AVX2 SIMD matmul: 3.3× vs ONNX; 12.3× faster *per core* than M0.
- **M4** — packed GEMM + spinning fork-join pool + fast-math exp/erf: pinned to
  6 P-cores, **1.458 ms vs ONNX 1.387 ms = 1.05×**, flagged parity. (The box is
  an i9-13900H: 6 fast P-cores plus 8 E-cores ~2.5× slower, which is why core
  pinning matters for a clean comparison.)

## int8

Weight-only int8 (`WithInt8`), pinned 6 P-cores:

- **int8 vs rembed fp32 — 1.369 ms vs 1.543 ms = 0.89×.**
- **int8 vs ONNX quint8 (avx2) — 1.333 ms vs 1.403 ms = 0.95×.**
- A sign test across sessions: rembed int8 beat ONNX in **6/6**.
- Accuracy: worst golden cosine `0.99906`, worst max-abs-diff `0.0126`.

Full int8 (`WithInt8Activations`) on AVX-VNNI:

- Reference laptop (medians of 200): MiniLM-L6 fp32 3.69 / weight-int8 3.57 /
  full-int8 **2.62 ms (1.36×)**; mpnet-base 18.2 / 16.2 / **12.7 ms (1.28×)** —
  margins of 28–36%, outside noise.
- Zen 4 EPYC (Genoa, 12-vCPU), mpnet: full int8 vs ONNX fp32 = **0.70–0.75×**,
  ahead in **4/4** rounds; vs ONNX's own AVX-512-VNNI int8, parity. The
  `gemm4x16vnni512` EVEX path is bit-exact with the 256-bit VEX path.

## Single-core vs multi-core

On a contended 20-core box, single-core fp32 lands ~1.26–1.28× *behind* ONNX
(MLAS's GEMM tuning), narrowing to ~1.16× after the `gemm6x16` micro-kernel;
rembed reaches parity multi-core. With full int8 the same single-core comparison
flips to **0.69× (1.45× faster)** on nomic-embed. In short: fp32 is at parity
where cores are available, and int8 is ahead broadly.

## Rejected experiments

Recorded honestly alongside the wins, because a null result is a result:

- **Matmul prefetch** — `PREFETCHT0` in the `gemm4x16` k-loop: 269 vs 270 ms, a
  wash. The kernel is FMA-port-bound, not memory-bound. Reverted.
- **Packed attention kernel** — 1.65× *slower* (~116 µs vs ~70 µs): attention
  K/V can't be pre-packed, so per-call packing cost dominates. Reverted.
- **k-unrolling `gemm6x16`** — a wash inside the spread (1.17× → 1.16×), and the
  forward pass ticked slightly slower; the 6×16 kernel is already
  FMA-throughput-bound. Reverted.
- **2×4 micro-kernel (M1)** — ~7% slower than 1×4 from register spills.
- **DRAM-streaming loop-order swap (M4)** — a no-op; the weights are L3-resident.

Full per-rung tables, per-engine CPU accounting, and every raw session are in
[`bench/RESULTS.md`](https://github.com/rostamlabs/rembed/blob/main/bench/RESULTS.md).
