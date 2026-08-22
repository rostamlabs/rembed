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

## M2 — goroutine parallelism (2026-08-22)

Same machine (20 logical CPUs). MatMulParallel partitions output columns
into 64-wide blocks over a fixed GOMAXPROCS fan-out (tensor.ParallelFor);
attention fans out across heads; GELU across rows. Every path stays
bit-identical to the serial kernels (disjoint outputs, unchanged
accumulation order), pinned by the cross-kernel test and a
concurrent-vs-serial e2e test under -race.

Kernel (same-session A/B, median of 3×20 runs):

| kernel | 128×384·384×1536 (FFN) | 12×384·384×384 |
|--------|------------------------|-----------------|
| blocked (serial) | 20.5 ms | 0.56 ms |
| parallel | 4.8 ms | 0.17 ms |

End-to-end (`bench/compare.py`, 40 runs, both orders):

| engine | threads | median | p10 | p90 | spread |
|--------|---------|--------|-----|-----|--------|
| rembed M2 | GOMAXPROCS | 15.5–17.3 ms | 11.3 ms | 23.6 ms | 58–79% (FLAGged) |
| ONNX Runtime | 1 (pinned) | 3.8 ms | 3.7 ms | 4.2 ms | 12% |
| ONNX Runtime | default pool | 1.8 ms | 1.7 ms | 2.2 ms | 30% |

**e2e 67.4 → ~16 ms ≈ 4.1× over M1 — the 2–4× target hit at the top end.
ours/onnx ≈ 9.6× against ORT's default pool, the like-for-like config now
that both engines are multithreaded** (the ladder so far: ~45× → ~25× →
9.6× in that configuration). Two honesty caveats, both FLAGged by the
harness: (1) this box swung ~1.7× between sessions (M1's blocked-kernel
FFN measured 35 ms then, 20.5 ms now; ORT moved too), so only same-session
A/B ratios are quoted; (2) rembed's spread is now 58–79% — goroutine
scheduling on a P/E-core laptop — versus ~4% when single-threaded, so the
median is soft. The pinned-cloud-box rule from DESIGN.md starts mattering
here, one rung before M3 was expected to need it.

Steady-state allocations grew from 29 to a fixed ~49 per forward pass (the
fan-out's goroutine/closure overhead).

### M2 review fixes (same day)

The review caught the original `minParallelWork = 1<<20` total-work gate
leaving every short-sequence projection serial — seq=3 e2e sped up only
1.20× over single-core while the same shapes parallelize ~2.4–2.9×. It also
disproved the "allocations are seq-independent" claim as then written: the
gate made alloc counts regime-dependent (31 at seq=3 vs 77 at seq=12), and
the test's two size classes sat on the same side of the threshold, so it
couldn't see the violation. Both fixed:

- Per-UNIT work gate (1<<14 MACs) + 2-D (row×column) output partitioning.
  Measured after: 3×384·384×384 135 µs → 57 µs (2.4×), 1×384·384×384
  43 µs → 30 µs, seq=3 e2e 8.7 ms → **4.4 ms**; seq=12 unchanged (~17 ms).
  The 2-D grid also lifts the 6-unit cap on projection shapes and lets
  n < 128 matmuls parallelize by rows.
- Alloc equality now asserted across THREE size classes (seq 3/12/~480) and
  genuinely holds — the gate no longer changes regime with seq.
- Repack/gather moved inside the head workers (no serial Amdahl bracket;
  pack-then-consume stays cache-hot); ParallelFor propagates worker panics
  to the caller (a library must not kill its host process), spawns
  min(GOMAXPROCS, units), and its unit-coverage test sweeps GOMAXPROCS.

Caveat carried from the review, unfixed by design this rung: concurrent
Embed callers each spawn their own fan-out — aggregate throughput saturates
around 4 in-flight requests (~190 embeds/s on 20 cores) and higher
concurrency mostly buys tail latency. The 9.6× figure is single-request
latency. A worker-cap option on Load is the natural M3+ knob if a server
workload needs it.

## M3 — AVX2 SIMD matmul (2026-08-22)

Hand-written plan9 asm (the kickoff allowed avo or hand-written; the whole
kernel is one dot4 primitive): 8-lane FMA accumulators over four bT rows
per call, sharing every load of the A row, with a scalar tail folded in
AFTER the ymm→xmm reduction (VEX xmm writes zero the upper lanes — the
first version lost half the vector sum on any k%8≠0 and the float64
reference sweep caught it immediately). Same 2-D partition and per-unit
gate as the scalar path; CPU detection via golang.org/x/sys/cpu, scalar
fallback everywhere else. SIMD accumulation order differs from the scalar
kernels (horizontal reduction), so it is held to the relative-tolerance
checks, not bit-identity; golden cases still match ONNX within 1e-4.

Kernel (same-session A/B, median of 3×30):

| shape | parallel (scalar) | SIMD | delta |
|-------|-------------------|------|-------|
| 128×384·384×1536 | 2.97 ms | 0.75 ms | 4.0× |
| 12×384·384×384 | 176 µs | 47 µs | 3.8× |
| 3×384·384×384 | 55 µs | 24 µs | 2.3× |

End-to-end (same session):

| config | median | note |
|--------|--------|------|
| seq=12, all cores | **4.5 ms** (compare.py: 4.8 ms, spread 57%, FLAGged) | was 17.4 ms at M2 → **3.9×** |
| seq=3, all cores | 2.5 ms | was 4.4 ms |
| seq=12, GOMAXPROCS=1 | 9.9 ms (spread 4.2%) | M0 was 122 ms serial → 12.3× per-core |
| ONNX default pool | 1.5 ms | |
| ONNX 1 thread | ~3.8–6.1 ms (this box, unstable) | |

**ours/onnx ≈ 3.3× against ORT's default pool — the M3 target ("within
~2–3× of ONNX Runtime") is hit at its edge, with both order-median FLAGs
firing.** Ladder to date in that configuration: ~45× → ~25× → 9.6× →
3.3×. Single-core, rembed is within ~1.6–2.6× of pinned ORT. This is the
rung DESIGN.md says needs a pinned cloud box before any publishable claim;
these laptop numbers remain relative-only, and the remaining gap is now of
the same magnitude as this machine's own noise floor.

## M4 — parity with ONNX Runtime (2026-08-22)

Five changes, each driven by a profile and several corrected by a failed
experiment the harness caught:

1. **Fast float32 exp/erf** replacing the float64 libm calls in Softmax and
   GELU (~28% of the pass), written FLAT in the loop bodies — the helper
   exceeds Go's inline budget (cost 99 vs 80), and a call per element
   stops the out-of-order core from overlapping consecutive elements'
   chains. Accuracy ~3e-7, pinned by tests against the stdlib.
2. **Attention head matmuls through the serial SIMD body**
   (tensor.MatMulSerial) — they were 27% of the pass on the scalar kernel.
3. **GEBP packed-weights kernel** (gemm4x16, plan9 asm): 4×16 tiles,
   broadcast-FMA, 8 accumulator chains saturating both FMA ports; weights
   packed ONCE at load (a structural edge over runtimes that pack per
   session/call), Q‖K‖V fused into one [3H×H] matmul. Measured 101 GFLOPS
   single-P-core (~69% of peak). Failed first cut, caught same-day: units
   of one tile ran SLOWER than serial (counter thrash + the interleaving
   handing adjacent 64-byte strips to different workers); a loop-order swap
   aimed at DRAM streaming was a no-op (per-layer weights are L3-resident).
4. **Spinning fork-join pool** (tensor.Pool), one per forward pass: the
   ~36 fan-outs per embed each paid goroutine wake latency — about HALF
   the short-sequence wall time. Workers spawn once, spin between tasks,
   exit at the end; per-task state makes stale workers harmless.
5. **Fan-out = GOMAXPROCS**: with coordination cheap, the earlier
   seq-scaled worker cap (which was compensating for wake latency)
   inverted — full width wins at every measured seq.

Also learned: this box is an i9-13900H — 6 P-cores at 101 GFLOPS/core on
the kernel, 8 E-cores 2.5× slower. E-core workers poison fan-out tails,
which is most of the unpinned spread.

Result (compare.py, 60 runs, both orders, after cool-down;
`taskset` pins BOTH engines identically):

| config | rembed | ONNX Runtime | ratio |
|--------|--------|--------------|-------|
| pinned 6 P-cores | 1.458 ms (p10 1.316) | 1.387 ms (p10 1.361) | **1.05× — FLAGged as parity** |
| unpinned, all cores | 2.48 ms (spread 123%) | 1.68 ms | 1.47× — FLAGged within noise |

**Statistical parity with ONNX Runtime on P-cores — the ladder reads
~45× → ~25× → 9.6× → 3.3× → 1.05×.** The harness FLAGs both configs as
inside measurement noise, so "faster than ONNX" is NOT claimable from this
laptop; the pinned cloud box (DESIGN.md rule) is where that claim gets
settled. The structural lever that goes decisively past parity is int8
(4× less weight traffic on a pass that streams 42 MB of weights per
embed). Trade recorded: the spinning pool burns idle worker cores for the
duration of a forward pass — right for latency, wrong for saturated
servers; a worker-cap Load option remains the follow-up.
