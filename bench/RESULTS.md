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

## M5 — weight-only int8: ahead of ONNX in every configuration (2026-08-22)

Weight-only int8 (`rembed.WithInt8()`): per-output-channel symmetric
quantization at load, activations stay float32, gemm4x16i8 dequantizes
in-register with one per-channel multiply in the epilogue. The physics: a
seq=12 embed is bound by streaming ~42 MB of fp32 weights; int8 cuts that
to ~10.5 MB, which fits L3. ORT's quint8 model instead dynamically
quantizes activations per call — overhead our weight-only design does not
pay at short sequence.

Accuracy (pinned by TestGoldenInt8 against the fp32 ONNX golden): worst
cosine 0.99906, worst maxAbsDiff 0.0126 on unit-norm embeddings. The int8
micro-benchmark is deliberately absent from the headline: with cache-hot
weights it measures SLOWER than fp32 (conversion cost without the
bandwidth win) — only end-to-end shows the truth, which is exactly why the
harness benchmarks embeds, not kernels, for claims.

Final measurements (compare.py, 80 runs, both orders, both engines pinned
to the 6 P-cores, cool-downs between sessions):

| matchup | rembed | ONNX Runtime | ratio |
|---------|--------|--------------|-------|
| fp32 vs fp32 | — | — | 0.92× (earlier session: 1.05×) |
| **int8 vs fp32** | **1.369 ms** (p10 1.110) | 1.543 ms | **0.89×** (earlier session: 0.70×) |
| **int8 vs ORT quint8_avx2** | **1.333 ms** (p10 1.134) | 1.403 ms | **0.95×** (earlier session: 0.93×) |

Every individual ratio is FLAGged — this laptop's noise floor (30–83%
spreads, order-median drift) exceeds each delta, so no single number here
is a claim. But the SIGN is consistent: across six independent measurement
sessions today, rembed's median beat ORT's in six. Read as a sign test,
not a magnitude claim. The pinned cloud box (DESIGN.md rule) is where the
magnitude gets settled; on this machine, rembed int8's best sessions
(median 1.17–1.25 ms, p10 1.11) sit clearly under anything ORT produced
all day (best 1.31 ms).

Ladder complete: ~45× → ~25× → 9.6× → 3.3× → 1.05× → **0.89×**.

## R3 — ARM64 NEON kernels (2026-08-22)

All three kernels ported: gemm4x16 (pure supported mnemonics — one
64-byte B load + 4 lane dups + 16 FMLAs per k step), gemm4x16i8 (the Go
assembler has no SXTL/SCVTF vector mnemonics, so the int8 widening
pipeline is WORD-encoded, documented instruction by instruction), and
dot4 (vector accumulate + frame-spill reduction). arm64 is now a
first-class kernel target: PackB/PackB8 engage, int8 works, the batch
path's worker budget holds.

Verified the only honest way available without ARM hardware: the ENTIRE
test suite — kernel sweeps against float64 references, the quantizer, and
the full golden suite (fp32 1e-4 vs ONNX, int8 bounds, CLS, the 5-model
matrix, batch bit-identity, token-level outputs) — passes on arm64 under
qemu-aarch64, locally and in a new CI job. Two war stories the sweeps
caught in minutes: scalar Fn ALIASES Vn.S[0] on arm64, so dot4's
reduction temporaries were silently clobbering un-reduced accumulator
lanes (temps moved to dead registers, hazard documented in the file); and
the WORD-encoded int8 widening passed its dequantized-weights reference
sweep on the first run — the encodings are right.

NO performance claims: qemu measures nothing. Real-ARM benchmarks (Apple
Silicon / Graviton) are the follow-up; the structural expectation is
ORT-competitive by the same weights-packed-at-load argument as amd64.

## R4 — MPNet architecture (2026-08-22)

Coverage rung, not a perf rung: all-mpnet-base-v2 — the most-downloaded
sentence-transformers model — now runs end to end. Two architectural
deltas over BERT: positions offset by pad_token_id+1 with no segment
embedding, and a bucketed relative-position bias (one [32×heads] table
shared by all 12 layers) added to attention scores before softmax.
Because positions are contiguous, the bias collapses to a
[heads×(2·seq−1)] delta table computed once per forward — HF
materializes [heads×seq×seq].

Validation against the repo's own ONNX export over the 12-case golden
(the longest case is 324 tokens, so token pairs reach |j−i| ≥ 128 and
the log-spaced far buckets plus the max-distance clamp are pinned end to
end, not just in the unit test): maxAbsDiff 3.3e-7, worst per-text
meanAbs 7.3e-8, cosine 1.000000000 — the same fidelity class as the
BERT matrix (1.5e-7–1.9e-7). The adversarial review additionally
validated against a LIVE HuggingFace MPNetModel forward pass at 4, 52,
402, and 512 tokens: maxAbs ≤ 2.05e-6, cosine 1.0 throughout.
Tokenization (WordPiece with <s>/</s> framing over the 4-shifted vocab)
matches HF's MPNetTokenizer token-for-token on every golden case,
including the accent/CJK/symbol torture cases. Weight-only int8 engages
unchanged (768-dim projections pack cleanly): cosine ≥ 0.9978 vs the
golden — the measured worst case is 0.997874 on the punctuation text,
now test-enforced in the golden matrix. (An earlier draft claimed
0.9989, measured on only 2 texts; the review caught it.)

The bucket function ports HF's float32 log computation to float64 —
verified to pick identical buckets for every |distance| ≤ 1000
(exhaustive check), double the 512-token maximum.

No benchmark entry: the forward pass is the same kernel ladder; MPNet
just runs 12 layers × 768 hidden instead of MiniLM's 6 × 384. R8 (pinned
cloud box) remains where cross-model magnitude claims get made.

## R5 — RoBERTa + byte-level BPE (2026-08-22)

Coverage rung. The encoder needed nothing: RoBERTa is BERT with the
fairseq position offset (pad_token_id+1) that R4 already plumbed, plus a
[1×H] segment table the loader already accepted. The work was the
tokenizer: a pure-Go byte-level BPE (GPT-2 family) — the byte-to-unicode
table, a hand-written pre-tokenization scanner that reproduces the GPT-2
pattern's \s+(?!\S) lookahead (Go's regexp has none; the whitespace-run
backtracking and the only-a-literal-space-joins rule are the subtle
parts), and the ranked-merge loop.

Validation: token-for-token against HF's RobertaTokenizer over a
27-case fixture that hits every pre-tokenizer branch (contractions and
their case-sensitivity, space joining, whitespace backtracking, tabs and
newlines, unicode letter/number classes including No/Nl, multi-byte
UTF-8, emoji) — passed on the first run — plus the golden's own
input_ids assertions. all-distilroberta-v1 against its repo's ONNX
export over the 12-case golden (326-token long case included): fp32
maxAbs 3.2e-7, cosine 1.0. int8: cosine ≥ 0.9985 (measured worst
0.998727 on the long case), enforced in the golden matrix.

No benchmark entry: same kernels, one fewer encoder layer than MPNet.

## R6 — SentencePiece unigram, multilingual models (2026-08-22)

Coverage rung, the deepest tokenizer of the three: SentencePiece Unigram
in pure Go. Three independently-validated layers — a minimal protobuf
wire reader for the .model file; the NMT-NFKC normalizer, driven by the
model's precompiled charsmap (a darts-clone double-array trie whose
traversal was ported unit-for-unit from the darts.h sentencepiece
bundles, replacement strings NUL-delimited in the trailing blob); and
Viterbi segmentation over 250k pieces with the unknown-character rules
ported from unigram_model.cc (unk node only when no single-char piece
covers a position; CONSECUTIVE unknowns merge into one piece — the one
divergence the first fixture run caught, visible as three <unk> where
sentencepiece emits one).

Validation, per layer, against the reference implementation over a
31-case battery spanning nine scripts (Persian, Arabic, CJK, Cyrillic,
Greek, Hebrew, Korean, Devanagari) plus fullwidth/ligature NFKC cases,
unicode spaces, emoji, and a >512-token Persian truncation case:
normalization BYTE-FOR-BYTE vs sentencepiece's own Normalize; pieces
exactly vs EncodeAsPieces; ids token-for-token vs HF XLMRobertaTokenizer
(fairseq remapping + truncation included).

paraphrase-multilingual-MiniLM-L12-v2 (a plain BERT encoder with a
250037-row padded embedding table) against its repo's ONNX export over
the 13-case golden (now including a Persian sentence): fp32 maxAbs
6.96e-7, cosine 1.0. int8: cosine ≥ 0.9995 (measured worst 0.999642),
enforced in the golden matrix. Ten validated models, three
architectures, three tokenizer families.

## R7 — AVX-VNNI full int8 (2026-08-23)

The last performance rung reachable on this machine. AVX-512-VNNI does
not exist on consumer Raptor Lake, but AVX-VNNI (the 256-bit VEX form of
VPDPBUSD) does — and Go's assembler only knows the EVEX encoding, which
faults without AVX-512. So: eight hand-encoded VEX byte sequences,
generated field-by-field (VEX.256.66.0F38.W0 50 /r) and verified
byte-for-byte against GNU as's {vex}-prefixed output before first run —
the amd64 mirror of R3's WORD-encoded NEON story.

The scheme: weights per-output-channel symmetric s8 (as M5, float64
scale division and all); activations per-ROW asymmetric u8 via the +128
offset trick — q = round(a/s)+128, so VPDPBUSD's u8·s8 accumulation
equals the true dot plus 128·colSum, and the column sums are precomputed
at pack time and subtracted in the epilogue. One VPDPBUSD does four
multiply-accumulates per int32 lane: 4× the MACs per instruction of the
fp32 FMA path, and activations stream at a quarter the bytes.

Correctness: the kernel is BIT-EXACT against a pure-Go integer
reference over every shape class (there is no floating-point rounding to
hide behind — the accumulation is integers until the epilogue), and
serial vs parallel outputs are bit-identical (per-row quantization,
disjoint tiles). The review additionally proved the no-intermediate-
saturation semantics adversarially (all four products maximal in every
lane, k up to 1536: bit-exact), decoded all eight VEX byte sequences
independently, and showed int32 accumulator wraparound CANCELS exactly
in the zero-point correction — the true overflow limit is k > 131072,
~85× past any transformer.

Golden accuracy, measured per model and enforced in the golden matrix:
worst cosine 0.9917 (MiniLM-L6), 0.9932 (L12), 0.9979 (L3), 0.9912
(mpnet), 0.9982 (multilingual), 0.9991 (gte) — and 0.9747 for
distilroberta, where RoBERTa's activation outliers (~2.7× larger row
absmax, measured) defeat the per-row absmax scheme and short texts
degrade most. That spread is why the mode is a separate opt-in
(WithInt8Activations), never a WithInt8 upgrade, and why the README
carries a per-model table with a RoBERTa warning instead of one blanket
bound.

Pinned same-machine A/B (taskset 0-11, medians of 200 after warmup):

| model | fp32 | int8 weights | int8 FULL | full vs weights |
|---|---|---|---|---|
| MiniLM-L6 | 3.69 ms | 3.57 ms | 2.62 ms | 1.36× |
| mpnet-base | 18.2 ms | 16.2 ms | 12.7 ms | 1.28× |

FLAG: single-machine numbers, same protocol caveats as M4/M5 — but the
margins here are 28-36%, far outside the noise band that made the M5
verdict a sign-test. Weight-only int8 already sat at 0.89× of ONNX
Runtime fp32; full int8 on VNNI hardware is decisively ahead. R8's
pinned cloud box remains where the cross-engine magnitude gets settled.

## R8 — Cloud-box cross-engine benchmark (2026-08-23)

The first rembed-vs-ONNX-Runtime numbers from OFF the development
laptop. Box: a 12-vCPU AMD EPYC Genoa (Zen 4) KVM guest, 2.0 GHz, 22 GB,
Ubuntu 24.04 (kernel 6.8), single NUMA node — a cloud VM, honestly NOT
the dedicated bare-metal machine the roadmap idealized. Hypervisor
jitter is real and the harness's FLAG discipline was in charge
throughout: every number below comes from bench/compare.py's
both-orders/warm-up/median protocol, and anything the noise rules
flagged is reported as parity, not victory.

The box also closed R7's loose end: Zen 4 has AVX-512-VNNI but NOT
AVX-VNNI, so the EVEX twin of the VNNI kernel (gemm4x16vnni512, Go's
native VPDPBUSD mnemonics this time) was built and PROVEN here on real
hardware — bit-exact against the integer reference, and its end-to-end
full-int8 worst golden cosine is 0.991652, IDENTICAL to the last bit
with the VEX path on Raptor Lake. Two encodings, two vendors, one
integer computation. Full int8 now engages on every VNNI-capable CPU whose OS saves AVX-512
state (macOS is the exception — x/sys disables AVX-512 there).

mpnet-base (the stable workload on this VM; L6's sub-2 ms runs drown in
hypervisor jitter), four independent rounds, 50-100 runs per order.
Flag types matter and are disclosed per round: "order" = the harness's
machine-unstable/rerun flag (order medians differ >10%), "noise" = the
delta-within-spread flag whose rule is "treat as parity".

| comparison | round ratios (ours/ORT) | flags per round | verdict |
|---|---|---|---|
| fp32 vs ORT fp32 | 1.08 / 1.13 / 1.07 | noise / noise / order+noise | parity (ORT nominally ahead, single digits) |
| full int8 vs ORT fp32 | 0.78 / 0.59 / 0.70 / 0.75 | order+noise / order+noise / none / **none** | **ahead in 4/4 rounds; the two flag-free rounds say 0.70× and 0.75×** |
| full int8 vs ORT qint8-avx512-vnni | 1.16 / 0.97 / 0.95 | order+order / noise / noise | parity with ORT's strongest int8 graph |

The flag-free variance row (round 4, 100 runs per order, both orders —
the checkable magnitudes DESIGN.md's variance rule asks for):

| engine | median | p10 | p90 | spread |
|---|---|---|---|---|
| rembed full int8 | 5.910 ms | 5.619 | 6.628 | 17.1% |
| ORT fp32 | 7.922 ms | 7.620 | 8.791 | 14.8% |

The headline, stated carefully: on Zen 4 server hardware, pure-Go
rembed fp32 is at parity with ONNX Runtime fp32; rembed full int8 beats
ORT fp32 in every round, and the two rounds with NO flags of either
kind measured 0.70× and 0.75× — a clean 25-30% win, not a sign test
this time; and against ORT's own hand-tuned AVX-512-VNNI int8 graphs,
rembed trades blows at parity while its VNNI kernel is still 256-bit
ymm. The obvious remaining headroom on 512-bit native parts is a
zmm-wide kernel — future work, now with hardware that can test it.

L6 on this VM: all four comparisons FLAG-parity (spreads 40-94% on
sub-2 ms runs); full int8 vs ORT qint8-avx512-vnni measured 0.96×.
Small-model latencies need quieter hardware than a shared KVM guest.

## zmm fp32 kernel (2026-08-23)

The fp32 counterpart of the VNNI-width story: gemm4x32, the AVX-512
twin of gemm4x16 — same 4-row micro-tile doubled to two zmm per row,
over 32-column B panels that PackB now builds automatically on AVX-512
hardware (every BERT-family dim divides by 32). Per-element accumulation
order is unchanged, so output is BIT-IDENTICAL to the ymm kernel —
pinned by TestPackedPanelWidthsAgree on the Zen 4 box, along with the
full packed sweep and golden battery over the 32-wide layout.

Measured honestly at kernel level on the Zen 4 box (serial pool, 200
iterations × 3 repeats, ymm vs zmm on identical packed data):

| shape | ymm (w16) | zmm (w32) | delta |
|---|---|---|---|
| 28×768×2304 (qkv) | ~1.04 ms | ~1.02 ms | wash |
| 28×768×3072 (ffn1) | ~1.39 ms | ~1.28 ms | ~6% |
| 28×3072×768 (ffn2) | ~1.40 ms | ~1.27 ms | ~8% |
| 160×768×3072 (long-seq ffn1) | 7.56 ms (stable) | 6.87 ms (stable) | **9.5%** |

Exactly what the kernel header predicts for Zen 4: 512-bit FMAs are
double-pumped through 256-bit ports there, so peak FLOPs are unchanged
and the win is front-end relief — half the instructions per k step —
worth 5-10% on large shapes and nothing on small ones. The end-to-end
PROJECTION from that is ~6% on mpnet — too small to resolve on this VM,
whose round-to-round drift exceeds it, and the ORT fp32 rematch
accordingly showed no movement (1.14×/1.18×, first round flagged). The
2× peak-FLOPs story belongs to Intel server parts with two native
512-bit FMA units (Sapphire Rapids+), where this same kernel is the
enabler — hardware to measure that is the open item. Never slower,
bit-identical, kept.

Review addendum, applied before merge: (a) the accumulator zeroing was
VXORPS on zmm — an AVX512DQ instruction behind an AVX512F-only gate, a
latent SIGILL on F-without-DQ parts (KNL, hypervisor feature masks);
now VPXORQ, pure F. (b) The parallel column chunk was counted in
PANELS, so 32-wide panels silently halved the unit count — the review
measured 18-26% lost on n=768 shapes of a many-core box via a ymm
proxy. The chunk is now fixed in COLUMNS (64), width-neutral; with the
fix the box re-measured the A/B under an 11-worker PARALLEL pool and
the zmm win survives intact there: ffn2 28×3072×768 380→348 µs (~9%),
long-seq ffn1 160×768×3072 740→662 µs (~11%), stable across repeats.
This is exactly the class of interaction the serial-pool A/B was blind
to, and the review caught it before it shipped.

## Coverage expansion — 15 validated models, 4 architectures (2026-08-23)

Seven models added in one pass. Six were free — architectures rembed
already runs, needing only ONNX golden validation: multilingual-e5-small
(fp32 1.7e-7 — the e5 family is now VALIDATED, not just "expected
compatible"), bge-base-en-v1.5 (7.4e-7, cls), gte-base (3.8e-6),
paraphrase-mpnet-base-v2 (1.2e-6), multi-qa-MiniLM-L6-cos-v1 (2.1e-7),
snowflake-arctic-embed-s (2.5e-7, cls). The seventh is a new (fourth)
architecture: DistilBERT — same post-LN flow as BERT with renamed
tensors and config keys, no token-type table —
multi-qa-distilbert-cos-v1 at 2.5e-7. Weight-only int8 bounds measured
and matrix-enforced for all seven.

Two real fixes fell out of the expansion, both caught by goldens:

- The WordPiece tokenizer kept Cf format characters that HF's
  _clean_text drops — most visibly U+200C ZWNJ, which Persian uses
  INSIDE words («می‌کند»). The distilbert golden's Persian case failed
  with ids-match=false and a 0.0277 embedding error while
  last_hidden_state matched at 5e-6 — the numerics were perfect and the
  tokenizer was the culprit, exactly the attribution the golden design
  intends. Fixed to exact _clean_text parity (\t\n\r split; all other
  Cc/Cf/Co/Cs deleted; Cn KEPT — the review's differential run showed
  Go's combined unicode.C table would delete unassigned codepoints HF
  keeps, i.e. anything newer than the toolchain's Unicode tables).
  Persian ids now match on every WordPiece model, with per-model fp32
  ranging 1.0e-7 (e5) to 3.8e-6 (gte-base) on the Persian case.
- paraphrase-mpnet-base-v2 writes its special tokens in AddedToken
  object form ({"content": "<s>", …}); both convert.py and the Go
  derivation now unwrap it — a long-standing backlog item closed by the
  first repo that actually shipped the form.

## Performance investigation (profile-driven, 2026-08)

Profiled end-to-end forward passes (BenchmarkEmbedProfile: serial, ~120
tokens, `-cpuprofile`) on an AVX2+AVX-VNNI (no AVX-512) box, encoder
(all-mpnet-base-v2) and decoder (Qwen3-Embedding-0.6B).

- **AVX2 GELU — SHIPPED, ~12%.** GELU was 9% of the encoder forward as a
  scalar Go polynomial. A vectorized kernel (gelu_amd64.s: 8-wide erf +
  Cephes exp) cut mpnet serial forward 270 ms -> 238 ms (~12%), validated
  bit-close to the scalar (<= 1e-6) and against every GELU-model golden.
- **Matmul prefetch — REJECTED (measured wash).** gemm4x16 is the biggest
  single hotspot (~58% encoder), but adding PREFETCHT0 to its k-loop
  measured 269 vs 270 ms — no change. The kernel is FMA-port-bound (its
  design saturates two FMA ports), not memory-bound, so prefetch cannot
  help; reverted rather than add a dead instruction. Going wider needs
  AVX-512 (the gemm4x32/VNNI-512 kernels already exist for CPUs that have
  it; this box does not).
- **Packed attention kernel — REJECTED (measured 1.65x slower).** The
  decoder spends ~34% in dot4 (attention Q·Kᵀ and probs·V via
  MatMulSerial), which has far fewer MACs than the projections yet takes
  more time — suggesting the packed gemm4x16 might win. A micro-benchmark
  on the real shape (m=120,k=120,n=128) refuted it: dot4 ~70 us vs packed
  gemm4x16 + per-call PackB ~116 us. Attention weights (K/V) are computed
  per forward and cannot be pre-packed, so the packing cost outweighs the
  kernel's efficiency (projections only look efficient because they pack
  once at load and amortize it). dot4 is the correct kernel for
  attention's small, non-reusable matmuls.
- **SiLU / RMSNorm / RoPE — not worth vectorizing.** Each < 1.5% of the
  decoder forward; the profile cleared them despite an obvious-looking
  float64 math.Exp in SiLU.

Net: on AVX2, the matmul is at the compute roofline and attention is on
the right kernel; GELU was the one remaining vectorization win.

## Cross-engine refresh — nomic-embed + baseline (2026-08, contended 20-core box)

Re-running the us-vs-ONNX protocol (bench/compare.py, both-orders,
median-of-40, warm-up discard) after the new architectures landed,
specifically on **nomic-embed-text-v1.5** — the first encoder combining
RoPE + gated SwiGLU + bidirectional attention, a harder shape than the
plain-BERT encoders R8 measured.

**Environment caveat.** This box runs a clickhouse-server, so 20-thread
runs are noise-dominated: rembed's spinning fork-join pool oversubscribes
against the other load (fp32 nomic multi-thread measured 63% spread and a
35% order-median drift — the harness FLAGGED it as unstable and refused to
call it), while ONNX Runtime's pool parks and stayed at ~6% spread. That
is a real, documented property — rembed's spin pool assumes a quiet/
dedicated box (R8's Zen4 cloud numbers) — not a kernel result. So the
numbers below are **single-core** (GOMAXPROCS=1 vs onnx-threads=1), which
both removes the scheduler artifact and isolates raw kernel efficiency;
all four runs were stable (spread <= 8%, no instability flags).

| model | mode | rembed | ONNX (fp32) | ratio |
|-------|------|--------|-------------|-------|
| all-MiniLM-L6-v2 (seq 93) | fp32 | 49.7 ms | 39.5 ms | 1.26× (slower) |
| nomic-embed-text-v1.5 (seq 93) | fp32 | 451 ms | 352 ms | 1.28× (slower) |
| nomic-embed-text-v1.5 (seq 93) | **full int8 (VNNI)** | **243 ms** | 352 ms | **0.69× (1.45× faster)** |

Reading it honestly:

- **Single-core fp32: ONNX Runtime is ~1.26–1.28× ahead**, and it is
  consistent across a plain BERT and the RoPE+SwiGLU encoder — so it is
  ORT's per-core MLAS GEMM microkernels being better-tuned, not an
  architecture-specific rembed weakness. This does NOT contradict R8's
  encoder "parity": that was measured MULTI-core on a quiet box, where
  rembed closes the per-core gap through parallelism. We could not
  re-verify the multi-core parity here (the box is contended; see above).
- **Full int8 (u8×s8 VNNI) beats ONNX fp32 by ~1.45×** on nomic, stable
  and flag-free, reproducing R8's headline that full int8 clears ORT fp32.
  The accuracy trade is real and per-model (nomic full-int8 worst cosine
  0.953 — see the README int8 table); it is opt-in for exactly that reason.

Net: at matched precision and per core, ORT's fp32 kernels lead ~1.26×;
rembed reaches fp32 parity via multi-core scaling (quiet box) and passes
ORT outright when the int8 accuracy trade is acceptable. Decoders (qwen3,
gemma) are not in this table — their ONNX exports are gated or not shipped,
so a like-for-like cross-engine decoder number needs an export step this
run did not take.
