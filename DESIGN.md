# rembed — design

Pure-Go embedding inference engine: text → embedding vectors for BERT-family
encoder models. No cgo, no ONNX Runtime, one static binary. It is both a
learning project (the whole transformer forward pass implemented from scratch)
and a performance project (an explicitly measured optimization ladder toward
ONNX-Runtime-competitive speed).

Eventually it should drop in behind Rostam's `-tags localembed` build and
replace the ONNX Runtime dependency; until then it stands alone.

**Non-goals (for now):** decoder/generative LLMs, GPU, training. int8 and
multi-model support come after v1.

## Public API

```go
emb, err := rembed.Load(modelDir)          // *rembed.Embedder
vecs, err := emb.Embed(ctx, []string{...}) // [][]float32, L2-normalized
emb.Model()                                // model name from the manifest
emb.Dim()                                  // embedding dimensionality
```

The interface deliberately mirrors what Rostam's localembed feature needs so
the engine can later slot in behind it:
`Embed(ctx, texts []string) ([][]float32, error)`, `Model() string`, `Dim() int`.

## Package layout

    rembed/                 public API (Load / Embedder)
      internal/tensor/      math kernels (matmul, softmax, layernorm, gelu, add)
                            — the perf-critical core
      internal/model/       BERT encoder: embeddings(token+pos+segment)
                            → N layers (attn + FFN, post-LN) → pool → L2 normalize
      internal/safetensors/ weight loader (JSON header + raw little-endian blobs)
      tokenizer/            pure-Go WordPiece (ported from rostam)
      cmd/rembed/           CLI: embed | bench | validate
      bench/                benchmark + validation harness
      models/convert.py     one-time HuggingFace → model-dir export + manifest
      testdata/             golden ONNX-reference outputs for fixed inputs

## Key decisions (settled — do not re-litigate)

1. **safetensors, not ONNX, for weights.** safetensors is a JSON header plus
   raw f32 blobs — trivial to parse in Go. `models/convert.py` produces the
   model dir once (HuggingFace already ships `model.safetensors` for the
   target models; the converter downloads, verifies, and writes a manifest).
   We do NOT parse ONNX protobuf.
2. **Pluggable kernel interface** is the heart of the perf plan: ONE matmul
   signature with swappable bodies (naive → cache-blocked → SIMD → int8),
   each behind its own benchmark. Kernels swap without touching model code,
   so every rung of the ladder is A/B-testable in isolation.
   The signature is `MatMul(dst, a, bT []float32, m, k, n int)` computing
   `C[m×n] = A[m×k] · Bᵀ` with `bT` laid out `[n×k]` row-major. That is
   exactly how HuggingFace stores `nn.Linear` weights (`[out, in]`), so
   linear layers need no transpose at load, and the naive body reduces to
   dot products of two contiguous rows (cache-friendly from day one).
3. **The WordPiece tokenizer is ported, not rewritten**, from
   `github.com/rostamlabs/rostam` branch `feat/localembed-release`, package
   `semcache/local` (tokenizer.go). It handles whitespace/punct splitting
   (IsSpace checked before IsControl), greedy longest-match WordPiece with
   `##` continuations, [CLS]/[SEP]/[UNK], configurable special tokens, and
   truncation to 512 tokens. Its unit tests come along in the port.

## Resolved in this session (small choices the kickoff left open)

- Module path `github.com/rostamlabs/rembed`, Go 1.26, **zero third-party Go
  dependencies** for v1 (avo appears at M3 as a build-time codegen tool only).
- v1 embeds each text independently (no cross-text padding/batching): the
  sequence length is the text's own token count, which matches the per-text
  ONNX reference run exactly and keeps masking trivial. Batching across texts
  is an optimization to revisit with data, not part of correctness.
- safetensors loader supports F32 only for v1 and validates header offsets
  against the file size.
- The model dir is `{model.safetensors, vocab.txt, manifest.json}`.
  `manifest.json` (written by convert.py) records: model name, hidden size,
  layers, heads, intermediate size, vocab size, max position embeddings,
  do_lower_case, special tokens, pooling strategy, normalize flag.
- Golden reference comes from ONNX Runtime **in Python** (models ship
  `onnx/model.onnx` on HuggingFace): convert.py runs fixed inputs through
  onnxruntime, applies mean pooling + L2 normalize in numpy, and writes
  `testdata/golden.json` with, per input: the text, the HF-tokenizer input
  ids, and the final embedding. Storing the ids separately lets a failure be
  attributed to the tokenizer vs the numerics immediately. Model weights
  (~90 MB) are never committed; tests that need them skip when the model dir
  is absent.

## The optimization ladder

Each rung is a milestone, measured against the previous rung AND against the
ONNX Runtime baseline. One milestone per PR, each with a benchmark delta.

- **M0 — correctness.** Naive fp32 forward pass; output matches the ONNX
  reference within 1e-4. Record the baseline latency. Build the full
  benchmark + validation harness at this rung, before optimizing anything.
- **M1 — alloc-free + cache-blocked matmul.** Target ~3–5× over M0.
- **M2 — goroutine parallelism** (across heads / matmul row-blocks).
  Target ~2–4×.
- **M3 — SIMD matmul, AVX2** (github.com/mmcloughlin/avo or hand-written
  plan9 asm). The big lever; target within ~2–3× of ONNX Runtime.
- **M4 — int8 quantization.** Parity-ish with ONNX int8, big size win.

## Benchmark + validation harness (built at M0)

A perf project is only as honest as its harness, and laptop numbers lie
(P/E-core lottery, thermal throttling, fixed-order runs, tiny-sample medians
under large run-to-run variance). Rules:

- **Validation:** golden ONNX-reference outputs for fixed inputs live in
  testdata; every kernel or model change must still match within tolerance
  (1e-4 max abs difference). This is the safety net that makes fearless
  optimization possible.
- **Benchmark rigor:** warm-up discard; medians over N runs; order-reversal
  control (run us-then-ONNX AND ONNX-then-us); report variance, not just the
  median; ONNX Runtime as the constant baseline; FLAG any delta that is
  within measurement noise.
- Laptop numbers are relative-only/provisional. Pin to a dedicated cloud box
  for any publishable us-vs-ONNX claim (reach for that around M3).
- `go test -bench` for kernel micro-benchmarks plus an end-to-end embed
  benchmark.

## v1 first slice

MiniLM-L6-v2 (`sentence-transformers/all-MiniLM-L6-v2`: 6 layers, 384-dim,
12 heads, GELU, post-LN, mean pooling, L2 normalize), fp32, CPU. Deliver
M0 → M2 with the harness. SIMD (M3) and int8 (M4) follow once the foundation
is solid.
