# Architecture

rembed implements the whole transformer forward pass in Go — embeddings, encoder
layers, pooling, normalization — and reaches ONNX-Runtime-competitive speed
through a measured optimization ladder, every rung validated against a golden
reference and A/B-tested in isolation. This page describes how it's built; the
numbers are in [Benchmarks](benchmarks.md).

## The forward pass

`internal/model/` holds the encoder: token + position (+ segment) embeddings →
N layers of attention + feed-forward → pooling → L2 normalization. Each text is
embedded **independently**, so a text's sequence length is exactly its own token
count. That keeps masking trivial and makes the output match a per-text ONNX
reference exactly, and it's why a batch fans out *across* texts with no padding
waste rather than padding a batch to a common length.

rembed spans eight architectures with their variations: absolute and rotary
(RoPE) positions, bidirectional / sliding-window / causal attention with GQA and
QK-norm, GELU / GeGLU / SwiGLU activations, LayerNorm and RMSNorm, and mean /
CLS / last-token pooling — with an optional two-layer Dense projection head
(EmbeddingGemma). See [Supported models](models.md).

## Weights: safetensors, not ONNX

rembed reads **safetensors** — a JSON header plus raw little-endian float blobs,
trivial to parse in Go — never ONNX protobuf. `models/convert.py` produces a
model directory once (`model.safetensors`, `vocab.txt`, `manifest.json`); the
`manifest.json` records the architecture's shape (hidden size, layers, heads,
pooling strategy, normalize flag, special tokens, …) so the Go loader needs no
framework. Loading straight from a Hugging Face checkout is also supported, with
the manifest derived on the fly.

## The pluggable kernel interface

The heart of the performance plan is a single matmul signature with swappable
bodies, each behind its own benchmark:

```
MatMul(dst, a, bT []float32, m, k, n int)   // C[m×n] = A[m×k] · Bᵀ,  bT laid out [n×k] row-major
```

That `Bᵀ` layout `[out, in]` is exactly how Hugging Face stores an `nn.Linear`
weight, so linear layers need no transpose at load — the naive body reduces to
dot products of two contiguous rows. Because every kernel implements one
interface, a ladder rung (naive → cache-blocked → SIMD → int8) can be swapped in
and A/B-tested without touching the model code. A second, packed-weight interface
was added once packing weights at load became the structural win.

## The optimization ladder

Each rung targets ONNX Runtime and is measured against it (reference laptop, an
i9-13900H). Targets vs measured:

| Rung | What it added | Measured |
|---|---|---|
| **M0** | Naive fp32 forward pass; build the validation + benchmark harness | matches ONNX within `1e-4` (`2.9e-7`); ~20× slower single-thread |
| **M1** | Alloc-free, cache-blocked matmul | 1.81× (the scalar-Go ceiling) |
| **M2** | Goroutine parallelism (heads / matmul row-blocks) | 4.1× |
| **M3** | AVX2 SIMD matmul (hand-written plan9 asm) | 3.3× |
| **M4** | Packed GEMM (`gemm4x16`), spinning fork-join pool, fast-math exp/erf | **parity, 1.05×** |
| **int8 (M5)** | Weight-only int8 | **ahead** of ORT fp32 (0.89×) and ORT quint8 (0.95×) |

Beyond the core ladder, an R-series added breadth and further kernels: ARM64
NEON (R3), the MPNet / RoBERTa+BPE / SentencePiece / ModernBERT / Qwen3 /
XLM-RoBERTa / EmbeddingGemma / nomic models (R4–R14), AVX-VNNI full int8 (R7),
and disk-backed weights (R11) — plus a 512-bit-wide fp32 kernel (`gemm4x32`) and
a `gemm6x16` micro-kernel that closed most of the residual single-core fp32 gap.

## int8 quantization

- **Weight-only** (`WithInt8`) — dense weights are quantized at load to
  per-output-channel symmetric int8; activations stay fp32, and `gemm4x16i8`
  dequantizes in-register with one per-channel multiply in the epilogue. The win
  is memory traffic: a `seq=12` embed streams ~42 MB of fp32 weights; int8 cuts
  that to ~10.5 MB, which fits in L3.
- **Full int8** (`WithInt8Activations`) — weights as above, plus activations
  quantized per-row to `u8` via the `+128` offset trick, so `VPDPBUSD` computes
  `u8·s8` as the true dot plus a precomputed `128·colSum` correction subtracted
  in the epilogue. One instruction does four multiply-accumulates per `int32`
  lane. Kernels exist for AVX-VNNI (256-bit, VEX-encoded) and AVX-512-VNNI
  (`gemm4x16vnni512`, EVEX); both are bit-exact against a pure-Go integer
  reference. The `k > 131072` overflow limit is ~85× past any transformer.

Both int8 paths fall back silently (int8-activations → weight-only → fp32) when
the CPU or a matrix shape can't take them — query `Quantized()` /
`QuantizedActivations()` when the mode matters.

## Correctness: the golden-ONNX harness

Built at M0, before any optimization. `models/convert.py` runs fixed inputs
through ONNX Runtime in Python, applies the manifest's pooling + L2 normalization
in numpy, and writes `testdata/golden.json`: per input, the text, the
Hugging-Face-tokenizer `input_ids`, and the final embedding. Storing the ids
separately lets a failure be attributed to the **tokenizer** vs the
**numerics**. The input set covers accents, non-ASCII symbols, and CJK.

Every kernel and model change must match within **`1e-4`** max absolute
difference, in CI. Model weights (~90 MB) are never committed — tests skip when
the model dir is absent. SIMD reductions accumulate in a different order than
scalar, so SIMD paths are held to a relative tolerance rather than bit-identity,
but still match the golden within `1e-4`; int8 is bounded per-model by
test-enforced cosine floors (weight-only ≥ 0.988 across models; full int8 lower
on some checkpoints — see [choosing a quantization
mode](models.md#choosing-a-quantization-mode)).

## Package layout

`rembed/` (public API) · `internal/tensor/` (matmul, softmax, layernorm, gelu) ·
`internal/model/` (the encoder) · `internal/safetensors/` (weight loader) ·
`internal/hub/` (pure-Go Hugging Face fetch, `sha256`-verified) ·
`internal/packfile/` + `internal/mmapfile/` (disk-backed weights) · `tokenizer/`
(+ `bpe`, `sentencepiece`, `gemma` subpackages) · `cmd/rembed/` (CLI) · `bench/`
(harness) · `models/convert.py` · `testdata/` (goldens).
