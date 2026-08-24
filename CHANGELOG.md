# Changelog

All notable changes to rembed are recorded here. rembed is a pure-Go,
cgo-free embedding inference engine validated end-to-end against each
model's own independent reference (ONNX Runtime, or the canonical PyTorch
model where an ONNX export is unavailable).

## Unreleased

- **nomic-embed** (`nomic-embed-text-v1.5`): the eighth architecture — a
  post-norm BERT encoder with RoPE, gated SwiGLU, and token_type
  embeddings, bias-free. Validated against its own ONNX export at fp32
  maxAbs < 1e-4.

## v0.3.0

Seven architectures, three tokenizer families, weight-only and full int8,
run-larger-than-RAM disk weights, and a pure-Go Hugging Face loader — all
golden-validated within 1e-4 of the reference (token-for-token for the
tokenizers).

### Added — architectures

- **MPNet** (`all-mpnet-base-v2`): offset positions, shared bucketed
  relative-position bias.
- **RoBERTa** + byte-level BPE tokenizer (`all-distilroberta-v1`): the
  fairseq position offset on the BERT encoder.
- **SentencePiece Unigram** tokenizer — multilingual encoders
  (`multilingual-e5-small`, `paraphrase-multilingual-MiniLM`): the XLM-R
  NMT-NFKC normalizer + Viterbi segmentation, matched to the C++ reference.
- **ModernBERT** (`modernbert-embed-base`): dual-theta RoPE, alternating
  global/local sliding-window attention, pre-norm bias-free GeGLU.
- **Qwen3-Embedding** (`Qwen3-Embedding-0.6B`+): the first decoder
  embedder — causal attention, RMSNorm, per-head QK-norm, grouped-query
  attention, SwiGLU, last-token pooling.
- **XLM-RoBERTa** model_type (`multilingual-e5-base`/`-large`, `bge-m3`):
  the RoBERTa encoder with the SentencePiece tokenizer.
- **EmbeddingGemma** (`embeddinggemma-300m`): a bidirectional Gemma 3
  backbone — unit-offset RMSNorm, four-LayerNorm sandwich, dual-theta RoPE,
  QK-norm + GQA, tanh-GELU GeGLU, a two-layer Dense head, and a new
  byte-fallback BPE tokenizer family.

### Added — features

- **`WithDim(d)`** / CLI `-dim`: Matryoshka truncation (EmbeddingGemma
  768→512/256/128), slice + re-normalize, validated at each dim.
- **`WithDiskWeights()`**: memory-mapped weights from a pack file — run a
  model larger than RAM (proven on Qwen3-4B), resident memory tracks the
  working set. Sharded safetensors supported.
- **`EmbedTokens`**: token-level output (the reference `last_hidden_state`),
  unpooled and unnormalized.
- **Batched throughput**: parallelism fans out across texts, no padding.
- **Pure-Go Hugging Face loader**: fetches straight from the Hub (no Python,
  no conversion), deriving the config in Go. **HF-token support** (`HF_TOKEN`
  / `HUGGING_FACE_HUB_TOKEN`, or the `hf login` token file) so gated repos
  (e.g. EmbeddingGemma) download too.

### Performance

- **Full int8 on AVX-VNNI** (`WithInt8Activations`): u8 activations × s8
  weights via VPDPBUSD, a further ~1.3× over weight-only int8; per-model
  accuracy bounds are test-enforced.
- **AVX2 GELU kernel**: vectorized erf + Cephes exp, ~12% faster encoder
  forward.
- **Causal flash-attention** (Qwen3 decoder): tiled online softmax that
  skips the causal upper triangle, ~2.1× the decoder forward, fp32 golden
  unchanged.
- **Cross-engine benchmark** vs ONNX Runtime: rembed fp32 at parity, full
  int8 ahead on the measured rounds (see `bench/RESULTS.md`).

### Notes

- fp32 encoders are at ONNX-Runtime parity at matched precision; int8 modes
  are opt-in with test-enforced accuracy bounds — check `Quantized()` /
  `QuantizedActivations()` when the mode matters.
- Serial and parallel forwards are bit-identical.

## v0.1.0

Initial release: BERT/DistilBERT sentence-transformers embedders,
mean/CLS pooling, WordPiece tokenizer, weight-only int8, and the golden
validation harness against ONNX Runtime.
