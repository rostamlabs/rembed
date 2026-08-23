# rembed

Pure-Go embedding inference engine for BERT-family encoder models.
Text in, L2-normalized embedding vectors out — no cgo, no ONNX Runtime,
one static binary.

```go
// Loads straight from the Hugging Face Hub (pure Go, cached locally) —
// no Python, no conversion step:
emb, err := rembed.Load("sentence-transformers/all-MiniLM-L6-v2")
vecs, err := emb.Embed(ctx, []string{"hello world"})
// vecs[0] is a []float32 of emb.Dim() (384 for MiniLM-L6-v2)
```

`EmbedTokens` returns per-token hidden states (ONNX Runtime's
`last_hidden_state`) for rerankers and late-interaction retrieval, and a
multi-text `Embed` call fans out across texts for near-linear batch
throughput (bit-identical to one-at-a-time results).

`Load` accepts a Hub model id (downloaded into `$REMBED_CACHE`, default
the user cache dir; `HF_TOKEN` honored), a git-cloned HF repo directory,
or a converted model dir. Options: `rembed.WithInt8()` (weight-only
quantization, ~4× less weight traffic, cosine ≥ 0.999 vs fp32) and
`rembed.WithWorkers(n)` (CPU cap for servers).

Status: the optimization ladder is complete — naive baseline to
**statistical parity with (and, with int8, consistently ahead of) ONNX
Runtime** on the reference laptop: ~45× → 0.89× across six rungs, every
step measured against a golden ONNX reference within 1e-4 (int8: cosine
≥ 0.999). See [DESIGN.md](DESIGN.md) for the architecture and
[bench/RESULTS.md](bench/RESULTS.md) for the full measured ladder,
including the failed experiments. Weight-only int8 is opt-in via
`rembed.WithInt8()`; `rembed.WithWorkers(n)` caps per-call CPU for
throughput-saturated servers.

## Supported models

Six architectures: BERT-family, DistilBERT, MPNet, RoBERTa, and
ModernBERT encoders, plus **Qwen3-Embedding** — a causal *decoder*
embedder (the current state of the art for retrieval). sentence-
transformers format: mean, CLS, or last-token pooling; WordPiece,
byte-level BPE, or SentencePiece Unigram tokenization (the XLM-R
tokenizer — multilingual models work, 50+ languages); absolute positions
(plus MPNet's bucketed relative-position bias) OR rotary positions (RoPE,
ModernBERT and Qwen3); alternating global/local sliding-window attention
(ModernBERT) or full causal attention with grouped-query attention and
QK-norm (Qwen3); exact GELU, ModernBERT's GeGLU, and Qwen3's SwiGLU;
LayerNorm and RMSNorm; F32/F16/BF16 safetensors. Validated end-to-end
against each model's own ONNX Runtime reference (ModernBERT and Qwen3
against the canonical PyTorch `ModernBertModel` / `Qwen3Model`, since
their ONNX exports bundle or omit the pooling rembed reproduces):

| model | pooling | dtype | fp32 vs ONNX | int8 |
|-------|---------|-------|--------------|------|
| sentence-transformers/all-MiniLM-L6-v2 | mean | F32 | 1.5e-7 | cosine ≥ 0.9991 |
| sentence-transformers/all-MiniLM-L12-v2 | mean | F32 | 1.9e-7 | in bounds |
| sentence-transformers/paraphrase-MiniLM-L3-v2 | mean | F32 | < 1e-4 | — |
| BAAI/bge-small-en-v1.5 | cls | F32 | < 1e-4 | in bounds |
| sentence-transformers/all-mpnet-base-v2 | mean | F32 | 3.3e-7 | cosine ≥ 0.9978 |
| sentence-transformers/all-distilroberta-v1 | mean | F32 | 3.2e-7 | cosine ≥ 0.9985 |
| sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2 | mean | F32 | 7e-7 | cosine ≥ 0.9995 |
| intfloat/multilingual-e5-small | mean | F32 | 1.7e-7 | cosine ≥ 0.9995 |
| BAAI/bge-base-en-v1.5 | cls | F32 | 7.4e-7 | cosine ≥ 0.995 |
| thenlper/gte-base | mean | F32 | 3.8e-6 | cosine ≥ 0.988 |
| sentence-transformers/paraphrase-mpnet-base-v2 | mean | F32 | 1.2e-6 | cosine ≥ 0.9945 |
| sentence-transformers/multi-qa-MiniLM-L6-cos-v1 | mean | F32 | 2.1e-7 | cosine ≥ 0.998 |
| Snowflake/snowflake-arctic-embed-s | cls | F32 | 2.5e-7 | cosine ≥ 0.995 |
| sentence-transformers/multi-qa-distilbert-cos-v1 | mean | F32 | 2.5e-7 | cosine ≥ 0.999 |
| nomic-ai/modernbert-embed-base | mean | F32 | < 1e-4 (vs PyTorch) | cosine ≥ 0.998 |
| Qwen/Qwen3-Embedding-0.6B | lasttoken | BF16 | < 1e-4 (vs PyTorch) | cosine ≥ 0.997 |
| thenlper/gte-small | mean | F16 | 2e-3 maxAbs + cosine ≥ 0.9999 + meanAbs ≤ 2e-4 (the repo's ONNX export is fp32 while its safetensors are f16, so maxAbs is dominated by the checkpoint's own rounding; the cosine/mean bounds are what actually constrain rembed) | — |

On CPUs with AVX-VNNI — Intel Alder Lake (2021) onward and Sapphire
Rapids+ servers, AMD Zen 5+; note that AVX-512-VNNI-only parts like Ice
Lake-SP and Zen 4 do NOT have it — `WithInt8Activations` selects full
int8 inference (u8 activations × s8 weights via VPDPBUSD) for a further
~1.3× over weight-only int8. The accuracy trade is real, PER-MODEL, and
test-enforced (worst golden cosine, full int8 vs weight-only):

| model | full int8 | weight-only int8 |
|-------|-----------|------------------|
| MiniLM-L6 / L12 / L3 | 0.9917 / 0.9932 / 0.9979 | ≥ 0.9990 |
| mpnet-base / paraphrase-mpnet | 0.9912 / 0.9867 | ≥ 0.9945 |
| multilingual MiniLM / multilingual-e5 | 0.9982 / 0.9988 | ≥ 0.9995 |
| gte-small / gte-base | 0.9991 / 0.9741 | ≥ 0.9880 |
| multi-qa MiniLM / distilbert | 0.9949 / 0.9854 | ≥ 0.9940 |
| arctic-embed-s | 0.9932 | 0.9953 |
| distilroberta | 0.9747 | 0.9987 |
| modernbert-embed | 0.9660 | 0.9984 |
| qwen3-embedding-0.6B | 0.9747 | 0.9978 |
| **bge-base** | **0.9593** | 0.9957 |

Activation outliers are a PER-CHECKPOINT property, not an architecture
one: bge-base (a plain BERT) measures worst of all at 0.9593, below
distilroberta's 0.9747 and modernbert-embed's 0.9660 (whose GeGLU gate
activations have a range the per-row u8 scale can't hold), while
bge-base's sibling bge-small is unremarkable. Qwen3-Embedding compounds
this: last-token pooling reads a single position, so there is no
averaging across tokens to soften activation-quantization error — prefer
`WithInt8` (weight-only) there.
Check the table before enabling full int8 for a model — anything below
~0.99 is a real retrieval-quality risk — and prefer `WithInt8`
(weight-only, ≥ 0.988 everywhere) when in doubt. Every figure above is
enforced in the golden matrix.

Cross-engine, measured on a Zen 4 cloud box with a both-orders/median
protocol (bench/RESULTS.md has the full data and every noise flag):
rembed fp32 sits at parity with ONNX Runtime fp32, and rembed full int8
beat ORT fp32 in every round — the flag-free rounds measured 0.70× and
0.75× (5.9 ms vs 7.9 ms on mpnet) — while trading blows at parity with
ORT's own AVX-512-VNNI int8 graphs.

**Disk-backed weights (run larger than RAM).** `WithDiskWeights()`
memory-maps the weights from a pack file instead of loading them into RAM:
the OS pages weights in on access and evicts under pressure, so resident
memory tracks the working set and a model larger than RAM runs
(disk-bandwidth-bound when it does not fit, full speed with a warm page
cache when it does — the same trade ORT's mmap mode makes). On first use
the safetensors (single-file or sharded) are streamed to a pack file one
tensor at a time, so even the pack step fits a small box. This is what
lets Qwen3-Embedding-4B run cgo-free on a laptop that cannot hold it in
fp32 RAM. Close the Embedder to unmap. Numerics are unchanged — only
where the bytes live. (Currently wired for qwen3.)

Expected compatible (same architecture, no committed golden yet): the
remaining e5 sizes, the largest BGE/GTE variants, the msmarco families,
other BERT/DistilBERT-based sentence-transformers checkpoints, and the
larger Qwen3-Embedding sizes (4B/8B — same architecture, far larger).
Caveat for retrieval models: e5 requires "query: "/"passage: " prefixes,
Qwen3-Embedding expects an instruction on queries only
("Instruct: {task}\nQuery:{text}", with documents left bare), and some models
(e.g. arctic) declare prompt handling in their pooling config — rembed
embeds exactly the text you pass and does not add prefixes; add them
yourself or retrieval quality silently degrades. Validated models in
that category: the e5 family ("query: "/"passage: "), bge
("Represent this sentence for searching relevant passages: " on
queries), and arctic-embed (its own query prefix) — rembed embeds
exactly the text you pass.

One deliberate tokenizer divergence: on NFD (decomposed) Hangul/kana —
routine output from macOS — HF's fast tokenizer skips ≥6-byte grapheme
clusters during normalization and shreds Korean into jamo; rembed
matches the sentencepiece C++ reference instead, which composes NFD
back so decomposed and composed text embed identically. 65k-input
fuzzing against the reference: zero mismatches.

## Dev: golden reference generation

The validation harness's golden files come from ONNX Runtime in Python
(ModernBERT from the canonical PyTorch `ModernBertModel` instead; this is
a dev-time tool; users never need it):

```sh
cd models
python3 -m venv .venv && .venv/bin/pip install -r requirements.txt
.venv/bin/python convert.py sentence-transformers/all-MiniLM-L6-v2
```

## Python

The same in-process engine is callable from Python through a C-shared
build (ctypes, ~µs call overhead; the Go library itself stays cgo-free —
the shared object is a separate artifact for foreign callers):

```sh
python/build.sh   # needs a C toolchain; produces python/rembed/librembed.so
```

```python
import sys; sys.path.insert(0, "python")
from rembed import Embedder

emb = Embedder("models/all-MiniLM-L6-v2")           # fp32
emb = Embedder("models/all-MiniLM-L6-v2", int8=True)  # weight-only int8
vecs = emb.embed(["hello world"])                    # (n, dim) float32 numpy
```

Validated against the same golden reference as the Go tests
(`python/test_rembed.py`); vectors cross the ABI bit-identically.

## CLI

```sh
go run ./cmd/rembed embed    -model models/all-MiniLM-L6-v2 "some text"
go run ./cmd/rembed validate -model models/all-MiniLM-L6-v2
go run ./cmd/rembed bench    -model models/all-MiniLM-L6-v2
```

## License

Apache-2.0
