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

BERT-family, MPNet, and RoBERTa encoders in sentence-transformers
format: mean or CLS pooling; WordPiece, byte-level BPE, or SentencePiece
Unigram tokenization (the XLM-R tokenizer — multilingual models work,
50+ languages); absolute positions (plus MPNet's bucketed
relative-position attention bias); exact GELU; F32/F16/BF16 safetensors.
Validated end-to-end against each model's own ONNX Runtime reference:

| model | pooling | dtype | fp32 vs ONNX | int8 |
|-------|---------|-------|--------------|------|
| sentence-transformers/all-MiniLM-L6-v2 | mean | F32 | 1.5e-7 | cosine ≥ 0.9991 |
| sentence-transformers/all-MiniLM-L12-v2 | mean | F32 | 1.9e-7 | in bounds |
| sentence-transformers/paraphrase-MiniLM-L3-v2 | mean | F32 | < 1e-4 | — |
| BAAI/bge-small-en-v1.5 | cls | F32 | < 1e-4 | in bounds |
| sentence-transformers/all-mpnet-base-v2 | mean | F32 | 3.3e-7 | cosine ≥ 0.9978 |
| sentence-transformers/all-distilroberta-v1 | mean | F32 | 3.2e-7 | cosine ≥ 0.9985 |
| sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2 | mean | F32 | 7e-7 | cosine ≥ 0.9995 |
| thenlper/gte-small | mean | F16 | 2e-3 maxAbs + cosine ≥ 0.9999 + meanAbs ≤ 2e-4 (the repo's ONNX export is fp32 while its safetensors are f16, so maxAbs is dominated by the checkpoint's own rounding; the cosine/mean bounds are what actually constrain rembed) | — |

Expected compatible (same architecture, no ONNX export on the Hub to
validate against): the e5 family, larger BGE/GTE sizes, and other
BERT-based sentence-transformers checkpoints. Caveat for retrieval
models: e5 requires "query: "/"passage: " prefixes and some models
(e.g. arctic) declare prompt handling in their pooling config — rembed
embeds exactly the text you pass and does not add prefixes; add them
yourself or retrieval quality silently degrades (multilingual-e5 is the
main SentencePiece model in this category).

## Dev: golden reference generation

The validation harness's golden files come from ONNX Runtime in Python
(this is a dev-time tool; users never need it):

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
