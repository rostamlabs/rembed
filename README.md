# rembed

Pure-Go embedding inference engine for BERT-family encoder models.
Text in, L2-normalized embedding vectors out — no cgo, no ONNX Runtime,
one static binary.

```go
emb, err := rembed.Load("models/all-MiniLM-L6-v2")
vecs, err := emb.Embed(ctx, []string{"hello world"})
// vecs[0] is a []float32 of emb.Dim() (384 for MiniLM-L6-v2)
```

Status: the optimization ladder is complete — naive baseline to
**statistical parity with (and, with int8, consistently ahead of) ONNX
Runtime** on the reference laptop: ~45× → 0.89× across six rungs, every
step measured against a golden ONNX reference within 1e-4 (int8: cosine
≥ 0.999). See [DESIGN.md](DESIGN.md) for the architecture and
[bench/RESULTS.md](bench/RESULTS.md) for the full measured ladder,
including the failed experiments. Weight-only int8 is opt-in via
`rembed.WithInt8()`; `rembed.WithWorkers(n)` caps per-call CPU for
throughput-saturated servers.

## Getting a model

Model weights are not committed. Export one from HuggingFace:

```sh
cd models
python3 -m venv .venv && .venv/bin/pip install -r requirements.txt
.venv/bin/python convert.py sentence-transformers/all-MiniLM-L6-v2
```

This writes `models/all-MiniLM-L6-v2/{model.safetensors, vocab.txt, manifest.json}`
and regenerates `testdata/golden.json` (ONNX Runtime reference outputs used
by the validation harness).

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
