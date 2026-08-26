# rembed

**Pure-Go embedding inference for text-embedding models — BERT-style encoders
and decoder-derived embedders.** Text in, L2-normalized embedding vectors out —
no cgo, no ONNX Runtime, no Python in the path, one static binary.

```go
emb, err := rembed.Load("sentence-transformers/all-MiniLM-L6-v2") // downloads from the HF Hub, pure Go
vecs, err := emb.Embed(ctx, []string{"hello world"})              // vecs[0] is a []float32 of emb.Dim() (384)
```

rembed implements the whole transformer forward pass in Go and reaches
**statistical parity with ONNX Runtime on CPU** — and, with weight-only int8,
consistently ahead of it — across a measured optimization ladder validated
against a golden ONNX reference within `1e-4`. It is both a from-scratch
implementation and a performance project; the honest measurements, including the
experiments that were rejected, are in [Benchmarks](benchmarks.md).

## Why

- **No native dependency.** ONNX Runtime is a large cgo/shared-library
  dependency that must be installed and version-matched at runtime. rembed is
  ordinary Go: `go get`, cross-compile, ship one static binary. Its only
  dependencies are `golang.org/x/sys` and `golang.org/x/text`.
- **Models load straight from the Hugging Face Hub** in pure Go (cached locally,
  `HF_TOKEN` honored) — or from a local directory. No conversion step, no
  Python export.
- **Competitive speed.** Measured parity with ONNX Runtime for fp32 on CPU, and
  ahead of it with int8, via hand-written AVX2 / AVX-VNNI kernels behind a
  pluggable interface. See [Architecture](architecture.md).
- **Correct.** Every model and every kernel is validated against an independent
  ONNX Runtime (or PyTorch) reference within `1e-4` max absolute difference, in
  CI. [20 models across 8 architectures](models.md) ship with committed goldens.

## Quickstart

=== "Go library"

    ```go
    package main

    import (
        "context"
        "fmt"

        "github.com/rostamlabs/rembed"
    )

    func main() {
        emb, err := rembed.Load("sentence-transformers/all-MiniLM-L6-v2")
        if err != nil {
            panic(err)
        }
        defer emb.Close()

        vecs, err := emb.Embed(context.Background(), []string{
            "the cat sat on the mat",
            "quantum chromodynamics",
        })
        if err != nil {
            panic(err)
        }
        fmt.Println(len(vecs), emb.Dim()) // 2 384
    }
    ```

=== "CLI"

    ```sh
    go install github.com/rostamlabs/rembed/cmd/rembed@latest

    rembed embed -model sentence-transformers/all-MiniLM-L6-v2 "some text"
    ```

=== "HTTP server (OpenAI-compatible)"

    ```sh
    rembed serve -model sentence-transformers/all-MiniLM-L6-v2 -addr :8080

    curl -s localhost:8080/v1/embeddings \
      -H 'content-type: application/json' \
      -d '{"input": ["hello world"]}'
    ```

=== "Python"

    ```python
    from rembed import Embedder

    emb = Embedder("models/all-MiniLM-L6-v2")
    vecs = emb.embed(["hello world"])   # (n, dim) float32 numpy array
    ```

## Feature highlights

- **`Embed`** — one L2-normalized vector per text; a batch fans out across texts
  for near-linear throughput, bit-identical to one-at-a-time.
- **`EmbedTokens`** — per-token hidden states (ONNX Runtime's
  `last_hidden_state`) for rerankers and late-interaction retrieval.
- **Weight-only int8** (`WithInt8`) — ~4× less weight traffic, cosine ≥ 0.999 vs
  fp32; **full int8** (`WithInt8Activations`) on AVX-VNNI CPUs.
- **Matryoshka** (`WithDim`) — truncate + re-normalize (e.g. EmbeddingGemma
  768→512/256/128) for cheaper storage and search.
- **Disk-backed weights** (`WithDiskWeights`) — mmap a model larger than RAM.
- **CPU cap** (`WithWorkers`) — tune latency vs throughput for servers.

See the [Go library](usage.md) reference for the full API, the
[CLI & server](cli.md) for command-line and HTTP use, [Python & C
bindings](python.md) for foreign callers, and [Supported models](models.md) for
the validated model matrix.

## Rostam

rembed is the pure-Go local embedder behind [Rostam](https://docs.rostamlabs.com/)'s
in-process embeddings (`ROSTAM_EMBED_LOCAL`). See [Rostam
integration](rostam-integration.md).

## License

Apache-2.0.
