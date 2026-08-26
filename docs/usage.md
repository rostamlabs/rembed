# Go library

```sh
go get github.com/rostamlabs/rembed
```

rembed is an ordinary Go module (`go 1.26.1`); its only dependencies are
`golang.org/x/sys` and `golang.org/x/text`. No cgo, no ONNX Runtime.

## Loading a model

```go
func Load(ref string, opts ...Option) (*Embedder, error)
```

`ref` selects the model:

- **A Hugging Face model id** — `"sentence-transformers/all-MiniLM-L6-v2"`, or
  explicitly `"hf:org/name"` to force the Hub. Files are downloaded straight
  from the Hub in pure Go into the local cache (see [below](#cache-and-tokens))
  and reused on later loads. `sha256`-verified on download.
- **A local directory** — either a converted model dir (`manifest.json`) or a
  plain Hugging Face repo checkout (`config.json` + `1_Pooling/…`, with the
  manifest derived on the fly).

!!! warning "Ambiguity guard — a typo never becomes a network fetch"
    A `ref` that could be *both* — an `org/name` that doesn't exist locally but
    whose first segment *is* a local directory (e.g. a mistyped
    `models/all-MiniLM-L6-v2`) — is treated as a **missing local path** and
    fails, rather than silently reaching out to the Hub carrying your
    `HF_TOKEN`. Use the `hf:` prefix to force a Hub load.

```go
emb, err := rembed.Load("sentence-transformers/all-MiniLM-L6-v2")
if err != nil {
    return err
}
defer emb.Close()
```

## Embedding text

```go
func (e *Embedder) Embed(ctx context.Context, texts []string) ([][]float32, error)
```

Returns one embedding per input text, each of length `Dim()`, L2-normalized when
the model's manifest says so. Texts are embedded **independently** — a batch
fans out *across texts*, so there is zero padding waste and the result is
bit-identical to calling `Embed` once per text. `ctx` is checked before each
text's forward pass.

```go
vecs, err := emb.Embed(ctx, []string{"the cat sat on the mat", "quantum chromodynamics"})
// len(vecs) == 2, len(vecs[0]) == emb.Dim()
```

### Per-token hidden states

```go
func (e *Embedder) EmbedTokens(ctx context.Context, texts []string) ([]TokenEmbeddings, error)

type TokenEmbeddings struct {
    IDs     []int64      // token ids, including [CLS]/[SEP] framing
    Vectors [][]float32  // len(IDs) rows of Dim()
}
```

`EmbedTokens` returns the **unpooled, unnormalized** per-token hidden states
(ONNX Runtime's `last_hidden_state`) — the raw material for rerankers,
late-interaction (ColBERT-style) retrieval, or custom pooling. Like `Embed`, a
batch fans out across texts.

!!! note "Memory"
    `EmbedTokens` holds every result live for the whole call — a 256-text batch
    of long inputs is on the order of 200 MB.

## Options

Pass options to `Load`:

| Option | Effect |
|---|---|
| `WithInt8()` | Weight-only int8 inference: dense weights are quantized at load (per-output-channel symmetric); activations stay fp32. ~4× less weight traffic per embed (~42 MB → ~10.5 MB for MiniLM), cosine ≥ 0.999 vs fp32. Token embeddings stay fp32, so resident memory drops ~1.5×, not 4×. On CPUs without AVX2+FMA it silently falls back to fp32 — check `Quantized()`. |
| `WithInt8Activations()` | Full int8 on AVX-VNNI CPUs (Alder Lake 2021+, AVX-512-VNNI servers): weights *and* activations quantized, using `VPDPBUSD` for four multiply-accumulates per lane per instruction. Falls back to weight-only int8, then fp32, where VNNI or a shape isn't available — check `QuantizedActivations()`. **Implies `WithInt8()`.** |
| `WithDim(d int)` | Matryoshka (MRL): truncate each embedding to its first `d` dimensions and re-L2-normalize. `d` must be in `[1, full dim]`. Meaningful only for MRL-trained models (EmbeddingGemma 768→512/256/128, nomic-embed). `Dim()` reflects the truncated size. |
| `WithWorkers(n int)` | Caps the CPU workers one `Embed` call uses. Default (`0`) uses `GOMAXPROCS` — a spinning fork-join pool that minimizes single-request latency by burning idle-core cycles. Servers saturating many concurrent calls should set a small cap; `WithWorkers(1)` is fully serial with zero spinning. |
| `WithDiskWeights()` | Memory-maps weights from an on-disk pack file instead of loading them into RAM, so a model larger than RAM runs (disk-bandwidth-bound) and resident memory tracks the working set. On first use, safetensors are packed to `<dir>/weights.rembedpack`; later loads mmap it directly. The `Embedder` **must** be `Close`d to unmap. **Currently qwen3 only; cannot be combined with int8 modes yet.** |

```go
emb, err := rembed.Load("google/embeddinggemma-300m",
    rembed.WithInt8(),
    rembed.WithDim(256),
    rembed.WithWorkers(2),
)
```

## Introspection and cleanup

| Method | Returns |
|---|---|
| `Dim() int` | Embedding dimensionality (the `WithDim` truncation if set, else the full hidden size). |
| `Model() string` | The model name from the manifest. |
| `Quantized() bool` | Whether a weight-only int8 path is actually active (requested *and* every dense weight packed as int8). |
| `QuantizedActivations() bool` | Whether the full u8-activation VNNI path is active for every dense weight. |
| `Close() error` | Releases resources (unmaps disk-backed weights); a safe no-op for RAM-loaded models. The `Embedder` must not be used after `Close`. |
| `Tokenize(text string) []int64` | The tokenizer's `input_ids` for one text. For the validation harness and debugging — **not** a stable part of the embedding API. |

Because int8 modes fall back silently by design, query `Quantized()` /
`QuantizedActivations()` when the mode matters (e.g. before reporting perf
numbers).

## Cache and tokens

| Variable | Meaning |
|---|---|
| `REMBED_CACHE` | Model cache root. If set it is used directly; otherwise `os.UserCacheDir()`. If neither resolves, a Hub load errors with `hub: no cache dir (set REMBED_CACHE)`. |
| `HF_TOKEN` → `HUGGING_FACE_HUB_TOKEN` → `hf login` token file | Checked in that order for gated/private repos. |

## Concurrency and memory

- An `*Embedder` is safe for concurrent use.
- A single `Embed` batch can hold up to `min(GOMAXPROCS, WithWorkers)` scratch
  buffers, ~25 MB each at the maximum sequence length — the reason to cap
  `WithWorkers` on a busy server.
- At low concurrency the default spinning pool trades idle-core CPU for latency;
  a throughput-bound server that keeps many cores busy with concurrent requests
  should set `WithWorkers` to a small number (or `1`) to stop the spin.
