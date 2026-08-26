# CLI & server

```sh
go install github.com/rostamlabs/rembed/cmd/rembed@latest
```

```
usage: rembed {embed|validate|bench|serve} -model DIR|HF-ID [flags] [args]
```

`-model` accepts the same `ref` as [`Load`](usage.md#loading-a-model) — a local
directory or a Hugging Face model id.

## `embed`

Embed one or more texts.

```sh
rembed embed -model sentence-transformers/all-MiniLM-L6-v2 "some text" "another"
```

| Flag | Default | Meaning |
|---|---|---|
| `-model` | `models/all-MiniLM-L6-v2` | Model directory or HF id. |
| `-full` | `false` | Print the full vectors as JSON (`[][]float32`) instead of a preview. |
| `-dim` | `0` | Matryoshka output dimension (`0` = full). |
| `-int8` | `false` | Weight-only int8. |
| `-int8act` | `false` | Full int8 (needs AVX-VNNI). |

Positional arguments are the texts (at least one required). Default output prints
each text with its dimension and the first 8 components; `-full` emits JSON.

## `validate`

Check the model against a committed golden reference — the same check CI runs.

```sh
rembed validate -model sentence-transformers/all-MiniLM-L6-v2
```

| Flag | Default | Meaning |
|---|---|---|
| `-model` | `models/all-MiniLM-L6-v2` | Model directory or HF id. |
| `-golden` | `testdata/golden.json` | Golden reference file. |
| `-tol` | `1e-4` (`0.03` with `-int8`) | Max-abs-difference tolerance. |
| `-int8` | `false` | Validate the weight-only int8 path (loosens the default `-tol` to `0.03`). |
| `-force` | `false` | Compare even if the golden is for a different model. |

For each golden case it first checks tokenizer `input_ids` equality (so a failure
can be attributed to the tokenizer vs the numerics), then compares the embedding.
It prints `ok`/`FAIL` per case with the `maxAbsDiff`, and on success
`validate: all N cases within <tol> of the ONNX reference`.

## `bench`

Time embeds, with warm-up discard and medians.

```sh
rembed bench -model sentence-transformers/all-MiniLM-L6-v2 -runs 30
```

| Flag | Default | Meaning |
|---|---|---|
| `-model` | `models/all-MiniLM-L6-v2` | Model directory or HF id. |
| `-runs` | `30` | Timed runs (after warm-up). |
| `-warmup` | `5` | Warm-up runs, discarded. |
| `-text` | `"The quick brown fox…"` | Text to embed. |
| `-int8` / `-int8act` | `false` | Weight-only / full int8. |
| `-json` | `false` | Machine-readable per-run latencies (for `bench/compare.py`). |

Default output reports `median`, `p10`, `p90`, and spread as a percentage of the
median, plus a reminder that single-machine numbers are relative-only. JSON mode
also emits `quantized` / `quantized_activations` so the comparison harness can
confirm which mode *actually* ran (int8 fallback is silent by design). See
[Benchmarks](benchmarks.md) for methodology.

## `serve`

An OpenAI-compatible embeddings endpoint.

```sh
rembed serve -model sentence-transformers/all-MiniLM-L6-v2 -addr :8080
```

| Flag | Default | Meaning |
|---|---|---|
| `-model` | `sentence-transformers/all-MiniLM-L6-v2` | Model directory or HF id. |
| `-addr` | `:8080` | Listen address. |
| `-workers` | `0` (all cores) | CPU cap per request; set low for many concurrent clients. |
| `-max-batch` | `256` | Maximum inputs per request (≥ 1). |
| `-int8` / `-int8act` | `false` | Weight-only / full int8. |

### Endpoints

**`POST /v1/embeddings`** — body `{"input": "text"}` or `{"input": ["a", "b"]}`
(an optional `"model"` is accepted and echoed; token-id arrays are rejected with
a clear error). Returns the OpenAI shape:

```json
{
  "object": "list",
  "data": [{"object": "embedding", "index": 0, "embedding": [/* … */]}],
  "model": "all-MiniLM-L6-v2",
  "usage": {"prompt_tokens": 4, "total_tokens": 4}
}
```

Request bodies are capped at 32 MiB; errors use the OpenAI error envelope
(`invalid_request_error`).

**`GET /healthz`** — `{"status": "ok", "model": "…", "dim": 384}`.

The server drains gracefully on `SIGINT`/`SIGTERM` (15 s shutdown timeout) and
sets a 10 s header-read timeout.
