# Python & C bindings

rembed ships a Python binding built over a small C ABI. The Go library itself
stays cgo-free — the shared object is a *separate* artifact for foreign callers.
Calls run in-process (ctypes over the C ABI, ~µs overhead against a ~1.4 ms
embed): no subprocess, no server, no serialization in the path.

## Python

### Build

The binding needs a C toolchain to compile the shared library once:

```sh
python/build.sh          # produces python/rembed/librembed.so
# equivalently:
CGO_ENABLED=1 go build -buildmode=c-shared -o python/rembed/librembed.so ./python/capi
```

The library is found via `REMBED_LIB`, else `librembed.so` next to the package's
`__init__.py`.

### Usage

```python
import sys; sys.path.insert(0, "python")
from rembed import Embedder

emb = Embedder("models/all-MiniLM-L6-v2")             # fp32
emb = Embedder("models/all-MiniLM-L6-v2", int8=True)  # weight-only int8
vecs = emb.embed(["hello world"])                     # (n, dim) float32 numpy array
```

### API

```python
class Embedder(model_dir, *, int8: bool = False, workers: int = 0)
```

An in-process handle. Thread-safe. Usable as a context manager (`with
Embedder(...) as emb:`), and `close()` runs on `__del__`.

| Member | Description |
|---|---|
| `.embed(texts)` | A list of texts, or a single string. Returns an `(n, dim)` float32 numpy array (or nested lists if numpy isn't installed). A single string returns the one row unwrapped. Empty batch → length 0; the empty string is well-defined (unit norm). |
| `.dim` | Embedding dimensionality (int). |
| `.model` | Model name from the manifest (str). |
| `.close()` | Release the handle. |
| `RembedError(RuntimeError)` | Raised on load/embed failure, with a real message. |

!!! note "Surface"
    The ctypes layer currently exposes `int8=` and `workers=` only.
    `WithInt8Activations`, `WithDim`, and `WithDiskWeights` are not surfaced
    through the Python binding yet — use the [Go library](usage.md) for those.

The bundled `python/test_rembed.py` validates against the same golden reference
as the Go tests (fp32 within `1e-4`, cosine ≥ 0.9999; int8 within `maxAbsDiff`
`0.03`, cosine ≥ 0.998), including a `workers=1` serial-path check and the
single-string-equals-batch-row invariant.

## C API

The shared library exports a flat, allocation-disciplined C ABI (from
`python/capi`) — the same surface the Python binding uses, and the entry point
for any other FFI caller. Handles are opaque `longlong` integers (cgo forbids
passing Go pointers across the boundary); the caller owns every output buffer,
and any returned string must be released with `RembedFreeString`.

| Symbol | Purpose |
|---|---|
| `RembedLoad(modelDir, useInt8, workers, errOut) → handle` | Opens a model dir; returns an opaque handle > 0. `useInt8 != 0` → `WithInt8()`; `workers > 0` → `WithWorkers`. On failure returns `0` and stores a freeable message in `errOut`. |
| `RembedDim(handle) → int` | Dimensionality, or `0` for a bad handle. |
| `RembedModel(handle, buf, bufLen) → int` | Writes the model name into `buf` (NUL-terminated); returns the full length (≥ `bufLen` means truncated), `-1` for a bad handle. |
| `RembedEmbedBatch(handle, texts, n, out, errOut) → int` | Embeds `n` texts row-major into the caller's `out` buffer (`n × RembedDim` floats). Returns `0` on success, `-1` on failure (fills `errOut`). |
| `RembedClose(handle)` | Releases a handle (safe with an unknown handle). |
| `RembedFreeString(s)` | Frees a string returned via an `errOut` parameter. |
