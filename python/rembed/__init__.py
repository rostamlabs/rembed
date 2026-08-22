# SPDX-License-Identifier: Apache-2.0
"""Python bindings for rembed, the pure-Go embedding inference engine.

The Go engine is loaded in-process through a C-shared library (call
overhead ~µs against a ~1.4 ms embed), so there is no subprocess, server,
or serialization in the path.

    from rembed import Embedder

    emb = Embedder("models/all-MiniLM-L6-v2")          # fp32
    emb = Embedder("models/all-MiniLM-L6-v2", int8=True)  # weight-only int8
    vecs = emb.embed(["hello world", "second text"])   # (n, dim) float32

Returns numpy arrays when numpy is importable, plain nested lists
otherwise. Build the shared library once from the repo root:

    CGO_ENABLED=1 go build -buildmode=c-shared \
        -o python/rembed/librembed.so ./python/capi

(or run python/build.sh). The library search order: REMBED_LIB env var,
then librembed.so next to this file.
"""

from __future__ import annotations

import ctypes
import os
from pathlib import Path

try:
    import numpy as _np
except ImportError:  # numpy is optional; lists are returned instead
    _np = None

__all__ = ["Embedder", "RembedError"]


class RembedError(RuntimeError):
    """An error reported by the rembed engine."""


def _find_library() -> Path:
    env = os.environ.get("REMBED_LIB")
    if env:
        p = Path(env)
        if p.is_file():
            return p
        raise RembedError(f"REMBED_LIB={env} does not exist")
    p = Path(__file__).with_name("librembed.so")
    if p.is_file():
        return p
    raise RembedError(
        "librembed.so not found — build it from the repo root with:\n"
        "  CGO_ENABLED=1 go build -buildmode=c-shared "
        "-o python/rembed/librembed.so ./python/capi"
    )


_lib = None


def _load_library() -> ctypes.CDLL:
    global _lib
    if _lib is None:
        lib = ctypes.CDLL(str(_find_library()))
        lib.RembedLoad.argtypes = [
            ctypes.c_char_p, ctypes.c_int, ctypes.c_int,
            ctypes.POINTER(ctypes.c_char_p),
        ]
        lib.RembedLoad.restype = ctypes.c_longlong
        lib.RembedDim.argtypes = [ctypes.c_longlong]
        lib.RembedDim.restype = ctypes.c_int
        lib.RembedModel.argtypes = [ctypes.c_longlong, ctypes.c_char_p, ctypes.c_int]
        lib.RembedModel.restype = ctypes.c_int
        lib.RembedEmbedBatch.argtypes = [
            ctypes.c_longlong, ctypes.POINTER(ctypes.c_char_p), ctypes.c_int,
            ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_char_p),
        ]
        lib.RembedEmbedBatch.restype = ctypes.c_int
        lib.RembedClose.argtypes = [ctypes.c_longlong]
        lib.RembedClose.restype = None
        lib.RembedFreeString.argtypes = [ctypes.c_char_p]
        lib.RembedFreeString.restype = None
        _lib = lib
    return _lib


def _take_error(lib: ctypes.CDLL, err: ctypes.c_char_p) -> str:
    # ctypes copies c_char_p values on access; free the Go-allocated original.
    msg = err.value.decode("utf-8", "replace") if err.value else "unknown error"
    lib.RembedFreeString(err)
    return msg


class Embedder:
    """In-process handle to a loaded rembed model.

    Thread-safe (the Go engine is); ``close()`` or use as a context
    manager to release the model's memory eagerly.
    """

    def __init__(self, model_dir: str | os.PathLike, *, int8: bool = False,
                 workers: int = 0):
        self._lib = _load_library()
        self._handle = 0
        err = ctypes.c_char_p()
        handle = self._lib.RembedLoad(
            str(model_dir).encode(), 1 if int8 else 0, workers,
            ctypes.byref(err),
        )
        if handle == 0:
            raise RembedError(_take_error(self._lib, err))
        self._handle = handle
        self._dim = self._lib.RembedDim(handle)

    @property
    def dim(self) -> int:
        """Embedding dimensionality."""
        return self._dim

    @property
    def model(self) -> str:
        """Model name from the manifest."""
        n = self._lib.RembedModel(self._handle, None, 0)
        buf = ctypes.create_string_buffer(n + 1)
        self._lib.RembedModel(self._handle, buf, n + 1)
        return buf.value.decode()

    def embed(self, texts):
        """Embed a list of texts (or one string) into L2-normalized vectors.

        Returns an (n, dim) float32 numpy array, or nested lists without
        numpy. A single string returns the 1-row result unwrapped.
        """
        single = isinstance(texts, (str, bytes))
        if single:
            texts = [texts]
        texts = list(texts)
        if self._handle == 0:
            raise RembedError("embedder is closed")
        n = len(texts)
        if _np is not None:
            out = _np.empty((n, self._dim), dtype=_np.float32)
            out_ptr = out.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
        else:
            raw = (ctypes.c_float * (n * self._dim))()
            out_ptr = ctypes.cast(raw, ctypes.POINTER(ctypes.c_float))
        if n > 0:
            arr = (ctypes.c_char_p * n)(
                *[t.encode() if isinstance(t, str) else bytes(t) for t in texts]
            )
            err = ctypes.c_char_p()
            rc = self._lib.RembedEmbedBatch(
                self._handle, arr, n, out_ptr, ctypes.byref(err)
            )
            if rc != 0:
                raise RembedError(_take_error(self._lib, err))
        if _np is not None:
            return out[0] if single else out
        rows = [list(raw[i * self._dim:(i + 1) * self._dim]) for i in range(n)]
        return rows[0] if single else rows

    def close(self) -> None:
        if self._handle:
            self._lib.RembedClose(self._handle)
            self._handle = 0

    def __enter__(self) -> "Embedder":
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass
