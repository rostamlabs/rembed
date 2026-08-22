# SPDX-License-Identifier: Apache-2.0
"""Binding correctness test: the vectors that cross the C ABI must match
the golden ONNX reference exactly as the Go tests do — fp32 within 1e-4,
int8 within the pinned quantization bounds (maxAbsDiff 0.03, cosine 0.998).

Run from the repo root (model dir exported, librembed.so built):
    models/.venv/bin/python python/test_rembed.py
"""

import json
import math
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "python"))

from rembed import Embedder  # noqa: E402


def check(emb, cases, max_diff, min_cos, label):
    texts = [c["text"] for c in cases]
    vecs = emb.embed(texts)
    worst_d, worst_c = 0.0, 1.0
    for row, c in zip(vecs, cases):
        want = c["embedding"]
        assert len(row) == len(want), f"dim {len(row)} != {len(want)}"
        d = max(abs(float(a) - b) for a, b in zip(row, want))
        cos = sum(float(a) * b for a, b in zip(row, want))
        worst_d, worst_c = max(worst_d, d), min(worst_c, cos)
        assert d <= max_diff, f"{label}: {c['text'][:40]!r} maxAbsDiff {d:.3g} > {max_diff}"
        assert cos >= min_cos, f"{label}: {c['text'][:40]!r} cosine {cos:.6f} < {min_cos}"
    # Single-string call unwraps to one row and matches the batch result.
    one = emb.embed(texts[0])
    assert max(abs(float(a) - float(b)) for a, b in zip(one, vecs[0])) == 0
    print(f"  {label}: {len(cases)} cases ok (worst maxAbsDiff {worst_d:.3g}, worst cosine {worst_c:.6f})")


def main():
    golden = json.loads((REPO / "testdata" / "golden.json").read_text())
    cases = golden["cases"]
    model_dir = str(REPO / "models" / "all-MiniLM-L6-v2")

    with Embedder(model_dir) as emb:
        assert emb.dim == len(cases[0]["embedding"])
        assert emb.model == golden["model"], emb.model
        check(emb, cases, 1e-4, 0.9999, "fp32")
        # Empty batch and empty string are well-defined.
        assert len(emb.embed([])) == 0
        v = emb.embed("")
        assert abs(math.fsum(float(x) * float(x) for x in v) - 1) < 1e-4

    with Embedder(model_dir, int8=True) as emb:
        check(emb, cases, 0.03, 0.998, "int8")

    with Embedder(model_dir, workers=1) as emb:  # serial path
        check(emb, cases[:3], 1e-4, 0.9999, "fp32 workers=1")

    # Bad model dir raises, with a real message.
    try:
        Embedder(str(REPO / "does-not-exist"))
    except Exception as e:
        assert "manifest" in str(e) or "no such file" in str(e).lower(), e
    else:
        raise AssertionError("bad model dir did not raise")

    with Embedder(model_dir) as emb:
        text = "The quick brown fox jumps over the lazy dog."
        emb.embed(text)  # warm
        t0 = time.perf_counter()
        n = 30
        for _ in range(n):
            emb.embed(text)
        ms = (time.perf_counter() - t0) / n * 1e3
        print(f"  latency from python: {ms:.2f} ms/embed (seq=12)")

    print("python bindings: all checks passed")


if __name__ == "__main__":
    main()
