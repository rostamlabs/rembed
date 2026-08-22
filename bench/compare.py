#!/usr/bin/env python3
"""us-vs-ONNX benchmark with the rigor rules from DESIGN.md.

Runs the Go engine (via the rembed CLI) and ONNX Runtime on the same input,
in BOTH orders (us-then-ONNX and ONNX-then-us) to control for thermal/cache
ordering effects, with warm-up discard, medians over N runs, and a spread
report. Deltas within the measurement noise are FLAGGED, not celebrated.

Laptop numbers are relative-only/provisional; pin to a dedicated cloud box
for any publishable claim.

Usage (from the repo root, model dir already exported by models/convert.py):
    models/.venv/bin/python bench/compare.py [--runs 30] [--warmup 5]
"""

import argparse
import json
import statistics
import subprocess
import time
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_TEXT = "The quick brown fox jumps over the lazy dog."


def bench_ours(binary: str, model_dir: str, text: str, runs: int, warmup: int) -> list[float]:
    """Per-run embed latencies (seconds) from the Go engine, in-process there."""
    out = subprocess.run(
        [binary, "bench", "-model", model_dir, "-runs", str(runs), "-warmup", str(warmup),
         "-text", text, "-json"],
        check=True, capture_output=True, text=True, cwd=REPO_ROOT,
    ).stdout
    return json.loads(out)["runs_sec"]


def bench_onnx(model_id: str, text: str, runs: int, warmup: int) -> list[float]:
    import onnxruntime as ort
    from huggingface_hub import hf_hub_download
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(model_id)
    sess = ort.InferenceSession(
        hf_hub_download(model_id, "onnx/model.onnx"), providers=["CPUExecutionProvider"]
    )
    input_names = {i.name for i in sess.get_inputs()}
    enc = tok(text, truncation=True, max_length=512)
    ids = np.array([enc["input_ids"]], dtype=np.int64)
    feeds = {"input_ids": ids, "attention_mask": np.array([enc["attention_mask"]], dtype=np.int64)}
    if "token_type_ids" in input_names:
        feeds["token_type_ids"] = np.zeros_like(ids)

    for _ in range(warmup):
        sess.run(["last_hidden_state"], feeds)
    durs = []
    for _ in range(runs):
        t0 = time.perf_counter()
        sess.run(["last_hidden_state"], feeds)
        durs.append(time.perf_counter() - t0)
    return durs


def summarize(name: str, durs: list[float]) -> tuple[float, float]:
    durs = sorted(durs)
    median = statistics.median(durs)
    p10, p90 = durs[len(durs) // 10], durs[len(durs) * 9 // 10]
    spread = (p90 - p10) / median
    print(f"  {name:8s} median={median * 1e3:8.3f}ms  p10={p10 * 1e3:8.3f}ms  "
          f"p90={p90 * 1e3:8.3f}ms  spread={spread * 100:5.1f}%")
    return median, spread


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir", default="models/all-MiniLM-L6-v2")
    ap.add_argument("--model-id", default="sentence-transformers/all-MiniLM-L6-v2")
    ap.add_argument("--binary", default=None, help="rembed binary (default: go build one)")
    ap.add_argument("--text", default=DEFAULT_TEXT)
    ap.add_argument("--runs", type=int, default=30)
    ap.add_argument("--warmup", type=int, default=5)
    args = ap.parse_args()

    binary = args.binary
    if binary is None:
        binary = str(REPO_ROOT / "bin" / "rembed")
        subprocess.run(["go", "build", "-o", binary, "./cmd/rembed"], check=True, cwd=REPO_ROOT)

    results: dict[str, list[list[float]]] = {"ours": [], "onnx": []}
    for order in ("us-then-onnx", "onnx-then-us"):
        print(f"order: {order}")
        if order == "us-then-onnx":
            results["ours"].append(bench_ours(binary, args.model_dir, args.text, args.runs, args.warmup))
            results["onnx"].append(bench_onnx(args.model_id, args.text, args.runs, args.warmup))
        else:
            results["onnx"].append(bench_onnx(args.model_id, args.text, args.runs, args.warmup))
            results["ours"].append(bench_ours(binary, args.model_dir, args.text, args.runs, args.warmup))
        summarize("ours", results["ours"][-1])
        summarize("onnx", results["onnx"][-1])

    ours_all = [d for r in results["ours"] for d in r]
    onnx_all = [d for r in results["onnx"] for d in r]
    print("combined (both orders):")
    ours_med, ours_spread = summarize("ours", ours_all)
    onnx_med, onnx_spread = summarize("onnx", onnx_all)
    ratio = ours_med / onnx_med
    print(f"\nours / onnx = {ratio:.2f}x  (lower is better; 1.00x = parity)")
    # Order-effect check: if either engine's two order-medians differ by more
    # than its own p10..p90 spread, the machine is not giving stable numbers.
    for name in ("ours", "onnx"):
        m1 = statistics.median(results[name][0])
        m2 = statistics.median(results[name][1])
        if abs(m1 - m2) / min(m1, m2) > 0.10:
            print(f"FLAG: {name} order-medians differ {abs(m1 - m2) / min(m1, m2) * 100:.0f}% "
                  f"(us-first {m1 * 1e3:.2f}ms vs onnx-first {m2 * 1e3:.2f}ms) — machine unstable, rerun")
    noise = max(ours_spread, onnx_spread)
    if abs(ratio - 1) < noise:
        print(f"FLAG: the ours-vs-onnx delta ({abs(ratio - 1) * 100:.0f}%) is within measurement "
              f"noise ({noise * 100:.0f}%) — treat as parity, not a win or loss")


if __name__ == "__main__":
    main()
