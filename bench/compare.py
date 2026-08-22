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


def bench_ours(binary: str, model_dir: str, text: str, runs: int, warmup: int,
               expect_seq: int, int8: "bool | str" = False) -> list[float]:
    """Per-run embed latencies (seconds) from the Go engine, in-process there."""
    cmd = [binary, "bench", "-model", model_dir, "-runs", str(runs), "-warmup", str(warmup),
           "-text", text, "-json"]
    if int8 == "full":
        cmd.append("-int8act")
    elif int8:
        cmd.append("-int8")
    out = subprocess.run(
        cmd,
        check=True, capture_output=True, text=True, cwd=REPO_ROOT,
    ).stdout
    payload = json.loads(out)
    # Both engines must tokenize to the same length or the ratio compares
    # different workloads.
    if payload["seq"] != expect_seq:
        raise SystemExit(
            f"seq mismatch: Go tokenized to {payload['seq']}, HF to {expect_seq} — "
            f"fix the tokenizer divergence before trusting any ratio"
        )
    # The engine falls back silently by design; a benchmark must not
    # label fallback latencies as the requested mode (same standard as
    # the seq check above).
    if int8 == "full" and not payload.get("quantized_activations"):
        raise SystemExit("full int8 requested but the engine fell back "
                         "(no VNNI on this CPU?) — refusing to mislabel the run")
    if int8 is True and not payload.get("quantized"):
        raise SystemExit("int8 requested but the engine fell back to fp32 — "
                         "refusing to mislabel the run")
    return payload["runs_sec"]


def bench_onnx(model_id: str, text: str, runs: int, warmup: int, threads: int,
               onnx_file: str = "onnx/model.onnx") -> list[float]:
    import onnxruntime as ort
    from huggingface_hub import hf_hub_download
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(model_id)
    so = ort.SessionOptions()
    # ORT thread pool control. Since M2 the Go engine is multithreaded, so
    # the like-for-like default is ORT's own default all-cores pool (0);
    # pin to 1 for single-core kernel-quality comparisons (report which
    # configuration a number came from!).
    if threads > 0:
        so.intra_op_num_threads = threads
        so.inter_op_num_threads = 1
    sess = ort.InferenceSession(
        hf_hub_download(model_id, onnx_file), so, providers=["CPUExecutionProvider"]
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
    ap.add_argument("--ours-int8", action="store_true",
                    help="run the Go engine with weight-only int8 (-int8)")
    ap.add_argument("--ours-int8act", action="store_true",
                    help="run the Go engine with FULL int8 (-int8act, needs VNNI)")
    ap.add_argument("--onnx-file", default="onnx/model.onnx",
                    help="ONNX graph to benchmark (e.g. onnx/model_quint8_avx2.onnx for ORT's int8)")
    ap.add_argument("--onnx-threads", type=int, default=0,
                    help="ORT intra-op threads (0 = default pool, like-for-like since M2's parallelism; 1 = single-core kernel comparison)")
    args = ap.parse_args()
    if args.runs <= 0 or args.warmup < 0:
        raise SystemExit("--runs must be > 0 and --warmup >= 0")

    from huggingface_hub import hf_hub_download
    from transformers import AutoTokenizer
    hf_seq = len(AutoTokenizer.from_pretrained(args.model_id)(args.text, truncation=True, max_length=512)["input_ids"])

    binary = args.binary
    if binary is None:
        binary = str(REPO_ROOT / "bin" / "rembed")
        subprocess.run(["go", "build", "-o", binary, "./cmd/rembed"], check=True, cwd=REPO_ROOT)

    ours_label = "int8-full" if args.ours_int8act else ("int8-weights" if args.ours_int8 else "fp32")
    print(f"seq={hf_seq}  ours={ours_label}  onnx-file={args.onnx_file}  onnx-threads={args.onnx_threads or 'default pool'}")
    ours_mode = "full" if args.ours_int8act else args.ours_int8
    results: dict[str, list[list[float]]] = {"ours": [], "onnx": []}
    for order in ("us-then-onnx", "onnx-then-us"):
        print(f"order: {order}")
        if order == "us-then-onnx":
            results["ours"].append(bench_ours(binary, args.model_dir, args.text, args.runs, args.warmup, hf_seq, ours_mode))
            results["onnx"].append(bench_onnx(args.model_id, args.text, args.runs, args.warmup, args.onnx_threads, args.onnx_file))
        else:
            results["onnx"].append(bench_onnx(args.model_id, args.text, args.runs, args.warmup, args.onnx_threads, args.onnx_file))
            results["ours"].append(bench_ours(binary, args.model_dir, args.text, args.runs, args.warmup, hf_seq, ours_mode))
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
