#!/usr/bin/env python3
"""Dev-only: EmbeddingGemma golden generator.

Independent reference for the gemma3 path: torch Gemma3TextModel (eager,
fp32) → masked mean pooling → the two bias-free Dense layers (H→D→H) →
L2 normalize, exactly the sentence-transformers module stack the repo
declares (Transformer, Pooling, Dense, Dense, Normalize). Writes
testdata/golden-embeddinggemma-300m.json for the Go golden matrix.

Usage: python gemma_golden.py [model_dir] [model_id]
"""

import json
import struct
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "models"))
from convert import GOLDEN_TEXTS  # noqa: E402


def load_linear(path: Path) -> np.ndarray:
    with open(path, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        header = json.loads(f.read(n))
        ent = header["linear.weight"]
        f.seek(8 + n + ent["data_offsets"][0])
        raw = f.read(ent["data_offsets"][1] - ent["data_offsets"][0])
    return np.frombuffer(raw, dtype=np.float32).reshape(ent["shape"])


def main() -> None:
    model_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else REPO_ROOT / "models" / "embeddinggemma-300m"
    model_id = sys.argv[2] if len(sys.argv) > 2 else "google/embeddinggemma-300m"

    import torch
    from transformers import AutoModel, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(model_id)
    model = AutoModel.from_pretrained(
        model_id, attn_implementation="eager", torch_dtype=torch.float32,
        low_cpu_mem_usage=True).eval()
    w1 = load_linear(model_dir / "2_Dense" / "model.safetensors")  # [D, H]
    w2 = load_linear(model_dir / "3_Dense" / "model.safetensors")  # [H, D]

    def embed(text: str):
        enc = tok(text, truncation=True, max_length=2048)
        ids = np.array([enc["input_ids"]], dtype=np.int64)
        mask = np.array([enc["attention_mask"]], dtype=np.int64)
        with torch.no_grad():
            out = model(input_ids=torch.tensor(ids), attention_mask=torch.tensor(mask))
        last = out.last_hidden_state[0].numpy().astype(np.float32)  # [seq, H]
        m = mask[0].astype(np.float32)[:, None]
        pooled = (last * m).sum(0) / np.maximum(m.sum(), 1e-9)       # mean pool
        pooled = pooled @ w1.T                                       # Dense H->D
        pooled = pooled @ w2.T                                       # Dense D->H
        n = np.linalg.norm(pooled)
        pooled = pooled / n if n > 0 else pooled                     # L2 normalize
        return enc["input_ids"], pooled

    golden = []
    for text in GOLDEN_TEXTS:
        ids, emb = embed(text)
        golden.append({
            "text": text,
            "input_ids": ids,
            "embedding": [float(f"{v:.8g}") for v in emb],
        })
    out = REPO_ROOT / "testdata" / "golden-embeddinggemma-300m.json"
    out.write_text(json.dumps({"model": model_id, "cases": golden}, ensure_ascii=False) + "\n")
    print(f"wrote {out} ({len(golden)} cases, dim {len(golden[0]['embedding'])})")


if __name__ == "__main__":
    main()
