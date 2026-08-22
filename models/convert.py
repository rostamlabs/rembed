#!/usr/bin/env python3
"""One-time HuggingFace -> rembed model-dir export, plus golden reference.

Downloads a sentence-transformers BERT-family model, assembles the rembed
model dir {model.safetensors, vocab.txt, manifest.json}, and regenerates
testdata/golden.json: ONNX Runtime reference embeddings for a fixed input set,
used by the Go validation harness (every kernel/model change must keep
matching these within tolerance).

Usage:
    python convert.py [model_id] [--out DIR] [--golden FILE]

Defaults: model_id=sentence-transformers/all-MiniLM-L6-v2,
out=models/<basename>, golden=testdata/golden.json (repo-relative).
"""

import argparse
import json
import shutil
import struct
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent

# Fixed golden inputs. Deliberately includes the historically-divergent
# tokenizer classes — accents (stripped when lowercasing), non-ASCII symbols
# (NOT separators in BERT), and CJK ideographs (space-padded) — so the golden
# set pins HF-fidelity of the tokenizer, not just the numerics.
GOLDEN_TEXTS = [
    "hello world",
    "The quick brown fox jumps over the lazy dog.",
    "a",
    "Rostam is a vector database written in Go, with BM25 and semantic caching built in.",
    "embedding inference engines should be boring, predictable, and fast",
    "punctuation, splitting: does it (really) work?! yes -- it does...",
    "supercalifragilisticexpialidocious antidisestablishmentarianism pseudopseudohypoparathyroidism",
    "café résumé naïve déjà vu at the sørensen-müller café",
    "€100 costs $5, or ±43 at 20° ± 3°",
    "你好世界 mixed with english text",
    "In 2024, the model processed 1,234,567 queries at 99.9% availability, "
    "averaging 3.14 ms per request across 42 regions. "
    "This sentence exists to push the sequence length up so the golden set "
    "covers more position embeddings than the short inputs do, including a "
    "second clause with commas, numbers like 8675309, and ordinary prose "
    "about nothing in particular that keeps going for a while longer.",
    # ~400 tokens: long enough that token pairs reach |j-i| >= 128, pinning
    # the log-spaced far buckets AND the max-distance clamp of relative-
    # position-bias models (MPNet) end to end — the shorter cases never
    # leave the near-exact buckets.
    "The history of mechanical computation stretches back further than most "
    "people assume, beginning with devices like the antikythera mechanism, "
    "a bronze assembly of interlocking gears recovered from a shipwreck in "
    "the aegean sea that modeled the motions of the sun and moon with "
    "startling precision. centuries later, charles babbage designed his "
    "difference engine to tabulate polynomial functions automatically, and "
    "ada lovelace, studying his more ambitious analytical engine, wrote what "
    "many consider the first published algorithm, recognizing that a machine "
    "manipulating symbols could do far more than arithmetic. the twentieth "
    "century accelerated everything: alan turing formalized the notion of "
    "computability with an abstract machine reading and writing symbols on "
    "an infinite tape, claude shannon showed that boolean algebra could "
    "describe switching circuits, and the engineers of eniac soldered "
    "seventeen thousand vacuum tubes into a room-sized calculator for "
    "artillery tables. transistors replaced tubes, integrated circuits "
    "replaced discrete transistors, and the number of components on a chip "
    "doubled with such regularity that the trend acquired a name and became "
    "a planning assumption for an entire industry. software grew alongside "
    "the hardware, from hand-assembled machine code through fortran and lisp "
    "to operating systems that shared a single expensive processor among "
    "many impatient users. networks stitched the machines together, first "
    "across campuses, then across continents, until a physicist at cern "
    "proposed a system of linked hypertext documents that anyone could "
    "publish to and anyone could read, and the resulting web rearranged "
    "commerce, journalism, friendship, and memory itself within a single "
    "human generation, leaving us to wonder what the next doubling will "
    "rearrange next.",
]


def sanity_check_safetensors(path: Path) -> None:
    """Verify every tensor dtype is one the Go loader handles."""
    with open(path, "rb") as f:
        (hlen,) = struct.unpack("<Q", f.read(8))
        header = json.loads(f.read(hlen))
    bad = {
        name: entry["dtype"]
        for name, entry in header.items()
        if name != "__metadata__"
        and entry["dtype"] not in ("F32", "F16", "BF16")
        # position_ids is a saved buffer (0..maxPos), not a weight; the Go
        # loader skips it.
        and not name.endswith("position_ids")
    }
    if bad:
        raise SystemExit(f"{path}: unsupported tensor dtypes: {bad}")


def pool(last_hidden: np.ndarray, mask: np.ndarray, mode: str, normalize: bool) -> np.ndarray:
    """[seq, H] token embeddings + [seq] attention mask -> pooled [H].

    `mode`/`normalize` must be the manifest's values: the Go engine honors
    them, so the golden must too, or the reference itself becomes wrong.
    """
    if mode == "cls":
        pooled = last_hidden[0].astype(np.float32)
    else:
        m = mask.astype(np.float32)[:, None]
        pooled = (last_hidden * m).sum(axis=0) / np.maximum(m.sum(), 1e-9)
    if normalize:
        n = np.linalg.norm(pooled)
        if n == 0:
            raise SystemExit("golden case pooled to a zero vector; refusing to divide")
        pooled = pooled / n
    return pooled


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("model_id", nargs="?", default="sentence-transformers/all-MiniLM-L6-v2")
    ap.add_argument("--out", type=Path, default=None, help="model dir to write")
    ap.add_argument("--tokens-golden", type=Path, default=None,
                    help="also write a token-level (last_hidden_state) golden here")
    ap.add_argument("--golden", type=Path, default=None,
                    help="golden output path (default: testdata/golden.json for the default model, "
                         "testdata/golden-<model>.json otherwise — never silently overwriting "
                         "another model's golden)")
    args = ap.parse_args()
    if args.golden is None:
        short = args.model_id.split("/")[-1]
        if args.model_id == "sentence-transformers/all-MiniLM-L6-v2":
            args.golden = REPO_ROOT / "testdata" / "golden.json"
        else:
            args.golden = REPO_ROOT / "testdata" / f"golden-{short}.json"

    from huggingface_hub import hf_hub_download
    import onnxruntime as ort
    from transformers import AutoTokenizer

    out = args.out or Path(__file__).resolve().parent / args.model_id.split("/")[-1]
    out.mkdir(parents=True, exist_ok=True)

    def fetch(filename: str) -> Path:
        return Path(hf_hub_download(args.model_id, filename))

    # --- model dir ---------------------------------------------------------
    st = fetch("model.safetensors")
    sanity_check_safetensors(st)
    shutil.copyfile(st, out / "model.safetensors")

    config = json.loads(fetch("config.json").read_text())
    # Byte-level BPE models (RoBERTa) ship vocab.json + merges.txt;
    # WordPiece models (BERT, MPNet) ship vocab.txt.
    if config.get("model_type") == "roberta":
        shutil.copyfile(fetch("vocab.json"), out / "vocab.json")
        shutil.copyfile(fetch("merges.txt"), out / "merges.txt")
    else:
        shutil.copyfile(fetch("vocab.txt"), out / "vocab.txt")
    tok_config = json.loads(fetch("tokenizer_config.json").read_text())
    pooling = json.loads(fetch("1_Pooling/config.json").read_text())
    modules = json.loads(fetch("modules.json").read_text())

    model_type = config.get("model_type")
    if model_type not in ("bert", "roberta", "mpnet"):
        raise SystemExit(f"model_type={model_type!r}: only bert, roberta, and mpnet models are supported")
    if model_type == "mpnet":
        # HF's MPNet code hardcodes num_buckets=32 and padding_idx=1
        # regardless of config; the Go loader refuses anything else, so
        # refuse at export time too.
        if config.get("relative_attention_num_buckets") != 32:
            raise SystemExit("mpnet: relative_attention_num_buckets must be 32 (HF hardcodes it)")
        if config.get("pad_token_id", 1) != 1:
            raise SystemExit("mpnet: pad_token_id must be 1 (HF hardcodes padding_idx=1)")
    # The Go engine hardcodes exact-erf GELU and absolute position embeddings;
    # anything else would produce a valid-looking model dir that computes the
    # wrong thing, so refuse at export time.
    if config.get("hidden_act", "gelu") != "gelu":
        raise SystemExit(f"hidden_act={config.get('hidden_act')!r}: only exact GELU is supported")
    if config.get("position_embedding_type", "absolute") != "absolute":
        raise SystemExit(f"position_embedding_type={config.get('position_embedding_type')!r}: only absolute is supported")
    mean = bool(pooling.get("pooling_mode_mean_tokens"))
    cls = bool(pooling.get("pooling_mode_cls_token"))
    other = any(pooling.get(k) for k in (
        "pooling_mode_max_tokens", "pooling_mode_lasttoken",
        "pooling_mode_mean_sqrt_len_tokens", "pooling_mode_weightedmean_tokens"))
    if other or mean == cls:
        raise SystemExit(f"unsupported pooling config (rembed supports mean or cls): {pooling}")
    pool_mode = "mean" if mean else "cls"
    # Any ST module beyond Transformer/Pooling/Normalize (e.g. a Dense head)
    # would make the real model differ from what rembed computes.
    known = ("models.Transformer", "models.Pooling", "models.Normalize")
    unknown = [m["type"] for m in modules if not m.get("type", "").endswith(known)]
    if unknown:
        raise SystemExit(f"unsupported sentence-transformers modules: {unknown}")
    normalize = any(m.get("type", "").endswith("models.Normalize") for m in modules)

    manifest = {
        "name": args.model_id,
        "hidden_size": config["hidden_size"],
        "num_hidden_layers": config["num_hidden_layers"],
        "num_attention_heads": config["num_attention_heads"],
        "intermediate_size": config["intermediate_size"],
        "vocab_size": config["vocab_size"],
        "max_position_embeddings": config["max_position_embeddings"],
        "layer_norm_eps": config.get("layer_norm_eps", 1e-12),
        "do_lower_case": bool(tok_config.get("do_lower_case", False)) if model_type == "roberta" else bool(tok_config.get("do_lower_case", True)),
        "cls_token": tok_config.get("cls_token", "[CLS]"),
        "sep_token": tok_config.get("sep_token", "[SEP]"),
        "unk_token": tok_config.get("unk_token", "<unk>" if model_type == "roberta" else "[UNK]"),
        "pooling": pool_mode,
        "normalize": normalize,
    }
    if model_type == "mpnet":
        # Positions are offset by pad_token_id+1 (fairseq convention), and
        # attention adds a shared bucketed relative-position bias.
        manifest["model_type"] = "mpnet"
        manifest["relative_attention_num_buckets"] = config["relative_attention_num_buckets"]
        manifest["pad_token_id"] = config.get("pad_token_id", 1)
    elif model_type == "roberta":
        # Same fairseq position offset as MPNet; plain BERT encoder inside.
        manifest["model_type"] = "roberta"
        manifest["pad_token_id"] = config["pad_token_id"]
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    vocab_files = "vocab.json merges.txt" if model_type == "roberta" else "vocab.txt"
    print(f"wrote {out}/[model.safetensors {vocab_files} manifest.json]")

    # --- golden reference (ONNX Runtime, per-text, no padding) -------------
    tokenizer = AutoTokenizer.from_pretrained(args.model_id)
    sess = ort.InferenceSession(fetch("onnx/model.onnx"), providers=["CPUExecutionProvider"])
    input_names = {i.name for i in sess.get_inputs()}

    max_len = manifest["max_position_embeddings"]
    if model_type in ("mpnet", "roberta"):
        max_len -= config.get("pad_token_id", 1) + 1

    golden = []
    for text in GOLDEN_TEXTS:
        enc = tokenizer(text, truncation=True, max_length=max_len)
        ids = np.array([enc["input_ids"]], dtype=np.int64)
        mask = np.array([enc["attention_mask"]], dtype=np.int64)
        feeds = {"input_ids": ids, "attention_mask": mask}
        if "token_type_ids" in input_names:
            feeds["token_type_ids"] = np.zeros_like(ids)
        (last_hidden,) = sess.run(["last_hidden_state"], feeds)
        emb = pool(last_hidden[0], mask[0], pool_mode, normalize)
        golden.append(
            {
                "text": text,
                "input_ids": enc["input_ids"],
                "embedding": [float(f"{v:.8g}") for v in emb],
            }
        )

    args.golden.parent.mkdir(parents=True, exist_ok=True)
    args.golden.write_text(
        json.dumps({"model": args.model_id, "cases": golden}, indent=1) + "\n"
    )
    print(f"wrote {args.golden} ({len(golden)} cases, dim {len(golden[0]['embedding'])})")

    if args.tokens_golden:
        # Token-level golden: raw last_hidden_state for a few SHORT texts,
        # validating EmbedTokens (no pooling, no normalization).
        tcases = []
        for text in GOLDEN_TEXTS[:3]:
            enc = tokenizer(text, truncation=True, max_length=max_len)
            ids = np.array([enc["input_ids"]], dtype=np.int64)
            feeds = {"input_ids": ids, "attention_mask": np.array([enc["attention_mask"]], dtype=np.int64)}
            if "token_type_ids" in input_names:
                feeds["token_type_ids"] = np.zeros_like(ids)
            (last_hidden,) = sess.run(["last_hidden_state"], feeds)
            tcases.append({
                "text": text,
                "input_ids": enc["input_ids"],
                "hidden": [[float(f"{v:.8g}") for v in row] for row in last_hidden[0]],
            })
        args.tokens_golden.parent.mkdir(parents=True, exist_ok=True)
        args.tokens_golden.write_text(json.dumps({"model": args.model_id, "cases": tcases}, indent=1) + "\n")
        print(f"wrote {args.tokens_golden} ({len(tcases)} cases)")


if __name__ == "__main__":
    main()
