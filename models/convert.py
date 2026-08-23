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
    "سلام دنیا! جستجوی معنایی به زبان فارسی هم کار می‌کند.",
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
    elif mode == "lasttoken":
        # Qwen3-Embedding: the last non-padded token (the appended
        # <|endoftext|>). Per-text with no padding, that is simply the
        # final row.
        idx = int(mask.sum()) - 1
        pooled = last_hidden[idx].astype(np.float32)
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
    from transformers import AutoTokenizer

    out = args.out or Path(__file__).resolve().parent / args.model_id.split("/")[-1]
    out.mkdir(parents=True, exist_ok=True)

    def fetch(filename: str) -> Path:
        return Path(hf_hub_download(args.model_id, filename))

    # --- model dir ---------------------------------------------------------
    # A single model.safetensors is the common case; large checkpoints
    # (Qwen3-4B/8B) ship a sharded set named by model.safetensors.index.json.
    from huggingface_hub.errors import EntryNotFoundError as _ENF
    try:
        st = fetch("model.safetensors")
        sanity_check_safetensors(st)
        shutil.copyfile(st, out / "model.safetensors")
    except _ENF:
        idxp = fetch("model.safetensors.index.json")
        shutil.copyfile(idxp, out / "model.safetensors.index.json")
        weight_map = json.loads(idxp.read_text())["weight_map"]
        for shard in sorted(set(weight_map.values())):
            # The index is downloaded/untrusted: a shard name is joined into
            # output paths, so reject anything that is not a plain filename.
            if shard in ("", ".", "..") or "/" in shard or "\\" in shard:
                raise SystemExit(f"unsafe shard name in index: {shard!r}")
            sp = fetch(shard)
            sanity_check_safetensors(sp)
            shutil.copyfile(sp, out / shard)

    config = json.loads(fetch("config.json").read_text())
    # The sentencepiece.bpe.model FILE is the authority — tokenizer_class
    # is unreliable in real repos (the multilingual MiniLM says
    # PreTrainedTokenizerFast).
    from huggingface_hub.errors import EntryNotFoundError
    try:
        fetch("sentencepiece.bpe.model")
        sentencepiece_tok = True
    except EntryNotFoundError:
        sentencepiece_tok = False
    # SentencePiece models (XLM-R tokenizer, any architecture) ship
    # sentencepiece.bpe.model; byte-level BPE (RoBERTa) ships vocab.json +
    # merges.txt; WordPiece (BERT, MPNet) ships vocab.txt.
    if sentencepiece_tok:
        shutil.copyfile(fetch("sentencepiece.bpe.model"), out / "sentencepiece.bpe.model")
    elif config.get("model_type") == "roberta":
        shutil.copyfile(fetch("vocab.json"), out / "vocab.json")
        shutil.copyfile(fetch("merges.txt"), out / "merges.txt")
    elif config.get("model_type") in ("modernbert", "qwen3"):
        # ModernBERT and Qwen3 ship only tokenizer.json (byte-level BPE,
        # vocab and merges embedded); no vocab.json/merges.txt.
        shutil.copyfile(fetch("tokenizer.json"), out / "tokenizer.json")
    else:
        shutil.copyfile(fetch("vocab.txt"), out / "vocab.txt")
    tok_config = json.loads(fetch("tokenizer_config.json").read_text())
    pooling = json.loads(fetch("1_Pooling/config.json").read_text())
    modules = json.loads(fetch("modules.json").read_text())

    model_type = config.get("model_type")
    if model_type not in ("bert", "distilbert", "modernbert", "qwen3", "roberta", "mpnet"):
        raise SystemExit(f"model_type={model_type!r}: only bert, distilbert, modernbert, qwen3, roberta, and mpnet models are supported")
    if model_type == "qwen3":
        # Causal decoder embedder: SwiGLU (silu), full causal attention.
        # Refuse a rope scaling (YaRN) or sliding window — geometry changes
        # rembed's path does not implement.
        if config.get("hidden_act", "silu") != "silu":
            raise SystemExit(f"hidden_act={config.get('hidden_act')!r}: qwen3 supports only silu")
        if config.get("attention_bias"):
            raise SystemExit("qwen3 with attention_bias is not supported (rembed's qwen3 path is bias-free)")
        if config.get("rope_scaling"):
            raise SystemExit(f"qwen3 rope_scaling={config.get('rope_scaling')!r} is not supported")
        if config.get("use_sliding_window") or config.get("sliding_window"):
            raise SystemExit("qwen3 sliding-window attention is not supported")
    if model_type == "modernbert":
        # rembed's ModernBERT path is bias-free; refuse a checkpoint that
        # enabled any bias (it would silently be dropped). Positions enter
        # via RoPE, so position_embedding_type is a vestige and is ignored.
        if config.get("norm_bias") or config.get("attention_bias") or config.get("mlp_bias"):
            raise SystemExit("modernbert with norm_bias/attention_bias/mlp_bias is not supported")
        if config.get("hidden_activation", "gelu") != "gelu":
            raise SystemExit(f"hidden_activation={config.get('hidden_activation')!r}: only exact GELU is supported")
    if model_type == "distilbert":
        # Fold DistilBERT's config keys into the BERT names.
        if config.get("sinusoidal_pos_embds"):
            raise SystemExit("sinusoidal_pos_embds=true is not supported")
        missing = [k for k in ("dim", "n_layers", "n_heads", "hidden_dim") if k not in config]
        if missing:
            raise SystemExit(f"distilbert config.json lacks {missing}")
        config = dict(config,
                      hidden_size=config["dim"], num_hidden_layers=config["n_layers"],
                      num_attention_heads=config["n_heads"], intermediate_size=config["hidden_dim"],
                      hidden_act=config.get("activation", "gelu"),
                      # HF hardcodes DistilBERT's LayerNorm eps; drop any stray key.
                      layer_norm_eps=1e-12)
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
    if model_type != "qwen3" and config.get("hidden_act", "gelu") != "gelu":
        raise SystemExit(f"hidden_act={config.get('hidden_act')!r}: only exact GELU is supported")
    if config.get("position_embedding_type", "absolute") != "absolute":
        raise SystemExit(f"position_embedding_type={config.get('position_embedding_type')!r}: only absolute is supported")
    mean = bool(pooling.get("pooling_mode_mean_tokens"))
    cls = bool(pooling.get("pooling_mode_cls_token"))
    last = bool(pooling.get("pooling_mode_lasttoken"))
    other = any(pooling.get(k) for k in (
        "pooling_mode_max_tokens",
        "pooling_mode_mean_sqrt_len_tokens", "pooling_mode_weightedmean_tokens"))
    if other or (mean + cls + last) != 1:
        raise SystemExit(f"unsupported pooling config (rembed supports mean, cls, or lasttoken): {pooling}")
    pool_mode = "mean" if mean else "cls" if cls else "lasttoken"
    # Any ST module beyond Transformer/Pooling/Normalize (e.g. a Dense head)
    # would make the real model differ from what rembed computes.
    known = ("models.Transformer", "models.Pooling", "models.Normalize")
    unknown = [m["type"] for m in modules if not m.get("type", "").endswith(known)]
    if unknown:
        raise SystemExit(f"unsupported sentence-transformers modules: {unknown}")
    normalize = any(m.get("type", "").endswith("models.Normalize") for m in modules)

    def tok_str(v, default):
        # HF writes special tokens either as plain strings or as
        # AddedToken objects ({"content": ..., "lstrip": ...}).
        if isinstance(v, dict):
            return v.get("content", default)
        return v if v is not None else default

    manifest = {
        "name": args.model_id,
        "hidden_size": config["hidden_size"],
        "num_hidden_layers": config["num_hidden_layers"],
        "num_attention_heads": config["num_attention_heads"],
        "intermediate_size": config["intermediate_size"],
        "vocab_size": config["vocab_size"],
        "max_position_embeddings": config["max_position_embeddings"],
        "layer_norm_eps": config.get("layer_norm_eps", 1e-12),
        "do_lower_case": bool(tok_config.get("do_lower_case", False)) if (model_type in ("roberta", "modernbert", "qwen3") or sentencepiece_tok) else bool(tok_config.get("do_lower_case", True)),
        "cls_token": tok_str(tok_config.get("cls_token"), "[CLS]"),
        "sep_token": tok_str(tok_config.get("sep_token"), "[SEP]"),
        "unk_token": tok_str(tok_config.get("unk_token"), "<unk>" if (model_type == "roberta" or sentencepiece_tok) else "[UNK]"),
        "pooling": pool_mode,
        "normalize": normalize,
    }
    if model_type == "modernbert":
        # ModernBERT: RoPE positions (no learned table), pre-norm bias-free
        # LayerNorm (norm_eps, not layer_norm_eps), GeGLU, and a global/local
        # attention schedule — all read from config, unlike MPNet's hardcoded
        # knobs.
        manifest["model_type"] = "modernbert"
        manifest["layer_norm_eps"] = config["norm_eps"]
        manifest["global_attn_every_n_layers"] = config["global_attn_every_n_layers"]
        manifest["local_attention"] = config["local_attention"]
        manifest["global_rope_theta"] = config["global_rope_theta"]
        manifest["local_rope_theta"] = config["local_rope_theta"]
    if model_type == "qwen3":
        # Qwen3 causal decoder embedder: RMSNorm (rms_norm_eps), explicit
        # head_dim (!= hidden/heads), GQA (num_key_value_heads), SwiGLU, and
        # last-token pooling. Framing appends <|endoftext|> (the token
        # config.eos_token_id points at, which last-token pooling reads);
        # there is no CLS prefix.
        manifest["model_type"] = "qwen3"
        manifest["layer_norm_eps"] = config["rms_norm_eps"]
        manifest["head_dim"] = config["head_dim"]
        manifest["num_key_value_heads"] = config["num_key_value_heads"]
        manifest["rope_theta"] = config["rope_theta"]
        manifest["cls_token"] = ""
        manifest["sep_token"] = "<|endoftext|>"
    if model_type == "distilbert":
        manifest["model_type"] = "distilbert"
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
    if sentencepiece_tok:
        manifest["tokenizer"] = "sentencepiece"
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    vocab_files = ("sentencepiece.bpe.model" if sentencepiece_tok
                   else "vocab.json merges.txt" if model_type == "roberta"
                   else "tokenizer.json" if model_type in ("modernbert", "qwen3") else "vocab.txt")
    print(f"wrote {out}/[model.safetensors {vocab_files} manifest.json]")

    # --- golden reference (per-text, no padding) --------------------------
    # ModernBERT has no ONNX Runtime golden here (the reference is the
    # canonical torch ModernBertModel, eager attention, fp32); every other
    # architecture uses the repo's ONNX export via ONNX Runtime. Both are
    # engines INDEPENDENT of the Go implementation under test — the point of
    # a golden.
    tokenizer = AutoTokenizer.from_pretrained(args.model_id)
    if model_type in ("modernbert", "qwen3"):
        import torch
        from transformers import AutoModel
        # low_cpu_mem_usage avoids the ~2× load-time spike (materializing
        # then copying) — essential for the 4B reference on a small box.
        torch_model = AutoModel.from_pretrained(
            args.model_id, attn_implementation="eager", torch_dtype=torch.float32,
            low_cpu_mem_usage=True).eval()

        def run_last_hidden(ids, mask):
            with torch.no_grad():
                out = torch_model(input_ids=torch.tensor(ids), attention_mask=torch.tensor(mask))
            return out.last_hidden_state.numpy()

        input_names = set()
    else:
        import onnxruntime as ort
        sess = ort.InferenceSession(fetch("onnx/model.onnx"), providers=["CPUExecutionProvider"])
        input_names = {i.name for i in sess.get_inputs()}

        def run_last_hidden(ids, mask):
            feeds = {"input_ids": ids, "attention_mask": mask}
            if "token_type_ids" in input_names:
                feeds["token_type_ids"] = np.zeros_like(ids)
            (last_hidden,) = sess.run(["last_hidden_state"], feeds)
            return last_hidden

    max_len = manifest["max_position_embeddings"]
    if model_type in ("mpnet", "roberta"):
        max_len -= config.get("pad_token_id", 1) + 1

    golden = []
    for text in GOLDEN_TEXTS:
        enc = tokenizer(text, truncation=True, max_length=max_len)
        ids = np.array([enc["input_ids"]], dtype=np.int64)
        mask = np.array([enc["attention_mask"]], dtype=np.int64)
        last_hidden = run_last_hidden(ids, mask)
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
            mask = np.array([enc["attention_mask"]], dtype=np.int64)
            last_hidden = run_last_hidden(ids, mask)
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
