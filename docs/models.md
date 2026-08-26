# Supported models

rembed covers eight encoder architectures — BERT, DistilBERT, MPNet, RoBERTa
(including XLM-RoBERTa), ModernBERT, nomic-embed, and two decoder-derived
embedders (Qwen3-Embedding and EmbeddingGemma) — across four tokenizer families
(WordPiece, byte-level BPE, SentencePiece Unigram, Gemma byte-fallback BPE).
Pass any of them to [`Load`](usage.md#loading-a-model) by Hugging Face id.

Every model below ships a **committed golden** and is validated in CI against an
independent ONNX Runtime (or PyTorch) reference. "fp32 vs reference" is the
worst-case max-absolute-difference; the int8 column is the worst-case cosine of
the weight-only int8 path vs fp32.

| Model | Pooling | fp32 vs reference | int8 (weight-only) |
|---|---|---|---|
| `sentence-transformers/all-MiniLM-L6-v2` | mean | 1.5e-7 | ≥ 0.9991 |
| `sentence-transformers/all-MiniLM-L12-v2` | mean | 1.9e-7 | in bounds |
| `sentence-transformers/paraphrase-MiniLM-L3-v2` | mean | < 1e-4 | — |
| `sentence-transformers/multi-qa-MiniLM-L6-cos-v1` | mean | 2.1e-7 | ≥ 0.998 |
| `BAAI/bge-small-en-v1.5` | cls | < 1e-4 | in bounds |
| `BAAI/bge-base-en-v1.5` | cls | 7.4e-7 | ≥ 0.995 |
| `thenlper/gte-small` | mean | 2e-3¹ | — |
| `thenlper/gte-base` | mean | 3.8e-6 | ≥ 0.988 |
| `sentence-transformers/all-mpnet-base-v2` | mean | 3.3e-7 | ≥ 0.9978 |
| `sentence-transformers/paraphrase-mpnet-base-v2` | mean | 1.2e-6 | ≥ 0.9945 |
| `sentence-transformers/all-distilroberta-v1` | mean | 3.2e-7 | ≥ 0.9985 |
| `sentence-transformers/multi-qa-distilbert-cos-v1` | mean | 2.5e-7 | ≥ 0.999 |
| `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2` | mean | 7e-7 | ≥ 0.9995 |
| `intfloat/multilingual-e5-small` | mean | 1.7e-7 | ≥ 0.9995 |
| `intfloat/multilingual-e5-base` | mean | 2.9e-7 | ≥ 0.999 |
| `Snowflake/snowflake-arctic-embed-s` | cls | 2.5e-7 | ≥ 0.995 |
| `nomic-ai/modernbert-embed-base` | mean | < 1e-4 | ≥ 0.998 |
| `nomic-ai/nomic-embed-text-v1.5` | mean | < 1e-4 | ≥ 0.996 |
| `Qwen/Qwen3-Embedding-0.6B` | last-token | < 1e-4 | ≥ 0.997 |
| `google/embeddinggemma-300m` | mean + Dense | < 1e-4 | ≥ 0.998 |

¹ `gte-small` ships as F16 weights: 2e-3 max-abs, but cosine ≥ 0.9999 and mean
abs ≤ 2e-4.

Other checkpoints of these architectures are **expected to work** without a
committed golden — the remaining e5 sizes, larger BGE/GTE, msmarco families,
other BERT/DistilBERT sentence-transformers, and the larger Qwen3-Embedding
(4B/8B, same architecture — run them cgo-free with
[`WithDiskWeights`](usage.md#options) on a machine that can't hold the fp32
weights in RAM).

## Choosing a quantization mode

- **fp32** — matches the reference within `1e-4`. The default.
- **Weight-only int8** ([`WithInt8`](usage.md#options)) — cosine ≥ 0.988 across
  *every* model above; the safe quantized choice.
- **Full int8** ([`WithInt8Activations`](usage.md#options)) — fastest on AVX-VNNI
  CPUs, but activation error is **per-checkpoint**, not per-architecture (worst
  golden cosine ranges from ≥ 0.999 down to ~0.95 on some checkpoints, e.g.
  nomic-embed 0.953, bge-base 0.959). Not recommended for the RoBERTa family,
  ModernBERT, nomic-embed, or Qwen3 last-token pooling — prefer weight-only int8
  when in doubt.

## Retrieval prefixes

rembed embeds **exactly** the text you pass — it adds no prefixes. Models trained
with instruction/role prefixes expect you to add them yourself:

| Model family | Prefix |
|---|---|
| e5 | `"query: "` / `"passage: "` |
| BGE | `"Represent this sentence for searching relevant passages: "` (queries) |
| Qwen3-Embedding | `"Instruct: {task}\nQuery:{text}"` on queries only; documents bare |
| arctic-embed | its own query prefix |

## Tokenizer fidelity

The WordPiece tokenizer reproduces Hugging Face's `BasicTokenizer` exactly —
NFD accent stripping, CJK space-padding, BERT's punctuation predicate, and
`_clean_text` parity: control and format characters (`Cc`/`Cf`/`Co`/`Cs`),
including U+200C ZWNJ, are removed (an earlier build that *kept* ZWNJ failed the
Persian golden).

The SentencePiece tokenizer (the XLM-RoBERTa family) has one **deliberate**
divergence: on NFD-decomposed Hangul/kana it matches the SentencePiece C++
reference (recomposing NFD) rather than Hugging Face's fast tokenizer (which
shreds Korean into jamo). A 65k-input fuzz run finds zero unintended mismatches.
