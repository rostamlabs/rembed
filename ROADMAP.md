# rembed roadmap — ONNX feature completeness

Goal: any user running sentence-embedding inference on ONNX Runtime can
switch to rembed with no feature they miss — and better performance. The
gap analysis behind this list is the ONNX-vs-rembed audit of 2026-08-22;
each milestone lands as its own reviewed PR with golden validation, in
the discipline established by DESIGN.md (nothing merges without an
independent reference to validate against).

Already at parity or ahead (see README/RESULTS.md): CPU fp32 at measured
parity, weight-only int8 measured ahead, built-in HF-faithful WordPiece
tokenization (ORT ships none), pure-Go Hub model fetching with sha256
verification, OpenAI-compatible serving, Python bindings, one static
binary, per-model committed golden references validated in CI.

## R1 — Token-level outputs (`EmbedTokens`)
ORT returns `last_hidden_state`; rembed only returned pooled vectors.
Expose per-token hidden states so rerankers, late-interaction retrieval
(ColBERT-style), and custom pooling work. Validated against a committed
token-level ONNX golden.

## R2 — Batched throughput mode
`Embed` with many texts fans the TEXTS across workers (each forward pass
serial) instead of running texts sequentially with per-text fan-outs:
near-linear throughput scaling for server batches with zero padding waste
— the shape CPU inference actually wants (ORT pads a batch through one
graph; on CPU that trades waste for parallelism we get for free).

## R3 — ARM64 NEON kernels
The AVX2 kernels have scalar fallbacks on ARM: correct but slow — ONNX
Runtime currently wins on Apple Silicon / Graviton. Port gemm4x16,
gemm4x16i8, and dot4 to NEON (FMLA-by-element removes even the broadcast
step; 32 vector registers fit the whole 4×16 tile). Verified under
qemu-aarch64 locally and in CI, benchmarked on real ARM when available.

## R4 — MPNet architecture (all-mpnet-base-v2) ✅
Done: model_type=mpnet end to end — offset positions (pad_token_id+1),
no segment table, and the shared bucketed relative-position bias added
to every layer's attention scores (computed once per forward as a
[heads×(2·seq−1)] delta table since positions are contiguous).
Validated against the repo's ONNX export: maxAbsDiff 3.3e-7, cosine
1.0 over the 12-case golden (longest case 324 tokens, so the far
relative-position buckets and the max-distance clamp are pinned end to
end); int8 cosine ≥ 0.9978 (measured worst 0.997874, test-enforced). The
original scope, for the record — needed:
relative position bias added to attention scores (bucketed, shared
across layers) on top of absolute embeddings, RoBERTa-style position
offsets, MPNet special tokens (the tokenizer already supports custom
framing tokens). Golden via the repo's ONNX export.

## R5 — RoBERTa family + byte-level BPE tokenizer ✅
Done: pure-Go byte-level BPE (tokenizer/bpe — GPT-2 byte table, a
hand-written scanner reproducing the pre-tokenization pattern's
lookahead, ranked merges), validated token-for-token against HF's
RobertaTokenizer over a 27-case fixture hitting every pre-tokenizer
branch, plus the 12-case golden's input_ids. The encoder is BERT
untouched — only the fairseq position offset (pad+1, shared with the
MPNet plumbing) differs. all-distilroberta-v1 vs its ONNX export:
fp32 maxAbs 3.2e-7 cosine 1.0; int8 cosine ≥ 0.9985 (measured worst
0.998727, test-enforced). The hub loader now selects tokenizer files by
model_type (vocab.json+merges.txt vs vocab.txt).

## R6 — SentencePiece unigram — multilingual models ✅
Done: pure-Go SentencePiece Unigram (tokenizer/sentencepiece) — a
minimal protobuf wire reader for the .model file, the NMT-NFKC
normalizer driven by the model's precompiled charsmap (darts-clone
double-array trie, traversal ported unit-for-unit), Viterbi
segmentation with sentencepiece's consecutive-unknown merging, and HF's
fairseq id remapping. Validated in three layers against the reference —
normalization byte-for-byte, pieces exactly, ids token-for-token vs
XLMRobertaTokenizer — over a 31-case battery spanning nine scripts
(Persian included) plus a >512-token truncation case.
paraphrase-multilingual-MiniLM-L12-v2 vs its ONNX export: fp32 maxAbs
7e-7, int8 cosine ≥ 0.9995 (measured worst 0.999642, test-enforced).
The multilingual-e5 family is architecture-identical (add the
query:/passage: prefixes yourself). Tokenizer selection keys on the
sentencepiece.bpe.model FILE, not tokenizer_class — older exports omit
the field, and their stale do_lower_case=true is ignored exactly as HF
ignores it.

## R7 — AVX-VNNI full int8 (u8 activations × s8 weights) ✅
Done — adapted to real hardware: consumer Alder/Raptor Lake has no
AVX-512, so the rung targets AVX-VNNI, the 256-bit VEX form of VPDPBUSD
(Alder Lake+ client, Sapphire Rapids+ server, Zen 5+). AVX-512-VNNI-only
parts (Cascade/Ice Lake-SP, Zen 4) do NOT report it and take the
weight-only fallback; the EVEX path is future work for hardware that can
actually test it.
Go's assembler only emits the EVEX encoding — which FAULTS without
AVX-512 — so the eight VPDPBUSD are hand-encoded VEX bytes, generated
field-by-field and verified against GNU as's {vex} output (the amd64
mirror of R3's ARM WORD-encoding story). Activations quantize per row
to u8 with the +128 offset trick (correction = 128·colSum, precomputed
at pack time); weights are the same per-channel symmetric int8 as M5.
The kernel is BIT-EXACT against a pure-Go integer reference (integer
math has no rounding to hide behind), and serial/parallel outputs are
bit-identical. Opt-in via WithInt8Activations, falling back to
weight-only int8 then fp32; QuantizedActivations() reports engagement.

Measured (i9-13900H, pinned, medians of 200):
| model | vs weight-int8 | vs fp32 | worst golden cosine |
|---|---|---|---|
| MiniLM-L6 | 1.36× | 1.41× | 0.9917 |
| mpnet-base | 1.28× | 1.44× | 0.9912 |
| multilingual-MiniLM | — | — | 0.9982 |
Weight-only int8 was already at 0.89× of ONNX Runtime fp32 (M5), so
full int8 puts rembed decisively ahead on VNNI hardware. Accuracy is
the honest cost, and it is PER-MODEL (each bound test-enforced in the
golden matrix): ≥0.991 for the BERT MiniLMs and mpnet, ≥0.998 for
multilingual and gte — but 0.9747 for distilroberta, whose activation
outliers defeat the per-row absmax scheme (the review measured ~2.7×
larger row absmax at the median). Full int8 is not recommended for
RoBERTa-family checkpoints.

## R8 — Publishable benchmark + beyond-CPU
A pinned cloud box turns the consistent sign-test wins over ONNX Runtime
into publishable magnitudes (DESIGN.md rule; laptop numbers stay
relative-only). GPU execution remains explicitly out of scope until the
CPU story is complete — revisit after R1–R7.

## Standing rules
Every rung: golden validation against an independent reference, its own
benchmark delta where perf-relevant, an adversarial review pass before
merge, honest RESULTS.md entries including failed experiments.
