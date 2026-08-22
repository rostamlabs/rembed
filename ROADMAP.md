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

## R6 — XLM-R + SentencePiece unigram (multilingual-e5)
The multilingual unlock. Needs a pure-Go SentencePiece unigram tokenizer
(minimal protobuf reader for the .model file + Viterbi segmentation),
then XLM-R is architecturally RoBERTa. Validated token-for-token against
HF across scripts (CJK, Cyrillic, Arabic, emoji).

## R7 — int8 depth: AVX-512-VNNI and activation quantization
VNNI (VPDPBUSD) computes int8×int8→int32 4-wide-per-lane — on capable
CPUs this roughly doubles int8 throughput over the current
dequantize-then-FMA scheme. Static activation quantization is measured
territory: adopt only if it beats weight-only on the harness (ORT's
dynamic-quant overhead is why weight-only currently wins at short seq).

## R8 — Publishable benchmark + beyond-CPU
A pinned cloud box turns the consistent sign-test wins over ONNX Runtime
into publishable magnitudes (DESIGN.md rule; laptop numbers stay
relative-only). GPU execution remains explicitly out of scope until the
CPU story is complete — revisit after R1–R7.

## Standing rules
Every rung: golden validation against an independent reference, its own
benchmark delta where perf-relevant, an adversarial review pass before
merge, honest RESULTS.md entries including failed experiments.
