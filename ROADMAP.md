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

## R8 — Cloud-box cross-engine benchmark ✅
Done on a user-provisioned 12-vCPU Zen 4 (EPYC Genoa) KVM guest — a
cloud VM, not bare metal, and the FLAG discipline stayed in charge.
Verdicts (mpnet, three rounds, both-orders protocol): fp32 at PARITY
with ONNX Runtime fp32; full int8 ahead of ORT fp32 in 3/3 rounds
(0.59-0.78×); full int8 at parity with ORT's own qint8-avx512-vnni
graphs (0.95-1.16×) despite rembed's kernel being 256-bit ymm. The box
also proved R7's EVEX twin on real Zen 4 (bit-identical to the VEX
path's outputs), so full int8 now covers every VNNI-capable CPU.
Remaining headroom: a zmm-wide kernel for 512-bit-native parts; a
dedicated bare-metal box would still tighten the noise bands. GPU
remains out of scope.

## R9 — ModernBERT (RoPE encoder) ✅
Done: model_type=modernbert end to end — the fifth architecture and the
first with rotary positions. Dual-theta RoPE (θ=160000 on the global
layers, every 3rd; θ=10000 on the local ones), a ±64 sliding-window mask
on the local layers, pre-norm with bias-free LayerNorms (layer 0's
attention norm is an Identity, as in HF), an embedding norm and a final
norm, and a 2-matrix GeGLU MLP. The tokenizer reuses the R5 byte-level
BPE scanner (ModernBERT's ByteLevel pre-tokenizer is the same GPT-2
regex) with two additions read from tokenizer.json: NFC normalization,
and leftmost-longest matching of the OLMo added tokens (whitespace runs
of 2–24 spaces, |||MARKER||| tokens, [unusedN], and [MASK]'s lstrip)
BEFORE the merges. Validated against the canonical PyTorch
ModernBertModel (nomic-ai/modernbert-embed-base, mean pooling): fp32
maxAbs < 1e-4 over the 13-case golden whose ~400-token case runs the
seq well past the 128-token window (global/local split and the ±64 mask
exercised end to end); the tokenizer matches HF token-for-token over a
64-case committed fixture and 6k+ adversarial fuzz inputs (zero
mismatches). int8: weight-only ≥ 0.998, but FULL int8 drops to 0.966 —
GeGLU's gated activations have outliers the per-row u8 scale can't hold,
so full int8 is not recommended for ModernBERT (same caveat as the
RoBERTa family). Parallel head fan-out is bit-identical to serial. The
original scope, for the record — needed:

The natural next architecture and a deliberate stepping
stone: it introduces rotary positions and bias-free norms while staying
inside rembed's encoder / mean-pool lane, so the golden harness,
convert.py, DeriveConfig, and the R5 byte-level BPE tokenizer all carry
over almost untouched. What is genuinely new:
- **RoPE** (`internal/model/rope.go`) — rotary applied to Q and K per
  head before the attention score, with DUAL theta selected by layer:
  θ=160000 on global layers (every 3rd — 0,3,6,…), θ=10000 on the local
  ones. This is the reusable asset R10 also needs.
- **Sliding-window attention** — the 2/3 non-global layers attend only
  ±64 tokens (local_attention=128 is the TOTAL window; HF splits it 64
  each side). Still bidirectional within the window.
- **GeGLU MLP** — two matrices, not three: Wi=Linear(H, 2·I) chunked into
  (value, gate), output Wo(gelu(value)·gate). Existing kernels do the
  matmuls; only the chunk/epilogue is new.
- **Bias-free LayerNorm + structural norms** — a no-bias LayerNorm
  variant (eps 1e-5); a LayerNorm right after the embeddings; layer 0's
  attn-norm is Identity (HF removes it to avoid double-norming); a
  final-norm after the stack. The ONLY bias in the model is the MLM
  decoder, which embeddings never touch.
Refuse-what-we-cannot-compute: the config's leftover
`position_embedding_type:"absolute"` is a documented HF vestige and must
be IGNORED (modernbert applies rotary regardless), not honored.
Tokenizer is byte-level BPE (OLMo-derived, vocab 50368, [CLS]=50281 /
[SEP]=50282 remapped onto the BPE vocab) — the R5 scanner should load it
once merges/vocab and the remapped specials are wired. Golden target:
nomic-ai/modernbert-embed-base (the de-facto ST embedding variant, mean
pooling; note its search_query:/search_document: prefixes). Sizes: base
22L/768/12h/1152, large 28L/1024/16h/2624, ctx 8192. Risk: low — the
only new math is RoPE and the window mask. Roughly one R3-sized rung.

## R10 — Qwen3-Embedding (decoder embedder)
Not started, and the largest rung on the board — the first DECODER-based
embedder, comparable in effort to R1–R8 combined, so realistically 3–4
reviewed sub-branches. It reuses the golden harness, the byte-BPE
tokenizer (Qwen2, GPT-2-style), L2 normalization, and the RoPE built in
R9; everything else is new attention math:
- **RMSNorm** (`internal/model/rmsnorm.go`) — replaces LayerNorm
  throughout, eps 1e-6, no mean-subtraction, no bias.
- **QK-norm** — Qwen3's signature addition: a per-head RMSNorm (length =
  head_dim = 128) applied to Q and to K BEFORE RoPE. Two extra norm
  weight vectors per layer.
- **GQA** — 16 query heads share 8 KV heads (group size 2), and
  head_dim=128 is INDEPENDENT of hidden/heads (16×128=2048 ≠ hidden
  1024). q_proj 1024→2048, k/v_proj 1024→1024, o_proj 2048→1024 — the
  config must carry head_dim explicitly, never infer it. KV heads
  broadcast across their group.
- **Causal mask** — rembed's first left-to-right masking; each query
  attends only to positions ≤ itself.
- **SwiGLU MLP** — three matrices (gate/up/down), down(silu(gate(x))·up(x)).
- **Last-token pooling** — a new pooling mode reading the final hidden
  state of the APPENDED `<|endoftext|>` token (id 151643, from
  config.json — NOT `<|im_end|>` 151645 from tokenizer_config, the trap).
  The tokenizer must auto-append 151643 and pooling reads that position.
- **Instruction layer** — asymmetric prompting: queries get
  `Instruct: {task}\nQuery:{text}` (no space after `Query:`), documents
  get NO prefix. Usage-layer, not model math, but load-bearing for
  correct retrieval — needs a clean API surface.
Config/derive: model_type=qwen3 / architectures=[Qwen3ForCausalLM];
carry head_dim, kv-heads, rope_theta (1e6), rms_eps; refuse a non-null
`rope_scaling` (YaRN is off in the shipped config — refuse rather than
silently ignore). Matryoshka (optional): truncate to the first N dims
then re-L2-normalize (0.6B native 1024, range 32–1024). The 0.6B is
~600M params (~1.2 GB fp32 / ~600 MB int8) — a different performance
regime where the int8/VNNI kernels stop being a nicety and become the
point, with a per-token-latency / batch-throughput benchmark story rather
than R8's. Golden target: Qwen/Qwen3-Embedding-0.6B (28L/1024 hidden/16 q
heads/8 kv/head_dim 128/3072 intermediate/vocab 151669, ctx 32768).
Risk: high — causal, GQA, and QK-norm each have their own
golden-attribution failure modes (a mis-broadcast KV head or wrong
QK-norm yields plausible-but-wrong cosines the golden matrix is built to
catch).

## Standing rules
Every rung: golden validation against an independent reference, its own
benchmark delta where perf-relevant, an adversarial review pass before
merge, honest RESULTS.md entries including failed experiments.
