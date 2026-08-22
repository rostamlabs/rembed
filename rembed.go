// SPDX-License-Identifier: Apache-2.0

// Package rembed is a pure-Go embedding inference engine for BERT-family
// encoder models: text in, L2-normalized embedding vectors out. No cgo, no
// ONNX Runtime.
//
//	emb, err := rembed.Load("models/all-MiniLM-L6-v2")
//	vecs, err := emb.Embed(ctx, []string{"hello world"})
package rembed

import (
	"context"
	"errors"
	"fmt"
	"os"
	"path/filepath"
	"runtime"
	"strings"

	"github.com/rostamlabs/rembed/internal/hub"
	"github.com/rostamlabs/rembed/internal/model"
	"github.com/rostamlabs/rembed/internal/tensor"
	"github.com/rostamlabs/rembed/tokenizer"
	"github.com/rostamlabs/rembed/tokenizer/bpe"
)

// textTokenizer is what an Embedder needs from a tokenizer; the WordPiece
// (BERT/MPNet) and byte-level BPE (RoBERTa) implementations both satisfy
// it, and Load picks by the model's architecture.
type textTokenizer interface {
	Encode(text string, maxLen int) (ids, mask []int64)
	VocabSize() int
}

// Embedder turns texts into fixed-size embedding vectors. It is safe for
// concurrent use. Latency is the default optimization target: each Embed
// call fans out across up to GOMAXPROCS cores and keeps a spinning worker
// pool for its duration, which trades idle-core burn for wall time — see
// WithWorkers to cap that for throughput-saturated servers.
type Embedder struct {
	cfg model.Config
	tok textTokenizer
	m   *model.Model
}

// Option configures Load.
type Option func(*loadOptions)

type loadOptions struct {
	int8    bool
	workers int
}

// WithInt8 selects weight-only int8 inference: transformer dense weights
// are quantized at load (per-output-channel symmetric scales; activations
// stay float32), cutting per-embed weight traffic ~4× (~42 MB → ~10.5 MB
// for MiniLM). Token embeddings stay fp32, so RESIDENT model memory drops
// ~1.5×, not 4×. Embeddings differ slightly from fp32 — see the int8
// golden test for the measured bound. On CPUs without AVX2+FMA the engine
// silently falls back to fp32; check Quantized() when the mode matters.
func WithInt8() Option {
	return func(o *loadOptions) { o.int8 = true }
}

// WithWorkers caps the number of CPU workers one Embed call uses.
// The default (0) uses GOMAXPROCS, minimizing single-request latency by
// keeping every core busy — including a spinning fork-join pool that burns
// idle-core cycles for the duration of each call (~10× the useful CPU at
// low concurrency). A server saturating many concurrent Embed calls should
// set a small cap; WithWorkers(1) is fully serial with zero spinning.
//
// The cap governs every fan-out: the packed SIMD path, the attention and
// GELU phases, and the unpacked fallback matmul (non-amd64, or weight
// shapes the packer rejects).
func WithWorkers(n int) Option {
	return func(o *loadOptions) { o.workers = n }
}

// Load opens a model. ref may be:
//
//   - a local directory: either a converted dir (manifest.json) or a plain
//     Hugging Face repo checkout (config.json + 1_Pooling/... — the
//     manifest is derived on the fly);
//   - a Hugging Face model id like "sentence-transformers/all-MiniLM-L6-v2"
//     (or explicitly "hf:org/name", which always means the Hub): the files
//     are downloaded straight from the Hub in pure Go into $REMBED_CACHE
//     (default: the user cache dir) and reused on later loads. Set
//     HF_TOKEN for gated repos.
//
// A ref that could be BOTH — an org/name that does not exist locally but
// whose first segment IS a local directory (e.g. a missing
// "models/all-MiniLM-L6-v2") — is treated as a missing LOCAL path: a typo
// must fail as one, not turn into silent network egress carrying HF_TOKEN.
// Use the "hf:" prefix to force the Hub in that situation.
func Load(ref string, opts ...Option) (*Embedder, error) {
	var o loadOptions
	for _, opt := range opts {
		opt(&o)
	}
	name := strings.TrimPrefix(ref, "hf:")
	forceHub := strings.HasPrefix(ref, "hf:")
	modelDir := name
	fi, statErr := os.Stat(modelDir)
	switch {
	case !forceHub && statErr == nil && fi.IsDir():
		// local directory
	case !forceHub && statErr != nil && !os.IsNotExist(statErr):
		return nil, fmt.Errorf("rembed: %q: %w", ref, statErr)
	default:
		if !hub.IsModelID(name) {
			return nil, fmt.Errorf("rembed: %q is neither a model directory nor a Hugging Face model id (org/name)", ref)
		}
		if !forceHub {
			if parent := filepath.Dir(name); parent != "." {
				if pfi, err := os.Stat(parent); err == nil && pfi.IsDir() {
					return nil, fmt.Errorf("rembed: model directory %q does not exist (its parent does — assuming a local path; use %q to load from the Hugging Face Hub)", name, "hf:"+name)
				}
			}
		}
		cache, err := hub.CacheDir()
		if err != nil {
			return nil, fmt.Errorf("rembed: %w", err)
		}
		dir, err := hub.Ensure(name, cache)
		if err != nil {
			return nil, fmt.Errorf("rembed: %w", err)
		}
		modelDir = dir
	}
	cfg, err := model.LoadConfigOrDerive(modelDir, name)
	if err != nil {
		return nil, fmt.Errorf("rembed: %w", err)
	}
	var tok textTokenizer
	if cfg.ModelType == "roberta" {
		tok, err = bpe.New(filepath.Join(modelDir, "vocab.json"), filepath.Join(modelDir, "merges.txt"),
			cfg.ClsToken, cfg.SepToken, cfg.UnkToken)
	} else {
		tok, err = tokenizer.New(filepath.Join(modelDir, "vocab.txt"), cfg.DoLowerCase, cfg.ClsToken, cfg.SepToken, cfg.UnkToken)
	}
	if err != nil {
		return nil, fmt.Errorf("rembed: %w", err)
	}
	if tok.VocabSize() != cfg.VocabSize {
		// A vocab and model.safetensors from different models would
		// otherwise produce silently wrong (or per-call failing) embeddings.
		return nil, fmt.Errorf("rembed: vocab has %d tokens but manifest says %d — mismatched model dir", tok.VocabSize(), cfg.VocabSize)
	}
	m, err := model.Load(filepath.Join(modelDir, "model.safetensors"), cfg, o.int8, o.workers)
	if err != nil {
		return nil, fmt.Errorf("rembed: %w", err)
	}
	return &Embedder{cfg: cfg, tok: tok, m: m}, nil
}

// Embed returns one embedding per input text, each of length Dim(),
// L2-normalized when the model manifest says so (true for the
// sentence-transformers models). Texts are embedded independently, and a
// batch of several texts is fanned out ACROSS texts (each forward pass
// serial or lightly parallel) — near-linear throughput scaling with zero
// padding waste, and results bit-identical to embedding one at a time.
// ctx is checked before each text's forward pass; a forward already in
// flight runs to completion. NOTE: a batch call can hold up to
// min(GOMAXPROCS, WithWorkers) scratch buffers (~25 MB each at max
// sequence length) simultaneously — size servers with WithWorkers.
func (e *Embedder) Embed(ctx context.Context, texts []string) ([][]float32, error) {
	out := make([][]float32, len(texts))
	// The manifest owns the sequence ceiling (the position-embedding count,
	// minus the offset rows MPNet reserves); tokenizer.MaxSeqLen is only
	// that package's standalone default.
	maxLen := e.cfg.MaxSeqLen()

	if len(texts) <= 1 {
		for i, text := range texts {
			if err := ctx.Err(); err != nil {
				return nil, err
			}
			ids, _ := e.tok.Encode(text, maxLen)
			vec, err := e.m.Forward(ids)
			if err != nil {
				return nil, fmt.Errorf("rembed: text %d: %w", i, err)
			}
			out[i] = vec
		}
		return out, nil
	}

	// Batch path: spread the machine across texts, splitting leftover
	// parallelism inside each forward when texts are fewer than cores.
	total := runtime.GOMAXPROCS(0)
	if e.m.Workers() > 0 {
		total = min(total, e.m.Workers())
	}
	across := min(total, len(texts))
	within := max(1, total/across)
	errs := make([]error, len(texts))
	tensor.ParallelForCap(across, len(texts), func(i int) {
		if err := ctx.Err(); err != nil {
			errs[i] = err
			return
		}
		ids, _ := e.tok.Encode(texts[i], maxLen)
		vec, err := e.m.ForwardWorkers(ids, within)
		if err != nil {
			errs[i] = fmt.Errorf("rembed: text %d: %w", i, err)
			return
		}
		out[i] = vec
	})
	return out, firstError(errs)
}

// firstError prefers cancellation (so a cancelled batch reports ctx.Err
// uniformly with the single-text path, not whichever per-text error has
// the lowest index), then the lowest-index error.
func firstError(errs []error) error {
	for _, err := range errs {
		if errors.Is(err, context.Canceled) || errors.Is(err, context.DeadlineExceeded) {
			return err
		}
	}
	for _, err := range errs {
		if err != nil {
			return err
		}
	}
	return nil
}

// TokenEmbeddings is one text's token-level output: the final encoder
// layer's hidden state for every token (ONNX Runtime's last_hidden_state),
// unpooled and unnormalized.
//
// Vectors' rows are views into a single backing array — one allocation of
// len(IDs)×Dim() float32s per text (~786 KB for a 512-token input).
// Retaining one row keeps the whole text's allocation alive, and rows are
// capacity-clamped so append on a row cannot bleed into its neighbor.
type TokenEmbeddings struct {
	IDs     []int64     // input token ids, including [CLS]/[SEP] framing
	Vectors [][]float32 // len(IDs) rows of Dim()
}

// EmbedTokens returns per-token hidden states for each text — the raw
// material for rerankers, late-interaction (ColBERT-style) retrieval, and
// custom pooling. Embed remains the API for sentence vectors; nothing here
// is pooled or normalized. Batches fan out across texts exactly like
// Embed. All results are held live for the whole call — a 256-text batch
// of long inputs is ~200 MB of hidden states.
func (e *Embedder) EmbedTokens(ctx context.Context, texts []string) ([]TokenEmbeddings, error) {
	out := make([]TokenEmbeddings, len(texts))
	maxLen := e.cfg.MaxSeqLen()
	if len(texts) <= 1 {
		for i, text := range texts {
			if err := ctx.Err(); err != nil {
				return nil, err
			}
			ids, _ := e.tok.Encode(text, maxLen)
			vecs, err := e.m.ForwardTokens(ids)
			if err != nil {
				return nil, fmt.Errorf("rembed: text %d: %w", i, err)
			}
			out[i] = TokenEmbeddings{IDs: ids, Vectors: vecs}
		}
		return out, nil
	}
	total := runtime.GOMAXPROCS(0)
	if e.m.Workers() > 0 {
		total = min(total, e.m.Workers())
	}
	across := min(total, len(texts))
	within := max(1, total/across)
	errs := make([]error, len(texts))
	tensor.ParallelForCap(across, len(texts), func(i int) {
		if err := ctx.Err(); err != nil {
			errs[i] = err
			return
		}
		ids, _ := e.tok.Encode(texts[i], maxLen)
		vecs, err := e.m.ForwardTokensWorkers(ids, within)
		if err != nil {
			errs[i] = fmt.Errorf("rembed: text %d: %w", i, err)
			return
		}
		out[i] = TokenEmbeddings{IDs: ids, Vectors: vecs}
	})
	if err := firstError(errs); err != nil {
		return nil, err
	}
	return out, nil
}

// Tokenize exposes the tokenizer's input_ids for one text. It exists for the
// validation harness (attributing golden mismatches to tokenization vs
// numerics) and debugging; it is not a stable part of the embedding API.
func (e *Embedder) Tokenize(text string) []int64 {
	ids, _ := e.tok.Encode(text, e.cfg.MaxSeqLen())
	return ids
}

// Model returns the model name from the manifest.
func (e *Embedder) Model() string { return e.cfg.Name }

// Quantized reports whether the weight-only int8 path is actually active
// (WithInt8 requested AND every dense weight packed as int8 — the engine
// falls back to fp32 per-matrix when the CPU or a shape cannot take it).
func (e *Embedder) Quantized() bool { return e.m.Quantized() }

// Dim returns the embedding dimensionality.
func (e *Embedder) Dim() int { return e.cfg.HiddenSize }
