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
	"fmt"
	"os"
	"path/filepath"
	"strings"

	"github.com/rostamlabs/rembed/internal/hub"
	"github.com/rostamlabs/rembed/internal/model"
	"github.com/rostamlabs/rembed/tokenizer"
)

// Embedder turns texts into fixed-size embedding vectors. It is safe for
// concurrent use. Latency is the default optimization target: each Embed
// call fans out across up to GOMAXPROCS cores and keeps a spinning worker
// pool for its duration, which trades idle-core burn for wall time — see
// WithWorkers to cap that for throughput-saturated servers.
type Embedder struct {
	cfg model.Config
	tok *tokenizer.Tokenizer
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
//     (optionally prefixed "hf:"): the files are downloaded straight from
//     the Hub in pure Go into $REMBED_CACHE (default: the user cache dir)
//     and reused on later loads. Set HF_TOKEN for gated repos.
func Load(ref string, opts ...Option) (*Embedder, error) {
	var o loadOptions
	for _, opt := range opts {
		opt(&o)
	}
	name := strings.TrimPrefix(ref, "hf:")
	modelDir := name
	if fi, statErr := os.Stat(modelDir); statErr != nil || !fi.IsDir() {
		if !hub.IsModelID(name) {
			return nil, fmt.Errorf("rembed: %q is neither a model directory nor a Hugging Face model id", ref)
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
	tok, err := tokenizer.New(filepath.Join(modelDir, "vocab.txt"), cfg.DoLowerCase, cfg.ClsToken, cfg.SepToken, cfg.UnkToken)
	if err != nil {
		return nil, fmt.Errorf("rembed: %w", err)
	}
	if tok.VocabSize() != cfg.VocabSize {
		// A vocab.txt and model.safetensors from different models would
		// otherwise produce silently wrong (or per-call failing) embeddings.
		return nil, fmt.Errorf("rembed: vocab.txt has %d tokens but manifest says %d — mismatched model dir", tok.VocabSize(), cfg.VocabSize)
	}
	m, err := model.Load(filepath.Join(modelDir, "model.safetensors"), cfg, o.int8, o.workers)
	if err != nil {
		return nil, fmt.Errorf("rembed: %w", err)
	}
	return &Embedder{cfg: cfg, tok: tok, m: m}, nil
}

// Embed returns one embedding per input text, each of length Dim(),
// L2-normalized when the model manifest says so (true for the
// sentence-transformers models). Texts are embedded independently; the
// context is checked between texts.
func (e *Embedder) Embed(ctx context.Context, texts []string) ([][]float32, error) {
	out := make([][]float32, len(texts))
	// The manifest owns the sequence ceiling (position-embedding count);
	// tokenizer.MaxSeqLen is only that package's standalone default.
	maxLen := e.cfg.MaxPositionEmbeddings
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

// Tokenize exposes the tokenizer's input_ids for one text. It exists for the
// validation harness (attributing golden mismatches to tokenization vs
// numerics) and debugging; it is not a stable part of the embedding API.
func (e *Embedder) Tokenize(text string) []int64 {
	ids, _ := e.tok.Encode(text, e.cfg.MaxPositionEmbeddings)
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
