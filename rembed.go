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
	"path/filepath"

	"github.com/rostamlabs/rembed/internal/model"
	"github.com/rostamlabs/rembed/tokenizer"
)

// Embedder turns texts into fixed-size embedding vectors. It is safe for
// concurrent use.
type Embedder struct {
	cfg model.Config
	tok *tokenizer.Tokenizer
	m   *model.Model
}

// Load opens a model directory produced by models/convert.py, containing
// model.safetensors, vocab.txt, and manifest.json.
func Load(modelDir string) (*Embedder, error) {
	cfg, err := model.LoadConfig(filepath.Join(modelDir, "manifest.json"))
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
	m, err := model.Load(filepath.Join(modelDir, "model.safetensors"), cfg)
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

// Dim returns the embedding dimensionality.
func (e *Embedder) Dim() int { return e.cfg.HiddenSize }
