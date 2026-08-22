// SPDX-License-Identifier: Apache-2.0

// Package sentencepiece implements the SentencePiece Unigram tokenizer
// used by the XLM-RoBERTa family (multilingual-e5, paraphrase-multilingual
// MiniLM, …) in pure Go: a minimal protobuf reader for the .model file,
// the NMT-NFKC normalizer driven by the model's precompiled charsmap, and
// Viterbi segmentation over the piece vocabulary — with HF's fairseq id
// remapping on top. Validated token-for-token against HF's
// XLMRobertaTokenizer over the committed fixture, and byte-for-byte
// against sentencepiece's own normalizer.
//
// Known, DELIBERATE divergence from HF's fast (Rust) tokenizer: NFD
// (decomposed) Hangul and kana. tokenizers-rs applies the charsmap per
// grapheme cluster and skips clusters ≥ 6 bytes, so decomposed Korean
// from e.g. macOS never gets NFC-composed there and shreds into jamo.
// This implementation matches the sentencepiece C++ reference (and the
// slow Python tokenizer), which composes them — 65k-input fuzzing
// against the reference found zero mismatches, with every HF-fast
// disagreement being HF's known quirk. The NFD fixture pins this.
package sentencepiece

import (
	"fmt"
	"math"
	"unicode/utf8"
)

// Tokenizer is a loaded sentencepiece.bpe.model ready to encode. Safe for
// concurrent use after New: all state is read-only.
type Tokenizer struct {
	norm      *normalizer
	pieceID   map[string]int32 // NORMAL pieces only → sentencepiece id
	pieceText []string         // by sentencepiece id
	scores    []float32        // by sentencepiece id
	maxPiece  int              // longest piece in bytes
	unkScore  float32
	nPieces   int

	// HF/fairseq id space: 0=<s> 1=<pad> 2=</s> 3=<unk>, sp ids shift +1,
	// <mask> appended last.
	cls, sep, unk int64
}

// unkPenalty matches unigram_model.cc: the unknown-character score is the
// minimum piece score minus 10.
const unkPenalty = 10.0

// New loads a SentencePiece Unigram model (the sentencepiece.bpe.model
// file XLM-R-family repos ship).
func New(modelPath string) (*Tokenizer, error) {
	m, err := parseModel(modelPath)
	if err != nil {
		return nil, err
	}
	norm, err := newNormalizer(m.norm)
	if err != nil {
		return nil, err
	}
	t := &Tokenizer{
		norm:      norm,
		pieceID:   make(map[string]int32, len(m.pieces)),
		pieceText: make([]string, len(m.pieces)),
		scores:    make([]float32, len(m.pieces)),
		nPieces:   len(m.pieces),
	}
	minScore := float32(math.MaxFloat32)
	for id, p := range m.pieces {
		t.scores[id] = p.score
		t.pieceText[id] = p.text
		switch p.kind {
		case 1: // NORMAL pieces participate in matching
			if _, dup := t.pieceID[p.text]; dup {
				return nil, fmt.Errorf("sentencepiece: %s: duplicate piece %q", modelPath, p.text)
			}
			t.pieceID[p.text] = int32(id)
			if len(p.text) > t.maxPiece {
				t.maxPiece = len(p.text)
			}
			if p.score < minScore {
				minScore = p.score
			}
		case 2, 3: // UNKNOWN, CONTROL — never matched against text
		default:
			// USER_DEFINED (4), UNUSED (5), and BYTE (6) pieces carry
			// semantics this implementation does not reproduce (special
			// matching rules, score bonuses, byte fallback) — loading such
			// a model would tokenize differently from sentencepiece, so
			// refuse loudly (none of the XLM-R-family models use them).
			return nil, fmt.Errorf("sentencepiece: %s: piece %q has unsupported type %d (USER_DEFINED/UNUSED/BYTE models are not supported)", modelPath, p.text, p.kind)
		}
	}
	if t.maxPiece == 0 {
		return nil, fmt.Errorf("sentencepiece: %s: no normal pieces", modelPath)
	}
	t.unkScore = minScore - unkPenalty
	// fairseq framing ids are fixed by construction in XLMRobertaTokenizer.
	t.cls, t.sep, t.unk = 0, 2, 3
	return t, nil
}

// VocabSize is the HF tokenizer's id-space size: every sentencepiece id
// shifted by the fairseq offset (+1), plus the appended <mask>. The
// model's embedding table may be padded LARGER than this (the multilingual
// MiniLM ships 250037 rows for 250002 ids); the loader allows that.
func (t *Tokenizer) VocabSize() int { return t.nPieces + 2 }

// hfID maps a sentencepiece id to HF's fairseq-offset id space.
func (t *Tokenizer) hfID(spID int32) int64 {
	switch spID {
	case 0: // <unk>
		return t.unk
	case 1: // <s>
		return t.cls
	case 2: // </s>
		return t.sep
	}
	return int64(spID) + 1
}

// span is one Viterbi segment: s[start:end] tokenized as sp id.
type span struct {
	start, end int32
	id         int32
}

// segment runs Viterbi over the normalized text: the best-scoring
// segmentation into vocabulary pieces, with unknown single characters
// scored at unkScore (ported from unigram_model.cc — an unk node is added
// only when no single-character piece covers the position, and CONSECUTIVE
// unknown characters merge into a single unk piece, exactly as
// sentencepiece emits them).
func (t *Tokenizer) segment(s string) []span {
	n := len(s)
	if n == 0 {
		return nil
	}
	const negInf = float32(-1e30)
	bestScore := make([]float32, n+1)
	bestEdge := make([]int32, n+1) // sp id of the piece ending at this pos
	bestFrom := make([]int32, n+1)
	for i := 1; i <= n; i++ {
		bestScore[i] = negInf
		bestFrom[i] = -1
	}
	for i := 0; i < n; i++ {
		if bestFrom[i] == -1 && i > 0 {
			continue // unreachable (mid-rune positions)
		}
		base := bestScore[i]
		// One-character (rune) span: piece if known, unk node otherwise.
		_, runeLen := utf8.DecodeRuneInString(s[i:])
		hasSingle := false
		limit := min(n, i+t.maxPiece)
		for j := i + 1; j <= limit; j++ {
			id, ok := t.pieceID[s[i:j]]
			if !ok {
				continue
			}
			if j-i == runeLen {
				hasSingle = true
			}
			if sc := base + t.scores[id]; sc > bestScore[j] {
				bestScore[j] = sc
				bestEdge[j] = id
				bestFrom[j] = int32(i)
			}
		}
		if !hasSingle {
			j := i + runeLen
			if sc := base + t.unkScore; sc > bestScore[j] {
				bestScore[j] = sc
				bestEdge[j] = 0 // sp <unk>
				bestFrom[j] = int32(i)
			}
		}
	}
	// Backtrack into spans.
	var rev []span
	for pos := int32(n); pos > 0; {
		from := bestFrom[pos]
		if from < 0 {
			// Cannot happen: every position is reachable through unk
			// nodes; guard against a malformed model anyway.
			return []span{{0, int32(n), 0}}
		}
		rev = append(rev, span{start: from, end: pos, id: bestEdge[pos]})
		pos = from
	}
	out := make([]span, 0, len(rev))
	for i := len(rev) - 1; i >= 0; i-- {
		sp := rev[i]
		// Merge runs of unknown characters into one unk piece.
		if sp.id == 0 && len(out) > 0 && out[len(out)-1].id == 0 && out[len(out)-1].end == sp.start {
			out[len(out)-1].end = sp.end
			continue
		}
		out = append(out, sp)
	}
	return out
}

// Encode tokenizes text as XLMRobertaTokenizer frames it — <s> pieces
// </s> — truncating to maxLen total ids. The mask is all ones (rembed
// never pads).
func (t *Tokenizer) Encode(text string, maxLen int) (ids, mask []int64) {
	ids = append(ids, t.cls)
	budget := max(maxLen-2, 0)
	norm := t.norm.normalize(text)
	// Bound the Viterbi work by what the budget can consume: a piece is
	// at least one byte, so (budget+margin)·maxPiece bytes of normalized
	// text always yields more than budget pieces. Without this, one 32 MB
	// request buys ~1.5 s of CPU and ~135 MB of allocation to produce 512
	// ids (measured by the review). Only inputs whose normalized form
	// exceeds the cap can theoretically differ from HF near the ceiling,
	// and those are far past the model's sequence limit anyway.
	if cap := (budget + 16) * t.maxPiece; len(norm) > cap {
		for cap > 0 && !utf8.RuneStart(norm[cap]) {
			cap--
		}
		norm = norm[:cap]
	}
	for _, sp := range t.segment(norm) {
		if budget <= 0 {
			break
		}
		ids = append(ids, t.hfID(sp.id))
		budget--
	}
	ids = append(ids, t.sep)
	mask = make([]int64, len(ids))
	for i := range mask {
		mask[i] = 1
	}
	return ids, mask
}

// Normalize exposes the model's text normalization (for tests).
func (t *Tokenizer) Normalize(text string) string { return t.norm.normalize(text) }

// Pieces exposes the sentencepiece-id segmentation of normalized text
// (for tests): the piece strings, pre-fairseq-mapping.
func (t *Tokenizer) Pieces(text string) []string {
	norm := t.norm.normalize(text)
	spans := t.segment(norm)
	out := make([]string, len(spans))
	for i, sp := range spans {
		if sp.id == 0 {
			// Unknown pieces surface as the raw text they cover, like
			// sentencepiece's EncodeAsPieces.
			out[i] = norm[sp.start:sp.end]
			continue
		}
		out[i] = t.pieceText[sp.id]
	}
	return out
}
