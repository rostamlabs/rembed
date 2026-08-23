// SPDX-License-Identifier: Apache-2.0

// Package gemma implements the Gemma tokenizer (EmbeddingGemma and the
// Gemma 3 family) in pure Go: a SentencePiece-style byte-level-fallback BPE
// read from the repo's tokenizer.json. It differs from the byte-level BPE in
// tokenizer/bpe (Qwen/ModernBERT) in three ways that make it its own family:
//
//   - Metaspace normalization: the only normalization is replacing the ASCII
//     space U+0020 with U+2581 (▁). No prefix space is prepended, so
//     "hello world" normalizes to "hello▁world" and tokenizes as
//     ["hello","▁world"] — the leading word carries no ▁.
//   - Whole-string BPE: the pre-tokenizer splits on the ASCII space, which
//     normalization has already removed, so BPE runs over the entire
//     normalized string at once (merges may cross former word boundaries).
//   - Byte fallback: a character absent from the vocab is emitted as its
//     UTF-8 bytes, each as a <0xNN> vocab token. No merge rule in the shipped
//     model references a <0xNN> token (verified at build time), so byte
//     symbols never merge further and a plain string-concatenation merge —
//     where the merged token's vocab string is exactly left+right — is exact.
//
// Framing is <bos> … <eos> (ids 2 and 1). Validated token-for-token against
// HF's tokenizer over the committed fixture.
package gemma

import (
	"container/heap"
	"encoding/json"
	"fmt"
	"os"
	"strings"
)

const metaspace = "▁" // ▁ — SentencePiece's visible space

// Tokenizer is a loaded Gemma tokenizer, safe for concurrent use after New.
type Tokenizer struct {
	vocab    map[string]int32
	rank     map[[2]string]int32 // adjacent (left,right) piece -> merge priority (lower = earlier)
	byteTok  [256]int32          // 0xNN -> id of the "<0xNN>" fallback piece
	bos, eos int32
	unk      int32
}

// tokJSON is the subset of tokenizer.json the Gemma BPE needs.
type tokJSON struct {
	Model struct {
		Vocab        map[string]int32 `json:"vocab"`
		Merges       [][]string       `json:"merges"`
		ByteFallback bool             `json:"byte_fallback"`
	} `json:"model"`
	AddedTokens []struct {
		ID      int32  `json:"id"`
		Content string `json:"content"`
	} `json:"added_tokens"`
}

// New loads a Gemma tokenizer from a tokenizer.json file.
func New(tokenizerPath string) (*Tokenizer, error) {
	raw, err := os.ReadFile(tokenizerPath)
	if err != nil {
		return nil, err
	}
	var tj tokJSON
	if err := json.Unmarshal(raw, &tj); err != nil {
		return nil, fmt.Errorf("gemma: %s: %w", tokenizerPath, err)
	}
	if len(tj.Model.Vocab) == 0 {
		return nil, fmt.Errorf("gemma: %s: empty vocab", tokenizerPath)
	}
	if !tj.Model.ByteFallback {
		return nil, fmt.Errorf("gemma: %s: byte_fallback is false — unsupported (rembed's gemma path requires byte fallback)", tokenizerPath)
	}
	t := &Tokenizer{
		vocab: tj.Model.Vocab,
		rank:  make(map[[2]string]int32, len(tj.Model.Merges)),
		bos:   -1, eos: -1, unk: -1,
	}
	for r, pair := range tj.Model.Merges {
		if len(pair) != 2 {
			return nil, fmt.Errorf("gemma: %s: merge %d is not a pair: %v", tokenizerPath, r, pair)
		}
		// Keep the earliest (highest-priority) rank if a pair repeats.
		key := [2]string{pair[0], pair[1]}
		if _, dup := t.rank[key]; !dup {
			t.rank[key] = int32(r)
		}
	}
	// Byte-fallback pieces: "<0xNN>" for every byte value.
	for b := range 256 {
		id, ok := t.vocab[fmt.Sprintf("<0x%02X>", b)]
		if !ok {
			return nil, fmt.Errorf("gemma: %s: byte fallback piece <0x%02X> missing from vocab", tokenizerPath, b)
		}
		t.byteTok[b] = id
	}
	// Framing / unk ids from the added-token table.
	for _, a := range tj.AddedTokens {
		switch a.Content {
		case "<bos>":
			t.bos = a.ID
		case "<eos>":
			t.eos = a.ID
		case "<unk>":
			t.unk = a.ID
		}
	}
	if t.bos < 0 || t.eos < 0 || t.unk < 0 {
		return nil, fmt.Errorf("gemma: %s: missing <bos>/<eos>/<unk> in added_tokens", tokenizerPath)
	}
	return t, nil
}

// VocabSize is the tokenizer's id-space size.
func (t *Tokenizer) VocabSize() int { return len(t.vocab) }

// sym is one node in the doubly-linked symbol list during BPE.
type sym struct {
	text       string
	prev, next int // -1 when absent
	alive      bool
}

// mergeCand is a pending adjacent-pair merge in the priority queue.
type mergeCand struct {
	rank int32
	left int // index of the left symbol
	ver  int // snapshot of the left symbol's mutation counter
}

type mcHeap []mergeCand

func (h mcHeap) Len() int           { return len(h) }
func (h mcHeap) Less(a, b int) bool { return h[a].rank < h[b].rank }
func (h mcHeap) Swap(a, b int)      { h[a], h[b] = h[b], h[a] }
func (h *mcHeap) Push(x any)        { *h = append(*h, x.(mergeCand)) }
func (h *mcHeap) Pop() any {
	old := *h
	n := len(old)
	c := old[n-1]
	*h = old[:n-1]
	return c
}

// bpe runs whole-string byte-fallback BPE over the normalized text and
// appends the resulting piece ids to dst.
func (t *Tokenizer) bpe(norm string, dst []int32) []int32 {
	if norm == "" {
		return dst
	}
	// Initial symbols: each rune that is a vocab piece stays whole; a rune
	// absent from the vocab is expanded to its UTF-8 bytes as <0xNN> pieces.
	syms := make([]sym, 0, len(norm))
	appendPiece := func(s string) {
		syms = append(syms, sym{text: s, alive: true})
	}
	for _, r := range norm {
		s := string(r)
		if _, ok := t.vocab[s]; ok {
			appendPiece(s)
			continue
		}
		for _, b := range []byte(s) {
			// byteTok[b] is guaranteed present; store the piece string so
			// the final lookup and any (never-occurring) merge are uniform.
			appendPiece(fmt.Sprintf("<0x%02X>", b))
		}
	}
	n := len(syms)
	ver := make([]int, n)
	for i := range syms {
		syms[i].prev = i - 1
		syms[i].next = i + 1
	}
	syms[n-1].next = -1

	h := &mcHeap{}
	pushPair := func(left int) {
		if left < 0 {
			return
		}
		right := syms[left].next
		if right < 0 {
			return
		}
		if r, ok := t.rank[[2]string{syms[left].text, syms[right].text}]; ok {
			heap.Push(h, mergeCand{rank: r, left: left, ver: ver[left]})
		}
	}
	for i := range n {
		pushPair(i)
	}
	for h.Len() > 0 {
		c := heap.Pop(h).(mergeCand)
		left := c.left
		// Stale if the left symbol mutated or died since this candidate was
		// queued, or its right neighbour is gone.
		if !syms[left].alive || ver[left] != c.ver {
			continue
		}
		right := syms[left].next
		if right < 0 || !syms[right].alive {
			continue
		}
		if r, ok := t.rank[[2]string{syms[left].text, syms[right].text}]; !ok || r != c.rank {
			continue
		}
		// Merge right into left; the merged piece's vocab string is left+right.
		syms[left].text += syms[right].text
		syms[left].next = syms[right].next
		if syms[right].next >= 0 {
			syms[syms[right].next].prev = left
		}
		syms[right].alive = false
		ver[left]++
		// New adjacencies (prev,left) and (left,next) may now be mergeable.
		pushPair(syms[left].prev)
		pushPair(left)
	}
	for i := 0; i >= 0; i = syms[i].next {
		if !syms[i].alive {
			continue
		}
		if id, ok := t.vocab[syms[i].text]; ok {
			dst = append(dst, id)
		} else {
			dst = append(dst, t.unk)
		}
	}
	return dst
}

// Encode tokenizes text as HF's Gemma tokenizer frames it — <bos> pieces
// <eos> — truncating to maxLen total ids. The mask is all ones (rembed
// never pads).
func (t *Tokenizer) Encode(text string, maxLen int) (ids, mask []int64) {
	norm := strings.ReplaceAll(text, " ", metaspace)
	pieces := t.bpe(norm, nil)
	budget := max(maxLen-2, 0)
	if len(pieces) > budget {
		pieces = pieces[:budget]
	}
	ids = make([]int64, 0, len(pieces)+2)
	ids = append(ids, int64(t.bos))
	for _, p := range pieces {
		ids = append(ids, int64(p))
	}
	ids = append(ids, int64(t.eos))
	mask = make([]int64, len(ids))
	for i := range mask {
		mask[i] = 1
	}
	return ids, mask
}

// Pieces exposes the piece-string segmentation of normalized text (for
// tests), pre-framing — mirroring HF's tokenize().
func (t *Tokenizer) Pieces(text string) []string {
	norm := strings.ReplaceAll(text, " ", metaspace)
	ids := t.bpe(norm, nil)
	inv := make(map[int32]string, len(t.vocab))
	for s, id := range t.vocab {
		inv[id] = s
	}
	out := make([]string, len(ids))
	for i, id := range ids {
		out[i] = inv[id]
	}
	return out
}
