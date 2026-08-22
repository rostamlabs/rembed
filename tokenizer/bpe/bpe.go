// SPDX-License-Identifier: Apache-2.0

// Package bpe implements the byte-level BPE tokenizer used by the
// RoBERTa/GPT-2 family, ported faithfully from HuggingFace's Python
// implementation: text is pre-tokenized with GPT-2's pattern, each
// pre-token's UTF-8 bytes are mapped through the byte-to-unicode table,
// and adjacent symbols are merged greedily by merge rank until no ranked
// pair remains. Every token-id sequence is validated token-for-token
// against HF's RobertaTokenizer by the committed fixture test.
package bpe

import (
	"bufio"
	"container/heap"
	"encoding/json"
	"fmt"
	"os"
	"strings"
	"unicode"
)

// Tokenizer is a loaded vocab.json + merges.txt byte-level BPE encoder.
// It is safe for concurrent use after New: all state is read-only.
type Tokenizer struct {
	vocab   map[string]int64
	ranks   map[[2]string]int
	byteEnc [256]rune
	cls     int64
	sep     int64
	unk     int64
}

// byteToUnicode builds GPT-2's bytes_to_unicode table: the three printable
// Latin-1 ranges map to themselves, every other byte b maps to 256+n in
// first-free order — a reversible byte→char code that keeps merge files
// printable and whitespace-free.
func byteToUnicode() [256]rune {
	var enc [256]rune
	printable := func(b int) bool {
		return (b >= '!' && b <= '~') || (b >= 0xA1 && b <= 0xAC) || (b >= 0xAE && b <= 0xFF)
	}
	n := 0
	for b := range 256 {
		if printable(b) {
			enc[b] = rune(b)
		} else {
			enc[b] = rune(256 + n)
			n++
		}
	}
	return enc
}

// New loads a byte-level BPE tokenizer. clsTok/sepTok/unkTok are the
// framing specials from the model config (RoBERTa: "<s>", "</s>",
// "<unk>"); they must exist in the vocab.
func New(vocabPath, mergesPath, clsTok, sepTok, unkTok string) (*Tokenizer, error) {
	raw, err := os.ReadFile(vocabPath)
	if err != nil {
		return nil, err
	}
	var vocab map[string]int64
	if err := json.Unmarshal(raw, &vocab); err != nil {
		return nil, fmt.Errorf("bpe: %s: %w", vocabPath, err)
	}

	f, err := os.Open(mergesPath)
	if err != nil {
		return nil, err
	}
	defer func() { _ = f.Close() }()
	ranks := make(map[[2]string]int)
	sc := bufio.NewScanner(f)
	sc.Buffer(make([]byte, 1<<16), 1<<20)
	rank := 0
	for sc.Scan() {
		line := sc.Text()
		// The header line and any blank tail lines carry no merge.
		if line == "" || strings.HasPrefix(line, "#version") {
			continue
		}
		a, b, ok := strings.Cut(line, " ")
		if !ok || a == "" || b == "" {
			return nil, fmt.Errorf("bpe: %s: malformed merge line %q", mergesPath, line)
		}
		ranks[[2]string{a, b}] = rank
		rank++
	}
	if err := sc.Err(); err != nil {
		return nil, fmt.Errorf("bpe: %s: %w", mergesPath, err)
	}
	if len(ranks) == 0 {
		return nil, fmt.Errorf("bpe: %s: no merges", mergesPath)
	}

	t := &Tokenizer{vocab: vocab, ranks: ranks, byteEnc: byteToUnicode()}
	lookup := func(tok, what string) (int64, error) {
		id, ok := vocab[tok]
		if !ok {
			return 0, fmt.Errorf("bpe: %s token %q not in %s", what, tok, vocabPath)
		}
		return id, nil
	}
	if t.cls, err = lookup(clsTok, "cls"); err != nil {
		return nil, err
	}
	if t.sep, err = lookup(sepTok, "sep"); err != nil {
		return nil, err
	}
	if t.unk, err = lookup(unkTok, "unk"); err != nil {
		return nil, err
	}
	return t, nil
}

// VocabSize returns the number of entries in vocab.json.
func (t *Tokenizer) VocabSize() int { return len(t.vocab) }

// isNum reports \p{N} membership (all Number categories — Nd, Nl, No)
// via unicode.IsNumber. The only divergence from the reference pattern is
// Unicode-database VERSION skew: Python's regex module bundles a newer
// UCD than Go's unicode tables, so a handful of recently-encoded scripts
// classify differently — and HF's actual tokenizer (Rust, its own UCD)
// sides with Go on the fuzzed cases, so token ids agree even there.
func isNum(r rune) bool { return unicode.IsNumber(r) }

// isSpace reports \s membership for the pattern. Verified by exhaustive
// code-point enumeration: the regex module's \s (what GPT-2/HF actually
// use — NOT Python's stdlib re, which adds \x1c-\x1f) and Go's
// unicode.IsSpace are IDENTICAL, zero differences.
func isSpace(r rune) bool { return unicode.IsSpace(r) }

// preTokenize splits text with GPT-2's pattern:
//
//	's|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+
//
// Go's regexp has no lookahead, so the scanner reproduces the regex
// engine's leftmost-first alternation by hand. The subtle branch is
// \s+(?!\S): a whitespace run followed by a non-space backtracks one
// character, leaving that character to serve as the next token's " ?"
// prefix — but only a literal space can actually join (the pattern says
// " ?", not "\s?"), so a run ending in '\n' before a word yields the
// '\n' as its own token via the final \s+ branch.
func (t *Tokenizer) preTokenize(text string) []string {
	var out []string
	rs := []rune(text)
	i := 0
	for i < len(rs) {
		r := rs[i]
		// Contractions: literal, case-sensitive, no leading space.
		if r == '\'' && i+1 < len(rs) {
			matched := ""
			rest := rs[i+1:]
			for _, suf := range [...]string{"s", "t", "re", "ve", "m", "ll", "d"} {
				sr := []rune(suf)
				if len(rest) >= len(sr) && string(rest[:len(sr)]) == suf {
					matched = suf
					break
				}
			}
			// The alternation tries 're/'ve/'ll before shorter ones in
			// pattern order; ordering above matches ('s,'t,'re,'ve,'m,'ll,'d)
			// — 's beats 're only when the next rune is 's' anyway, so
			// first-match here equals the regex's preference.
			if matched != "" {
				out = append(out, "'"+matched)
				i += 1 + len([]rune(matched))
				continue
			}
		}
		// " ?" + letters / digits / other-run branches. Only a literal
		// space (U+0020) can join as prefix — the pattern says " ?", not
		// "\s?". An apostrophe under the other-class is included: a
		// contraction only wins when the match STARTS at the apostrophe,
		// so " 're" tokenizes as [" '", "re"], and "!!!'s" keeps the
		// apostrophe inside the punctuation run.
		j := i
		if r == ' ' {
			j++
		}
		if j < len(rs) {
			switch c := rs[j]; {
			case unicode.IsLetter(c):
				k := j
				for k < len(rs) && unicode.IsLetter(rs[k]) {
					k++
				}
				out = append(out, string(rs[i:k]))
				i = k
				continue
			case isNum(c):
				k := j
				for k < len(rs) && isNum(rs[k]) {
					k++
				}
				out = append(out, string(rs[i:k]))
				i = k
				continue
			case !isSpace(c):
				k := j
				for k < len(rs) && !isSpace(rs[k]) && !unicode.IsLetter(rs[k]) && !isNum(rs[k]) {
					k++
				}
				out = append(out, string(rs[i:k]))
				i = k
				continue
			}
		}
		// Whitespace branches: \s+(?!\S) then \s+. Every non-space rune
		// was consumed by a branch above, so reaching here means rs[i] is
		// whitespace (a bare trailing space included).
		k := i
		for k < len(rs) && isSpace(rs[k]) {
			k++
		}
		switch {
		case k == len(rs):
			out = append(out, string(rs[i:k])) // run to EOS
			i = k
		case k-i > 1:
			out = append(out, string(rs[i:k-1])) // backtrack one: it prefixes the next token
			i = k - 1
		default:
			out = append(out, string(rs[i:k])) // single ws char via \s+
			i = k
		}
	}
	return out
}

// bpe merges one byte-encoded pre-token into vocabulary symbols with a
// doubly-linked symbol list and a min-heap of merge candidates ordered by
// (rank, position) — the algorithm HF's fast (Rust) tokenizer uses.
// O(n log n), so a megabyte-long unbroken pre-token costs milliseconds
// where the textbook round-based rescan is quadratic (measured 3.28 s for
// a 50k-char word — a real hazard for a server fed base64 or minified
// JSON). Output is IDENTICAL to GPT-2's round-based algorithm for any
// well-formed merge table: a merge's result can only appear in
// higher-ranked (later-learned) merges, so the heap never reorders
// rounds. bpeReference in the tests is the round-based version, and a
// differential test pins the equivalence on this vocab.
func (t *Tokenizer) bpe(token string) []string {
	runes := []rune(token)
	if len(runes) < 2 {
		if len(runes) == 0 {
			return nil
		}
		return []string{string(runes[0])}
	}
	syms := make([]string, len(runes))
	next := make([]int, len(runes))
	prev := make([]int, len(runes))
	alive := make([]bool, len(runes))
	for i, r := range runes {
		syms[i] = string(r)
		next[i] = i + 1
		prev[i] = i - 1
		alive[i] = true
	}
	next[len(runes)-1] = -1

	h := &candHeap{}
	push := func(left int) {
		right := next[left]
		if right == -1 {
			return
		}
		if rank, ok := t.ranks[[2]string{syms[left], syms[right]}]; ok {
			// Snapshot the pair: a popped candidate is stale (skipped)
			// unless both symbols still read exactly as they did here.
			heap.Push(h, cand{rank: rank, pos: left, left: syms[left], right: syms[right]})
		}
	}
	for i := 0; i+1 < len(runes); i++ {
		push(i)
	}
	for h.Len() > 0 {
		c := heap.Pop(h).(cand)
		i := c.pos
		j := next[i]
		if !alive[i] || j == -1 || syms[i] != c.left || syms[j] != c.right {
			continue // stale: one side was already merged away
		}
		syms[i] = c.left + c.right
		alive[j] = false
		next[i] = next[j]
		if next[j] != -1 {
			prev[next[j]] = i
		}
		if prev[i] != -1 {
			push(prev[i])
		}
		push(i)
	}

	out := make([]string, 0, 8)
	for i := 0; i != -1; i = next[i] {
		if alive[i] {
			out = append(out, syms[i])
		}
	}
	return out
}

// cand is one prospective merge: the pair (left,right) starting at node
// pos, valid only while both symbols still read as snapshotted.
type cand struct {
	rank, pos   int
	left, right string
}

// candHeap orders candidates by rank then position, making merge order
// deterministic and left-to-right among equal ranks (ranks are unique in
// a merges.txt, so the position tie-break is belt-and-suspenders).
type candHeap []cand

func (h candHeap) Len() int { return len(h) }
func (h candHeap) Less(a, b int) bool {
	if h[a].rank != h[b].rank {
		return h[a].rank < h[b].rank
	}
	return h[a].pos < h[b].pos
}
func (h candHeap) Swap(a, b int) { h[a], h[b] = h[b], h[a] }
func (h *candHeap) Push(x any)   { *h = append(*h, x.(cand)) }
func (h *candHeap) Pop() any {
	old := *h
	n := len(old)
	x := old[n-1]
	*h = old[:n-1]
	return x
}

// Encode tokenizes text as RoBERTa frames it — cls + tokens + sep —
// truncating to maxLen total ids. The mask is all ones (rembed never
// pads). Byte-level coverage means the unk id is unreachable for any
// input, but the lookup keeps it as a guard.
func (t *Tokenizer) Encode(text string, maxLen int) (ids, mask []int64) {
	ids = append(ids, t.cls)
	// Framing always survives: like the WordPiece path, maxLen < 2 clamps
	// the content budget to zero and still returns [cls, sep].
	budget := max(maxLen-2, 0)
	var sb strings.Builder
outer:
	for _, pre := range t.preTokenize(text) {
		sb.Reset()
		for _, b := range []byte(pre) {
			sb.WriteRune(t.byteEnc[b])
		}
		for _, sym := range t.bpe(sb.String()) {
			if budget <= 0 {
				break outer
			}
			id, ok := t.vocab[sym]
			if !ok {
				id = t.unk
			}
			ids = append(ids, id)
			budget--
		}
	}
	ids = append(ids, t.sep)
	mask = make([]int64, len(ids))
	for i := range mask {
		mask[i] = 1
	}
	return ids, mask
}
