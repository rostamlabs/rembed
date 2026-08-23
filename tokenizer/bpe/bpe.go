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
	"unicode/utf8"

	"golang.org/x/text/unicode/norm"
)

// Tokenizer is a loaded byte-level BPE encoder (vocab.json + merges.txt for
// RoBERTa, or an embedded vocab/merges from a tokenizer.json for
// ModernBERT). It is safe for concurrent use after New: all state is
// read-only.
type Tokenizer struct {
	vocab   map[string]int64
	ranks   map[[2]string]int
	byteEnc [256]rune
	cls     int64
	sep     int64
	unk     int64
	// nfc applies Unicode NFC normalization to the input before
	// pre-tokenization. ModernBERT's tokenizer declares an NFC normalizer;
	// RoBERTa/GPT-2 declare none, so it stays off there.
	nfc bool
	// added holds the tokenizer's added tokens (content → id) — ModernBERT's
	// OLMo-inherited whitespace runs (2–24 spaces), |||MARKER||| tokens,
	// specials and [unusedN]. HF matches these against the (normalized)
	// text BEFORE BPE, leftmost-longest and non-overlapping; the gaps
	// between matches then go through the byte-BPE pipeline. nil on the
	// RoBERTa path (New), which frames with specials but never matches
	// added tokens inside the text. addedFirst is a fast first-byte filter
	// (most positions can't start an added token), addedMin/addedMax bound
	// the match-length scan.
	added              map[string]addedTok
	addedFirst         [256]bool
	addedMin, addedMax int
	// qwenPre selects the GPT-NeoX/Qwen pre-tokenizer instead of GPT-2's.
	// noPrefix skips the cls/bos prefix at Encode (Qwen3 frames with only a
	// trailing eos in sep).
	qwenPre  bool
	noPrefix bool
	// lstripToks are the added tokens with lstrip=true ([MASK]). HF matches
	// them as `\s*<content>` leftmost-longest, so a preceding whitespace run
	// is consumed by the token (and never claimed as a whitespace-run added
	// token). Tiny set, scanned linearly at whitespace positions.
	lstripToks []addedContent
}

// addedContent pairs an added token's literal content with its id.
type addedContent struct {
	content string
	id      int64
}

// addedTok is one added token's id plus its whitespace-absorption flags.
// lstrip (true for [MASK]) eats whitespace immediately to its left;
// rstrip eats whitespace to its right — matching HF's AddedToken flags.
type addedTok struct {
	id             int64
	lstrip, rstrip bool
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

// parseBPEJSON builds a base Tokenizer (vocab, merges, added tokens,
// normalizer) from a HuggingFace tokenizer.json — the shared core of the
// ModernBERT and Qwen3 loaders, which then differ only in framing and
// pre-tokenizer variant. The framing tokens (cls/sep/unk) and the
// qwenPre/noPrefix flags are left for the caller to set.
func parseBPEJSON(path string) (*Tokenizer, error) {
	raw, err := os.ReadFile(path)
	if err != nil {
		return nil, err
	}
	var tj struct {
		Normalizer struct {
			Type string `json:"type"`
		} `json:"normalizer"`
		AddedTokens []struct {
			ID      int64  `json:"id"`
			Content string `json:"content"`
			Lstrip  bool   `json:"lstrip"`
			Rstrip  bool   `json:"rstrip"`
		} `json:"added_tokens"`
		Model struct {
			Type   string           `json:"type"`
			Vocab  map[string]int64 `json:"vocab"`
			Merges json.RawMessage  `json:"merges"`
		} `json:"model"`
	}
	if err := json.Unmarshal(raw, &tj); err != nil {
		return nil, fmt.Errorf("bpe: %s: %w", path, err)
	}
	if tj.Model.Type != "BPE" {
		return nil, fmt.Errorf("bpe: %s: model.type %q is not BPE", path, tj.Model.Type)
	}
	if len(tj.Model.Vocab) == 0 {
		return nil, fmt.Errorf("bpe: %s: empty model.vocab", path)
	}
	// added_tokens (specials, whitespace runs, OLMo markers) live outside
	// model.vocab; merge them so the framing-token lookups resolve. They
	// never collide with byte-encoded BPE symbols (those map space→Ġ etc.).
	// The same set is kept separately in `added` for the leftmost-longest
	// pre-BPE matching HF performs — Encode splits the text on these before
	// running byte-BPE on the gaps.
	vocab := tj.Model.Vocab
	added := make(map[string]addedTok, len(tj.AddedTokens))
	addedMin, addedMax := 1<<30, 0
	var addedFirst [256]bool
	var lstripToks []addedContent
	for _, at := range tj.AddedTokens {
		if at.Content == "" {
			continue
		}
		vocab[at.Content] = at.ID
		added[at.Content] = addedTok{id: at.ID, lstrip: at.Lstrip, rstrip: at.Rstrip}
		addedFirst[at.Content[0]] = true
		if at.Lstrip {
			lstripToks = append(lstripToks, addedContent{content: at.Content, id: at.ID})
		}
		n := len(at.Content)
		if n < addedMin {
			addedMin = n
		}
		if n > addedMax {
			addedMax = n
		}
	}

	// merges is either the modern [["a","b"],…] or the legacy ["a b",…].
	ranks := make(map[[2]string]int)
	var pairs [][2]string
	var arr2 [][]string
	if err := json.Unmarshal(tj.Model.Merges, &arr2); err == nil {
		for _, m := range arr2 {
			if len(m) != 2 {
				return nil, fmt.Errorf("bpe: %s: malformed merge %v", path, m)
			}
			pairs = append(pairs, [2]string{m[0], m[1]})
		}
	} else {
		var arr1 []string
		if err := json.Unmarshal(tj.Model.Merges, &arr1); err != nil {
			return nil, fmt.Errorf("bpe: %s: merges are neither [[a,b]] nor [\"a b\"]: %w", path, err)
		}
		for _, line := range arr1 {
			a, b, ok := strings.Cut(line, " ")
			if !ok || a == "" || b == "" {
				return nil, fmt.Errorf("bpe: %s: malformed merge %q", path, line)
			}
			pairs = append(pairs, [2]string{a, b})
		}
	}
	if len(pairs) == 0 {
		return nil, fmt.Errorf("bpe: %s: no merges", path)
	}
	for rank, p := range pairs {
		ranks[p] = rank
	}

	if addedMin > addedMax {
		addedMin = addedMax
	}
	t := &Tokenizer{
		vocab: vocab, ranks: ranks, byteEnc: byteToUnicode(),
		nfc:   tj.Normalizer.Type == "NFC",
		added: added, addedFirst: addedFirst, addedMin: addedMin, addedMax: addedMax,
		lstripToks: lstripToks,
	}
	return t, nil
}

// lookupTok resolves a token content to its id in the loaded vocab.
func (t *Tokenizer) lookupTok(tok, what, path string) (int64, error) {
	id, ok := t.vocab[tok]
	if !ok {
		return 0, fmt.Errorf("bpe: %s token %q not in %s", what, tok, path)
	}
	return id, nil
}

// NewFromTokenizerJSON loads a byte-level BPE tokenizer from a HuggingFace
// tokenizer.json (the ModernBERT case: no vocab.json/merges.txt ship),
// framing with cls…sep like RoBERTa and using the GPT-2 pre-tokenizer. NFC
// normalization is enabled to match ModernBERT's declared normalizer.
func NewFromTokenizerJSON(path, clsTok, sepTok, unkTok string) (*Tokenizer, error) {
	t, err := parseBPEJSON(path)
	if err != nil {
		return nil, err
	}
	if t.cls, err = t.lookupTok(clsTok, "cls", path); err != nil {
		return nil, err
	}
	if t.sep, err = t.lookupTok(sepTok, "sep", path); err != nil {
		return nil, err
	}
	if t.unk, err = t.lookupTok(unkTok, "unk", path); err != nil {
		return nil, err
	}
	return t, nil
}

// NewQwenFromTokenizerJSON loads Qwen3's byte-level BPE tokenizer: the
// GPT-NeoX/Qwen pre-tokenizer (different from GPT-2 — case-insensitive
// contractions, single-digit splitting, a broader letter prefix), NFC
// normalization, and SUFFIX-ONLY framing — no CLS/BOS prefix, and eosTok
// (<|endoftext|>) appended, which last-token pooling then reads.
func NewQwenFromTokenizerJSON(path, eosTok string) (*Tokenizer, error) {
	t, err := parseBPEJSON(path)
	if err != nil {
		return nil, err
	}
	t.qwenPre = true
	t.noPrefix = true
	if t.sep, err = t.lookupTok(eosTok, "eos", path); err != nil {
		return nil, err
	}
	t.unk = t.sep // byte-level BPE never emits unk; keep a valid guard id
	return t, nil
}

// VocabSize returns the number of entries in the loaded vocabulary.
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

// preTok dispatches to the pre-tokenizer variant this tokenizer uses.
func (t *Tokenizer) preTok(text string) []string {
	if t.qwenPre {
		return t.preTokenizeQwen(text)
	}
	return t.preTokenize(text)
}

// preTokenizeQwen splits text with the GPT-NeoX/Qwen pattern (the ByteLevel
// pre-tokenizer's Split step):
//
//	(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+
//
// Go's regexp has no lookahead or ordered-alternation semantics the way HF's
// Rust regex applies them, so the scanner reproduces the leftmost-first
// alternation by hand. Differences from GPT-2 (preTokenize): contractions
// are case-INSENSITIVE; a letter run may take ANY single non-newline
// non-alphanumeric prefix (not just a space); digits are emitted ONE at a
// time; punctuation runs may absorb trailing newlines; and a whitespace run
// containing a newline is cut at its last newline.
func (t *Tokenizer) preTokenizeQwen(text string) []string {
	isNL := func(r rune) bool { return r == '\r' || r == '\n' }
	var out []string
	rs := []rune(text)
	i, n := 0, len([]rune(text))
	for i < n {
		r := rs[i]
		// 1. (?i:'s|'t|'re|'ve|'m|'ll|'d) — case-insensitive, apostrophe-led.
		if r == '\'' && i+1 < n {
			rest := rs[i+1:]
			matched := ""
			for _, suf := range [...]string{"s", "t", "re", "ve", "m", "ll", "d"} {
				sr := []rune(suf)
				if len(rest) >= len(sr) && strings.EqualFold(string(rest[:len(sr)]), suf) {
					matched = string(rest[:len(sr)])
					break
				}
			}
			if matched != "" {
				out = append(out, "'"+matched)
				i += 1 + len([]rune(matched))
				continue
			}
		}
		// 2. [^\r\n\p{L}\p{N}]?\p{L}+ — letters with an optional single
		// non-newline, non-alphanumeric prefix (space, punct, tab, …).
		{
			j := i
			if !isNL(r) && !unicode.IsLetter(r) && !isNum(r) && i+1 < n && unicode.IsLetter(rs[i+1]) {
				j = i + 1
			}
			if j < n && unicode.IsLetter(rs[j]) {
				k := j
				for k < n && unicode.IsLetter(rs[k]) {
					k++
				}
				out = append(out, string(rs[i:k]))
				i = k
				continue
			}
		}
		// 3. \p{N} — a single digit.
		if isNum(r) {
			out = append(out, string(r))
			i++
			continue
		}
		// 4.  ?[^\s\p{L}\p{N}]+[\r\n]* — optional space, a run of
		// non-space/non-alphanumeric, then trailing newlines.
		{
			j := i
			if r == ' ' {
				j++
			}
			if j < n && !unicode.IsSpace(rs[j]) && !unicode.IsLetter(rs[j]) && !isNum(rs[j]) {
				k := j
				for k < n && !unicode.IsSpace(rs[k]) && !unicode.IsLetter(rs[k]) && !isNum(rs[k]) {
					k++
				}
				for k < n && isNL(rs[k]) {
					k++
				}
				out = append(out, string(rs[i:k]))
				i = k
				continue
			}
		}
		// Remaining branches are whitespace runs. Find the run [i,k).
		k := i
		for k < n && unicode.IsSpace(rs[k]) {
			k++
		}
		// 5. \s*[\r\n]+ — a run containing a newline is cut after its LAST
		// newline; the rest re-enters the loop.
		last := -1
		for p := i; p < k; p++ {
			if isNL(rs[p]) {
				last = p
			}
		}
		if last >= 0 {
			out = append(out, string(rs[i:last+1]))
			i = last + 1
			continue
		}
		// 6. \s+(?!\S) / 7. \s+ — a newline-free space run: the whole run at
		// end-of-text, else all but the last char (which prefixes the next
		// token, matching GPT-2's backtrack).
		switch {
		case k == n:
			out = append(out, string(rs[i:k]))
			i = k
		case k-i > 1:
			out = append(out, string(rs[i:k-1]))
			i = k - 1
		default:
			out = append(out, string(rs[i:k]))
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
	if t.nfc {
		text = norm.NFC.String(text)
	}
	// Framing: RoBERTa/ModernBERT prepend cls and append sep; Qwen3 appends
	// only the eos (in sep) with no prefix. The suffix always survives, so
	// maxLen below the framing count clamps the content budget to zero and
	// still returns the framing.
	framing := 1
	if !t.noPrefix {
		ids = append(ids, t.cls)
		framing++
	}
	budget := max(maxLen-framing, 0)

	// emitBPE runs the byte-BPE pipeline over one gap of ordinary text,
	// appending ids until the budget runs out; returns false when it does.
	var sb strings.Builder
	emitBPE := func(gap string) bool {
		for _, pre := range t.preTok(gap) {
			sb.Reset()
			for _, b := range []byte(pre) {
				sb.WriteRune(t.byteEnc[b])
			}
			for _, sym := range t.bpe(sb.String()) {
				if budget <= 0 {
					return false
				}
				id, ok := t.vocab[sym]
				if !ok {
					id = t.unk
				}
				ids = append(ids, id)
				budget--
			}
		}
		return true
	}

	if t.added == nil {
		emitBPE(text)
	} else {
		// Leftmost-longest, non-overlapping match of added tokens against
		// the (normalized) text — HF's AddedVocabulary pass. The gaps
		// between matches go through byte-BPE; a match emits its id
		// directly. Added-token contents are all ASCII, so byte scanning
		// never splits a multibyte rune.
		b := []byte(text)
		gapStart := 0
		i := 0
		for i < len(b) {
			ml, id, ok := t.matchAdded(b, i)
			if !ok {
				i++
				continue
			}
			// The consumed span [i, i+ml) — which for an lstrip token
			// already includes the preceding whitespace run — emits only the
			// token id; the gap before it goes through BPE.
			if gapStart < i && !emitBPE(string(b[gapStart:i])) {
				gapStart = len(b)
				break
			}
			if budget <= 0 {
				gapStart = len(b)
				break
			}
			ids = append(ids, id)
			budget--
			i += ml
			gapStart = i
		}
		if gapStart < len(b) {
			emitBPE(string(b[gapStart:]))
		}
	}

	ids = append(ids, t.sep)
	mask = make([]int64, len(ids))
	for i := range mask {
		mask[i] = 1
	}
	return ids, mask
}

// matchAdded finds the longest added-token match whose consumed span
// starts at b[i], returning the span's byte length, the token id, and ok.
// It reproduces HF's leftmost-longest AddedVocabulary regex: a plain token
// matches its literal content; an rstrip token additionally consumes the
// whitespace run after it; and an lstrip token ([MASK]) matches as
// `\s*<content>`, so when b[i] begins a whitespace run followed by the
// content, the whole run+content is one span (the whitespace is eaten, not
// tokenized). The longest competing span wins.
func (t *Tokenizer) matchAdded(b []byte, i int) (int, int64, bool) {
	best, bestID, found := 0, int64(0), false
	// Plain content match at i (longest content first), plus rstrip's
	// trailing-whitespace extension.
	if t.addedFirst[b[i]] {
		hi := t.addedMax
		if rem := len(b) - i; rem < hi {
			hi = rem
		}
		for l := hi; l >= t.addedMin; l-- {
			if at, ok := t.added[string(b[i:i+l])]; ok {
				consumed := l
				if at.rstrip {
					consumed += wsRun(b, i+l)
				}
				best, bestID, found = consumed, at.id, true
				break
			}
		}
	}
	// lstrip tokens ([MASK]): a whitespace run at i followed by the content.
	if w := i + wsRun(b, i); w > i {
		for _, lt := range t.lstripToks {
			if hasPrefixAt(b, w, lt.content) {
				consumed := (w - i) + len(lt.content)
				if at := t.added[lt.content]; at.rstrip {
					consumed += wsRun(b, w+len(lt.content))
				}
				if consumed > best {
					best, bestID, found = consumed, lt.id, true
				}
			}
		}
	}
	return best, bestID, found
}

// wsRun returns the byte length of the Unicode-whitespace run starting at
// b[pos] (matching HF's regex \s*, which is Unicode-aware).
func wsRun(b []byte, pos int) int {
	n := 0
	for pos+n < len(b) {
		r, sz := utf8.DecodeRune(b[pos+n:])
		if !unicode.IsSpace(r) {
			break
		}
		n += sz
	}
	return n
}

// hasPrefixAt reports whether b[pos:] begins with s.
func hasPrefixAt(b []byte, pos int, s string) bool {
	return pos+len(s) <= len(b) && string(b[pos:pos+len(s)]) == s
}
