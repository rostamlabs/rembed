// SPDX-License-Identifier: Apache-2.0

package bpe

import (
	"encoding/json"
	"math/rand"
	"os"
	"path/filepath"
	"slices"
	"strings"
	"testing"
	"time"
)

// modelDir resolves the distilroberta model dir: $REMBED_MODEL_ROBERTA
// (CI's golden job points it at the hub cache) or the local convert.py
// output.
func modelDir() string {
	if dir := os.Getenv("REMBED_MODEL_ROBERTA"); dir != "" {
		return dir
	}
	return "../../models/all-distilroberta-v1"
}

func loadTok(t *testing.T) *Tokenizer {
	t.Helper()
	dir := modelDir()
	vocabPath := filepath.Join(dir, "vocab.json")
	if _, err := os.Stat(vocabPath); err != nil {
		t.Skipf("vocab not present (%v) — run models/convert.py for all-distilroberta-v1", err)
	}
	tok, err := New(vocabPath, filepath.Join(dir, "merges.txt"), "<s>", "</s>", "<unk>")
	if err != nil {
		t.Fatal(err)
	}
	return tok
}

// TestFixtureAgainstHF requires token-for-token equality with HF's
// RobertaTokenizer over the committed fixture (models/bpe_fixture.py) —
// a battery hitting every pre-tokenizer branch: contractions (and their
// case-sensitivity), space-joining, whitespace-run backtracking,
// non-space whitespace, unicode letter/number classes, multi-byte UTF-8,
// and emoji.
func TestFixtureAgainstHF(t *testing.T) {
	tok := loadTok(t)
	raw, err := os.ReadFile("testdata/fixture.json")
	if err != nil {
		t.Fatal(err)
	}
	var fixture struct {
		Cases []struct {
			Text     string  `json:"text"`
			InputIDs []int64 `json:"input_ids"`
		} `json:"cases"`
	}
	if err := json.Unmarshal(raw, &fixture); err != nil {
		t.Fatal(err)
	}
	if len(fixture.Cases) < 20 {
		t.Fatalf("suspiciously small fixture: %d cases", len(fixture.Cases))
	}
	for _, c := range fixture.Cases {
		ids, mask := tok.Encode(c.Text, 512)
		if !slices.Equal(ids, c.InputIDs) {
			t.Errorf("%.50q:\n got %v\nwant %v", c.Text, ids, c.InputIDs)
		}
		if len(mask) != len(ids) {
			t.Errorf("%.50q: mask length %d != ids length %d", c.Text, len(mask), len(ids))
		}
	}
}

// mbTok resolves and loads the ModernBERT tokenizer from its
// tokenizer.json ($REMBED_MODEL_MODERNBERT in CI, or the convert.py dir).
func mbTok(t *testing.T) *Tokenizer {
	t.Helper()
	dir := os.Getenv("REMBED_MODEL_MODERNBERT")
	if dir == "" {
		dir = "../../models/modernbert-embed-base"
	}
	path := filepath.Join(dir, "tokenizer.json")
	if _, err := os.Stat(path); err != nil {
		t.Skipf("tokenizer.json not present (%v) — run models/convert.py for nomic-ai/modernbert-embed-base", err)
	}
	tok, err := NewFromTokenizerJSON(path, "[CLS]", "[SEP]", "[UNK]")
	if err != nil {
		t.Fatal(err)
	}
	return tok
}

// TestModernBERTFixtureAgainstHF requires token-for-token equality with
// HF's ModernBERT tokenizer over the committed fixture (models/mb_fixture.py).
// Beyond the shared byte-BPE branches it pins the two ModernBERT-specific
// behaviors: NFC normalization (decomposed → composed folds), and the
// leftmost-longest matching of the OLMo added tokens (whitespace runs of
// 2..24 spaces, the |||MARKER||| tokens, and [unusedN]) before BPE.
func TestModernBERTFixtureAgainstHF(t *testing.T) {
	tok := mbTok(t)
	raw, err := os.ReadFile("testdata/mb_fixture.json")
	if err != nil {
		t.Fatal(err)
	}
	var fixture struct {
		Cases []struct {
			Text     string  `json:"text"`
			InputIDs []int64 `json:"input_ids"`
		} `json:"cases"`
	}
	if err := json.Unmarshal(raw, &fixture); err != nil {
		t.Fatal(err)
	}
	if len(fixture.Cases) < 40 {
		t.Fatalf("suspiciously small fixture: %d cases", len(fixture.Cases))
	}
	for _, c := range fixture.Cases {
		ids, mask := tok.Encode(c.Text, 512)
		if !slices.Equal(ids, c.InputIDs) {
			t.Errorf("%.50q:\n got %v\nwant %v", c.Text, ids, c.InputIDs)
		}
		if len(mask) != len(ids) {
			t.Errorf("%.50q: mask length %d != ids length %d", c.Text, len(mask), len(ids))
		}
	}
}

// qwenTok resolves and loads the Qwen3 tokenizer from its tokenizer.json.
func qwenTok(t *testing.T) *Tokenizer {
	t.Helper()
	dir := os.Getenv("REMBED_MODEL_QWEN3")
	if dir == "" {
		dir = "../../models/Qwen3-Embedding-0.6B"
	}
	path := filepath.Join(dir, "tokenizer.json")
	if _, err := os.Stat(path); err != nil {
		t.Skipf("tokenizer.json not present (%v) — run models/convert.py for Qwen/Qwen3-Embedding-0.6B", err)
	}
	tok, err := NewQwenFromTokenizerJSON(path, "<|endoftext|>")
	if err != nil {
		t.Fatal(err)
	}
	return tok
}

// TestQwenFixtureAgainstHF requires token-for-token equality with HF's
// Qwen3 tokenizer over the committed fixture (models/qwen_fixture.py). It
// pins the GPT-NeoX/Qwen pre-tokenizer's departures from GPT-2
// (case-insensitive contractions, single-digit splitting, the broad letter
// prefix, newline handling), NFC, and the suffix-only framing (a trailing
// <|endoftext|>, no CLS).
func TestQwenFixtureAgainstHF(t *testing.T) {
	tok := qwenTok(t)
	raw, err := os.ReadFile("testdata/qwen_fixture.json")
	if err != nil {
		t.Fatal(err)
	}
	var fixture struct {
		Cases []struct {
			Text     string  `json:"text"`
			InputIDs []int64 `json:"input_ids"`
		} `json:"cases"`
	}
	if err := json.Unmarshal(raw, &fixture); err != nil {
		t.Fatal(err)
	}
	if len(fixture.Cases) < 20 {
		t.Fatalf("suspiciously small fixture: %d cases", len(fixture.Cases))
	}
	for _, c := range fixture.Cases {
		ids, mask := tok.Encode(c.Text, 512)
		if !slices.Equal(ids, c.InputIDs) {
			t.Errorf("%.50q:\n got %v\nwant %v", c.Text, ids, c.InputIDs)
		}
		if len(mask) != len(ids) {
			t.Errorf("%.50q: mask length %d != ids length %d", c.Text, len(mask), len(ids))
		}
	}
	// Framing: no prefix, a trailing <|endoftext|> (151643).
	ids, _ := tok.Encode("hi", 512)
	if len(ids) < 2 || ids[len(ids)-1] != 151643 {
		t.Fatalf("qwen framing wrong: %v (want trailing 151643, no prefix)", ids)
	}
	if ids[0] == 151643 {
		t.Fatalf("qwen must not prepend a prefix token: %v", ids)
	}
}

// TestVocabSize pins the distilroberta vocab count (specials included).
func TestVocabSize(t *testing.T) {
	tok := loadTok(t)
	if got := tok.VocabSize(); got != 50265 {
		t.Fatalf("VocabSize = %d, want 50265", got)
	}
}

// TestTruncation: maxLen bounds the TOTAL sequence, framing included,
// and the sep always survives truncation.
func TestTruncation(t *testing.T) {
	tok := loadTok(t)
	long := strings.Repeat("supercalifragilisticexpialidocious ", 200)
	ids, _ := tok.Encode(long, 16)
	if len(ids) != 16 {
		t.Fatalf("truncated length %d, want 16", len(ids))
	}
	if ids[0] != 0 || ids[len(ids)-1] != 2 {
		t.Fatalf("framing lost under truncation: first=%d last=%d", ids[0], ids[len(ids)-1])
	}
}

// TestByteToUnicode pins the GPT-2 table's fixed points: 256 distinct
// outputs, printable Latin-1 identity, and the space mapping to Ġ
// (U+0120) that the merges file is written in.
func TestByteToUnicode(t *testing.T) {
	enc := byteToUnicode()
	seen := map[rune]bool{}
	for _, r := range enc {
		if seen[r] {
			t.Fatalf("byteToUnicode not injective at %q", r)
		}
		seen[r] = true
	}
	if enc[' '] != 'Ġ' {
		t.Fatalf("space maps to %q, want Ġ", enc[' '])
	}
	if enc['A'] != 'A' || enc['~'] != '~' || enc[0xFF] != 'ÿ' {
		t.Fatal("printable identity broken")
	}
	if enc['\n'] != 'Ċ' || enc['\t'] != 'ĉ' {
		t.Fatalf("control mappings: \\n=%q \\t=%q, want Ċ ĉ", enc['\n'], enc['\t'])
	}
}

// bpeReference is GPT-2's textbook round-based algorithm — find the
// lowest-ranked adjacent pair, merge every non-overlapping occurrence
// left to right, repeat. O(n²) and kept ONLY as the differential oracle
// for the heap-based production bpe.
func (t *Tokenizer) bpeReference(token string) []string {
	word := make([]string, 0, len(token))
	for _, r := range token {
		word = append(word, string(r))
	}
	for len(word) > 1 {
		bestRank := -1
		var best [2]string
		for i := 0; i+1 < len(word); i++ {
			if r, ok := t.ranks[[2]string{word[i], word[i+1]}]; ok && (bestRank < 0 || r < bestRank) {
				bestRank = r
				best = [2]string{word[i], word[i+1]}
			}
		}
		if bestRank < 0 {
			break
		}
		merged := make([]string, 0, len(word))
		for i := 0; i < len(word); i++ {
			if i+1 < len(word) && word[i] == best[0] && word[i+1] == best[1] {
				merged = append(merged, best[0]+best[1])
				i++
			} else {
				merged = append(merged, word[i])
			}
		}
		word = merged
	}
	return word
}

// TestBPEHeapMatchesReference pins the heap-based bpe against the
// round-based GPT-2 oracle on the real merge table: random byte soup,
// repeated-character words (the overlapping-pair edge), and every
// fixture pre-token. The equivalence argument (a merge's result only
// appears in later-learned, higher-ranked merges) holds for any real
// merges.txt; this test would catch a table that violates it.
func TestBPEHeapMatchesReference(t *testing.T) {
	tok := loadTok(t)
	rng := rand.New(rand.NewSource(11))
	var words []string
	alphabet := []rune("abcdefghijklmnopqrstuvwxyzABC0123456789 .,!?'-éü你🚀")
	for range 400 {
		n := 1 + rng.Intn(24)
		w := make([]rune, n)
		for i := range w {
			w[i] = alphabet[rng.Intn(len(alphabet))]
		}
		words = append(words, string(w))
	}
	for _, r := range []string{"a", "e", "s", "Ġ", "0"} {
		for _, n := range []int{2, 3, 7, 64, 257} {
			words = append(words, strings.Repeat(r, n))
		}
	}
	raw, err := os.ReadFile("testdata/fixture.json")
	if err == nil {
		var fixture struct {
			Cases []struct {
				Text string `json:"text"`
			} `json:"cases"`
		}
		if json.Unmarshal(raw, &fixture) == nil {
			for _, c := range fixture.Cases {
				words = append(words, tok.preTokenize(c.Text)...)
			}
		}
	}
	for _, w := range words {
		var sb strings.Builder
		for _, b := range []byte(w) {
			sb.WriteRune(tok.byteEnc[b])
		}
		enc := sb.String()
		if got, want := tok.bpe(enc), tok.bpeReference(enc); !slices.Equal(got, want) {
			t.Fatalf("bpe(%q):\n heap %v\n ref  %v", w, got, want)
		}
	}
}

// TestEncodeLongWordFast bounds the cost of a pathological unbroken
// pre-token — the review measured the quadratic version at 3.28 s for a
// 50k-char word; the heap version must stay in the tens of milliseconds.
func TestEncodeLongWordFast(t *testing.T) {
	tok := loadTok(t)
	long := strings.Repeat("abcdefghij", 5000) // 50k chars, no spaces
	start := time.Now()
	ids, _ := tok.Encode(long, 512)
	if d := time.Since(start); d > 2*time.Second {
		t.Fatalf("50k-char word took %v; the quadratic hazard is back", d)
	}
	if len(ids) != 512 {
		t.Fatalf("expected 512 ids, got %d", len(ids))
	}
}

// TestEncodeTinyMaxLen: framing survives even absurd ceilings.
func TestEncodeTinyMaxLen(t *testing.T) {
	tok := loadTok(t)
	for _, maxLen := range []int{0, 1, 2} {
		ids, _ := tok.Encode("hello world", maxLen)
		if len(ids) != 2 || ids[0] != 0 || ids[1] != 2 {
			t.Fatalf("maxLen=%d: got %v, want [0 2]", maxLen, ids)
		}
	}
}
