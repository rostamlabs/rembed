// SPDX-License-Identifier: Apache-2.0

package bpe

import (
	"encoding/json"
	"os"
	"path/filepath"
	"slices"
	"strings"
	"testing"
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
