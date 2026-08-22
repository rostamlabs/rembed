// SPDX-License-Identifier: Apache-2.0

package sentencepiece

import (
	"encoding/json"
	"os"
	"path/filepath"
	"slices"
	"testing"
)

// modelDir resolves the multilingual model dir: $REMBED_MODEL_XLMR (CI's
// golden job points it at the hub cache) or the local convert.py output.
func modelDir() string {
	if dir := os.Getenv("REMBED_MODEL_XLMR"); dir != "" {
		return dir
	}
	return "../../models/paraphrase-multilingual-MiniLM-L12-v2"
}

func loadTok(t *testing.T) *Tokenizer {
	t.Helper()
	path := filepath.Join(modelDir(), "sentencepiece.bpe.model")
	if _, err := os.Stat(path); err != nil {
		t.Skipf("sentencepiece model not present (%v) — run models/convert.py for paraphrase-multilingual-MiniLM-L12-v2", err)
	}
	tok, err := New(path)
	if err != nil {
		t.Fatal(err)
	}
	return tok
}

type fixture struct {
	Cases []struct {
		Text       string   `json:"text"`
		Normalized string   `json:"normalized"`
		Pieces     []string `json:"pieces"`
		InputIDs   []int64  `json:"input_ids"`
	} `json:"cases"`
}

func loadFixture(t *testing.T) fixture {
	t.Helper()
	raw, err := os.ReadFile("testdata/fixture.json")
	if err != nil {
		t.Fatal(err)
	}
	var f fixture
	if err := json.Unmarshal(raw, &f); err != nil {
		t.Fatal(err)
	}
	if len(f.Cases) < 25 {
		t.Fatalf("suspiciously small fixture: %d cases", len(f.Cases))
	}
	return f
}

// TestNormalizeAgainstReference pins the charsmap-driven NMT-NFKC
// normalizer byte-for-byte against sentencepiece's own Normalize over the
// fixture battery — fullwidth folding, ligatures, unicode spaces,
// whitespace collapsing, the dummy prefix, and nine scripts.
func TestNormalizeAgainstReference(t *testing.T) {
	tok := loadTok(t)
	for _, c := range loadFixture(t).Cases {
		if got := tok.Normalize(c.Text); got != c.Normalized {
			t.Errorf("Normalize(%.40q):\n got %q\nwant %q", c.Text, got, c.Normalized)
		}
	}
}

// TestPiecesAgainstReference pins the Viterbi segmentation against
// sentencepiece's EncodeAsPieces.
func TestPiecesAgainstReference(t *testing.T) {
	tok := loadTok(t)
	for _, c := range loadFixture(t).Cases {
		if got := tok.Pieces(c.Text); !slices.Equal(got, c.Pieces) {
			t.Errorf("Pieces(%.40q):\n got %v\nwant %v", c.Text, got, c.Pieces)
		}
	}
}

// TestFixtureAgainstHF requires token-for-token id equality with HF's
// XLMRobertaTokenizer, fairseq mapping and truncation included (the long
// Persian case exceeds 512 tokens).
func TestFixtureAgainstHF(t *testing.T) {
	tok := loadTok(t)
	for _, c := range loadFixture(t).Cases {
		ids, mask := tok.Encode(c.Text, 512)
		if !slices.Equal(ids, c.InputIDs) {
			got, want := ids, c.InputIDs
			if len(got) > 16 {
				got = got[:16]
			}
			if len(want) > 16 {
				want = want[:16]
			}
			t.Errorf("%.40q: ids mismatch (len %d vs %d)\n got %v…\nwant %v…", c.Text, len(ids), len(c.InputIDs), got, want)
		}
		if len(mask) != len(ids) {
			t.Errorf("%.40q: mask length %d != ids %d", c.Text, len(mask), len(ids))
		}
	}
}

// TestVocabSize pins the fairseq id-space arithmetic: 250000 pieces + the
// +1 offset + <mask> = 250002 (HF's len(tokenizer)).
func TestVocabSize(t *testing.T) {
	tok := loadTok(t)
	if got := tok.VocabSize(); got != 250002 {
		t.Fatalf("VocabSize = %d, want 250002", got)
	}
}

// TestEncodeTinyMaxLen: framing survives absurd ceilings.
func TestEncodeTinyMaxLen(t *testing.T) {
	tok := loadTok(t)
	for _, maxLen := range []int{0, 1, 2} {
		ids, _ := tok.Encode("hello world", maxLen)
		if len(ids) != 2 || ids[0] != 0 || ids[1] != 2 {
			t.Fatalf("maxLen=%d: got %v, want [0 2]", maxLen, ids)
		}
	}
}
