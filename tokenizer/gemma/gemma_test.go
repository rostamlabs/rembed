// SPDX-License-Identifier: Apache-2.0

package gemma

import (
	"encoding/json"
	"os"
	"path/filepath"
	"testing"
)

// gemmaTok resolves and loads the EmbeddingGemma tokenizer from its
// tokenizer.json ($REMBED_MODEL_GEMMA in CI, or the convert.py dir).
func gemmaTok(t *testing.T) *Tokenizer {
	t.Helper()
	dir := os.Getenv("REMBED_MODEL_GEMMA")
	if dir == "" {
		dir = filepath.Join("..", "..", "models", "embeddinggemma-300m")
	}
	path := filepath.Join(dir, "tokenizer.json")
	if _, err := os.Stat(path); err != nil {
		t.Skipf("tokenizer.json not present (%v) — run models/gemma_fixture.py for google/embeddinggemma-300m", err)
	}
	tok, err := New(path)
	if err != nil {
		t.Fatal(err)
	}
	return tok
}

type fixture struct {
	Model string `json:"model"`
	Cases []struct {
		Text     string  `json:"text"`
		InputIDs []int64 `json:"input_ids"`
	} `json:"cases"`
}

// TestGemmaFixtureAgainstHF pins the Go Gemma BPE token-for-token against
// HF's tokenizer over the committed fixture (metaspace normalization, byte
// fallback, whole-string merges, and the <bos>…<eos> framing).
func TestGemmaFixtureAgainstHF(t *testing.T) {
	tok := gemmaTok(t)
	raw, err := os.ReadFile(filepath.Join("testdata", "gemma_fixture.json"))
	if err != nil {
		t.Fatal(err)
	}
	var fx fixture
	if err := json.Unmarshal(raw, &fx); err != nil {
		t.Fatal(err)
	}
	if len(fx.Cases) == 0 {
		t.Fatal("empty fixture")
	}
	for _, c := range fx.Cases {
		ids, _ := tok.Encode(c.Text, 4096)
		if len(ids) != len(c.InputIDs) {
			t.Errorf("%.50q: got %d ids, want %d\n got  %v\n want %v", c.Text, len(ids), len(c.InputIDs), ids, c.InputIDs)
			continue
		}
		for i := range ids {
			if ids[i] != c.InputIDs[i] {
				t.Errorf("%.50q: id[%d]=%d want %d\n got  %v\n want %v", c.Text, i, ids[i], c.InputIDs[i], ids, c.InputIDs)
				break
			}
		}
	}
}

// TestGemmaFraming checks the bos/eos wrapping and truncation budget.
func TestGemmaFraming(t *testing.T) {
	tok := gemmaTok(t)
	ids, mask := tok.Encode("", 4096)
	if len(ids) != 2 || ids[0] != int64(tok.bos) || ids[1] != int64(tok.eos) {
		t.Fatalf("empty input should frame to [bos,eos], got %v", ids)
	}
	if len(mask) != len(ids) {
		t.Fatalf("mask/ids length mismatch: %d vs %d", len(mask), len(ids))
	}
	// maxLen leaves room for exactly (maxLen-2) content pieces plus framing.
	ids, _ = tok.Encode("the quick brown fox jumps over the lazy dog", 5)
	if len(ids) != 5 || ids[0] != int64(tok.bos) || ids[len(ids)-1] != int64(tok.eos) {
		t.Fatalf("truncation to maxLen=5 wrong: got %d ids %v", len(ids), ids)
	}
}
