// SPDX-License-Identifier: Apache-2.0

package tokenizer

import (
	"os"
	"path/filepath"
	"strings"
	"testing"
)

func writeVocab(t *testing.T, lines []string) string {
	t.Helper()
	dir := t.TempDir()
	p := filepath.Join(dir, "vocab.txt")
	if err := os.WriteFile(p, []byte(strings.Join(lines, "\n")), 0o600); err != nil {
		t.Fatal(err)
	}
	return p
}

func TestEncodeWordPiece(t *testing.T) {
	// ids:      0     1     2     3     4       5     6      7
	vocab := []string{"[PAD]", "[UNK]", "[CLS]", "[SEP]", "play", "##ing", "hello", "world"}
	p := writeVocab(t, vocab)
	tok, err := New(p, true, "", "", "")
	if err != nil {
		t.Fatal(err)
	}
	ids, mask := tok.Encode("playing hello", MaxSeqLen)
	// [CLS] play ##ing hello [SEP]
	want := []int64{2, 4, 5, 6, 3}
	if len(ids) != len(want) {
		t.Fatalf("ids=%v want len %d", ids, len(want))
	}
	for i := range want {
		if ids[i] != want[i] {
			t.Fatalf("ids=%v want %v", ids, want)
		}
	}
	for _, m := range mask {
		if m != 1 {
			t.Fatalf("mask has non-1 for real tokens: %v", mask)
		}
	}
}

func TestEncodeOverLongWordIsSingleUnk(t *testing.T) {
	vocab := []string{"[PAD]", "[UNK]", "[CLS]", "[SEP]", "a", "##a", "hello"}
	tok, _ := New(writeVocab(t, vocab), true, "", "", "")

	// A >100-rune no-space word must collapse to one [UNK]=1 without the
	// O(n^2) prefix search: [CLS] [UNK] [SEP].
	long := strings.Repeat("a", 101)
	ids, _ := tok.Encode(long, MaxSeqLen)
	want := []int64{2, 1, 3}
	if len(ids) != len(want) {
		t.Fatalf("ids=%v want %v", ids, want)
	}
	for i := range want {
		if ids[i] != want[i] {
			t.Fatalf("ids=%v want %v", ids, want)
		}
	}

	// A <=100-char word still tokenizes normally (not forced to [UNK]).
	ids2, _ := tok.Encode("hello", MaxSeqLen)
	want2 := []int64{2, 6, 3} // [CLS] hello [SEP]
	if len(ids2) != len(want2) {
		t.Fatalf("ids2=%v want %v", ids2, want2)
	}
	for i := range want2 {
		if ids2[i] != want2[i] {
			t.Fatalf("ids2=%v want %v", ids2, want2)
		}
	}
}

func TestEncodeWhitespaceControlCharsSplit(t *testing.T) {
	// \t and \n are BOTH unicode.IsControl and unicode.IsSpace; they must
	// split tokens like ordinary whitespace, not get silently dropped and
	// merge the surrounding words into one unknown token.
	vocab := []string{"[PAD]", "[UNK]", "[CLS]", "[SEP]", "hello", "world"}
	p := writeVocab(t, vocab)
	tok, err := New(p, true, "", "", "")
	if err != nil {
		t.Fatal(err)
	}
	want, _ := tok.Encode("hello world", MaxSeqLen)
	for _, sep := range []string{"\t", "\n"} {
		got, _ := tok.Encode("hello"+sep+"world", MaxSeqLen)
		if len(got) != len(want) {
			t.Fatalf("Encode(%q)=%v want same shape as %v", "hello"+sep+"world", got, want)
		}
		for i := range want {
			if got[i] != want[i] {
				t.Fatalf("Encode(%q)=%v want %v", "hello"+sep+"world", got, want)
			}
		}
	}
}

func TestEncodeCustomSpecialTokens(t *testing.T) {
	// MPNet-style framing: "<s>"=2, "</s>"=3, "[UNK]"=1, no [CLS]/[SEP] at all.
	vocab := []string{"[PAD]", "[UNK]", "<s>", "</s>", "hello", "world"}
	tok, err := New(writeVocab(t, vocab), true, "<s>", "</s>", "[UNK]")
	if err != nil {
		t.Fatal(err)
	}
	ids, _ := tok.Encode("hello world", MaxSeqLen)
	want := []int64{2, 4, 5, 3} // <s> hello world </s>
	if len(ids) != len(want) {
		t.Fatalf("ids=%v want %v", ids, want)
	}
	for i := range want {
		if ids[i] != want[i] {
			t.Fatalf("ids=%v want %v", ids, want)
		}
	}
}

func TestNewMissingSpecialTokenErrors(t *testing.T) {
	// Configured "<s>" is absent from a BERT vocab => construction must fail.
	vocab := []string{"[PAD]", "[UNK]", "[CLS]", "[SEP]", "hello"}
	if _, err := New(writeVocab(t, vocab), true, "<s>", "</s>", "[UNK]"); err == nil {
		t.Fatal("expected error for missing special token, got nil")
	}
}

func TestEncodeUnknownAndTruncate(t *testing.T) {
	vocab := []string{"[PAD]", "[UNK]", "[CLS]", "[SEP]", "hello"}
	tok, _ := New(writeVocab(t, vocab), true, "", "", "")
	ids, _ := tok.Encode("hello zzz", MaxSeqLen) // zzz -> [UNK]=1
	want := []int64{2, 4, 1, 3}
	for i := range want {
		if ids[i] != want[i] {
			t.Fatalf("ids=%v want %v", ids, want)
		}
	}
	// Truncation: maxLen=3 keeps [CLS] + 1 token + [SEP].
	ids2, _ := tok.Encode("hello hello hello", 3)
	if len(ids2) != 3 || ids2[0] != 2 || ids2[2] != 3 {
		t.Fatalf("truncated ids=%v want [2 4 3]", ids2)
	}
}
