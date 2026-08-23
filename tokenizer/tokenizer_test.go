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

func TestEncodeStripsAccentsWhenLowercasing(t *testing.T) {
	// HF couples accent stripping to do_lower_case: "café" must tokenize as
	// "cafe", both for precomposed é and for e + combining acute.
	vocab := []string{"[PAD]", "[UNK]", "[CLS]", "[SEP]", "cafe", "resume"}
	tok, _ := New(writeVocab(t, vocab), true, "", "", "")
	for _, text := range []string{"café résumé", "cafe\u0301 re\u0301sume\u0301"} {
		ids, _ := tok.Encode(text, MaxSeqLen)
		want := []int64{2, 4, 5, 3} // [CLS] cafe resume [SEP]
		if !equalIDs(ids, want) {
			t.Fatalf("Encode(%q)=%v want %v", text, ids, want)
		}
	}
	// Without lowercasing, accents are preserved (HF strip_accents=None).
	tokCased, _ := New(writeVocab(t, []string{"[PAD]", "[UNK]", "[CLS]", "[SEP]", "café"}), false, "", "", "")
	ids, _ := tokCased.Encode("café", MaxSeqLen)
	if !equalIDs(ids, []int64{2, 4, 3}) {
		t.Fatalf("cased Encode(café)=%v want [2 4 3]", ids)
	}
}

func TestEncodeCJKIdeographsSplitIndividually(t *testing.T) {
	// HF space-pads CJK ideographs: each is its own word, never merged into
	// one [UNK] run. 你=4 好=5, 界 unknown → [UNK]=1.
	vocab := []string{"[PAD]", "[UNK]", "[CLS]", "[SEP]", "你", "好"}
	tok, _ := New(writeVocab(t, vocab), true, "", "", "")
	ids, _ := tok.Encode("你好界", MaxSeqLen)
	want := []int64{2, 4, 5, 1, 3}
	if !equalIDs(ids, want) {
		t.Fatalf("ids=%v want %v", ids, want)
	}
}

func TestEncodeNonASCIISymbolsAreNotSeparators(t *testing.T) {
	// BERT's _is_punctuation treats ASCII symbols ($) as separators but NOT
	// non-ASCII symbols (€): "€100" is one word → € + ##100, while "$100"
	// splits into $ + 100 (no continuation).
	vocab := []string{"[PAD]", "[UNK]", "[CLS]", "[SEP]", "€", "##100", "$", "100"}
	tok, _ := New(writeVocab(t, vocab), true, "", "", "")
	ids, _ := tok.Encode("€100 $100", MaxSeqLen)
	want := []int64{2, 4, 5, 6, 7, 3}
	if !equalIDs(ids, want) {
		t.Fatalf("ids=%v want %v", ids, want)
	}
}

func TestEncodeDropsReplacementChar(t *testing.T) {
	vocab := []string{"[PAD]", "[UNK]", "[CLS]", "[SEP]", "hello"}
	tok, _ := New(writeVocab(t, vocab), true, "", "", "")
	// U+FFFD is dropped in place (HF _clean_text), so "hel�lo" = "hello".
	ids, _ := tok.Encode("hel�lo", MaxSeqLen)
	if !equalIDs(ids, []int64{2, 4, 3}) {
		t.Fatalf("ids=%v want [2 4 3]", ids)
	}
}

func equalIDs(a, b []int64) bool {
	if len(a) != len(b) {
		return false
	}
	for i := range a {
		if a[i] != b[i] {
			return false
		}
	}
	return true
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

// TestCleanTextCategoryC pins HF _clean_text parity for category-C
// characters, the review's differential findings distilled: \t\n\r are
// the ONLY C characters that split; every other Cc/Cf/Co/Cs character is
// deleted (joining its neighbors); unassigned codepoints (Cn) are KEPT —
// HF keeps them, and Go's combined unicode.C table would have silently
// deleted anything newer than the toolchain's Unicode tables.
func TestCleanTextCategoryC(t *testing.T) {
	tok, err := New(writeVocab(t, []string{"[PAD]", "[UNK]", "[CLS]", "[SEP]", "hello", "world", "a", "b"}), true, "", "", "")
	if err != nil {
		t.Fatal(err)
	}
	cases := []struct {
		name, text, want string
	}{
		{"tab splits", "hello\tworld", "hello world"},
		{"vertical tab deletes+joins", "hello\vworld", "helloworld"},
		{"form feed deletes+joins", "hello\fworld", "helloworld"},
		{"NEL U+0085 deletes+joins", "hello\u0085world", "helloworld"},
		{"soft hyphen U+00AD deletes", "hy\u00adphen", "hyphen"},
		{"ZWSP U+200B deletes", "a\u200bb", "ab"},
		{"ZWNJ U+200C deletes (Persian)", "a\u200cb", "ab"},
		{"BOM U+FEFF deletes", "\ufeffhello", "hello"},
		{"private use U+E000 deletes", "a\ue000b", "ab"},
		{"unassigned U+0378 KEPT", "a\u0378b", "a\u0378b"},
	}
	for _, c := range cases {
		got := strings.Join(tok.basicTokens(c.text), "|")
		want := strings.Join(tok.basicTokens(c.want), "|")
		if got != want {
			t.Errorf("%s: basicTokens(%q) = %q, want same as %q (= %q)", c.name, c.text, got, c.want, want)
		}
	}
}
