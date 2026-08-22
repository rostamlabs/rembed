// SPDX-License-Identifier: Apache-2.0

// Package tokenizer implements BERT WordPiece tokenization in pure Go.
// Ported from github.com/rostamlabs/rostam (semcache/local, -tags localembed).
package tokenizer

import (
	"bufio"
	"fmt"
	"os"
	"strings"
	"unicode"
)

// MaxSeqLen is the default maximum sequence length (BERT position-embedding
// limit); Encode truncates so [CLS] + tokens + [SEP] never exceeds it.
const MaxSeqLen = 512

// maxWordChars mirrors BERT's WordPiece max_input_chars_per_word: a single word
// longer than this (in runes) is emitted as one [UNK] without attempting the
// O(n^2) longest-prefix search, bounding work on pathological no-space input.
const maxWordChars = 100

type Tokenizer struct {
	vocab     map[string]int64
	lowerCase bool
	clsID     int64
	sepID     int64
	unkID     int64
}

// New loads a WordPiece vocab (one token per line, id = line number) and
// resolves the sequence-framing special tokens. clsTok/sepTok/unkTok name the
// tokens to frame with; an empty string selects the BERT default
// ("[CLS]"/"[SEP]"/"[UNK]"). It errors if any configured special token is
// absent from the vocab.
func New(vocabPath string, lowerCase bool, clsTok, sepTok, unkTok string) (*Tokenizer, error) {
	if clsTok == "" {
		clsTok = "[CLS]"
	}
	if sepTok == "" {
		sepTok = "[SEP]"
	}
	if unkTok == "" {
		unkTok = "[UNK]"
	}
	f, err := os.Open(vocabPath)
	if err != nil {
		return nil, err
	}
	defer func() { _ = f.Close() }()
	vocab := map[string]int64{}
	sc := bufio.NewScanner(f)
	sc.Buffer(make([]byte, 0, 64*1024), 1024*1024)
	var i int64
	for sc.Scan() {
		vocab[sc.Text()] = i
		i++
	}
	if err := sc.Err(); err != nil {
		return nil, err
	}
	for _, tk := range []string{clsTok, sepTok, unkTok} {
		if _, ok := vocab[tk]; !ok {
			return nil, fmt.Errorf("vocab %q is missing required special token %q", vocabPath, tk)
		}
	}
	t := &Tokenizer{vocab: vocab, lowerCase: lowerCase, clsID: vocab[clsTok], sepID: vocab[sepTok], unkID: vocab[unkTok]}
	return t, nil
}

// VocabSize returns the number of entries in the loaded vocab.
func (t *Tokenizer) VocabSize() int { return len(t.vocab) }

// basicTokens splits on whitespace and separates punctuation, mirroring BERT's
// BasicTokenizer (control-char strip, optional lowercase). CJK handling is out
// of scope for the MVP English models.
func (t *Tokenizer) basicTokens(text string) []string {
	if t.lowerCase {
		text = strings.ToLower(text)
	}
	var out []string
	var cur strings.Builder
	flush := func() {
		if cur.Len() > 0 {
			out = append(out, cur.String())
			cur.Reset()
		}
	}
	for _, r := range text {
		switch {
		case unicode.IsSpace(r):
			flush()
		case unicode.IsControl(r):
			// drop
		case unicode.IsPunct(r) || unicode.IsSymbol(r):
			flush()
			out = append(out, string(r))
		default:
			cur.WriteRune(r)
		}
	}
	flush()
	return out
}

// wordPiece greedily matches the longest vocab prefix, using "##" for
// continuations, falling back to [UNK] for the whole word.
func (t *Tokenizer) wordPiece(word string) []int64 {
	runes := []rune(word)
	if len(runes) > maxWordChars {
		return []int64{t.unkID} // over-long word: skip the O(n^2) search
	}
	var out []int64
	start := 0
	for start < len(runes) {
		end := len(runes)
		var curID int64 = -1
		for start < end {
			sub := string(runes[start:end])
			if start > 0 {
				sub = "##" + sub
			}
			if id, ok := t.vocab[sub]; ok {
				curID = id
				break
			}
			end--
		}
		if curID < 0 {
			return []int64{t.unkID} // whole word is unknown
		}
		out = append(out, curID)
		start = end
	}
	return out
}

// Encode returns input_ids and attention_mask for one text: [CLS] tokens [SEP],
// truncated so the total length never exceeds maxLen.
func (t *Tokenizer) Encode(text string, maxLen int) (ids []int64, mask []int64) {
	if maxLen < 2 {
		maxLen = 2
	}
	ids = append(ids, t.clsID)
	for _, w := range t.basicTokens(text) {
		for _, id := range t.wordPiece(w) {
			if len(ids) >= maxLen-1 { // reserve room for [SEP]
				goto done
			}
			ids = append(ids, id)
		}
	}
done:
	ids = append(ids, t.sepID)
	mask = make([]int64, len(ids))
	for i := range mask {
		mask[i] = 1
	}
	return ids, mask
}
