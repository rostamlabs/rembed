// SPDX-License-Identifier: Apache-2.0

package sentencepiece

import (
	"encoding/binary"
	"fmt"
	"strings"
	"unicode/utf8"
)

// normalizer ports sentencepiece's Normalizer (normalizer.cc): NMT-NFKC
// text normalization driven by the model's precompiled_charsmap — a
// darts-clone double-array trie mapping source byte sequences to
// replacement strings — plus the whitespace protocol (collapse runs, trim
// ends, dummy " " prefix, escape ' ' to ▁ U+2581).
type normalizer struct {
	trie       *doubleArray
	normalized string // replacement blob; entries are NUL-terminated
	spec       normalizerSpec
}

const spaceSymbol = "▁" // LOWER ONE EIGHTH BLOCK, "▁"

// newNormalizer decodes the precompiled charsmap:
// <trie size uint32 LE><double-array units><replacement blob>.
func newNormalizer(spec normalizerSpec) (*normalizer, error) {
	n := &normalizer{spec: spec}
	blob := spec.precompiledCharsmap
	if len(blob) == 0 {
		// A model may carry no charsmap (e.g. "identity" normalization);
		// the whitespace protocol still applies.
		return n, nil
	}
	if len(blob) < 4 {
		return nil, fmt.Errorf("sentencepiece: charsmap too short (%d bytes)", len(blob))
	}
	trieSize := binary.LittleEndian.Uint32(blob[:4])
	if int(trieSize)%4 != 0 || 4+int(trieSize) > len(blob) {
		return nil, fmt.Errorf("sentencepiece: charsmap trie size %d out of bounds (%d-byte blob)", trieSize, len(blob))
	}
	trieBlob := blob[4 : 4+int(trieSize)]
	units := make([]uint32, len(trieBlob)/4)
	for i := range units {
		units[i] = binary.LittleEndian.Uint32(trieBlob[i*4:])
	}
	n.trie = &doubleArray{units: units}
	n.normalized = string(blob[4+int(trieSize):])
	return n, nil
}

// replacement returns the NUL-terminated string at offset v of the blob.
func (n *normalizer) replacement(v int) string {
	if v < 0 || v >= len(n.normalized) {
		return ""
	}
	end := strings.IndexByte(n.normalized[v:], 0)
	if end < 0 {
		return n.normalized[v:]
	}
	return n.normalized[v : v+end]
}

// normalizePrefix is Normalizer::NormalizePrefix: the longest charsmap
// match at the head of input wins and maps to its replacement; otherwise
// one valid UTF-8 rune passes through unchanged, and one INVALID byte
// becomes U+FFFD (consuming exactly one byte).
func (n *normalizer) normalizePrefix(input string) (normalized string, consumed int) {
	if input == "" {
		return "", 0
	}
	longestLen, longestVal := 0, 0
	if n.trie != nil {
		n.trie.commonPrefixSearch(input, func(value, length int) {
			if longestLen == 0 || length > longestLen {
				longestLen = length
				longestVal = value
			}
		})
	}
	if longestLen == 0 {
		r, size := utf8.DecodeRuneInString(input)
		if r == utf8.RuneError && size <= 1 {
			return "�", 1
		}
		return input[:size], size
	}
	return n.replacement(longestVal), longestLen
}

// normalize is Normalizer::Normalize, ported flow-for-flow.
func (n *normalizer) normalize(input string) string {
	if input == "" {
		return ""
	}
	// Ignore heading space: drop prefixes that normalize to exactly " ".
	if n.spec.removeExtraWhitespaces {
		for input != "" {
			p, consumed := n.normalizePrefix(input)
			if p != " " {
				break
			}
			input = input[consumed:]
		}
	}
	if input == "" {
		return ""
	}

	var out strings.Builder
	out.Grow(len(input) + 8)
	addWS := func() {
		if n.spec.escapeWhitespaces {
			out.WriteString(spaceSymbol)
		} else {
			out.WriteByte(' ')
		}
	}
	if n.spec.addDummyPrefix {
		addWS()
	}

	isPrevSpace := n.spec.removeExtraWhitespaces
	for input != "" {
		sp, consumed := n.normalizePrefix(input)
		// Strip heading spaces of this chunk while the previous output
		// ended in whitespace (collapses runs to one).
		for isPrevSpace && strings.HasPrefix(sp, " ") {
			sp = sp[1:]
		}
		if sp != "" {
			for i := 0; i < len(sp); i++ {
				if n.spec.escapeWhitespaces && sp[i] == ' ' {
					out.WriteString(spaceSymbol)
				} else {
					out.WriteByte(sp[i])
				}
			}
			isPrevSpace = strings.HasSuffix(sp, " ")
		}
		input = input[consumed:]
		if !n.spec.removeExtraWhitespaces {
			isPrevSpace = false
		}
	}

	s := out.String()
	if n.spec.removeExtraWhitespaces {
		trailer := " "
		if n.spec.escapeWhitespaces {
			trailer = spaceSymbol
		}
		for strings.HasSuffix(s, trailer) {
			s = s[:len(s)-len(trailer)]
		}
	}
	return s
}

// doubleArray is a read-only darts-clone double-array trie, traversal
// ported from the darts.h sentencepiece bundles. Each unit packs label,
// offset, and a has-leaf bit; children are found by XORing offsets with
// byte labels.
type doubleArray struct {
	units []uint32
}

func (d *doubleArray) commonPrefixSearch(key string, emit func(value, length int)) {
	if len(d.units) == 0 {
		return
	}
	// unit accessors, exactly as darts.h defines them.
	hasLeaf := func(u uint32) bool { return (u>>8)&1 == 1 }
	value := func(u uint32) int { return int(u & ((1 << 31) - 1)) }
	label := func(u uint32) uint32 { return u & ((1 << 31) | 0xFF) }
	offset := func(u uint32) uint32 { return (u >> 10) << ((u & (1 << 9)) >> 6) }

	nodePos := uint32(0)
	unit := d.units[nodePos]
	nodePos ^= offset(unit)
	for i := 0; i < len(key); i++ {
		c := uint32(key[i])
		if c == 0 {
			// darts keys are NUL-terminated; a NUL in the query ends it.
			return
		}
		nodePos ^= c
		if nodePos >= uint32(len(d.units)) {
			return
		}
		unit = d.units[nodePos]
		if label(unit) != c {
			return
		}
		nodePos ^= offset(unit)
		if hasLeaf(unit) {
			if nodePos < uint32(len(d.units)) {
				emit(value(d.units[nodePos]), i+1)
			}
		}
	}
}
