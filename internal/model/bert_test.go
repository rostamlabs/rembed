// SPDX-License-Identifier: Apache-2.0

package model

import (
	"testing"

	"github.com/rostamlabs/rembed/internal/tensor"
)

// TestScratchResizeTilesExactly pins the per-head slice arithmetic that
// Forward relies on: for any (seq, H, I, dh) the golden tests' single MiniLM
// config would never exercise, the per-head sub-slices of qh/kh/ch/vhT and
// scores must tile their parent buffers exactly — no gap (stale pooled data
// would leak through) and no overlap (parallel head workers would race).
func TestScratchResizeTilesExactly(t *testing.T) {
	cases := [][4]int{ // seq, H, I, dh
		{1, 384, 1536, 32},   // MiniLM at minimum seq
		{12, 384, 1536, 32},  // MiniLM typical
		{512, 384, 1536, 32}, // MiniLM max
		{7, 768, 3072, 64},   // BERT-base-like, odd seq
		{3, 512, 2048, 64},   // dh=64 variant
		{5, 384, 1536, 384},  // single head (heads=1)
	}
	for _, c := range cases {
		seq, H, I, dh := c[0], c[1], c[2], c[3]
		heads := H / dh
		if heads*dh != H {
			t.Fatalf("bad case %v: H not divisible by dh", c)
		}
		var s scratch
		s.resize(seq, H, I, dh)

		mPad := tensor.PackAPad(seq)
		wantLens := map[string][2]int{
			"x":         {len(s.x), seq * H},
			"qkv":       {len(s.qkv), mPad * 3 * H},
			"ctxOut":    {len(s.ctxOut), seq * H},
			"attnOut":   {len(s.attnOut), mPad * H},
			"ffnOut":    {len(s.ffnOut), mPad * H},
			"ffnHidden": {len(s.ffnHidden), mPad * I},
			"aPack":     {len(s.aPack), mPad * max(H, I)},
			"scores":    {len(s.scores), heads * seq * seq},
			"qh":        {len(s.qh), seq * H},
			"kh":        {len(s.kh), seq * H},
			"ch":        {len(s.ch), seq * H},
			"vhT":       {len(s.vhT), H * seq},
		}
		for name, lens := range wantLens {
			if lens[0] != lens[1] {
				t.Fatalf("case %v: %s has len %d, want %d", c, name, lens[0], lens[1])
			}
		}

		// The per-head ranges Forward slices out must partition [0, len).
		if heads*seq*dh != len(s.qh) {
			t.Fatalf("case %v: head panels cover %d of qh's %d", c, heads*seq*dh, len(s.qh))
		}
		if heads*dh*seq != len(s.vhT) {
			t.Fatalf("case %v: head panels cover %d of vhT's %d", c, heads*dh*seq, len(s.vhT))
		}
		if heads*seq*seq != len(s.scores) {
			t.Fatalf("case %v: head panels cover %d of scores' %d", c, heads*seq*seq, len(s.scores))
		}

		// Growing to a bigger shape and shrinking back must keep lengths
		// exact (grow-only capacity, exact lengths).
		s.resize(seq+9, H, I, dh)
		s.resize(seq, H, I, dh)
		if len(s.scores) != heads*seq*seq || len(s.qh) != seq*H {
			t.Fatalf("case %v: re-resize broke lengths: scores=%d qh=%d", c, len(s.scores), len(s.qh))
		}
	}
}
