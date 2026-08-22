// SPDX-License-Identifier: Apache-2.0

package tensor

import (
	"math"
	"math/rand"
	"testing"
)

// TestDot4 is the load-bearing test for the hand-written assembly (AVX2 and NEON): an
// asserted sweep over every k in [1, 40] — covering zero, one, and many
// vector iterations combined with every scalar-tail length 0..7 — using
// index- and lane-dependent random values, so a wrong-lane reduction, a
// dropped tail element, or a wrong intra-row offset cannot cancel out.
// Guard slots around dst catch out-of-bounds writes.
func TestDot4(t *testing.T) {
	if !hasSIMD {
		t.Skip("no native dot kernel on this architecture")
	}
	rng := rand.New(rand.NewSource(7))
	for k := 1; k <= 40; k++ {
		a := make([]float32, k)
		bs := make([][]float32, 4)
		for c := range bs {
			bs[c] = make([]float32, k)
		}
		for p := range k {
			a[p] = rng.Float32()*2 - 1
			for c := range bs {
				bs[c][p] = rng.Float32()*2 - 1
			}
		}
		want := [4]float64{}
		for c := range bs {
			for p := range k {
				want[c] += float64(a[p]) * float64(bs[c][p])
			}
		}
		// dst embedded in a guarded buffer: any write outside dst[0:4]
		// lands on a sentinel.
		buf := []float32{-999, -999, 0, 0, 0, 0, -999, -999}
		dst := buf[2:6]
		dot4(&dst[0], &a[0], &bs[0][0], &bs[1][0], &bs[2][0], &bs[3][0], k)
		for c := range want {
			if d := math.Abs(float64(dst[c]) - want[c]); d > 1e-5*(1+math.Abs(want[c])) {
				t.Fatalf("k=%d lane %d: got %v want %v (diff %g)", k, c, dst[c], want[c], d)
			}
		}
		for _, gi := range []int{0, 1, 6, 7} {
			if buf[gi] != -999 {
				t.Fatalf("k=%d: guard slot %d overwritten: %v", k, gi, buf[gi])
			}
		}
	}
}
