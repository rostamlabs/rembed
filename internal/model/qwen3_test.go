// SPDX-License-Identifier: Apache-2.0

package model

import (
	"math"
	"math/rand"
	"testing"
)

// naiveCausalAttn is the straightforward reference: per query i, softmax over
// keys [0,i] of (q_i·k_j)·scale, then the weighted sum of v_j. Used to pin
// flashAttnCausalHead's tiled online-softmax against ground truth at the
// block boundaries the golden's fixed sequence lengths don't all hit.
func naiveCausalAttn(qH, kH, vH []float32, seq, dh int, scale float32) []float32 {
	out := make([]float32, seq*dh)
	for i := 0; i < seq; i++ {
		s := make([]float64, i+1)
		mx := math.Inf(-1)
		for j := 0; j <= i; j++ {
			var d float64
			for c := 0; c < dh; c++ {
				d += float64(qH[i*dh+c]) * float64(kH[j*dh+c])
			}
			s[j] = d * float64(scale)
			if s[j] > mx {
				mx = s[j]
			}
		}
		var sum float64
		for j := range s {
			s[j] = math.Exp(s[j] - mx)
			sum += s[j]
		}
		for c := 0; c < dh; c++ {
			var acc float64
			for j := 0; j <= i; j++ {
				acc += s[j] / sum * float64(vH[j*dh+c])
			}
			out[i*dh+c] = float32(acc)
		}
	}
	return out
}

// TestFlashAttnCausalHead pins the tiled online-softmax attention against the
// naive reference across sizes below, at, and straddling the flashBlk (64)
// boundary — where the block-skip and diagonal-mask logic is most fragile.
func TestFlashAttnCausalHead(t *testing.T) {
	rng := rand.New(rand.NewSource(7))
	dh := 64
	for _, seq := range []int{1, 2, 5, 63, 64, 65, 120, 127, 128, 129, 200} {
		qH := make([]float32, seq*dh)
		kH := make([]float32, seq*dh)
		vH := make([]float32, seq*dh)
		for i := range qH {
			qH[i] = rng.Float32()*2 - 1
			kH[i] = rng.Float32()*2 - 1
			vH[i] = rng.Float32()*2 - 1
		}
		scale := float32(1 / math.Sqrt(float64(dh)))
		want := naiveCausalAttn(qH, kH, vH, seq, dh, scale)

		bq := min(flashBlk, seq)
		got := make([]float32, seq*dh)
		flashAttnCausalHead(qH, kH, vH, got, seq, dh, scale,
			make([]float32, bq*bq), make([]float32, bq*dh),
			make([]float32, bq), make([]float32, bq))

		var maxd float64
		for i := range want {
			if d := math.Abs(float64(got[i] - want[i])); d > maxd {
				maxd = d
			}
		}
		if maxd > 1e-4 {
			t.Errorf("seq=%d: flash vs naive maxAbs=%g > 1e-4", seq, maxd)
		}
	}
}
