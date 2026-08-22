// SPDX-License-Identifier: Apache-2.0

package tensor

import (
	"fmt"
	"math"
	"math/rand"
	"testing"
)

// TestMatMulPacked8AgainstDequantizedReference: the int8 kernel's result
// must equal a float64 GEMM over the DEQUANTIZED weights (scale·q) within
// fp32 rounding — quantization error is a property of PackB8's rounding,
// not of the kernel, and this reference excludes it entirely. Same shape
// coverage and guard checks as the fp32 packed test.
func TestMatMulPacked8AgainstDequantizedReference(t *testing.T) {
	if !hasSIMD {
		t.Skip("no AVX2+FMA on this CPU")
	}
	shapes := [][3]int{
		{1, 1, 16},
		{3, 7, 16},
		{4, 33, 32},
		{5, 384, 48},
		{12, 384, 384},
		{12, 384, 1152},
		{13, 1536, 384},
		{64, 100, 64},
	}
	rng := rand.New(rand.NewSource(23))
	for _, sh := range shapes {
		m, k, n := sh[0], sh[1], sh[2]
		a := make([]float32, m*k)
		bT := make([]float32, n*k)
		for i := range a {
			a[i] = rng.Float32()*2 - 1
		}
		for i := range bT {
			bT[i] = rng.Float32()*2 - 1
		}
		pb, err := PackB8(bT, k, n)
		if err != nil {
			t.Fatal(err)
		}
		// Dequantized weights: what the kernel actually multiplies by.
		deq := make([]float64, n*k)
		for j := range n {
			s := float64(pb.scales[j])
			panel := pb.data[(j/16)*k*16:]
			for p := range k {
				deq[j*k+p] = float64(panel[p*16+j%16]) * s
			}
		}
		mPad := PackAPad(m)
		const guard = 32
		dstBuf := make([]float32, mPad*n+guard)
		for i := range dstBuf {
			dstBuf[i] = -999
		}
		aPack := make([]float32, mPad*k)
		MatMulPacked8(dstBuf[:mPad*n], a, pb, m, aPack, nil)

		for i := range m {
			for j := range n {
				var want float64
				for p := range k {
					want += float64(a[i*k+p]) * deq[j*k+p]
				}
				got := float64(dstBuf[i*n+j])
				tol := max(1e-5, 1e-6*math.Sqrt(float64(k))) * (1 + math.Abs(want))
				if d := math.Abs(got - want); d > tol {
					t.Fatalf("%dx%dx%d: [%d,%d]=%g want %g (diff %g)", m, k, n, i, j, got, want, d)
				}
			}
		}
		for i := m; i < mPad; i++ {
			for j := range n {
				if dstBuf[i*n+j] != 0 {
					t.Fatalf("%dx%dx%d: pad row %d col %d = %v, want 0", m, k, n, i, j, dstBuf[i*n+j])
				}
			}
		}
		for g := range guard {
			if dstBuf[mPad*n+g] != -999 {
				t.Fatalf("%dx%dx%d: guard slot %d overwritten", m, k, n, g)
			}
		}
	}
}

// TestPackB8QuantizationError pins the quantization error bound itself:
// per-channel symmetric int8 keeps every dequantized weight within
// scale/2 = maxabs/254 of the original.
func TestPackB8QuantizationError(t *testing.T) {
	if !hasSIMD {
		t.Skip("no AVX2+FMA on this CPU")
	}
	rng := rand.New(rand.NewSource(29))
	const k, n = 100, 32
	bT := make([]float32, n*k)
	for i := range bT {
		bT[i] = (rng.Float32()*2 - 1) * float32(math.Exp(float64(rng.Intn(6)))) // varied magnitudes
	}
	pb, err := PackB8(bT, k, n)
	if err != nil {
		t.Fatal(err)
	}
	for j := range n {
		var maxAbs float64
		for p := range k {
			if a := math.Abs(float64(bT[j*k+p])); a > maxAbs {
				maxAbs = a
			}
		}
		bound := maxAbs/254 + 1e-9
		panel := pb.data[(j/16)*k*16:]
		for p := range k {
			deq := float64(panel[p*16+j%16]) * float64(pb.scales[j])
			if d := math.Abs(deq - float64(bT[j*k+p])); d > bound {
				t.Fatalf("channel %d elem %d: |deq-orig|=%g exceeds scale/2=%g", j, p, d, bound)
			}
		}
	}
}

func BenchmarkMatMulPacked8(b *testing.B) {
	if !hasSIMD {
		b.Skip("no AVX2+FMA on this CPU")
	}
	for _, sh := range [][3]int{{128, 384, 1536}, {12, 384, 384}, {12, 384, 1152}} {
		m, k, n := sh[0], sh[1], sh[2]
		b.Run(fmt.Sprintf("%dx%dx%d", m, k, n), func(b *testing.B) {
			a := make([]float32, m*k)
			bT := make([]float32, n*k)
			for i := range a {
				a[i] = float32(i%7) * 0.1
			}
			for i := range bT {
				bT[i] = float32(i%5) * 0.1
			}
			pb, err := PackB8(bT, k, n)
			if err != nil {
				b.Fatal(err)
			}
			mPad := PackAPad(m)
			dst := make([]float32, mPad*n)
			aPack := make([]float32, mPad*k)
			b.SetBytes(int64(m*k+n*k+m*n) * 4)
			for b.Loop() {
				MatMulPacked8(dst, a, pb, m, aPack, nil)
			}
		})
	}
}
