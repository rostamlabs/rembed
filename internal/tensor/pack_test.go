// SPDX-License-Identifier: Apache-2.0

package tensor

import (
	"fmt"
	"math"
	"math/rand"
	"testing"
)

// TestMatMulPackedAgainstFloat64Reference is the packed path's safety net:
// random matrices over shapes exercising m%4 remainders (pad rows), k
// values with and without vector tails inside gemm's k-loop (any k works —
// the loop steps by 1), and single/multi panel n. Guard region beyond the
// padded rows catches the kernel writing where it must not.
func TestMatMulPackedAgainstFloat64Reference(t *testing.T) {
	if !hasSIMD {
		t.Skip("no AVX2+FMA on this CPU")
	}
	shapes := [][3]int{ // m, k, n (n % 16 == 0)
		{1, 1, 16},
		{3, 7, 16},   // pad row + odd k, single panel
		{4, 33, 32},  // exact row panel
		{5, 384, 48}, // pad rows + model-like k
		{12, 384, 384},
		{12, 384, 1152}, // fused-QKV shape
		{13, 1536, 384}, // FFN2-like with pad row
		{64, 100, 64},
	}
	rng := rand.New(rand.NewSource(11))
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
		pb, err := PackB(bT, k, n)
		if err != nil {
			t.Fatal(err)
		}
		mPad := PackAPad(m)
		const guard = 32
		dstBuf := make([]float32, mPad*n+guard)
		for i := range dstBuf {
			dstBuf[i] = -999
		}
		aPack := make([]float32, mPad*k)
		MatMulPacked(dstBuf[:mPad*n], a, pb, m, aPack, nil)

		// fp32 error of a sequential k-chain grows ~√k (the gemm kernel
		// accumulates one chain per element, unlike dot4's 8-lane split),
		// so the tolerance scales with it: at k=1536 legitimate noise
		// reaches ~1.4e-5. A real indexing bug is off by whole products
		// (~0.25), three orders of magnitude beyond this.
		for i := range m {
			for j := range n {
				var want float64
				for p := range k {
					want += float64(a[i*k+p]) * float64(bT[j*k+p])
				}
				got := float64(dstBuf[i*n+j])
				tol := max(1e-5, 1e-6*math.Sqrt(float64(k))) * (1 + math.Abs(want))
				if d := math.Abs(got - want); d > tol {
					t.Fatalf("%dx%dx%d: [%d,%d]=%g want %g (diff %g)", m, k, n, i, j, got, want, d)
				}
			}
		}
		// Pad rows must be exact zeros (packA4 zero-fills), and the guard
		// region untouched.
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

func TestPackBRejectsBadShapes(t *testing.T) {
	if !hasSIMD {
		t.Skip("no AVX2+FMA on this CPU")
	}
	if _, err := PackB(make([]float32, 15*3), 3, 15); err == nil {
		t.Fatal("expected error for n%16 != 0")
	}
	if _, err := PackB(make([]float32, 10), 5, 16); err == nil {
		t.Fatal("expected error for wrong length")
	}
}

func BenchmarkMatMulPacked(b *testing.B) {
	if !hasSIMD {
		b.Skip("no AVX2+FMA on this CPU")
	}
	// Same shapes as benchMatMul so packed vs dot4 vs scalar A/B directly.
	for _, sh := range [][3]int{{128, 384, 1536}, {12, 384, 384}, {3, 384, 384}, {1, 384, 384}} {
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
			pb, err := PackB(bT, k, n)
			if err != nil {
				b.Fatal(err)
			}
			mPad := PackAPad(m)
			dst := make([]float32, mPad*n)
			aPack := make([]float32, mPad*k)
			b.SetBytes(int64(m*k+n*k+m*n) * 4)
			for b.Loop() {
				MatMulPacked(dst, a, pb, m, aPack, nil)
			}
		})
	}
}

// TestPackedPanelWidthsAgree: on AVX-512 hardware PackB builds 32-column
// panels for n%32==0 — this test forces BOTH widths on the same data and
// requires bit-identical output (per-element accumulation order is the
// same sequential FMA chain in gemm4x16 and gemm4x32). Skips where only
// the 16-wide kernel exists.
func TestPackedPanelWidthsAgree(t *testing.T) {
	if !hasAVX512 {
		t.Skip("no AVX-512F on this CPU")
	}
	rng := rand.New(rand.NewSource(21))
	pool := NewPool(0)
	defer pool.Stop()
	for _, sh := range [][3]int{{1, 3, 32}, {4, 8, 32}, {5, 129, 64}, {12, 384, 384}, {13, 384, 1536}, {64, 100, 128}} {
		m, k, n := sh[0], sh[1], sh[2]
		a := make([]float32, m*k)
		bT := make([]float32, n*k)
		for i := range a {
			a[i] = rng.Float32()*2 - 1
		}
		for i := range bT {
			bT[i] = rng.Float32()*2 - 1
		}
		wide, err := PackB(bT, k, n) // panelW=32 on this hardware
		if err != nil {
			t.Fatal(err)
		}
		if wide.panelW != 32 {
			t.Fatalf("expected 32-wide panels on AVX-512 hardware, got %d", wide.panelW)
		}
		// 16-wide pack via the REAL packer (shared helper), so this test
		// cannot drift from PackB's layout.
		narrow := &PackedB{K: k, N: n, panelW: 16, data: make([]float32, k*n)}
		packBInto(narrow.data, bT, k, n, 16)
		mPad := PackAPad(m)
		aPack := make([]float32, mPad*k)
		got := make([]float32, mPad*n)
		want := make([]float32, mPad*n)
		MatMulPacked(got, a, wide, m, aPack, pool)
		MatMulPacked(want, a, narrow, m, aPack, pool)
		for i := 0; i < m*n; i++ {
			if got[i] != want[i] {
				t.Fatalf("%dx%dx%d: [%d] 32-wide %v != 16-wide %v (must be bit-identical)", m, k, n, i, got[i], want[i])
			}
		}
	}
}
