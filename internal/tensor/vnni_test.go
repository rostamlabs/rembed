// SPDX-License-Identifier: Apache-2.0

package tensor

import (
	"math"
	"math/rand"
	"testing"
)

// vnniReference computes exactly what MatMulPackedVNNI must: quantize
// activations per row (u8, +128 offset), weights per column (s8,
// symmetric), accumulate in int32, dequantize. Pure Go, no assembly —
// the kernel's integer accumulation must match this BIT-EXACTLY (integer
// math has no rounding to hide behind).
func vnniReference(dst, a, bT []float32, m, k, n int) {
	for c := 0; c < n; c++ {
		var maxAbs float64
		for p := 0; p < k; p++ {
			if x := math.Abs(float64(bT[c*k+p])); x > maxAbs {
				maxAbs = x
			}
		}
		scale := maxAbs / 127
		if scale == 0 {
			scale = 1
		}
		var colSum int32
		qw := make([]int32, k)
		for p := 0; p < k; p++ {
			qw[p] = int32(int8(math.RoundToEven(float64(bT[c*k+p]) / scale)))
			colSum += qw[p]
		}
		for r := 0; r < m; r++ {
			var rowMax float32
			for p := 0; p < k; p++ {
				if x := float32(math.Abs(float64(a[r*k+p]))); x > rowMax {
					rowMax = x
				}
			}
			rs := rowMax / 127
			if rs == 0 {
				rs = 1
			}
			inv := 1 / rs
			var acc int32
			for p := 0; p < k; p++ {
				q := int32(math.RoundToEven(float64(a[r*k+p]*inv))) + 128
				if q < 0 {
					q = 0
				} else if q > 255 {
					q = 255
				}
				acc += q * qw[p]
			}
			dst[r*n+c] = float32(acc-128*colSum) * rs * float32(scale)
		}
	}
}

// TestMatMulPackedVNNIExact: the assembly path must equal the pure-Go
// reference bit-for-bit across shapes covering k%4 tails, multi-panel n,
// row padding, and values that graze the u8 clamp rails.
func TestMatMulPackedVNNIExact(t *testing.T) {
	if !hasVNNI && !hasVNNI512 {
		t.Skip("no AVX-VNNI or AVX-512-VNNI on this CPU")
	}
	rng := rand.New(rand.NewSource(99))
	shapes := [][3]int{
		{1, 1, 16}, {1, 3, 16}, {2, 4, 16}, {3, 5, 32},
		{4, 8, 16}, {5, 7, 48}, {7, 384, 384}, {12, 384, 1536},
		{13, 129, 64}, {64, 100, 128},
	}
	// Both an inline pool (deterministic) and a real 4-worker pool: the
	// review noted the exact test never exercised the parallel path.
	for _, workers := range []int{0, 4} {
		pool := NewPool(workers)
		runShapes(t, rng, shapes, pool)
		pool.Stop()
	}
}

func runShapes(t *testing.T, rng *rand.Rand, shapes [][3]int, pool *Pool) {
	t.Helper()
	for _, sh := range shapes {
		m, k, n := sh[0], sh[1], sh[2]
		a := make([]float32, m*k)
		bT := make([]float32, n*k)
		for i := range a {
			a[i] = rng.Float32()*4 - 2
		}
		for i := range bT {
			bT[i] = rng.Float32()*2 - 1
		}
		// One extreme row to graze the clamp rails.
		if k > 2 {
			a[0], a[1], a[2] = 127.4, -127.4, 0.0001
		}
		pb, err := PackB8VNNI(bT, k, n)
		if err != nil {
			t.Fatal(err)
		}
		mPad := PackAPad(m)
		qa := make([]uint8, mPad*pb.kg*4)
		rowScales := make([]float32, mPad)
		got := make([]float32, mPad*n)
		MatMulPackedVNNI(got, a, pb, m, qa, rowScales, pool)
		want := make([]float32, m*n)
		vnniReference(want, a, bT, m, k, n)
		for i := 0; i < m*n; i++ {
			if got[i] != want[i] {
				t.Fatalf("%dx%dx%d: [%d]=%v want %v (must be bit-exact: integer math)", m, k, n, i, got[i], want[i])
			}
		}
	}
}

// TestQuantizeRowU8Rails pins the clamp and zero-row behavior.
func TestQuantizeRowU8Rails(t *testing.T) {
	dst := make([]uint8, 8)
	s := QuantizeRowU8(dst, []float32{127, -127, 0, 63.5})
	if s == 0 {
		t.Fatal("zero scale for nonzero row")
	}
	if dst[0] != 255 || dst[1] != 1 || dst[2] != 128 {
		t.Fatalf("rails wrong: %v", dst[:4])
	}
	if dst[4] != 0 || dst[7] != 0 {
		t.Fatalf("tail not zeroed: %v", dst)
	}
	if s := QuantizeRowU8(dst[:4], []float32{0, 0, 0, 0}); s != 1 {
		t.Fatalf("zero row scale = %v, want 1", s)
	}
}

// TestVNNIEncodingsAgree runs the VEX and EVEX kernels directly on
// identical inputs and requires identical int32 accumulators. It can only
// run on hardware reporting BOTH encodings (Sapphire Rapids, Zen 5, or
// Intel SDE) — everywhere else exactly one kernel is legal and the other
// #UDs, so the test skips. This is the mechanical lock on twin drift:
// the two .s files must stay in step forever, and on both-capable
// hardware that is provable rather than absent-by-inspection.
func TestVNNIEncodingsAgree(t *testing.T) {
	if !hasVNNI || !hasVNNI512 {
		t.Skip("needs BOTH AVX-VNNI and AVX-512-VNNI (SPR/Zen5/SDE)")
	}
	rng := rand.New(rand.NewSource(7))
	for _, kg := range []int{1, 2, 13, 96, 384} {
		qa := make([]uint8, 4*kg*4)
		pb := make([]int8, kg*64)
		for i := range qa {
			qa[i] = uint8(rng.Intn(256))
		}
		for i := range pb {
			pb[i] = int8(rng.Intn(256) - 128)
		}
		var vex, evex [64]int32
		gemm4x16vnni(&vex[0], 16, &qa[0], kg*4, &pb[0], kg)
		gemm4x16vnni512(&evex[0], 16, &qa[0], kg*4, &pb[0], kg)
		if vex != evex {
			t.Fatalf("kg=%d: VEX and EVEX kernels disagree", kg)
		}
	}
}
