// SPDX-License-Identifier: Apache-2.0

package tensor

import (
	"fmt"
	"math"
	"math/rand"
	"runtime"
	"sync/atomic"
	"testing"
)

func almostEqual(t *testing.T, got, want []float32, tol float64, what string) {
	t.Helper()
	if len(got) != len(want) {
		t.Fatalf("%s: len=%d want %d", what, len(got), len(want))
	}
	for i := range want {
		if d := math.Abs(float64(got[i] - want[i])); d > tol {
			t.Fatalf("%s: [%d]=%v want %v (diff %g)", what, i, got[i], want[i], d)
		}
	}
}

func TestMatMulNaiveSmall(t *testing.T) {
	// A = [1 2 3; 4 5 6] (2×3), B = [7 8; 9 10; 11 12] (3×2) → C = A·B (2×2).
	// bT is Bᵀ = [7 9 11; 8 10 12] laid out [n×k] = [2×3].
	a := []float32{1, 2, 3, 4, 5, 6}
	bT := []float32{7, 9, 11, 8, 10, 12}
	dst := make([]float32, 4)
	MatMulNaive(dst, a, bT, 2, 3, 2)
	almostEqual(t, dst, []float32{58, 64, 139, 154}, 0, "matmul")
}

// kernels lists every matmul body; each must pass the float64 cross-check
// below — it is the per-kernel safety net of the optimization ladder.
// exact marks kernels whose k-accumulation order matches naive (bit-identical
// on non-FMA-fusing targets); the SIMD kernel reduces 8 lanes horizontally,
// so it is held to the relative-tolerance check only.
var kernels = func() map[string]struct {
	fn    MatMulFunc
	exact bool
} {
	m := map[string]struct {
		fn    MatMulFunc
		exact bool
	}{
		"naive":    {MatMulNaive, true},
		"blocked":  {MatMulBlocked, true},
		"parallel": {MatMulParallel, true},
	}
	if hasSIMD {
		m["simd"] = struct {
			fn    MatMulFunc
			exact bool
		}{MatMulSIMD, false}
	}
	return m
}()

// TestMatMulAgainstFloat64Reference cross-checks every kernel body against a
// float64 reference on random matrices, over shapes that exercise the
// blocking edges: micro-kernel remainders (n%4 != 0), tile remainders
// (m > blockM with a partial last tile), and degenerate sizes.
func TestMatMulAgainstFloat64Reference(t *testing.T) {
	shapes := [][3]int{
		{17, 33, 29},          // nothing divides anything
		{1, 1, 1},             // degenerate
		{12, 384, 384},        // MiniLM projection shape
		{12, 384, 1536},       // MiniLM FFN shape (n % 4 == 0)
		{blockM + 5, 16, 7},   // partial last i-tile AND n remainder
		{2 * blockM, 8, 4},    // exact tile boundaries
		{2*blockM + 7, 33, 7}, // multiple full i-tiles PLUS both remainders
		{3, 5, blockN - 1},    // n smaller than the micro-kernel
		// Shapes that clear MatMulParallel's per-unit gate with awkward
		// column remainders, so the parallel path's partial final column
		// block is pinned (the model's own dims are all 64-aligned and
		// would never catch it):
		{blockM, 384, 129},
		{blockM + 6, 400, 191},
		{2, 384, 384}, // tiny m through the parallel path (per-unit gate)
		{5, 20, 9},    // SIMD path with k%8=4: vector loop + a multi-element scalar tail
	}
	rng := rand.New(rand.NewSource(42))
	for _, sh := range shapes {
		m, k, n := sh[0], sh[1], sh[2]
		// fp32 accumulation error grows with k and |value| — and on arm64
		// the compiler fuses mul+add even in the "naive" kernel, shifting
		// accumulation slightly — so tolerances scale with √k like the
		// packed sweep's.
		tolF := max(1e-5, 1e-6*math.Sqrt(float64(k)))
		a := make([]float32, m*k)
		bT := make([]float32, n*k)
		for i := range a {
			a[i] = rng.Float32()*2 - 1
		}
		for i := range bT {
			bT[i] = rng.Float32()*2 - 1
		}
		want := make([]float32, m*n)
		for i := range m {
			for j := range n {
				var sum float64
				for p := range k {
					sum += float64(a[i*k+p]) * float64(bT[j*k+p])
				}
				want[i*n+j] = float32(sum)
			}
		}
		results := map[string][]float32{}
		for name, kern := range kernels {
			got := make([]float32, m*n)
			kern.fn(got, a, bT, m, k, n)
			results[name] = got
			for i := range want {
				if d := math.Abs(float64(got[i] - want[i])); d > tolF*(1+math.Abs(float64(want[i]))) {
					t.Fatalf("%s %dx%dx%d: [%d]=%v want %v (diff %g)", name, m, k, n, i, got[i], want[i], d)
				}
			}
		}
		// Cross-kernel check against naive: exact kernels share naive's
		// accumulation order, so they are bit-identical where the compiler
		// does not fuse mul+add (amd64); every kernel — the SIMD one
		// included — must stay within tight fp32 tolerance of naive. This
		// is a far stronger net than the float64 tolerance alone.
		bitwise := runtime.GOARCH == "amd64"
		for name, got := range results {
			ref := results["naive"]
			// Reordered-accumulation kernels (SIMD lanes) legitimately
			// differ from naive by more than exact ones can — and the gap
			// grows with √k (on arm64 the naive reference itself is
			// FMA-fused, widening it further).
			tol := 1e-6
			if !kernels[name].exact {
				tol = tolF
			}
			for i := range ref {
				if bitwise && kernels[name].exact && got[i] != ref[i] {
					t.Fatalf("%s %dx%dx%d: [%d]=%v differs from naive %v (expected bit-identical on amd64)", name, m, k, n, i, got[i], ref[i])
				}
				if d := math.Abs(float64(got[i] - ref[i])); d > tol*(1+math.Abs(float64(ref[i]))) {
					t.Fatalf("%s %dx%dx%d: [%d]=%v vs naive %v (diff %g)", name, m, k, n, i, got[i], ref[i], d)
				}
			}
		}
	}
}

func TestSoftmax(t *testing.T) {
	x := []float32{1, 2, 3, 0, 0, 0}
	Softmax(x, 2, 3)
	// Row 0: softmax(1,2,3); row 1: uniform.
	e1, e2, e3 := math.Exp(1), math.Exp(2), math.Exp(3)
	s := e1 + e2 + e3
	want := []float32{float32(e1 / s), float32(e2 / s), float32(e3 / s), 1.0 / 3, 1.0 / 3, 1.0 / 3}
	almostEqual(t, x, want, 1e-6, "softmax")

	// Large values must not overflow (max-subtraction).
	y := []float32{1000, 1000}
	Softmax(y, 1, 2)
	almostEqual(t, y, []float32{0.5, 0.5}, 1e-6, "softmax large")
}

func TestLayerNorm(t *testing.T) {
	// Row [1,2,3]: mean=2, biased var=2/3.
	x := []float32{1, 2, 3}
	gamma := []float32{1, 1, 1}
	beta := []float32{0, 0, 0}
	LayerNorm(x, gamma, beta, 1, 3, 0)
	inv := 1 / math.Sqrt(2.0/3.0)
	want := []float32{float32(-inv), 0, float32(inv)}
	almostEqual(t, x, want, 1e-6, "layernorm")

	// gamma/beta applied after normalization.
	x2 := []float32{1, 2, 3}
	LayerNorm(x2, []float32{2, 2, 2}, []float32{1, 1, 1}, 1, 3, 0)
	want2 := []float32{float32(1 - 2*inv), 1, float32(1 + 2*inv)}
	almostEqual(t, x2, want2, 1e-6, "layernorm affine")
}

func TestGELU(t *testing.T) {
	// Reference in float64 with exact erf; the fast float32 erf is within
	// ~3e-7, so 1e-6 absolute over this range.
	x := make([]float32, 0, 100)
	for v := float32(-6); v <= 6; v += 0.25 {
		x = append(x, v)
	}
	want := make([]float32, len(x))
	for i, v := range x {
		want[i] = float32(0.5 * float64(v) * (1 + math.Erf(float64(v)/math.Sqrt2)))
	}
	GELU(x)
	almostEqual(t, x, want, 1e-6, "gelu")
}

func TestL2Normalize(t *testing.T) {
	x := []float32{3, 4}
	L2Normalize(x)
	almostEqual(t, x, []float32{0.6, 0.8}, 1e-7, "l2")
	z := []float32{0, 0}
	L2Normalize(z) // must not NaN
	almostEqual(t, z, []float32{0, 0}, 0, "l2 zero")
}

func TestAddBiasAndAdd(t *testing.T) {
	x := []float32{1, 2, 3, 4}
	AddBias(x, []float32{10, 20}, 2, 2)
	almostEqual(t, x, []float32{11, 22, 13, 24}, 0, "addbias")
	Add(x, []float32{1, 1, 1, 1})
	almostEqual(t, x, []float32{12, 23, 14, 25}, 0, "add")
}

// benchMatMul runs one kernel over the two shapes that dominate the forward
// pass: the FFN panel at a large seq, and the skinny seq=12 projection that
// the e2e bench text produces.
func benchMatMul(b *testing.B, kern MatMulFunc) {
	// FFN panel at large seq, the seq=12 projection, and the short-query
	// projections that defend the per-unit parallelization gate.
	for _, sh := range [][3]int{{128, 384, 1536}, {12, 384, 384}, {3, 384, 384}, {1, 384, 384}} {
		m, k, n := sh[0], sh[1], sh[2]
		b.Run(fmt.Sprintf("%dx%dx%d", m, k, n), func(b *testing.B) {
			a := make([]float32, m*k)
			bT := make([]float32, n*k)
			dst := make([]float32, m*n)
			for i := range a {
				a[i] = float32(i%7) * 0.1
			}
			for i := range bT {
				bT[i] = float32(i%5) * 0.1
			}
			b.SetBytes(int64(m*k+n*k+m*n) * 4)
			for b.Loop() {
				kern(dst, a, bT, m, k, n)
			}
		})
	}
}

func BenchmarkMatMulNaive(b *testing.B)    { benchMatMul(b, MatMulNaive) }
func BenchmarkMatMulBlocked(b *testing.B)  { benchMatMul(b, MatMulBlocked) }
func BenchmarkMatMulParallel(b *testing.B) { benchMatMul(b, MatMulParallel) }

func BenchmarkMatMulSIMD(b *testing.B) {
	if !hasSIMD {
		b.Skip("no AVX2+FMA on this CPU")
	}
	benchMatMul(b, MatMulSIMD)
}

func TestParallelForCoversEveryUnitOnce(t *testing.T) {
	// Sweep GOMAXPROCS so the atomic-counter path is exercised even when
	// the test host has few cores, including units < workers.
	defer runtime.GOMAXPROCS(runtime.GOMAXPROCS(0))
	for _, procs := range []int{1, 2, 4, 16} {
		runtime.GOMAXPROCS(procs)
		for _, units := range []int{0, 1, 2, 3, 7, 64, 1000} {
			hits := make([]atomic.Int32, units)
			ParallelFor(units, func(u int) { hits[u].Add(1) })
			for u := range hits {
				if got := hits[u].Load(); got != 1 {
					t.Fatalf("procs=%d units=%d: unit %d executed %d times", procs, units, u, got)
				}
			}
		}
	}
}

// TestDefaultCappedMatchesDefault: capping the fan-out must change
// scheduling only, never results — capped kernels share the serial bodies.
func TestDefaultCappedMatchesDefault(t *testing.T) {
	rng := rand.New(rand.NewSource(99))
	const m, k, n = 37, 129, 192
	a := make([]float32, m*k)
	bT := make([]float32, n*k)
	for i := range a {
		a[i] = rng.Float32()*2 - 1
	}
	for i := range bT {
		bT[i] = rng.Float32()*2 - 1
	}
	want := make([]float32, m*n)
	Default()(want, a, bT, m, k, n)
	for _, workers := range []int{1, 2, 5} {
		got := make([]float32, m*n)
		DefaultCapped(workers)(got, a, bT, m, k, n)
		for i := range want {
			if got[i] != want[i] {
				t.Fatalf("workers=%d: [%d]=%v differs from uncapped %v", workers, i, got[i], want[i])
			}
		}
	}
}

func TestParallelForPropagatesPanic(t *testing.T) {
	// A worker panic must surface on the calling goroutine (where the
	// host's recover can handle it), not kill the process.
	defer func() {
		if r := recover(); r != "boom" {
			t.Fatalf("recovered %v, want \"boom\"", r)
		}
	}()
	ParallelFor(64, func(u int) {
		if u == 13 {
			panic("boom")
		}
	})
	t.Fatal("ParallelFor returned instead of panicking")
}
