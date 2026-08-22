// SPDX-License-Identifier: Apache-2.0

package tensor

import (
	"math"
	"math/rand"
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

// TestMatMulAgainstFloat64Reference cross-checks the active kernel against a
// float64 reference on random matrices. Every future kernel body must pass
// this same test — it is the per-kernel safety net of the optimization ladder.
func TestMatMulAgainstFloat64Reference(t *testing.T) {
	rng := rand.New(rand.NewSource(42))
	const m, k, n = 17, 33, 29
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
	got := make([]float32, m*n)
	MatMul(got, a, bT, m, k, n)
	almostEqual(t, got, want, 1e-5, "matmul vs float64")
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
	x := []float32{0, 1, -1, 2}
	GELU(x)
	ref := func(v float64) float32 { return float32(0.5 * v * (1 + math.Erf(v/math.Sqrt2))) }
	want := []float32{0, ref(1), ref(-1), ref(2)}
	almostEqual(t, x, want, 1e-7, "gelu")
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

func BenchmarkMatMulNaive(b *testing.B) {
	// Representative FFN shape for MiniLM at seq=128: [128×384]·[384×1536].
	const m, k, n = 128, 384, 1536
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
		MatMulNaive(dst, a, bT, m, k, n)
	}
}
