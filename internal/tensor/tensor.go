// SPDX-License-Identifier: Apache-2.0

// Package tensor holds the math kernels for the transformer forward pass.
// This is the perf-critical core: the optimization ladder (naive →
// cache-blocked → SIMD → int8) swaps kernel bodies behind the signatures in
// this package without touching model code.
package tensor

import "math"

// MatMulFunc computes C[m×n] = A[m×k] · Bᵀ, where bT is the [n×k] row-major
// transpose of B. This layout is chosen deliberately: HuggingFace stores
// nn.Linear weights as [out, in] = [n, k], so linear layers pass weights
// straight through with no transpose, and every dot product reads two
// contiguous rows.
type MatMulFunc func(dst, a, bT []float32, m, k, n int)

// Default returns the best matmul body for this platform. Model code binds
// it ONCE at load time (a mutable package variable would race with in-flight
// forward passes); benchmarks A/B the named implementations directly.
func Default() MatMulFunc { return MatMulBlocked }

// MatMulNaive is the M0 reference body: three plain loops, dot products of
// contiguous rows.
func MatMulNaive(dst, a, bT []float32, m, k, n int) {
	for i := range m {
		ar := a[i*k : i*k+k]
		for j := range n {
			br := bT[j*k : j*k+k]
			var sum float32
			for p := range k {
				sum += ar[p] * br[p]
			}
			dst[i*n+j] = sum
		}
	}
}

// Blocking parameters for MatMulBlocked.
const (
	// blockM: A-panel rows per tile. 64 rows × 512 floats × 4 B ≈ 128 KB
	// worst-case (k=I=1536 gives 384 KB — still L2-resident), so the panel
	// is reused from cache across every bT row of the tile.
	blockM = 64
	// blockN: bT rows per micro-kernel pass. 4 independent accumulator
	// chains give the FP units ILP, and each loaded a[p] is reused 4×.
	blockN = 4
)

// MatMulBlocked is the M1 body: cache-blocked with a 1×4 register
// micro-kernel. The naive body streams ALL of bT from memory for every row
// of A (m × n×k floats); here the j-loop is outermost within an A-tile, so
// bT streams through cache once per tile of blockM rows and the A panel
// stays resident. Four independent accumulator chains give the FP units
// ILP; a 2×4 variant measured ~7% SLOWER here (register spills — Go keeps
// only so many of 8 accumulators + 6 temps live), so 1×4 it is until M3's
// SIMD kernel. Accumulation order over k matches naive, so results are
// bit-identical wherever the compiler does not fuse mul+add (amd64 today),
// and within fp32 rounding otherwise.
func MatMulBlocked(dst, a, bT []float32, m, k, n int) {
	if m*n > 0 {
		// Fail fast on an undersized dst: the 4-wide stores below slice with
		// an explicit upper bound, which bounds-checks against cap — on a
		// pooled buffer with cap > len an undersized dst would otherwise be
		// corrupted silently instead of panicking like the naive body does.
		_ = dst[m*n-1]
	}
	for i0 := 0; i0 < m; i0 += blockM {
		i1 := min(i0+blockM, m)
		var j int
		for j = 0; j+blockN <= n; j += blockN {
			b0 := bT[(j+0)*k : (j+0)*k+k]
			b1 := bT[(j+1)*k : (j+1)*k+k]
			b2 := bT[(j+2)*k : (j+2)*k+k]
			b3 := bT[(j+3)*k : (j+3)*k+k]
			for i := i0; i < i1; i++ {
				ar := a[i*k : i*k+k]
				var s0, s1, s2, s3 float32
				for p, av := range ar {
					s0 += av * b0[p]
					s1 += av * b1[p]
					s2 += av * b2[p]
					s3 += av * b3[p]
				}
				d := dst[i*n+j : i*n+j+4]
				d[0], d[1], d[2], d[3] = s0, s1, s2, s3
			}
		}
		for ; j < n; j++ { // n % blockN remainder columns
			br := bT[j*k : j*k+k]
			for i := i0; i < i1; i++ {
				ar := a[i*k : i*k+k]
				var s float32
				for p, av := range ar {
					s += av * br[p]
				}
				dst[i*n+j] = s
			}
		}
	}
}

// AddBias adds bias (length n) to each of the m rows of x.
func AddBias(x, bias []float32, m, n int) {
	for i := range m {
		row := x[i*n : i*n+n]
		for j := range row {
			row[j] += bias[j]
		}
	}
}

// Add accumulates src into dst element-wise (residual connections).
func Add(dst, src []float32) {
	for i := range dst {
		dst[i] += src[i]
	}
}

// Softmax normalizes each of the m rows of x (length n) in place, with
// max-subtraction for numerical stability.
func Softmax(x []float32, m, n int) {
	for i := range m {
		row := x[i*n : i*n+n]
		maxv := row[0]
		for _, v := range row[1:] {
			if v > maxv {
				maxv = v
			}
		}
		var sum float32
		for j, v := range row {
			e := float32(math.Exp(float64(v - maxv)))
			row[j] = e
			sum += e
		}
		inv := 1 / sum
		for j := range row {
			row[j] *= inv
		}
	}
}

// LayerNorm normalizes each of the m rows of x (length n) in place:
// (x-mean)/sqrt(variance+eps) * gamma + beta, with biased variance, matching
// torch.nn.LayerNorm.
func LayerNorm(x, gamma, beta []float32, m, n int, eps float32) {
	for i := range m {
		row := x[i*n : i*n+n]
		var mean float32
		for _, v := range row {
			mean += v
		}
		mean /= float32(n)
		var variance float32
		for _, v := range row {
			d := v - mean
			variance += d * d
		}
		variance /= float32(n)
		inv := 1 / float32(math.Sqrt(float64(variance+eps)))
		for j, v := range row {
			row[j] = (v-mean)*inv*gamma[j] + beta[j]
		}
	}
}

// GELU applies the exact (erf-based) Gaussian Error Linear Unit in place,
// matching HuggingFace's "gelu" activation: 0.5x(1+erf(x/√2)).
func GELU(x []float32) {
	for i, v := range x {
		x[i] = float32(0.5 * float64(v) * (1 + math.Erf(float64(v)/math.Sqrt2)))
	}
}

// L2Normalize scales x to unit Euclidean norm in place. A zero vector is
// left unchanged.
func L2Normalize(x []float32) {
	var sum float64
	for _, v := range x {
		sum += float64(v) * float64(v)
	}
	if sum == 0 {
		return
	}
	inv := float32(1 / math.Sqrt(sum))
	for i := range x {
		x[i] *= inv
	}
}
