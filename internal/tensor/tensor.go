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

// MatMul is the active matmul kernel. Model code calls only this; ladder
// rungs swap the body (and benchmarks A/B the named implementations below).
var MatMul MatMulFunc = MatMulNaive

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
