// SPDX-License-Identifier: Apache-2.0

package tensor

import "math"

// Float32 exp/erf used by Softmax and GELU. The stdlib routes through
// float64 libm calls (math.Exp ~20 ns, math.Erf ~25 ns each); at seq×I
// GELU calls and heads×seq² softmax calls per layer they were ~30% of the
// whole forward pass. These float32 versions are ~4-6× faster. Accuracy:
// exp is ~2e-7 relative near zero, degrading to ~4e-6 at |x|≈85 (the
// two-part float32 range reduction cancels two ~|x|-magnitude values, so
// the error grows ~ulp(|x|)/2); erf is ~4e-7 absolute. All three orders
// of magnitude inside the 1e-4 golden tolerance, and pinned at exactly
// these bounds against the float64 stdlib in fastmath_test.go.

const (
	expLog2e = 1.44269504088896341
	expC1    = 0.693359375    // ln2 high part
	expC2    = -2.12194440e-4 // ln2 low part
)

// expf32 computes e^x for float32 (Cephes expf: range reduction by ln2 in
// two parts, degree-5 polynomial, exponent reassembled through the float32
// bit layout). See the accuracy note above.
func expf32(x float32) float32 {
	if x > 88.72 {
		return float32(math.Inf(1))
	}
	if x < -87.33 {
		return 0
	}
	// n = round(x / ln2)
	nf := float32(int32(x*expLog2e + sign05(x)))
	x -= nf * expC1
	x -= nf * expC2

	z := x * x
	y := float32(1.9875691500e-4)
	y = y*x + 1.3981999507e-3
	y = y*x + 8.3334519073e-3
	y = y*x + 4.1665795894e-2
	y = y*x + 1.6666665459e-1
	y = y*x + 5.0000001201e-1
	y = y*z + x + 1

	// y * 2^n via the exponent bits. n can reach 128 for x in
	// (~88.03, 88.72] where the result is still a finite float32 —
	// (128+127)<<23 is the +Inf bit pattern, so split one power of two off.
	n := int32(nf)
	if n >= 128 {
		return y * math.Float32frombits(uint32(127+127)<<23) * 2
	}
	return y * math.Float32frombits(uint32(n+127)<<23)
}

func sign05(x float32) float32 {
	if x < 0 {
		return -0.5
	}
	return 0.5
}

// erff32 computes erf(x) for float32 with ~3e-7 absolute error
// (Abramowitz & Stegun 7.1.26 on |x|, sign restored by symmetry).
func erff32(x float32) float32 {
	neg := x < 0
	if neg {
		x = -x
	}
	t := 1 / (1 + 0.3275911*x)
	poly := t * (0.254829592 + t*(-0.284496736+t*(1.421413741+t*(-1.453152027+t*1.061405429))))
	r := 1 - poly*expf32(-x*x)
	if neg {
		return -r
	}
	return r
}

// expNonPos is expf32 specialized for x <= 0 and flattened branch-free so
// the compiler inlines it into hot loops (Softmax rows after
// max-subtraction, GELU's exp(-a²)): with no calls in the loop body the
// out-of-order core overlaps consecutive elements' latency chains, which
// measured ~3× faster than the called version per element. Accuracy
// follows expf32 (see the note above: ~2e-7 near zero, ~4e-6 at |x|≈85).
// Inputs below the underflow clamp return exp(-87.33) ≈ 1.2e-38 rather
// than 0 — an absolute error < 1.3e-38, far below the golden tolerance.
func expNonPos(x float32) float32 {
	if x < -87.33 {
		x = -87.33
	}
	nf := float32(int32(x*expLog2e - 0.5))
	x -= nf * expC1
	x -= nf * expC2
	z := x * x
	y := float32(1.9875691500e-4)
	y = y*x + 1.3981999507e-3
	y = y*x + 8.3334519073e-3
	y = y*x + 4.1665795894e-2
	y = y*x + 1.6666665459e-1
	y = y*x + 5.0000001201e-1
	y = y*z + x + 1
	return y * math.Float32frombits(uint32(int32(nf)+127)<<23)
}
