// SPDX-License-Identifier: Apache-2.0

//go:build amd64

package tensor

// geluAVX2 applies erf-GELU in place to n floats (n a multiple of 8),
// implemented in gelu_amd64.s. It reproduces geluScalar's polynomial, so
// results track it within fp32 rounding.
//
//go:noescape
func geluAVX2(x *float32, n int)

// GELU vectorizes the erf-based GELU with AVX2 where available, handling the
// sub-8 tail (and non-AVX2 CPUs) with the scalar path.
func GELU(x []float32) {
	if hasSIMD && len(x) >= 8 {
		m := len(x) &^ 7
		geluAVX2(&x[0], m)
		if m < len(x) {
			geluScalar(x[m:])
		}
		return
	}
	geluScalar(x)
}
