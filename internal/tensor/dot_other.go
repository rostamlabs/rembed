// SPDX-License-Identifier: Apache-2.0

//go:build !amd64

package tensor

// hasSIMD is false off amd64; Default() falls back to MatMulParallel.
const hasSIMD = false

// dot4AVX2 is never called when hasSIMD is false; the stub keeps the
// SIMD kernel compiling on every platform.
func dot4AVX2(dst, a, b0, b1, b2, b3 *float32, k int) {
	panic("tensor: dot4AVX2 called on non-amd64")
}
