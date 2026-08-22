// SPDX-License-Identifier: Apache-2.0

//go:build !amd64

package tensor

// hasSIMD is false off amd64; Default() falls back to MatMulParallel.
const hasSIMD = false

// dot4AVX2 and gemm4x16 are never called when hasSIMD is false; the stubs
// keep the SIMD kernels compiling on every platform.
func dot4AVX2(dst, a, b0, b1, b2, b3 *float32, k int) {
	panic("tensor: dot4AVX2 called on non-amd64")
}

func gemm4x16(dst *float32, n int, pa, pb *float32, k int) {
	panic("tensor: gemm4x16 called on non-amd64")
}
