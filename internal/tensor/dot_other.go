// SPDX-License-Identifier: Apache-2.0

//go:build !amd64 && !arm64

package tensor

// No native kernels on this architecture; Default() falls back to the
// scalar parallel body.
const (
	hasSIMD  = false
	hasSIMD8 = false
)

// dot4 and gemm4x16 are never called when hasSIMD is false; the stubs
// keep the SIMD kernels compiling on every platform.
func dot4(dst, a, b0, b1, b2, b3 *float32, k int) {
	panic("tensor: dot4 called on non-amd64")
}

func gemm4x16(dst *float32, n int, pa, pb *float32, k int) {
	panic("tensor: gemm4x16 called on non-amd64")
}

func gemm4x16i8(dst *float32, n int, pa *float32, pb *int8, k int, scales *float32) {
	panic("tensor: gemm4x16i8 called on non-amd64")
}
