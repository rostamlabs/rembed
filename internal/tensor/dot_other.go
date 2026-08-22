// SPDX-License-Identifier: Apache-2.0

//go:build !amd64 && !arm64

package tensor

// No native kernels on this architecture; Default() falls back to the
// scalar parallel body.
const (
	hasSIMD    = false
	hasSIMD8   = false
	hasVNNI    = false
	hasVNNI512 = false
	hasAVX512  = false
)

// The native kernels are never called when hasSIMD/hasSIMD8 are false;
// the stubs keep the callers compiling on every platform.
func dot4(dst, a, b0, b1, b2, b3 *float32, k int) {
	panic("tensor: dot4 called without a native kernel for this architecture")
}

func gemm4x16(dst *float32, n int, pa, pb *float32, k int) {
	panic("tensor: gemm4x16 called without a native kernel for this architecture")
}

func gemm4x16i8(dst *float32, n int, pa *float32, pb *int8, k int, scales *float32) {
	panic("tensor: gemm4x16i8 called without a native kernel for this architecture")
}

func gemm4x16vnni(dst *int32, n int, qa *uint8, aStride int, pb *int8, kg int) {
	panic("tensor: gemm4x16vnni called without a native kernel for this architecture")
}

func gemm4x16vnni512(dst *int32, n int, qa *uint8, aStride int, pb *int8, kg int) {
	panic("tensor: gemm4x16vnni512 is amd64-only")
}

func gemm4x32(dst *float32, n int, pa, pb *float32, k int) {
	panic("tensor: gemm4x32 is amd64-only")
}
