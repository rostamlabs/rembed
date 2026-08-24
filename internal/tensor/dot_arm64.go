// SPDX-License-Identifier: Apache-2.0

//go:build arm64

package tensor

// NEON is baseline on arm64, so the native kernels are unconditional.
// hasSIMD8 covers the int8 gemm, whose widening/convert instructions are
// WORD-encoded (the Go assembler lacks SXTL/SCVTF mnemonics) and verified
// under qemu and CI like everything else.
const (
	hasVNNI    = false
	hasVNNI512 = false
	hasAVX512  = false // x86-only; NEON int8 uses the widening pipeline
	hasSIMD    = true
	hasSIMD8   = true
	has6x16    = false // the wider fp32 micro-kernel is an amd64/AVX2 tuning
)

// dot4 computes dst[0..3] = dot(a, b0..b3) over k floats
// (implemented in dot_arm64.s). Vector-lane accumulation then a scalar
// reduction: results are within fp32 rounding of the scalar kernels, not
// bit-identical — same contract as the amd64 kernel.
//
//go:noescape
func dot4(dst, a, b0, b1, b2, b3 *float32, k int)

// gemm4x16 writes C[4×16] = packed-A-panel × packed-B-panel to dst with
// row stride n floats (implemented in gemm_arm64.s). k must be >= 1.
//
//go:noescape
func gemm4x16(dst *float32, n int, pa, pb *float32, k int)

// gemm4x16i8 is gemm4x16 over an int8 B panel with per-column dequant
// scales applied in the epilogue (implemented in gemm8_arm64.s). k >= 1.
//
//go:noescape
func gemm4x16i8(dst *float32, n int, pa *float32, pb *int8, k int, scales *float32)

func gemm6x16(dst *float32, n int, pa, pb *float32, k int) {
	panic("tensor: gemm6x16 is an amd64-only tuning (has6x16 is false here)")
}

func gemm4x16vnni(dst *int32, n int, qa *uint8, aStride int, pb *int8, kg int) {
	panic("tensor: gemm4x16vnni is amd64-only")
}

func gemm4x16vnni512(dst *int32, n int, qa *uint8, aStride int, pb *int8, kg int) {
	panic("tensor: gemm4x16vnni512 is amd64-only")
}

func gemm4x32(dst *float32, n int, pa, pb *float32, k int) {
	panic("tensor: gemm4x32 is amd64-only")
}
