// SPDX-License-Identifier: Apache-2.0

//go:build amd64

package tensor

import "golang.org/x/sys/cpu"

// hasSIMD gates the native fp32 kernels (AVX2+FMA covers every x86-64-v3
// machine, Haswell 2013 onward); hasSIMD8 gates the int8 gemm. Checked
// once at init.
var (
	hasSIMD  = cpu.X86.HasAVX2 && cpu.X86.HasFMA
	hasSIMD8 = hasSIMD
	// hasVNNI gates the u8·s8 VPDPBUSD kernel. AVX-VNNI is the 256-bit
	// VEX form: Alder Lake (2021) onward on client, Sapphire Rapids
	// onward on server, Zen 5 onward on AMD. NOTE the trap the review
	// caught: AVX-512-VNNI parts (Cascade/Ice Lake-SP, Zen 4) do NOT
	// report AVX-VNNI, and our VEX encoding #UDs there — those CPUs
	// would need the EVEX form Go emits natively, an untested path this
	// codebase refuses to ship until hardware is available (R8's box).
	hasVNNI = hasSIMD && cpu.X86.HasAVXVNNI
)

// dot4 computes dst[0..3] = dot(a, b0..b3) over k floats with 8-lane FMA
// accumulators (implemented in dot_amd64.s). The horizontal reduction at the
// end means its accumulation order differs from the scalar kernels: results
// are within fp32 rounding, not bit-identical.
//
//go:noescape
func dot4(dst, a, b0, b1, b2, b3 *float32, k int)

// gemm4x16 writes C[4×16] = packed-A-panel × packed-B-panel to dst with row
// stride n floats (implemented in gemm_amd64.s). k must be >= 1.
//
//go:noescape
func gemm4x16(dst *float32, n int, pa, pb *float32, k int)

// gemm4x16i8 is gemm4x16 over an int8 B panel with per-column dequant
// scales applied in the epilogue (implemented in gemm8_amd64.s). k >= 1.
//
//go:noescape
func gemm4x16i8(dst *float32, n int, pa *float32, pb *int8, k int, scales *float32)

// gemm4x16vnni accumulates dst[4×16] int32 += u8(qa)·s8(pb) over kg
// groups of 4 k-values (implemented in vnni_amd64.s; requires AVX-VNNI).
//
//go:noescape
func gemm4x16vnni(dst *int32, n int, qa *uint8, aStride int, pb *int8, kg int)
