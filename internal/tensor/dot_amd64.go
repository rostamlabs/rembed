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
	// hasVNNI gates the VEX-encoded u8·s8 VPDPBUSD kernel. AVX-VNNI is
	// the 256-bit VEX form: Alder Lake (2021) onward on client, Sapphire
	// Rapids onward on server, Zen 5 onward on AMD. AVX-512-VNNI parts
	// (Cascade/Ice Lake-SP, Zen 4) do NOT report it — our VEX encoding
	// #UDs there — so they take the EVEX twin below, proven on R8's
	// Zen 4 box.
	hasVNNI = hasSIMD && cpu.X86.HasAVXVNNI
	// hasVNNI512 covers the complementary set: AVX-512-VNNI parts
	// (Cascade/Ice Lake-SP, Zen 4) where Go's native EVEX VPDPBUSD is
	// legal (AVX512VL needed for the ymm form). Together the two gates
	// give full int8 on every VNNI-capable CPU whose OS saves AVX-512
	// state (x/sys forces that false on macOS, so Intel Macs stay on
	// weight-only int8 — correct, just not "every"), each with the one
	// encoding its hardware accepts.
	hasVNNI512 = hasSIMD && cpu.X86.HasAVX512VNNI && cpu.X86.HasAVX512VL
	// hasAVX512 gates the zmm fp32 gemm (AVX512F covers 512-bit fp32
	// FMA; x/sys folds OS state support into the flag). Server parts and
	// Zen 4+ have it; consumer Alder/Raptor Lake do not.
	hasAVX512 = hasSIMD && cpu.X86.HasAVX512F
	// has6x16 gates the wider AVX2 fp32 micro-kernel (6 rows × 16 cols, 12
	// ymm accumulators — better B-reuse and loop-overhead amortization than
	// the 8-accumulator 4×16). Used only for the 16-wide panel path (AVX2
	// without AVX-512; AVX-512 boxes pack 32-wide and take gemm4x32).
	has6x16 = hasSIMD
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

// gemm6x16 writes C[6×16] = packed-A-panel (6 rows) × packed-B-panel to dst
// with row stride n floats (implemented in gemm_amd64.s). Per-element
// accumulation order is identical to gemm4x16, so results are bit-identical;
// only the tile shape differs. k must be >= 1.
//
//go:noescape
func gemm6x16(dst *float32, n int, pa, pb *float32, k int)

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

// gemm4x16vnni512 is gemm4x16vnni with the EVEX (AVX-512-VNNI) encoding
// (implemented in vnni512_amd64.s; requires AVX512-VNNI + AVX512VL).
//
//go:noescape
func gemm4x16vnni512(dst *int32, n int, qa *uint8, aStride int, pb *int8, kg int)

// gemm4x32 is gemm4x16 doubled to zmm width over a 32-column B panel
// (implemented in gemm512_amd64.s; requires AVX-512F).
//
//go:noescape
func gemm4x32(dst *float32, n int, pa, pb *float32, k int)
