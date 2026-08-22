// SPDX-License-Identifier: Apache-2.0
//go:build amd64

#include "textflag.h"

// func gemm4x16i8(dst *float32, n int, pa *float32, pb *int8, k int, scales *float32)
//
// Like gemm4x16, but the B panel is int8 (weight-only quantization): each
// k step loads 16 BYTES of B instead of 64, sign-extends to two 8-lane
// dword vectors and converts to float32 in-register, then runs the same
// broadcast-FMA scheme. The per-output-channel dequant scales are applied
// once to the accumulators in the epilogue — 4× less weight traffic for
// two extra port-0/1 ops per step. k must be >= 1.
TEXT ·gemm4x16i8(SB), NOSPLIT, $0-48
	MOVQ dst+0(FP), DI
	MOVQ n+8(FP), DX
	MOVQ pa+16(FP), SI
	MOVQ pb+24(FP), BX
	MOVQ k+32(FP), CX
	MOVQ scales+40(FP), R8
	SHLQ $2, DX // row stride in bytes

	VXORPS Y0, Y0, Y0 // row 0, cols 0-7
	VXORPS Y1, Y1, Y1 // row 0, cols 8-15
	VXORPS Y2, Y2, Y2 // row 1
	VXORPS Y3, Y3, Y3
	VXORPS Y4, Y4, Y4 // row 2
	VXORPS Y5, Y5, Y5
	VXORPS Y6, Y6, Y6 // row 3
	VXORPS Y7, Y7, Y7

loop:
	VPMOVSXBD    (BX), Y8  // cols 0-7: 8 int8 -> 8 int32
	VPMOVSXBD    8(BX), Y9 // cols 8-15
	VCVTDQ2PS    Y8, Y8
	VCVTDQ2PS    Y9, Y9
	VBROADCASTSS (SI), Y10
	VBROADCASTSS 4(SI), Y11
	VBROADCASTSS 8(SI), Y12
	VBROADCASTSS 12(SI), Y13
	VFMADD231PS  Y8, Y10, Y0
	VFMADD231PS  Y9, Y10, Y1
	VFMADD231PS  Y8, Y11, Y2
	VFMADD231PS  Y9, Y11, Y3
	VFMADD231PS  Y8, Y12, Y4
	VFMADD231PS  Y9, Y12, Y5
	VFMADD231PS  Y8, Y13, Y6
	VFMADD231PS  Y9, Y13, Y7
	ADDQ         $16, SI
	ADDQ         $16, BX
	DECQ         CX
	JNZ          loop

	// Dequantize: per-column scales, once per tile.
	VMOVUPS (R8), Y8
	VMOVUPS 32(R8), Y9
	VMULPS  Y8, Y0, Y0
	VMULPS  Y9, Y1, Y1
	VMULPS  Y8, Y2, Y2
	VMULPS  Y9, Y3, Y3
	VMULPS  Y8, Y4, Y4
	VMULPS  Y9, Y5, Y5
	VMULPS  Y8, Y6, Y6
	VMULPS  Y9, Y7, Y7

	VMOVUPS Y0, (DI)
	VMOVUPS Y1, 32(DI)
	ADDQ    DX, DI
	VMOVUPS Y2, (DI)
	VMOVUPS Y3, 32(DI)
	ADDQ    DX, DI
	VMOVUPS Y4, (DI)
	VMOVUPS Y5, 32(DI)
	ADDQ    DX, DI
	VMOVUPS Y6, (DI)
	VMOVUPS Y7, 32(DI)

	VZEROUPPER
	RET
