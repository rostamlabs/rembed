// SPDX-License-Identifier: Apache-2.0

//go:build amd64

#include "textflag.h"

// func gemm4x16vnni512(dst *int32, n int, qa *uint8, aStride int, pb *int8, kg int)
//
// The EVEX twin of gemm4x16vnni for AVX-512-VNNI parts that lack
// AVX-VNNI (Cascade/Ice Lake-SP, Zen 4): identical structure, but the
// VPDPBUSD here are Go's native mnemonics, which emit the EVEX encoding
// — legal on these CPUs (with AVX512VL for the ymm form) and #UD on
// AVX-VNNI-only clients, exactly mirroring how the VEX kernel #UDs
// here. hasVNNI512 gates it; first proven on real Zen 4 hardware
// (the R8 benchmark box).
TEXT ·gemm4x16vnni512(SB), NOSPLIT, $0-48
	MOVQ dst+0(FP), DI
	MOVQ n+8(FP), DX
	MOVQ qa+16(FP), R8
	MOVQ aStride+24(FP), R12
	MOVQ pb+32(FP), BX
	MOVQ kg+40(FP), CX

	SHLQ $2, DX // row stride in bytes

	LEAQ (R8)(R12*1), R9
	LEAQ (R9)(R12*1), R10
	LEAQ (R10)(R12*1), R11

	VPXOR Y0, Y0, Y0
	VPXOR Y1, Y1, Y1
	VPXOR Y2, Y2, Y2
	VPXOR Y3, Y3, Y3
	VPXOR Y4, Y4, Y4
	VPXOR Y5, Y5, Y5
	VPXOR Y6, Y6, Y6
	VPXOR Y7, Y7, Y7

loop:
	VMOVDQU (BX), Y8
	VMOVDQU 32(BX), Y9
	ADDQ    $64, BX
	VPBROADCASTD (R8), Y10
	VPBROADCASTD (R9), Y11
	VPBROADCASTD (R10), Y12
	VPBROADCASTD (R11), Y13
	ADDQ $4, R8
	ADDQ $4, R9
	ADDQ $4, R10
	ADDQ $4, R11

	VPDPBUSD Y8, Y10, Y0
	VPDPBUSD Y9, Y10, Y1
	VPDPBUSD Y8, Y11, Y2
	VPDPBUSD Y9, Y11, Y3
	VPDPBUSD Y8, Y12, Y4
	VPDPBUSD Y9, Y12, Y5
	VPDPBUSD Y8, Y13, Y6
	VPDPBUSD Y9, Y13, Y7

	DECQ CX
	JNZ  loop

	VMOVDQU Y0, (DI)
	VMOVDQU Y1, 32(DI)
	ADDQ    DX, DI
	VMOVDQU Y2, (DI)
	VMOVDQU Y3, 32(DI)
	ADDQ    DX, DI
	VMOVDQU Y4, (DI)
	VMOVDQU Y5, 32(DI)
	ADDQ    DX, DI
	VMOVDQU Y6, (DI)
	VMOVDQU Y7, 32(DI)
	VZEROUPPER
	RET
