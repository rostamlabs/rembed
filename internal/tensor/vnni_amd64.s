// SPDX-License-Identifier: Apache-2.0

//go:build amd64

#include "textflag.h"

// func gemm4x16vnni(dst *int32, n int, qa *uint8, aStride int, pb *int8, kg int)
//
// 4×16 int8 micro-kernel on AVX-VNNI: dst[4][16] (row stride n int32s)
// accumulates Σ_k u8(activation)·s8(weight) as int32 over kg groups of 4
// k-values. qa rows are VNNI-packed activations (aStride bytes apart); pb
// is the VNNI-packed B panel (per k-group: 16 columns × 4 bytes).
//
// Go's assembler only emits the EVEX (AVX-512-VNNI) encoding for
// VPDPBUSD, which FAULTS on AVX-VNNI-only CPUs (all consumer Alder/
// Raptor Lake) — so the eight VPDPBUSD are hand-encoded in their VEX
// (AVX-VNNI) form. Each encoding was generated field-by-field
// (VEX.256.66.0F38.W0 50 /r) and verified byte-for-byte against GNU as's
// {vex}-prefixed output.
TEXT ·gemm4x16vnni(SB), NOSPLIT, $0-48
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
	VMOVDQU (BX), Y8    // B panel cols 0-7 (8 dwords × 4 k-bytes)
	VMOVDQU 32(BX), Y9  // B panel cols 8-15
	ADDQ    $64, BX
	VPBROADCASTD (R8), Y10 // row 0's k-group, broadcast to all lanes
	VPBROADCASTD (R9), Y11
	VPBROADCASTD (R10), Y12
	VPBROADCASTD (R11), Y13
	ADDQ $4, R8
	ADDQ $4, R9
	ADDQ $4, R10
	ADDQ $4, R11

	BYTE $0xc4; BYTE $0xc2; BYTE $0x2d; BYTE $0x50; BYTE $0xc0 // VPDPBUSD Y8, Y10, Y0
	BYTE $0xc4; BYTE $0xc2; BYTE $0x2d; BYTE $0x50; BYTE $0xc9 // VPDPBUSD Y9, Y10, Y1
	BYTE $0xc4; BYTE $0xc2; BYTE $0x25; BYTE $0x50; BYTE $0xd0 // VPDPBUSD Y8, Y11, Y2
	BYTE $0xc4; BYTE $0xc2; BYTE $0x25; BYTE $0x50; BYTE $0xd9 // VPDPBUSD Y9, Y11, Y3
	BYTE $0xc4; BYTE $0xc2; BYTE $0x1d; BYTE $0x50; BYTE $0xe0 // VPDPBUSD Y8, Y12, Y4
	BYTE $0xc4; BYTE $0xc2; BYTE $0x1d; BYTE $0x50; BYTE $0xe9 // VPDPBUSD Y9, Y12, Y5
	BYTE $0xc4; BYTE $0xc2; BYTE $0x15; BYTE $0x50; BYTE $0xf0 // VPDPBUSD Y8, Y13, Y6
	BYTE $0xc4; BYTE $0xc2; BYTE $0x15; BYTE $0x50; BYTE $0xf9 // VPDPBUSD Y9, Y13, Y7

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
