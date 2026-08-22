// SPDX-License-Identifier: Apache-2.0
//go:build amd64

#include "textflag.h"

// func gemm4x32(dst *float32, n int, pa, pb *float32, k int)
//
// The AVX-512 twin of gemm4x16, doubled in width: C[4×32] = A-panel ×
// B-panel, written (not accumulated) to dst with row stride n floats.
// pa is the SAME packed A panel gemm4x16 uses (k-major, 4 rows
// interleaved); pb is a 32-column panel (pb[p*32+c]). Per k step: 2 zmm
// B loads + 4 A broadcasts feed 8 FMAs into 8 independent zmm chains.
// Per-element accumulation order is IDENTICAL to gemm4x16 (one
// sequential FMA chain per output element), so a 32-panel result is
// bit-identical to the 16-panel result for the same data — pinned by
// TestPackedPanelWidthsAgree on AVX-512 hardware.
//
// Honest throughput note: on Zen 4 the 512-bit FMAs are double-pumped
// through 256-bit ports (same peak FLOPs as gemm4x16); the win there is
// frontend pressure — half the instructions per k step. On Intel server
// parts with two native 512-bit FMA units it is a real 2× peak.
TEXT ·gemm4x32(SB), NOSPLIT, $0-40
	MOVQ dst+0(FP), DI
	MOVQ n+8(FP), DX
	MOVQ pa+16(FP), SI
	MOVQ pb+24(FP), BX
	MOVQ k+32(FP), CX
	SHLQ $2, DX // row stride in bytes

	VXORPS Z0, Z0, Z0 // row 0, cols 0-15
	VXORPS Z1, Z1, Z1 // row 0, cols 16-31
	VXORPS Z2, Z2, Z2 // row 1
	VXORPS Z3, Z3, Z3
	VXORPS Z4, Z4, Z4 // row 2
	VXORPS Z5, Z5, Z5
	VXORPS Z6, Z6, Z6 // row 3
	VXORPS Z7, Z7, Z7

loop:
	VMOVUPS      (BX), Z8
	VMOVUPS      64(BX), Z9
	VBROADCASTSS (SI), Z10
	VBROADCASTSS 4(SI), Z11
	VBROADCASTSS 8(SI), Z12
	VBROADCASTSS 12(SI), Z13
	VFMADD231PS  Z8, Z10, Z0
	VFMADD231PS  Z9, Z10, Z1
	VFMADD231PS  Z8, Z11, Z2
	VFMADD231PS  Z9, Z11, Z3
	VFMADD231PS  Z8, Z12, Z4
	VFMADD231PS  Z9, Z12, Z5
	VFMADD231PS  Z8, Z13, Z6
	VFMADD231PS  Z9, Z13, Z7
	ADDQ         $16, SI
	ADDQ         $128, BX
	DECQ         CX
	JNZ          loop

	VMOVUPS Z0, (DI)
	VMOVUPS Z1, 64(DI)
	ADDQ    DX, DI
	VMOVUPS Z2, (DI)
	VMOVUPS Z3, 64(DI)
	ADDQ    DX, DI
	VMOVUPS Z4, (DI)
	VMOVUPS Z5, 64(DI)
	ADDQ    DX, DI
	VMOVUPS Z6, (DI)
	VMOVUPS Z7, 64(DI)

	VZEROUPPER
	RET
