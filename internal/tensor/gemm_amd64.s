// SPDX-License-Identifier: Apache-2.0
//go:build amd64

#include "textflag.h"

// func gemm4x16(dst *float32, n int, pa, pb *float32, k int)
//
// C[4×16] = A-panel × B-panel, written (not accumulated) to dst with row
// stride n floats. pa is a packed A panel (k-major, 4 rows interleaved:
// pa[p*4+r]), pb a packed B panel (k-major, 16 columns interleaved:
// pb[p*16+c]). Per k step: 2 B loads + 4 A broadcasts feed 8 FMAs into 8
// independent ymm accumulator chains — exactly saturating two FMA ports at
// 4-cycle latency. k must be >= 1 (Go side guarantees it).
TEXT ·gemm4x16(SB), NOSPLIT, $0-40
	MOVQ dst+0(FP), DI
	MOVQ n+8(FP), DX
	MOVQ pa+16(FP), SI
	MOVQ pb+24(FP), BX
	MOVQ k+32(FP), CX
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
	VMOVUPS      (BX), Y8
	VMOVUPS      32(BX), Y9
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
	ADDQ         $64, BX
	DECQ         CX
	JNZ          loop

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
