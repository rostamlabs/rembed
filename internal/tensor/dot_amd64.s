// SPDX-License-Identifier: Apache-2.0
//go:build amd64

#include "textflag.h"

// func dot4(dst, a, b0, b1, b2, b3 *float32, k int)
//
// dst[0..3] = dot(a, b0..b3) over k floats, using 8-lane FMA accumulators.
// The four dots share every load of a, so the inner loop is 1 load + 4
// memory-operand FMAs per 8 floats. Lane sums are reduced horizontally at
// the end, so the accumulation order differs from the scalar kernels: the
// result is within fp32 rounding of them, not bit-identical.
TEXT ·dot4(SB), NOSPLIT, $0-56
	MOVQ dst+0(FP), DI
	MOVQ a+8(FP), SI
	MOVQ b0+16(FP), R8
	MOVQ b1+24(FP), R9
	MOVQ b2+32(FP), R10
	MOVQ b3+40(FP), R11
	MOVQ k+48(FP), CX

	VXORPS Y0, Y0, Y0 // acc for b0
	VXORPS Y1, Y1, Y1 // acc for b1
	VXORPS Y2, Y2, Y2 // acc for b2
	VXORPS Y3, Y3, Y3 // acc for b3

	MOVQ CX, DX
	SHRQ $3, DX  // DX = k / 8 vector steps
	JZ   tail

loop8:
	VMOVUPS     (SI), Y4
	VFMADD231PS (R8), Y4, Y0
	VFMADD231PS (R9), Y4, Y1
	VFMADD231PS (R10), Y4, Y2
	VFMADD231PS (R11), Y4, Y3
	ADDQ        $32, SI
	ADDQ        $32, R8
	ADDQ        $32, R9
	ADDQ        $32, R10
	ADDQ        $32, R11
	DECQ        DX
	JNZ         loop8

tail:
	// Fold each ymm accumulator's upper 128 into the lower half BEFORE the
	// scalar tail: VEX xmm writes zero bits 255:128, so accumulating the
	// tail into X0..X3 first would silently destroy lanes 4-7.
	VEXTRACTF128 $1, Y0, X4
	VADDPS       X4, X0, X0
	VEXTRACTF128 $1, Y1, X4
	VADDPS       X4, X1, X1
	VEXTRACTF128 $1, Y2, X4
	VADDPS       X4, X2, X2
	VEXTRACTF128 $1, Y3, X4
	VADDPS       X4, X3, X3

	ANDQ $7, CX // CX = k % 8 scalar steps, accumulated into lane 0
	JZ   reduce

tailloop:
	VMOVSS      (SI), X4
	VFMADD231SS (R8), X4, X0
	VFMADD231SS (R9), X4, X1
	VFMADD231SS (R10), X4, X2
	VFMADD231SS (R11), X4, X3
	ADDQ        $4, SI
	ADDQ        $4, R8
	ADDQ        $4, R9
	ADDQ        $4, R10
	ADDQ        $4, R11
	DECQ        CX
	JNZ         tailloop

reduce:
	// Horizontal-sum each folded accumulator's 4 lanes into dst[0..3].
	VHADDPS X0, X0, X0
	VHADDPS X0, X0, X0
	VMOVSS  X0, (DI)

	VHADDPS X1, X1, X1
	VHADDPS X1, X1, X1
	VMOVSS  X1, 4(DI)

	VHADDPS X2, X2, X2
	VHADDPS X2, X2, X2
	VMOVSS  X2, 8(DI)

	VHADDPS X3, X3, X3
	VHADDPS X3, X3, X3
	VMOVSS  X3, 12(DI)

	VZEROUPPER
	RET
