// SPDX-License-Identifier: Apache-2.0
//go:build amd64

#include "textflag.h"

// Constants, one float32 (or int32) each, broadcast per use. Values are
// identical to the scalar GELU in tensor.go / fastmath.go, so the vectorized
// result tracks the scalar to fp32 rounding (VDIVPS is exact, like the scalar
// reciprocal; the only divergence is FMA fusion) — well inside 1e-4.
DATA gc<>+0(SB)/4, $0.7071067811865476   // invSqrt2
DATA gc<>+4(SB)/4, $0x7fffffff           // abs mask
DATA gc<>+8(SB)/4, $0x80000000           // sign mask (also used to negate)
DATA gc<>+12(SB)/4, $0.3275911           // erf p
DATA gc<>+16(SB)/4, $1.0                 // one
DATA gc<>+20(SB)/4, $0.254829592         // erf c1
DATA gc<>+24(SB)/4, $-0.284496736        // erf c2
DATA gc<>+28(SB)/4, $1.421413741         // erf c3
DATA gc<>+32(SB)/4, $-1.453152027        // erf c4
DATA gc<>+36(SB)/4, $1.061405429         // erf c5
DATA gc<>+40(SB)/4, $-87.33              // exp underflow clamp
DATA gc<>+44(SB)/4, $1.44269504088896341 // log2e
DATA gc<>+48(SB)/4, $0.693359375         // ln2 high (expC1)
DATA gc<>+52(SB)/4, $-2.12194440e-4      // ln2 low  (expC2)
DATA gc<>+56(SB)/4, $1.9875691500e-4     // exp e0
DATA gc<>+60(SB)/4, $1.3981999507e-3     // exp e1
DATA gc<>+64(SB)/4, $8.3334519073e-3     // exp e2
DATA gc<>+68(SB)/4, $4.1665795894e-2     // exp e3
DATA gc<>+72(SB)/4, $1.6666665459e-1     // exp e4
DATA gc<>+76(SB)/4, $5.0000001201e-1     // exp e5
DATA gc<>+80(SB)/4, $0.5                 // half
DATA gc<>+84(SB)/4, $-0.5                // neg half (nf rounding for x<=0)
DATA gc<>+88(SB)/4, $127                 // exponent bias (int32)
GLOBL gc<>(SB), RODATA|NOPTR, $92

// func geluAVX2(x *float32, n int)
// Applies erf-based GELU in place to n floats (n a multiple of 8).
TEXT ·geluAVX2(SB), NOSPLIT, $0-16
	MOVQ x+0(FP), SI
	MOVQ n+8(FP), CX
	SHRQ $3, CX // 8 floats per iteration
	TESTQ CX, CX
	JZ   done
	VBROADCASTSS gc<>+16(SB), Y14 // one (persistent)

loop:
	VMOVUPS (SI), Y0 // v

	VBROADCASTSS gc<>+0(SB), Y15
	VMULPS       Y15, Y0, Y1 // a = v*invSqrt2
	VBROADCASTSS gc<>+4(SB), Y15
	VANDPS       Y15, Y1, Y2 // |a|
	VBROADCASTSS gc<>+8(SB), Y15
	VANDPS       Y15, Y1, Y3 // sign(a) bits

	// t = one / (p*|a| + one)
	VBROADCASTSS gc<>+12(SB), Y15
	VMOVAPS      Y14, Y5
	VFMADD231PS  Y15, Y2, Y5 // Y5 = p*|a| + one
	VDIVPS       Y5, Y14, Y6 // t = one / Y5

	// poly = t*(c1 + t*(c2 + t*(c3 + t*(c4 + t*c5))))
	VBROADCASTSS gc<>+36(SB), Y7 // h = c5
	VBROADCASTSS gc<>+32(SB), Y15
	VFMADD213PS  Y15, Y6, Y7 // h = h*t + c4
	VBROADCASTSS gc<>+28(SB), Y15
	VFMADD213PS  Y15, Y6, Y7 // h = h*t + c3
	VBROADCASTSS gc<>+24(SB), Y15
	VFMADD213PS  Y15, Y6, Y7 // h = h*t + c2
	VBROADCASTSS gc<>+20(SB), Y15
	VFMADD213PS  Y15, Y6, Y7 // h = h*t + c1
	VMULPS       Y6, Y7, Y7  // poly = t*h

	// xe = max(-(|a|*|a|), -87.33)
	VMULPS       Y2, Y2, Y8
	VBROADCASTSS gc<>+8(SB), Y15
	VXORPS       Y15, Y8, Y8 // xe = -(a^2)
	VBROADCASTSS gc<>+40(SB), Y15
	VMAXPS       Y15, Y8, Y8

	// nf = trunc(xe*log2e - 0.5); reduce xe by nf*ln2 (two parts)
	VBROADCASTSS gc<>+44(SB), Y15
	VBROADCASTSS gc<>+84(SB), Y9
	VFMADD231PS  Y15, Y8, Y9 // Y9 = xe*log2e - 0.5
	VCVTTPS2DQ   Y9, Y10     // nfi (int32, trunc toward zero)
	VCVTDQ2PS    Y10, Y11    // nf (float)
	VBROADCASTSS gc<>+48(SB), Y15
	VFNMADD231PS Y15, Y11, Y8 // xe -= expC1*nf
	VBROADCASTSS gc<>+52(SB), Y15
	VFNMADD231PS Y15, Y11, Y8 // xe -= expC2*nf

	VMULPS Y8, Y8, Y9 // z = xe*xe

	// y = exp polynomial (Horner in xe)
	VBROADCASTSS gc<>+56(SB), Y12 // y = e0
	VBROADCASTSS gc<>+60(SB), Y15
	VFMADD213PS  Y15, Y8, Y12 // y = y*xe + e1
	VBROADCASTSS gc<>+64(SB), Y15
	VFMADD213PS  Y15, Y8, Y12
	VBROADCASTSS gc<>+68(SB), Y15
	VFMADD213PS  Y15, Y8, Y12
	VBROADCASTSS gc<>+72(SB), Y15
	VFMADD213PS  Y15, Y8, Y12
	VBROADCASTSS gc<>+76(SB), Y15
	VFMADD213PS  Y15, Y8, Y12 // y = y*xe + e5

	// e = (y*z + xe + 1) * 2^nf
	VFMADD213PS Y8, Y9, Y12       // y = y*z + xe
	VADDPS      Y14, Y12, Y12     // y += one
	VBROADCASTSS gc<>+88(SB), Y15 // 127
	VPADDD      Y15, Y10, Y10     // nfi + 127
	VPSLLD      $23, Y10, Y10     // << 23  -> 2^nf bits
	VMULPS      Y10, Y12, Y12     // e = y * 2^nf

	// erf = sign(a) XOR (one - poly*e); out = (v*0.5) * (one + erf)
	VMOVAPS      Y14, Y13
	VFNMADD231PS Y7, Y12, Y13 // val = one - poly*e
	VXORPS       Y3, Y13, Y13  // apply sign of a
	VADDPS       Y14, Y13, Y13 // one + erf
	VBROADCASTSS gc<>+80(SB), Y15
	VMULPS       Y15, Y0, Y0 // v*0.5
	VMULPS       Y13, Y0, Y0 // out
	VMOVUPS      Y0, (SI)

	ADDQ $32, SI
	DECQ CX
	JNZ  loop

done:
	VZEROUPPER
	RET
