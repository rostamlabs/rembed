// SPDX-License-Identifier: Apache-2.0
//go:build arm64

#include "textflag.h"

// func gemm4x16(dst *float32, n int, pa, pb *float32, k int)
//
// NEON port of the amd64 kernel: C[4×16] = A-panel × B-panel, written
// (not accumulated) to dst with row stride n floats. pa is k-major with 4
// rows interleaved, pb k-major with 16 columns interleaved. Per k step:
// one 64-byte B load (4 vectors), one 16-byte A load, 4 lane dups, 16
// FMLAs into 16 independent accumulators (V0-V15) — FMLA's two source
// operands commute, so operand-order ambiguity in the assembler cannot
// change results. k must be >= 1.
TEXT ·gemm4x16(SB), NOSPLIT, $0-40
	MOVD dst+0(FP), R0
	MOVD n+8(FP), R1
	MOVD pa+16(FP), R2
	MOVD pb+24(FP), R3
	MOVD k+32(FP), R4
	LSL  $2, R1, R1 // row stride in bytes

	VEOR V0.B16, V0.B16, V0.B16
	VEOR V1.B16, V1.B16, V1.B16
	VEOR V2.B16, V2.B16, V2.B16
	VEOR V3.B16, V3.B16, V3.B16
	VEOR V4.B16, V4.B16, V4.B16
	VEOR V5.B16, V5.B16, V5.B16
	VEOR V6.B16, V6.B16, V6.B16
	VEOR V7.B16, V7.B16, V7.B16
	VEOR V8.B16, V8.B16, V8.B16
	VEOR V9.B16, V9.B16, V9.B16
	VEOR V10.B16, V10.B16, V10.B16
	VEOR V11.B16, V11.B16, V11.B16
	VEOR V12.B16, V12.B16, V12.B16
	VEOR V13.B16, V13.B16, V13.B16
	VEOR V14.B16, V14.B16, V14.B16
	VEOR V15.B16, V15.B16, V15.B16

loop:
	VLD1.P 64(R3), [V16.S4, V17.S4, V18.S4, V19.S4]
	VLD1.P 16(R2), [V20.S4]
	VDUP   V20.S[0], V21.S4
	VDUP   V20.S[1], V22.S4
	VDUP   V20.S[2], V23.S4
	VDUP   V20.S[3], V24.S4
	VFMLA  V16.S4, V21.S4, V0.S4
	VFMLA  V17.S4, V21.S4, V1.S4
	VFMLA  V18.S4, V21.S4, V2.S4
	VFMLA  V19.S4, V21.S4, V3.S4
	VFMLA  V16.S4, V22.S4, V4.S4
	VFMLA  V17.S4, V22.S4, V5.S4
	VFMLA  V18.S4, V22.S4, V6.S4
	VFMLA  V19.S4, V22.S4, V7.S4
	VFMLA  V16.S4, V23.S4, V8.S4
	VFMLA  V17.S4, V23.S4, V9.S4
	VFMLA  V18.S4, V23.S4, V10.S4
	VFMLA  V19.S4, V23.S4, V11.S4
	VFMLA  V16.S4, V24.S4, V12.S4
	VFMLA  V17.S4, V24.S4, V13.S4
	VFMLA  V18.S4, V24.S4, V14.S4
	VFMLA  V19.S4, V24.S4, V15.S4
	SUBS   $1, R4, R4
	BNE    loop

	VST1 [V0.S4, V1.S4, V2.S4, V3.S4], (R0)
	ADD  R1, R0, R0
	VST1 [V4.S4, V5.S4, V6.S4, V7.S4], (R0)
	ADD  R1, R0, R0
	VST1 [V8.S4, V9.S4, V10.S4, V11.S4], (R0)
	ADD  R1, R0, R0
	VST1 [V12.S4, V13.S4, V14.S4, V15.S4], (R0)
	RET
