// SPDX-License-Identifier: Apache-2.0
//go:build arm64

#include "textflag.h"

// func dot4(dst, a, b0, b1, b2, b3 *float32, k int)
//
// NEON port of the amd64 dot kernel: dst[0..3] = dot(a, b0..b3) over k
// floats. Four 4-lane accumulators; the k%4 tail uses scalar FMADDS into
// separate scalar accumulators; the final lane reduction extracts lanes
// through GPRs (VMOV Vn.S[i] is UMOV; FMOVS Rn moves the bits back to an
// FP register) and sums with scalar FADDS — the Go assembler has no float
// horizontal-add mnemonic. Register-only reduction keeps the function
// frameless: an earlier version spilled to (RSP), which in a $16 frame is
// the saved-LR slot, not the locals — harmless to a leaf's execution but
// it corrupts SIGPROF unwinding (runtime reads the innermost caller PC
// from frame.sp), silently truncating CPU profile samples. Accumulation
// order differs from the scalar kernels — within fp32 rounding, same
// contract as amd64.
TEXT ·dot4(SB), NOSPLIT, $0-56
	MOVD dst+0(FP), R0
	MOVD a+8(FP), R1
	MOVD b0+16(FP), R2
	MOVD b1+24(FP), R3
	MOVD b2+32(FP), R4
	MOVD b3+40(FP), R5
	MOVD k+48(FP), R6

	VEOR  V0.B16, V0.B16, V0.B16
	VEOR  V1.B16, V1.B16, V1.B16
	VEOR  V2.B16, V2.B16, V2.B16
	VEOR  V3.B16, V3.B16, V3.B16
	FMOVS ZR, F24 // scalar tail accumulators
	FMOVS ZR, F25
	FMOVS ZR, F26
	FMOVS ZR, F27

	LSR $2, R6, R7 // vector steps
	CBZ R7, tail

loop:
	VLD1.P 16(R1), [V4.S4]
	VLD1.P 16(R2), [V5.S4]
	VLD1.P 16(R3), [V6.S4]
	VLD1.P 16(R4), [V7.S4]
	VLD1.P 16(R5), [V16.S4]
	VFMLA  V4.S4, V5.S4, V0.S4
	VFMLA  V4.S4, V6.S4, V1.S4
	VFMLA  V4.S4, V7.S4, V2.S4
	VFMLA  V4.S4, V16.S4, V3.S4
	SUBS   $1, R7, R7
	BNE    loop

tail:
	ANDS $3, R6, R6
	BEQ  reduce

tailloop:
	FMOVS.P 4(R1), F16
	FMOVS.P 4(R2), F17
	FMOVS.P 4(R3), F18
	FMOVS.P 4(R4), F19
	FMOVS.P 4(R5), F20
	FMADDS  F16, F24, F17, F24
	FMADDS  F16, F25, F18, F25
	FMADDS  F16, F26, F19, F26
	FMADDS  F16, F27, F20, F27
	SUBS    $1, R6, R6
	BNE     tailloop

reduce:
	// Sum each accumulator's four lanes lane-by-lane through R8-R11, add
	// the scalar tail, store dst[c]. The FP temporaries are F16-F19:
	// scalar Fn ALIASES Vn.S[0] on arm64, so using F1-F3 here would
	// clobber the low lanes of the not-yet-reduced accumulators V1-V3
	// (found by the qemu sweep).
	VMOV  V0.S[0], R8
	VMOV  V0.S[1], R9
	VMOV  V0.S[2], R10
	VMOV  V0.S[3], R11
	FMOVS R8, F16
	FMOVS R9, F17
	FMOVS R10, F18
	FMOVS R11, F19
	FADDS F17, F16, F16
	FADDS F19, F18, F18
	FADDS F18, F16, F16
	FADDS F24, F16, F16
	FMOVS F16, (R0)

	VMOV  V1.S[0], R8
	VMOV  V1.S[1], R9
	VMOV  V1.S[2], R10
	VMOV  V1.S[3], R11
	FMOVS R8, F16
	FMOVS R9, F17
	FMOVS R10, F18
	FMOVS R11, F19
	FADDS F17, F16, F16
	FADDS F19, F18, F18
	FADDS F18, F16, F16
	FADDS F25, F16, F16
	FMOVS F16, 4(R0)

	VMOV  V2.S[0], R8
	VMOV  V2.S[1], R9
	VMOV  V2.S[2], R10
	VMOV  V2.S[3], R11
	FMOVS R8, F16
	FMOVS R9, F17
	FMOVS R10, F18
	FMOVS R11, F19
	FADDS F17, F16, F16
	FADDS F19, F18, F18
	FADDS F18, F16, F16
	FADDS F26, F16, F16
	FMOVS F16, 8(R0)

	VMOV  V3.S[0], R8
	VMOV  V3.S[1], R9
	VMOV  V3.S[2], R10
	VMOV  V3.S[3], R11
	FMOVS R8, F16
	FMOVS R9, F17
	FMOVS R10, F18
	FMOVS R11, F19
	FADDS F17, F16, F16
	FADDS F19, F18, F18
	FADDS F18, F16, F16
	FADDS F27, F16, F16
	FMOVS F16, 12(R0)
	RET
