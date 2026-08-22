// SPDX-License-Identifier: Apache-2.0

//go:build amd64

package tensor

import "golang.org/x/sys/cpu"

// hasSIMD gates the AVX2+FMA kernel; both features together cover every
// x86-64-v3 machine (Haswell 2013 onward). Checked once at init.
var hasSIMD = cpu.X86.HasAVX2 && cpu.X86.HasFMA

// dot4AVX2 computes dst[0..3] = dot(a, b0..b3) over k floats with 8-lane FMA
// accumulators (implemented in dot_amd64.s). The horizontal reduction at the
// end means its accumulation order differs from the scalar kernels: results
// are within fp32 rounding, not bit-identical.
//
//go:noescape
func dot4AVX2(dst, a, b0, b1, b2, b3 *float32, k int)
