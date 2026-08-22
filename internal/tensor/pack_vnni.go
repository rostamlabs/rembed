// SPDX-License-Identifier: Apache-2.0

package tensor

import (
	"fmt"
	"math"
)

// PackedB8V is a weight matrix packed for the AVX-VNNI int8 kernel:
// activations AND weights integer, VPDPBUSD accumulating u8·s8 into
// int32. Weights are per-output-channel symmetric int8 (same scheme as
// PackedB8); activations are quantized per row at matmul time (see
// MatMulPackedVNNI). Layout: 16-column panels, each k-group of 4 storing
// 16 columns × 4 consecutive-k bytes — the dword lanes VPDPBUSD consumes.
type PackedB8V struct {
	K, N   int
	kg     int       // number of 4-k groups (k rounded up)
	data   []int8    // [n/16 panels][kg][16 cols][4]
	colSum []int32   // per-column Σ_k w — the u8 zero-point correction
	scales []float32 // per-column dequant scale
}

// PackB8VNNI packs bT (HF [out,in] layout) for the VNNI kernel. Same
// preconditions as PackB8: n%16==0, n>=16, k>=1, finite weights, and the
// CPU must have AVX-VNNI.
func PackB8VNNI(bT []float32, k, n int) (*PackedB8V, error) {
	if !hasVNNI && !hasVNNI512 {
		return nil, fmt.Errorf("tensor: PackB8VNNI requires AVX-VNNI or AVX-512-VNNI on this CPU")
	}
	if n < 16 || n%16 != 0 {
		return nil, fmt.Errorf("tensor: PackB8VNNI needs n%%16==0 and n>=16, got n=%d", n)
	}
	if k < 1 {
		return nil, fmt.Errorf("tensor: PackB8VNNI needs k>=1, got k=%d", k)
	}
	if len(bT) != k*n {
		return nil, fmt.Errorf("tensor: PackB8VNNI bT has %d elems, want %d", len(bT), k*n)
	}
	kg := (k + 3) / 4
	pb := &PackedB8V{
		K: k, N: n, kg: kg,
		data:   make([]int8, (n/16)*kg*64),
		colSum: make([]int32, n),
		scales: make([]float32, n),
	}
	// Per-column symmetric scales, float64 division (the R5-era lesson:
	// float32 reciprocals double-round and collapse subnormal channels).
	for c := range n {
		var maxAbs float64
		for p := range k {
			v := float64(bT[c*k+p])
			if math.IsNaN(v) || math.IsInf(v, 0) {
				return nil, fmt.Errorf("tensor: PackB8VNNI weight [%d,%d] is not finite", c, p)
			}
			if a := math.Abs(v); a > maxAbs {
				maxAbs = a
			}
		}
		scale := maxAbs / 127
		if scale == 0 {
			scale = 1
		}
		pb.scales[c] = float32(scale)
		panel, col := c/16, c%16
		base := panel * kg * 64
		var sum int32
		for p := range k {
			q := int8(math.RoundToEven(float64(bT[c*k+p]) / scale))
			sum += int32(q)
			pb.data[base+(p/4)*64+col*4+(p%4)] = q
		}
		pb.colSum[c] = sum
	}
	return pb, nil
}

// QuantizeRowU8 quantizes one activation row to u8 with the +128 offset
// trick: q = round(a/s) + 128, s = absmax/127, so VPDPBUSD's u8·s8
// accumulation equals the true product plus 128·colSum — subtracted in
// the epilogue. dst must hold kg*4 bytes; the k..kg*4 tail bytes are
// zeroed (the matching weight bytes are zero, so their contribution is
// zero regardless). Returns the row scale.
func QuantizeRowU8(dst []uint8, a []float32) float32 {
	var maxAbs float32
	for _, v := range a {
		if x := float32(math.Abs(float64(v))); x > maxAbs {
			maxAbs = x
		}
	}
	scale := maxAbs / 127
	if scale == 0 {
		scale = 1
	}
	// float32 reciprocal ON PURPOSE, unlike the weight path's float64
	// division: this runs per activation row per matmul (hot), weights
	// quantize once at load (cold). The double-rounding hazard that
	// float64 guards against needs subnormal-scale rows, which real
	// activations never produce — and the test reference mirrors this
	// exact expression, so bit-exactness is preserved either way.
	inv := 1 / scale
	for i, v := range a {
		q := int32(math.RoundToEven(float64(v*inv))) + 128
		// Rounding can graze the rails at ±127.5; clamp to u8.
		if q < 0 {
			q = 0
		} else if q > 255 {
			q = 255
		}
		dst[i] = uint8(q)
	}
	for i := len(a); i < len(dst); i++ {
		dst[i] = 0
	}
	return scale
}

// MatMulPackedVNNI computes dst[m×n] = a[m×k]·Bᵀ using the AVX-VNNI
// int8 kernel with per-row-quantized activations. qa must hold
// PackAPad(m)·kg·4 bytes and rowScales PackAPad(m) floats (scratch,
// caller-owned); dst must be sized for PackAPad(m) rows. Work fans out
// over the pool exactly like MatMulPacked.
func MatMulPackedVNNI(dst, a []float32, pb *PackedB8V, m int, qa []uint8, rowScales []float32, pool *Pool) {
	k, n, kg := pb.K, pb.N, pb.kg
	mPad := PackAPad(m)
	aBytes := kg * 4
	// Quantize rows (pool-parallel over 4-row panels, mirroring packA4).
	rowPanels := mPad / 4
	pool.Run(rowPanels, func(p int) {
		for r := p * 4; r < p*4+4; r++ {
			row := qa[r*aBytes : (r+1)*aBytes]
			if r < m {
				rowScales[r] = QuantizeRowU8(row, a[r*k:r*k+k])
			} else {
				clear(row)
				rowScales[r] = 1
			}
		}
	})

	colPanels := n / 16
	units := rowPanels * colPanels
	pool.Run(units, func(u int) {
		rp := u / colPanels
		cp := u % colPanels
		var acc [64]int32
		// Two encodings of the same instruction, disjoint hardware: VEX
		// for AVX-VNNI clients, EVEX for AVX-512-VNNI servers.
		if hasVNNI {
			gemm4x16vnni(&acc[0], 16, &qa[rp*4*aBytes], aBytes, &pb.data[cp*kg*64], kg)
		} else {
			gemm4x16vnni512(&acc[0], 16, &qa[rp*4*aBytes], aBytes, &pb.data[cp*kg*64], kg)
		}
		// Dequant epilogue: (acc − 128·colSum)·rowScale·colScale. int32
		// wraparound in acc CANCELS exactly against the correction (both
		// are exact mod 2^32; the true Σt·w is bounded by 128·128·k), so
		// the real overflow limit is k > 131072 — ~85× past any
		// transformer dimension. Unlike MatMulPacked8, dst's padding rows
		// are left unwritten (the row >= m break); every consumer reads
		// only seq rows.
		for r := 0; r < 4; r++ {
			row := rp*4 + r
			if row >= m {
				break
			}
			d := dst[row*n+cp*16:]
			rs := rowScales[row]
			for c := 0; c < 16; c++ {
				col := cp*16 + c
				real := acc[r*16+c] - 128*pb.colSum[col]
				d[c] = float32(real) * rs * pb.scales[col]
			}
		}
	})
}
