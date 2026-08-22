// SPDX-License-Identifier: Apache-2.0

package tensor

import (
	"fmt"
	"math"
)

// PackedB8 is a weight matrix quantized to int8 (weight-only, symmetric,
// one scale per OUTPUT channel) and packed into the same 16-column k-major
// panel layout as PackedB. Activations stay float32 end to end — the only
// error introduced is the weights' rounding to 8 bits (~0.4% of each
// channel's max magnitude), and the kernel dequantizes with a single
// per-channel multiply in the epilogue. The payoff is 4× less weight
// traffic on a forward pass that is bound by streaming weights.
type PackedB8 struct {
	K, N   int
	data   []int8
	scales []float32 // [N] dequant scale per output channel
}

// PackB8 quantizes and packs a bT ([n×k] row-major) weight matrix.
// Same preconditions as PackB (SIMD kernel, n%16 == 0).
func PackB8(bT []float32, k, n int) (*PackedB8, error) {
	if !hasSIMD {
		return nil, fmt.Errorf("tensor: PackB8 requires AVX2+FMA")
	}
	if n%16 != 0 {
		return nil, fmt.Errorf("tensor: PackB8 needs n%%16==0, got n=%d", n)
	}
	if k < 1 {
		return nil, fmt.Errorf("tensor: PackB8 needs k>=1, got k=%d", k)
	}
	if len(bT) != k*n {
		return nil, fmt.Errorf("tensor: PackB8 bT has %d floats, want %d", len(bT), k*n)
	}
	p := &PackedB8{K: k, N: n, data: make([]int8, k*n), scales: make([]float32, n)}
	for j := range n {
		row := bT[j*k : j*k+k]
		var maxAbs float32
		for _, v := range row {
			if a := float32(math.Abs(float64(v))); a > maxAbs {
				maxAbs = a
			}
		}
		scale := maxAbs / 127
		if scale == 0 {
			scale = 1 // all-zero channel: quantized zeros, any scale works
		}
		p.scales[j] = scale
		panel := p.data[(j/16)*k*16:]
		c := j % 16
		inv := 1 / scale
		for pp, v := range row {
			q := math.RoundToEven(float64(v * inv))
			if q > 127 {
				q = 127
			}
			if q < -127 {
				q = -127
			}
			panel[pp*16+c] = int8(q)
		}
	}
	return p, nil
}

// MatMulPacked8 is MatMulPacked over an int8-quantized weight matrix:
// identical partitioning and contracts, gemm4x16i8 as the micro-kernel.
func MatMulPacked8(dst, a []float32, pb *PackedB8, m int, aPack []float32, pool *Pool) {
	if m == 0 {
		return
	}
	mPad := PackAPad(m)
	k, n := pb.K, pb.N
	_ = dst[mPad*n-1] // fail fast before the asm writes anything
	_ = aPack[mPad*k-1]
	packA4(aPack, a, m, k)

	rowPanels := mPad / 4
	colPanels := n / 16
	const rowChunk, colChunk = 32, 4
	rowUnits := (rowPanels + rowChunk - 1) / rowChunk
	colUnits := (colPanels + colChunk - 1) / colChunk
	units := rowUnits * colUnits
	unitMACs := min(rowPanels, rowChunk) * 4 * k * min(colPanels, colChunk) * 16
	if units < 2 || unitMACs < minUnitWork {
		gemm8Chunk(dst, aPack, pb, 0, rowPanels, 0, colPanels, k, n)
		return
	}
	body := func(u int) {
		ip0 := (u / colUnits) * rowChunk
		jp0 := (u % colUnits) * colChunk
		gemm8Chunk(dst, aPack, pb, ip0, min(ip0+rowChunk, rowPanels), jp0, min(jp0+colChunk, colPanels), k, n)
	}
	if pool != nil {
		pool.Run(units, body)
	} else {
		ParallelFor(units, body)
	}
}

func gemm8Chunk(dst, aPack []float32, pb *PackedB8, ip0, ip1, jp0, jp1, k, n int) {
	for jp := jp0; jp < jp1; jp++ {
		pbPanel := &pb.data[jp*k*16]
		scales := &pb.scales[jp*16]
		for ip := ip0; ip < ip1; ip++ {
			off := ip*4*n + jp*16
			d := dst[off : off+3*n+16]
			gemm4x16i8(&d[0], n, &aPack[ip*k*4], pbPanel, k, scales)
		}
	}
}
