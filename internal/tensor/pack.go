// SPDX-License-Identifier: Apache-2.0

package tensor

import "fmt"

// PackedB is a weight matrix pre-packed ONCE (at model load) into the
// panel layout a gemm micro-kernel streams: panelW-column panels, k-major
// within each panel (data[jp*K*pw + p*pw + c] = bT[(jp*pw+c)*K + p]).
// panelW is 16 (AVX2 gemm4x16) or 32 (AVX-512 gemm4x32) — decided at
// pack time by what the CPU has and what n divides by. Weights are
// static for the life of a model, so rembed pays the packing cost once
// where a runtime pays it per session or per call — this is the
// structural advantage the packed path exists to exploit.
type PackedB struct {
	K, N   int
	panelW int
	data   []float32
}

// PackB packs a bT ([n×k] row-major) weight matrix. It requires the SIMD
// kernel (the packed layout exists only for gemm4x16) and n a multiple of
// 16 (true for every BERT-family dimension: hidden, 3·hidden,
// intermediate); callers keep the unpacked path for anything else.
func PackB(bT []float32, k, n int) (*PackedB, error) {
	if !hasSIMD {
		return nil, fmt.Errorf("tensor: PackB requires a native SIMD kernel on this architecture (packed layout feeds it only)")
	}
	if n < 16 || n%16 != 0 {
		return nil, fmt.Errorf("tensor: PackB needs n%%16==0 and n>=16, got n=%d", n)
	}
	if k < 1 {
		return nil, fmt.Errorf("tensor: PackB needs k>=1, got k=%d", k)
	}
	if len(bT) != k*n {
		return nil, fmt.Errorf("tensor: PackB bT has %d floats, want %d", len(bT), k*n)
	}
	// The zmm kernel wants 32-wide panels; every BERT-family dim (384,
	// 768, 1152, 1536, 2304, 3072) divides by 32, so on AVX-512 hardware
	// this is the layout that gets built. Per-element accumulation order
	// is identical either way, so results are bit-identical across panel
	// widths (test-pinned on AVX-512 hardware).
	pw := 16
	if hasAVX512 && n%32 == 0 {
		pw = 32
	}
	p := &PackedB{K: k, N: n, panelW: pw, data: make([]float32, k*n)}
	for jp := range n / pw {
		panel := p.data[jp*k*pw:]
		for c := range pw {
			col := bT[(jp*pw+c)*k:]
			for pp := range k {
				panel[pp*pw+c] = col[pp]
			}
		}
	}
	return p, nil
}

// PackAPad returns the padded row count MatMulPacked uses for m rows.
func PackAPad(m int) int { return (m + 3) &^ 3 }

// packA4 packs a ([m×k] row-major) into 4-row k-major panels
// (dst[ip*k*4 + p*4 + r]), zero-filling pad rows so the kernel's extra
// output rows are deterministic zeros.
func packA4(dst, a []float32, m, k int, pool *Pool) {
	mPad := PackAPad(m)
	rowPanels := mPad / 4
	// The pack is a cache-hostile stride-4 scatter and, run serially before
	// the fan-out, it measured 21% of the whole ffn2 matmul at seq=512 — a
	// hard Amdahl ceiling. Row panels are independent, so fan them out too
	// (tiny inputs stay inline; the pack for 3 panels is a few µs).
	packOne := func(ip int) {
		panel := dst[ip*k*4 : ip*k*4+k*4]
		for r := range 4 {
			i := ip*4 + r
			if i >= m {
				for p := range k {
					panel[p*4+r] = 0
				}
				continue
			}
			row := a[i*k : i*k+k]
			for p, v := range row {
				panel[p*4+r] = v
			}
		}
	}
	if pool == nil || rowPanels < 8 {
		for ip := range rowPanels {
			packOne(ip)
		}
		return
	}
	pool.Run(rowPanels, packOne)
}

// MatMulPacked computes dst[m×N] = a[m×K] · Bᵀ from a pre-packed weight
// matrix. dst must have PackAPad(m)·N floats (the kernel writes whole 4-row
// tiles; pad rows receive zeros and must simply be writable — model scratch
// is sized for this) and aPack PackAPad(m)·K floats of caller scratch.
// A non-nil pool runs the fan-out on its spinning workers (the latency
// path); nil falls back to ParallelFor.
// Accumulation is scalar-ordered per element within the kernel's broadcast
// scheme; results match the float64 reference within fp32 rounding.
func MatMulPacked(dst, a []float32, pb *PackedB, m int, aPack []float32, pool *Pool) {
	if m == 0 {
		return
	}
	mPad := PackAPad(m)
	k, n := pb.K, pb.N
	_ = dst[mPad*n-1] // fail fast before the asm writes anything
	_ = aPack[mPad*k-1]
	packA4(aPack, a, m, k, pool)

	rowPanels := mPad / 4
	colPanels := n / pb.panelW
	// A parallel unit is a CHUNK of micro-tiles, not one 4×16 tile: one
	// tile is ~2-5 µs of SIMD work, and a first cut that fanned out single
	// tiles ran SLOWER than serial — 200+ units thrashed the atomic
	// counter, and the counter's interleaving handed adjacent 64-byte
	// output strips to different workers, maximizing false sharing. Chunks
	// of 8×4 panels (32 rows × 64 cols) restore ~the granularity the
	// unpacked path uses, and the B chunk is reused across the chunk's row
	// panels while cache-hot.
	const rowChunk, colChunk = 32, 4
	rowUnits := (rowPanels + rowChunk - 1) / rowChunk
	colUnits := (colPanels + colChunk - 1) / colChunk
	units := rowUnits * colUnits
	unitMACs := min(rowPanels, rowChunk) * 4 * k * min(colPanels, colChunk) * pb.panelW
	if units < 2 || unitMACs < minUnitWork {
		gemmChunk(dst, aPack, pb, 0, rowPanels, 0, colPanels, k, n)
		return
	}
	body := func(u int) {
		ip0 := (u / colUnits) * rowChunk
		jp0 := (u % colUnits) * colChunk
		gemmChunk(dst, aPack, pb, ip0, min(ip0+rowChunk, rowPanels), jp0, min(jp0+colChunk, colPanels), k, n)
	}
	if pool != nil {
		pool.Run(units, body)
	} else {
		ParallelFor(units, body)
	}
}

// gemmChunk runs the micro-kernel over row panels [ip0,ip1) × column panels
// [jp0,jp1). The COLUMN loop is outermost: at short seq the whole pass is
// bound by streaming the weights from DRAM, and a row-outer order
// re-streams every B panel once per row panel (measured: 3× the weight
// traffic at seq=12, 41 cycles per k-step against ~4 of compute). With jp
// outer, each B panel (k×16 floats, L1/L2-resident) is loaded from memory
// once and reused across every row panel of the chunk.
func gemmChunk(dst, aPack []float32, pb *PackedB, ip0, ip1, jp0, jp1, k, n int) {
	pw := pb.panelW
	for jp := jp0; jp < jp1; jp++ {
		// Reslice every operand to exactly what the asm touches, matching
		// the dst discipline — a raw &s[i] checks only the first element.
		bp := pb.data[jp*k*pw : (jp+1)*k*pw]
		for ip := ip0; ip < ip1; ip++ {
			ap := aPack[ip*k*4 : (ip+1)*k*4]
			off := ip*4*n + jp*pw
			// Reslice so every float the asm writes (3 full rows of stride
			// n plus the final panel-wide row) is bounds-checked up front.
			d := dst[off : off+3*n+pw]
			if pw == 32 {
				gemm4x32(&d[0], n, &ap[0], &bp[0], k)
			} else {
				gemm4x16(&d[0], n, &ap[0], &bp[0], k)
			}
		}
	}
}
