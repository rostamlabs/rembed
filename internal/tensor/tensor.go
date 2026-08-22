// SPDX-License-Identifier: Apache-2.0

// Package tensor holds the math kernels for the transformer forward pass.
// This is the perf-critical core: the optimization ladder (naive →
// cache-blocked → SIMD → int8) swaps kernel bodies behind the signatures in
// this package without touching model code.
package tensor

import (
	"math"
	"runtime"
	"sync"
	"sync/atomic"
)

// MatMulFunc computes C[m×n] = A[m×k] · Bᵀ, where bT is the [n×k] row-major
// transpose of B. This layout is chosen deliberately: HuggingFace stores
// nn.Linear weights as [out, in] = [n, k], so linear layers pass weights
// straight through with no transpose, and every dot product reads two
// contiguous rows.
type MatMulFunc func(dst, a, bT []float32, m, k, n int)

// Default returns the best matmul body for this platform: the AVX2+FMA
// SIMD kernel where the CPU supports it, the scalar parallel kernel
// otherwise. Model code binds it ONCE at load time (a mutable package
// variable would race with in-flight forward passes); benchmarks A/B the
// named implementations directly.
func Default() MatMulFunc {
	if hasSIMD {
		return MatMulSIMD
	}
	return MatMulParallel
}

// MatMulNaive is the M0 reference body: three plain loops, dot products of
// contiguous rows.
func MatMulNaive(dst, a, bT []float32, m, k, n int) {
	for i := range m {
		ar := a[i*k : i*k+k]
		for j := range n {
			br := bT[j*k : j*k+k]
			var sum float32
			for p := range k {
				sum += ar[p] * br[p]
			}
			dst[i*n+j] = sum
		}
	}
}

// Blocking parameters for MatMulBlocked.
const (
	// blockM: A-panel rows per tile. 64 rows × 512 floats × 4 B ≈ 128 KB
	// worst-case (k=I=1536 gives 384 KB — still L2-resident), so the panel
	// is reused from cache across every bT row of the tile.
	blockM = 64
	// blockN: bT rows per micro-kernel pass. 4 independent accumulator
	// chains give the FP units ILP, and each loaded a[p] is reused 4×.
	blockN = 4
)

// MatMulBlocked is the M1 body: cache-blocked with a 1×4 register
// micro-kernel. The naive body streams ALL of bT from memory for every row
// of A (m × n×k floats); here the j-loop is outermost within an A-tile, so
// bT streams through cache once per tile of blockM rows and the A panel
// stays resident. Four independent accumulator chains give the FP units
// ILP; a 2×4 variant measured ~7% SLOWER here (register spills — Go keeps
// only so many of 8 accumulators + 6 temps live), so 1×4 it is until M3's
// SIMD kernel. Accumulation order over k matches naive, so results are
// bit-identical wherever the compiler does not fuse mul+add (amd64 today),
// and within fp32 rounding otherwise.
func MatMulBlocked(dst, a, bT []float32, m, k, n int) {
	if m*n > 0 {
		// Fail fast on an undersized dst: the 4-wide stores below slice with
		// an explicit upper bound, which bounds-checks against cap — on a
		// pooled buffer with cap > len an undersized dst would otherwise be
		// corrupted silently instead of panicking like the naive body does.
		_ = dst[m*n-1]
	}
	matMulBlockedCols(dst, a, bT, m, k, n, 0, n)
}

// matMulBlockedCols is the blocked kernel restricted to output columns
// [jLo, jHi). Workers on disjoint column ranges write disjoint dst elements,
// and each element's k-accumulation is unchanged, so a column-partitioned
// run is bit-identical to a whole-matrix run.
func matMulBlockedCols(dst, a, bT []float32, m, k, n, jLo, jHi int) {
	for i0 := 0; i0 < m; i0 += blockM {
		i1 := min(i0+blockM, m)
		var j int
		for j = jLo; j+blockN <= jHi; j += blockN {
			b0 := bT[(j+0)*k : (j+0)*k+k]
			b1 := bT[(j+1)*k : (j+1)*k+k]
			b2 := bT[(j+2)*k : (j+2)*k+k]
			b3 := bT[(j+3)*k : (j+3)*k+k]
			for i := i0; i < i1; i++ {
				ar := a[i*k : i*k+k]
				var s0, s1, s2, s3 float32
				for p, av := range ar {
					s0 += av * b0[p]
					s1 += av * b1[p]
					s2 += av * b2[p]
					s3 += av * b3[p]
				}
				d := dst[i*n+j : i*n+j+4]
				d[0], d[1], d[2], d[3] = s0, s1, s2, s3
			}
		}
		for ; j < jHi; j++ { // remainder columns of this range
			br := bT[j*k : j*k+k]
			for i := i0; i < i1; i++ {
				ar := a[i*k : i*k+k]
				var s float32
				for p, av := range ar {
					s += av * br[p]
				}
				dst[i*n+j] = s
			}
		}
	}
}

// ParallelFor runs fn(u) for every u in [0, units), fanned out over
// min(GOMAXPROCS, units) goroutines pulling unit indexes from an atomic
// counter. The fan-out is bounded by GOMAXPROCS so nested or concurrent
// calls cannot explode the goroutine count, and the counter load-balances
// uneven units for free. units <= 1 (or a single-core box) runs inline.
// A panic in fn is captured and re-raised on the calling goroutine after
// all workers finish — a library must not let a worker panic kill the host
// process when the caller's recover() could have handled it.
func ParallelFor(units int, fn func(u int)) {
	ParallelForCap(runtime.GOMAXPROCS(0), units, fn)
}

// ParallelForCap is ParallelFor with the fan-out additionally capped at
// cap workers. When the total work is small (a short-sequence forward
// pass), goroutine wake latency dominates wide fan-outs — measured at
// seq=12, 4 workers beat 20 by ~20% — so the model caps the fan-out from
// what it knows about the sequence length.
func ParallelForCap(cap, units int, fn func(u int)) {
	workers := min(cap, runtime.GOMAXPROCS(0), units)
	if workers <= 1 {
		for u := range units {
			fn(u)
		}
		return
	}
	var panicked atomic.Pointer[any]
	var next atomic.Int64
	var wg sync.WaitGroup
	// The caller is worker zero: it starts on units immediately instead of
	// blocking in Wait while spawned workers wake. For small fan-outs the
	// caller often drains the counter before the workers are running, and
	// Wait returns without ever parking — cutting the wake latency that
	// dominated short-sequence forward passes.
	wg.Add(workers - 1)
	for range workers - 1 {
		go func() {
			defer wg.Done()
			defer func() {
				if r := recover(); r != nil {
					panicked.CompareAndSwap(nil, &r)
				}
			}()
			for {
				u := int(next.Add(1)) - 1
				if u >= units {
					return
				}
				fn(u)
			}
		}()
	}
	func() {
		defer func() {
			if r := recover(); r != nil {
				panicked.CompareAndSwap(nil, &r)
			}
		}()
		for {
			u := int(next.Add(1)) - 1
			if u >= units {
				return
			}
			fn(u)
		}
	}()
	wg.Wait()
	if p := panicked.Load(); p != nil {
		panic(*p)
	}
}

// parallelCols is the column-block width of one parallel unit; rows
// partition by blockM, giving a 2-D unit grid. It MUST stay a multiple of
// blockN: that is what guarantees a 4-wide micro-kernel store can never
// straddle a unit's column boundary, which is the disjointness argument for
// the parallel path's safety (-race does not instrument assembly stores, so
// this invariant — not the race detector — is the evidence).
const parallelCols = 64

// minUnitWork is the per-unit size (multiply-adds) below which a unit is
// not worth a worker's while: 1<<14 ≈ 5 µs of scalar work, comfortably
// above spawn/counter overhead. The gate is per-UNIT, not total work — a
// 1×384·384×384 projection is only ~1.8 M MACs total but its 6 units are
// each worth parallelizing (the original total-work gate of 1<<20 left
// every short-sequence projection serial, costing seq<8 inputs most of the
// rung's benefit).
const minUnitWork = 1 << 14

// MatMulParallel is the M2 body: MatMulBlocked with the output partitioned
// into blockM×parallelCols tiles executed via ParallelFor. Workers own
// disjoint output tiles (no synchronization beyond the final wait), and
// each element's k-accumulation is untouched, so the result is
// bit-identical to MatMulBlocked.
func MatMulParallel(dst, a, bT []float32, m, k, n int) {
	if m*n > 0 {
		_ = dst[m*n-1] // same fail-fast as MatMulBlocked
	}
	rowTiles := (m + blockM - 1) / blockM
	colBlocks := (n + parallelCols - 1) / parallelCols
	units := rowTiles * colBlocks
	perUnit := min(m, blockM) * k * min(n, parallelCols)
	if units < 2 || perUnit < minUnitWork {
		matMulBlockedCols(dst, a, bT, m, k, n, 0, n)
		return
	}
	ParallelFor(units, func(u int) {
		i0 := (u / colBlocks) * blockM
		i1 := min(i0+blockM, m)
		jLo := (u % colBlocks) * parallelCols
		jHi := min(jLo+parallelCols, n)
		// A row block is contiguous in dst and a, so the row range is just
		// a sub-slice; the kernel's own i-tiling sees a single tile.
		matMulBlockedCols(dst[i0*n:i1*n], a[i0*k:i1*k], bT, i1-i0, k, n, jLo, jHi)
	})
}

// MatMulSerial is the best SERIAL matmul body: the SIMD kernel where the
// CPU supports it, scalar blocked otherwise. It exists for callers that are
// already inside a parallel region (the per-head attention workers) — using
// MatMulSIMD there would nest fan-outs and oversubscribe the scheduler.
func MatMulSerial(dst, a, bT []float32, m, k, n int) {
	if m*n > 0 {
		_ = dst[m*n-1] // fail fast on undersized dst
	}
	if !hasSIMD || k == 0 {
		matMulBlockedCols(dst, a, bT, m, k, n, 0, n)
		return
	}
	matMulSIMDCols(dst, a, bT, m, k, n, 0, n)
}

// matMulSIMDCols is matMulBlockedCols with the 1×4 micro-kernel replaced by
// the AVX2+FMA dot4 primitive. Same tiling, same remainder handling; the
// only numeric difference is dot4's 8-lane accumulation order (within fp32
// rounding of the scalar kernels, not bit-identical).
func matMulSIMDCols(dst, a, bT []float32, m, k, n, jLo, jHi int) {
	for i0 := 0; i0 < m; i0 += blockM {
		i1 := min(i0+blockM, m)
		var j int
		for j = jLo; j+blockN <= jHi; j += blockN {
			b0 := bT[(j+0)*k : (j+0)*k+k]
			b1 := bT[(j+1)*k : (j+1)*k+k]
			b2 := bT[(j+2)*k : (j+2)*k+k]
			b3 := bT[(j+3)*k : (j+3)*k+k]
			for i := i0; i < i1; i++ {
				ar := a[i*k : i*k+k]
				// The 4-wide reslice restores MatMulBlocked's fail-fast
				// contract: the asm writes 4 floats, and &dst[i*n+j] alone
				// would bounds-check only the first — on a pooled buffer
				// (cap > len) the other three could corrupt silently.
				d := dst[i*n+j : i*n+j+4]
				dot4AVX2(&d[0], &ar[0], &b0[0], &b1[0], &b2[0], &b3[0], k)
			}
		}
		for ; j < jHi; j++ { // remainder columns of this range (scalar)
			br := bT[j*k : j*k+k]
			for i := i0; i < i1; i++ {
				ar := a[i*k : i*k+k]
				var s float32
				for p, av := range ar {
					s += av * br[p]
				}
				dst[i*n+j] = s
			}
		}
	}
}

// MatMulSIMD is the M3 body: the AVX2+FMA dot kernel under the same 2-D
// output partition and per-unit gate as MatMulParallel. Falls back to the
// scalar parallel body when the CPU lacks AVX2/FMA — or when k is 0, which
// the Go call site cannot express (&ar[0] on an empty slice panics; the
// asm itself handles k=0 fine).
func MatMulSIMD(dst, a, bT []float32, m, k, n int) {
	if !hasSIMD || k == 0 {
		MatMulParallel(dst, a, bT, m, k, n)
		return
	}
	if m*n > 0 {
		_ = dst[m*n-1] // same fail-fast as MatMulBlocked
	}
	rowTiles := (m + blockM - 1) / blockM
	colBlocks := (n + parallelCols - 1) / parallelCols
	units := rowTiles * colBlocks
	perUnit := min(m, blockM) * k * min(n, parallelCols)
	// Same gate as the scalar path. The SIMD kernel shrinks a unit's work
	// ~6×, but raising the gate proportionally would flip short-sequence
	// projections back to serial — the regression the M2 review caught —
	// and at the gate boundary the fan-out still wins (a seq=3 projection
	// is ~36 µs serial vs ~10 µs fanned out). It also keeps allocation
	// counts seq-independent, which the alloc test enforces.
	if units < 2 || perUnit < minUnitWork {
		matMulSIMDCols(dst, a, bT, m, k, n, 0, n)
		return
	}
	ParallelFor(units, func(u int) {
		i0 := (u / colBlocks) * blockM
		i1 := min(i0+blockM, m)
		jLo := (u % colBlocks) * parallelCols
		jHi := min(jLo+parallelCols, n)
		matMulSIMDCols(dst[i0*n:i1*n], a[i0*k:i1*k], bT, i1-i0, k, n, jLo, jHi)
	})
}

// AddBias adds bias (length n) to each of the m rows of x.
func AddBias(x, bias []float32, m, n int) {
	for i := range m {
		row := x[i*n : i*n+n]
		for j := range row {
			row[j] += bias[j]
		}
	}
}

// Add accumulates src into dst element-wise (residual connections).
func Add(dst, src []float32) {
	for i := range dst {
		dst[i] += src[i]
	}
}

// Softmax normalizes each of the m rows of x (length n) in place, with
// max-subtraction for numerical stability.
func Softmax(x []float32, m, n int) {
	for i := range m {
		row := x[i*n : i*n+n]
		maxv := row[0]
		for _, v := range row[1:] {
			if v > maxv {
				maxv = v
			}
		}
		// The exp is written flat in the loop (expNonPos exceeds the inline
		// budget — cost 99 vs 80 — and a call per element stops the OoO
		// core from overlapping consecutive elements' chains).
		var sum float32
		for j, v := range row {
			xe := v - maxv // <= 0 by construction
			if xe < -87.33 {
				xe = -87.33
			}
			nf := float32(int32(xe*expLog2e - 0.5))
			xe -= nf * expC1
			xe -= nf * expC2
			z := xe * xe
			y := float32(1.9875691500e-4)
			y = y*xe + 1.3981999507e-3
			y = y*xe + 8.3334519073e-3
			y = y*xe + 4.1665795894e-2
			y = y*xe + 1.6666665459e-1
			y = y*xe + 5.0000001201e-1
			e := (y*z + xe + 1) * math.Float32frombits(uint32(int32(nf)+127)<<23)
			row[j] = e
			sum += e
		}
		inv := 1 / sum
		for j := range row {
			row[j] *= inv
		}
	}
}

// LayerNorm normalizes each of the m rows of x (length n) in place:
// (x-mean)/sqrt(variance+eps) * gamma + beta, with biased variance, matching
// torch.nn.LayerNorm.
func LayerNorm(x, gamma, beta []float32, m, n int, eps float32) {
	for i := range m {
		row := x[i*n : i*n+n]
		var mean float32
		for _, v := range row {
			mean += v
		}
		mean /= float32(n)
		var variance float32
		for _, v := range row {
			d := v - mean
			variance += d * d
		}
		variance /= float32(n)
		inv := 1 / float32(math.Sqrt(float64(variance+eps)))
		for j, v := range row {
			row[j] = (v-mean)*inv*gamma[j] + beta[j]
		}
	}
}

// GELU applies the erf-based Gaussian Error Linear Unit in place, matching
// HuggingFace's "gelu" activation: 0.5x(1+erf(x/√2)), with the float32 erf
// (~3e-7 absolute error — see fastmath.go; the golden tolerance is 1e-4).
// The erf is written flat in the loop body (|x| fold via multiply, A&S
// polynomial, inlinable exp) rather than calling erff32: a function call
// per element blocks the out-of-order core from overlapping consecutive
// elements' division/exp latency chains, and this loop runs on seq×I
// elements per layer.
func GELU(x []float32) {
	const invSqrt2 = float32(0.7071067811865476)
	for i, v := range x {
		a := v * invSqrt2
		sign := float32(1)
		if a < 0 {
			sign = -1
			a = -a
		}
		t := 1 / (1 + 0.3275911*a)
		poly := t * (0.254829592 + t*(-0.284496736+t*(1.421413741+t*(-1.453152027+t*1.061405429))))
		// exp(-a²), flat (see Softmax for why this is not a call).
		xe := -a * a
		if xe < -87.33 {
			xe = -87.33
		}
		nf := float32(int32(xe*expLog2e - 0.5))
		xe -= nf * expC1
		xe -= nf * expC2
		z := xe * xe
		y := float32(1.9875691500e-4)
		y = y*xe + 1.3981999507e-3
		y = y*xe + 8.3334519073e-3
		y = y*xe + 4.1665795894e-2
		y = y*xe + 1.6666665459e-1
		y = y*xe + 5.0000001201e-1
		e := (y*z + xe + 1) * math.Float32frombits(uint32(int32(nf)+127)<<23)
		erf := sign * (1 - poly*e)
		x[i] = 0.5 * v * (1 + erf)
	}
}

// L2Normalize scales x to unit Euclidean norm in place. A zero vector is
// left unchanged.
func L2Normalize(x []float32) {
	var sum float64
	for _, v := range x {
		sum += float64(v) * float64(v)
	}
	if sum == 0 {
		return
	}
	inv := float32(1 / math.Sqrt(sum))
	for i := range x {
		x[i] *= inv
	}
}
