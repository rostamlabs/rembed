// SPDX-License-Identifier: Apache-2.0

// Package model implements the BERT encoder forward pass:
// embeddings(token+position+segment) → N layers (self-attention + FFN,
// post-LayerNorm) → mean pooling → L2 normalize.
package model

import (
	"fmt"
	"math"
	"runtime"
	"sync"

	"github.com/rostamlabs/rembed/internal/safetensors"
	"github.com/rostamlabs/rembed/internal/tensor"
)

// denseWeight is one linear layer's weights. Where the SIMD gemm kernel is
// available (and out%16==0, true for all BERT dims) the weight is
// pre-packed ONCE at load into the panel layout the kernel streams — the
// raw slice is dropped so weights are never held twice. Otherwise raw keeps
// HuggingFace's [out, in] layout, exactly the bT operand of MatMulFunc.
type denseWeight struct {
	packed  *tensor.PackedB
	packed8 *tensor.PackedB8
	raw     []float32
	bias    []float32
	in, out int
}

// layer holds one encoder layer's weights. Q/K/V are fused into a single
// [3H × H] projection: one matmul instead of three (fewer fan-outs, bigger
// parallel units), with the head repack reading q/k/v at row offsets 0, H,
// 2H of the fused output.
type layer struct {
	qkv              denseWeight // [3H×H]
	attnOut          denseWeight // [H×H]
	attnLNg, attnLNb []float32   // [H]
	ffn1             denseWeight // [I×H]
	ffn2             denseWeight // [H×I]
	outLNg, outLNb   []float32   // [H]
}

// Model is a loaded BERT encoder ready to embed token-id sequences.
// It is safe for concurrent use: the matmul kernel is bound once at Load,
// and per-call scratch comes from an internal pool.
type Model struct {
	cfg     Config
	matmul  tensor.MatMulFunc
	workers int // fan-out cap per Forward; 0 = GOMAXPROCS

	wordEmb []float32 // [vocab×H]
	posEmb  []float32 // [maxPos×H]
	typeEmb []float32 // [types×H]; segment 0 is the only one used
	embLNg  []float32 // [H]
	embLNb  []float32 // [H]

	layers []layer

	scratchPool sync.Pool // *scratch, buffers grown on demand
}

// scratch holds every intermediate buffer one Forward call needs. Buffers
// keep their capacity between uses (via the pool), so steady-state forward
// passes allocate nothing but the returned vector.
//
// Buffers only grow: one max-length (512-token) input inflates a scratch to
// ~25 MB (the per-head scores panel dominates), and sync.Pool retains up to
// one scratch PER P until GC drains it — worst case ~25 MB × GOMAXPROCS
// (~500 MB on a 20-core box) after a burst of concurrent max-length embeds.
// That is a deliberate trade (reuse over footprint); if it ever bites,
// scores is the buffer to shrink (a worker-slot ParallelFor variant would
// cut it from heads·seq² to min(heads, workers)·seq²).
// Matmul destination buffers (qkv, attnOut, ffnHidden, ffnOut) and aPack
// are sized for tensor.PackAPad(seq) rows: the packed gemm kernel writes
// whole 4-row tiles, so up to 3 pad rows receive zeros and are never read.
type scratch struct {
	fanout     int          // worker count for this call's fan-outs (scales with seq)
	pool       *tensor.Pool // spinning fork-join pool, one Forward's lifetime
	x          []float32    // [seq×H]
	qkv        []float32    // [mPad×3H] fused q‖k‖v projection output
	ctxOut     []float32    // [seq×H]
	attnOut    []float32    // [mPad×H]
	ffnOut     []float32    // [mPad×H]
	ffnHidden  []float32    // [mPad×I]
	aPack      []float32    // [mPad×max(H,I)] packed-A panel for MatMulPacked
	scores     []float32    // [heads×seq×seq]
	qh, kh, ch []float32    // [heads][seq×dh] = [seq×H]
	vhT        []float32    // [heads][dh×seq] = [H×seq]
}

// grow reslices buf to n floats, reallocating only when capacity is short.
func grow(buf []float32, n int) []float32 {
	if cap(buf) < n {
		return make([]float32, n)
	}
	return buf[:n]
}

func (s *scratch) resize(seq, H, I, dh int) {
	heads := H / dh
	mPad := tensor.PackAPad(seq)
	s.x = grow(s.x, seq*H)
	s.qkv = grow(s.qkv, mPad*3*H)
	s.ctxOut = grow(s.ctxOut, seq*H)
	s.attnOut = grow(s.attnOut, mPad*H)
	s.ffnOut = grow(s.ffnOut, mPad*H)
	s.ffnHidden = grow(s.ffnHidden, mPad*I)
	s.aPack = grow(s.aPack, mPad*max(H, I))
	s.scores = grow(s.scores, heads*seq*seq)
	s.qh = grow(s.qh, seq*H)
	s.kh = grow(s.kh, seq*H)
	s.ch = grow(s.ch, seq*H)
	s.vhT = grow(s.vhT, H*seq)
}

// newDense builds a denseWeight, packing eagerly where the SIMD gemm can
// consume it and keeping the raw bT layout otherwise. Only one layout is
// ever retained. quantize selects weight-only int8 (per-channel symmetric;
// activations stay float32): 4× less weight traffic on a pass bound by
// streaming weights, at the cost of the weights' 8-bit rounding.
func newDense(raw, bias []float32, in, out int, quantize bool) denseWeight {
	if quantize {
		if pb, err := tensor.PackB8(raw, in, out); err == nil {
			return denseWeight{packed8: pb, bias: bias, in: in, out: out}
		}
	}
	if pb, err := tensor.PackB(raw, in, out); err == nil {
		return denseWeight{packed: pb, bias: bias, in: in, out: out}
	}
	return denseWeight{raw: raw, bias: bias, in: in, out: out}
}

// applyDense computes dst = x·Wᵀ + bias for seq rows. dst must be sized for
// PackAPad(seq) rows (scratch is); x must hold exactly seq×in floats.
func (m *Model) applyDense(dst, x []float32, w *denseWeight, seq int, s *scratch) {
	if w.packed8 != nil {
		tensor.MatMulPacked8(dst, x, w.packed8, seq, s.aPack[:tensor.PackAPad(seq)*w.in], s.pool)
	} else if w.packed != nil {
		tensor.MatMulPacked(dst, x, w.packed, seq, s.aPack[:tensor.PackAPad(seq)*w.in], s.pool)
	} else {
		m.matmul(dst, x, w.raw, seq, w.in, w.out)
	}
	tensor.AddBias(dst, w.bias, seq, w.out)
}

// Load builds a Model from a safetensors file and a validated Config.
// Tensor names follow HuggingFace BertModel conventions, with or without a
// leading "bert." prefix. quantize selects weight-only int8 inference (see
// newDense).
func Load(weightsPath string, cfg Config, quantize bool, workers int) (*Model, error) {
	tensors, err := safetensors.Load(weightsPath)
	if err != nil {
		return nil, err
	}
	prefix := ""
	if _, ok := tensors["bert.embeddings.word_embeddings.weight"]; ok {
		prefix = "bert."
	}
	get := func(name string, wantShape ...int) ([]float32, error) {
		t, ok := tensors[prefix+name]
		if !ok {
			return nil, fmt.Errorf("weights %s: missing tensor %q", weightsPath, prefix+name)
		}
		if len(t.Shape) != len(wantShape) {
			return nil, fmt.Errorf("weights %s: tensor %q has shape %v, want %v", weightsPath, prefix+name, t.Shape, wantShape)
		}
		for i, d := range wantShape {
			if t.Shape[i] != d {
				return nil, fmt.Errorf("weights %s: tensor %q has shape %v, want %v", weightsPath, prefix+name, t.Shape, wantShape)
			}
		}
		return t.Data, nil
	}

	H, I := cfg.HiddenSize, cfg.IntermediateSize
	// LoadConfig enforces this, but Forward's per-head writes cover ctxOut
	// completely ONLY when it holds — and with pooled scratch a violation
	// would read the previous call's residue, not zeros. Re-assert here so an
	// unvalidated Config cannot reach that state.
	if cfg.NumAttentionHeads <= 0 || H%cfg.NumAttentionHeads != 0 {
		return nil, fmt.Errorf("weights %s: hidden_size %d not divisible by num_attention_heads %d", weightsPath, H, cfg.NumAttentionHeads)
	}
	m := &Model{cfg: cfg, matmul: tensor.DefaultCapped(workers), workers: workers}
	m.scratchPool.New = func() any { return new(scratch) }

	type load struct {
		dst   *[]float32
		name  string
		shape []int
	}
	loads := []load{
		{&m.wordEmb, "embeddings.word_embeddings.weight", []int{cfg.VocabSize, H}},
		{&m.posEmb, "embeddings.position_embeddings.weight", []int{cfg.MaxPositionEmbeddings, H}},
		{&m.embLNg, "embeddings.LayerNorm.weight", []int{H}},
		{&m.embLNb, "embeddings.LayerNorm.bias", []int{H}},
	}
	for _, ld := range loads {
		data, err := get(ld.name, ld.shape...)
		if err != nil {
			return nil, err
		}
		*ld.dst = data
	}

	m.layers = make([]layer, cfg.NumHiddenLayers)
	for i := range m.layers {
		l := &m.layers[i]
		p := fmt.Sprintf("encoder.layer.%d.", i)
		var raw struct {
			qW, qB, kW, kB, vW, vB []float32
			attnOutW, attnOutB     []float32
			ffn1W, ffn1B           []float32
			ffn2W, ffn2B           []float32
		}
		for _, ld := range []load{
			{&raw.qW, p + "attention.self.query.weight", []int{H, H}},
			{&raw.qB, p + "attention.self.query.bias", []int{H}},
			{&raw.kW, p + "attention.self.key.weight", []int{H, H}},
			{&raw.kB, p + "attention.self.key.bias", []int{H}},
			{&raw.vW, p + "attention.self.value.weight", []int{H, H}},
			{&raw.vB, p + "attention.self.value.bias", []int{H}},
			{&raw.attnOutW, p + "attention.output.dense.weight", []int{H, H}},
			{&raw.attnOutB, p + "attention.output.dense.bias", []int{H}},
			{&l.attnLNg, p + "attention.output.LayerNorm.weight", []int{H}},
			{&l.attnLNb, p + "attention.output.LayerNorm.bias", []int{H}},
			{&raw.ffn1W, p + "intermediate.dense.weight", []int{I, H}},
			{&raw.ffn1B, p + "intermediate.dense.bias", []int{I}},
			{&raw.ffn2W, p + "output.dense.weight", []int{H, I}},
			{&raw.ffn2B, p + "output.dense.bias", []int{H}},
			{&l.outLNg, p + "output.LayerNorm.weight", []int{H}},
			{&l.outLNb, p + "output.LayerNorm.bias", []int{H}},
		} {
			data, err := get(ld.name, ld.shape...)
			if err != nil {
				return nil, err
			}
			*ld.dst = data
		}

		// Fuse Q‖K‖V into one [3H×H] projection (rows concatenate cleanly
		// in the bT layout) so a layer does one big matmul, not three.
		qkvW := make([]float32, 0, 3*H*H)
		qkvW = append(append(append(qkvW, raw.qW...), raw.kW...), raw.vW...)
		qkvB := make([]float32, 0, 3*H)
		qkvB = append(append(append(qkvB, raw.qB...), raw.kB...), raw.vB...)
		l.qkv = newDense(qkvW, qkvB, H, 3*H, quantize)
		l.attnOut = newDense(raw.attnOutW, raw.attnOutB, H, H, quantize)
		l.ffn1 = newDense(raw.ffn1W, raw.ffn1B, H, I, quantize)
		l.ffn2 = newDense(raw.ffn2W, raw.ffn2B, I, H, quantize)
	}

	// token_type_embeddings may have any first dimension; only row 0 is used.
	tt, ok := tensors[prefix+"embeddings.token_type_embeddings.weight"]
	if !ok {
		return nil, fmt.Errorf("weights %s: missing tensor %q", weightsPath, prefix+"embeddings.token_type_embeddings.weight")
	}
	if len(tt.Shape) != 2 || tt.Shape[1] != H || tt.Shape[0] < 1 {
		return nil, fmt.Errorf("weights %s: token_type_embeddings has shape %v", weightsPath, tt.Shape)
	}
	m.typeEmb = tt.Data

	return m, nil
}

// Config returns the manifest the model was loaded with.
func (m *Model) Config() Config { return m.cfg }

// Quantized reports whether EVERY dense weight actually took the int8
// path. Quantization can silently fall back per-matrix (no AVX2, or an
// out-dim not divisible by 16) — callers who requested int8 can check the
// mode took effect instead of discovering fp32 memory and speed later.
func (m *Model) Quantized() bool {
	for i := range m.layers {
		l := &m.layers[i]
		for _, w := range []*denseWeight{&l.qkv, &l.attnOut, &l.ffn1, &l.ffn2} {
			if w.packed8 == nil {
				return false
			}
		}
	}
	return len(m.layers) > 0
}

// Forward embeds one token-id sequence (no padding: the sequence is exactly
// len(ids) long, so the attention mask is implicit all-ones) and returns the
// pooled sentence vector of length HiddenSize.
func (m *Model) Forward(ids []int64) ([]float32, error) {
	seq := len(ids)
	if seq == 0 {
		return nil, fmt.Errorf("empty token sequence")
	}
	if seq > m.cfg.MaxPositionEmbeddings {
		return nil, fmt.Errorf("sequence length %d exceeds max position embeddings %d", seq, m.cfg.MaxPositionEmbeddings)
	}
	H := m.cfg.HiddenSize
	heads := m.cfg.NumAttentionHeads
	dh := H / heads
	I := m.cfg.IntermediateSize
	eps := m.cfg.LayerNormEps

	s := m.scratchPool.Get().(*scratch)
	defer m.scratchPool.Put(s)
	s.resize(seq, H, I, dh)
	// The fan-out workers live in a spinning pool for the duration of this
	// call — spawned once, never parked between the ~36 fan-outs — because
	// per-fan-out goroutine wake latency was ~half the short-sequence
	// forward pass. With coordination that cheap, the full machine wins at
	// every seq (the earlier seq-scaled cap was compensating for wake
	// latency, not for parallelism itself).
	s.fanout = runtime.GOMAXPROCS(0)
	if m.workers > 0 {
		s.fanout = min(s.fanout, m.workers)
	}
	s.pool = tensor.NewPool(s.fanout - 1)
	defer s.pool.Stop()

	// Embeddings: word + position + segment(0), then LayerNorm.
	x := s.x
	for i, id := range ids {
		if id < 0 || int(id) >= m.cfg.VocabSize {
			return nil, fmt.Errorf("token id %d out of vocab range %d", id, m.cfg.VocabSize)
		}
		row := x[i*H : i*H+H]
		w := m.wordEmb[int(id)*H : int(id)*H+H]
		p := m.posEmb[i*H : i*H+H]
		t := m.typeEmb[:H]
		for j := range row {
			row[j] = w[j] + p[j] + t[j]
		}
	}
	tensor.LayerNorm(x, m.embLNg, m.embLNb, seq, H, eps)

	qkv := s.qkv
	ctxOut, attnOut := s.ctxOut, s.attnOut
	scores := s.scores
	qh, kh, vhT, ch := s.qh, s.kh, s.vhT, s.ch
	ffnHidden, ffnOut := s.ffnHidden, s.ffnOut
	scale := float32(1 / math.Sqrt(float64(dh)))

	for li := range m.layers {
		l := &m.layers[li]

		// Fused Q‖K‖V projection: row i of qkv holds q_i at [0,H),
		// k_i at [H,2H), v_i at [2H,3H).
		m.applyDense(qkv, x[:seq*H], &l.qkv, seq, s)

		// Scaled dot-product attention, heads fanned out in parallel. The
		// whole per-head pipeline — repack, matmuls, softmax, gather — runs
		// inside the worker: repack-then-consume keeps the head's panels
		// cache-hot, and no phase is left serial to bound the speedup.
		// Race-free by construction: worker h reads qkv (written before
		// the fan-out) and writes only its own slices of qh/kh/vhT/ch/scores
		// plus the disjoint ctxOut columns [h·dh, (h+1)·dh) of each row.
		s.pool.Run(heads, func(h int) {
			off := h * dh
			qhH := qh[h*seq*dh : (h+1)*seq*dh]
			khH := kh[h*seq*dh : (h+1)*seq*dh]
			vhTH := vhT[h*dh*seq : (h+1)*dh*seq]
			chH := ch[h*seq*dh : (h+1)*seq*dh]
			sc := scores[h*seq*seq : (h+1)*seq*seq]
			// Repack: qh/kh as [seq×dh] panels, vhT as Vᵀ [dh×seq] so that
			// probs·V fits the single MatMul (bT) signature.
			for i := range seq {
				row := qkv[i*3*H:]
				copy(qhH[i*dh:i*dh+dh], row[off:off+dh])
				copy(khH[i*dh:i*dh+dh], row[H+off:H+off+dh])
				vRow := row[2*H+off:]
				for d := range dh {
					vhTH[d*seq+i] = vRow[d]
				}
			}
			// The heads ARE the parallelism here, so the serial body is
			// deliberate: nesting MatMulParallel would oversubscribe the
			// scheduler and make allocation counts depend on seq.
			// scores = Qh·Khᵀ / √dh; Kh as bT gives exactly Qh·Khᵀ.
			tensor.MatMulSerial(sc, qhH, khH, seq, dh, seq)
			for i := range sc {
				sc[i] *= scale
			}
			tensor.Softmax(sc, seq, seq)
			// ch = probs·Vh, with Vhᵀ as the bT operand; then gather into
			// this head's ctxOut columns.
			tensor.MatMulSerial(chH, sc, vhTH, seq, seq, dh)
			for i := range seq {
				copy(ctxOut[i*H+off:i*H+off+dh], chH[i*dh:i*dh+dh])
			}
		})

		// Attention output projection + residual + LayerNorm.
		m.applyDense(attnOut, ctxOut[:seq*H], &l.attnOut, seq, s)
		tensor.Add(x, attnOut[:seq*H])
		tensor.LayerNorm(x, l.attnLNg, l.attnLNb, seq, H, eps)

		// FFN: GELU(x·W1ᵀ+b1)·W2ᵀ+b2 + residual + LayerNorm.
		m.applyDense(ffnHidden, x[:seq*H], &l.ffn1, seq, s)
		// GELU's erf is the priciest non-matmul op (seq×I calls); rows fan
		// out with the same fixed-worker ParallelFor.
		s.pool.Run(seq, func(i int) {
			tensor.GELU(ffnHidden[i*I : i*I+I])
		})
		m.applyDense(ffnOut, ffnHidden[:seq*I], &l.ffn2, seq, s)
		tensor.Add(x, ffnOut[:seq*H])
		tensor.LayerNorm(x, l.outLNg, l.outLNb, seq, H, eps)
	}

	// Pooling (no padding ⇒ every position counts), then optional L2
	// normalization. cls takes the first token's hidden state (BGE-style
	// models); mean averages all positions (sentence-transformers style).
	pooled := make([]float32, H)
	if m.cfg.Pooling == "cls" {
		copy(pooled, x[:H])
	} else {
		for i := range seq {
			row := x[i*H : i*H+H]
			for j := range pooled {
				pooled[j] += row[j]
			}
		}
		inv := 1 / float32(seq)
		for j := range pooled {
			pooled[j] *= inv
		}
	}
	if m.cfg.Normalize {
		tensor.L2Normalize(pooled)
	}
	return pooled, nil
}
