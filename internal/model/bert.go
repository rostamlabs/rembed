// SPDX-License-Identifier: Apache-2.0

// Package model implements the BERT-family encoder forward pass:
// embeddings(token+position+segment) → N layers (self-attention + FFN,
// post-LayerNorm) → mean pooling → L2 normalize. MPNet shares the entire
// pipeline with two deltas: positions are offset by pad_token_id+1 and
// there is no segment embedding, and every layer's attention scores add a
// shared bucketed relative-position bias before softmax. ModernBERT
// (rotary positions, pre-norm bias-free LayerNorms, GeGLU, sliding-window
// local attention) and Qwen3 (a causal decoder embedder: RMSNorm, QK-norm,
// grouped-query attention, SwiGLU, last-token pooling) are each different
// enough to have their own load + forward path — modernbert.go and
// qwen3.go — reusing this package's kernels and scratch pool.
package model

import (
	"fmt"
	"io"
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
	packed   *tensor.PackedB
	packed8  *tensor.PackedB8
	packed8v *tensor.PackedB8V // VNNI u8·s8 path (weights AND activations int8)
	raw      []float32
	bias     []float32
	in, out  int
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

// Model is a loaded BERT-family encoder ready to embed token-id
// sequences. It is safe for concurrent use: the matmul kernel is bound
// once at Load, and per-call scratch comes from an internal pool.
type Model struct {
	cfg     Config
	workers int // fan-out cap per Forward; 0 = GOMAXPROCS
	posOff  int // position-embedding row offset (0 for BERT, pad+1 for MPNet)

	wordEmb []float32 // [vocab×H]
	posEmb  []float32 // [maxPos×H]
	typeEmb []float32 // [types×H]; segment 0 is the only one used; nil for MPNet
	embLNg  []float32 // [H]
	embLNb  []float32 // [H]

	// relBias is MPNet's relative-position bias table [buckets×heads],
	// shared by every layer; nil for BERT.
	relBias []float32

	layers []layer

	// ModernBERT path (cfg.ModelType == "modernbert"); nil otherwise.
	// wordEmb doubles as the token embedding table; there are no position
	// or segment tables. embNormG normalizes the embeddings; finalNormG
	// normalizes the stack output; zeroBeta is the shared all-zeros beta
	// for the bias-free LayerNorms.
	mbLayers   []mbLayer
	embNormG   []float32
	finalNormG []float32
	zeroBeta   []float32

	// Qwen3 path (cfg.ModelType == "qwen3"); nil otherwise. A causal
	// decoder embedder — see qwen3.go. wordEmb is embed_tokens; finalNormG
	// is the trailing RMSNorm (reused). There is no embedding norm.
	qwLayers []qwLayer

	// pack is the mmapped disk-weights file when the model was loaded with
	// WithDiskWeights (nil otherwise); Close unmaps it. Weight slices alias
	// this mapping, so it must outlive every Forward.
	pack io.Closer

	scratchPool sync.Pool // *scratch, buffers grown on demand
}

// Close releases resources held by the model — currently the mmapped
// disk-weights file, if any. After Close, the model must not be used.
// Models loaded fully into RAM need no Close (it is a safe no-op).
func (m *Model) Close() error {
	if m.pack != nil {
		err := m.pack.Close()
		m.pack = nil
		return err
	}
	return nil
}

// relMaxDistance is the bucketing horizon of MPNet's relative-position
// bias. HF hardcodes it (relative_position_bucket's default; it is not in
// config.json), so rembed does too.
const relMaxDistance = 128

// relPosBucket maps a relative position (memory j − context i) to a bias
// bucket — HF's MPNetEncoder.relative_position_bucket with its n =
// −relative_position sign flip folded in. Distances up to maxExact-1 get
// their own bucket; larger ones share logarithmically-spaced buckets up to
// relMaxDistance; the sign selects the table's two halves. HF computes the
// log in float32 — float64 here picks identical buckets for every
// |distance| ≤ 1000 (checked exhaustively), far past the 512-position max.
func relPosBucket(rel, numBuckets int) int {
	n := -rel
	half := numBuckets / 2
	ret := 0
	if n < 0 {
		ret = half
		n = -n
	}
	maxExact := half / 2
	if n < maxExact {
		return ret + n
	}
	v := maxExact + int(math.Log(float64(n)/float64(maxExact))/
		math.Log(float64(relMaxDistance)/float64(maxExact))*float64(half-maxExact))
	return ret + min(v, half-1)
}

// scratch holds every intermediate buffer one Forward call needs. Buffers
// keep their capacity between uses (via the pool), so steady-state forward
// passes allocate nothing but the returned vector.
//
// Buffers only grow: one max-length (512-token) input inflates a scratch to
// ~25 MB (the per-head scores panel dominates), and sync.Pool retains up to
// one scratch PER P until GC drains it — worst case ~25 MB × GOMAXPROCS
// (~500 MB on a 20-core box) after a burst of concurrent max-length embeds.
// A single batched Embed call can inflate up to min(GOMAXPROCS, workers)
// scratches at once (the across-texts fan-out), so the ×P worst case no
// longer requires the CALLER to be concurrent.
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
	biasDelta  []float32    // [heads×(2seq−1)] MPNet rel-pos bias by (j−i); nil-length for BERT
	qact       []uint8      // [mPad×kgMax·4] VNNI-quantized activations
	ascales    []float32    // [mPad] per-row activation scales

	// ModernBERT-only buffers (pre-norm sublayer input, GeGLU projection
	// and its gated activation, and the RoPE cos/sin tables for the global
	// and local thetas). Unused (nil-length) on the BERT path.
	normed     []float32 // [seq×H]
	wiOut      []float32 // [mPad×2I]
	geglu      []float32 // [seq×I]
	cosG, sinG []float32 // [seq×dh/2] global-theta RoPE table
	cosL, sinL []float32 // [seq×dh/2] local-theta RoPE table

	// Qwen3-only buffers. q/k/v are the separate (non-fused) projections;
	// kHead/vHeadT are the per-kv-head normed+RoPE'd K and Vᵀ (computed once
	// and shared across a GQA group); qHead/cHead are per-q-head scratch;
	// swUp holds the SwiGLU up-projection.
	qwQ, qwK, qwV     []float32 // [mPad×qDim], [mPad×kvDim], [mPad×kvDim]
	qwKHead, qwVHeadT []float32 // [nkv×seq×dh], [nkv×dh×seq]
	qwQHead, qwCHead  []float32 // [nq×seq×dh]
	qwScores          []float32 // [nq×seq×seq]
	qwCtx             []float32 // [seq×qDim]
	qwUp              []float32 // [mPad×I]
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
	kgMax := (max(H, I) + 3) / 4
	if cap(s.qact) < mPad*kgMax*4 {
		s.qact = make([]uint8, mPad*kgMax*4)
	}
	s.qact = s.qact[:mPad*kgMax*4]
	s.ascales = grow(s.ascales, mPad)
}

// QuantMode selects how a model's dense weights (and optionally
// activations) are quantized at load.
type QuantMode int

const (
	QuantNone    QuantMode = iota
	QuantWeights           // weight-only int8, fp32 activations (M5)
	QuantFull              // int8 weights + u8 activations on VNNI (R7)
)

// newDense builds a denseWeight, packing eagerly where the SIMD gemm can
// consume it and keeping the raw bT layout otherwise. Only one layout is
// ever retained. QuantWeights selects weight-only int8 (per-channel
// symmetric; activations stay float32): 4× less weight traffic on a pass
// bound by streaming weights, at the cost of the weights' 8-bit rounding.
// QuantFull additionally quantizes activations per row to u8 at matmul
// time, letting VPDPBUSD do 4 multiply-accumulates per lane per
// instruction — falling back to QuantWeights when the CPU lacks both
// VNNI encodings (AVX-VNNI and AVX-512-VNNI).
func newDense(raw, bias []float32, in, out int, quantize QuantMode) denseWeight {
	if quantize == QuantFull {
		if pb, err := tensor.PackB8VNNI(raw, in, out); err == nil {
			return denseWeight{packed8v: pb, bias: bias, in: in, out: out}
		}
	}
	if quantize >= QuantWeights {
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
	if w.packed8v != nil {
		mPad := tensor.PackAPad(seq)
		kg := (w.in + 3) / 4
		tensor.MatMulPackedVNNI(dst, x, w.packed8v, seq, s.qact[:mPad*kg*4], s.ascales[:mPad], s.pool)
	} else if w.packed8 != nil {
		tensor.MatMulPacked8(dst, x, w.packed8, seq, s.aPack[:tensor.PackAPad(seq)*w.in], s.pool)
	} else if w.packed != nil {
		tensor.MatMulPacked(dst, x, w.packed, seq, s.aPack[:tensor.PackAPad(seq)*w.in], s.pool)
	} else {
		// The unpacked fallback must honor THIS call's cap too (the review
		// caught the batch path nesting GOMAXPROCS-wide matmul fan-outs
		// inside the across-texts fan-out on non-packed targets).
		tensor.MatMulWorkers(dst, x, w.raw, seq, w.in, w.out, s.fanout)
	}
	// ModernBERT's linear layers are bias-free; nil bias means "no add".
	if w.bias != nil {
		tensor.AddBias(dst, w.bias, seq, w.out)
	}
}

// Load builds a Model from a safetensors file and a validated Config.
// Tensor names follow HuggingFace BertModel/MPNetModel conventions, with
// or without a leading "bert."/"mpnet." prefix. quantize selects the int8
// mode (see newDense).
func Load(weightsPath string, cfg Config, quantize QuantMode, workers int) (*Model, error) {
	tensors, err := safetensors.Load(weightsPath)
	if err != nil {
		return nil, err
	}
	if cfg.ModelType == "modernbert" {
		// ModernBERT diverges enough (RoPE, pre-norm, GeGLU, sliding-window
		// attention, bias-free, no position/segment tables) to warrant a
		// dedicated load + forward path — see modernbert.go.
		return loadModernBERT(tensors, cfg, quantize, workers, weightsPath)
	}
	if cfg.ModelType == "qwen3" {
		// Qwen3: a causal decoder embedder (RMSNorm, QK-norm, GQA, RoPE,
		// SwiGLU, last-token pooling) — dedicated path in qwen3.go.
		return loadQwen3(tensors, cfg, quantize, workers, weightsPath)
	}
	prefix := ""
	for _, p := range []string{"bert.", "roberta.", "mpnet.", "distilbert."} {
		if _, ok := tensors[p+"embeddings.word_embeddings.weight"]; ok {
			prefix = p
		}
	}
	// RoBERTa loads through the BERT branch untouched: identical tensor
	// names, a [1×H] token_type table (row 0, the only one, is used), and
	// the position offset already comes from cfg.PositionOffset().
	mpnet := cfg.ModelType == "mpnet"
	distil := cfg.ModelType == "distilbert"
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
	m := &Model{cfg: cfg, workers: workers, posOff: cfg.PositionOffset()}
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
		if distil {
			p = fmt.Sprintf("transformer.layer.%d.", i)
		}
		var raw struct {
			qW, qB, kW, kB, vW, vB []float32
			attnOutW, attnOutB     []float32
			ffn1W, ffn1B           []float32
			ffn2W, ffn2B           []float32
		}
		// MPNet names its attention projections attn.{q,k,v,o} and hangs
		// the post-attention LayerNorm directly off `attention.`; BERT
		// spells them out under attention.self / attention.output. The FFN
		// halves share names.
		attn := []load{
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
		}
		if distil {
			// DistilBERT: same post-LN flow, renamed everything.
			attn = []load{
				{&raw.qW, p + "attention.q_lin.weight", []int{H, H}},
				{&raw.qB, p + "attention.q_lin.bias", []int{H}},
				{&raw.kW, p + "attention.k_lin.weight", []int{H, H}},
				{&raw.kB, p + "attention.k_lin.bias", []int{H}},
				{&raw.vW, p + "attention.v_lin.weight", []int{H, H}},
				{&raw.vB, p + "attention.v_lin.bias", []int{H}},
				{&raw.attnOutW, p + "attention.out_lin.weight", []int{H, H}},
				{&raw.attnOutB, p + "attention.out_lin.bias", []int{H}},
				{&l.attnLNg, p + "sa_layer_norm.weight", []int{H}},
				{&l.attnLNb, p + "sa_layer_norm.bias", []int{H}},
			}
		}
		if mpnet {
			attn = []load{
				{&raw.qW, p + "attention.attn.q.weight", []int{H, H}},
				{&raw.qB, p + "attention.attn.q.bias", []int{H}},
				{&raw.kW, p + "attention.attn.k.weight", []int{H, H}},
				{&raw.kB, p + "attention.attn.k.bias", []int{H}},
				{&raw.vW, p + "attention.attn.v.weight", []int{H, H}},
				{&raw.vB, p + "attention.attn.v.bias", []int{H}},
				{&raw.attnOutW, p + "attention.attn.o.weight", []int{H, H}},
				{&raw.attnOutB, p + "attention.attn.o.bias", []int{H}},
				{&l.attnLNg, p + "attention.LayerNorm.weight", []int{H}},
				{&l.attnLNb, p + "attention.LayerNorm.bias", []int{H}},
			}
		}
		ffn := []load{
			{&raw.ffn1W, p + "intermediate.dense.weight", []int{I, H}},
			{&raw.ffn1B, p + "intermediate.dense.bias", []int{I}},
			{&raw.ffn2W, p + "output.dense.weight", []int{H, I}},
			{&raw.ffn2B, p + "output.dense.bias", []int{H}},
			{&l.outLNg, p + "output.LayerNorm.weight", []int{H}},
			{&l.outLNb, p + "output.LayerNorm.bias", []int{H}},
		}
		if distil {
			ffn = []load{
				{&raw.ffn1W, p + "ffn.lin1.weight", []int{I, H}},
				{&raw.ffn1B, p + "ffn.lin1.bias", []int{I}},
				{&raw.ffn2W, p + "ffn.lin2.weight", []int{H, I}},
				{&raw.ffn2B, p + "ffn.lin2.bias", []int{H}},
				{&l.outLNg, p + "output_layer_norm.weight", []int{H}},
				{&l.outLNb, p + "output_layer_norm.bias", []int{H}},
			}
		}
		for _, ld := range append(attn, ffn...) {
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

	if distil {
		// DistilBERT has no segment embedding and no extras: done.
		return m, nil
	}
	if mpnet {
		// MPNet has no segment embedding; instead every layer shares one
		// bucketed relative-position bias table over the heads.
		relBias, err := get("encoder.relative_attention_bias.weight",
			cfg.RelativeAttentionNumBuckets, cfg.NumAttentionHeads)
		if err != nil {
			return nil, err
		}
		m.relBias = relBias
		return m, nil
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

// Workers returns the configured fan-out cap (0 = GOMAXPROCS).
func (m *Model) Workers() int { return m.workers }

// Quantized reports whether EVERY dense weight actually took an int8
// path (weight-only or full). Quantization can silently fall back
// per-matrix (no AVX2, or an out-dim not divisible by 16) — callers who
// requested int8 can check the mode took effect instead of discovering
// fp32 memory and speed later.
func (m *Model) Quantized() bool {
	if m.qwLayers != nil {
		for i := range m.qwLayers {
			l := &m.qwLayers[i]
			for _, w := range []*denseWeight{&l.qProj, &l.kProj, &l.vProj, &l.oProj, &l.gateProj, &l.upProj, &l.downProj} {
				if w.packed8 == nil && w.packed8v == nil {
					return false
				}
			}
		}
		return len(m.qwLayers) > 0
	}
	if m.mbLayers != nil {
		for i := range m.mbLayers {
			l := &m.mbLayers[i]
			for _, w := range []*denseWeight{&l.qkv, &l.attnOut, &l.wi, &l.mlpOut} {
				if w.packed8 == nil && w.packed8v == nil {
					return false
				}
			}
		}
		return len(m.mbLayers) > 0
	}
	for i := range m.layers {
		l := &m.layers[i]
		for _, w := range []*denseWeight{&l.qkv, &l.attnOut, &l.ffn1, &l.ffn2} {
			if w.packed8 == nil && w.packed8v == nil {
				return false
			}
		}
	}
	return len(m.layers) > 0
}

// QuantizedActivations reports whether EVERY dense weight took the VNNI
// u8-activation path (QuantFull requested AND AVX-VNNI present AND every
// shape packed).
func (m *Model) QuantizedActivations() bool {
	if m.qwLayers != nil {
		for i := range m.qwLayers {
			l := &m.qwLayers[i]
			for _, w := range []*denseWeight{&l.qProj, &l.kProj, &l.vProj, &l.oProj, &l.gateProj, &l.upProj, &l.downProj} {
				if w.packed8v == nil {
					return false
				}
			}
		}
		return len(m.qwLayers) > 0
	}
	if m.mbLayers != nil {
		for i := range m.mbLayers {
			l := &m.mbLayers[i]
			for _, w := range []*denseWeight{&l.qkv, &l.attnOut, &l.wi, &l.mlpOut} {
				if w.packed8v == nil {
					return false
				}
			}
		}
		return len(m.mbLayers) > 0
	}
	for i := range m.layers {
		l := &m.layers[i]
		for _, w := range []*denseWeight{&l.qkv, &l.attnOut, &l.ffn1, &l.ffn2} {
			if w.packed8v == nil {
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
	return m.ForwardWorkers(ids, m.workers)
}

// ForwardWorkers is Forward with an explicit per-call worker cap,
// overriding the model default. workers=1 is fully serial — the batched
// Embed path runs one serial forward per text so the parallelism lives
// ACROSS texts with zero fan-out coordination inside each.
func (m *Model) ForwardWorkers(ids []int64, workers int) ([]float32, error) {
	s, err := m.encodeWorkers(ids, workers)
	if err != nil {
		return nil, err
	}
	defer m.scratchPool.Put(s)
	seq, H := len(ids), m.cfg.HiddenSize
	x := s.x

	// Pooling (no padding ⇒ every position counts), then optional L2
	// normalization. cls takes the first token's hidden state (BGE-style
	// models); mean averages all positions (sentence-transformers style).
	pooled := make([]float32, H)
	switch m.cfg.Pooling {
	case "cls":
		copy(pooled, x[:H])
	case "lasttoken":
		// Qwen3: the final token's hidden state (the appended <|endoftext|>).
		// No padding, so it is the last row.
		copy(pooled, x[(seq-1)*H:seq*H])
	default:
		poolMean(pooled, x, seq, H)
	}
	if m.cfg.Normalize {
		tensor.L2Normalize(pooled)
	}
	return pooled, nil
}

// poolMean averages all seq rows of x[seq×H] into pooled[H].
func poolMean(pooled, x []float32, seq, H int) {
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

// ForwardTokens returns the final-layer hidden state for every token —
// ONNX Runtime's last_hidden_state — as a fresh [seq][H] matrix. No
// pooling and no normalization are applied: this is the raw material for
// rerankers, late-interaction retrieval, and custom pooling.
func (m *Model) ForwardTokens(ids []int64) ([][]float32, error) {
	return m.ForwardTokensWorkers(ids, m.workers)
}

// ForwardTokensWorkers is ForwardTokens with an explicit per-call worker
// cap (the batch path spreads texts across workers).
func (m *Model) ForwardTokensWorkers(ids []int64, workers int) ([][]float32, error) {
	s, err := m.encodeWorkers(ids, workers)
	if err != nil {
		return nil, err
	}
	defer m.scratchPool.Put(s)
	seq, H := len(ids), m.cfg.HiddenSize
	out := make([][]float32, seq)
	flat := make([]float32, seq*H)
	copy(flat, s.x[:seq*H])
	for i := range out {
		// Three-index slice: without the cap clamp, append on row i would
		// silently overwrite row i+1 (rows share one backing array).
		out[i] = flat[i*H : (i+1)*H : (i+1)*H]
	}
	return out, nil
}

// encode runs the transformer stack, leaving the final hidden states in
// the returned scratch's x[:seq*H]. The CALLER returns the scratch to the
// pool once done reading.
func (m *Model) encode(ids []int64) (*scratch, error) {
	return m.encodeWorkers(ids, m.workers)
}

func (m *Model) encodeWorkers(ids []int64, workers int) (*scratch, error) {
	seq := len(ids)
	if seq == 0 {
		return nil, fmt.Errorf("empty token sequence")
	}
	if seq > m.cfg.MaxSeqLen() {
		return nil, fmt.Errorf("sequence length %d exceeds the model's maximum %d", seq, m.cfg.MaxSeqLen())
	}
	if m.mbLayers != nil {
		return m.encodeModernBERT(ids, workers)
	}
	if m.qwLayers != nil {
		return m.encodeQwen3(ids, workers)
	}
	H := m.cfg.HiddenSize
	heads := m.cfg.NumAttentionHeads
	dh := H / heads
	I := m.cfg.IntermediateSize
	eps := m.cfg.LayerNormEps

	s := m.scratchPool.Get().(*scratch)
	// encode does NOT return s to the pool on success — the caller reads
	// s.x after this returns and is responsible for scratchPool.Put. On
	// error OR panic this deferred reclaim runs; it is registered BEFORE
	// the pool's Stop defer, so LIFO order guarantees the spin pool is
	// stopped before the scratch is published back to the pool.
	committed := false
	defer func() {
		if !committed {
			m.scratchPool.Put(s)
		}
	}()
	s.resize(seq, H, I, dh)
	// The fan-out workers live in a spinning pool for the duration of this
	// call — spawned once, never parked between the ~36 fan-outs — because
	// per-fan-out goroutine wake latency was ~half the short-sequence
	// forward pass. With coordination that cheap, the full machine wins at
	// every seq (the earlier seq-scaled cap was compensating for wake
	// latency, not for parallelism itself).
	s.fanout = runtime.GOMAXPROCS(0)
	if workers > 0 {
		s.fanout = min(s.fanout, workers)
	}
	s.pool = tensor.NewPool(s.fanout - 1)
	defer s.pool.Stop()

	// Embeddings: word + position (+ segment 0 for BERT; MPNet has no
	// segment table and offsets positions past its padding rows), then
	// LayerNorm.
	x := s.x
	for i, id := range ids {
		if id < 0 || int(id) >= m.cfg.VocabSize {
			return nil, fmt.Errorf("token id %d out of vocab range %d", id, m.cfg.VocabSize)
		}
		row := x[i*H : i*H+H]
		w := m.wordEmb[int(id)*H : int(id)*H+H]
		p := m.posEmb[(i+m.posOff)*H : (i+m.posOff)*H+H]
		if m.typeEmb != nil {
			t := m.typeEmb[:H]
			for j := range row {
				row[j] = w[j] + p[j] + t[j]
			}
		} else {
			for j := range row {
				row[j] = w[j] + p[j]
			}
		}
	}
	tensor.LayerNorm(x, m.embLNg, m.embLNb, seq, H, eps)

	// MPNet's relative-position bias depends only on j−i (positions are
	// contiguous), so one [heads×(2seq−1)] delta table — computed once per
	// forward, shared by every layer — replaces the [heads×seq×seq] tensor
	// HF materializes.
	if m.relBias != nil {
		nDelta := 2*seq - 1
		s.biasDelta = grow(s.biasDelta, heads*nDelta)
		nb := m.cfg.RelativeAttentionNumBuckets
		for d := -(seq - 1); d <= seq-1; d++ {
			b := relPosBucket(d, nb)
			for h := range heads {
				s.biasDelta[h*nDelta+d+seq-1] = m.relBias[b*heads+h]
			}
		}
	}

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
			if m.relBias != nil {
				// MPNet: scores/√dh + bias(j−i), matching HF's order
				// (scale first, then add). The explicit float32 conversion
				// forces the product to round before the add — without it
				// the compiler fuses an FMA on arm64, and results stop
				// being bit-identical across architectures.
				bd := s.biasDelta[h*(2*seq-1):]
				for i := range seq {
					row := sc[i*seq : i*seq+seq]
					bdi := bd[seq-1-i:] // index j-i+seq-1 = (seq-1-i)+j
					for j := range row {
						row[j] = float32(row[j]*scale) + bdi[j]
					}
				}
			} else {
				for i := range sc {
					sc[i] *= scale
				}
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

	committed = true
	return s, nil
}
