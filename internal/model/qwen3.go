// SPDX-License-Identifier: Apache-2.0

package model

import (
	"fmt"
	"math"
	"runtime"

	"github.com/rostamlabs/rembed/internal/safetensors"
	"github.com/rostamlabs/rembed/internal/tensor"
)

// qwLayer holds one Qwen3 decoder layer. Everything is bias-free. q/k/v are
// SEPARATE projections (not fused: k/v are narrower under GQA). qNormG and
// kNormG are the per-head QK-norm RMSNorm weights (length head_dim), applied
// to each head's q/k vector BEFORE RoPE.
type qwLayer struct {
	inNormG  []float32   // input_layernorm (pre-attention RMSNorm) [H]
	qProj    denseWeight // [nq·dh × H]
	kProj    denseWeight // [nkv·dh × H]
	vProj    denseWeight // [nkv·dh × H]
	oProj    denseWeight // [H × nq·dh]
	qNormG   []float32   // [dh]
	kNormG   []float32   // [dh]
	postNorm []float32   // post_attention_layernorm (pre-MLP RMSNorm) [H]
	gateProj denseWeight // [I × H]
	upProj   denseWeight // [I × H]
	downProj denseWeight // [H × I]
}

// loadQwen3 builds a Qwen3 causal-decoder embedder. Tensor names carry no
// prefix (embed_tokens.weight, layers.N.*, norm.weight); weights are tied
// so there is no lm_head, and the LM head is unused for embedding anyway.
func loadQwen3(tensors map[string]safetensors.Tensor, cfg Config, quantize QuantMode, workers int, weightsPath string) (*Model, error) {
	H, I := cfg.HiddenSize, cfg.IntermediateSize
	nq, nkv, dh := cfg.NumAttentionHeads, cfg.NumKeyValueHeads, cfg.HeadDim
	qDim, kvDim := nq*dh, nkv*dh
	get := func(name string, wantShape ...int) ([]float32, error) {
		t, ok := tensors[name]
		if !ok {
			return nil, fmt.Errorf("weights %s: missing tensor %q", weightsPath, name)
		}
		if len(t.Shape) != len(wantShape) {
			return nil, fmt.Errorf("weights %s: tensor %q has shape %v, want %v", weightsPath, name, t.Shape, wantShape)
		}
		for i, d := range wantShape {
			if t.Shape[i] != d {
				return nil, fmt.Errorf("weights %s: tensor %q has shape %v, want %v", weightsPath, name, t.Shape, wantShape)
			}
		}
		return t.Data, nil
	}

	m := &Model{cfg: cfg, workers: workers, posOff: 0}
	m.scratchPool.New = func() any { return new(scratch) }

	var err error
	if m.wordEmb, err = get("embed_tokens.weight", cfg.VocabSize, H); err != nil {
		return nil, err
	}
	if m.finalNormG, err = get("norm.weight", H); err != nil {
		return nil, err
	}

	m.qwLayers = make([]qwLayer, cfg.NumHiddenLayers)
	for i := range m.qwLayers {
		l := &m.qwLayers[i]
		p := fmt.Sprintf("layers.%d.", i)
		var qw, kw, vw, ow, gw, uw, dw []float32
		loads := []struct {
			dst   *[]float32
			name  string
			shape []int
		}{
			{&l.inNormG, p + "input_layernorm.weight", []int{H}},
			{&qw, p + "self_attn.q_proj.weight", []int{qDim, H}},
			{&kw, p + "self_attn.k_proj.weight", []int{kvDim, H}},
			{&vw, p + "self_attn.v_proj.weight", []int{kvDim, H}},
			{&ow, p + "self_attn.o_proj.weight", []int{H, qDim}},
			{&l.qNormG, p + "self_attn.q_norm.weight", []int{dh}},
			{&l.kNormG, p + "self_attn.k_norm.weight", []int{dh}},
			{&l.postNorm, p + "post_attention_layernorm.weight", []int{H}},
			{&gw, p + "mlp.gate_proj.weight", []int{I, H}},
			{&uw, p + "mlp.up_proj.weight", []int{I, H}},
			{&dw, p + "mlp.down_proj.weight", []int{H, I}},
		}
		for _, ld := range loads {
			data, err := get(ld.name, ld.shape...)
			if err != nil {
				return nil, err
			}
			*ld.dst = data
		}
		l.qProj = newDense(qw, nil, H, qDim, quantize)
		l.kProj = newDense(kw, nil, H, kvDim, quantize)
		l.vProj = newDense(vw, nil, H, kvDim, quantize)
		l.oProj = newDense(ow, nil, qDim, H, quantize)
		l.gateProj = newDense(gw, nil, H, I, quantize)
		l.upProj = newDense(uw, nil, H, I, quantize)
		l.downProj = newDense(dw, nil, I, H, quantize)
	}
	return m, nil
}

// encodeQwen3 runs the Qwen3 causal-decoder stack, leaving the final hidden
// states in the returned scratch's x[:seq*H]. Last-token pooling happens in
// ForwardWorkers (reads the final row). Caller returns the scratch.
func (m *Model) encodeQwen3(ids []int64, workers int) (*scratch, error) {
	seq := len(ids)
	H := m.cfg.HiddenSize
	nq, nkv, dh := m.cfg.NumAttentionHeads, m.cfg.NumKeyValueHeads, m.cfg.HeadDim
	qDim, kvDim := nq*dh, nkv*dh
	group := nq / nkv
	half := dh / 2
	I := m.cfg.IntermediateSize
	eps := m.cfg.LayerNormEps

	s := m.scratchPool.Get().(*scratch)
	committed := false
	defer func() {
		if !committed {
			m.scratchPool.Put(s)
		}
	}()

	mPad := tensor.PackAPad(seq)
	s.x = grow(s.x, seq*H)
	s.normed = grow(s.normed, seq*H)
	s.qwQ = grow(s.qwQ, mPad*qDim)
	s.qwK = grow(s.qwK, mPad*kvDim)
	s.qwV = grow(s.qwV, mPad*kvDim)
	s.qwKHead = grow(s.qwKHead, nkv*seq*dh)
	s.qwVHeadT = grow(s.qwVHeadT, nkv*dh*seq)
	s.qwQHead = grow(s.qwQHead, nq*seq*dh)
	s.qwCHead = grow(s.qwCHead, nq*seq*dh)
	s.qwScores = grow(s.qwScores, nq*seq*seq)
	s.qwCtx = grow(s.qwCtx, seq*qDim)
	s.attnOut = grow(s.attnOut, mPad*H)
	s.wiOut = grow(s.wiOut, mPad*I) // SwiGLU gate projection
	s.qwUp = grow(s.qwUp, mPad*I)   // SwiGLU up projection
	s.geglu = grow(s.geglu, seq*I)  // silu(gate)*up
	s.ffnOut = grow(s.ffnOut, mPad*H)
	s.aPack = grow(s.aPack, mPad*max(qDim, I))
	s.cosG, s.sinG = grow(s.cosG, seq*half), grow(s.sinG, seq*half)
	kgMax := (max(qDim, I) + 3) / 4
	if cap(s.qact) < mPad*kgMax*4 {
		s.qact = make([]uint8, mPad*kgMax*4)
	}
	s.qact = s.qact[:mPad*kgMax*4]
	s.ascales = grow(s.ascales, mPad)

	s.fanout = runtime.GOMAXPROCS(0)
	if workers > 0 {
		s.fanout = min(s.fanout, workers)
	}
	s.pool = tensor.NewPool(s.fanout - 1)
	defer s.pool.Stop()

	cos, sin := s.cosG, s.sinG
	ropeTable(cos, sin, seq, dh, m.cfg.RopeTheta)

	// Embeddings: token embeddings only (positions enter via RoPE). No
	// embedding norm (unlike ModernBERT).
	x := s.x
	for i, id := range ids {
		if id < 0 || int(id) >= m.cfg.VocabSize {
			return nil, fmt.Errorf("token id %d out of vocab range %d", id, m.cfg.VocabSize)
		}
		copy(x[i*H:i*H+H], m.wordEmb[int(id)*H:int(id)*H+H])
	}

	normed := s.normed
	q, k, v := s.qwQ, s.qwK, s.qwV
	kHead, vHeadT := s.qwKHead, s.qwVHeadT
	qHead, cHead := s.qwQHead, s.qwCHead
	scores := s.qwScores
	ctx := s.qwCtx
	attnOut := s.attnOut
	gate, up, act, down := s.wiOut, s.qwUp, s.geglu, s.ffnOut
	scale := float32(1 / math.Sqrt(float64(dh)))

	for li := range m.qwLayers {
		l := &m.qwLayers[li]

		// --- Attention block (pre-norm, causal, GQA, QK-norm) ---
		copy(normed[:seq*H], x[:seq*H])
		tensor.RMSNorm(normed, l.inNormG, seq, H, eps)
		m.applyDense(q, normed[:seq*H], &l.qProj, seq, s)
		m.applyDense(k, normed[:seq*H], &l.kProj, seq, s)
		m.applyDense(v, normed[:seq*H], &l.vProj, seq, s)

		// Per kv-head (shared across a GQA group): repack K and V; QK-norm
		// and RoPE the K once here so the group's query heads reuse it.
		s.pool.Run(nkv, func(kv int) {
			kH := kHead[kv*seq*dh : (kv+1)*seq*dh]
			vHT := vHeadT[kv*dh*seq : (kv+1)*dh*seq]
			for i := range seq {
				copy(kH[i*dh:i*dh+dh], k[i*kvDim+kv*dh:i*kvDim+kv*dh+dh])
				vRow := v[i*kvDim+kv*dh:]
				for d := range dh {
					vHT[d*seq+i] = vRow[d]
				}
			}
			tensor.RMSNorm(kH, l.kNormG, seq, dh, eps)
			applyRoPE(kH, seq, dh, cos, sin)
		})

		// Per query-head: repack Q, QK-norm + RoPE, causal scaled-dot-product
		// attention against its kv head, gather into ctx.
		s.pool.Run(nq, func(hh int) {
			qH := qHead[hh*seq*dh : (hh+1)*seq*dh]
			for i := range seq {
				copy(qH[i*dh:i*dh+dh], q[i*qDim+hh*dh:i*qDim+hh*dh+dh])
			}
			tensor.RMSNorm(qH, l.qNormG, seq, dh, eps)
			applyRoPE(qH, seq, dh, cos, sin)

			kv := hh / group
			kH := kHead[kv*seq*dh : (kv+1)*seq*dh]
			vHT := vHeadT[kv*dh*seq : (kv+1)*dh*seq]
			sc := scores[hh*seq*seq : (hh+1)*seq*seq]
			tensor.MatMulSerial(sc, qH, kH, seq, dh, seq)
			for i := range seq {
				row := sc[i*seq : i*seq+seq]
				for j := range row {
					if j > i {
						row[j] = -1e30 // causal: no attention to the future
					} else {
						row[j] *= scale
					}
				}
			}
			tensor.Softmax(sc, seq, seq)
			chH := cHead[hh*seq*dh : (hh+1)*seq*dh]
			tensor.MatMulSerial(chH, sc, vHT, seq, seq, dh)
			for i := range seq {
				copy(ctx[i*qDim+hh*dh:i*qDim+hh*dh+dh], chH[i*dh:i*dh+dh])
			}
		})

		m.applyDense(attnOut, ctx[:seq*qDim], &l.oProj, seq, s)
		tensor.Add(x, attnOut[:seq*H])

		// --- MLP block (pre-norm SwiGLU) ---
		copy(normed[:seq*H], x[:seq*H])
		tensor.RMSNorm(normed, l.postNorm, seq, H, eps)
		m.applyDense(gate, normed[:seq*H], &l.gateProj, seq, s)
		m.applyDense(up, normed[:seq*H], &l.upProj, seq, s)
		s.pool.Run(seq, func(i int) {
			g := gate[i*I : i*I+I]
			u := up[i*I : i*I+I]
			d := act[i*I : i*I+I]
			copy(d, g)
			tensor.SiLU(d)
			for j := range d {
				d[j] *= u[j]
			}
		})
		m.applyDense(down, act[:seq*I], &l.downProj, seq, s)
		tensor.Add(x, down[:seq*H])
	}

	tensor.RMSNorm(x, m.finalNormG, seq, H, eps)

	committed = true
	return s, nil
}
