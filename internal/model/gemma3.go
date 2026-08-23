// SPDX-License-Identifier: Apache-2.0

package model

import (
	"fmt"
	"math"
	"path/filepath"
	"runtime"

	"github.com/rostamlabs/rembed/internal/safetensors"
	"github.com/rostamlabs/rembed/internal/tensor"
)

// gmLayer holds one Gemma 3 decoder layer used as a bidirectional encoder.
// Everything is bias-free. All the *Norm weights are stored UNIT-OFFSET
// (the checkpoint's weight + 1) so the standard tensor.RMSNorm reproduces
// Gemma's x̂·(1+w). q/k/v are separate projections (k/v narrower under GQA);
// qNormG/kNormG are the per-head QK-norm weights (length head_dim), applied
// to each head before RoPE. The four LayerNorms follow Gemma's sandwich:
// input (pre-attn) and preFFN are pre-norms; postAttn and postFFN normalize
// the sublayer OUTPUT before it is added back to the residual.
type gmLayer struct {
	inNormG   []float32 // input_layernorm [H]
	qProj     denseWeight
	kProj     denseWeight
	vProj     denseWeight
	oProj     denseWeight
	qNormG    []float32 // [dh]
	kNormG    []float32 // [dh]
	postAttnG []float32 // post_attention_layernorm [H]
	preFFNG   []float32 // pre_feedforward_layernorm [H]
	postFFNG  []float32 // post_feedforward_layernorm [H]
	gateProj  denseWeight
	upProj    denseWeight
	downProj  denseWeight
	global    bool // full attention (vs sliding window)
}

// unitOffset returns a fresh copy of v with 1.0 added to every element —
// Gemma's RMSNorm scales by (1 + weight), so folding the +1 in at load lets
// the standard RMSNorm kernel run unchanged. A copy (not in-place) keeps the
// mmap/safetensors buffer pristine.
func unitOffset(v []float32) []float32 {
	out := make([]float32, len(v))
	for i, x := range v {
		out[i] = x + 1
	}
	return out
}

// loadGemma3 builds an EmbeddingGemma model: the Gemma3TextModel backbone
// (tensor names carry no prefix — embed_tokens.weight, layers.N.*,
// norm.weight) plus the sentence-transformers Dense head, which ships as two
// separate single-file safetensors under 2_Dense/ and 3_Dense/ alongside the
// backbone weights.
func loadGemma3(tensors map[string]safetensors.Tensor, cfg Config, quantize QuantMode, workers int, weightsPath string) (*Model, error) {
	H, I := cfg.HiddenSize, cfg.IntermediateSize
	nq, nkv, dh := cfg.NumAttentionHeads, cfg.NumKeyValueHeads, cfg.HeadDim
	qDim, kvDim := nq*dh, nkv*dh

	f32 := func(name string, wantShape ...int) ([]float32, error) {
		t, ok := tensors[name]
		if !ok {
			return nil, fmt.Errorf("weights %s: missing tensor %q", weightsPath, name)
		}
		if !shapeEq(t.Shape, wantShape) {
			return nil, fmt.Errorf("weights %s: tensor %q has shape %v, want %v", weightsPath, name, t.Shape, wantShape)
		}
		return t.Data, nil
	}

	m := &Model{cfg: cfg, workers: workers, posOff: 0}
	m.scratchPool.New = func() any { return new(scratch) }

	var err error
	if m.wordEmb, err = f32("embed_tokens.weight", cfg.VocabSize, H); err != nil {
		return nil, err
	}
	// Trailing RMSNorm (unit-offset), reusing the shared finalNormG slot.
	nrm, err := f32("norm.weight", H)
	if err != nil {
		return nil, err
	}
	m.finalNormG = unitOffset(nrm)

	m.gmLayers = make([]gmLayer, cfg.NumHiddenLayers)
	for i := range m.gmLayers {
		l := &m.gmLayers[i]
		p := fmt.Sprintf("layers.%d.", i)
		l.global = (i+1)%cfg.SlidingWindowPattern == 0
		norm := func(name string, dst *[]float32, dim int) error {
			v, e := f32(p+name, dim)
			if e != nil {
				return e
			}
			*dst = unitOffset(v)
			return nil
		}
		proj := func(name string, in, out int) (denseWeight, error) {
			raw, e := f32(p+name, out, in)
			if e != nil {
				return denseWeight{}, e
			}
			return newDense(raw, nil, in, out, quantize), nil
		}
		if err = norm("input_layernorm.weight", &l.inNormG, H); err != nil {
			return nil, err
		}
		if err = norm("self_attn.q_norm.weight", &l.qNormG, dh); err != nil {
			return nil, err
		}
		if err = norm("self_attn.k_norm.weight", &l.kNormG, dh); err != nil {
			return nil, err
		}
		if err = norm("post_attention_layernorm.weight", &l.postAttnG, H); err != nil {
			return nil, err
		}
		if err = norm("pre_feedforward_layernorm.weight", &l.preFFNG, H); err != nil {
			return nil, err
		}
		if err = norm("post_feedforward_layernorm.weight", &l.postFFNG, H); err != nil {
			return nil, err
		}
		for _, pr := range []struct {
			dst     *denseWeight
			name    string
			in, out int
		}{
			{&l.qProj, "self_attn.q_proj.weight", H, qDim},
			{&l.kProj, "self_attn.k_proj.weight", H, kvDim},
			{&l.vProj, "self_attn.v_proj.weight", H, kvDim},
			{&l.oProj, "self_attn.o_proj.weight", qDim, H},
			{&l.gateProj, "mlp.gate_proj.weight", H, I},
			{&l.upProj, "mlp.up_proj.weight", H, I},
			{&l.downProj, "mlp.down_proj.weight", I, H},
		} {
			if *pr.dst, err = proj(pr.name, pr.in, pr.out); err != nil {
				return nil, err
			}
		}
	}

	// Dense head: two bias-free linear layers (H→DenseHidden→H) that
	// sentence-transformers stores as 2_Dense/model.safetensors and
	// 3_Dense/model.safetensors, each with a single "linear.weight" [out,in].
	dir := filepath.Dir(weightsPath)
	if m.gmDense1, err = loadDenseLinear(filepath.Join(dir, "2_Dense", "model.safetensors"), cfg.DenseHidden, H); err != nil {
		return nil, err
	}
	if m.gmDense2, err = loadDenseLinear(filepath.Join(dir, "3_Dense", "model.safetensors"), H, cfg.DenseHidden); err != nil {
		return nil, err
	}
	return m, nil
}

// loadDenseLinear reads a sentence-transformers Dense module's [out,in]
// linear.weight from a single-file safetensors.
func loadDenseLinear(path string, out, in int) ([]float32, error) {
	ts, err := safetensors.Load(path)
	if err != nil {
		return nil, fmt.Errorf("gemma3 dense head %s: %w", path, err)
	}
	t, ok := ts["linear.weight"]
	if !ok {
		return nil, fmt.Errorf("gemma3 dense head %s: missing linear.weight", path)
	}
	if !shapeEq(t.Shape, []int{out, in}) {
		return nil, fmt.Errorf("gemma3 dense head %s: linear.weight shape %v, want %v", path, t.Shape, []int{out, in})
	}
	return t.Data, nil
}

// applyGemmaDenseHead projects the pooled [H] vector through the two bias-free
// linear layers (H→DenseHidden→H), returning a fresh [H] vector. Both layers
// are Identity-activated, so this is two matrix-vector products.
func (m *Model) applyGemmaDenseHead(pooled []float32) []float32 {
	H, D := m.cfg.HiddenSize, m.cfg.DenseHidden
	hid := make([]float32, D)
	for o := range D {
		row := m.gmDense1[o*H : o*H+H]
		var acc float32
		for i, w := range row {
			acc += w * pooled[i]
		}
		hid[o] = acc
	}
	out := make([]float32, H)
	for o := range H {
		row := m.gmDense2[o*D : o*D+D]
		var acc float32
		for i, w := range row {
			acc += w * hid[i]
		}
		out[o] = acc
	}
	return out
}

// encodeGemma3 runs EmbeddingGemma's Gemma 3 backbone as a bidirectional
// encoder: scaled token embeddings, then 24 decoder layers with per-head
// QK-norm, grouped-query attention, dual-theta RoPE, an alternating
// sliding-window/global bidirectional mask, and Gemma's four-LayerNorm
// sandwich, followed by the trailing RMSNorm. Pooling and the Dense head run
// in ForwardWorkers.
func (m *Model) encodeGemma3(ids []int64, workers int) (*scratch, error) {
	seq := len(ids)
	H := m.cfg.HiddenSize
	nq, nkv, dh := m.cfg.NumAttentionHeads, m.cfg.NumKeyValueHeads, m.cfg.HeadDim
	qDim, kvDim := nq*dh, nkv*dh
	group := nq / nkv
	half := dh / 2
	I := m.cfg.IntermediateSize
	eps := m.cfg.LayerNormEps
	window := m.cfg.SlidingWindow

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
	s.qwKHead = grow(s.qwKHead, nkv*seq*dh) // per-kv-head K (row-major), normed+RoPE'd
	s.qwVHead = grow(s.qwVHead, nkv*dh*seq) // per-kv-head V TRANSPOSED [dh×seq] for P·V
	s.qwQHead = grow(s.qwQHead, nq*seq*dh)
	s.qwCHead = grow(s.qwCHead, nq*seq*dh)
	s.qwCtx = grow(s.qwCtx, seq*qDim)
	s.scores = grow(s.scores, nq*seq*seq)
	s.attnOut = grow(s.attnOut, mPad*H)
	s.wiOut = grow(s.wiOut, mPad*I) // GeGLU gate projection
	s.qwUp = grow(s.qwUp, mPad*I)   // GeGLU up projection
	s.geglu = grow(s.geglu, seq*I)  // gelu_tanh(gate)*up
	s.ffnOut = grow(s.ffnOut, mPad*H)
	s.aPack = grow(s.aPack, mPad*max(qDim, I))
	s.cosG, s.sinG = grow(s.cosG, seq*half), grow(s.sinG, seq*half)
	s.cosL, s.sinL = grow(s.cosL, seq*half), grow(s.sinL, seq*half)
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

	ropeTable(s.cosG, s.sinG, seq, dh, m.cfg.GlobalRopeTheta)
	ropeTable(s.cosL, s.sinL, seq, dh, m.cfg.LocalRopeTheta)

	// Embeddings scaled by √H (Gemma3TextScaledWordEmbedding). Positions enter
	// via RoPE; there is no learned position table and no embedding norm.
	embScale := float32(math.Sqrt(float64(H)))
	x := s.x
	for i, id := range ids {
		if id < 0 || int(id) >= m.cfg.VocabSize {
			return nil, fmt.Errorf("token id %d out of vocab range %d", id, m.cfg.VocabSize)
		}
		src := m.wordEmb[int(id)*H : int(id)*H+H]
		dst := x[i*H : i*H+H]
		for j := range dst {
			dst[j] = src[j] * embScale
		}
	}

	normed := s.normed
	q, k, v := s.qwQ, s.qwK, s.qwV
	kHead, vHeadT := s.qwKHead, s.qwVHead
	qHead, cHead := s.qwQHead, s.qwCHead
	ctx := s.qwCtx
	scores := s.scores
	attnOut := s.attnOut
	gate, up, act, down := s.wiOut, s.qwUp, s.geglu, s.ffnOut
	scale := float32(1 / math.Sqrt(float64(m.cfg.QueryPreAttnScalar)))

	for li := range m.gmLayers {
		l := &m.gmLayers[li]
		cos, sin := s.cosL, s.sinL
		if l.global {
			cos, sin = s.cosG, s.sinG
		}

		// --- Attention block (pre-norm, bidirectional, GQA, QK-norm) ---
		copy(normed[:seq*H], x[:seq*H])
		tensor.RMSNorm(normed, l.inNormG, seq, H, eps)
		m.applyDense(q, normed[:seq*H], &l.qProj, seq, s)
		m.applyDense(k, normed[:seq*H], &l.kProj, seq, s)
		m.applyDense(v, normed[:seq*H], &l.vProj, seq, s)

		// Per kv-head (shared across a GQA group): repack + QK-norm + RoPE K,
		// and transpose V to [dh×seq] for the score·V product.
		s.pool.Run(nkv, func(kv int) {
			kH := kHead[kv*seq*dh : (kv+1)*seq*dh]
			vHT := vHeadT[kv*dh*seq : (kv+1)*dh*seq]
			for i := range seq {
				copy(kH[i*dh:i*dh+dh], k[i*kvDim+kv*dh:i*kvDim+kv*dh+dh])
				vRow := v[i*kvDim+kv*dh : i*kvDim+kv*dh+dh]
				for d := range dh {
					vHT[d*seq+i] = vRow[d]
				}
			}
			tensor.RMSNorm(kH, l.kNormG, seq, dh, eps)
			applyRoPE(kH, seq, dh, cos, sin)
		})

		// Per query-head: repack + QK-norm + RoPE Q, bidirectional
		// scaled-dot-product attention against its kv head (sliding-window
		// masked on local layers), gather into ctx.
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
			chH := cHead[hh*seq*dh : (hh+1)*seq*dh]
			sc := scores[hh*seq*seq : (hh+1)*seq*seq]
			tensor.MatMulSerial(sc, qH, kH, seq, dh, seq)
			for i := range seq {
				scRow := sc[i*seq : i*seq+seq]
				for j := range scRow {
					scRow[j] *= scale
				}
				// Bidirectional sliding window on local layers: a query
				// attends only where |i−j| < window. Global layers see all.
				if !l.global {
					for j := range scRow {
						if j <= i-window || j >= i+window {
							scRow[j] = -1e30
						}
					}
				}
			}
			tensor.Softmax(sc, seq, seq)
			tensor.MatMulSerial(chH, sc, vHT, seq, seq, dh)
			for i := range seq {
				copy(ctx[i*qDim+hh*dh:i*qDim+hh*dh+dh], chH[i*dh:i*dh+dh])
			}
		})

		m.applyDense(attnOut, ctx[:seq*qDim], &l.oProj, seq, s)
		tensor.RMSNorm(attnOut, l.postAttnG, seq, H, eps) // post-norm on the sublayer output
		tensor.Add(x, attnOut[:seq*H])

		// --- MLP block (pre-norm GeGLU, post-normed output) ---
		copy(normed[:seq*H], x[:seq*H])
		tensor.RMSNorm(normed, l.preFFNG, seq, H, eps)
		m.applyDense(gate, normed[:seq*H], &l.gateProj, seq, s)
		m.applyDense(up, normed[:seq*H], &l.upProj, seq, s)
		s.pool.Run(seq, func(i int) {
			g := gate[i*I : i*I+I]
			u := up[i*I : i*I+I]
			d := act[i*I : i*I+I]
			copy(d, g)
			tensor.GELUTanh(d)
			for j := range d {
				d[j] *= u[j]
			}
		})
		m.applyDense(down, act[:seq*I], &l.downProj, seq, s)
		tensor.RMSNorm(down, l.postFFNG, seq, H, eps) // post-norm on the sublayer output
		tensor.Add(x, down[:seq*H])
	}

	tensor.RMSNorm(x, m.finalNormG, seq, H, eps)

	committed = true
	return s, nil
}
