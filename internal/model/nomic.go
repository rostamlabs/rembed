// SPDX-License-Identifier: Apache-2.0

package model

import (
	"fmt"
	"math"
	"runtime"

	"github.com/rostamlabs/rembed/internal/safetensors"
	"github.com/rostamlabs/rembed/internal/tensor"
)

// nmLayer holds one nomic-embed encoder layer. A BERT-style POST-norm block
// (residual add THEN LayerNorm) but bias-free in the projections, with RoPE
// on the attention and a gated SwiGLU MLP. Wqkv is the fused query‖key‖value
// projection; the SwiGLU MLP is fc11(x)·silu(fc12(x)) then fc2.
type nmLayer struct {
	wqkv           denseWeight // [3H×H]
	attnOut        denseWeight // [H×H]
	norm1G, norm1B []float32   // post-attention LayerNorm [H]
	fc11           denseWeight // [I×H] SwiGLU linear operand
	fc12           denseWeight // [I×H] SwiGLU gated operand (silu)
	fc2            denseWeight // [H×I] down
	norm2G, norm2B []float32   // post-MLP LayerNorm [H]
}

// loadNomic builds a nomic-embed model. Embeddings are word + token_type[0]
// (no learned positions — RoPE supplies them) followed by an embedding
// LayerNorm; the stack is post-norm, so the last layer's norm2 is the final
// normalization (there is no separate trailing norm).
func loadNomic(tensors map[string]safetensors.Tensor, cfg Config, quantize QuantMode, workers int, weightsPath string) (*Model, error) {
	H, I := cfg.HiddenSize, cfg.IntermediateSize

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
	if m.wordEmb, err = f32("embeddings.word_embeddings.weight", cfg.VocabSize, H); err != nil {
		return nil, err
	}
	if m.typeEmb, err = f32("embeddings.token_type_embeddings.weight", 2, H); err != nil {
		return nil, err
	}
	if m.embLNg, err = f32("emb_ln.weight", H); err != nil {
		return nil, err
	}
	if m.embLNb, err = f32("emb_ln.bias", H); err != nil {
		return nil, err
	}

	m.nmLayers = make([]nmLayer, cfg.NumHiddenLayers)
	for i := range m.nmLayers {
		l := &m.nmLayers[i]
		p := fmt.Sprintf("encoder.layers.%d.", i)
		vec := func(name string, dst *[]float32) error {
			v, e := f32(p+name, H)
			*dst = v
			return e
		}
		proj := func(name string, in, out int) (denseWeight, error) {
			raw, e := f32(p+name, out, in)
			if e != nil {
				return denseWeight{}, e
			}
			return newDense(raw, nil, in, out, quantize), nil
		}
		if err = vec("norm1.weight", &l.norm1G); err != nil {
			return nil, err
		}
		if err = vec("norm1.bias", &l.norm1B); err != nil {
			return nil, err
		}
		if err = vec("norm2.weight", &l.norm2G); err != nil {
			return nil, err
		}
		if err = vec("norm2.bias", &l.norm2B); err != nil {
			return nil, err
		}
		for _, pr := range []struct {
			dst     *denseWeight
			name    string
			in, out int
		}{
			{&l.wqkv, "attn.Wqkv.weight", H, 3 * H},
			{&l.attnOut, "attn.out_proj.weight", H, H},
			{&l.fc11, "mlp.fc11.weight", H, I},
			{&l.fc12, "mlp.fc12.weight", H, I},
			{&l.fc2, "mlp.fc2.weight", I, H},
		} {
			if *pr.dst, err = proj(pr.name, pr.in, pr.out); err != nil {
				return nil, err
			}
		}
	}
	return m, nil
}

// encodeNomic runs the nomic-embed stack: word+segment embeddings and an
// embedding LayerNorm, then post-norm layers with RoPE attention
// (bidirectional, full) and a SwiGLU MLP.
func (m *Model) encodeNomic(ids []int64, workers int) (*scratch, error) {
	seq := len(ids)
	H := m.cfg.HiddenSize
	heads := m.cfg.NumAttentionHeads
	dh := m.cfg.HeadDim
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
	s.qkv = grow(s.qkv, mPad*3*H)
	s.ctxOut = grow(s.ctxOut, seq*H)
	s.attnOut = grow(s.attnOut, mPad*H)
	s.wiOut = grow(s.wiOut, mPad*I) // SwiGLU gate projection (fc11)
	s.qwUp = grow(s.qwUp, mPad*I)   // SwiGLU up projection (fc12)
	s.geglu = grow(s.geglu, seq*I)  // silu(gate)*up
	s.ffnOut = grow(s.ffnOut, mPad*H)
	s.aPack = grow(s.aPack, mPad*max(3*H, I))
	s.scores = grow(s.scores, heads*seq*seq)
	s.qh = grow(s.qh, seq*H)
	s.kh = grow(s.kh, seq*H)
	s.ch = grow(s.ch, seq*H)
	s.vhT = grow(s.vhT, H*seq)
	s.cosG, s.sinG = grow(s.cosG, seq*half), grow(s.sinG, seq*half)
	kgMax := (max(3*H, I) + 3) / 4
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

	// Embeddings: word + token_type 0 (single sequence), then LayerNorm.
	x := s.x
	for i, id := range ids {
		if id < 0 || int(id) >= m.cfg.VocabSize {
			return nil, fmt.Errorf("token id %d out of vocab range %d", id, m.cfg.VocabSize)
		}
		row := x[i*H : i*H+H]
		w := m.wordEmb[int(id)*H : int(id)*H+H]
		t := m.typeEmb[:H]
		for j := range row {
			row[j] = w[j] + t[j]
		}
	}
	tensor.LayerNorm(x, m.embLNg, m.embLNb, seq, H, eps)

	qkv := s.qkv
	ctxOut, attnOut := s.ctxOut, s.attnOut
	scores := s.scores
	qh, kh, vhT, ch := s.qh, s.kh, s.vhT, s.ch
	// fc11 is the LINEAR operand, fc12 the SILU-GATED operand: fc11·silu(fc12).
	// (The names deliberately track the weight names, not "gate"/"up", so a
	// future edit does not silently flip the operands — that flip made every
	// vector near-orthogonal until the reference settled the order.)
	fc11Out, fc12Out, act, mlpOut := s.wiOut, s.qwUp, s.geglu, s.ffnOut
	scale := float32(1 / math.Sqrt(float64(dh)))

	for li := range m.nmLayers {
		l := &m.nmLayers[li]

		// --- Attention (bidirectional, RoPE, post-norm) ---
		m.applyDense(qkv, x[:seq*H], &l.wqkv, seq, s)
		s.pool.Run(heads, func(h int) {
			off := h * dh
			qhH := qh[h*seq*dh : (h+1)*seq*dh]
			khH := kh[h*seq*dh : (h+1)*seq*dh]
			vhTH := vhT[h*dh*seq : (h+1)*dh*seq]
			chH := ch[h*seq*dh : (h+1)*seq*dh]
			sc := scores[h*seq*seq : (h+1)*seq*seq]
			for i := range seq {
				r := qkv[i*3*H:]
				copy(qhH[i*dh:i*dh+dh], r[off:off+dh])
				copy(khH[i*dh:i*dh+dh], r[H+off:H+off+dh])
				vRow := r[2*H+off:]
				for d := range dh {
					vhTH[d*seq+i] = vRow[d]
				}
			}
			applyRoPE(qhH, seq, dh, cos, sin)
			applyRoPE(khH, seq, dh, cos, sin)
			tensor.MatMulSerial(sc, qhH, khH, seq, dh, seq)
			for i := range seq {
				scRow := sc[i*seq : i*seq+seq]
				for j := range scRow {
					scRow[j] *= scale
				}
			}
			tensor.Softmax(sc, seq, seq)
			tensor.MatMulSerial(chH, sc, vhTH, seq, seq, dh)
			for i := range seq {
				copy(ctxOut[i*H+off:i*H+off+dh], chH[i*dh:i*dh+dh])
			}
		})
		m.applyDense(attnOut, ctxOut[:seq*H], &l.attnOut, seq, s)
		tensor.Add(x, attnOut[:seq*H])
		tensor.LayerNorm(x, l.norm1G, l.norm1B, seq, H, eps) // post-norm

		// --- MLP (SwiGLU, post-norm) ---
		m.applyDense(fc11Out, x[:seq*H], &l.fc11, seq, s)
		m.applyDense(fc12Out, x[:seq*H], &l.fc12, seq, s)
		s.pool.Run(seq, func(i int) {
			lin := fc11Out[i*I : i*I+I]  // linear operand
			gated := fc12Out[i*I : i*I+I] // silu-gated operand
			d := act[i*I : i*I+I]
			// nomic's GatedMLP: fc11(x) * silu(fc12(x)) — silu is on fc12.
			copy(d, gated)
			tensor.SiLU(d)
			for j := range d {
				d[j] *= lin[j]
			}
		})
		m.applyDense(mlpOut, act[:seq*I], &l.fc2, seq, s)
		tensor.Add(x, mlpOut[:seq*H])
		tensor.LayerNorm(x, l.norm2G, l.norm2B, seq, H, eps) // post-norm
	}

	committed = true
	return s, nil
}
