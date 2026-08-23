// SPDX-License-Identifier: Apache-2.0

package model

import (
	"fmt"
	"math"
	"runtime"

	"github.com/rostamlabs/rembed/internal/safetensors"
	"github.com/rostamlabs/rembed/internal/tensor"
)

// mbLayer holds one ModernBERT encoder layer. Everything is bias-free.
// attnNormG is nil for layer 0 (HF replaces its attention norm with an
// Identity to avoid re-normalizing the already-normed embeddings). global
// selects the attention type: a global layer attends to every token with
// GlobalRopeTheta, a local layer within a ±LocalAttention/2 window with
// LocalRopeTheta.
type mbLayer struct {
	attnNormG []float32   // [H]; nil ⇒ Identity (layer 0 only)
	qkv       denseWeight // Wqkv [3H×H]
	attnOut   denseWeight // Wo [H×H]
	mlpNormG  []float32   // [H]
	wi        denseWeight // GeGLU Wi [2I×H]
	mlpOut    denseWeight // GeGLU Wo [H×I]
	global    bool
}

// loadModernBERT builds a ModernBERT Model. The tensor names carry no
// architecture prefix (embeddings.tok_embeddings.weight, layers.N.*,
// final_norm.weight) and no bias tensors at all.
func loadModernBERT(tensors map[string]safetensors.Tensor, cfg Config, quantize QuantMode, workers int, weightsPath string) (*Model, error) {
	H, I := cfg.HiddenSize, cfg.IntermediateSize
	if cfg.NumAttentionHeads <= 0 || H%cfg.NumAttentionHeads != 0 {
		return nil, fmt.Errorf("weights %s: hidden_size %d not divisible by num_attention_heads %d", weightsPath, H, cfg.NumAttentionHeads)
	}
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
	if m.wordEmb, err = get("embeddings.tok_embeddings.weight", cfg.VocabSize, H); err != nil {
		return nil, err
	}
	if m.embNormG, err = get("embeddings.norm.weight", H); err != nil {
		return nil, err
	}
	if m.finalNormG, err = get("final_norm.weight", H); err != nil {
		return nil, err
	}
	m.zeroBeta = make([]float32, H)

	m.mbLayers = make([]mbLayer, cfg.NumHiddenLayers)
	for i := range m.mbLayers {
		l := &m.mbLayers[i]
		p := fmt.Sprintf("layers.%d.", i)
		l.global = i%cfg.GlobalAttnEveryNLayers == 0
		// Layer 0's attention norm is nn.Identity in HF; the tensor is
		// absent from the checkpoint, so a nil gamma means "skip the norm".
		if i != 0 {
			if l.attnNormG, err = get(p+"attn_norm.weight", H); err != nil {
				return nil, err
			}
		}
		wqkv, err := get(p+"attn.Wqkv.weight", 3*H, H)
		if err != nil {
			return nil, err
		}
		wo, err := get(p+"attn.Wo.weight", H, H)
		if err != nil {
			return nil, err
		}
		if l.mlpNormG, err = get(p+"mlp_norm.weight", H); err != nil {
			return nil, err
		}
		wi, err := get(p+"mlp.Wi.weight", 2*I, H)
		if err != nil {
			return nil, err
		}
		wmo, err := get(p+"mlp.Wo.weight", H, I)
		if err != nil {
			return nil, err
		}
		l.qkv = newDense(wqkv, nil, H, 3*H, quantize)
		l.attnOut = newDense(wo, nil, H, H, quantize)
		l.wi = newDense(wi, nil, H, 2*I, quantize)
		l.mlpOut = newDense(wmo, nil, I, H, quantize)
	}
	return m, nil
}

// ropeTable precomputes RoPE cos/sin for positions [0,seq) over the
// half=dh/2 frequency pairs of one theta. inv_freq[d] = theta^(-2d/dh);
// the trig is done in float64 and rounded to float32 (well inside the
// golden tolerance, and more accurate than HF's float32 table).
func ropeTable(cos, sin []float32, seq, dh int, theta float64) {
	half := dh / 2
	for d := range half {
		invFreq := math.Pow(theta, -2*float64(d)/float64(dh))
		for p := range seq {
			ang := float64(p) * invFreq
			cos[p*half+d] = float32(math.Cos(ang))
			sin[p*half+d] = float32(math.Sin(ang))
		}
	}
}

// applyRoPE rotates one head's [seq×dh] projection in place with the
// GPT-NeoX "rotate_half" convention: the first and second halves of each
// head vector form the (x1, x2) pairs. The float32() wrappers force each
// product to round before the add, so no arch fuses an FMA and the output
// stays bit-identical across amd64/arm64 (the same guard the MPNet bias
// add uses).
func applyRoPE(x []float32, seq, dh int, cos, sin []float32) {
	half := dh / 2
	for p := range seq {
		row := x[p*dh : p*dh+dh]
		c := cos[p*half : p*half+half]
		s := sin[p*half : p*half+half]
		for d := range half {
			x1, x2 := row[d], row[d+half]
			cc, ss := c[d], s[d]
			row[d] = float32(x1*cc) - float32(x2*ss)
			row[d+half] = float32(x2*cc) + float32(x1*ss)
		}
	}
}

// encodeModernBERT runs the ModernBERT stack, leaving the final hidden
// states in the returned scratch's x[:seq*H]. The caller returns the
// scratch to the pool (mirrors encodeWorkers' contract exactly).
func (m *Model) encodeModernBERT(ids []int64, workers int) (*scratch, error) {
	seq := len(ids)
	H := m.cfg.HiddenSize
	heads := m.cfg.NumAttentionHeads
	dh := H / heads
	half := dh / 2
	I := m.cfg.IntermediateSize
	eps := m.cfg.LayerNormEps
	window := m.cfg.LocalAttention / 2

	s := m.scratchPool.Get().(*scratch)
	committed := false
	defer func() {
		if !committed {
			m.scratchPool.Put(s)
		}
	}()

	// Scratch sizing: qkv/attnOut/ffnOut/scores/qh/kh/vhT/ch match the BERT
	// path; normed/wiOut/geglu and the RoPE tables are ModernBERT-only.
	mPad := tensor.PackAPad(seq)
	s.x = grow(s.x, seq*H)
	s.normed = grow(s.normed, seq*H)
	s.qkv = grow(s.qkv, mPad*3*H)
	s.ctxOut = grow(s.ctxOut, seq*H)
	s.attnOut = grow(s.attnOut, mPad*H)
	s.wiOut = grow(s.wiOut, mPad*2*I)
	s.geglu = grow(s.geglu, seq*I)
	s.ffnOut = grow(s.ffnOut, mPad*H)
	s.aPack = grow(s.aPack, mPad*max(H, 2*I))
	s.scores = grow(s.scores, heads*seq*seq)
	s.qh = grow(s.qh, seq*H)
	s.kh = grow(s.kh, seq*H)
	s.ch = grow(s.ch, seq*H)
	s.vhT = grow(s.vhT, H*seq)
	s.cosG, s.sinG = grow(s.cosG, seq*half), grow(s.sinG, seq*half)
	s.cosL, s.sinL = grow(s.cosL, seq*half), grow(s.sinL, seq*half)
	kgMax := (max(H, 2*I) + 3) / 4
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

	// Embeddings: token embeddings only (positions enter via RoPE), then a
	// LayerNorm — bias-free, hence the shared zero beta.
	x := s.x
	for i, id := range ids {
		if id < 0 || int(id) >= m.cfg.VocabSize {
			return nil, fmt.Errorf("token id %d out of vocab range %d", id, m.cfg.VocabSize)
		}
		copy(x[i*H:i*H+H], m.wordEmb[int(id)*H:int(id)*H+H])
	}
	tensor.LayerNorm(x, m.embNormG, m.zeroBeta, seq, H, eps)

	normed := s.normed
	qkv := s.qkv
	ctxOut, attnOut := s.ctxOut, s.attnOut
	scores := s.scores
	qh, kh, vhT, ch := s.qh, s.kh, s.vhT, s.ch
	wiOut, geglu, mlpOut := s.wiOut, s.geglu, s.ffnOut
	scale := float32(1 / math.Sqrt(float64(dh)))

	for li := range m.mbLayers {
		l := &m.mbLayers[li]
		cos, sin := s.cosL, s.sinL
		if l.global {
			cos, sin = s.cosG, s.sinG
		}

		// --- Attention block (pre-norm) ---
		// attn_norm(x) into a separate buffer so the residual keeps the
		// un-normed x; layer 0's norm is Identity, so just copy.
		if l.attnNormG == nil {
			copy(normed[:seq*H], x[:seq*H])
		} else {
			copy(normed[:seq*H], x[:seq*H])
			tensor.LayerNorm(normed, l.attnNormG, m.zeroBeta, seq, H, eps)
		}
		m.applyDense(qkv, normed[:seq*H], &l.qkv, seq, s)

		s.pool.Run(heads, func(h int) {
			off := h * dh
			qhH := qh[h*seq*dh : (h+1)*seq*dh]
			khH := kh[h*seq*dh : (h+1)*seq*dh]
			vhTH := vhT[h*dh*seq : (h+1)*dh*seq]
			chH := ch[h*seq*dh : (h+1)*seq*dh]
			sc := scores[h*seq*seq : (h+1)*seq*seq]
			for i := range seq {
				row := qkv[i*3*H:]
				copy(qhH[i*dh:i*dh+dh], row[off:off+dh])
				copy(khH[i*dh:i*dh+dh], row[H+off:H+off+dh])
				vRow := row[2*H+off:]
				for d := range dh {
					vhTH[d*seq+i] = vRow[d]
				}
			}
			// RoPE rotates q and k (not v) before the score product.
			applyRoPE(qhH, seq, dh, cos, sin)
			applyRoPE(khH, seq, dh, cos, sin)
			tensor.MatMulSerial(sc, qhH, khH, seq, dh, seq)
			for i := range seq {
				scRow := sc[i*seq : i*seq+seq]
				for j := range scRow {
					scRow[j] *= scale
				}
				// Sliding-window mask on local layers: a query attends only
				// within ±window (LocalAttention/2). Masked scores go to a
				// large negative so softmax gives them ~0 weight, matching
				// HF's finfo.min fill. Global layers see the whole sequence.
				if !l.global {
					for j := range scRow {
						if j < i-window || j > i+window {
							scRow[j] = -1e30
						}
					}
				}
			}
			tensor.Softmax(sc, seq, seq)
			tensor.MatMulSerial(chH, sc, vhTH, seq, seq, dh)
			for i := range seq {
				copy(ctxOut[i*H+off:i*H+off+dh], chH[i*dh:i*dh+dh])
			}
		})

		m.applyDense(attnOut, ctxOut[:seq*H], &l.attnOut, seq, s)
		tensor.Add(x, attnOut[:seq*H]) // residual onto the un-normed x

		// --- MLP block (pre-norm GeGLU) ---
		copy(normed[:seq*H], x[:seq*H])
		tensor.LayerNorm(normed, l.mlpNormG, m.zeroBeta, seq, H, eps)
		m.applyDense(wiOut, normed[:seq*H], &l.wi, seq, s)
		// GeGLU: Wi outputs [seq×2I]; the first I are the value, the second
		// I the gate. act = gelu(value) * gate.
		s.pool.Run(seq, func(i int) {
			in := wiOut[i*2*I : i*2*I+I]
			gate := wiOut[i*2*I+I : i*2*I+2*I]
			dst := geglu[i*I : i*I+I]
			copy(dst, in)
			tensor.GELU(dst)
			for d := range dst {
				dst[d] *= gate[d]
			}
		})
		m.applyDense(mlpOut, geglu[:seq*I], &l.mlpOut, seq, s)
		tensor.Add(x, mlpOut[:seq*H])
	}

	// Final norm before pooling.
	tensor.LayerNorm(x, m.finalNormG, m.zeroBeta, seq, H, eps)

	committed = true
	return s, nil
}
