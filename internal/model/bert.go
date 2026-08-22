// SPDX-License-Identifier: Apache-2.0

// Package model implements the BERT encoder forward pass:
// embeddings(token+position+segment) → N layers (self-attention + FFN,
// post-LayerNorm) → mean pooling → L2 normalize.
package model

import (
	"fmt"
	"math"
	"sync"

	"github.com/rostamlabs/rembed/internal/safetensors"
	"github.com/rostamlabs/rembed/internal/tensor"
)

// layer holds one encoder layer's weights. Linear weights keep HuggingFace's
// [out, in] layout, which is exactly the bT operand of the matmul kernel.
type layer struct {
	qW, qB             []float32 // [H×H], [H]
	kW, kB             []float32
	vW, vB             []float32
	attnOutW, attnOutB []float32 // [H×H], [H]
	attnLNg, attnLNb   []float32 // [H]
	ffn1W, ffn1B       []float32 // [I×H], [I]
	ffn2W, ffn2B       []float32 // [H×I], [H]
	outLNg, outLNb     []float32 // [H]
}

// Model is a loaded BERT encoder ready to embed token-id sequences.
// It is safe for concurrent use: the matmul kernel is bound once at Load,
// and per-call scratch comes from an internal pool.
type Model struct {
	cfg    Config
	matmul tensor.MatMulFunc

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
type scratch struct {
	x, q, k, v      []float32 // [seq×H]
	ctxOut, attnOut []float32 // [seq×H]
	ffnOut          []float32 // [seq×H]
	ffnHidden       []float32 // [seq×I]
	scores          []float32 // [seq×seq]
	qh, kh, ch      []float32 // [seq×dh]
	vhT             []float32 // [dh×seq]
}

// grow reslices buf to n floats, reallocating only when capacity is short.
func grow(buf []float32, n int) []float32 {
	if cap(buf) < n {
		return make([]float32, n)
	}
	return buf[:n]
}

func (s *scratch) resize(seq, H, I, dh int) {
	s.x = grow(s.x, seq*H)
	s.q = grow(s.q, seq*H)
	s.k = grow(s.k, seq*H)
	s.v = grow(s.v, seq*H)
	s.ctxOut = grow(s.ctxOut, seq*H)
	s.attnOut = grow(s.attnOut, seq*H)
	s.ffnOut = grow(s.ffnOut, seq*H)
	s.ffnHidden = grow(s.ffnHidden, seq*I)
	s.scores = grow(s.scores, seq*seq)
	s.qh = grow(s.qh, seq*dh)
	s.kh = grow(s.kh, seq*dh)
	s.ch = grow(s.ch, seq*dh)
	s.vhT = grow(s.vhT, dh*seq)
}

// Load builds a Model from a safetensors file and a validated Config.
// Tensor names follow HuggingFace BertModel conventions, with or without a
// leading "bert." prefix.
func Load(weightsPath string, cfg Config) (*Model, error) {
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
	m := &Model{cfg: cfg, matmul: tensor.Default()}
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
	m.layers = make([]layer, cfg.NumHiddenLayers)
	for i := range m.layers {
		l := &m.layers[i]
		p := fmt.Sprintf("encoder.layer.%d.", i)
		loads = append(loads,
			load{&l.qW, p + "attention.self.query.weight", []int{H, H}},
			load{&l.qB, p + "attention.self.query.bias", []int{H}},
			load{&l.kW, p + "attention.self.key.weight", []int{H, H}},
			load{&l.kB, p + "attention.self.key.bias", []int{H}},
			load{&l.vW, p + "attention.self.value.weight", []int{H, H}},
			load{&l.vB, p + "attention.self.value.bias", []int{H}},
			load{&l.attnOutW, p + "attention.output.dense.weight", []int{H, H}},
			load{&l.attnOutB, p + "attention.output.dense.bias", []int{H}},
			load{&l.attnLNg, p + "attention.output.LayerNorm.weight", []int{H}},
			load{&l.attnLNb, p + "attention.output.LayerNorm.bias", []int{H}},
			load{&l.ffn1W, p + "intermediate.dense.weight", []int{I, H}},
			load{&l.ffn1B, p + "intermediate.dense.bias", []int{I}},
			load{&l.ffn2W, p + "output.dense.weight", []int{H, I}},
			load{&l.ffn2B, p + "output.dense.bias", []int{H}},
			load{&l.outLNg, p + "output.LayerNorm.weight", []int{H}},
			load{&l.outLNb, p + "output.LayerNorm.bias", []int{H}},
		)
	}
	for _, ld := range loads {
		data, err := get(ld.name, ld.shape...)
		if err != nil {
			return nil, err
		}
		*ld.dst = data
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

	q, k, v := s.q, s.k, s.v
	ctxOut, attnOut := s.ctxOut, s.attnOut
	scores := s.scores
	qh, kh, vhT, ch := s.qh, s.kh, s.vhT, s.ch
	ffnHidden, ffnOut := s.ffnHidden, s.ffnOut
	scale := float32(1 / math.Sqrt(float64(dh)))

	for li := range m.layers {
		l := &m.layers[li]

		// Q, K, V projections.
		m.matmul(q, x, l.qW, seq, H, H)
		tensor.AddBias(q, l.qB, seq, H)
		m.matmul(k, x, l.kW, seq, H, H)
		tensor.AddBias(k, l.kB, seq, H)
		m.matmul(v, x, l.vW, seq, H, H)
		tensor.AddBias(v, l.vB, seq, H)

		// Scaled dot-product attention, one head at a time.
		for h := range heads {
			off := h * dh
			// Repack this head's slices: qh,kh are [seq×dh]; vhT is Vᵀ [dh×seq]
			// so that probs·V fits the single MatMul (bT) signature.
			for i := range seq {
				copy(qh[i*dh:i*dh+dh], q[i*H+off:i*H+off+dh])
				copy(kh[i*dh:i*dh+dh], k[i*H+off:i*H+off+dh])
				for d := range dh {
					vhT[d*seq+i] = v[i*H+off+d]
				}
			}
			// scores = Qh·Khᵀ / √dh; Kh as bT gives exactly Qh·Khᵀ.
			m.matmul(scores, qh, kh, seq, dh, seq)
			for i := range scores {
				scores[i] *= scale
			}
			tensor.Softmax(scores, seq, seq)
			// ch = probs·Vh, with Vhᵀ as the bT operand.
			m.matmul(ch, scores, vhT, seq, seq, dh)
			for i := range seq {
				copy(ctxOut[i*H+off:i*H+off+dh], ch[i*dh:i*dh+dh])
			}
		}

		// Attention output projection + residual + LayerNorm.
		m.matmul(attnOut, ctxOut, l.attnOutW, seq, H, H)
		tensor.AddBias(attnOut, l.attnOutB, seq, H)
		tensor.Add(x, attnOut)
		tensor.LayerNorm(x, l.attnLNg, l.attnLNb, seq, H, eps)

		// FFN: GELU(x·W1ᵀ+b1)·W2ᵀ+b2 + residual + LayerNorm.
		m.matmul(ffnHidden, x, l.ffn1W, seq, H, I)
		tensor.AddBias(ffnHidden, l.ffn1B, seq, I)
		tensor.GELU(ffnHidden)
		m.matmul(ffnOut, ffnHidden, l.ffn2W, seq, I, H)
		tensor.AddBias(ffnOut, l.ffn2B, seq, H)
		tensor.Add(x, ffnOut)
		tensor.LayerNorm(x, l.outLNg, l.outLNb, seq, H, eps)
	}

	// Mean pooling over all positions (no padding ⇒ every position counts),
	// then optional L2 normalization.
	pooled := make([]float32, H)
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
	if m.cfg.Normalize {
		tensor.L2Normalize(pooled)
	}
	return pooled, nil
}
