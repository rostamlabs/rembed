// SPDX-License-Identifier: Apache-2.0

package model

import (
	"fmt"
	"math"
	"os"
	"path/filepath"
	"runtime"

	"github.com/rostamlabs/rembed/internal/packfile"
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
	src := qwSource{
		f32: func(name string, wantShape ...int) ([]float32, error) {
			t, ok := tensors[name]
			if !ok {
				return nil, fmt.Errorf("weights %s: missing tensor %q", weightsPath, name)
			}
			if !shapeEq(t.Shape, wantShape) {
				return nil, fmt.Errorf("weights %s: tensor %q has shape %v, want %v", weightsPath, name, t.Shape, wantShape)
			}
			return t.Data, nil
		},
		dense: func(raw []float32, in, out int) denseWeight {
			return newDense(raw, nil, in, out, quantize)
		},
	}
	return loadQwen3FromSource(src, cfg, workers)
}

// qwSource abstracts where a Qwen3 weight comes from: f32 returns a tensor's
// data (from an in-RAM safetensors map, or a mmapped pack file); dense turns
// a projection's raw [out,in] weight into a denseWeight (packed for the
// in-RAM path, or kept raw pointing into the mmap for the disk path). This
// lets one loader serve both without the compute path knowing the difference.
type qwSource struct {
	f32   func(name string, shape ...int) ([]float32, error)
	dense func(raw []float32, in, out int) denseWeight
}

func shapeEq(got, want []int) bool {
	if len(got) != len(want) {
		return false
	}
	for i := range got {
		if got[i] != want[i] {
			return false
		}
	}
	return true
}

// qwen3Tensors lists every tensor a Qwen3 model needs, with its shape, in the
// order the loader consumes them — the authority for both packing (specs) and
// loading (names/shapes are re-checked on fetch, so any drift fails loudly).
func qwen3Tensors(cfg Config) []packfile.Spec {
	H, I := cfg.HiddenSize, cfg.IntermediateSize
	nq, nkv, dh := cfg.NumAttentionHeads, cfg.NumKeyValueHeads, cfg.HeadDim
	qDim, kvDim := nq*dh, nkv*dh
	specs := []packfile.Spec{
		{Name: "embed_tokens.weight", Shape: []int{cfg.VocabSize, H}},
		{Name: "norm.weight", Shape: []int{H}},
	}
	for i := range cfg.NumHiddenLayers {
		p := fmt.Sprintf("layers.%d.", i)
		specs = append(specs,
			packfile.Spec{Name: p + "input_layernorm.weight", Shape: []int{H}},
			packfile.Spec{Name: p + "self_attn.q_proj.weight", Shape: []int{qDim, H}},
			packfile.Spec{Name: p + "self_attn.k_proj.weight", Shape: []int{kvDim, H}},
			packfile.Spec{Name: p + "self_attn.v_proj.weight", Shape: []int{kvDim, H}},
			packfile.Spec{Name: p + "self_attn.o_proj.weight", Shape: []int{H, qDim}},
			packfile.Spec{Name: p + "self_attn.q_norm.weight", Shape: []int{dh}},
			packfile.Spec{Name: p + "self_attn.k_norm.weight", Shape: []int{dh}},
			packfile.Spec{Name: p + "post_attention_layernorm.weight", Shape: []int{H}},
			packfile.Spec{Name: p + "mlp.gate_proj.weight", Shape: []int{I, H}},
			packfile.Spec{Name: p + "mlp.up_proj.weight", Shape: []int{I, H}},
			packfile.Spec{Name: p + "mlp.down_proj.weight", Shape: []int{H, I}},
		)
	}
	return specs
}

func loadQwen3FromSource(src qwSource, cfg Config, workers int) (*Model, error) {
	H, I := cfg.HiddenSize, cfg.IntermediateSize
	nq, nkv, dh := cfg.NumAttentionHeads, cfg.NumKeyValueHeads, cfg.HeadDim
	qDim, kvDim := nq*dh, nkv*dh

	m := &Model{cfg: cfg, workers: workers, posOff: 0}
	m.scratchPool.New = func() any { return new(scratch) }

	var err error
	if m.wordEmb, err = src.f32("embed_tokens.weight", cfg.VocabSize, H); err != nil {
		return nil, err
	}
	if m.finalNormG, err = src.f32("norm.weight", H); err != nil {
		return nil, err
	}

	m.qwLayers = make([]qwLayer, cfg.NumHiddenLayers)
	for i := range m.qwLayers {
		l := &m.qwLayers[i]
		p := fmt.Sprintf("layers.%d.", i)
		norm := func(name string, dst *[]float32, dim int) error {
			v, e := src.f32(p+name, dim)
			*dst = v
			return e
		}
		proj := func(name string, in, out int) (denseWeight, error) {
			raw, e := src.f32(p+name, out, in)
			if e != nil {
				return denseWeight{}, e
			}
			return src.dense(raw, in, out), nil
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
		if err = norm("post_attention_layernorm.weight", &l.postNorm, H); err != nil {
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
	return m, nil
}

// PackQwen3ToDisk streams a Qwen3 checkpoint into a rembed pack file: each
// tensor is read from the (possibly sharded) safetensors, widened to
// float32, and appended — so peak memory is a single tensor, and an 8B model
// packs on a small box. The result mmaps for inference (see loadQwen3Disk).
func PackQwen3ToDisk(weightsPath, packPath string, cfg Config) error {
	r, err := safetensors.OpenReader(weightsPath)
	if err != nil {
		return err
	}
	defer func() { _ = r.Close() }()
	specs := qwen3Tensors(cfg)
	shapeOf := make(map[string][]int, len(specs))
	for _, s := range specs {
		shapeOf[s.Name] = s.Shape
	}
	return packfile.Write(packPath, specs, func(name string) ([]float32, error) {
		return r.F32(name, shapeOf[name]...)
	})
}

// LoadDisk builds a Model whose weights are memory-mapped from a rembed pack
// file in modelDir, packing the safetensors to disk on first use. Only qwen3
// is supported today — the architecture whose size (4B/8B) motivates running
// larger than RAM. The caller MUST Close the returned Model to unmap.
func LoadDisk(modelDir string, cfg Config, workers int) (*Model, error) {
	if cfg.ModelType != "qwen3" {
		return nil, fmt.Errorf("disk-backed weights are only supported for qwen3 models (got %q)", cfg.ModelType)
	}
	packPath := filepath.Join(modelDir, "weights.rembedpack")
	if _, err := os.Stat(packPath); err != nil {
		if err := PackQwen3ToDisk(filepath.Join(modelDir, "model.safetensors"), packPath, cfg); err != nil {
			return nil, fmt.Errorf("packing weights to %s: %w", packPath, err)
		}
	}
	pk, err := packfile.Open(packPath)
	if err != nil {
		return nil, err
	}
	m, err := loadQwen3Disk(pk, cfg, workers)
	if err != nil {
		_ = pk.Close()
		return nil, err
	}
	return m, nil
}

// loadQwen3Disk builds a Qwen3 Model whose weights alias the mmapped pack
// file: projections keep their raw [out,in] slice (the unpacked matmul reads
// it directly), so the resident cost is only the pages the OS keeps live.
// The Model owns the Pack and unmaps it on Close.
func loadQwen3Disk(pk *packfile.Pack, cfg Config, workers int) (*Model, error) {
	src := qwSource{
		f32: func(name string, shape ...int) ([]float32, error) {
			return pk.F32(name, shape...)
		},
		dense: func(raw []float32, in, out int) denseWeight {
			return denseWeight{raw: raw, in: in, out: out}
		},
	}
	m, err := loadQwen3FromSource(src, cfg, workers)
	if err != nil {
		return nil, err
	}
	m.pack = pk
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
