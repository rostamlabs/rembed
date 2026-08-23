// SPDX-License-Identifier: Apache-2.0

package model

import (
	"encoding/json"
	"fmt"
	"math"
	"os"
	"path/filepath"
	"runtime"
	"sort"
	"strings"

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
// packs on a small box. source is a fingerprint of the input weights, stored
// so a later load can detect a stale pack. The result mmaps for inference.
func PackQwen3ToDisk(weightsPath, packPath, source string, cfg Config) error {
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
	return packfile.Write(packPath, source, specs, func(name string) ([]float32, error) {
		return r.F32(name, shapeOf[name]...)
	})
}

// sourceFingerprint identifies the model dir's weight files by name, size,
// and mtime — enough to detect that the safetensors changed under a cached
// pack file (a re-download or retrain with the same shapes, which the
// per-tensor shape check alone would not catch).
func sourceFingerprint(modelDir string) (string, error) {
	var files []string
	single := filepath.Join(modelDir, "model.safetensors")
	if _, err := os.Stat(single); err == nil {
		files = []string{single}
	} else {
		idxPath := filepath.Join(modelDir, "model.safetensors.index.json")
		raw, err := os.ReadFile(idxPath)
		if err != nil {
			return "", err
		}
		var index struct {
			WeightMap map[string]string `json:"weight_map"`
		}
		if err := json.Unmarshal(raw, &index); err != nil {
			return "", err
		}
		files = append(files, idxPath)
		seen := map[string]struct{}{}
		for _, f := range index.WeightMap {
			if !safetensors.ValidShardName(f) {
				return "", fmt.Errorf("unsafe shard name %q in index", f)
			}
			if _, ok := seen[f]; ok {
				continue
			}
			seen[f] = struct{}{}
			files = append(files, filepath.Join(modelDir, f))
		}
	}
	sort.Strings(files)
	var b strings.Builder
	for _, f := range files {
		fi, err := os.Stat(f)
		if err != nil {
			return "", err
		}
		fmt.Fprintf(&b, "%s:%d:%d\n", filepath.Base(f), fi.Size(), fi.ModTime().UnixNano())
	}
	return b.String(), nil
}

// LoadDisk builds a Model whose weights are memory-mapped from a rembed pack
// file in modelDir, packing the safetensors to disk on first use (and
// rebuilding a stale pack whose source fingerprint no longer matches). Only
// qwen3 is supported today — the architecture whose size (4B/8B) motivates
// running larger than RAM. The caller MUST Close the returned Model to unmap.
func LoadDisk(modelDir string, cfg Config, workers int) (*Model, error) {
	if cfg.ModelType != "qwen3" {
		return nil, fmt.Errorf("disk-backed weights are only supported for qwen3 models (got %q)", cfg.ModelType)
	}
	packPath := filepath.Join(modelDir, "weights.rembedpack")
	fp, err := sourceFingerprint(modelDir)
	if err != nil {
		return nil, fmt.Errorf("fingerprinting weights: %w", err)
	}
	// Reuse an existing pack only if its recorded source matches; otherwise
	// (missing or stale) rebuild it.
	if pk, err := packfile.Open(packPath); err == nil {
		if pk.Source() == fp {
			m, err := loadQwen3Disk(pk, cfg, workers)
			if err != nil {
				_ = pk.Close()
				return nil, err
			}
			return m, nil
		}
		_ = pk.Close()
	}
	if err := PackQwen3ToDisk(filepath.Join(modelDir, "model.safetensors"), packPath, fp, cfg); err != nil {
		return nil, fmt.Errorf("packing weights to %s: %w", packPath, err)
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

// flashBlk is the query/key block size for flash-attention.
const flashBlk = 64

// flashAttnCausalHead computes causal scaled-dot-product attention for one
// head with online (streaming) softmax, writing [seq×dh] into out. It equals
// softmax(mask(Q·Kᵀ·scale))·V but never materializes the full seq×seq score
// matrix and SKIPS key blocks beyond each query block — the causal upper
// triangle rembed used to compute and then throw away — roughly halving the
// attention matmul work. qH/kH/vH are [seq×dh] row-major (key-contiguous);
// sblk (>= bq·bq), acc (>= bq·dh), mrow/lrow (>= bq) are per-head scratch
// with bq=min(flashBlk,seq). Online softmax is numerically exact; only the
// fp32 accumulation order differs from a full-row softmax (within tolerance).
func flashAttnCausalHead(qH, kH, vH, out []float32, seq, dh int, scale float32, sblk, acc, mrow, lrow []float32) {
	const negInf = float32(-1e30)
	for q0 := 0; q0 < seq; q0 += flashBlk {
		q1 := min(q0+flashBlk, seq)
		nq := q1 - q0
		for r := 0; r < nq; r++ {
			mrow[r], lrow[r] = negInf, 0
		}
		clear(acc[:nq*dh])
		// Keys only up to q1: no query in [q0,q1) attends past q1-1 (causal),
		// so every key block at k0 >= q1 is skipped entirely.
		for k0 := 0; k0 < q1; k0 += flashBlk {
			k1 := min(k0+flashBlk, q1)
			nk := k1 - k0
			// S = Qblk[nq×dh] · Kblk[nk×dh]ᵀ (kH row-major is the bT layout).
			tensor.MatMulSerial(sblk, qH[q0*dh:q1*dh], kH[k0*dh:k1*dh], nq, dh, nk)
			for r := 0; r < nq; r++ {
				i := q0 + r
				srow := sblk[r*nk : r*nk+nk]
				rmax := negInf
				for c := 0; c < nk; c++ {
					if k0+c > i {
						srow[c] = negInf // causal (diagonal block only)
					} else {
						srow[c] *= scale
						if srow[c] > rmax {
							rmax = srow[c]
						}
					}
				}
				if rmax == negInf {
					continue // whole block is in row i's future
				}
				mprev := mrow[r]
				mnew := mprev
				if rmax > mnew {
					mnew = rmax
				}
				ar := acc[r*dh : r*dh+dh]
				if mprev != negInf && mnew != mprev {
					// A new running max: rescale the accumulator and denom.
					corr := tensor.ExpNeg(mprev - mnew)
					lrow[r] *= corr
					for d := range ar {
						ar[d] *= corr
					}
				}
				lr := lrow[r]
				for c := 0; c < nk; c++ {
					if k0+c > i {
						continue
					}
					p := tensor.ExpNeg(srow[c] - mnew)
					lr += p
					vk := vH[(k0+c)*dh : (k0+c)*dh+dh]
					for d := range ar {
						ar[d] += p * vk[d]
					}
				}
				lrow[r], mrow[r] = lr, mnew
			}
		}
		for r := 0; r < nq; r++ {
			ar := acc[r*dh : r*dh+dh]
			inv := 1 / lrow[r]
			o := out[(q0+r)*dh : (q0+r)*dh+dh]
			for d := range o {
				o[d] = ar[d] * inv
			}
		}
	}
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
	s.qwVHead = grow(s.qwVHead, nkv*seq*dh)
	s.qwQHead = grow(s.qwQHead, nq*seq*dh)
	s.qwCHead = grow(s.qwCHead, nq*seq*dh)
	s.qwCtx = grow(s.qwCtx, seq*qDim)
	// Flash-attention per-head scratch.
	bq := min(flashBlk, seq)
	s.qwSblk = grow(s.qwSblk, nq*bq*bq)
	s.qwAcc = grow(s.qwAcc, nq*bq*dh)
	s.qwM = grow(s.qwM, nq*bq)
	s.qwL = grow(s.qwL, nq*bq)
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
	kHead, vHead := s.qwKHead, s.qwVHead
	qHead, cHead := s.qwQHead, s.qwCHead
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
			vH := vHead[kv*seq*dh : (kv+1)*seq*dh]
			for i := range seq {
				copy(kH[i*dh:i*dh+dh], k[i*kvDim+kv*dh:i*kvDim+kv*dh+dh])
				// V stored row-major (key-contiguous) so flash-attention's
				// P·V is a contiguous axpy that can skip future keys.
				copy(vH[i*dh:i*dh+dh], v[i*kvDim+kv*dh:i*kvDim+kv*dh+dh])
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
			vH := vHead[kv*seq*dh : (kv+1)*seq*dh]
			chH := cHead[hh*seq*dh : (hh+1)*seq*dh]
			bq := min(flashBlk, seq)
			flashAttnCausalHead(qH, kH, vH, chH, seq, dh, scale,
				s.qwSblk[hh*bq*bq:(hh+1)*bq*bq],
				s.qwAcc[hh*bq*dh:(hh+1)*bq*dh],
				s.qwM[hh*bq:(hh+1)*bq], s.qwL[hh*bq:(hh+1)*bq])
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
