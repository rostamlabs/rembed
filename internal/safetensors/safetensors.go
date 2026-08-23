// SPDX-License-Identifier: Apache-2.0

// Package safetensors reads the safetensors weight format: an 8-byte
// little-endian header length, a JSON header mapping tensor names to
// {dtype, shape, data_offsets}, then the raw tensor data. F32 loads
// directly; F16 and BF16 checkpoints (common on the Hub — gte ships F16)
// are converted to float32 at load, so the engine always computes in f32.
package safetensors

import (
	"encoding/binary"
	"encoding/json"
	"fmt"
	"math"
	"os"
	"path/filepath"
	"sort"
	"strings"
)

// Tensor is one named weight: Shape and its row-major F32 data.
type Tensor struct {
	Shape []int
	Data  []float32
}

// NumElements returns the product of the tensor's dimensions.
func (t Tensor) NumElements() int {
	n := 1
	for _, d := range t.Shape {
		n *= d
	}
	return n
}

type headerEntry struct {
	Dtype       string    `json:"dtype"`
	Shape       []int     `json:"shape"`
	DataOffsets [2]uint64 `json:"data_offsets"`
}

// LoadAny loads weights from a single model.safetensors file, or — when
// that file is absent but a model.safetensors.index.json sits beside it —
// from the sharded set the index names. Large checkpoints ship 2+ shards
// (Qwen3-4B: 2, 8B: 4); this loads them through the same path as
// single-file models. path is the expected single-file location; its
// directory is where the index and shards are looked for.
func LoadAny(path string) (map[string]Tensor, error) {
	if _, err := os.Stat(path); err == nil {
		return Load(path)
	}
	idx := filepath.Join(filepath.Dir(path), "model.safetensors.index.json")
	if _, err := os.Stat(idx); err != nil {
		return nil, fmt.Errorf("safetensors: %s not found (and no model.safetensors.index.json beside it)", path)
	}
	return loadSharded(idx)
}

// loadSharded reads a HuggingFace sharded checkpoint: model.safetensors.index.json
// maps every tensor name to the shard file that holds it; each distinct
// shard is loaded once and the maps merged. Shards load in sorted order so
// a partial-load failure is deterministic.
func loadSharded(indexPath string) (map[string]Tensor, error) {
	raw, err := os.ReadFile(indexPath)
	if err != nil {
		return nil, err
	}
	var idx struct {
		WeightMap map[string]string `json:"weight_map"`
	}
	if err := json.Unmarshal(raw, &idx); err != nil {
		return nil, fmt.Errorf("safetensors %s: bad index JSON: %w", indexPath, err)
	}
	if len(idx.WeightMap) == 0 {
		return nil, fmt.Errorf("safetensors %s: index weight_map is empty", indexPath)
	}
	shardSet := make(map[string]struct{})
	for _, f := range idx.WeightMap {
		shardSet[f] = struct{}{}
	}
	shards := make([]string, 0, len(shardSet))
	for f := range shardSet {
		shards = append(shards, f)
	}
	sort.Strings(shards)
	dir := filepath.Dir(indexPath)
	out := make(map[string]Tensor, len(idx.WeightMap))
	for _, f := range shards {
		m, err := Load(filepath.Join(dir, filepath.FromSlash(f)))
		if err != nil {
			return nil, err
		}
		for k, v := range m {
			out[k] = v
		}
	}
	for name := range idx.WeightMap {
		if _, ok := out[name]; !ok {
			return nil, fmt.Errorf("safetensors %s: index names %q but no shard provided it", indexPath, name)
		}
	}
	return out, nil
}

// Load reads every tensor in the file into memory. Offsets are validated
// against the data section before any allocation sized from them.
func Load(path string) (map[string]Tensor, error) {
	raw, err := os.ReadFile(path)
	if err != nil {
		return nil, err
	}
	if len(raw) < 8 {
		return nil, fmt.Errorf("safetensors %s: file too short for header length", path)
	}
	headerLen := binary.LittleEndian.Uint64(raw)
	if headerLen > uint64(len(raw)-8) {
		return nil, fmt.Errorf("safetensors %s: header length %d exceeds file size %d", path, headerLen, len(raw))
	}
	headerBytes := raw[8 : 8+headerLen]
	data := raw[8+headerLen:]

	var header map[string]json.RawMessage
	if err := json.Unmarshal(headerBytes, &header); err != nil {
		return nil, fmt.Errorf("safetensors %s: bad JSON header: %w", path, err)
	}

	out := make(map[string]Tensor, len(header))
	for name, rawEntry := range header {
		if name == "__metadata__" {
			continue
		}
		var e headerEntry
		if err := json.Unmarshal(rawEntry, &e); err != nil {
			return nil, fmt.Errorf("safetensors %s: bad entry %q: %w", path, name, err)
		}
		var elemSize int
		switch e.Dtype {
		case "F32":
			elemSize = 4
		case "F16", "BF16":
			elemSize = 2
		default:
			// The I64 position_ids buffer BERT checkpoints carry is not a
			// weight; skip it. Any OTHER unsupported dtype fails here with
			// the real reason rather than as a confusing "missing tensor"
			// later.
			if strings.HasSuffix(name, "position_ids") {
				continue
			}
			return nil, fmt.Errorf("safetensors %s: tensor %q has dtype %s; supported: F32, F16, BF16", path, name, e.Dtype)
		}
		start, end := e.DataOffsets[0], e.DataOffsets[1]
		if start > end || end > uint64(len(data)) {
			return nil, fmt.Errorf("safetensors %s: tensor %q offsets [%d,%d) out of range (data size %d)", path, name, start, end, len(data))
		}
		n := 1
		for _, d := range e.Shape {
			if d < 0 {
				return nil, fmt.Errorf("safetensors %s: tensor %q has negative dimension", path, name)
			}
			n *= d
		}
		if uint64(n)*uint64(elemSize) != end-start {
			return nil, fmt.Errorf("safetensors %s: tensor %q shape %v (%d elems × %d B) does not match byte range %d", path, name, e.Shape, n, elemSize, end-start)
		}
		out[name] = Tensor{Shape: e.Shape, Data: decodeF32(e.Dtype, data[start:end], n)}
	}
	return out, nil
}

// decodeF32 widens a tensor's raw bytes (F32/F16/BF16, little-endian) into a
// fresh float32 slice of n elements. Shared by Load and the mmap Reader.
func decodeF32(dtype string, buf []byte, n int) []float32 {
	vals := make([]float32, n)
	switch dtype {
	case "F32":
		for i := range vals {
			vals[i] = math.Float32frombits(binary.LittleEndian.Uint32(buf[i*4:]))
		}
	case "F16":
		for i := range vals {
			vals[i] = f16to32(binary.LittleEndian.Uint16(buf[i*2:]))
		}
	case "BF16":
		for i := range vals {
			// bfloat16 is float32's top 16 bits.
			vals[i] = math.Float32frombits(uint32(binary.LittleEndian.Uint16(buf[i*2:])) << 16)
		}
	}
	return vals
}

// f16to32 widens an IEEE-754 half-precision value to float32, covering
// normals, subnormals, zeros, infinities, and NaN.
func f16to32(h uint16) float32 {
	sign := uint32(h>>15) << 31
	exp := uint32(h>>10) & 0x1f
	man := uint32(h) & 0x3ff
	switch exp {
	case 0:
		if man == 0 {
			return math.Float32frombits(sign) // ±0
		}
		// Subnormal half: renormalize into a float32 normal.
		e := uint32(127 - 15 + 1)
		for man&0x400 == 0 {
			man <<= 1
			e--
		}
		return math.Float32frombits(sign | e<<23 | (man&0x3ff)<<13)
	case 0x1f:
		return math.Float32frombits(sign | 0xff<<23 | man<<13) // ±Inf / NaN
	default:
		return math.Float32frombits(sign | (exp+127-15)<<23 | man<<13)
	}
}
