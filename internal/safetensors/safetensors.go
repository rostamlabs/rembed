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
		buf := data[start:end]
		vals := make([]float32, n)
		switch e.Dtype {
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
		out[name] = Tensor{Shape: e.Shape, Data: vals}
	}
	return out, nil
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
