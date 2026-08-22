// SPDX-License-Identifier: Apache-2.0

// Package safetensors reads the safetensors weight format: an 8-byte
// little-endian header length, a JSON header mapping tensor names to
// {dtype, shape, data_offsets}, then the raw tensor data. v1 supports F32 only.
package safetensors

import (
	"encoding/binary"
	"encoding/json"
	"fmt"
	"math"
	"os"
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
		if e.Dtype != "F32" {
			// Skip non-F32 tensors (e.g. the I64 position_ids buffer BERT
			// checkpoints carry). Real weights are all F32 in the supported
			// models; a missing weight surfaces as a clear error in the
			// model loader.
			continue
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
		if uint64(n)*4 != end-start {
			return nil, fmt.Errorf("safetensors %s: tensor %q shape %v (%d floats) does not match byte range %d", path, name, e.Shape, n, end-start)
		}
		buf := data[start:end]
		vals := make([]float32, n)
		for i := range vals {
			vals[i] = math.Float32frombits(binary.LittleEndian.Uint32(buf[i*4:]))
		}
		out[name] = Tensor{Shape: e.Shape, Data: vals}
	}
	return out, nil
}
