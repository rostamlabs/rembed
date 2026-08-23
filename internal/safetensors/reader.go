// SPDX-License-Identifier: Apache-2.0

package safetensors

import (
	"encoding/binary"
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"sort"

	"github.com/rostamlabs/rembed/internal/mmapfile"
)

// Reader gives on-demand access to a safetensors checkpoint (single file or
// sharded) without materializing every tensor: the file is mmapped and each
// tensor is widened to float32 only when requested. This is what lets a
// large model be packed to disk one tensor at a time, so peak memory is a
// single tensor rather than the whole model.
type Reader struct {
	maps    []*mmapfile.File
	entries map[string]readerEntry
}

type readerEntry struct {
	dtype string
	shape []int
	buf   []byte // slice into one of the mmaps
}

// OpenReader mmaps the checkpoint at path (a model.safetensors), or the
// sharded set named by a sibling model.safetensors.index.json when the
// single file is absent. Close releases the mappings.
func OpenReader(path string) (*Reader, error) {
	var files []string
	if _, err := os.Stat(path); err == nil {
		files = []string{path}
	} else {
		idx := filepath.Join(filepath.Dir(path), "model.safetensors.index.json")
		raw, err := os.ReadFile(idx)
		if err != nil {
			return nil, fmt.Errorf("safetensors: %s not found (and no index beside it)", path)
		}
		var index struct {
			WeightMap map[string]string `json:"weight_map"`
		}
		if err := json.Unmarshal(raw, &index); err != nil {
			return nil, fmt.Errorf("safetensors %s: bad index JSON: %w", idx, err)
		}
		set := make(map[string]struct{})
		for _, f := range index.WeightMap {
			set[f] = struct{}{}
		}
		dir := filepath.Dir(idx)
		for f := range set {
			files = append(files, filepath.Join(dir, filepath.FromSlash(f)))
		}
		sort.Strings(files)
	}

	r := &Reader{entries: make(map[string]readerEntry)}
	for _, f := range files {
		if err := r.addFile(f); err != nil {
			_ = r.Close()
			return nil, err
		}
	}
	return r, nil
}

func (r *Reader) addFile(path string) error {
	mf, err := mmapfile.Open(path)
	if err != nil {
		return err
	}
	r.maps = append(r.maps, mf)
	raw := mf.Data()
	if len(raw) < 8 {
		return fmt.Errorf("safetensors %s: file too short", path)
	}
	headerLen := binary.LittleEndian.Uint64(raw)
	if headerLen > uint64(len(raw)-8) {
		return fmt.Errorf("safetensors %s: header length %d exceeds file size", path, headerLen)
	}
	var header map[string]json.RawMessage
	if err := json.Unmarshal(raw[8:8+headerLen], &header); err != nil {
		return fmt.Errorf("safetensors %s: bad JSON header: %w", path, err)
	}
	data := raw[8+headerLen:]
	for name, rawEntry := range header {
		if name == "__metadata__" {
			continue
		}
		var e headerEntry
		if err := json.Unmarshal(rawEntry, &e); err != nil {
			return fmt.Errorf("safetensors %s: bad entry %q: %w", path, name, err)
		}
		switch e.Dtype {
		case "F32", "F16", "BF16":
		default:
			continue // skip non-weight buffers (e.g. I64 position_ids)
		}
		start, end := e.DataOffsets[0], e.DataOffsets[1]
		if start > end || end > uint64(len(data)) {
			return fmt.Errorf("safetensors %s: tensor %q offsets out of range", path, name)
		}
		r.entries[name] = readerEntry{dtype: e.Dtype, shape: e.Shape, buf: data[start:end]}
	}
	return nil
}

// Shape returns the tensor's dimensions and whether it exists.
func (r *Reader) Shape(name string) ([]int, bool) {
	e, ok := r.entries[name]
	if !ok {
		return nil, false
	}
	return e.shape, true
}

// F32 widens tensor `name` to a fresh float32 slice (the only allocation is
// that slice). It returns an error if the shape does not match wantShape.
func (r *Reader) F32(name string, wantShape ...int) ([]float32, error) {
	e, ok := r.entries[name]
	if !ok {
		return nil, fmt.Errorf("safetensors: missing tensor %q", name)
	}
	if len(e.shape) != len(wantShape) {
		return nil, fmt.Errorf("safetensors: tensor %q shape %v, want %v", name, e.shape, wantShape)
	}
	n := 1
	for i, d := range wantShape {
		if e.shape[i] != d {
			return nil, fmt.Errorf("safetensors: tensor %q shape %v, want %v", name, e.shape, wantShape)
		}
		n *= d
	}
	elem := 4
	if e.dtype != "F32" {
		elem = 2
	}
	if n*elem != len(e.buf) {
		return nil, fmt.Errorf("safetensors: tensor %q byte length %d != %d elems × %d", name, len(e.buf), n, elem)
	}
	return decodeF32(e.dtype, e.buf, n), nil
}

// Close releases all mmaps.
func (r *Reader) Close() error {
	var err error
	for _, m := range r.maps {
		if e := m.Close(); e != nil && err == nil {
			err = e
		}
	}
	r.maps = nil
	return err
}
