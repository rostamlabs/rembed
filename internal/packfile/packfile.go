// SPDX-License-Identifier: Apache-2.0

// Package packfile stores a model's weights, already widened to float32, in
// one mmap-friendly file: an 8-byte little-endian header length, a JSON
// header naming each tensor and its byte range, then the contiguous f32
// data. Writing streams one tensor at a time (peak memory is a single
// tensor, so an 8B model packs on a small box); reading mmaps the file and
// hands out []float32 slices that point INTO the mapping (zero copy), so the
// OS pages weights in on access and evicts under pressure — letting rembed
// run a model larger than RAM.
//
// Little-endian only (rembed's SIMD kernels are amd64/arm64, both LE); the
// f32 bytes are reinterpreted directly, matching safetensors' own layout.
package packfile

import (
	"bufio"
	"encoding/binary"
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"unsafe"

	"github.com/rostamlabs/rembed/internal/mmapfile"
)

const magic = 1 // header "rembedpack" version

// maxInt is the largest value the platform int holds — a >2GB tensor's
// element count overflows int on 32-bit (linux/386), so guard before the
// int64→int narrowing rather than wrap silently.
const maxInt = int64(^uint(0) >> 1)

// Spec names a tensor to write and its shape (row-major).
type Spec struct {
	Name  string
	Shape []int
}

type entry struct {
	Name  string `json:"name"`
	Shape []int  `json:"shape"`
	Off   int64  `json:"off"`    // byte offset within the data section
	NByte int64  `json:"nbytes"` // = prod(shape) * 4
}

type header struct {
	RembedPack int     `json:"rembedpack"`
	Source     string  `json:"source,omitempty"` // fingerprint of the source weights (for staleness)
	Tensors    []entry `json:"tensors"`
}

// dataStart returns the 64-byte-aligned offset where the data section
// begins, given the JSON header's byte length (the 8-byte length prefix
// precedes the header).
func dataStart(headerLen int64) int64 {
	off := 8 + headerLen
	return (off + 63) &^ 63
}

func numElem(shape []int) int64 {
	n := int64(1)
	for _, d := range shape {
		n *= int64(d)
	}
	return n
}

// Write creates a pack file at path for the given specs. provide(name)
// returns that tensor's float32 data (exactly prod(shape) elements); it is
// called once per spec in order, so the caller can convert lazily and keep
// only one tensor resident at a time. source is an opaque fingerprint of the
// input weights, stored in the header so a stale pack can be detected.
func Write(path, source string, specs []Spec, provide func(name string) ([]float32, error)) (err error) {
	h := header{RembedPack: magic, Source: source}
	var off int64
	for _, s := range specs {
		nb := numElem(s.Shape) * 4
		h.Tensors = append(h.Tensors, entry{Name: s.Name, Shape: s.Shape, Off: off, NByte: nb})
		off += nb
	}
	hb, err := json.Marshal(h)
	if err != nil {
		return err
	}

	// A unique temp (not a fixed path+".tmp") so two concurrent packs of the
	// same model dir cannot interleave writes into one file; the rename at the
	// end is atomic.
	f, err := os.CreateTemp(filepath.Dir(path), ".rembedpack-*")
	if err != nil {
		return err
	}
	tmp := f.Name()
	defer func() {
		if err != nil {
			_ = f.Close()
			_ = os.Remove(tmp)
		}
	}()
	w := bufio.NewWriterSize(f, 1<<20)
	var lenbuf [8]byte
	binary.LittleEndian.PutUint64(lenbuf[:], uint64(len(hb)))
	if _, err = w.Write(lenbuf[:]); err != nil {
		return err
	}
	if _, err = w.Write(hb); err != nil {
		return err
	}
	// Pad so the data section starts on a 64-byte boundary: mmap's base is
	// page-aligned, so the reinterpreted []float32 is then aligned too
	// (matters on arm64, and lets SIMD loads stay aligned).
	if pad := int(dataStart(int64(len(hb))) - int64(8+len(hb))); pad > 0 {
		if _, err = w.Write(make([]byte, pad)); err != nil {
			return err
		}
	}
	for _, s := range specs {
		vals, perr := provide(s.Name)
		if perr != nil {
			return perr
		}
		if int64(len(vals)) != numElem(s.Shape) {
			return fmt.Errorf("packfile: %s produced %d elems, want %d", s.Name, len(vals), numElem(s.Shape))
		}
		if len(vals) > 0 {
			b := unsafe.Slice((*byte)(unsafe.Pointer(&vals[0])), len(vals)*4)
			if _, err = w.Write(b); err != nil {
				return err
			}
		}
	}
	if err = w.Flush(); err != nil {
		return err
	}
	if err = f.Close(); err != nil {
		return err
	}
	return os.Rename(tmp, path)
}

// Pack is an mmapped pack file. F32 slices are valid until Close.
type Pack struct {
	mf      *mmapfile.File
	source  string
	dataOff int64
	entries map[string]entry
}

// Open mmaps a pack file written by Write.
func Open(path string) (*Pack, error) {
	mf, err := mmapfile.Open(path)
	if err != nil {
		return nil, err
	}
	raw := mf.Data()
	if len(raw) < 8 {
		_ = mf.Close()
		return nil, fmt.Errorf("packfile %s: too short", path)
	}
	hlen := binary.LittleEndian.Uint64(raw)
	if hlen > uint64(len(raw)-8) {
		_ = mf.Close()
		return nil, fmt.Errorf("packfile %s: header length %d exceeds file", path, hlen)
	}
	var h header
	if err := json.Unmarshal(raw[8:8+hlen], &h); err != nil {
		_ = mf.Close()
		return nil, fmt.Errorf("packfile %s: bad header: %w", path, err)
	}
	if h.RembedPack != magic {
		_ = mf.Close()
		return nil, fmt.Errorf("packfile %s: version %d, want %d", path, h.RembedPack, magic)
	}
	p := &Pack{mf: mf, source: h.Source, dataOff: dataStart(int64(hlen)), entries: make(map[string]entry, len(h.Tensors))}
	dataLen := int64(len(raw)) - p.dataOff
	for _, e := range h.Tensors {
		// Validate every entry up front so F32's unsafe reinterpret is always
		// on a well-formed, in-bounds, 4-byte-aligned range even if the pack
		// is truncated or corrupt (Write guarantees these; Open must not trust
		// them blindly).
		if e.Off%4 != 0 || e.NByte%4 != 0 {
			_ = mf.Close()
			return nil, fmt.Errorf("packfile %s: tensor %q offset/length not 4-byte aligned", path, e.Name)
		}
		if e.NByte != numElem(e.Shape)*4 {
			_ = mf.Close()
			return nil, fmt.Errorf("packfile %s: tensor %q length %d != shape %v", path, e.Name, e.NByte, e.Shape)
		}
		if e.NByte/4 > maxInt {
			_ = mf.Close()
			return nil, fmt.Errorf("packfile %s: tensor %q too large for this platform's int", path, e.Name)
		}
		if e.Off < 0 || e.NByte < 0 || e.Off+e.NByte > dataLen {
			_ = mf.Close()
			return nil, fmt.Errorf("packfile %s: tensor %q range [%d,%d) out of data section %d", path, e.Name, e.Off, e.Off+e.NByte, dataLen)
		}
		p.entries[e.Name] = e
	}
	return p, nil
}

// Source returns the fingerprint of the weights this pack was built from
// (empty if none was recorded), for staleness checks.
func (p *Pack) Source() string { return p.source }

// Shape returns a tensor's dimensions and whether it exists.
func (p *Pack) Shape(name string) ([]int, bool) {
	e, ok := p.entries[name]
	if !ok {
		return nil, false
	}
	return e.Shape, true
}

// F32 returns tensor `name` as a []float32 aliasing the mmap (zero copy),
// validating it against wantShape. The slice is read-only and valid until
// Close.
func (p *Pack) F32(name string, wantShape ...int) ([]float32, error) {
	e, ok := p.entries[name]
	if !ok {
		return nil, fmt.Errorf("packfile: missing tensor %q", name)
	}
	if len(e.Shape) != len(wantShape) {
		return nil, fmt.Errorf("packfile: tensor %q shape %v, want %v", name, e.Shape, wantShape)
	}
	for i, d := range wantShape {
		if e.Shape[i] != d {
			return nil, fmt.Errorf("packfile: tensor %q shape %v, want %v", name, e.Shape, wantShape)
		}
	}
	raw := p.mf.Data()
	start := p.dataOff + e.Off
	end := start + e.NByte
	if start < 0 || end > int64(len(raw)) {
		return nil, fmt.Errorf("packfile: tensor %q range [%d,%d) out of file %d", name, start, end, len(raw))
	}
	n := int(e.NByte / 4)
	if n == 0 {
		return nil, nil
	}
	return unsafe.Slice((*float32)(unsafe.Pointer(&raw[start])), n), nil
}

// Close unmaps the file.
func (p *Pack) Close() error { return p.mf.Close() }
