// SPDX-License-Identifier: Apache-2.0

package packfile

import (
	"os"
	"path/filepath"
	"testing"
)

// TestRoundTrip writes a few tensors and reads them back through the mmap
// path, checking values, shapes, alignment (F32 must not corrupt), and that
// a missing tensor and a wrong shape are reported.
func TestRoundTrip(t *testing.T) {
	dir := t.TempDir()
	path := filepath.Join(dir, "w.rembedpack")
	src := map[string][]float32{
		"a":     {1, 2, 3, 4, 5, 6},    // [2,3]
		"scale": {0.5},                 // [1]
		"big":   make([]float32, 1000), // [10,100]
		"empty": {},                    // [0]
	}
	for i := range src["big"] {
		src["big"][i] = float32(i) * 0.25
	}
	specs := []Spec{
		{Name: "a", Shape: []int{2, 3}},
		{Name: "scale", Shape: []int{1}},
		{Name: "big", Shape: []int{10, 100}},
		{Name: "empty", Shape: []int{0}},
	}
	if err := Write(path, "src-fingerprint-v1", specs, func(name string) ([]float32, error) {
		return src[name], nil
	}); err != nil {
		t.Fatal(err)
	}

	p, err := Open(path)
	if err != nil {
		t.Fatal(err)
	}
	defer func() { _ = p.Close() }()

	for _, s := range specs {
		got, err := p.F32(s.Name, s.Shape...)
		if err != nil {
			t.Fatalf("%s: %v", s.Name, err)
		}
		want := src[s.Name]
		if len(got) != len(want) {
			t.Fatalf("%s: len %d, want %d", s.Name, len(got), len(want))
		}
		for i := range want {
			if got[i] != want[i] {
				t.Fatalf("%s[%d]=%v, want %v", s.Name, i, got[i], want[i])
			}
		}
	}
	if p.Source() != "src-fingerprint-v1" {
		t.Errorf("Source() = %q, want the stored fingerprint", p.Source())
	}
	if _, err := p.F32("missing", 1); err == nil {
		t.Fatal("expected error for missing tensor")
	}
	if _, err := p.F32("a", 3, 2); err == nil {
		t.Fatal("expected shape-mismatch error")
	}
}

// TestOpenRejectsCorruptHeader: Open must validate offsets/lengths rather
// than trust a (possibly truncated) header, so F32's unsafe reinterpret is
// always on a well-formed range.
func TestOpenRejectsCorruptHeader(t *testing.T) {
	dir := t.TempDir()
	path := filepath.Join(dir, "w.rembedpack")
	if err := Write(path, "fp", []Spec{{Name: "a", Shape: []int{4}}}, func(string) ([]float32, error) {
		return []float32{1, 2, 3, 4}, nil
	}); err != nil {
		t.Fatal(err)
	}
	// Truncate the file so tensor "a"'s recorded range runs past EOF.
	fi, _ := os.Stat(path)
	if err := os.Truncate(path, fi.Size()-8); err != nil {
		t.Fatal(err)
	}
	if p, err := Open(path); err == nil {
		_ = p.Close()
		t.Fatal("expected Open to reject a truncated pack")
	}
}
