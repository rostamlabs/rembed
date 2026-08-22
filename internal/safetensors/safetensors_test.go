// SPDX-License-Identifier: Apache-2.0

package safetensors

import (
	"encoding/binary"
	"encoding/json"
	"math"
	"os"
	"path/filepath"
	"testing"
)

// writeFile builds a safetensors file from a JSON header string and raw data.
func writeFile(t *testing.T, header string, data []byte) string {
	t.Helper()
	p := filepath.Join(t.TempDir(), "w.safetensors")
	buf := make([]byte, 8)
	binary.LittleEndian.PutUint64(buf, uint64(len(header)))
	buf = append(buf, header...)
	buf = append(buf, data...)
	if err := os.WriteFile(p, buf, 0o600); err != nil {
		t.Fatal(err)
	}
	return p
}

func f32bytes(vals ...float32) []byte {
	out := make([]byte, 4*len(vals))
	for i, v := range vals {
		binary.LittleEndian.PutUint32(out[i*4:], math.Float32bits(v))
	}
	return out
}

func TestLoadRoundTrip(t *testing.T) {
	data := f32bytes(1, 2, 3, 4, 5, 6)
	header, _ := json.Marshal(map[string]any{
		"__metadata__": map[string]string{"format": "pt"},
		"w": map[string]any{
			"dtype": "F32", "shape": []int{2, 3}, "data_offsets": []int{0, 24},
		},
	})
	tensors, err := Load(writeFile(t, string(header), data))
	if err != nil {
		t.Fatal(err)
	}
	w, ok := tensors["w"]
	if !ok {
		t.Fatalf("missing tensor w, got %v", tensors)
	}
	if len(w.Shape) != 2 || w.Shape[0] != 2 || w.Shape[1] != 3 {
		t.Fatalf("shape=%v want [2 3]", w.Shape)
	}
	for i, want := range []float32{1, 2, 3, 4, 5, 6} {
		if w.Data[i] != want {
			t.Fatalf("data=%v", w.Data)
		}
	}
}

func TestLoadSkipsPositionIDsBufferOnly(t *testing.T) {
	// BERT's I64 position_ids buffer is skipped, not fatal; F32 tensors in
	// the same file still load.
	header := `{"embeddings.position_ids":{"dtype":"I64","shape":[1],"data_offsets":[0,8]},` +
		`"w":{"dtype":"F32","shape":[1],"data_offsets":[8,12]}}`
	data := append(make([]byte, 8), f32bytes(7)...)
	tensors, err := Load(writeFile(t, header, data))
	if err != nil {
		t.Fatal(err)
	}
	if _, ok := tensors["embeddings.position_ids"]; ok {
		t.Fatal("position_ids buffer should be skipped")
	}
	if w := tensors["w"]; len(w.Data) != 1 || w.Data[0] != 7 {
		t.Fatalf("w=%v", tensors["w"])
	}

	// Any OTHER non-F32 tensor (e.g. an fp16 weight) must fail loudly with
	// the dtype as the reason, not surface later as "missing tensor".
	badHeader := `{"embeddings.word_embeddings.weight":{"dtype":"F16","shape":[1],"data_offsets":[0,2]}}`
	if _, err := Load(writeFile(t, badHeader, []byte{0, 0})); err == nil {
		t.Fatal("expected dtype error for non-F32 weight")
	}
}

func TestLoadRejectsBadOffsets(t *testing.T) {
	// Offsets point past the end of the data section.
	header := `{"w":{"dtype":"F32","shape":[4],"data_offsets":[0,16]}}`
	if _, err := Load(writeFile(t, header, f32bytes(1, 2))); err == nil {
		t.Fatal("expected offset error")
	}
	// Shape does not match the byte range.
	header2 := `{"w":{"dtype":"F32","shape":[3],"data_offsets":[0,8]}}`
	if _, err := Load(writeFile(t, header2, f32bytes(1, 2))); err == nil {
		t.Fatal("expected shape/range mismatch error")
	}
}

func TestLoadRejectsTruncatedHeader(t *testing.T) {
	p := filepath.Join(t.TempDir(), "bad.safetensors")
	// Header length claims 100 bytes but the file ends immediately.
	buf := make([]byte, 8)
	binary.LittleEndian.PutUint64(buf, 100)
	if err := os.WriteFile(p, buf, 0o600); err != nil {
		t.Fatal(err)
	}
	if _, err := Load(p); err == nil {
		t.Fatal("expected header length error")
	}
}
