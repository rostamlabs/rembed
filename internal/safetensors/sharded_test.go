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

// writeShard writes a minimal single-tensor F32 safetensors file.
func writeShard(t *testing.T, path, name string, vals []float32) {
	t.Helper()
	hdr := map[string]any{
		name: map[string]any{"dtype": "F32", "shape": []int{len(vals)}, "data_offsets": []int{0, len(vals) * 4}},
	}
	hb, _ := json.Marshal(hdr)
	buf := make([]byte, 8+len(hb)+len(vals)*4)
	binary.LittleEndian.PutUint64(buf, uint64(len(hb)))
	copy(buf[8:], hb)
	for i, v := range vals {
		binary.LittleEndian.PutUint32(buf[8+len(hb)+i*4:], math.Float32bits(v))
	}
	if err := os.WriteFile(path, buf, 0o644); err != nil {
		t.Fatal(err)
	}
}

// TestLoadAnySharded pins that a sharded checkpoint (index.json + shards)
// loads into RAM through LoadAny exactly like a single file — the RAM path
// must work for the large models the hub now fetches sharded.
func TestLoadAnySharded(t *testing.T) {
	dir := t.TempDir()
	writeShard(t, filepath.Join(dir, "model-00001-of-00002.safetensors"), "a", []float32{1, 2, 3})
	writeShard(t, filepath.Join(dir, "model-00002-of-00002.safetensors"), "b", []float32{4, 5})
	index := map[string]any{"weight_map": map[string]string{
		"a": "model-00001-of-00002.safetensors",
		"b": "model-00002-of-00002.safetensors",
	}}
	ib, _ := json.Marshal(index)
	if err := os.WriteFile(filepath.Join(dir, "model.safetensors.index.json"), ib, 0o644); err != nil {
		t.Fatal(err)
	}

	// No model.safetensors present: LoadAny must fall back to the shards.
	m, err := LoadAny(filepath.Join(dir, "model.safetensors"))
	if err != nil {
		t.Fatal(err)
	}
	if len(m) != 2 {
		t.Fatalf("got %d tensors, want 2", len(m))
	}
	if got := m["a"].Data; len(got) != 3 || got[0] != 1 || got[2] != 3 {
		t.Errorf("tensor a = %v", got)
	}
	if got := m["b"].Data; len(got) != 2 || got[1] != 5 {
		t.Errorf("tensor b = %v", got)
	}

	// The streaming Reader must agree.
	r, err := OpenReader(filepath.Join(dir, "model.safetensors"))
	if err != nil {
		t.Fatal(err)
	}
	defer func() { _ = r.Close() }()
	if v, err := r.F32("b", 2); err != nil || v[0] != 4 {
		t.Errorf("reader b = %v, %v", v, err)
	}
}

// TestShardIndexTraversalRejected pins that an untrusted index cannot name a
// shard outside the model directory.
func TestShardIndexTraversalRejected(t *testing.T) {
	for _, bad := range []string{"../evil.safetensors", "/etc/passwd", "sub/dir.safetensors", "..", ""} {
		if ValidShardName(bad) {
			t.Errorf("ValidShardName(%q) = true, want false", bad)
		}
	}
	if !ValidShardName("model-00001-of-00002.safetensors") {
		t.Error("rejected a legitimate shard name")
	}

	dir := t.TempDir()
	index := map[string]any{"weight_map": map[string]string{"a": "../escape.safetensors"}}
	ib, _ := json.Marshal(index)
	_ = os.WriteFile(filepath.Join(dir, "model.safetensors.index.json"), ib, 0o644)
	if _, err := LoadAny(filepath.Join(dir, "model.safetensors")); err == nil {
		t.Fatal("expected LoadAny to reject a traversal shard name")
	}
	if _, err := OpenReader(filepath.Join(dir, "model.safetensors")); err == nil {
		t.Fatal("expected OpenReader to reject a traversal shard name")
	}
}
