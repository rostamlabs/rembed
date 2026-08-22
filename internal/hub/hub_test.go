// SPDX-License-Identifier: Apache-2.0

package hub

import (
	"path/filepath"
	"testing"
)

// TestIsModelID pins the id/path discrimination that guards network
// egress: rembed.Load must never treat these as equivalent, and the
// review that found "models/foo" reaching the Hub is why the local-path
// preference in Load exists. This table is the regression net for it.
func TestIsModelID(t *testing.T) {
	yes := []string{
		"sentence-transformers/all-MiniLM-L6-v2",
		"BAAI/bge-small-en-v1.5",
		"a/b",
		"org-name/model.v2_x",
	}
	no := []string{
		"", "a", "a/b/c", "/abs/path", "a//b", "/a/b",
		"../x", "a/..", "..", "./a", "a/.b/..", "-lead/x", ".lead/x",
		"C:/x", // sub-path chars outside the class
	}
	for _, id := range yes {
		if !IsModelID(id) {
			t.Errorf("IsModelID(%q) = false, want true", id)
		}
	}
	for _, id := range no {
		if IsModelID(id) {
			t.Errorf("IsModelID(%q) = true, want false", id)
		}
	}
	// NOTE deliberately documented: path-like ids such as "models/foo" DO
	// match (they are valid org/name syntax) — rembed.Load's local-path
	// preference is what keeps them off the network, not this regexp.
	if !IsModelID("models/foo") {
		t.Error("models/foo is valid org/name syntax; Load-level logic handles the ambiguity")
	}
}

func TestCacheDir(t *testing.T) {
	t.Setenv("REMBED_CACHE", "/tmp/custom-cache")
	dir, err := CacheDir()
	if err != nil || dir != "/tmp/custom-cache" {
		t.Fatalf("dir=%q err=%v", dir, err)
	}
	t.Setenv("REMBED_CACHE", "")
	dir, err = CacheDir()
	if err != nil || filepath.Base(dir) != "rembed" {
		t.Fatalf("dir=%q err=%v", dir, err)
	}
}
