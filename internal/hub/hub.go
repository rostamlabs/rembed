// SPDX-License-Identifier: Apache-2.0

// Package hub fetches model files straight from the Hugging Face Hub in
// pure Go — no Python, no conversion step. HuggingFace ships
// model.safetensors natively for the supported models, so "conversion" was
// only ever a download plus a manifest, and the manifest is now derived in
// Go (internal/model). Files land in a local cache directory and are
// reused on every later Load.
package hub

import (
	"fmt"
	"io"
	"net/http"
	"os"
	"path/filepath"
	"regexp"
	"strings"
	"time"
)

// required are the files a sentence-transformers-format repo must provide.
var required = []string{
	"model.safetensors",
	"vocab.txt",
	"config.json",
	"tokenizer_config.json",
	"1_Pooling/config.json",
}

// optional files improve fidelity but their absence is meaningful, not an
// error (a repo without modules.json simply has no Normalize module).
var optional = []string{
	"modules.json",
}

var idRe = regexp.MustCompile(`^[A-Za-z0-9][A-Za-z0-9._-]*/[A-Za-z0-9._-]+$`)

// IsModelID reports whether ref looks like a Hugging Face model id
// (org/name) rather than a filesystem path.
func IsModelID(ref string) bool {
	return idRe.MatchString(ref) && !strings.Contains(ref, "..")
}

// CacheDir returns the model cache root: $REMBED_CACHE if set, else
// <user cache dir>/rembed.
func CacheDir() (string, error) {
	if env := os.Getenv("REMBED_CACHE"); env != "" {
		return env, nil
	}
	base, err := os.UserCacheDir()
	if err != nil {
		return "", fmt.Errorf("hub: no cache dir (set REMBED_CACHE): %w", err)
	}
	return filepath.Join(base, "rembed"), nil
}

// Ensure downloads modelID's files into the cache (skipping files already
// present) and returns the model directory. Set HF_TOKEN for gated or
// rate-limited access.
func Ensure(modelID, cacheDir string) (string, error) {
	if !IsModelID(modelID) {
		return "", fmt.Errorf("hub: %q is not a model id (want org/name)", modelID)
	}
	dir := filepath.Join(cacheDir, strings.ReplaceAll(modelID, "/", "--"))
	if err := os.MkdirAll(dir, 0o755); err != nil {
		return "", err
	}
	client := &http.Client{Timeout: 15 * time.Minute}
	for _, f := range required {
		if err := fetch(client, modelID, f, dir, true); err != nil {
			return "", err
		}
	}
	for _, f := range optional {
		if err := fetch(client, modelID, f, dir, false); err != nil {
			return "", err
		}
	}
	return dir, nil
}

// fetch downloads one file if it is not cached, atomically (tmp + rename),
// so a concurrent or interrupted download never leaves a truncated file
// under the final name.
func fetch(client *http.Client, modelID, name, dir string, must bool) error {
	dst := filepath.Join(dir, filepath.FromSlash(name))
	if _, err := os.Stat(dst); err == nil {
		return nil
	}
	if err := os.MkdirAll(filepath.Dir(dst), 0o755); err != nil {
		return err
	}
	url := fmt.Sprintf("https://huggingface.co/%s/resolve/main/%s", modelID, name)
	req, err := http.NewRequest(http.MethodGet, url, nil)
	if err != nil {
		return err
	}
	if tok := os.Getenv("HF_TOKEN"); tok != "" {
		req.Header.Set("Authorization", "Bearer "+tok)
	}
	resp, err := client.Do(req)
	if err != nil {
		return fmt.Errorf("hub: fetching %s/%s: %w", modelID, name, err)
	}
	defer func() { _ = resp.Body.Close() }()
	if resp.StatusCode == http.StatusNotFound {
		if must {
			return fmt.Errorf("hub: %s has no %s — not a sentence-transformers-format BERT repo", modelID, name)
		}
		return nil
	}
	if resp.StatusCode != http.StatusOK {
		return fmt.Errorf("hub: fetching %s/%s: HTTP %s", modelID, name, resp.Status)
	}
	tmp, err := os.CreateTemp(filepath.Dir(dst), ".rembed-dl-*")
	if err != nil {
		return err
	}
	defer func() { _ = os.Remove(tmp.Name()) }()
	n, err := io.Copy(tmp, resp.Body)
	if err == nil {
		err = tmp.Close()
	} else {
		_ = tmp.Close()
	}
	if err != nil {
		return fmt.Errorf("hub: downloading %s/%s: %w", modelID, name, err)
	}
	if resp.ContentLength > 0 && n != resp.ContentLength {
		return fmt.Errorf("hub: %s/%s truncated: %d of %d bytes", modelID, name, n, resp.ContentLength)
	}
	return os.Rename(tmp.Name(), dst)
}
