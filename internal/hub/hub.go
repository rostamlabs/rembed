// SPDX-License-Identifier: Apache-2.0

// Package hub fetches model files straight from the Hugging Face Hub in
// pure Go — no Python, no conversion step. HuggingFace ships
// model.safetensors natively for the supported models, so "conversion" was
// only ever a download plus a manifest, and the manifest is now derived in
// Go (internal/model). Files land in a local cache directory and are
// reused on every later Load.
package hub

import (
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"os"
	"path/filepath"
	"regexp"
	"strings"
	"time"
)

// required are the files every sentence-transformers-format repo must
// provide, regardless of architecture. modules.json is required
// deliberately: silently assuming "no Normalize module" for a repo that
// actually has one would shift every downstream cosine threshold, and
// every genuine ST repo ships the file. config.json comes first: the
// tokenizer files to fetch depend on its model_type.
var required = []string{
	"config.json",
	"tokenizer_config.json",
	"1_Pooling/config.json",
	"modules.json",
	"model.safetensors",
}

// tokenizerFiles returns the tokenizer artifacts for a model_type:
// byte-level BPE models (RoBERTa family) ship vocab.json + merges.txt;
// WordPiece models (BERT, MPNet) ship vocab.txt.
func tokenizerFiles(modelType string) []string {
	if modelType == "roberta" || modelType == "xlm-roberta" {
		return []string{"vocab.json", "merges.txt"}
	}
	return []string{"vocab.txt"}
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
	sweepStaleTemps(dir)
	cleanup := func() {
		// A failed first fetch should not litter the cache with empty
		// org--name directories (os.Remove refuses non-empty ones).
		_ = os.Remove(filepath.Join(dir, "1_Pooling"))
		_ = os.Remove(dir)
	}
	for _, f := range required {
		if err := fetch(modelID, f, dir); err != nil {
			cleanup()
			return "", err
		}
	}
	var hf struct {
		ModelType string `json:"model_type"`
	}
	raw, err := os.ReadFile(filepath.Join(dir, "config.json"))
	if err == nil {
		err = json.Unmarshal(raw, &hf)
	}
	if err != nil {
		cleanup()
		return "", fmt.Errorf("hub: %s: config.json: %w", modelID, err)
	}
	for _, f := range tokenizerFiles(hf.ModelType) {
		if err := fetch(modelID, f, dir); err != nil {
			cleanup()
			return "", err
		}
	}
	return dir, nil
}

// sweepStaleTemps removes download temp files older than an hour — a
// SIGKILL mid-download orphans them, and nothing else reclaims the space.
func sweepStaleTemps(dir string) {
	matches, _ := filepath.Glob(filepath.Join(dir, ".rembed-dl-*"))
	for _, m := range matches {
		if fi, err := os.Stat(m); err == nil && time.Since(fi.ModTime()) > time.Hour {
			_ = os.Remove(m)
		}
	}
}

// fetch downloads one file if it is not cached, atomically (tmp + rename),
// so a concurrent or interrupted download never leaves a truncated file
// under the final name. Large (LFS) files carry a sha256 in the redirect
// hop's X-Linked-Etag header; when present the body is verified against
// it, so a proxy-served error page or truncated transfer can never poison
// the cache.
func fetch(modelID, name, dir string) error {
	dst := filepath.Join(dir, filepath.FromSlash(name))
	if _, err := os.Stat(dst); err == nil {
		return nil
	}
	if err := os.MkdirAll(filepath.Dir(dst), 0o755); err != nil {
		return err
	}
	// The integrity headers appear on the FIRST response (the redirect to
	// the CDN), so capture them in CheckRedirect. One client per fetch
	// keeps that capture race-free.
	var wantSHA string
	var wantSize int64 = -1
	client := &http.Client{
		Timeout: 15 * time.Minute,
		CheckRedirect: func(req *http.Request, via []*http.Request) error {
			if r := via[len(via)-1].Response; r != nil {
				if etag := strings.Trim(r.Header.Get("X-Linked-Etag"), `W/"`); len(etag) == 64 {
					wantSHA = etag
				}
				if v := r.Header.Get("X-Linked-Size"); v != "" {
					_, _ = fmt.Sscan(v, &wantSize)
				}
			}
			return nil
		},
	}
	url := fmt.Sprintf("https://huggingface.co/%s/resolve/main/%s", modelID, name)
	req, err := http.NewRequest(http.MethodGet, url, nil)
	if err != nil {
		return err
	}
	req.Header.Set("User-Agent", "rembed/0.1")
	if tok := os.Getenv("HF_TOKEN"); tok != "" {
		req.Header.Set("Authorization", "Bearer "+tok)
	}
	resp, err := client.Do(req)
	if err != nil {
		return fmt.Errorf("hub: fetching %s/%s: %w", modelID, name, err)
	}
	defer func() { _ = resp.Body.Close() }()
	switch resp.StatusCode {
	case http.StatusOK:
	case http.StatusNotFound:
		return fmt.Errorf("hub: %s has no %s — not a sentence-transformers-format BERT repo", modelID, name)
	case http.StatusUnauthorized, http.StatusForbidden:
		// HF answers 401 for repos that DO NOT EXIST as well as for gated
		// and private ones — never assume this is an auth problem.
		hint := "no HF_TOKEN is set"
		if os.Getenv("HF_TOKEN") != "" {
			hint = "HF_TOKEN is set but may be invalid or lack access"
		}
		return fmt.Errorf("hub: %s: HTTP %s — the repo does not exist, is gated, or is private (%s)", modelID, resp.Status, hint)
	default:
		return fmt.Errorf("hub: fetching %s/%s: HTTP %s", modelID, name, resp.Status)
	}
	tmp, err := os.CreateTemp(filepath.Dir(dst), ".rembed-dl-*")
	if err != nil {
		return err
	}
	defer func() { _ = os.Remove(tmp.Name()) }()
	hasher := sha256.New()
	n, err := io.Copy(io.MultiWriter(tmp, hasher), resp.Body)
	if err == nil {
		err = tmp.Close()
	} else {
		_ = tmp.Close()
	}
	if err != nil {
		return fmt.Errorf("hub: downloading %s/%s: %w", modelID, name, err)
	}
	if wantSize > 0 && n != wantSize {
		return fmt.Errorf("hub: %s/%s truncated: %d of %d bytes", modelID, name, n, wantSize)
	}
	if resp.ContentLength > 0 && n != resp.ContentLength {
		return fmt.Errorf("hub: %s/%s truncated: %d of %d bytes", modelID, name, n, resp.ContentLength)
	}
	if wantSHA != "" {
		if got := hex.EncodeToString(hasher.Sum(nil)); got != wantSHA {
			return fmt.Errorf("hub: %s/%s content hash mismatch (got %s want %s) — corrupt transfer or interfering proxy", modelID, name, got, wantSHA)
		}
	}
	return os.Rename(tmp.Name(), dst)
}
