// SPDX-License-Identifier: Apache-2.0

// Package main is the C ABI for rembed's Python (and any other FFI)
// bindings, built as a shared library:
//
//	CGO_ENABLED=1 go build -buildmode=c-shared -o python/rembed/librembed.so ./python/capi
//
// The rembed Go library itself stays pure Go / cgo-free; this package is a
// separate build artifact for foreign callers only. The ABI is deliberately
// flat and allocation-disciplined: the caller owns the output buffer (a
// numpy array on the Python side), handles are opaque integers (cgo forbids
// passing Go pointers across the boundary), and every string this library
// returns must be released with RembedFreeString.
package main

/*
#include <stdlib.h>
*/
import "C"

import (
	"context"
	"sync"
	"unsafe"

	"github.com/rostamlabs/rembed"
)

var (
	mu      sync.Mutex
	handles            = map[C.longlong]*rembed.Embedder{}
	nextH   C.longlong = 1
)

func setErr(errOut **C.char, msg string) {
	if errOut != nil {
		*errOut = C.CString(msg)
	}
}

// RembedLoad opens a model directory and returns an opaque handle (> 0).
// useInt8 != 0 selects weight-only int8; workers caps per-embed CPU
// workers (0 = all cores). On failure returns 0 and, if errOut is
// non-NULL, stores a message the caller must free with RembedFreeString.
//
//export RembedLoad
func RembedLoad(modelDir *C.char, useInt8, workers C.int, errOut **C.char) C.longlong {
	var opts []rembed.Option
	if useInt8 != 0 {
		opts = append(opts, rembed.WithInt8())
	}
	if workers > 0 {
		opts = append(opts, rembed.WithWorkers(int(workers)))
	}
	emb, err := rembed.Load(C.GoString(modelDir), opts...)
	if err != nil {
		setErr(errOut, err.Error())
		return 0
	}
	mu.Lock()
	defer mu.Unlock()
	h := nextH
	nextH++
	handles[h] = emb
	return h
}

func get(h C.longlong) *rembed.Embedder {
	mu.Lock()
	defer mu.Unlock()
	return handles[h]
}

// RembedDim returns the embedding dimensionality, or 0 for a bad handle.
//
//export RembedDim
func RembedDim(h C.longlong) C.int {
	if emb := get(h); emb != nil {
		return C.int(emb.Dim())
	}
	return 0
}

// RembedModel writes the model name into buf (at most bufLen bytes,
// NUL-terminated) and returns the full name's length, or -1 for a bad
// handle. A return >= bufLen means the name was truncated.
//
//export RembedModel
func RembedModel(h C.longlong, buf *C.char, bufLen C.int) C.int {
	emb := get(h)
	if emb == nil {
		return -1
	}
	name := emb.Model()
	if buf != nil && bufLen > 0 {
		n := min(len(name), int(bufLen)-1)
		dst := unsafe.Slice((*byte)(unsafe.Pointer(buf)), int(bufLen))
		copy(dst, name[:n])
		dst[n] = 0
	}
	return C.int(len(name))
}

// RembedEmbedBatch embeds n texts and writes the vectors row-major into
// out, which the CALLER allocates with n*RembedDim(h) float32s (a numpy
// array on the Python side — no ownership crosses the boundary). Returns 0
// on success; on failure returns -1 and fills errOut as in RembedLoad.
//
//export RembedEmbedBatch
func RembedEmbedBatch(h C.longlong, texts **C.char, n C.int, out *C.float, errOut **C.char) C.int {
	emb := get(h)
	if emb == nil {
		setErr(errOut, "rembed: invalid handle")
		return -1
	}
	if n < 0 {
		setErr(errOut, "rembed: negative text count")
		return -1
	}
	if n == 0 {
		return 0
	}
	cTexts := unsafe.Slice(texts, int(n))
	goTexts := make([]string, int(n))
	for i, ct := range cTexts {
		goTexts[i] = C.GoString(ct)
	}
	vecs, err := emb.Embed(context.Background(), goTexts)
	if err != nil {
		setErr(errOut, err.Error())
		return -1
	}
	dim := emb.Dim()
	dst := unsafe.Slice((*float32)(unsafe.Pointer(out)), int(n)*dim)
	for i, v := range vecs {
		copy(dst[i*dim:(i+1)*dim], v)
	}
	return 0
}

// RembedClose releases a handle. Safe to call with an unknown handle.
//
//export RembedClose
func RembedClose(h C.longlong) {
	mu.Lock()
	defer mu.Unlock()
	delete(handles, h)
}

// RembedFreeString releases a string returned through an errOut parameter.
//
//export RembedFreeString
func RembedFreeString(s *C.char) {
	C.free(unsafe.Pointer(s))
}

func main() {}
