// SPDX-License-Identifier: Apache-2.0

//go:build !unix

// Package mmapfile: on platforms without unix mmap (Windows), fall back to
// reading the whole file into the heap. Correct and identical in behavior;
// it just loses the paging/eviction benefit, so a model larger than RAM
// will not fit there.
package mmapfile

import "os"

// File holds file bytes (read into the heap on this platform).
type File struct {
	data []byte
}

// Open reads path into memory.
func Open(path string) (*File, error) {
	b, err := os.ReadFile(path)
	if err != nil {
		return nil, err
	}
	return &File{data: b}, nil
}

// Data returns the file bytes.
func (m *File) Data() []byte { return m.data }

// Close releases the buffer.
func (m *File) Close() error { m.data = nil; return nil }
