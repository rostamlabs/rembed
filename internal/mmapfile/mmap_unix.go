// SPDX-License-Identifier: Apache-2.0

//go:build unix

// Package mmapfile memory-maps a file read-only so its bytes can be handed
// out as slices without copying into the heap. The OS pages the bytes in on
// access and evicts them under memory pressure (they are clean, file-backed),
// so a mapped weights file has a resident cost of only its working set —
// which is what lets rembed run a model larger than RAM.
package mmapfile

import (
	"os"

	"golang.org/x/sys/unix"
)

// File is a read-only memory mapping. Data is valid until Close.
type File struct {
	data []byte
	f    *os.File
}

// Open mmaps path read-only (MAP_SHARED). An empty file maps to nil Data.
func Open(path string) (*File, error) {
	f, err := os.Open(path)
	if err != nil {
		return nil, err
	}
	fi, err := f.Stat()
	if err != nil {
		_ = f.Close()
		return nil, err
	}
	size := fi.Size()
	if size == 0 {
		return &File{f: f}, nil
	}
	data, err := unix.Mmap(int(f.Fd()), 0, int(size), unix.PROT_READ, unix.MAP_SHARED)
	if err != nil {
		_ = f.Close()
		return nil, err
	}
	// Weights are streamed in order each forward pass; hint the kernel so it
	// reads ahead and drops behind rather than caching aggressively.
	_ = unix.Madvise(data, unix.MADV_SEQUENTIAL)
	return &File{data: data, f: f}, nil
}

// Data returns the mapped bytes (nil for an empty file).
func (m *File) Data() []byte { return m.data }

// Close unmaps and closes the file. After Close, slices derived from Data
// must not be used.
func (m *File) Close() error {
	if m.data != nil {
		_ = unix.Munmap(m.data)
		m.data = nil
	}
	if m.f != nil {
		return m.f.Close()
	}
	return nil
}
