// SPDX-License-Identifier: Apache-2.0

//go:build !amd64

package tensor

// GELU applies the erf-based GELU in place. No vectorized kernel off amd64
// (arm64's NEON GELU is future work), so this is the scalar path.
func GELU(x []float32) { geluScalar(x) }
