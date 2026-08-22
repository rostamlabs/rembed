// SPDX-License-Identifier: Apache-2.0

package tensor

import (
	"math"
	"testing"
)

// TestExpf32AgainstFloat64 pins the fast exp to ~2e-7 relative error over
// the range Softmax and GELU actually use (softmax inputs are ≤ 0 after
// max-subtraction; GELU feeds -x²).
func TestExpf32AgainstFloat64(t *testing.T) {
	for x := float32(-87); x <= 88; x += 0.001 {
		got := float64(expf32(x))
		want := math.Exp(float64(x))
		if rel := math.Abs(got-want) / want; rel > 5e-7 {
			t.Fatalf("expf32(%v)=%g want %g (rel %g)", x, got, want, rel)
		}
	}
	// Clamp regions.
	if v := expf32(-100); v != 0 {
		t.Fatalf("expf32(-100)=%v want 0", v)
	}
	if v := expf32(200); !math.IsInf(float64(v), 1) {
		t.Fatalf("expf32(200)=%v want +Inf", v)
	}
	if v := expf32(0); v != 1 {
		t.Fatalf("expf32(0)=%v want 1", v)
	}
}

// TestErff32AgainstFloat64 pins the fast erf to ~5e-7 absolute error.
func TestErff32AgainstFloat64(t *testing.T) {
	for x := float32(-10); x <= 10; x += 0.001 {
		got := float64(erff32(x))
		want := math.Erf(float64(x))
		if d := math.Abs(got - want); d > 5e-7 {
			t.Fatalf("erff32(%v)=%g want %g (diff %g)", x, got, want, d)
		}
	}
	if v := erff32(0); v != 0 {
		t.Fatalf("erff32(0)=%v want 0", v)
	}
}

func BenchmarkStdlibExp(b *testing.B) {
	var s float64
	x := 0.0
	for b.Loop() {
		s += math.Exp(x)
		x -= 1e-6
	}
	_ = s
}

func BenchmarkExpf32(b *testing.B) {
	var s float32
	x := float32(0)
	for b.Loop() {
		s += expf32(x)
		x -= 1e-6
	}
	_ = s
}

func BenchmarkStdlibErf(b *testing.B) {
	var s float64
	x := 0.0
	for b.Loop() {
		s += math.Erf(x)
		x += 1e-6
	}
	_ = s
}

func BenchmarkErff32(b *testing.B) {
	var s float32
	x := float32(0)
	for b.Loop() {
		s += erff32(x)
		x += 1e-6
	}
	_ = s
}
