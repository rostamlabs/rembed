package tensor

import "testing"

func TestDot4Probe(t *testing.T) {
	if !hasSIMD {
		t.Skip("no simd")
	}
	for _, k := range []int{1, 4, 8, 9, 16, 33} {
		a := make([]float32, k)
		b0 := make([]float32, k)
		b1 := make([]float32, k)
		b2 := make([]float32, k)
		b3 := make([]float32, k)
		for i := range a {
			a[i] = 1
			b0[i] = 1
			b1[i] = 2
			b2[i] = 3
			b3[i] = 4
		}
		dst := make([]float32, 4)
		dot4AVX2(&dst[0], &a[0], &b0[0], &b1[0], &b2[0], &b3[0], k)
		t.Logf("k=%d dst=%v want=[%d %d %d %d]", k, dst, k, 2*k, 3*k, 4*k)
	}
}
