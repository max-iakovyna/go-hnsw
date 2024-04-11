package distances

import (
	"go-hnsw/hnsw/vectors"
	"testing"
)

func TestEuclidian(t *testing.T) {

	v1 := vectors.Vector{0, 3}
	v2 := vectors.Vector{4, 0}

	distance := Euclidian(v1, v2)

	if distance != 5.0 {
		t.Fatalf("Euclidian distance expected to be 5.0 but %f found", distance)
	}
}

func TestCosine(t *testing.T) {
	v1 := vectors.Vector{3, 3}
	v2 := vectors.Vector{30, 30}

	distance := Cosine(v1, v2)

	if distance != 0 {
		t.Fatalf("Euclidian distance expected to be 5.0 but %f found", distance)
	}
}
