package hnsw

import (
	"fmt"
	"go-hnsw/hnsw/vectors"
	"go-hnsw/hnsw/vectors/distances"
	"testing"
)

func TestLayerDeleteNode(t *testing.T) {

	layer := NewLayer(distances.Euclidian)

	layer.Add(1, vectors.Vector{0, 1}, []byte("1"), 3)
	layer.Add(2, vectors.Vector{1, 0}, []byte("2"), 3)
	layer.Add(3, vectors.Vector{1, 1}, []byte("3"), 3)
	layer.Add(4, vectors.Vector{-1, -1}, []byte("4"), 3)

	for i := 1; i <= 4; i += 1 {
		n, ok := layer.Get(uint64(i))

		if !ok {
			t.Fatalf("Node with id %d is not found", i)
		}

		if string(n.Value) != fmt.Sprintf("%d", i) {
			t.Fatalf("Node alue is incorret expected %s found %s", fmt.Sprintf("%d", i), n.Value)
		}
	}

	if !layer.Remove(1) {
		t.Fatal("layer.Delete(1) must return true but return false")
	}

	if !layer.Remove(3) {
		t.Fatal("layer.Delete(3) must return true but return false")
	}

	_, ok := layer.Get(1)
	if ok {
		t.Fatal("layer.Get(1) must return empty response")
	}

	_, ok = layer.Get(3)
	if ok {
		t.Fatal("layer.Get(3) must return empty response")
	}

	for _, i := range []int{2, 4} {
		n, ok := layer.Get(uint64(i))

		if !ok {
			t.Fatalf("Node with id %d is not found", i)
		}

		if string(n.Value) != fmt.Sprintf("%d", i) {
			t.Fatalf("Node alue is incorret expected %s found %s", fmt.Sprintf("%d", i), n.Value)
		}
	}

}
