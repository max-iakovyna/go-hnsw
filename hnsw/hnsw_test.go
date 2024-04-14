package hnsw

import (
	"go-hnsw/hnsw/vectors"
	"go-hnsw/hnsw/vectors/distances"
	"math/rand/v2"
	"testing"
)

func toVector(arr []float64) vectors.Vector {
	vector := make([]vectors.VFloat, len(arr))
	for i, v := range arr {
		vector[i] = vectors.VFloat(v)
	}

	return vectors.Vector(vector)
}

func nnToMap(nodes []*Node) map[string]uint64 {
	resMap := map[string]uint64{}

	for _, node := range nodes {
		resMap[string(node.Value)] = node.Id
	}

	return resMap
}

func validateANN(t *testing.T, nodes []*Node, expectedCount int, nodeType string) {
	if len(nodes) != expectedCount {
		t.Fatalf("The number of found '%s' nodes is expeccted to be %d but found %d", nodeType, expectedCount, len(nodes))
	}
	for _, r := range nodes {
		if string(r.Value) != nodeType {
			t.Fatalf("The node value is expected to be '%s' but %s found", nodeType, r.Value)
		}
	}
}

func TestHnsw(t *testing.T) {

	baseVectorA := []float64{10.0, 20.0, 30.0, 10.0, 10.0, 20.0, 30.0, 10.0, 10.0, 20.0, 30.0, 10.0, 10.0, 20.0, 30.0, 10.0}
	baseVectorB := make([]float64, 16)

	for i, v := range baseVectorA {
		baseVectorB[i] = -v
	}

	hnswCollection := NewHnswCollection(3, 16, distances.Euclidian, 5, 3)

	vectorToAdd := make([]float64, 16)

	for i := 0; i < 100; i += 1 {
		for i, a := range baseVectorA {
			vectorToAdd[i] = a + (rand.Float64()-0.5)*10.0
		}

		hnswCollection.Add(toVector(vectorToAdd), []byte("A"))

		for i, b := range baseVectorB {
			vectorToAdd[i] = b + (rand.Float64()-0.5)*10.0
		}

		hnswCollection.Add(toVector(vectorToAdd), []byte("B"))
	}

	resA := hnswCollection.NNearest(toVector(baseVectorA), 20)

	validateANN(t, resA, 20, "A")

	resB := hnswCollection.NNearest(toVector(baseVectorB), 20)

	validateANN(t, resB, 20, "B")

}

func HnswRemoveNodeTest(t *testing.T) {
	hnswCollection := NewHnswCollection(3, 16, distances.Euclidian, 5, 3)

	v1 := vectors.Vector{1.0, 0, 1.0}
	v2 := vectors.Vector{1.0, 0, 2.0}
	v3 := vectors.Vector{-1.0, 1.0, 0}

	hnswCollection.Add(v1, []byte("v1"))
	hnswCollection.Add(v2, []byte("v2"))
	lastId := hnswCollection.Add(v3, []byte("v3"))

	originalResult := nnToMap(hnswCollection.NNearest(vectors.Vector{0, 0, 0}, 3))

	if len(originalResult) != 3 {
		t.Fatalf("Number of found nodes expected to be 3 but found %d", len(originalResult))
	}

	removeResult := hnswCollection.Remove(lastId)

	if !removeResult {
		t.Fatal("Remove method must return 'true'")
	}

	modifiedResult := nnToMap(hnswCollection.NNearest(vectors.Vector{0, 0, 0}, 3))

	if len(modifiedResult) != 2 {
		t.Fatalf("Number of found nodes expected to be 3 but found %d", len(modifiedResult))
	}

	_, ok := modifiedResult["v3"]

	if ok {
		t.Fatalf("The node with 'v3' value must not be present in the response")
	}
}

func TestPanicOnwrongVectorDim(t *testing.T) {
	defer func() {
		err := recover()
		if err == nil {
			t.Fatal("The 'Add' must panic on wrong vector size")
		}
		if err != "Vector dimension must be 5" {
			t.Fatal("Wrong error")
		}
	}()

	hnsw := NewHnswCollection(3, 5, distances.Euclidian, 5, 5)

	hnsw.Add(vectors.Vector{1, 2, 3}, []byte("data"))

	hnsw.Add(vectors.Vector{1, 2, 3, 4, 5, 6}, []byte("data"))

}
