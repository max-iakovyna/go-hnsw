package hnsw

import (
	"fmt"
	"go-hnsw/hnsw/vectors"
	"go-hnsw/hnsw/vectors/distances"
	"math/rand/v2"
)

type HnswCollection struct {
	layers          []*Layer
	idCounter       uint64
	connectivity    int
	prefetchFactor  int
	vectorDimension int
}

func NewHnswCollection(nLayers int, vectorDimension int, distance distances.Distance, connectivity int, prefetchFactor int) *HnswCollection {
	if nLayers <= 0 {
		panic("nLayers must be > 0")
	}
	hnsw := new(HnswCollection)
	for i := 0; i < nLayers; i += 1 {
		hnsw.layers = append(hnsw.layers, NewLayer(distance))
	}

	hnsw.idCounter = 0
	hnsw.connectivity = connectivity
	hnsw.prefetchFactor = prefetchFactor
	hnsw.vectorDimension = vectorDimension

	return hnsw
}

func (hnsw *HnswCollection) Add(vector vectors.Vector, value []byte) uint64 {

	if len(vector) != hnsw.vectorDimension {
		panic(fmt.Sprintf("Vector dimension must be %d", hnsw.vectorDimension))
	}

	insertIdx := 0

	if !hnsw.layers[0].IsEmpty() {
		insertIdx = hnsw.findInsertIndex()
	}

	node := &Node{}
	id := hnsw.generateNewId()
	for i := insertIdx; i < len(hnsw.layers); i += 1 {
		layer := hnsw.layers[i]
		newNode := layer.Add(id, vector, value, hnsw.connectivity)
		node.NextLevel = newNode
		node = newNode
	}
	return id
}

func (hnsw *HnswCollection) NNearest(vector vectors.Vector, n int) []*Node {
	node := hnsw.layers[0].Nearest(vector)

	if node == nil {
		return []*Node{}
	}

	for i := 1; i < len(hnsw.layers)-1; i += 1 {
		node = hnsw.layers[i].NearestFrom(vector, node.NextLevel)
	}

	return hnsw.layers[len(hnsw.layers)-1].NNearest(node.NextLevel, n, hnsw.prefetchFactor)
}

func (hnsw *HnswCollection) findInsertIndex() int {
	return rand.IntN(len(hnsw.layers))
}

func (hnsw *HnswCollection) generateNewId() uint64 {
	id := hnsw.idCounter
	hnsw.idCounter += 1
	return id
}

func (hnsw *HnswCollection) Remove(id uint64) bool {
	res := false
	for _, layer := range hnsw.layers {
		res = res || layer.Remove(id)
	}
	return res
}
