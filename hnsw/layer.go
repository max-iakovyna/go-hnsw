package hnsw

import (
	"container/heap"
	"encoding/binary"
	"go-hnsw/hnsw/vectors"
	"go-hnsw/hnsw/vectors/distances"
	"io"
	"math/rand/v2"
)

type Layer struct {
	nodes       []*Node
	DistanceFnc distances.Distance
	rindex      map[uint64]int
}

func NewLayer(distanceFnc distances.Distance) *Layer {
	layer := new(Layer)
	layer.DistanceFnc = distanceFnc
	layer.rindex = map[uint64]int{}
	return layer
}

func (layer *Layer) IsEmpty() bool {
	return len(layer.nodes) == 0
}

func (layer *Layer) Add(id uint64, vector vectors.Vector, value []byte, connectivity int) *Node {

	nearestNode := layer.Nearest(vector)

	newNode := Node{Id: id, Vector: vector, Value: value, Layer: layer, neighbors: map[uint64]*Node{}}
	if nearestNode != nil {
		neighbors := layer.NNearest(nearestNode, connectivity, 3)
		for _, nbh := range neighbors {
			newNode.neighbors[nbh.Id] = nbh
			nbh.neighbors[id] = &newNode
		}
	}
	layer.nodes = append(layer.nodes, &newNode)

	layer.rindex[id] = len(layer.nodes) - 1

	return &newNode
}

func (layer *Layer) NNearest(node *Node, n int, overfetchFactor int) []*Node {
	nearestNodes := make([]*Node, n*overfetchFactor)
	start, end := 0, 1

	nearestNodes[0] = node
	added := true
	visited := map[uint64]bool{}
	visited[node.Id] = true
	for added {
		lEnd := end
		added = false
		for start < lEnd && end < len(nearestNodes) {
			currentNode := nearestNodes[start]
			for _, nd := range currentNode.neighbors {
				_, seen := visited[nd.Id]
				if seen {
					continue
				}
				visited[nd.Id] = true
				nearestNodes[end] = nd
				added = true
				end += 1
				if end == len(nearestNodes) {
					break
				}
			}
			start += 1
		}
	}

	knearest := NewKClosestNodes(n, node.Vector, layer.DistanceFnc)

	for _, n := range nearestNodes[:end] {
		heap.Push(knearest, n)
	}

	return knearest.nodes
}

func (layer *Layer) Nearest(vector vectors.Vector) *Node {
	if len(layer.nodes) == 0 {
		return nil
	}

	nth := rand.IntN(len(layer.nodes))
	node := layer.nodes[nth]

	return layer.NearestFrom(vector, node)
}

func (layer *Layer) NearestFrom(vector vectors.Vector, startNode *Node) *Node {
	if len(layer.nodes) == 0 {
		return nil
	}

	distance := layer.DistanceFnc(startNode.Vector, vector)

	visited := map[uint64]bool{}
	visited[startNode.Id] = true

	node := startNode

	for {
		updated := false
		for _, nbh := range node.neighbors {
			_, seen := visited[nbh.Id]
			if seen {
				continue
			}
			visited[nbh.Id] = true
			dst := layer.DistanceFnc(vector, nbh.Vector)
			if dst <= distance {
				distance = dst
				node = nbh
				updated = true
			}
		}

		if !updated {
			break
		}
	}

	return node
}

func (layer *Layer) Remove(id uint64) bool {
	index, ok := layer.rindex[id]

	if !ok {
		return false
	}

	delNode := layer.swapAndPop(index)
	delete(layer.rindex, id)

	for _, neighbor := range delNode.neighbors {
		delete(neighbor.neighbors, id)
	}
	return true
}

func (layer *Layer) swapAndPop(index int) *Node {
	delNode := layer.nodes[index]
	nNodes := len(layer.nodes)
	lastNodeIndex := nNodes - 1
	lastNode := layer.nodes[lastNodeIndex]

	layer.nodes[index], layer.nodes[lastNodeIndex] = layer.nodes[lastNodeIndex], layer.nodes[index]
	layer.rindex[lastNode.Id] = index

	layer.nodes[nNodes-1] = nil

	layer.nodes = layer.nodes[:nNodes-1]
	return delNode
}

func (layer Layer) Get(id uint64) (*Node, bool) {
	index, ok := layer.rindex[id]

	if !ok {
		return nil, false
	}

	return layer.nodes[index], true
}

func (layer Layer) Serrialize(writer io.Writer) (int, error) {

	size := 4
	err := binary.Write(writer, binary.LittleEndian, int32(len(layer.nodes)))
	if err != nil {
		return size, err
	}

	for _, node := range layer.nodes {
		nodeSize, err := node.SerializeCompact(writer)
		size += nodeSize
		if err != nil {
			return size, err
		}
	}

	return size, nil
}

func DesserializeLayer(reader io.Reader, distanceFnc distances.Distance, nextLayer *Layer) (*Layer, error) {
	var nNodes int32
	err := binary.Read(reader, binary.LittleEndian, &nNodes)
	if err != nil {
		return nil, err
	}

	layer := Layer{
		nodes:       make([]*Node, nNodes),
		rindex:      map[uint64]int{},
		DistanceFnc: distanceFnc,
	}

	for i := 0; i < int(nNodes); i += 1 {
		node, err := DesserializeNode(reader)
		if err != nil {
			return nil, err
		}
		if nextLayer != nil {
			node.NextLevel, _ = nextLayer.Get(node.Id)
		}
		layer.nodes[i] = node
		layer.rindex[node.Id] = i

		for id, _ := range node.neighbors {

			idx, ok := layer.rindex[id]
			if ok {
				nNode := layer.nodes[idx]
				node.neighbors[id] = nNode
				nNode.neighbors[node.Id] = node
			}
		}
	}

	return &layer, nil
}
