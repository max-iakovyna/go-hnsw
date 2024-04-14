package hnsw

import (
	"bytes"
	"container/heap"
	"fmt"
	"go-hnsw/hnsw/vectors"
	"go-hnsw/hnsw/vectors/distances"
	"reflect"
	"testing"
)

func TestKClosesNodesSequential(t *testing.T) {

	kclosest := NewKClosestNodes(
		5,
		vectors.Vector{0, 0},
		distances.Euclidian,
	)

	for i := 10; i > 1; i -= 1 {

		v := vectors.Vector{0.0, vectors.VFloat(i)}

		heap.Push(kclosest, &Node{Vector: v, Value: []byte(fmt.Sprintf("%d", i))})
	}

	if kclosest.Len() != 5 {
		t.Fatalf("Expected len is 5 but actual was %d", kclosest.Len())
	}

	for i := 6; i >= 2; i -= 1 {
		node := heap.Pop(kclosest)
		value := string(node.(*Node).Value)
		expected := fmt.Sprintf("%d", i)
		if value != expected {
			t.Fatalf("Next node mus have value %s, but %s found", expected, value)
		}
	}

}

func TestKClosesNodesRandom(t *testing.T) {
	kclosest := NewKClosestNodes(
		5,
		vectors.Vector{0, 0},
		distances.Euclidian,
	)

	vectors := []vectors.Vector{
		{0, 1},
		{0, 4},
		{0, 9},
		{0, 6},
		{0, 2},
		{0, 8},
		{0, 3},
		{0, 7},
		{0, 0},
	}

	for _, v := range vectors {
		heap.Push(kclosest, &Node{Vector: v, Value: []byte(fmt.Sprintf("%d", int(v[1])))})
	}

	if kclosest.Len() != 5 {
		t.Fatalf("Expected len is 5 but actual was %d", kclosest.Len())
	}

	for i := 4; i >= 0; i -= 1 {
		node := heap.Pop(kclosest)
		value := string(node.(*Node).Value)
		expected := fmt.Sprintf("%d", i)
		if value != expected {
			t.Fatalf("The next node must have a value %s, but %s found", expected, value)
		}
	}

}

func TestSerDe(t *testing.T) {

	nNode := Node{Id: 2}

	node := Node{
		Id:        1,
		neighbors: map[uint64]*Node{2: &nNode, 3: &nNode},
		Vector:    vectors.Vector{1, 2, 3, 4},
		Value:     []byte("payload"),
	}

	buff := new(bytes.Buffer)

	_, e := node.SerializeCompact(buff)

	if e != nil {
		t.Fatalf("'Node.SerializeCompact' returned error: %s", e)
	}

	desserNode, e := DesserializeNode(buff)

	if e != nil {
		t.Fatalf("'DesserializeCompact' returned error: %s", e)
	}

	if desserNode.Id != 1 {
		t.Fatalf("The node Id must be 1 but %d found", desserNode.Id)
	}

	expectedVector := vectors.Vector{1, 2, 3, 4}
	if !reflect.DeepEqual(desserNode.Vector, expectedVector) {
		t.Fatalf("Expected desserialized vector: %s, bu found: %s", expectedVector.String(), desserNode.Vector.String())
	}

	if string(desserNode.Value) != "payload" {
		t.Fatal("The Value of the desserialized node is not equal to the original")
	}

	_, ok := desserNode.neighbors[2]
	if !ok {
		t.Fatal("Node with id = 2 not in neighbors")
	}

	_, ok = desserNode.neighbors[3]
	if !ok {
		t.Fatal("Node with id = 3 not in neighbors")
	}
}
