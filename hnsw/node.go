package hnsw

import (
	"container/heap"
	"encoding/binary"
	"go-hnsw/hnsw/vectors"
	"go-hnsw/hnsw/vectors/distances"
	"io"
	"unsafe"
)

type Node struct {
	Id        uint64
	neighbors map[uint64]*Node
	Vector    vectors.Vector
	Value     []byte
	NextLevel *Node
	Layer     *Layer
}

func (node *Node) SerializeCompact(writer io.Writer) (int, error) {
	var size int = 8

	err := binary.Write(writer, binary.LittleEndian, node.Id)
	if err != nil {
		return size, err
	}

	err = binary.Write(writer, binary.LittleEndian, int32(len(node.neighbors)))
	if err != nil {
		return size, err
	}
	size += 4

	for k := range node.neighbors {
		err = binary.Write(writer, binary.LittleEndian, k)
		if err != nil {
			return size, err
		}
		size += 8
	}

	err = binary.Write(writer, binary.LittleEndian, int32(len(node.Vector)))
	if err != nil {
		return size, err
	}
	size += 4

	for _, v := range node.Vector {
		err = binary.Write(writer, binary.LittleEndian, v)
		if err != nil {
			return size, err
		}
		size += int(unsafe.Sizeof(v))
	}

	var dataLen int32
	if len(node.Value) > 0 {
		dataLen = int32(len(node.Value))
	}
	err = binary.Write(writer, binary.LittleEndian, dataLen)
	if err != nil {
		return size, err
	}
	size += 4

	if dataLen != 0 {
		s, err := writer.Write(node.Value)
		if err != nil {
			return size, err
		}
		size += s
	}

	return size, nil
}

func DesserializeNode(reader io.Reader) (*Node, error) {
	var id uint64
	err := binary.Read(reader, binary.LittleEndian, &id)
	if err != nil {
		return nil, err
	}

	node := Node{Id: id, neighbors: map[uint64]*Node{}}

	var nNeightbors int32
	err = binary.Read(reader, binary.LittleEndian, &nNeightbors)
	if err != nil {
		return nil, err
	}

	var nId uint64
	for i := 0; i < int(nNeightbors); i += 1 {
		err = binary.Read(reader, binary.LittleEndian, &nId)
		if err != nil {
			return nil, err
		}
		node.neighbors[nId] = nil
	}

	var vectorSize int32
	err = binary.Read(reader, binary.LittleEndian, &vectorSize)
	if err != nil {
		return nil, err
	}

	vector := make([]vectors.VFloat, vectorSize)
	var v vectors.VFloat
	for i := 0; i < int(vectorSize); i += 1 {
		err = binary.Read(reader, binary.LittleEndian, &v)
		if err != nil {
			return nil, err
		}
		vector[i] = v
	}

	node.Vector = vectors.Vector(vector)

	var dataLen int32
	err = binary.Read(reader, binary.LittleEndian, &dataLen)
	if err != nil {
		return nil, err
	}

	if dataLen > 0 {
		node.Value = make([]byte, dataLen)
		_, err := reader.Read(node.Value)
		if err != nil {
			return nil, err
		}
	}

	return &node, nil
}

type KClosestNodes struct {
	nodes        []*Node
	targetLen    int
	targetVector vectors.Vector
	distanceFnc  distances.Distance
}

func NewKClosestNodes(k int, targetVector vectors.Vector, distanceFnc distances.Distance) *KClosestNodes {
	kcls := new(KClosestNodes)

	kcls.distanceFnc = distanceFnc
	kcls.targetLen = k
	kcls.targetVector = targetVector
	kcls.nodes = make([]*Node, 0, k)

	return kcls
}

func (hp *KClosestNodes) Len() int {
	return len(hp.nodes)
}

func (hp *KClosestNodes) Swap(i, j int) {
	hp.nodes[i], hp.nodes[j] = hp.nodes[j], hp.nodes[i]
}

func (hp *KClosestNodes) Less(i, j int) bool {
	dstI := hp.distanceFnc(hp.targetVector, hp.nodes[i].Vector)
	dstJ := hp.distanceFnc(hp.targetVector, hp.nodes[j].Vector)

	return dstI > dstJ
}

func (hp *KClosestNodes) Push(x any) {
	node := x.(*Node)

	if len(hp.nodes) < hp.targetLen {
		hp.nodes = append(hp.nodes, node)
	} else {
		dstToNode := hp.distanceFnc(node.Vector, hp.targetVector)
		dstToHead := hp.distanceFnc(hp.nodes[0].Vector, hp.targetVector)

		if dstToNode < dstToHead {
			heap.Pop(hp)
			hp.nodes = append(hp.nodes, node)
		}
	}
}

func (hp *KClosestNodes) Pop() any {
	n := len(hp.nodes)
	node := hp.nodes[n-1]
	hp.nodes = hp.nodes[:n-1]
	return node
}
