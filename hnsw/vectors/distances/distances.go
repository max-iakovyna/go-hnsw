package distances

import (
	"go-hnsw/hnsw/vectors"
	"math"
)

type Distance func(v1, v2 vectors.Vector) vectors.VFloat

var Euclidian Distance = func(vector1, vector2 vectors.Vector) vectors.VFloat {
	var dst vectors.VFloat = 0
	for i := 0; i < len(vector1); i += 1 {
		dst += vectors.VFloat(math.Pow(float64(vector1[i]-vector2[i]), 2))
	}

	return vectors.VFloat(math.Sqrt(float64(dst)))
}

var Cosine Distance = func(vector1, vector2 vectors.Vector) vectors.VFloat {
	var dotProd vectors.VFloat = 0.0
	for i := 0; i < len(vector1); i += 1 {
		dotProd += vector1[i] * vector2[i]
	}

	abs1 := vectors.VectorAbs(vector1)
	abs2 := vectors.VectorAbs(vector2)

	return dotProd / (abs1 * abs2)
}
