package vectors

import (
	"fmt"
	"math"
)

type VFloat float64

type Vector []VFloat

func (vector Vector) String() string {

	s := "Vector("

	for i, v := range vector {
		s += fmt.Sprintf("%f", float64(v))
		if i != len(vector) {
			s += ", "
		}
	}
	s += ")"
	return s
}

func VectorAbs(vector Vector) VFloat {
	var abs VFloat = 0.0
	for i := 0; i < len(vector); i += 1 {
		abs += vector[i] * vector[i]
	}
	return VFloat(math.Sqrt(float64(abs)))
}
