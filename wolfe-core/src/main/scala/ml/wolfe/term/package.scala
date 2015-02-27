package ml.wolfe

/**
 * @author riedel
 */
package object term {
  type DoubleVar = DoubleDom#DomVar
  type VectorVar = VectorDom#DomVar
  type MatrixVar = MatrixDom#DomVar
  type DiscVar[T] = GenericDiscreteDom[T]#DomVar
  type Constant[D <: Dom] = D#Constant

  type TypedDom[T] = Dom { type Value = T}
  type TypedTerm[T] = Term[TypedDom[T]]
  type DomWithTerm[T] = Dom { type Term = T}
  type DoubleTerm = TypedTerm[Double]
  type VectorTerm = Term[GenericVectorDom]
  type MatrixTerm = Term[MatrixDom]
  type BoolDom = DiscreteDom[Boolean]
  type BoolTerm = Term[BoolDom]
  type DiscreteTerm[T] = Term[DiscreteDom[T]]
//  type IntTerm = DiscreteTerm[Int]
  type DoubleSeqDom = SeqDom[DoubleDom]

}
