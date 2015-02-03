package ml.wolfe.term

import cc.factorie.Factorie.DenseTensor1
import cc.factorie.la.DenseTensor2
import ml.wolfe.{FactorieMatrix, FactorieVector}
import scala.language.implicitConversions

/**
 * @author riedel
 */
object TermImplicits {

  val doubles = Dom.doubles
  val bools = Dom.bools

  def vectors(dim: Int) = new VectorDom(dim)

  def matrices(dim1: Int, dim2: Int) = new MatrixDom(dim1: Int, dim2: Int)

  def discrete[T](args: T*) = new DiscreteDom[T](args.toIndexedSeq)

  def vector(values: Double*) = new DenseTensor1(values.toArray)

  def matrix(values: Seq[Double]*) = {
    val tmp = new DenseTensor2(values.length, values.head.length)

    (0 until values.length).foreach(row => {
      (0 until values(row).length).foreach(col => {
        tmp(row, col) = values(row)(col)
      })
    })

    tmp
  }

  def seqs[D <: Dom](elements: D, length: Int) = new SeqDom(elements, length)

  def seq[E<:Dom](dom:SeqDom[E])(elems:dom.elementDom.Term*) = new SeqTermImpl[E] {

      val domain:dom.type = dom
      def elements = elems.toIndexedSeq

  }

  def sigm[T <: DoubleTerm](term: T) = new Sigmoid(term)

  def log[T <: DoubleTerm](term: T) = new Log(term)

  def I[T <: BoolTerm](term: T) = new Iverson(term)

  implicit def genericToConstant[T,D<:TypedDom[T]](t:T)(implicit dom:D):dom.Term = dom.const(t)

  implicit def seqOfTermsToSeqTerm[T <: Term[Dom], D <: DomWithTerm[T],S<:SeqDom[D]](seq:IndexedSeq[T])(implicit dom:S):dom.Term =
    new SeqTermImpl[D] {
      val elements = seq
      val domain = dom
    }


  //implicit def seqToConstant[T,D<:TypedDom[T]](seq:IndexedSeq[T])(implicit dom:SeqDom[D]):dom.TermType = dom.const(seq)

  //implicit def seqToSeqTerm[E <: Dom : SeqDom](elems:Seq[Term[E]]) = seq(implicitly[SeqDom[E]])(elems: _*)

  implicit def doubleToConstant(d: Double): Constant[DoubleDom] = new Constant[DoubleDom](Dom.doubles, d)

  implicit def vectToConstant(d: FactorieVector): Constant[VectorDom] = new Constant[VectorDom](vectors(d.dim1), d)

  implicit def matToConstant(d: FactorieMatrix): Constant[MatrixDom] = new Constant[MatrixDom](matrices(d.dim1, d.dim2), d)

//  implicit def discToConstant[T: DiscreteDom](value: T): Constant[DiscreteDom[T]] =
//    new Constant[DiscreteDom[T]](implicitly[DiscreteDom[T]], value)

//  def argmax[D <: Dom](dom: D)(obj: dom.Variable => DoubleTerm): dom.Value = {
//    val variable = dom.variable("_hidden")
//    val term = obj(variable)
//    term.argmax(variable).asInstanceOf[dom.Value]
//  }

  def argmax[D <: Dom](dom: D)(obj: dom.Variable => DoubleTerm):Argmax[dom.This] = {
    val variable = dom.variable("_hidden")
    val term = obj(variable)
    new Argmax(term,variable)
  }

  def max[D <: Dom](dom: D)(obj: dom.Variable => DoubleTerm) = {
    val variable = dom.variable("_hidden")
    val term = obj(variable)
    new Max(term, Seq(variable))
  }

  def sum[T](dom:Seq[T])(arg:T => DoubleTerm) = new Sum(dom.toIndexedSeq.map(arg))

  implicit class RichDoubleTerm(term: DoubleTerm) {
    def +(that: DoubleTerm) = new Sum(IndexedSeq(term, that))
    def -(that:DoubleTerm) = new Sum(IndexedSeq(term, that * (-1.0)))
    def *(that: DoubleTerm): Product = new Product(IndexedSeq(term, that))
    def argmaxBy(factory: ArgmaxerFactory) = new TermProxy[DoubleDom] {
      def self = term
      override def argmaxer(wrt: Seq[Var[Dom]]) = factory.argmaxer(term,wrt)
    }
  }

  implicit class RichBoolTerm(term: BoolTerm) {
    def &&(that: BoolTerm) = new And(term, that)

    def ||(that: BoolTerm) = new Or(term, that)

    def -->(that: BoolTerm) = new Implies(term, that)
  }

  implicit class RichDiscreteTerm[T](term: DiscreteTerm[T]) {
    def ===(that: DiscreteTerm[T]) = new DiscreteEquals(term, that)
  }

  implicit class RichTerm[D <: Dom](val term: Term[D]) {
    def apply(args: Any*) = term.apply(args)
  }

  implicit class RichDom[D <: Dom](val dom: D) {
    def x[D2 <: Dom](that: D2) = new Tuple2Dom[D, D2](dom, that)
  }

  implicit class RichVectTerm(val vect: Term[VectorDom]) {
    def dot(that: Term[VectorDom]) = new DotProduct(vect, that)
  }

  implicit class RichMatrixTerm(val mat: Term[MatrixDom]) {
    //def dot(that: Term[MatrixDom]) = new DotProduct(mat, that)
    def *(that: Term[VectorDom]) = new MatrixVectorProduct(mat, that)
  }

}
