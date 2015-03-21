package ml.wolfe.term

import ml.wolfe.term.TermImplicits._

import scala.util.Random

/**
 * @author riedel
 */
object LearningObjective {

  implicit val random = new Random(0)

  //want to write perceptron(train)(Labels)(x => y => model(w)(x)(y)

  def perceptron[X <: Dom, Y <: Dom, W <: Dom](data:SeqTerm[X x Y])
                                              (space: Y)
                                              (model: X#Term => Y#Term => DoubleTerm):DoubleTerm = {

    //shuffled(data) { i => max(space) {model(i._1)} - model(i._1)(i._2)}
    shuffled(data) { i => model(i._1)(i._2) - max(space) {model(i._1)}}
  }

}
