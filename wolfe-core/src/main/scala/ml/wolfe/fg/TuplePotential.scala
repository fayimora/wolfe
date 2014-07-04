package ml.wolfe.fg

import cc.factorie.la.SingletonTensor
import ml.wolfe.FactorGraph.{Factor, FGPrinter, Edge, Node}
import ml.wolfe.{FactorGraph, FactorieVector}
import ml.wolfe.MoreArrayOps._
import ml.wolfe.util.Multidimensional._
import scalaxy.loops._

/**
 * @author Luke
 */

trait TuplePotential extends Potential {
  val baseNodes:Array[Node] // All nodes in the original factor graph that pertain to this potential
}

/**
 * A potential that maintains consistency between two nodes of the junction tree
 */
final class TupleConsistencyPotential(edge1: Edge, edge2: Edge) extends TuplePotential {
  val v1 = edge1.n.variable.asTuple
  val v2 = edge2.n.variable.asTuple
  val m1 = edge1.msgs.asTuple
  val m2 = edge2.msgs.asTuple

  val baseNodes = v1.componentNodes intersect v2.componentNodes
  val baseVariables = baseNodes.map(_.variable.asDiscrete)

  override def toString = "Consistency " + baseNodes.map(_.index.toString).mkString("(",",",")")

  override def valueForCurrentSetting() = {
    if(baseVariables.forall (v => v1.componentSetting(v) == v2.componentSetting(v))) 0.0
    else Double.NegativeInfinity
  }

  override def maxMarginalF2N(edge: Edge) = {
    val (thisMsg, thatMsg) = if (edge == edge1) (m1, m2) else (m2, m1)

    thatMsg.n2f.foldInto(Double.NegativeInfinity, math.max, thisMsg.f2n)
    maxNormalize(thisMsg.f2n.array)
  }

  override def marginalF2N(edge: Edge) = {
    val (thisMsg, thatMsg) = if (edge == edge1) (m1, m2) else (m2, m1)

    thatMsg.n2f.foldInto(0.0, (sum:Double, x:Double) => sum + math.exp(x), thisMsg.f2n)

    normalize(thisMsg.f2n.array)
    log(thisMsg.f2n.array)
  }

  override def maxMarginalExpectationsAndObjective(result: FactorieVector) = {
    val positive1:LabelledTensor[DiscreteVar, Boolean] =
      m1.n2f.fold(baseVariables, false, (pos:Boolean, x:Double) => pos || x > Double.NegativeInfinity)
    val positive2:LabelledTensor[DiscreteVar, Boolean] =
      m1.n2f.fold(baseVariables, false, (pos:Boolean, x:Double) => pos || x > Double.NegativeInfinity)

    if((0 until positive1.array.length).exists(i => positive1.array(i) && positive2.array(i)))
      0.0 else Double.NegativeInfinity
  }

}

/**
 * A wrapper potential, which sums over a collection of component potentials
 * @param components The component factors in the junction tree
 * @param edge the edge between this GroupPotential and the TupleNode it communicates with
 */
final class GroupPotential(val components: Array[Factor], val edge:Edge, val baseNodes:Array[Node]) extends TuplePotential {
  val v = edge.n.variable.asTuple
  val m = edge.msgs.asTuple

  def componentVariables(f:Factor) = f.edges.map(_.n.variable.asDiscrete)

  override def toString = "Group " + baseNodes.map(_.index.toString).mkString("(",",",")")

  override def valueForCurrentSetting() = {
    v.updateComponentSettings()
    components.map(_.potential.valueForCurrentSetting()).sum
  }

  override def marginalF2N(edge: Edge) = {
    val scoretables = components.map(f => f.potential.getScoreTable(componentVariables(f)))
    m.f2n.copyFrom(scoretables.head)
    for(t <- scoretables.tail) m.f2n += t
    maxNormalize(m.f2n.array)
  }
  override def maxMarginalF2N(edge:Edge) = marginalF2N(edge)

  override def maxMarginalExpectationsAndObjective(result: FactorieVector) = {
    val scoretables = components.map(f => f.potential.getScoreTable(componentVariables(f)))
    val scoreSums:LabelledTensor[DiscreteVar, Double] =
      LabelledTensor.onNewArray[DiscreteVar, Double](v.components, _.dim, 0.0)
    for(t <- scoretables) scoreSums += t

    val scorePairs : LabelledTensor[DiscreteVar, (Double, Double)] =
      m.n2f.elementWiseOp[Double, (Double, Double)](scoreSums, (n2f, score) => (score, score+n2f))

    var maxScore = Double.NegativeInfinity
    var maxPenalisedScore = Double.NegativeInfinity
    var maxIndices:Int = 0
    for(i <- (0 until scorePairs.array.length).optimized) {
      if(scorePairs.array(i)._2 > maxPenalisedScore) {
        maxPenalisedScore = scorePairs.array(i)._2
        maxScore = scorePairs.array(i)._1
        maxIndices = 1
      } else if(scorePairs.array(i)._2 == maxPenalisedScore) {
        maxIndices = maxIndices + 1
      }
    }

    val prob = 1d / maxIndices
    for (f <- components if f.potential.isLinear) { //todo: yuck!
      val maxMarginal = scorePairs.fold[Double](componentVariables(f), 0, {
        (acc:Double, x:(Double, Double)) =>
          if(x._2 == maxPenalisedScore) acc + prob else acc
      })
      f.potential match {
        case p: LinearPotential =>
          val R = maxMarginal.permute(p.edges.map(_.n.variable.asDiscrete), allowSameArray = true)
          for(i <- 0 until R.array.length if R.array(i)!=0)
              result +=(p.statistics.vectors(i), R.array(i))
      }
    }
    maxScore
  }
}