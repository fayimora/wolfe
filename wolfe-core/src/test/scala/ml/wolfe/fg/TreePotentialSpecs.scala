package ml.wolfe.fg

import ml.wolfe.FactorGraph.Edge
import ml.wolfe.macros.OptimizedOperators
import ml.wolfe.{BruteForceOperators, Wolfe, WolfeSpec}

/**
 * @author Sebastian Riedel
 */
class TreePotentialSpecs extends WolfeSpec {


  import TreePotential._

  def graph[T](pairs: (T, T)*) = (pairs map (x => x -> true)).toMap withDefaultValue false

  "A tree constraint checker" should {
    "find a loop" in {
      isFullyConnectedNonProjectiveTree(graph(0 -> 1, 1 -> 2, 2 -> 0)) should be(false)
    }

    "find a legal tree" in {
      isFullyConnectedNonProjectiveTree(graph(0 -> 1, 1 -> 2)) should be(true)
    }
  }

  "A TreePotential" should {
    "Blah blah blah" in {

      import Wolfe._
      import BruteForceOperators._

      @Wolfe.Potential(new TreePotential(_: Map[(Any, Any), Edge], true))
      def tree[T](tree: Map[(T, T), Boolean]) =
        if (isFullyConnectedNonProjectiveTree(tree)) 0.0 else Double.NegativeInfinity

      type Node = Int
      type Graph = Pred[(Node,Node)]
      val slen = 3
      val tokens = 0 until slen
      def edges = tokens x tokens
      def graphs = preds(edges)
      def query(graph:Graph) = sum(edges) {e => oneHot(e, I(graph(e)))}

      val result = graphs map (g => tree(g))
      val marginals = BruteForceOperators.expect(graphs)(tree)(query)

      //def bp = OptimizedOperators.expect(graphs)(tree)(query)

      println("Yo")
      println(marginals)



    }
  }

}
