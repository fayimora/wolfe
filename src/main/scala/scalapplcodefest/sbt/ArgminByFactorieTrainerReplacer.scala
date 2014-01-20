package scalapplcodefest.sbt

import scala.tools.nsc.Global
import scalapplcodefest._
import cc.factorie.WeightsSet
import cc.factorie.optimize.{Example, Trainer}
import cc.factorie.la.WeightsMapAccumulator
import cc.factorie.util.DoubleAccumulator
import scalapplcodefest.newExamples.SumOfQuadraticFunctions
import scala.reflect.internal.Flags

/**
 * @author Sebastian Riedel
 */
class ArgminByFactorieTrainerReplacer(val env: GeneratorEnvironment)
  extends CodeStringReplacer with WolfePatterns {

  this: Differentiator =>

  import env.global._

  def replace(tree: env.global.Tree, modification: ModifiedSourceText) = {
    //assume a sum
    val replaced = env.replaceMethods(tree)
    val reduced = env.simplifyBlocks(env.betaReduce(replaced))

    reduced match {
      //      case ApplyArgmin2(_, _, _, _, _) =>
      //        println(replaced)
      //        println(reduced)
      //        false
      case ApplyArgmin2(_, _, _, _, Function(List(w), ApplySum2(_, _, data, _, Function(List(y_i), perInstance), _)), _) =>
        val indexId = "index"
        val weightsId = "weights"
        differentiate(perInstance, w.symbol, indexId, weightsId) match {
          case Some(gradientValue) =>
            val indentation = modification.indentationOfLineAt(tree.pos.start)
            println(indentation)
            val replacement = ArgminByFactorieTrainer.generateCode(
              data.symbol.name.toString, y_i.symbol.name.toString, indexId, weightsId, gradientValue, indentation + 2)
            modification.replace(tree.pos.start, tree.pos.end, replacement)
            true
          case _ => false
        }

      case other =>
        false

    }
  }
}


object ArgminByFactorieTrainerReplacer {
  def main(args: Array[String]) {
    val className = classOf[SumOfQuadraticFunctions].getName.replaceAll("\\.", "/")
    GenerateSources.generate(
      sourcePath = s"src/main/scala/$className.scala",
      replacers = List(g => new ArgminByFactorieTrainerReplacer(g) with SimpleDifferentiator))
  }
}


trait Differentiator extends InGeneratorEnvironment {

  import env.global._

  //this should return a string representation of (factorie gradient vector, objective value)
  def differentiate(objective: Tree, variable: Symbol,
                    indexIdentifier: String, weightIdentifier: String): Option[String]

}

trait SimpleDifferentiator extends Differentiator with WolfePatterns {

  //this should return a string representation of (factorie gradient vector, objective value)
  def differentiate(objective: env.global.Tree, variable: env.global.Symbol,
                    indexIdentifier: String, weightIdentifier: String) = {

    println(s"Differentiating $objective wrt $variable")
    import env.global._
    val DoubleSymbol = definitions.DoubleClass
    val minus = definitions.getMemberMethod(definitions.DoubleClass, newTermName("$minus"))

    objective match {
      case ApplyDoubleMinus(arg1, arg2) =>
        for (g1 <- differentiate(arg1, variable, indexIdentifier, weightIdentifier);
             g2 <- differentiate(arg2, variable, indexIdentifier, weightIdentifier)) yield
          s"{val (g1,v1) = $g1; val (g2,v2) = $g2; (g1 - g2, v1 - v2)}"

      case ApplyMax2(se, types, dom, pred, obj@Function(List(y), body), num) =>
        //todo: this should be an optimized version of the argmax
        //todo: need to create appropriate select
        println(se)
        val argmax = ApplyArgmax2(Select(Select(Ident(scalapplcodefest),wolfe),argmax2), types, dom, pred, obj, num)
        val argmaxSymbol = objective.symbol.owner.newValue(newTermName("_best"))
        val binding = Map(y.symbol -> Ident(argmaxSymbol))
        val atOptimum = env.substitute(body, binding)
        for (g <- differentiate(atOptimum, variable, indexIdentifier, weightIdentifier)) yield {
          s"{val ${argmaxSymbol.encodedName} = $argmax; $g}"
        }

      case DotProduct(arg1, arg2) if !arg1.exists(_.symbol == variable) && arg2.symbol == variable =>
        val toSparseFVector = "scalapplcodefest.sbt.FactorieConverter.toFactorieSparseVector"
        Some(s"($toSparseFVector($arg1,$indexIdentifier),$objective)")
        Some(s"($toSparseFVector(Wolfe.VectorZero,$indexIdentifier),0.0)")

      case _ =>
        val toSparseFVector = "scalapplcodefest.sbt.FactorieConverter.toFactorieSparseVector"
        Some(s"($toSparseFVector(Wolfe.VectorZero,$indexIdentifier),0.0)")
    }

  }
}

object FactorieConverter {

  import Wolfe.{Vector => WVector}
  import scalapplcodefest.{Vector => FVector}

  def toFactorieSparseVector(vector: WVector, index: Index): FVector = {
    val sparse = new SparseVector(vector.size)
    for ((key, value) <- vector) sparse(index(Seq(key))) = value
    sparse
  }
  def toWolfeVector(fvector: FVector, index: Index): WVector = {
    val inverse = index.inverse()
    val map = for ((key, value) <- fvector.activeElements) yield inverse(key)(0) -> value
    map.toMap
  }

}

object ArgminByFactorieTrainer {

  import scalapplcodefest.Vector

  def argmin[T](data: Seq[T],
                gradientValue: (Index, Vector) => (Vector, Double),
                trainerFor: WeightsSet => Trainer) = {
    val index = new Index
    val weightsSet = new WeightsSet
    val key = weightsSet.newWeights(new DenseVector(10000))
    val examples = for (instance <- data) yield new Example {
      def accumulateValueAndGradient(value: DoubleAccumulator, gradient: WeightsMapAccumulator) = {
        val weights = weightsSet(key).asInstanceOf[Vector]
        val (g, v) = gradientValue(index, weights)
        value.accumulate(v)
        gradient.accumulate(key, g, -1.0)
      }
    }
    val trainer = trainerFor(weightsSet)
    trainer.trainFromExamples(examples)
    weightsSet(key).asInstanceOf[Vector]
    ??? //convert to Wolfe.Vector
  }

  def generateCode(data: String, instanceVar: String, indexId: String, weightId: String, gradientValue: String,
                   indent: Int = 6) =
    s"""{ //this code calls the factorie learner
      |import cc.factorie.WeightsSet
      |import cc.factorie.optimize.{Example, Trainer}
      |import cc.factorie.la.WeightsMapAccumulator
      |import cc.factorie.util.DoubleAccumulator
      |import cc.factorie.optimize.{Perceptron, OnlineTrainer}
      |import scalapplcodefest._
      |val $indexId = new Index
      |val _weightsSet = new WeightsSet
      |val _key = _weightsSet.newWeights(new DenseVector(10000))
      |val examples = for ($instanceVar <- $data) yield new Example {
      |  def accumulateValueAndGradient(value: DoubleAccumulator, gradient: WeightsMapAccumulator) = {
      |    val $weightId = _weightsSet(_key).asInstanceOf[Vector]
      |    val (g,v) = $gradientValue
      |    value.accumulate(v)
      |    gradient.accumulate(_key, g, -1.0)
      |  }
      |}
      |val _trainer = new OnlineTrainer(_weightsSet, new Perceptron, 5)
      | _trainer.trainFromExamples(examples)
      |val fweights = _weightsSet(_key).asInstanceOf[Vector]
      |scalapplcodefest.sbt.FactorieConverter.toWolfeVector(fweights,$indexId)}
    """.replaceAll("\\|", Array.fill(indent)(" ").mkString("|", "", "")).stripMargin

}

