package ml.wolfe.term

import cc.factorie.la.{SparseBinaryTensor1, SparseIndexedTensor1, DenseTensor1}
import com.typesafe.scalalogging.slf4j.LazyLogging
import ml.wolfe.term.TermImplicits.Assignment
import ml.wolfe.util.Math._

import scala.collection.mutable
import scala.util.Random

/**
 * A term is an object that represents a computation of a value based on an assignment to free variables in the term.
 * The term is typed, in the sense that it is guaranteed to produce values in a domain [[ml.wolfe.term.Dom]].
 * The term's computation is defined through the term's evaluator object. Terms can also provide means to differentiate,
 * optimize or marginalize the term with respect to some variables.
 * @tparam D the type of domain this term produces values in.
 */
trait Term[+D <: Dom] extends TermHelper[D]  {

  term =>

  /**
   * The domain of the term. Evaluating the term will yield a value within this domain.
   */
  val domain: D

  /**
   * The sequence of (distinct) free variables in this term. Notice that the order of this sequence determines the order
   * of arguments to be passed into the evaluation methods.
   * @return the sequence of free variables in this term.
   */
  def vars: Seq[AnyVar]

  /**
   * Is this term guaranteed to evaluate to the same value each it is called
   * @return true iff the evaluator always evaluates to the same value (over all executions)
   */
  lazy val isStatic = vars.isEmpty && !isStochastic

  /**
   * Does this term have some stochastic subterms.
   */
  lazy val isStochastic = toList.exists(_.isInstanceOf[SampleTerm])

  /**
   * Return a list of all sub terms in this term (including the term itself).
   * @return all subterms.
   */
  def toList:List[AnyTerm] = this :: Nil

  def precalculate:domain.Term =
    if (isStatic) domain.own(Precalculated(this).asInstanceOf[TypedTerm[domain.Value]])
    else domain.own(this.asInstanceOf[TypedTerm[domain.Value]])

  def evaluatorImpl(in: Settings): Evaluator = ???

  def differentiatorImpl(wrt: Seq[AnyVar])(in: Settings, err: Setting, gradientAcc: Settings): Differentiator =
    new EmptyDifferentiator(in, err, gradientAcc, wrt)

  def maxMarginalizerImpl(wrt: Seq[AnyVar], observed: Seq[AnyVar])
                         (input: Settings, inputMsgs: Msgs, reverseMsgsAlso: Boolean = false): MaxMarginalizer = {
    //todo: this could be type safe, for example by adding the maxMarginalizerImpl method to the RichDoubleTerm
    val varying = vars filterNot observed.contains
    if (!domain.isDouble) sys.error("MaxMarginals only supported for real valued terms")
    else if (varying.forall(_.domain.isDiscrete))
      new ExhaustiveSearchMaxMarginalizer(this.asInstanceOf[DoubleTerm], wrt, observed, input, inputMsgs, reverseMsgsAlso)
    else ???
  }

  def marginalizerImpl(wrt: Seq[AnyVar], observed: Seq[AnyVar])
                         (input: Settings, inputMsgs: Msgs, reverseMsgsAlso: Boolean = false): Marginalizer = {
    //todo: this could be type safe, for example by adding the marginalizerImpl method to the RichDoubleTerm
    val varying = vars filterNot observed.contains
    if (!domain.isDouble) sys.error("Marginals only supported for real valued terms")
    else if (varying.forall(_.domain.isDiscrete))
      new ExhaustiveSearchMarginalizer(this.asInstanceOf[DoubleTerm], wrt, observed, input, inputMsgs, reverseMsgsAlso)
    else ???
  }


  def argmaxerImpl(wrt: Seq[AnyVar])(observed: Settings, msgs: Msgs): Argmaxer = {
    //todo: this could be type safe, for example by adding the argmax method to the RichDoubleTerm
    if (!domain.isDouble) sys.error("Argmax only supported for real valued terms")
    else if (wrt.forall(_.domain.isDiscrete)) new ExhaustiveSearchArgmaxer(this.asInstanceOf[DoubleTerm],wrt)(observed,msgs) else ???
  }

  def samplerImpl(wrt: Seq[AnyVar])(observed: Settings, msgs: Msgs)(implicit random:Random): Sampler = {
    //todo: this could be type safe, for example by adding the argmax method to the RichDoubleTerm
    if (!domain.isDouble) sys.error("Sampler only supported for real valued terms")
    else if (wrt.forall(_.domain.isDiscrete)) new ExhaustiveSampler(this.asInstanceOf[DoubleTerm],wrt)(observed,msgs) else ???
  }


  trait Evaluator extends ml.wolfe.term.Evaluator {

    override def toString = term.toString + ".evaluator"
  }

  trait Differentiator extends ml.wolfe.term.Differentiator  {
    val needBackward = withRespectTo.exists(vars.contains)

    def optimizedBackward()(implicit execution: Execution): Unit = {
      if (needBackward) backward()
    }

    override def toString = term.toString + ".differentiator"


  }

  class ProxyEvaluator(val self: ml.wolfe.term.Evaluator) extends Evaluator {
    val input = self.input
    val output = self.output
    def eval()(implicit execution: Execution) = self.eval()
  }

  class ProxyDifferentiator(val self: ml.wolfe.term.Differentiator) extends Differentiator {
    val gradientAccumulator = self.gradientAccumulator
    val error = self.error
    val input = self.input
    val output = self.output

    def forward()(implicit execution: Execution) = self.forward()
    def backward()(implicit execution: Execution) = self.backward()
    def withRespectTo = self.withRespectTo
  }


  abstract class AbstractEvaluator(val input: Settings) extends Evaluator

  trait ArgmaxEvaluator {
    val maxer:Argmaxer
  }

  abstract class AbstractDifferentiator(val input: Settings,
                                        val error: Setting,
                                        val gradientAccumulator: Settings,
                                        val withRespectTo: Seq[AnyVar]) extends Differentiator

  class EmptyDifferentiator(in: Settings, err: Setting, gradientAcc: Settings, withRespectTo: Seq[AnyVar]) extends AbstractDifferentiator(in, err, gradientAcc,withRespectTo) {
    val eval = evaluatorImpl(in)
    val output = eval.output

    def forward()(implicit execution: Execution): Unit = {
      eval.eval()(execution.copy(typ = Execution.EmptyDiff)) //ensures that log terms still render
    }

    def backward()(implicit execution: Execution) {}
  }


}


/**
 * Provides a range of convenience methods to access Term functionality in more type safe ways.
 * @tparam D
 */
trait TermHelper[+D <: Dom] extends LazyLogging {
  this: Term[D] =>

  def evaluator() = new ClientEvaluator

  def differentiator[T <: Dom](wrt: Var[T]) = new ClientDifferentiator[T](wrt)

  class ClientEvaluator() {
    val input = createInputSettings()
    val eval = evaluatorImpl(input)
    var count = 0

    def evalUntyped(args: Any*): domain.Value = {
      for ((v, i) <- vars.zipWithIndex) v.domain.copyValue(args(i).asInstanceOf[v.domain.Value], input(i))
      eval.eval()(Execution(count))
      count += 1
      domain.toValue(eval.output)
    }

    def eval(at: Assignment[Dom]*): domain.Value = {
      val values = at.map(a => a.variable -> a.value).toMap
      val args = vars.map(values)
      evalUntyped(args: _*)
    }


  }

  class ClientDifferentiator[T <: Dom](val wrt: Var[T]) {
    val input = createInputSettings()
    val error = domain.toSetting(domain.one)
    val gradient = createZeroInputSettings()
    val diff = differentiatorImpl(Seq(wrt))(input, error, gradient)
    val indexOfWrt = vars.indexOf(wrt)
    var count = 0

    def differentiate(args: Any*): wrt.domain.Value = {
      for ((v, i) <- vars.zipWithIndex) v.domain.copyValue(args(i).asInstanceOf[v.domain.Value], input(i))
      gradient.foreach(_.resetToZero())
      diff.differentiate()(Execution(count, Execution.Diff))
      count += 1
      wrt.domain.toValue(gradient(indexOfWrt))
    }
  }

  def maxMarginals[T <: Dom](target: Var[T], wrt: Var[T])(incoming: wrt.domain.Marginals)(args: Any*): target.domain.Marginals = {
    val observed = vars.filter(v => v != wrt && v != target)
    val observedSettings = Dom.createSettings(observed, args)
    val incomingMsgs = Msgs(Seq(wrt.domain.toMsgs(incoming)))
    val maxMarger = maxMarginalizerImpl(Seq(wrt), observed)(observedSettings, incomingMsgs)
    maxMarger.updateMessages()(Execution(0))
    target.domain.toMarginals(maxMarger.outputMsgs(0))
  }

  def marginals[T <: Dom](target: Var[T], wrt: Var[T])(incoming: wrt.domain.Marginals)(args: Any*): target.domain.Marginals = {
    val observed = vars.filter(v => v != wrt && v != target)
    val observedSettings = Dom.createSettings(observed, args)
    val incomingMsgs = Msgs(Seq(wrt.domain.toMsgs(incoming)))
    val maxMarger = marginalizerImpl(Seq(wrt), observed)(observedSettings, incomingMsgs)
    maxMarger.updateMessages()(Execution(0))
    target.domain.toMarginals(maxMarger.outputMsgs(0))
  }


  def argmax[V <: Dom](wrt: Var[V], args: Any*): wrt.domain.Value = {
    val observed = vars.filter(_ != wrt)
    val observedSettings = for ((a, v) <- args zip observed) yield v.domain.toSetting(a.asInstanceOf[v.domain.Value])
    val am = argmaxerImpl(Seq(wrt))(Settings.fromSeq(observedSettings),null)
    val result = wrt.domain.createSetting()
    am.argmax()(Execution(0))
    wrt.domain.toValue(am.result(0))
  }


  def evalUntyped(args: Any*): domain.Value = {
    val argSettings = createSettings(args)
    val ev = evaluatorImpl(argSettings)
    ev.eval()(Execution(0))
    domain.toValue(ev.output)
  }

  def evalResultUntyped(args: Any*): EvalResult[domain.Value] = {
    val argSettings = createSettings(args)
    val ev = evaluatorImpl(argSettings)
    ev.eval()(Execution(0))
    new EvalResult(domain.toValue(ev.output), ev)
  }

//  def !! = eval()


  def gradient[V <: Dom](wrt: Var[V], args: Any*): wrt.domain.Value = {
    require(args.length == vars.length, s"You need as many arguments as there are free variables (${vars.length})")
    val gradient = createZeroInputSettings()
    val diff = differentiatorImpl(Seq(wrt))(createSettings(args), domain.toSetting(domain.one), gradient)
    diff.differentiate()(Execution(0, Execution.Diff))
    val indexOfWrt = vars.indexOf(wrt)
    wrt.domain.toValue(gradient(indexOfWrt))
  }


  def createVariableSettings() = vars.map(_.domain.createSetting()).toArray

  def createSettings(args: Seq[Any]) = {
    Settings.fromSeq((args zip vars) map { case (a, v) => v.domain.toSetting(a.asInstanceOf[v.domain.Value])})
  }

  def createInputSettings() = {
    Settings.fromSeq(vars.map(_.domain.createSetting()))
  }

  def createZeroInputSettings() = {
    Settings.fromSeq(vars.map(_.domain.createZeroSetting()))
  }

  def createSparseZeroInputSettings() = {
    Settings.fromSeq(vars.map(_.domain.createSparseZeroSetting()))
  }


}

trait ProxyTerm[D <: Dom] extends Term[D] with NAry {
  def self: Term[D]

  val domain = self.domain

  def vars = self.vars


  /**
   * Is this term guaranteed to evaluate to the same value each it is called
   * @return true iff the evaluator always evaluates to the same value (over all executions)
   */
  //def isStatic = self.isStatic

  type ArgumentType = Term[D]

  def arguments = IndexedSeq(self)

  override def evaluatorImpl(in: Settings) = new ProxyEvaluator(self.evaluatorImpl(in))

  override def differentiatorImpl(wrt: Seq[AnyVar])(in: Settings, err: Setting, gradientAcc: Settings) =
    new ProxyDifferentiator(self.differentiatorImpl(wrt)(in, err, gradientAcc))

  override def toString = self.toString

  override def equals(obj: scala.Any) = obj match {
    case t: ProxyTerm[_] =>
      t.self == self && t.getClass == getClass
    case _ => false
  }
}

trait OwnedTerm[T] extends ProxyTerm[TypedDom[T]] {

  def keep:Boolean = false

}




class Div[T1 <: DoubleTerm, T2 <: DoubleTerm](val arg1: T1, val arg2: T2) extends ComposedDoubleTerm {

  self =>

  type ArgumentType = DoubleTerm

  val arguments = IndexedSeq(arg1, arg2)

  def copy(args: IndexedSeq[ArgumentType]) = new Div(args(0), args(1))


  override def composer(args: Settings) = new Composer(args) {
    def eval()(implicit execution: Execution) = {
      output.cont(0) = input(0).cont(0) / input(1).cont(0)
    }
  }

  override def differentiatorImpl(wrt: Seq[AnyVar])(in: Settings, err: Setting, gradientAcc: Settings) =
    new ComposedDifferentiator(wrt, in, err, gradientAcc) {
      def localBackProp()(implicit execution: Execution) = {
        val scale = error.cont(0)

        val x = argOutputs(0).cont(0)
        val y = argOutputs(1).cont(0)

        argErrors(0).cont(0) = scale / y
        argErrors(1).cont(0) = -(scale * x) / (y * y)
      }
    }
}


trait ComposedDoubleTerm extends DoubleTerm with Composed[Dom.doubles.type] {
  val domain:Dom.doubles.type = Dom.doubles
}


trait UnaryTerm[T <: Term[Dom], D <: Dom] extends Composed[D] {
  def arg: T

  type ArgumentType = T

  val arguments = IndexedSeq(arg)
}


case class Product(arguments: IndexedSeq[DoubleTerm]) extends ComposedDoubleTerm {

  type ArgumentType = DoubleTerm


  override def composer(args: Settings) = new Composer(args) {
    def eval()(implicit execution: Execution) = {
      output.cont(0) = 1.0
      for (i <- 0 until input.length)
        output.cont(0) *= input(i).cont(0)
    }
  }


  override def differentiatorImpl(wrt: Seq[AnyVar])(in: Settings, err: Setting, gradientAcc: Settings) =
    new ComposedDifferentiator(wrt, in, err, gradientAcc) {
      def localBackProp()(implicit execution: Execution) = {
        var total = error.cont(0)
        var zeros = 0
        var zeroIndex = -1
        for (i <- 0 until argOutputs.length) {
          //set to 0.0 by default
          argErrors(i) := 0.0
          if (argOutputs(i).cont(0) == 0.0) {
            zeros += 1
            zeroIndex = i
          } else
            total *= argOutputs(i).cont(0)
        }
        if (zeros == 1) {
          argErrors(zeroIndex).cont(0) = total
        } else if (zeros == 0) {
          for (i <- 0 until argOutputs.length)
            argErrors(i).cont(0) = total / argOutputs(i).cont(0)
        }
      }
    }

  def copy(args: IndexedSeq[ArgumentType]) = new Product(args)

}




case class Iverson[T <: BoolTerm](arg: T) extends UnaryTerm[T, DoubleDom] with ComposedDoubleTerm {


  override def composer(args: Settings) = new Composer(args) {
    def eval()(implicit execution: Execution) = {
      output.cont(0) = if (input(0).disc(0) == 0) 0.0 else 1.0
    }
  }

  def copy(args: IndexedSeq[ArgumentType]) = new Iverson(args(0))

  def differentiatorOld(wrt: Seq[AnyVar]) = ???

  override def toString = s"I($arg)"
}



case class LogIverson[T <: BoolTerm](arg: T) extends UnaryTerm[T, DoubleDom] with ComposedDoubleTerm {


  override def composer(args: Settings) = new Composer(args) {
    def eval()(implicit execution: Execution) = {
      output.cont(0) = if (input(0).disc(0) == 0) Double.NegativeInfinity else 0.0
    }
  }

  def copy(args: IndexedSeq[ArgumentType]) = new LogIverson(args(0))

  def differentiatorOld(wrt: Seq[AnyVar]) = ???

  override def toString = s"logI($arg)"
}

class IntToDouble[T <: IntTerm](val int: T) extends ComposedDoubleTerm {
  type ArgumentType = T

  def arguments = IndexedSeq(int)

  def copy(args: IndexedSeq[ArgumentType]) = new IntToDouble(args(0))


  override def composer(args: Settings) = new Composer(args) {
    def eval()(implicit execution: Execution) = {
      output.cont(0) = input(0).disc(0)
    }
  }


  def differentiatorOld(wrt: Seq[AnyVar]) = ???
}

class DoubleToDouble[T <: DoubleTerm](val double: T) extends ComposedDoubleTerm {
  type ArgumentType = T

  def arguments = IndexedSeq(double)

  def copy(args: IndexedSeq[ArgumentType]) = new DoubleToDouble(args(0))

  override def composer(args: Settings) = new Composer(args) {
    def eval()(implicit execution: Execution) = {
      output.cont(0) = double.domain.toValue(input(0))
    }
  }

}

case class Identity[D <: Dom, T <: Term[D]](self: T) extends ProxyTerm[D] {
  def copy(args: IndexedSeq[ArgumentType]) = new Identity[D, T](args(0).asInstanceOf[T])

}


trait BinaryDiscreteOperator[D <: Dom, A <: Dom] extends Composed[D] {

  self =>

  type ArgumentType = Term[A]

  def arg1: Term[A]

  def arg2: Term[A]

  def op(a1: Int, a2: Int): Int

  def name:String

  override def toString = s"$arg1 $name $arg2"

  val arguments = IndexedSeq(arg1, arg2)


  override def composer(args: Settings) = new Composer(args) {
    def eval()(implicit execution: Execution) = {
      output.disc(0) = op(input(0).disc(0), input(1).disc(0))
    }
  }


  def copy(args: IndexedSeq[ArgumentType]) = new BinaryDiscreteOperator[D, A] {

    val domain: D = self.domain

    def arg1 = args(0)

    def arg2 = args(1)

    def op(a1: Int, a2: Int) = self.op(a1, a2)

    def name = self.name
  }
}

case class Unary_!(arg: BoolTerm) extends UnaryTerm[BoolTerm, Dom.bools.type] {
  override def copy(args: IndexedSeq[BoolTerm]) = new Unary_!(args(0))

  override val domain: Dom.bools.type = Dom.bools

  override def composer(args: Settings) = new Composer(args) {
    def eval()(implicit execution: Execution) = {
      output.disc(0) = 1 - input(0).disc(0)
    }
  }
}

case class InverseLogIverson(arg:DoubleTerm) extends UnaryTerm[DoubleTerm, Dom.bools.type] {
  override def copy(args: IndexedSeq[DoubleTerm]) = InverseLogIverson(args(0))

  override val domain: Dom.bools.type = Dom.bools

  override def composer(args: Settings) = new Composer(args) {
    def eval()(implicit execution: Execution) = {
      output.disc(0) = if (input(0).cont(0) == Double.NegativeInfinity) 0 else 1
    }
  }

}


case class DiscreteEquals[T](arg1: Term[GenericDiscreteDom[T]], arg2: Term[GenericDiscreteDom[T]]) extends BinaryDiscreteOperator[BoolDom, GenericDiscreteDom[T]] {
  val domain = Dom.bools

  def op(a1: Int, a2: Int) = if (a1 == a2) 1 else 0

  def name = "==="
}

case class And(arg1: BoolTerm, arg2: BoolTerm) extends BinaryDiscreteOperator[BoolDom, BoolDom] {
  def op(a1: Int, a2: Int) = if (a1 != 0 && a2 != 0) 1 else 0

  val domain = Dom.bools

  def name = "&&"

}

case class Or(arg1: BoolTerm, arg2: BoolTerm) extends BinaryDiscreteOperator[BoolDom, BoolDom] {
  def op(a1: Int, a2: Int) = if (a1 != 0 || a2 != 0) 1 else 0

  val domain = Dom.bools

  def name = "||"

}

case class Implies(arg1: BoolTerm, arg2: BoolTerm) extends BinaryDiscreteOperator[BoolDom, BoolDom] {
  def op(a1: Int, a2: Int) = if (a1 == 0 || a2 != 0) 1 else 0

  val domain = Dom.bools

  def name = "-->"

}



