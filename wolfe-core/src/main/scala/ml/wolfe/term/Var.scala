package ml.wolfe.term

/**
 * @author riedel
 */
trait Var[+D <: Dom] extends Term[D] {
  self =>
  def name: String

  def vars = Seq(this)

  def isStatic = false

  override def evaluatorImpl(in: Settings) = new Evaluator {
    def eval()(implicit execution: Execution) = {}

    val input = in
    val output = in(0)
  }


  override def differentiatorImpl(wrt: Seq[Var[Dom]])(in: Settings, err: Setting, gradientAcc: Settings) =
    new AbstractDifferentiator(in, err, gradientAcc, wrt) {
      val output = in(0)
      val isWrt = wrt.contains(self)

      def forward()(implicit execution: Execution) {}

      def backward()(implicit execution: Execution): Unit = {
        if (isWrt) gradientAccumulator(0).addIfChanged(error)
      }
    }

  override def toString = name
}

trait Atom[+D <: Dom] extends Var[D] {

  self =>

  trait Grounder {
    def ground()(implicit execution: Execution): GroundAtom[D]
  }

  def varsToGround: Seq[AnyVar]

  def grounder(settings: Settings): Grounder

  def owner: AnyVar

}

trait GroundAtom[+D <: Dom] extends Atom[D] {
  self =>
  def offsets: Offsets

  def varsToGround = Nil

  def grounder(settings: Settings) = new Grounder {
    def ground()(implicit execution: Execution) = self
  }
}

case class VarAtom[D <: Dom](variable: Var[D]) extends GroundAtom[D] {

  def offsets = Offsets.zero

  val domain = variable.domain

  def owner = variable

  def name = variable.name
}

//case class _1Atom[D <: Dom](parent:Atom[Tuple2Dom[D,_]]) extends Atom[D] {
//  val domain = parent.domain.dom1
//
//}
//case class _2Atom[D <: Dom](parent:Atom[Tuple2Dom[_,D]]) extends Atom[D] {
//  val domain = parent.domain.dom2
//
//}

case class SeqGroundAtom[E <: Dom, S <: VarSeqDom[E]](seq: GroundAtom[S], index: Int) extends GroundAtom[E] {
  val domain = seq.domain.elementDom

  val offsets = seq.offsets + domain.lengths * index

  def owner = seq.owner

  def name = toString
}

case class SeqAtom[E <: Dom, S <: VarSeqDom[E]](seq: Atom[S], index: IntTerm) extends Atom[E] {

  val domain = seq.domain.elementDom
  val varsToGround = (seq.varsToGround ++ index.vars).distinct

  def grounder(settings: Settings) = {
    new Grounder {
      val seqGrounder = seq.grounder(settings.linkedSettings(varsToGround, seq.varsToGround))
      val indexEval = index.evaluatorImpl(settings.linkedSettings(varsToGround, index.vars))

      def ground()(implicit execution: Execution) = {
        val parent = seqGrounder.ground()
        indexEval.eval()
        val groundIndex = indexEval.output.disc(0)
        SeqGroundAtom[E,S](parent,groundIndex)
      }
    }
  }
  def name = toString

  def owner = seq.owner
}








