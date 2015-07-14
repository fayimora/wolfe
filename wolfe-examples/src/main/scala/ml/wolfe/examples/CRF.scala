package ml.wolfe.examples

import ml.wolfe.term.LearningObjective._
import ml.wolfe.term.TermImplicits._
import ml.wolfe.term._
import ml.wolfe._

import scala.util.Random

/**
 * Created by fayimora on 10/07/15.
 */

object CRF extends App {

  implicit val random = new Random(0)

  object Data {

    def readWsjPos(filename: String): IndexedSeq[(IndexedSeq[String], IndexedSeq[String])] = {
      import scala.collection.mutable.ArrayBuffer

      val lines  = io.Source.fromFile(filename).getLines().map(_.split(" "))
      val result = ArrayBuffer[(ArrayBuffer[String], ArrayBuffer[String])]()
      var tokens = ArrayBuffer[String]()
      var labels = ArrayBuffer[String]()

      for(line <- lines) {
        if (line.size <= 1) { // end of sentence
          result += ((tokens, labels))
          tokens = ArrayBuffer[String]()
          labels = ArrayBuffer[String]()
        } else {
          tokens += line(0)
          labels += line(1)
        }
      }
      result
    }

    // dev=5527, train=38219, test=5462
//    val data = readWsjPos("/Users/fayimora/Downloads/wsj-pos/dev.txt")
    val train = readWsjPos("/Users/fayimora/Downloads/wsj-pos/train.txt").take(1000)
    val test = readWsjPos("/Users/fayimora/Downloads/wsj-pos/dev.txt")

    val words = (train ++ test).flatMap(_._1).distinct
    val tags = (train ++ test).flatMap(_._2).distinct
    val maxLength = (train ++ test).map(_._1.length).max

    println(s"train: ${train.length}, test: ${test.length}")
    println(s"words: ${words.length}, tags: ${tags.length}, maxLength: $maxLength")

    // number of labels
    val L = tags.length
    // number of unary features
    val N = words.length

    // data domains
    implicit val Words = words.toDom withOOV "[OOV]"
    implicit val Tags = tags.toDom
    implicit val Y = Seqs(Tags, 0, maxLength)
    implicit val X = Seqs(Words, 0, maxLength)
    implicit val Instances = Pairs(X, Y)

    val numFeats = (N+1) * L + L
    implicit val Features = Vectors(numFeats)
    println("numFeats: " + numFeats)
    val TransitionFeats = Vectors(L * L)
  }

  trait Model {
    def predict: IndexedSeq[String] => IndexedSeq[String]
  }

  object CRFModel extends Model {
    import Data._

    // model domains
    @domain case class Theta(w: Vect, wb: Vect)

    implicit val Thetas = Theta.Values(Features, TransitionFeats)
    implicit val maxProductParams = BPParameters(iterations = 1, cachePotentials = true)

    val localIndex = new SimpleFeatureIndex(Features)
    val transIndex = new SimpleFeatureIndex(TransitionFeats)

    def local(x: X.Term, y: Y.Term, i: IntTerm) = {
      localIndex.oneHot('bias, y(i)) + localIndex.oneHot('word, y(i), x(i))
    }

    def transition(x: X.Term, y: Y.Term, i: IntTerm) = {
      transIndex.oneHot('pair, y(i), y(i + 1))
    }

    def model(w: Thetas.Term)(x: X.Term)(y: Y.Term) = {
      sum(0 until x.length) { i => w.w dot local(x, y, i) } +
        sum(0 until x.length - 1) { i => w.wb dot transition(x, y, i) }
    } subjectTo (y.length === x.length) argmaxBy Argmaxer.maxProduct

    val init = Settings(Thetas.createZeroSetting())
    // epochs, learningRate, delta
    val params = AdaGradParameters(10, 0.1, 0.1, initParams = init)

    lazy val thetaStar =
      learn(Thetas)(t => perceptron(train.toConst)(Y)(model(t))) using Argmaxer.adaGrad(params)

    val predict = fun(X) { x => argmax(Y)(model(Thetas.Const(thetaStar))(x)) }
  }

  object NeuralModel extends Model {

    import Data._

    val localIndex = new SimpleFeatureIndex(Features)
    val transIndex = new SimpleFeatureIndex(TransitionFeats)

    val ak = 3
    val bk = 3

    // model domains
    @domain case class Theta(a: Mat, b: Mat, w: Vect, wb: Vect)

    implicit val Thetas = Theta.Values(Matrices(ak, bk), Matrices(bk, numFeats), Vectors(ak), TransitionFeats)
    implicit val maxProductParams = BPParameters(iterations = 1)

    def local(x: X.Term, y: Y.Term, i: IntTerm) = {
      localIndex.oneHot('bias, y(i)) + localIndex.oneHot('word, y(i), x(i))
    }

    def transition(x: X.Term, y: Y.Term, i: IntTerm) = {
      transIndex.oneHot('pair, y(i), y(i + 1))
    }

    def model(w: Thetas.Term)(x: X.Term)(y: Y.Term) = {
      sum(0 until x.length) { i => w.w dot (w.a * sigmVec(w.b * local(x,y,i))) } +
       sum(0 until x.length - 1) { i => w.wb dot transition(x,y,i) }
    } subjectTo (y.length === x.length) argmaxBy Argmaxer.maxProduct

    val init = Settings(Thetas.createRandomSetting(random.nextGaussian() * 0.1))
//    val init = Settings(Thetas.createZeroSetting())
    val params = AdaGradParameters(10, 0.1, 0.1, initParams = init) // epochs, alpha, delta

    lazy val thetaStar =
      learn(Thetas)(t => perceptron(train.toConst)(Y)(model(t))) using Argmaxer.adaGrad(params)

    val predict = fun(X) { x => argmax(Y)(model(Thetas.Const(thetaStar))(x)) }
  }

  def time(block: => Double) = {
    import java.util.concurrent.TimeUnit

    val t0 = System.currentTimeMillis()
    val result = block    // call-by-name
    val t1 = System.currentTimeMillis()
    val millis = t1 - t0
    val mins = TimeUnit.MILLISECONDS.toMinutes(millis)
    val secs = TimeUnit.MILLISECONDS.toSeconds(millis) - TimeUnit.MINUTES.toSeconds(TimeUnit.MILLISECONDS.toMinutes(millis))

    println(s"Elapsed time: ${mins} mins, ${secs} secs")
    result
  }

  def run(m: Model, test: Seq[(IndexedSeq[String], IndexedSeq[String])]): Double = {
    var errs = 0.0
    var total = 0.0
    for ((x, y) <- test) {
      val yh = m.predict(x)
      assert(yh.length == y.length)
      for (i <- 0 until y.length) {
        if (yh(i) != y(i)) errs += 1.0
        total += 1.0
      }
    }
    errs / total
  }

//  val linErr = time { run(CRFModel, Data.test) }
//  println(s"===== linErr: ${linErr*100}% error =====")

  val neuralErr = time { run(NeuralModel, Data.test) }
  println(s"===== neuralErr: ${neuralErr*100}% error =====")

}
