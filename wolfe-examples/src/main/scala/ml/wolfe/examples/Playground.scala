package ml.wolfe.examples


class Playground {
  def readWsjPos(filename: String): IndexedSeq[(IndexedSeq[String], IndexedSeq[String])] = {
    import scala.collection.mutable.ArrayBuffer

    val lines = io.Source.fromFile(filename).getLines().map(_.split(" "))
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
}
