package ml.wolfe.ui

import java.io.{FileWriter, File}

import org.json4s.NoTypeHints
import org.json4s.jackson.Serialization

import scala.collection.mutable.ArrayBuffer

/**
 * @author Sebastian Riedel
 */
class MutableMoroNotebook extends MutableNotebook[MutableMoroNotebook] {
  val cells = new ArrayBuffer[Cell]()

  object OutputFormats extends Enumeration {
    val string, html, javascript = Value
  }

  case class Input(code: String, outputFormat: String = OutputFormats.html.toString, extraFields: Map[String, String] = Map.empty)

  case class Cell(id: Int, compiler: String, input: Input)

  case class DocumentData(name: String, cells: Seq[Cell])

  def stripLeadingWhitespace(text: String) = text.split("\n").map(_.trim).mkString("\n")

  def html(text: String) = {
    cells += Cell(cells.size, "html", Input(text, OutputFormats.html.toString))
    this
  }

  def h2(text: String) = {
    cells += Cell(cells.size, "heading2", Input(text, OutputFormats.html.toString))
    this
  }

  def trimEmptyLines(text: String) = {
    val lines = text.split("\n").toList
    val prefixRemoved = lines.dropWhile(_.trim == "")
    val postfixRemoved = prefixRemoved.reverse.dropWhile(_.trim == "").reverse
    postfixRemoved.mkString("\n")
  }

  def source(src: String) = {
    cells += Cell(cells.size, "scala", Input(trimEmptyLines(src.stripMargin), OutputFormats.string.toString))
    this
  }

  def wolfe(src: String) = {
    cells += Cell(cells.size, "wolfe", Input(trimEmptyLines(src.stripMargin), OutputFormats.string.toString))
    this
  }

  def wolfeNoEval(src: String) = {
    cells += Cell(cells.size, "wolfeNoEval", Input(trimEmptyLines(src.stripMargin), OutputFormats.string.toString))
    this
  }

  def section(name: String)(body: => MutableMoroNotebook = this) = {
    cells += Cell(cells.size, "section", Input(name, OutputFormats.html.toString))
    body
  }

  def frame = section _

  def h1(text: String) = {
    cells += Cell(cells.size, "heading1", Input(text, OutputFormats.html.toString))
    this
  }

  private def escape(code: String) = code
    .replaceAll("</script>", "\\\\x3C/script>")
    .replaceAll("'", "\\\"")

  def raw(code: String) = {
    cells += Cell(cells.size, "raw", Input(escape(code)))
    this
  }

  def rawOutside(code: String) = {
    cells += Cell(cells.size, "rawOutside", Input(escape(code)))
    this
  }

  def md(markdown: String) = {
    cells += Cell(cells.size, "markdown", Input(trimEmptyLines(markdown.stripMargin), OutputFormats.html.toString))
    this
  }

  def latex(text: String) = {
    cells += Cell(cells.size, "latex", Input(text, OutputFormats.html.toString))
    this
  }

  def pdflatex(text: String) = {
    cells += Cell(cells.size, "pdflatex", Input(text, OutputFormats.string.toString, Map("png" -> "true")))
    //cells += Cell(cells.size, "pdflatex", Input(text, OutputFormats.string.toString))
    this
  }

  def notes(text: String) = {
    cells += Cell(cells.size, "html", Input("<aside class=\"notes\">" + text + "</aside>", OutputFormats.html.toString))
    this
  }

  //todo: speed up
  //fixme: doesn't work for raw cells
  def config(configs: Map[String, String])(body: => MutableMoroNotebook = this): MutableMoroNotebook = {
    val before = this.cells.size
    val after = body.cells.size

    val affected = this.cells.slice(before, after)
    cells.remove(before, after - before)

    for {
      cell <- affected
    } {
      val input = cell.input
      val newConfig = input.extraFields ++ configs
      val newInput = input.copy(extraFields = newConfig).asInstanceOf[this.Input]
      cells += Cell(cell.id, cell.compiler, newInput)
    }

    this
  }

  def fragment(body: => MutableMoroNotebook = this) = config(Map("fragment" -> "true")) {
    body
  }

  def saveTo(name: String, file: File): Unit = {
    implicit val formats = Serialization.formats(NoTypeHints)
    val ser = Serialization.writePretty(DocumentData(name, cells))
    val fw = new FileWriter(file)
    fw.write(ser)
    fw.close()

  }
}
