package ml.wolfe.macros

import scala.reflect.macros.Context
import scala.reflect.api.Universe
import scala.collection.mutable

/**
 * Represents code that generates structures for a given sample space.
 *
 * @author Sebastian Riedel
 */
trait MetaStructure extends HasUniverse {

  import universe._

  /**
   * @return The name of the structure class.
   */
  def className: TypeName

  /**
   * @return A list of definitions of domain values needed in the structure generation code.
   */
  def domainDefs: List[ValDef]

  /**
   * Creates the code the defines the structure class.
   * @param graphName the structure class needs an underlying factor graph, and this parameter provides its name.
   *                  This means that when instantiating the structure class a factor graph with the given name needs
   *                  to be in scope.
   * @return code that defines the structure class.
   */
  def classDef(graphName: TermName): ClassDef

  /**
   * @return the type of objects this structure represents (i.e. the type of objects in the sample space).
   */
  def argType: Type

  /**
   * @return any meta structure used within this meta structure.
   */
  def children: List[MetaStructure]

  /**
   * @return this and all descendant metastructures.
   */
  def all: List[MetaStructure] = this :: children.flatMap(_.all)

  /**
   * A matcher takes a tree and returns a tree that represents the sub-structure that corresponds to the
   * tree, if any. This method creates matchers for the structures of this meta-structure.
   * @param parent a matcher for the parent structure.
   * @param result the current result matcher.
   * @return a matcher...
   */
  def matcher(parent: Tree => Option[Tree], result: Tree => Option[Tree]): Tree => Option[Tree]

  def injectStructure(tree: Tree, root: Tree => Option[Tree]) = {
    val matcher = this.matcher(root, root)
    val transformer = new Transformer {
      val functionStack = new mutable.Stack[Function]()
      override def transform(tree: Tree) = {
        tree match {
          case f: Function => functionStack.push(f)
          case _ =>
        }
        val result = matcher(tree) match {
          case Some(structure) => {
            //get symbols in tree
            val symbols = tree.collect({case i: Ident => i}).map(_.name).toSet //todo: this shouldn't just be by name
            val hasFunctionArg = functionStack.exists(_.vparams.exists(p => symbols(p.name)))
            if (hasFunctionArg)
              super.transform(tree)
            else
              q"$structure.value()"
          }
          case _ => super.transform(tree)
        }
        tree match {
          case _: Function => functionStack.pop
          case _ =>
        }
        result
      }
    }
    transformer transform tree
  }


}


object MetaStructure {

  import scala.language.experimental.macros

  /**
   * Creates a meta structure for the given code repository and sample space.
   * @param repo the code repository to use when inlining.
   * @param sampleSpace the sample space to create a meta structure for.
   * @return the meta structure for the given sample space.
   */
  def apply(repo: CodeRepository)(sampleSpace: repo.universe.Tree): MetaStructure {type U = repo.universe.type} = {
    //todo: assert that domain is an iterable
    import repo.universe._
    //get symbol for all, unwrap ...
    val symbols = WolfeSymbols(repo.universe)
    sampleSpace match {
      case q"$all[..${_}]($unwrap[..${_}]($constructor))($cross(..$sets))"
        if all.symbol == symbols.all && symbols.unwraps(unwrap.symbol) && symbols.crosses(cross.symbol) =>
        val t: Type = constructor.tpe
        val scope = t.declarations
        val applySymbol = t.member(newTermName("apply")).asMethod
        val args = applySymbol.paramss.head
        new MetaCaseClassStructure {
          type U = repo.universe.type
          val universe: repo.universe.type = repo.universe
          val tpe                          = constructor.tpe
          val fieldStructures              = sets.map(apply(repo)(_))
          val fields                       = args
          def repository = repo
        }
      case _ =>
        repo.inlineOnce(sampleSpace) match {
          case Some(inlined) => apply(repo)(inlined)
          case None =>
            new MetaAtomicStructure {
              type U = repo.universe.type
              val universe: repo.universe.type = repo.universe
              def repository = repo
              def domain = sampleSpace
            }
        }
    }
  }

  /**
   * Useful for unit tests.
   * @param sampleSpace the sample space to inspect.
   * @tparam T type of objects in sample space.
   * @return a structure corresponding to the given sample space.
   */
  def createStructure[T](sampleSpace: Iterable[T]): Structure[T] = macro createStructureImpl[T]

  def createStructureAndProjection[T1, T2](sampleSpace: Iterable[T1],
                                           projection: T1 => T2): (Structure[T1], Structure[T1] => T2) = macro createStructureAndProjectionImpl[T1, T2]

  def createStructureAndProjectionImpl[T1: c.WeakTypeTag, T2: c.WeakTypeTag](c: Context)
                                                                            (sampleSpace: c.Expr[Iterable[T1]],
                                                                             projection: c.Expr[T1 => T2]) = {
    import c.universe._
    val repo = CodeRepository.fromContext(c)
    val meta = MetaStructure(repo)(sampleSpace.tree)
    val graphName = newTermName("_graph")
    val structName = newTermName("structure")
    val structArgName = newTermName("structArg")
    val cls = meta.classDef(graphName)
    val q"($arg) => $rhs" = projection.tree
    val root = (tree: Tree) => tree match {
      case i: Ident if i.symbol == arg.symbol => Some(q"$structArgName.asInstanceOf[${meta.className}]")
      case _ => None
    }
    val injectedRhs = meta.injectStructure(rhs, root)

    val injectedProj = q"($structArgName:ml.wolfe.macros.Structure[${meta.argType.typeSymbol.name.toTypeName}]) => $injectedRhs"
    val code = q"""
      val $graphName = new ml.wolfe.MPGraph
      $cls
      val $structName = new ${meta.className}
      $graphName.setupNodes()
      ($structName,$injectedProj)
    """
    c.Expr[(Structure[T1], Structure[T1] => T2)](code)
//    println(code)
//    c.Expr[(Structure[T1], Structure[T1] => T2)](q"(null,???)")
  }


  def createStructureImpl[T: c.WeakTypeTag](c: Context)(sampleSpace: c.Expr[Iterable[T]]): c.Expr[Structure[T]] = {
    import c.universe._
    val repo = CodeRepository.fromContext(c)
    val meta = MetaStructure(repo)(sampleSpace.tree)
    val graphName = newTermName("_graph")
    val cls = meta.classDef(graphName)
    val code = q"""
      val $graphName = new ml.wolfe.MPGraph
      $cls
      val structure = new ${meta.className}
      $graphName.setupNodes()
      structure
    """
    c.Expr[Structure[T]](code)
  }

}
