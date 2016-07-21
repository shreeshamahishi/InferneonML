package org.inferneon.bayesnet.hillclimber

import org.inferneon.bayesnet.core.Node

/**
  * Types of scoring techniques.
  */
object ScoringType extends Enumeration {
  val BAYES, BDeu, MDL, ENTROPY, AIC = Value
}

/**
  * Types of operations.
  */
object OperationType extends Enumeration {
  val Add, Delete, Reverse = Value
}

/**
  * Denotes an operation that can be performed on the target node.
  * @param source                          The source node
  * @param target                          The target node
  * @param operationType                   The type of operation (add or delete)
  */
case class Operation(val source: Node, val target: Node, val operationType: OperationType.Value) {

  def getProposedParents(currentParents : Option[List[Node]]) : List[Node] = {
    val newList = if(operationType == OperationType.Add){
      if(currentParents.isDefined){
        currentParents.get.::(source)
      }
      else{
        source :: Nil
      }
    }
    else{
      if(currentParents.isDefined){
        currentParents.get diff List(source)
      }
      else{
        // should not happen
        List.empty
      }
    }
    newList
  }
}

/**
  * A cache of change in scores for every possible operation on every node in the DAG. This is used later to confirm
  * get the optimal operation given the structure of the network.
  *
  * @param numFeatures    The total number of features or nodes in the DAG.
  */
class ScoreDeltasCache(val numFeatures: Int) extends Serializable{

  private val addEdgeDeltas = Array.fill(numFeatures)(Array.fill(numFeatures)(0.0))
  private val deleteEdgeDeltas = Array.fill(numFeatures)(Array.fill(numFeatures)(0.0))

  def put(operation: Operation, delta: Double) {
    if (operation.operationType == OperationType.Add) {
      addEdgeDeltas(operation.source.id)(operation.target.id) = delta;
    }
    else {
      deleteEdgeDeltas(operation.source.id)(operation.target.id) = delta;
    }
  }

  def get(op: Operation) : Double = {
    if(op.operationType == OperationType.Add){
      addEdgeDeltas(op.source.id)(op.target.id)
    }
    else{
      deleteEdgeDeltas(op.source.id)(op.target.id)
    }
  }

  def printCache(): Unit = {
    println("Add edge scores: ")
    Range(0,numFeatures) foreach{fIndex1 => print(fIndex1)
            Range(0, numFeatures) foreach {fIndex2 =>
            print (" to " + fIndex2 + " = " + addEdgeDeltas(fIndex1)(fIndex2))} ; println("")}
    println("Delete edge scores: ")
    Range(0,numFeatures) foreach{fIndex1 => print(fIndex1)
      Range(0, numFeatures) foreach {fIndex2 =>
        print (" from " + fIndex2 + " = " + deleteEdgeDeltas(fIndex1)(fIndex2))} ; println("")}
  }
}
