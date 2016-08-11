package org.inferneon.bayesnet.core

import java.util

import org.inferneon.bayesnet.DataUtils

import scala.collection.mutable
import scala.collection.JavaConverters._
import scala.language.postfixOps
/**
  * Represent the DAG (directed acyclic graph) of the Bayesian belief network. It consists of a collection of nodes,
  * each node denoting a feature, a collection of directed edges between nodes in the graph and conditional probability
  * tables for each node in the graph.
  */

class BayesianBeliefNetwork(var allNodes: List[Node]) extends Serializable {
  var test = 0.0
  var edges: collection.mutable.Set[Edge] = new collection.mutable.HashSet[Edge]
  var cpts: collection.mutable.Map[Int, ConditionalProbabilityTable] = new collection.mutable.HashMap[Int, ConditionalProbabilityTable]

  def add(node: Node): Boolean = {
    if(allNodes.contains(node)) false
    else {
      allNodes :+ node
      true
    }
  }

  def addEdge(source: Node, target: Node): Boolean ={
    require(allNodes.contains(source), "The graph does not contain " + source)
    require(allNodes.contains(target), "The graph does not contain " + target)
    require(!addingEdgeCreatesCycle(source, target), "Addding an edge between " + source + " and " + target + " creates a cycle in the graph.")

    if(!edgeExists(source, target)) {
      edges += Edge(source, target)
      true
    }
    else{
      false
    }
  }

  def edgeExists(source: Node, target: Node): Boolean = {
    require(allNodes.contains(source), "The graph does not contain " + source)
    require(allNodes.contains(target), "The graph does not contain " + target)
    val edge = Edge(source, target)
    if(edges.contains(edge)){ true } else{ false }
  }

  def getChildNodes(source:Node): List[Node] = {
    edges.filter(edge => edge.source == source).map(_.target).toList
  }

  /**
    * Implements a simple depth first search (DFS) satisfying the condition that this network represents a DAG (directed acyclic
    * graph). The DFS search simply traverses all nodes reachable from the target and checks if the source node is visited along the
    * way. If it encounters the source node during the search, it exits with a flag set to true, else the traversal will complete
    * without having found the source node and then we know a cycle will not be created by adding the proposed edge.
    *
    * @param source The source node of the proposed edge.
    * @param target The target node of the proposed edge.
    * @return A flag indicating that the proposed edge introduces an cycle (or not).
    */
  def addingEdgeCreatesCycle(source: Node, target: Node) : Boolean = {
    def continueLoop(stack: mutable.Stack[Node], foundSource: Boolean) : Boolean = {
      if(stack.isEmpty){
        false
      }
      else{
        if(foundSource){
          false
        }
        else {
          true
        }
      }
    }

    require(allNodes.contains(source), "The graph does not contain " + source)
    require(allNodes.contains(target), "The graph does not contain " + target)
    val visitInfo = new mutable.HashMap[Node, Boolean]()
    var foundSource = false
    allNodes foreach {node => visitInfo(node) = false}
    val stack = new mutable.Stack[Node]()
    stack.push(target)
    while(continueLoop(stack, foundSource)){
      val currentNode = stack.pop()
      if(currentNode == source){
        foundSource = true
      }
      else{
        if(!visitInfo(currentNode)){
          visitInfo(currentNode) = true
          val childNodes = getChildNodes(currentNode).toArray
          val numChildNodes = childNodes.length
          var index = 0
          while(index < numChildNodes && !foundSource){
            val cn = childNodes(index)
            if(cn == source){
              foundSource = true
            }
            else {
              if (!visitInfo(cn)) {
                stack.push(cn)
              }
            }
            index = index + 1
          }
        }
      }
    }
    foundSource
  }

  def getParents(node: Node) : Option[List[Node]] = {
    require(allNodes.contains(node), "The graph does not contain node " + node)
    val sources = edges filter  {_.target == node} map {_.source}
    if(sources.isEmpty){  None }  else{ Some(sources.toList) }
  }

  def removeEdge(source: Node, target: Node): Boolean = {
    require(allNodes.contains(source), "The graph does not contain " + source)
    require(allNodes.contains(target), "The graph does not contain " + target)
    val edge = Edge(source, target)
    if(!edges.contains(edge)){ false } else{ edges.remove(edge) }
  }

  def rootNodes() : Set[Node] = {
    val rootNodes =  allNodes.toSet diff(edges map {edge => edge.target} toSet )
    rootNodes
  }

  def treeDescription(schema: Array[(String, Array[String])]): String = {
    def populateTreeDesc(nodes: List[Node], level: Int, sbuf: StringBuffer): Unit = {
      nodes foreach { node =>
        val cNodes = getChildNodes(node)
        if(cNodes.nonEmpty){
          cNodes foreach{ target =>
            sbuf.append(System.getProperty("line.separator"))
            Range(0, level).foreach(times => sbuf.append("|   "))
            sbuf.append(schema(node.id)._1)
            if(getChildNodes(target).isEmpty) {
              sbuf.append(": ")
            }
            populateTreeDesc(List[Node](target), level + 1, sbuf)
          }
        }
        else{
          sbuf.append(schema(node.id)._1)
        }
      }
    }

    val sbuf = new StringBuffer
    populateTreeDesc(rootNodes().toList, 0, sbuf)

    sbuf.append(System.getProperty("line.separator"))
    sbuf.append("CONDITIONAL PROBABILITY DISTRIBUTION TABLES:")
    sbuf.append(System.getProperty("line.separator"))

    cpts foreach {table =>
      sbuf.append("Table for " + schema(table._1)._1 + ":")
      sbuf.append(System.getProperty("line.separator"))
      sbuf.append(" " + table._2.description(schema))
      sbuf.append(System.getProperty("line.separator"))
    }
    sbuf.toString
  }

  def treeDescription(format: java.util.List[java.util.Map[String, java.util.List[String]]]): String = {
    treeDescription(DataUtils.schemaFromJava(format))
  }

}

class JavaBayesianBeliefNetwork(val nodes : java.util.List[Node], val edges: java.util.Set[Edge], val cpts: java.util.Map[Int, JavaCPT])

object BayesianBeliefNetwork {
  def emptyNetwork() : BayesianBeliefNetwork = new BayesianBeliefNetwork(List.empty)

  def getJavaBayesianBeliefNetwork(network : BayesianBeliefNetwork) : JavaBayesianBeliefNetwork = {
    val nodes : java.util.List[Node] =  network.allNodes.asJava
    val edges : java.util.Set[Edge] = network.edges.asJava
    val cpts = new java.util.HashMap[Int, JavaCPT]()

    network.cpts foreach { case (id, cpt ) =>
      cpts.put(id, ConditionalProbabilityTable.getJavaCPT(cpt))
    }

    new JavaBayesianBeliefNetwork(nodes, edges, cpts)
  }

}
