package org.inferneon.bayesnet.core

/**
  * Node in a Bayesian belief network.
  */
case class Node(val id: Int, val isLabel : Boolean) extends Serializable

object Node {
  def emptyNode(id:Int, isLabel: Boolean = false): Node = {
    new Node(id, isLabel)
  }
}