package org.inferneon.bayesnet.core

/**
  * Edge in a Bayesian belief network.
  */
case class Edge(var source: Node, var target : Node)
