package org.inferneon.bayesnet.core

import org.scalatest.FunSuite

/**
  * Tests for network creation.
  */
class NetworkSuite extends FunSuite {

  test("Check child nodes"){
    val nodes = (1 to 10) map (id => Node.emptyNode(id))
    val network = new BayesianBeliefNetwork(nodes.toList)
    val edgesInfo = List(1 -> 2, 1 -> 3, 2 -> 4, 2 -> 5, 5 -> 6, 4 -> 7, 4 -> 8, 4 -> 9, 7 -> 10)
    edgesInfo foreach{case (src, target) =>
      network.addEdge(Node.emptyNode(src), Node.emptyNode(target))
    }

    val childNodesOfRoot = network.getChildNodes(nodes.head)
    assert(childNodesOfRoot.contains(Node.emptyNode(2)))
    assert(childNodesOfRoot.contains(Node.emptyNode(3)))

    val childrenOf4 = network.getChildNodes(Node.emptyNode(4))
    assert(childrenOf4.contains(Node.emptyNode(7)))
    assert(childrenOf4.contains(Node.emptyNode(8)))
    assert(childrenOf4.contains(Node.emptyNode(9)))
  }

  test("Adding edges, some create cycles") {
    val nodes = (1 to 10) map (id => Node.emptyNode(id))
    val network = new BayesianBeliefNetwork(nodes.toList)
    val edgesInfo = List(1 -> 2, 1 -> 3, 2 -> 4, 2 -> 5, 5 -> 6, 4 -> 7, 4 -> 8, 4 -> 9, 7 -> 10)
    edgesInfo foreach{case (src, target) =>
        network.addEdge(Node.emptyNode(src), Node.emptyNode(target))
    }

    val flag1 = network.addingEdgeCreatesCycle(Node.emptyNode(2), Node.emptyNode(3))
    assert(!flag1)

    val flag2 = network.addingEdgeCreatesCycle(Node.emptyNode(8), Node.emptyNode(1))
    assert(flag2)

    val flag3 = network.addingEdgeCreatesCycle(Node.emptyNode(10), Node.emptyNode(1))
    assert(flag3)

    val flag4 = network.addingEdgeCreatesCycle(Node.emptyNode(6), Node.emptyNode(2))
    assert(flag4)

    val flag5 = network.addingEdgeCreatesCycle(Node.emptyNode(3), Node.emptyNode(1))
    assert(flag5)

    val flag6 = network.addingEdgeCreatesCycle(Node.emptyNode(8), Node.emptyNode(2))
    assert(flag6)

    val flag7 = network.addingEdgeCreatesCycle(Node.emptyNode(2), Node.emptyNode(10))
    assert(!flag7)

  }
}
