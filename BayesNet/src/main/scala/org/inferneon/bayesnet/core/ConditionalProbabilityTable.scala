package org.inferneon.bayesnet.core

import java.util
import scala.collection.JavaConverters._
import scala.collection.mutable.ArrayBuffer

/**
  * Represents a conditional probability distribution table corresponding a node (feature) in a Bayesian belief network.
  */
case class ConditionalProbabilityTable (id: Int, conditionalProbabilites: Option[Map[ArrayBuffer[(Int, Int)], Array[Double]]],
                                        priorProbabilites: Option[Array[Double]]) extends Serializable {

  val isConditional = if(conditionalProbabilites.isDefined) true else false

  def description(schema : Array[(String, Array[String])]): String ={
    if(isConditional){
      val cps = conditionalProbabilites.get
      val strs = cps map { cp =>
        val ftrAndValIndices = cp._1
        val cpstr = ftrAndValIndices map { case (fIndex, valueIndex) =>
          schema(fIndex)._1 + ":" + schema(fIndex)._2(valueIndex) + " "
        }
        cpstr.mkString("") + " => " + cp._2.mkString(", ")
      }
      strs.mkString(System.getProperty("line.separator") + "  ")
    }
    else{
      priorProbabilites.get mkString ", "
    }
  }
}

class JavaCPT(val id: Int,
              val conditionalProbabilites: java.util.Map[java.util.ArrayList[java.util.Map[Int, Int]], java.util.List[Double]],
              val isConditional: Boolean) {

}

object ConditionalProbabilityTable {
  def getJavaCPT(cpt : ConditionalProbabilityTable) : JavaCPT = {

    if(cpt.conditionalProbabilites.isDefined){
      val javaConditionalCPT = new java.util.HashMap[java.util.ArrayList[java.util.Map[Int, Int]], java.util.List[Double]]()
      cpt.conditionalProbabilites.get foreach {case (combos, probabilities) => {
        val javaCombos = new java.util.ArrayList[java.util.Map[Int, Int]]()
        combos foreach {case (fIndex, valueIndex) =>
                      val fIndexValueMap = new java.util.HashMap[Int, Int]()
                      fIndexValueMap.put(fIndex, valueIndex)
                      javaCombos.add(fIndexValueMap)
        }
        val probabilitiesList : java.util.List[Double] = probabilities.toList.asJava
        javaConditionalCPT.put(javaCombos, probabilitiesList)
      }
      }
      new JavaCPT(cpt.id, javaConditionalCPT, true)
    }
    else{
      val prs : java.util.List[Double] = cpt.priorProbabilites.get.toList.asJava
      val javaConditionalCPT = new java.util.HashMap[java.util.ArrayList[java.util.Map[Int, Int]], java.util.List[Double]]()
      javaConditionalCPT.put(new java.util.ArrayList[java.util.Map[Int, Int]](), prs)
      new JavaCPT(cpt.id, javaConditionalCPT, false)
    }
  }
}
