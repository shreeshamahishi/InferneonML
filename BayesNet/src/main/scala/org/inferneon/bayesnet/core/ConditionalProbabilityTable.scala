package org.inferneon.bayesnet.core

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