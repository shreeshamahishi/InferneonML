package org.inferneon.bayesnet.core

import scala.collection.mutable.ArrayBuffer

/**
  * A combination of values across different features. It is defined by a collection of tuples, each tuple containg the
  * feature index and the corresponding value of that feature and the value of the dependent feature.
  */
case class ValuesCombination(featureIndexAndValue: ArrayBuffer[(Int, Int)], valueOfDependentFeature: Int) extends Serializable{
  def valueOfDependentForFeatureCombination(featuresCombination: FeaturesCombination): Option[Int] ={
    if(featuresCombination.featureIndexAndValue.toSet == featureIndexAndValue.toSet){
      Some(valueOfDependentFeature)
    }
    else{
      None
    }
  }

  def getFeaturesCombination() : FeaturesCombination = {
    new FeaturesCombination(featureIndexAndValue)
  }
}

/**
  * Similar to ValuesCombination, but without a reference to the dependent feature.
  */
case class FeaturesCombination(featureIndexAndValue: ArrayBuffer[(Int, Int)]) extends Serializable