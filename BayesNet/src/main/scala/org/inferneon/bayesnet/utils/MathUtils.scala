package org.inferneon.bayesnet.utils

/**
  * Utility and helper function for math functions.
  */
object MathUtils {
  def approximatelyEqual(x: Double, y: Double, precision: Double) = {
    if ((x - y).abs < precision) true else false
  }
}
