package org.inferneon.bayesnet.hillclimber

import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.rdd.RDD
import org.apache.spark.{SparkConf, SparkContext}
import org.inferneon.bayesnet.core.Node
import org.inferneon.datautils.DataUtils
import org.scalatest.{BeforeAndAfterAll, FunSuite}

/**
  * Tests for hill climber algorithm.
  */
class HillClimberSuite extends FunSuite with BeforeAndAfterAll {

  @transient var sc: SparkContext = _

  override def beforeAll() {
    super.beforeAll()
    val conf = new SparkConf()
      .setMaster("local[2]")
      .setAppName("HillClimberUnitTest")
    sc = new SparkContext(conf)
  }

  override def afterAll() {
    if (sc != null) {
      sc.stop()
    }
    sc = null
    super.afterAll()
  }

  // TODO Add more meaningful test cases

  test("Diagnostic network") {
    val filePath = "/SampleSales.csv"
    val fullPath = getClass.getResource(filePath).getFile
    val result = DataUtils.inferSchema(fullPath, 3, false)
    val schema = result._2
    val labeledPoints = result._3

    val rdd: RDD[Option[LabeledPoint]] = sc.parallelize(labeledPoints)
    val pointsRDD: RDD[LabeledPoint] = rdd filter {lp => lp.isDefined} map {point => point.get}
    val network = HillClimber.learnNetwork(pointsRDD, 2, 0.5, false, 3, schema)
    assert(network.allNodes.length == 4)
    assert(network.getParents(Node(3, true)).isEmpty)   // Sales is the root
    println("Network 1: ")
    println(network.treeDescription(schema))
  }

  test("Causal network") {
    val filePath = "/SampleSales.csv"
    val fullPath = getClass.getResource(filePath).getFile
    val result = DataUtils.inferSchema(fullPath, 3, false)
    val schema = result._2
    val labeledPoints = result._3

    val rdd: RDD[Option[LabeledPoint]] = sc.parallelize(labeledPoints)
    val pointsRDD: RDD[LabeledPoint] = rdd filter {lp => lp.isDefined} map {point => point.get}
    val network = HillClimber.learnNetwork(pointsRDD, 2, 0.5, true, 3, schema)
    assert(network.allNodes.length == 4)
    assert(network.getParents(Node(1, false)).isEmpty)   // Midage is the root
    println("Network 2: ")
    println(network.treeDescription(schema))
  }
}