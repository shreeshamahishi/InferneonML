package org.inferneon.bayesnet.simulatedannealing

import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.rdd.RDD
import org.apache.spark.{SparkConf, SparkContext}
import org.inferneon.datautils.DataUtils
import org.scalatest.{BeforeAndAfterAll, FunSuite}

/**
  * Tests for simulated annealing algorithm.
  */
class SimulatedAnnealingSuite extends FunSuite with BeforeAndAfterAll {

  @transient var sc: SparkContext = _

  override def beforeAll() {
    super.beforeAll()
    val conf = new SparkConf()
      .setMaster("local[2]")
      .setAppName("HillClimberUnitTest")
    sc = new SparkContext(conf)
    sc.setLogLevel("WARN")
  }

  override def afterAll() {
    if (sc != null) {
      sc.stop()
    }
    sc = null
    super.afterAll()
  }

  // TODO Add more meaningful test cases. For now we only illustrate how to run the algorithm.

  test("Diagnostic network") {
    val filePath = "/SampleSales.csv"
    val fullPath = getClass.getResource(filePath).getFile
    val result = DataUtils.inferSchema(fullPath, 3, false)
    val schema = result._2
    val labeledPoints = result._3

    val rdd: RDD[Option[LabeledPoint]] = sc.parallelize(labeledPoints)
    val pointsRDD: RDD[LabeledPoint] = rdd filter {lp => lp.isDefined} map {point => point.get}
    val network = SimulatedAnnealing.learnNetwork(pointsRDD, 2, 0.5, false, 3, schema, 10.0, 200, 0.999)
    assert(network.allNodes.length == 4)
    //assert(network.getParents(Node(3, true)).isEmpty)   // Sales is the root
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
    val network = SimulatedAnnealing.learnNetwork(pointsRDD, 2, 0.5, true, 3, schema, 10.0, 200, 0.999)
    assert(network.allNodes.length == 4)
    // assert(network.getParents(Node(3, true)).isEmpty)   // Sales is the root
    println("Network 1: ")
    println(network.treeDescription(schema))
  }

}
