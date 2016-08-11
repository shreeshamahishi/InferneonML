package org.inferneon.bayesnet

import org.apache.spark.mllib.linalg.Vectors
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.{SparkConf, SparkContext}
import org.scalatest.{BeforeAndAfterAll, FunSuite}

import scala.io.Source

class DataUtilsSuite extends FunSuite with BeforeAndAfterAll {

  @transient var sc: SparkContext = _

  override def beforeAll() {
    super.beforeAll()
    val conf = new SparkConf()
      .setMaster("local[2]")
      .setAppName("DataUtilsUnitTest")
    sc = new SparkContext(conf)
  }

  override def afterAll() {
    if (sc != null) {
      sc.stop()
    }
    sc = null
    super.afterAll()
  }

  // TODO Minimize warnings.

  test("All nominal values, no missing data") {
    val filePath = "/NominalValuesOnlyNoMissing.csv"
    val fullPath = getClass.getResource(filePath).getFile
    val (errors, schema, points) = DataUtils.inferSchema(fullPath, 4, true)

    assert(errors.isEmpty)
    assert(schema.length == 5)
    assert(schema(0)._1 == "Outlook")
    assert(schema(0)._2.sameElements(Array("Sunny", "Overcast", "Rain")))
    assert(schema(1)._1 == "Temperature")
    assert(schema(1)._2.sameElements(Array("Hot", "Mild", "Cool")))
    assert(schema(2)._1 == "Humidity")
    assert(schema(2)._2.sameElements(Array("High", "Normal")))
    assert(schema(3)._1 == "Wind")
    assert(schema(3)._2.sameElements(Array("Weak", "Strong")))
    assert(schema(4)._1 == "PlayTennis")
    assert(schema(4)._2.sameElements(Array("No", "Yes")))

    pointsForNominalValuesOnlyNoMissing(points)
  }

  private def pointsForNominalValuesOnlyNoMissing(points : List[Option[LabeledPoint]]): Unit ={
    assert(points.length == 14)
    assert(points(0).get.label === 0.0)
    assert(points(0).get.features === Vectors.dense(Array(0.0,0.0,0.0,0.0)))
    assert(points(1).get.label === 0.0)
    assert(points(1).get.features === Vectors.dense(Array(0.0,0.0,0.0,1.0)))
    assert(points(2).get.label === 1.0)
    assert(points(2).get.features === Vectors.dense(Array(1.0,0.0,0.0,0.0)))
    assert(points(3).get.label === 1.0)
    assert(points(3).get.features === Vectors.dense(Array(2.0,1.0,0.0,0.0)))
    assert(points(4).get.label === 1.0)
    assert(points(4).get.features === Vectors.dense(Array(2.0,2.0,1.0,0.0)))
    assert(points(5).get.label === 0.0)
    assert(points(5).get.features === Vectors.dense(Array(2.0,2.0,1.0,1.0)))
    assert(points(6).get.label === 1.0)
    assert(points(6).get.features === Vectors.dense(Array(1.0,2.0,1.0,1.0)))
    assert(points(7).get.label === 0.0)
    assert(points(7).get.features === Vectors.dense(Array(0.0,1.0,0.0,0.0)))
    assert(points(8).get.label === 1.0)
    assert(points(8).get.features === Vectors.dense(Array(0.0,2.0,1.0,0.0)))
    assert(points(9).get.label === 1.0)
    assert(points(9).get.features === Vectors.dense(Array(2.0,1.0,1.0,0.0)))
    assert(points(10).get.label === 1.0)
    assert(points(10).get.features === Vectors.dense(Array(0.0,1.0,1.0,1.0)))
    assert(points(11).get.label === 1.0)
    assert(points(11).get.features === Vectors.dense(Array(1.0,1.0,0.0,1.0)))
    assert(points(12).get.label === 1.0)
    assert(points(12).get.features === Vectors.dense(Array(1.0,0.0,1.0,0.0)))
    assert(points(13).get.label === 0.0)
    assert(points(13).get.features === Vectors.dense(Array(2.0,1.0,0.0,1.0)))
  }


  test("All nominal values, some missing data") {
    val filePath = "/NominalValuesOnlySomeMissing.csv"
    val fullPath = getClass.getResource(filePath).getFile
    val (errors, schema, points) = DataUtils.inferSchema(fullPath, 4, true)

    assert(errors.isEmpty)
    assert(schema.length == 5)
    assert(schema(0)._1 == "Outlook")
    assert(schema(0)._2.sameElements(Array("Sunny", "Overcast", "Rain")))
    assert(schema(1)._1 == "Temperature")
    assert(schema(1)._2.sameElements(Array("Hot", "Mild", "Cool")))
    assert(schema(2)._1 == "Humidity")
    assert(schema(2)._2.sameElements(Array("High", "Normal")))
    assert(schema(3)._1 == "Wind")
    assert(schema(3)._2.sameElements(Array("Weak", "Strong")))
    assert(schema(4)._1 == "PlayTennis")
    assert(schema(4)._2.sameElements(Array("No", "Yes")))

    pointsForNominalValuesOnlySomeMissing(points)
  }

  private def pointsForNominalValuesOnlySomeMissing(points : List[Option[LabeledPoint]]): Unit ={
    assert(points.length == 14)
    assert(points(0).get.label === 0.0)
    assert(points(0).get.features === Vectors.dense(Array(0.0,0.0,0.0,0.0)))
    assert(points(1).get.label === 0.0)
    assert(points(1).get.features === Vectors.dense(Array(0.0,0.0,0.0,1.0)))
    assert(points(2).get.label === 1.0)
    assert(points(2).get.features === Vectors.dense(Array(1.0,0.0,0.0,0.0)))
    assert(points(3).get.label === 1.0)
    assert(points(3).get.features === Vectors.dense(Array(2.0,1.0,0.0,0.0)))
    assert(points(4).get.label === 1.0)
    assert(points(4).get.features === Vectors.dense(Array(2.0,2.0,1.0,0.0)))
    assert(points(5).get.label === 0.0)
    assert(points(5).get.features === Vectors.dense(Array(2.0,2.0,1.0,1.0)))
    assert(points(6).get.label === 1.0)
    assert(points(6).get.features === Vectors.sparse(4, Seq((0, 1.0), (2, 1.0))))
    assert(points(7).get.label === 0.0)
    assert(points(7).get.features === Vectors.dense(Array(0.0,1.0,0.0,0.0)))
    assert(points(8).get.label === 1.0)
    assert(points(8).get.features === Vectors.sparse(4, Seq((0, 0.0), (2, 1.0), (3, 0.0))))
    assert(points(9).get.label === 1.0)
    assert(points(9).get.features === Vectors.dense(Array(2.0,1.0,1.0,0.0)))
    assert(points(10).get.label === 1.0)
    assert(points(10).get.features === Vectors.sparse(4, Seq((0, 0.0), (1, 1.0), (2, 1.0))))
    assert(points(11).get.label === 1.0)
    assert(points(11).get.features === Vectors.dense(Array(1.0,1.0,0.0,1.0)))
    assert(points(12).get.label === 1.0)
    assert(points(12).get.features === Vectors.sparse(4, Seq((0, 1.0), (2, 1.0), (3, 0.0))))
    assert(points(13).get.label === 0.0)
    assert(points(13).get.features === Vectors.dense(Array(2.0,1.0,0.0,1.0)))
  }

  test("All nominal values, no missing data and class index at first position") {
    val filePath = "/NominalValuesOnlyNoMissingCIAtFirstPos.csv"
    val fullPath = getClass.getResource(filePath).getFile
    val (errors, schema, points) = DataUtils.inferSchema(fullPath, 0, true)
    assert(errors.isEmpty)
    assert(schema.length == 5)
    assert(schema(0)._1 == "PlayTennis")
    assert(schema(0)._2.sameElements(Array("No", "Yes")))
    assert(schema(1)._1 == "Outlook")
    assert(schema(1)._2.sameElements(Array("Sunny", "Overcast", "Rain")))
    assert(schema(2)._1 == "Temperature")
    assert(schema(2)._2.sameElements(Array("Hot", "Mild", "Cool")))
    assert(schema(3)._1 == "Humidity")
    assert(schema(3)._2.sameElements(Array("High", "Normal")))
    assert(schema(4)._1 == "Wind")
    assert(schema(4)._2.sameElements(Array("Weak", "Strong")))

    pointsForNominalValuesOnlyNoMissingCIAtFirstPos(points)
  }

  private def pointsForNominalValuesOnlyNoMissingCIAtFirstPos(points : List[Option[LabeledPoint]]): Unit ={
    assert(points.length == 14)
    assert(points(0).get.label === 0.0)
    assert(points(0).get.features === Vectors.dense(Array(0.0,0.0,0.0,0.0)))
    assert(points(1).get.label === 0.0)
    assert(points(1).get.features === Vectors.dense(Array(0.0,0.0,0.0,1.0)))
    assert(points(2).get.label === 1.0)
    assert(points(2).get.features === Vectors.dense(Array(1.0,0.0,0.0,0.0)))
    assert(points(3).get.label === 1.0)
    assert(points(3).get.features === Vectors.dense(Array(2.0,1.0,0.0,0.0)))
    assert(points(4).get.label === 1.0)
    assert(points(4).get.features === Vectors.dense(Array(2.0,2.0,1.0,0.0)))
    assert(points(5).get.label === 0.0)
    assert(points(5).get.features === Vectors.dense(Array(2.0,2.0,1.0,1.0)))
    assert(points(6).get.label === 1.0)
    assert(points(6).get.features === Vectors.dense(Array(1.0,2.0,1.0,1.0)))
    assert(points(7).get.label === 0.0)
    assert(points(7).get.features === Vectors.dense(Array(0.0,1.0,0.0,0.0)))
    assert(points(8).get.label === 1.0)
    assert(points(8).get.features === Vectors.dense(Array(0.0,2.0,1.0,0.0)))
    assert(points(9).get.label === 1.0)
    assert(points(9).get.features === Vectors.dense(Array(2.0,1.0,1.0,0.0)))
    assert(points(10).get.label === 1.0)
    assert(points(10).get.features === Vectors.dense(Array(0.0,1.0,1.0,1.0)))
    assert(points(11).get.label === 1.0)
    assert(points(11).get.features === Vectors.dense(Array(1.0,1.0,0.0,1.0)))
    assert(points(12).get.label === 1.0)
    assert(points(12).get.features === Vectors.dense(Array(1.0,0.0,1.0,0.0)))
    assert(points(13).get.label === 0.0)
    assert(points(13).get.features === Vectors.dense(Array(2.0,1.0,0.0,1.0)))
  }

  test("All nominal values, no missing data and class index at middle position") {
    val filePath = "/NominalValuesOnlyNoMissingCIAtMidPos.csv"
    val fullPath = getClass.getResource(filePath).getFile
    val (errors, schema, points) = DataUtils.inferSchema(fullPath, 2, true)

    assert(errors.isEmpty)
    assert(schema.length == 5)
    assert(schema(0)._1 == "Outlook")
    assert(schema(0)._2.sameElements(Array("Sunny", "Overcast", "Rain")))
    assert(schema(1)._1 == "Temperature")
    assert(schema(1)._2.sameElements(Array("Hot", "Mild", "Cool")))
    assert(schema(2)._1 == "PlayTennis")
    assert(schema(2)._2.sameElements(Array("No", "Yes")))
    assert(schema(3)._1 == "Humidity")
    assert(schema(3)._2.sameElements(Array("High", "Normal")))
    assert(schema(4)._1 == "Wind")
    assert(schema(4)._2.sameElements(Array("Weak", "Strong")))

    pointsForNominalValuesOnlyNoMissingCIAtMidPos(points)
  }

  private def pointsForNominalValuesOnlyNoMissingCIAtMidPos(points : List[Option[LabeledPoint]]): Unit ={
    assert(points.length == 14)
    assert(points(0).get.label === 0.0)
    assert(points(0).get.features === Vectors.dense(Array(0.0,0.0,0.0,0.0)))
    assert(points(1).get.label === 0.0)
    assert(points(1).get.features === Vectors.dense(Array(0.0,0.0,0.0,1.0)))
    assert(points(2).get.label === 1.0)
    assert(points(2).get.features === Vectors.dense(Array(1.0,0.0,0.0,0.0)))
    assert(points(3).get.label === 1.0)
    assert(points(3).get.features === Vectors.dense(Array(2.0,1.0,0.0,0.0)))
    assert(points(4).get.label === 1.0)
    assert(points(4).get.features === Vectors.dense(Array(2.0,2.0,1.0,0.0)))
    assert(points(5).get.label === 0.0)
    assert(points(5).get.features === Vectors.dense(Array(2.0,2.0,1.0,1.0)))
    assert(points(6).get.label === 1.0)
    assert(points(6).get.features === Vectors.dense(Array(1.0,2.0,1.0,1.0)))
    assert(points(7).get.label === 0.0)
    assert(points(7).get.features === Vectors.dense(Array(0.0,1.0,0.0,0.0)))
    assert(points(8).get.label === 1.0)
    assert(points(8).get.features === Vectors.dense(Array(0.0,2.0,1.0,0.0)))
    assert(points(9).get.label === 1.0)
    assert(points(9).get.features === Vectors.dense(Array(2.0,1.0,1.0,0.0)))
    assert(points(10).get.label === 1.0)
    assert(points(10).get.features === Vectors.dense(Array(0.0,1.0,1.0,1.0)))
    assert(points(11).get.label === 1.0)
    assert(points(11).get.features === Vectors.dense(Array(1.0,1.0,0.0,1.0)))
    assert(points(12).get.label === 1.0)
    assert(points(12).get.features === Vectors.dense(Array(1.0,0.0,1.0,0.0)))
    assert(points(13).get.label === 0.0)
    assert(points(13).get.features === Vectors.dense(Array(2.0,1.0,0.0,1.0)))
  }

  test("All nominal values, some missing data and class index at middle position") {
    val filePath = "/NominalValuesOnlySomeMissingCIAtMidPos.csv"
    val fullPath = getClass.getResource(filePath).getFile
    val (errors, schema, points) = DataUtils.inferSchema(fullPath, 2, true)

    assert(errors.isEmpty)
    assert(schema.length == 5)
    assert(schema(0)._1 == "Outlook")
    assert(schema(0)._2.sameElements(Array("Sunny", "Overcast", "Rain")))
    assert(schema(1)._1 == "Temperature")
    assert(schema(1)._2.sameElements(Array("Hot", "Mild", "Cool")))
    assert(schema(2)._1 == "PlayTennis")
    assert(schema(2)._2.sameElements(Array("No", "Yes")))
    assert(schema(3)._1 == "Humidity")
    assert(schema(3)._2.sameElements(Array("High", "Normal")))
    assert(schema(4)._1 == "Wind")
    assert(schema(4)._2.sameElements(Array("Weak", "Strong")))

    pointsForNominalValuesOnlySomeMissingCIAtMidPos(points)
  }

  private def pointsForNominalValuesOnlySomeMissingCIAtMidPos(points : List[Option[LabeledPoint]]): Unit ={
    assert(points.length == 15)
    assert(points(0).get.label === 0.0)
    assert(points(0).get.features === Vectors.dense(Array(0.0,0.0,0.0,0.0)))
    assert(points(1).get.label === 0.0)
    assert(points(1).get.features === Vectors.dense(Array(0.0,0.0,0.0,1.0)))
    assert(points(2).get.label === 1.0)
    assert(points(2).get.features === Vectors.dense(Array(1.0,0.0,0.0,0.0)))
    assert(points(3).get.label === 1.0)
    assert(points(3).get.features === Vectors.dense(Array(2.0,1.0,0.0,0.0)))
    assert(points(4).get.label === 1.0)
    assert(points(4).get.features === Vectors.dense(Array(2.0,2.0,1.0,0.0)))
    assert(points(5).get.label === 0.0)
    assert(points(5).get.features === Vectors.sparse(4, Seq((0, 2.0), (1, 2.0), (2, 1.0))))
    assert(points(6).get.label === 1.0)
    assert(points(6).get.features === Vectors.sparse(4, Seq((0, 1.0), (1, 2.0), (3, 1.0))))
    assert(points(7).get.label === 0.0)
    assert(points(7).get.features === Vectors.sparse(4, Seq((0, 0.0), (1, 1.0))))
    assert(points(8).get.label === 1.0)
    assert(points(8).get.features === Vectors.sparse(4, Seq((2, 1.0), (3, 0.0))))
    assert(points(9).get.label === 1.0)
    assert(points(9).get.features === Vectors.sparse(4, Seq((0, 2.0), (2, 1.0))))
    assert(points(10).get.label === 1.0)
    assert(points(10).get.features === Vectors.dense(Array(0.0,1.0,1.0,1.0)))
    assert(points(11).get.label === 1.0)
    assert(points(11).get.features === Vectors.dense(Array(1.0,1.0,0.0,1.0)))
    assert(points(12).get.label === 1.0)
    assert(points(12).get.features === Vectors.dense(Array(1.0,0.0,1.0,0.0)))
    assert(points(13).get.label === 0.0)
    assert(points(13).get.features === Vectors.dense(Array(2.0,1.0,0.0,1.0)))
    assert(points(14).get.label === 1.0)
    assert(points(14).get.features === Vectors.sparse(4, Seq((0, 0.0), (1, 2.0))))
  }

  test("Test empty file") {
    val filePath = "/EmptyFile.csv"
    val fullPath = getClass.getResource(filePath).getFile
    try {
      val (errors, schema, points) = DataUtils.inferSchema(fullPath, 4, true)
    }
    catch {
      case e: IllegalArgumentException => assert(e.getMessage == "requirement failed: File is empty, cannot infer the schema.")
    }

    // Check loading labeled points
    val schema: Array[(String, Array[String])] = Array[(String, Array[String])](
      ("Outlook", Array("Sunny", "Overcast", "Rain")),
      ("Temperature", Array("Hot", "Mild", "Cool")),
      ("PlayTennis", Array("No", "Yes")),
      ("Humidity", Array("High", "Normal")),
      ("Wind", Array("Weak", "Strong")))
    try {
      val (errors , points) = DataUtils.loadLabeledPoints(filePath, schema, 2, true)
    }
    catch {
      case e: IllegalArgumentException => assert(e.getMessage == "requirement failed: File is empty, cannot infer the schema.")
    }
  }

  test("Only valid header, no data") {
    val filePath = "/ValidHeadersNoData.csv"
    val fullPath = getClass.getResource(filePath).getFile
    try {
      DataUtils.inferSchema(fullPath, 0, true)
    }
    catch {
      case e: IllegalArgumentException => assert(e.getMessage == "requirement failed: No data found.")
    }
  }

  test("Only valid header, multiple lines with no data") {
    val filePath = "/ValidHeadersMultipleLinesNoData.csv"
    val fullPath = getClass.getResource(filePath).getFile
    val errors = DataUtils.inferSchema(fullPath, 4, true)._1
    assert(errors.length == 1)
    assert(errors(0) == "No data found.")
  }

  test("Empty column names") {
    val filePath = "/EmptyColumnNames.csv"
    val fullPath = getClass.getResource(filePath).getFile
    val errors = DataUtils.inferSchema(fullPath, 4, caseSensitive = true)._1
    assert(errors.length == 1)
    assert(errors(0) == "Found empty column names.")
  }

  test("Only one column") {
    val filePath = "/OnlyOneColumn.csv"
    val fullPath = getClass.getResource(filePath).getFile
    val result = DataUtils.inferSchema(fullPath, 0, true)
    val errors = result._1
    assert(errors.length == 1)
    assert(errors(0) == "The data must contain at least two columns.")
  }

  test("Too few columns") {
    val filePath = "/TooFewColumns.csv"
    val fullPath = getClass.getResource(filePath).getFile
    val (errors, schema, points) = DataUtils.inferSchema(fullPath, 4, true)
    assert(errors.length == 1)
    assert(errors(0) == "There are too few columns for the class index specified.")
  }

  test("Unknown column") {
    val filePath = "/NominalValuesOnlyEntireColumnMissing.csv"
    val fullPath = getClass.getResource(filePath).getFile
    val (errors, schema, points) = DataUtils.inferSchema(fullPath, 4, true)
    assert(errors.length == 1)
    assert(errors(0) == "Unable to identify field at column index 1")
  }

  test("No missing data but some empty lines") {
    val filePath = "/NominalValuesOnlyNoMissingSomeEmptyLines.csv"
    val fullPath = getClass.getResource(filePath).getFile
    val (errors, schema, points) = DataUtils.inferSchema(fullPath, 4, true)

    assert(errors.isEmpty)
    assert(schema.length == 5)
    assert(schema(0)._1 == "Outlook")
    assert(schema(0)._2.sameElements(Array("Sunny", "Overcast", "Rain")))
    assert(schema(1)._1 == "Temperature")
    assert(schema(1)._2.sameElements(Array("Hot", "Mild", "Cool")))
    assert(schema(2)._1 == "Humidity")
    assert(schema(2)._2.sameElements(Array("High", "Normal")))
    assert(schema(3)._1 == "Wind")
    assert(schema(3)._2.sameElements(Array("Strong", "Weak")))
    assert(schema(4)._1 == "PlayTennis")
    assert(schema(4)._2.sameElements(Array("No", "Yes")))

    assert(points.length == 14)
    assert(points(0).isEmpty)
    assert(points(1).get.label === 0.0)
    assert(points(1).get.features === Vectors.dense(Array(0.0,0.0,0.0,0.0)))
    assert(points(2).get.label === 1.0)
    assert(points(2).get.features === Vectors.dense(Array(1.0,0.0,0.0,1.0)))
    assert(points(3).get.label === 1.0)
    assert(points(3).get.features === Vectors.dense(Array(2.0,1.0,0.0,1.0)))
    assert(points(4).isEmpty)
    assert(points(5).get.label === 0.0)
    assert(points(5).get.features === Vectors.dense(Array(2.0,2.0,1.0,0.0)))
    assert(points(6).isEmpty)
    assert(points(7).get.label === 0.0)
    assert(points(7).get.features === Vectors.dense(Array(0.0,1.0,0.0,1.0)))
    assert(points(8).get.label === 1.0)
    assert(points(8).get.features === Vectors.dense(Array(0.0,2.0,1.0,1.0)))
    assert(points(9).isEmpty)
    assert(points(10).get.label === 1.0)
    assert(points(10).get.features === Vectors.dense(Array(0.0,1.0,1.0,0.0)))
    assert(points(11).isEmpty)
    assert(points(12).get.label === 1.0)
    assert(points(12).get.features === Vectors.dense(Array(1.0,0.0,1.0,1.0)))
    assert(points(13).get.label === 0.0)
    assert(points(13).get.features === Vectors.dense(Array(2.0,1.0,0.0,0.0)))

  }

  test("One row with all missing features") {
    val filePath = "/NominalValuesOnlyRowWithAllMissingValues.csv"
    val fullPath = getClass.getResource(filePath).getFile
    val (errors, schema, points) = DataUtils.inferSchema(fullPath, 4, true)

    assert(errors.isEmpty)
    assert(schema.length == 5)
    assert(schema(0)._1 == "Outlook")
    assert(schema(0)._2.sameElements(Array("Sunny", "Overcast", "Rain")))
    assert(schema(1)._1 == "Temperature")
    assert(schema(1)._2.sameElements(Array("Hot", "Mild", "Cool")))
    assert(schema(2)._1 == "Humidity")
    assert(schema(2)._2.sameElements(Array("High", "Normal")))
    assert(schema(3)._1 == "Wind")
    assert(schema(3)._2.sameElements(Array("Weak", "Strong")))
    assert(schema(4)._1 == "PlayTennis")
    assert(schema(4)._2.sameElements(Array("No", "Yes")))

    assert(points.length == 8)
    assert(points(0).get.label === 0.0)
    assert(points(0).get.features === Vectors.dense(Array(0.0,0.0,0.0,0.0)))
    assert(points(1).get.label === 0.0)
    assert(points(1).get.features === Vectors.dense(Array(0.0,0.0,0.0,1.0)))
    assert(points(2).get.label === 1.0)
    assert(points(2).get.features === Vectors.dense(Array(1.0,0.0,0.0,0.0)))
    assert(points(3).get.label === 1.0)
    assert(points(3).get.features === Vectors.dense(Array(2.0,1.0,0.0,0.0)))
    assert(points(4).get.label === 1.0)
    assert(points(4).get.features === Vectors.dense(Array(2.0,2.0,1.0,0.0)))
    assert(points(5).get.label === 0.0)
    assert(points(5).get.features === Vectors.dense(Array(2.0,2.0,1.0,1.0)))
    assert(points(6).get.label === 1.0)
    assert(points(6).get.features === Vectors.sparse(4, Seq()))
    assert(points(7).get.label === 0.0)
    assert(points(7).get.features === Vectors.dense(Array(0.0,1.0,0.0,0.0)))

  }

  test("All numeric, no missing data") {
    val filePath = "/NumericValuesOnlyNoMissing.csv"
    val fullPath = getClass.getResource(filePath).getFile
    val (errors, schema, points) = DataUtils.inferSchema(fullPath, 3, true)

    assert(errors.isEmpty)
    assert(schema.length == 4)
    assert(schema(0)._1 == "F1")
    assert(schema(0)._2.isEmpty)
    assert(schema(1)._1 == "F2")
    assert(schema(1)._2.isEmpty)
    assert(schema(2)._1 == "F3")
    assert(schema(2)._2.isEmpty)
    assert(schema(3)._1 == "F4")
    assert(schema(3)._2.isEmpty)

    assert(points.length == 4)
    assert(points(0).get.label === 0.9)
    assert(points(0).get.features === Vectors.dense(Array(0.1,0.2,0.3)))
    assert(points(1).get.label === 11.3)
    assert(points(1).get.features === Vectors.dense(Array(0.4,0.8,0.9)))
    assert(points(2).get.label === 9.1)
    assert(points(2).get.features === Vectors.dense(Array(5.6, 7.8, 8.9)))
    assert(points(3).get.label === 14.0)
    assert(points(3).get.features === Vectors.dense(Array(1.2, 2.3, 4)))
  }

  test("Mixed types, no missing data") {
    val filePath = "/MixedTypesNoMissing.csv"
    val fullPath = getClass.getResource(filePath).getFile
    val (errors, schema, points) = DataUtils.inferSchema(fullPath, 3, true)

    assert(errors.isEmpty)

    assert(schema.length == 4)
    assert(schema(0)._1 == "F1")
    assert(schema(0)._2.sameElements(Array("M", "F")))
    assert(schema(1)._1 == "F2")
    assert(schema(1)._2.isEmpty)
    assert(schema(2)._1 == "F3")
    assert(schema(2)._2.sameElements(Array("Student", "Professional", "Teacher")))
    assert(schema(3)._1 == "F4")
    assert(schema(3)._2.sameElements(Array("Y", "N")))

    pointsForMixedTypeNoMissingData(points)
  }

  private def pointsForMixedTypeNoMissingData(points : List[Option[LabeledPoint]]): Unit ={
    assert(points.length == 4)
    assert(points(0).get.label === 0.0)
    assert(points(0).get.features === Vectors.dense(Array(0.0, 23.0, 0.0)))
    assert(points(1).get.label === 0.0)
    assert(points(1).get.features === Vectors.dense(Array(1.0,41.0,1.0)))
    assert(points(2).get.label === 1.0)
    assert(points(2).get.features === Vectors.dense(Array(0.0, 22.0, 0.0)))
    assert(points(3).get.label === 1.0)
    assert(points(3).get.features === Vectors.dense(Array(1.0, 60.0, 2.0)))
  }

  test("Mixed types, some missing data") {
    val filePath = "/MixedTypesSomeMissing.csv"
    val fullPath = getClass.getResource(filePath).getFile
    val (errors, schema, points) = DataUtils.inferSchema(fullPath, 3, true)

    assert(errors.isEmpty)
    assert(schema.length == 4)
    assert(schema(0)._1 == "F1")
    assert(schema(0)._2.sameElements(Array("M", "F")))
    assert(schema(1)._1 == "F2")
    assert(schema(1)._2.isEmpty)
    assert(schema(2)._1 == "F3")
    assert(schema(2)._2.sameElements(Array("Student", "Professional", "Teacher")))
    assert(schema(3)._1 == "F4")
    assert(schema(3)._2.sameElements(Array("Y", "N")))

    assert(points.length == 9)
    assert(points(0).get.label === 0.0)
    assert(points(0).get.features === Vectors.dense(Array(0.0, 23.0, 0.0)))
    assert(points(1).get.label === 0.0)
    assert(points(1).get.features === Vectors.dense(Array(1.0,41.0,1.0)))
    assert(points(2).get.label === 1.0)
    assert(points(2).get.features === Vectors.dense(Array(0.0, 22.0, 0.0)))
    assert(points(3).get.label === 1.0)
    assert(points(3).get.features === Vectors.dense(Array(1.0, 60.0, 2.0)))
    assert(points(4).get.label === 1.0)
    assert(points(4).get.features === Vectors.sparse(3, Seq((0, 1.0), (1, 60.0))))
    assert(points(5).get.label === 1.0)
    assert(points(5).get.features === Vectors.sparse(3, Seq((0, 0.0), (2, 2.0))))
    assert(points(6).get.label === 0.0)
    assert(points(6).get.features === Vectors.dense(Array(1.0, 19.0, 0.0)))
    assert(points(7).get.label === 0.0)
    assert(points(7).get.features === Vectors.sparse(3, Seq((0, 1.0))))
    assert(points(8).get.label === 1.0)
    assert(points(8).get.features === Vectors.dense(Array(1.0, 60.0, 2.0)))
  }

  test("Mixed types, categorical data changed to numerical in second column") {
    val filePath = "/MixedTypesCategoricalToNumericalChangeInCol.csv"
    val fullPath = getClass.getResource(filePath).getFile
    val (errors, schema, points) = DataUtils.inferSchema(fullPath, 3, true)

    assert(errors.size == 0)
    assert(schema.length == 4)
    assert(schema(0)._1 == "F1")
    assert(schema(0)._2.sameElements(Array("M", "F")))
    assert(schema(1)._1 == "F2")
    assert(schema(1)._2.isEmpty)
    assert(schema(2)._1 == "F3")
    assert(schema(2)._2.sameElements(Array("Student", "Professional", "Teacher", "15.8", "16.9", "20.3")))
    assert(schema(3)._1 == "F4")
    assert(schema(3)._2.sameElements(Array("Y", "N")))

    assert(points.length == 9)
    assert(points(3).get.label === 1.0)
    assert(points(3).get.features === Vectors.dense(Array(1.0, 60.0, 2.0)))
    assert(points(4).get.label === 0.0)
    assert(points(4).get.features === Vectors.dense(Array(0.0, 23.0, 3.0)))
    assert(points(5).get.label === 0.0)
    assert(points(5).get.features === Vectors.dense(Array(1.0, 41.0, 4.0)))
    assert(points(6).get.label === 1.0)
    assert(points(6).get.features === Vectors.dense(Array(0.0, 22.0, 1.0)))
    assert(points(7).get.label === 1.0)
    assert(points(7).get.features === Vectors.dense(Array(1.0, 43.0, 4.0)))
    assert(points(8).get.label === 0.0)
    assert(points(8).get.features === Vectors.dense(Array(0.0, 20.0, 5.0)))
  }

  test("Mixed types, numerical data changed to categorical in first column") {
    val filePath = "/MixedTypesNumericalToCategoricalChangeInCol.csv"
    val fullPath = getClass.getResource(filePath).getFile
    val (errors, schema, points) = DataUtils.inferSchema(fullPath, 3, true)

    assert(errors.size == 2)
    assert(schema.length == 4)
    assert(schema(0)._1 == "F1")
    assert(schema(0)._2.sameElements(Array("M", "F")))
    assert(schema(1)._1 == "F2")
    assert(schema(1)._2.isEmpty)
    assert(schema(2)._1 == "F3")
    assert(schema(2)._2.sameElements(Array("Student", "Professional", "Teacher")))
    assert(schema(3)._1 == "F4")
    assert(schema(3)._2.sameElements(Array("Y", "N")))

    assert(points.length == 9)
    assert(points(3).get.label === 1.0)
    assert(points(3).get.features === Vectors.dense(Array(1.0, 60.0, 2.0)))
    assert(points(4).get.label === 0.0)
    assert(points(4).get.features === Vectors.dense(Array(0.0, 23.0, 2.0)))
    assert(points(5).get.label === 0.0)
    assert(points(5).get.features === Vectors.sparse(3, Seq((0, 1.0), (2, 0.0))))
    assert(points(6).get.label === 1.0)
    assert(points(6).get.features === Vectors.dense(Array(0.0, 22.0, 1.0)))
    assert(points(7).get.label === 1.0)
    assert(points(7).get.features === Vectors.sparse(3, Seq((0, 1.0), (2, 0.0))))
    assert(points(8).get.label === 0.0)
    assert(points(8).get.features === Vectors.dense(Array(0.0, 20.0, 1.0)))
  }

  test("File not found"){
    val fullPath = "SomeMissingFile"
    val (errors, schema, points) = DataUtils.inferSchema(fullPath, 3, true)
    assert(errors.size == 1 && errors(0) == "File not found.")
  }

  test("Invalid file"){
    val filePath = "/InvalidFile.csv"
    val fullPath = getClass.getResource(filePath).getFile
    val (errors, schema, points) = DataUtils.inferSchema(fullPath, 3, true)
    assert(errors.size == 1 && (errors(0) == "Invalid file. Please use a valid CSV file." ||
      errors(0) == "Unknown error. Please use a valid CSV file."))
  }

  test("All nominal values, no missing data, case insensitive") {
    val filePath = "/NominalValuesOnlyCaseInsensitive.csv"
    val fullPath = getClass.getResource(filePath).getFile
    val (errors, schema, points) = DataUtils.inferSchema(fullPath, 4, false)

    assert(errors.isEmpty)
    assert(schema.length == 5)
    assert(schema(0)._1 == "Outlook")
    assert(schema(0)._2.map(_.toLowerCase()).sameElements(Array("Sunny", "Overcast", "Rain").map(_.toLowerCase())))
    assert(schema(1)._1 == "Temperature")

    assert(schema(1)._2.map(_.toLowerCase()).sameElements(Array("Hot", "Mild", "Cool").map(_.toLowerCase())))
    assert(schema(2)._1 == "Humidity")
    assert(schema(2)._2.map(_.toLowerCase()).sameElements(Array("High", "Normal").map(_.toLowerCase())))
    assert(schema(3)._1 == "Wind")
    assert(schema(3)._2.map(_.toLowerCase()).sameElements(Array("Weak", "Strong").map(_.toLowerCase())))
    assert(schema(4)._1 == "PlayTennis")
    assert(schema(4)._2.map(_.toLowerCase()).sameElements(Array("No", "Yes").map(_.toLowerCase())))

    assert(points.length == 14)
    assert(points(0).get.label === 0.0)
    assert(points(0).get.features === Vectors.dense(Array(0.0,0.0,0.0,0.0)))
    assert(points(1).get.label === 0.0)
    assert(points(1).get.features === Vectors.dense(Array(0.0,0.0,0.0,1.0)))
    assert(points(2).get.label === 1.0)
    assert(points(2).get.features === Vectors.dense(Array(1.0,0.0,0.0,0.0)))
    assert(points(3).get.label === 1.0)
    assert(points(3).get.features === Vectors.dense(Array(2.0,1.0,0.0,0.0)))
    assert(points(4).get.label === 1.0)
    assert(points(4).get.features === Vectors.dense(Array(2.0,2.0,1.0,0.0)))
    assert(points(5).get.label === 0.0)
    assert(points(5).get.features === Vectors.dense(Array(2.0,2.0,1.0,1.0)))
    assert(points(6).get.label === 1.0)
    assert(points(6).get.features === Vectors.dense(Array(1.0,2.0,1.0,1.0)))
    assert(points(7).get.label === 0.0)
    assert(points(7).get.features === Vectors.dense(Array(0.0,1.0,0.0,0.0)))
    assert(points(8).get.label === 1.0)
    assert(points(8).get.features === Vectors.dense(Array(0.0,2.0,1.0,0.0)))
    assert(points(9).get.label === 1.0)
    assert(points(9).get.features === Vectors.dense(Array(2.0,1.0,1.0,0.0)))
    assert(points(10).get.label === 1.0)
    assert(points(10).get.features === Vectors.dense(Array(0.0,1.0,1.0,1.0)))
    assert(points(11).get.label === 1.0)
    assert(points(11).get.features === Vectors.dense(Array(1.0,1.0,0.0,1.0)))
    assert(points(12).get.label === 1.0)
    assert(points(12).get.features === Vectors.dense(Array(1.0,0.0,1.0,0.0)))
    assert(points(13).get.label === 0.0)
    assert(points(13).get.features === Vectors.dense(Array(2.0,1.0,0.0,1.0)))
  }

  test("Load labeled points from schema, all categorical data, no missing data"){
    val filePath = "/NominalValuesOnlyNoMissing1.csv"
    val fullPath = getClass.getResource(filePath).getFile
    val schema: Array[(String, Array[String])] = Array[(String, Array[String])](("Outlook", Array("Sunny", "Overcast", "Rain")),
      ("Temperature", Array("Hot", "Mild", "Cool")),
      ("Humidity", Array("High", "Normal")),
      ("Wind", Array("Weak", "Strong")),
      ("PlayTennis", Array("No", "Yes")))
    val (errors, points) = DataUtils.loadLabeledPoints(fullPath, schema, 4, true)

    assert(errors.isEmpty)
    pointsForNominalValuesOnlyNoMissing(points.toList)
  }

  test("Labeled points from schema, All nominal values, some missing data") {

    val filePath = "/NominalValuesOnlySomeMissing1.csv"
    val fullPath = getClass.getResource(filePath).getFile
    val schema: Array[(String, Array[String])] = Array[(String, Array[String])](("Outlook", Array("Sunny", "Overcast", "Rain")),
      ("Temperature", Array("Hot", "Mild", "Cool")),
      ("Humidity", Array("High", "Normal")),
      ("Wind", Array("Weak", "Strong")),
      ("PlayTennis", Array("No", "Yes")))
    val (errors, points) = DataUtils.loadLabeledPoints(fullPath, schema, 4, true)
    assert(errors.isEmpty)
    pointsForNominalValuesOnlySomeMissing(points.toList)
  }

  test("Labeled points from schema, all nominal values, no missing data and class index at first position") {
    val filePath = "/NominalValuesOnlyNoMissingCIAtFirstPos1.csv"
    val fullPath = getClass.getResource(filePath).getFile
    val schema: Array[(String, Array[String])] = Array[(String, Array[String])](("PlayTennis", Array("No", "Yes")),
      ("Outlook", Array("Sunny", "Overcast", "Rain")),
      ("Temperature", Array("Hot", "Mild", "Cool")),
      ("Humidity", Array("High", "Normal")),
      ("Wind", Array("Weak", "Strong")))
    val (errors, points) = DataUtils.loadLabeledPoints(fullPath, schema, 0, true)

    pointsForNominalValuesOnlyNoMissingCIAtFirstPos(points.toList)
  }

  test("Labeled points from schema, all nominal values, no missing data and class index at middle position") {
    val filePath = "/NominalValuesOnlyNoMissingCIAtMidPos1.csv"
    val fullPath = getClass.getResource(filePath).getFile
    val schema: Array[(String, Array[String])] = Array[(String, Array[String])](
      ("Outlook", Array("Sunny", "Overcast", "Rain")),
      ("Temperature", Array("Hot", "Mild", "Cool")),
      ("PlayTennis", Array("No", "Yes")),
      ("Humidity", Array("High", "Normal")),
      ("Wind", Array("Weak", "Strong")))
    val (errors, points) = DataUtils.loadLabeledPoints(fullPath, schema, 2, true)

    pointsForNominalValuesOnlyNoMissingCIAtMidPos(points.toList)
  }

  test("Labeled points from schema, all nominal values, some missing data and class index at middle position") {
    val filePath = "/NominalValuesOnlySomeMissingCIAtMidPos1.csv"
    val fullPath = getClass.getResource(filePath).getFile
    val schema: Array[(String, Array[String])] = Array[(String, Array[String])](
      ("Outlook", Array("Sunny", "Overcast", "Rain")),
      ("Temperature", Array("Hot", "Mild", "Cool")),
      ("PlayTennis", Array("No", "Yes")),
      ("Humidity", Array("High", "Normal")),
      ("Wind", Array("Weak", "Strong")))
    val (errors, points) = DataUtils.loadLabeledPoints(fullPath, schema, 2, true)

    pointsForNominalValuesOnlySomeMissingCIAtMidPos(points.toList)
  }

  test("schemaSpec") {
    val filePath = "/EmptyFile.csv"
    val fullPath = getClass.getResource(filePath).getFile
    val schema1: Array[(String, Array[String])] = Array[(String, Array[String])](
      ("Col1", Array("F1", "F2", "Rain")))
    val (errors, points) = DataUtils.loadLabeledPoints(fullPath, schema1, 0, true)
    assert(errors.size == 1 && errors.contains("The schema must specify at least two columns."))

    val schema2: Array[(String, Array[String])] = Array[(String, Array[String])](
      ("Col1", Array("F1", "F2", "Rain")),
      ("Col2", Array("F3", "F4")),
      ("Col3", Array("F35", "F46")))
    val (errors1, points1) = DataUtils.loadLabeledPoints(fullPath, schema2, 3, true)
    assert(errors1.size == 1 && errors1.contains("The number of columns are too few."))

    val schema3: Array[(String, Array[String])] = Array[(String, Array[String])](
      ("Col1", Array("F1", "F2", "Rain")),
      ("", Array("F3", "F4")),
      ("Col3", Array("F35", "F46")))
    val (errors2, points2) = DataUtils.loadLabeledPoints(fullPath, schema3, 1, true)
    assert(errors2.size == 1 && errors2.contains("Found empty column names."))

    val schema4: Array[(String, Array[String])] = Array[(String, Array[String])](
      ("Col1", Array("F1", "F2", "Rain")),
      ("Col2", Array("F3", "F4")),
      ("Col2", Array("F3", "F4")),
      ("Col3", Array("F35", "F46")))
    val (errors3, points3) = DataUtils.loadLabeledPoints(fullPath, schema4, 1, true)
    assert(errors3.size == 1 && errors3.contains("Found columns with duplicate names."))

    val schema5: Array[(String, Array[String])] = Array[(String, Array[String])](
      ("Col1", Array("F1", "F2", "Rain")),
      ("Col2", Array("F3", "F4", "F5", "F3", "F5")),
      ("Col3", Array("F3", "F4", "F3")),
      ("Col4", Array("F35", "F46")))
    val (errors4, points4) = DataUtils.loadLabeledPoints(fullPath, schema5, 1, true)
    assert(errors4.size == 1 && errors4.contains("Found category values with duplicate names."))
  }

  test("Labeled points from schema, mixed types, no missing data") {
    val filePath = "/MixedTypesNoMissing1.csv"
    val fullPath = getClass.getResource(filePath).getFile
    val schema: Array[(String, Array[String])] = Array[(String, Array[String])](
      ("F1", Array("M", "F")),
      ("F2", Array.empty),
      ("F3", Array("Student", "Professional", "Teacher")),
      ("F4", Array("Y", "N")))
    val (errors, points) = DataUtils.loadLabeledPoints(fullPath, schema, 3, true)

    pointsForMixedTypeNoMissingData(points.toList)
  }

  test("Labeled points from schema, mixed types, incorrect categorical value, no missing data") {
    val filePath = "/MixedTypesNoMissingIncorrectCategoricalValue.csv"
    val fullPath = getClass.getResource(filePath).getFile
    val schema: Array[(String, Array[String])] = Array[(String, Array[String])](
      ("F1", Array("M", "F")),
      ("F2", Array.empty),
      ("F3", Array("Student", "Professional", "Teacher")),
      ("F4", Array("Y", "N")))
    val (errors, points) = DataUtils.loadLabeledPoints(fullPath, schema, 3, true)

    assert(errors.length == 1 && errors(0) == "1: Cannot recognize token Doctor at category index 2")

    assert(points(0).get.label === 0.0)
    assert(points(0).get.features === Vectors.dense(Array(0.0,23.0,0.0)))
    assert(!points(1).isDefined)
    assert(points(2).get.label === 1.0)
    assert(points(2).get.features === Vectors.dense(Array(0.0,22.0,0.0)))
    assert(points(3).get.label === 1.0)
    assert(points(3).get.features === Vectors.dense(Array(1.0,60.0,2.0)))
  }

  test("Labeled points from schema, mixed types categorical value in numerical column") {
    val filePath = "/MixedTypeNoMissingCategoricalValueInNumericCol.csv"
    val fullPath = getClass.getResource(filePath).getFile
    val schema: Array[(String, Array[String])] = Array[(String, Array[String])](
      ("F1", Array("M", "F")),
      ("F2", Array.empty),
      ("F3", Array("Student", "Professional", "Teacher")),
      ("F4", Array("Y", "N")))
    val (errors, points) = DataUtils.loadLabeledPoints(fullPath, schema, 3, true)

    assert(errors.length == 1 && errors(0) == "2: Found a categorical value in a field that was regarded as numerical so far.")
    assert(points(0).get.label === 0.0)
    assert(points(0).get.features === Vectors.dense(Array(0.0,23.0,0.0)))
    assert(points(1).get.label == 0.0)
    assert(points(1).get.features === Vectors.dense(Array(1.0,41.0,1.0)))
    assert(!points(2).isDefined)
    assert(points(3).get.label === 1.0)
    assert(points(3).get.features === Vectors.dense(Array(1.0,60.0,2.0)))
  }

  test("Labeled points from schema, mixed types numerical value in categorical type column") {
    val filePath = "/MixedTypeNoMissingNumValueInNominalCol.csv"
    val fullPath = getClass.getResource(filePath).getFile
    val schema: Array[(String, Array[String])] = Array[(String, Array[String])](
      ("F1", Array("M", "F")),
      ("F2", Array.empty),
      ("F3", Array("Student", "Professional", "Teacher")),
      ("F4", Array("Y", "N")))
    val (errors, points) = DataUtils.loadLabeledPoints(fullPath, schema, 3, true)

    assert(errors.length == 1 && errors(0) == "1: Found a numerical value in a field that is actually categorical.")
    assert(points(0).get.label === 0.0)
    assert(points(0).get.features === Vectors.dense(Array(0.0,23.0,0.0)))
    assert(!points(1).isDefined)
    assert(points(2).get.label == 1.0)
    assert(points(2).get.features === Vectors.dense(Array(0.0,12.0,0.0)))
    assert(points(3).get.label === 1.0)
    assert(points(3).get.features === Vectors.dense(Array(1.0,60.0,2.0)))
  }

  test("RDD creation, all nominal values, no missing data") {
    val filePath = "/NominalValuesOnlyNoMissing1.csv"
    val fullPath = getClass.getResource(filePath).getFile
    val schema: Array[(String, Array[String])] = Array[(String, Array[String])](("Outlook", Array("Sunny", "Overcast", "Rain")),
      ("Temperature", Array("Hot", "Mild", "Cool")),
      ("Humidity", Array("High", "Normal")),
      ("Wind", Array("Weak", "Strong")),
      ("PlayTennis", Array("No", "Yes")))

    val rdd = sc.textFile(fullPath, sc.defaultMinPartitions)

    val (errors, points) = DataUtils.loadLabeledPointsRDD(sc, rdd, schema, 4, true)

    assert(errors.isEmpty)
    pointsForNominalValuesOnlyNoMissing(points.collect().toList)
  }

  test("RDD, Labeled points from schema, mixed types, incorrect categorical value, no missing data") {
    val filePath = "/MixedTypesNoMissingIncorrectCategoricalValue.csv"
    val fullPath = getClass.getResource(filePath).getFile
    val schema: Array[(String, Array[String])] = Array[(String, Array[String])](
      ("F1", Array("M", "F")),
      ("F2", Array.empty),
      ("F3", Array("Student", "Professional", "Teacher")),
      ("F4", Array("Y", "N")))

    val rdd = sc.textFile(fullPath, sc.defaultMinPartitions)
    val (errors, pointsRDD) = DataUtils.loadLabeledPointsRDD(sc, rdd, schema, 3, true)
    val points = pointsRDD.collect()
    assert(points(0).get.label === 0.0)
    assert(points(0).get.features === Vectors.dense(Array(0.0,23.0,0.0)))
    assert(!points(1).isDefined)
    assert(points(2).get.label === 1.0)
    assert(points(2).get.features === Vectors.dense(Array(0.0,22.0,0.0)))
    assert(points(3).get.label === 1.0)
    assert(points(3).get.features === Vectors.dense(Array(1.0,60.0,2.0)))
  }
}
