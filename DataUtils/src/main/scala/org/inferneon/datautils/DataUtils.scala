package org.inferneon.datautils

import java.io.FileNotFoundException
import java.nio.charset.MalformedInputException
import java.nio.file.{Files, Paths}

import org.apache.spark.SparkContext
import org.apache.spark.mllib.linalg.Vectors
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.rdd.RDD

import scala.collection.mutable
import scala.collection.mutable.ArrayBuffer
import scala.io.{BufferedSource, Source}

/**
  * Helper methods to infer schema given a CSV file or load LabeledPoints given data and a schema.
  * There are three primary helper methods available:
  *
  * inferSchema : Infers the schema from given CSV data and also returns a list of labelled points if a schema could
  *               be determined.

  * loadLabeledPoints: Returns a list of LabeledPoints given the CSV data and a schema defining the data.
  *
  * loadLabeledPointsRDD: Returns a RDD of LabeledPoints given the CSV data as an RDD and a schema defining the data.
  *
  */

object DataUtils {
  val pattern = """(("\s*[^"]*?")\s*,)|(\s*([^,|^"]*?)(,|\n))""".r

  object FeatureType extends Enumeration {
    val Unknown, Categorical, Numerical = Value
  }

  /**
    * Given the path of a comma-separated file (CSV) file which contains data with header information, this method
    * attempts to infer the schema from the data. The schema is inferred based on the following assumptions:
    *
    * 1. The first row of the file represents the header information. Each comma-delimited field in this header
    * is considered to denote the name of the feature.
    * 2. The data can contain missing information. This can either show up in the file as empty strings (or
    * whitespaces) or can be represented by a question mark ( ? ).
    * 3. The data can contain commas; however in such cases, that data item must be enclosed in double-quotes.
    * E.g.: "Hello, World". Moreover, escaped strings are not handled; in such cases, behaviour is undefined.
    * 4. The data can consist of both categorical (nominal) features as well as numerical data.
    *
    * If a schema is successfully inferred (or whatever best is inferred), the schema as well as a list of labeled
    * points are returned. A labeled point is returned for each row in the data with a the label value and a
    * dense or sparse Vector. For a categorical feature, the corresponding entry in the Vector is a zero-indexed
    * integer that corresponds to the index of that value int that categorical feature. For a numerical feature,
    * the entry in the Vector will be the number itself. If a schema cannot be inferred, empty values for both the
    * schema as well as the list of labeled points will be returned. Please read the description of the return
    * value for further information.
    *
    * @param file Path of the file representing the input file.
    * @param classIndex  The class or target index indicating which column represents the class. This must be
    *                    a zero-indexed integer and must be lesser than the number of columns in the header.
    * @param caseSensitive If this is specified as "true" categorical values will be checked in a case-
    *                      sensitive manner, and case-insensitive otherwise.
    * @return A three-tuple value which represents the following:
    *         1. The first element of the tuple is an array containing a descriptive Strings of errors. Errors
    *         might be found due to inconsistency in the data. If no errors are found, this array buffer
    *         will be empty. Each error description starts with the line number at which the error was seen.
    *         The line numbers are 0-based indexed. It is also possible that even when a reasonable schema was inferred,
    *         this array may not be empty.
    *         2. The second element of the tuple is an array representing the schema. Each element of this
    *         array identifies a feature at a corresponding column in the data. A feature is again represented
    *         by a 2-tuple. The first element of the feature tuple is the name of the feature as found in the
    *         first row (header) of the data. The second element of the feature tuple is an array of categorical
    *         values for that feature as found in the data. If a feature is inferred to be numerical,
    *         this corresponding array will be empty.
    *         3. The third element of the tuple is the list of labeled points. They are wrapped in an Optional
    *         value to address the possibility of inconsistency of the data at the corresponding row. If a row has
    *         missing data or some elements of a row could not be inferred, a LabeledPoint is created with a
    *         sparse Vector for that row; else if a row has data that could be inferred correctly for all fields, a
    *         LabeledPoint is created with a sparse Vector for that row.
    */

  def inferSchema(file: String, classIndex: Int, caseSensitive: Boolean): (ArrayBuffer[String], Array[(String, Array[String])],
    List[Option[LabeledPoint]]) = {

    def infer(columnNames: Array[String], linesWithoutHeader: Array[String])
    : (ArrayBuffer[String], Array[(String, Array[String])], List[Option[LabeledPoint]]) = {
      val allFields = new mutable.HashMap[Int, ArrayBuffer[String]]()
      var lineCount = 0
      var missingLineCount = 0
      val fieldsAndInferredTypes = new mutable.HashMap[Int, FeatureType.Value]
      val numColumns = columnNames.length
      Range(0, numColumns).foreach(fieldsAndInferredTypes(_) = FeatureType.Unknown)

      // Iterate over the lines. For each line a labeled point is created. The schema is also inferred as part of this
      // iteration.
      val errorsAndLabeledPoints  = linesWithoutHeader map { lineInFile =>
        var labeledPoint: Option[LabeledPoint] = None
        var errors = ArrayBuffer[String]()
        val line = lineInFile + "\n"
        if (line.trim.length == 0) {
          missingLineCount += 1
        }
        else {
          val result = createLabeledPointFromLine(line, lineCount, numColumns,
            classIndex, allFields, fieldsAndInferredTypes, caseSensitive, true)
          errors = result._1
          labeledPoint = result._2
        }
        lineCount += 1
        (errors, labeledPoint)
      }

      val (errors, labeledPoints) = errorsAndLabeledPoints.unzip
      val allErrors = errors.flatMap(err => err).to[ArrayBuffer]
      if (!labeledPoints.exists(_.isDefined)) {
        val errs = ArrayBuffer[String]("No data found.")
        (errs, Array.empty, List.empty)
      }
      else {
        var errorsIdentifyingField = new ArrayBuffer[String]()
        val namesAndTypes = Range(0, numColumns) map { idx =>
          val fieldsInBuffer = allFields.get(idx)
          if (fieldsInBuffer.isEmpty) {
            errorsIdentifyingField += "Unable to identify field at column index " + idx
            ("", Array[String]())
          }
          else {
            val fields = fieldsInBuffer.get.toArray
            (columnNames(idx), fields)
          }
        }
        if (errorsIdentifyingField.nonEmpty) {
          allErrors ++= errorsIdentifyingField
          (allErrors, Array.empty, List.empty)
        }
        else {
          (allErrors, namesAndTypes.toArray, labeledPoints.toList)
        }
      }
    }

    val errors = new ArrayBuffer[String]()
    var schema : Array[(String, Array[String])] = Array.empty
    var labeledPoint : List[Option[LabeledPoint]] = List.empty
    var source: Option[BufferedSource] = None
    try {
      source = Some(Source.fromFile(file))
      val lines = source.get.getLines().toArray
      var columnNames = Array[String]()
      var numColumns = 0
      var linesWithoutHeader = Array[String]()
      val indexOfFirstNonEmptyLine = getIndexOfFirstNonEmptyLine(lines)

      require(!lines.isEmpty, "File is empty, cannot infer the schema.")
      val splits = lines.splitAt(indexOfFirstNonEmptyLine + 1)
      val first = splits._1
      linesWithoutHeader = splits._2
      require(linesWithoutHeader.length > 0, "No data found.")

      val firstLine = first.head + "\n"
      columnNames = pattern.findAllIn(firstLine).toArray.map(line => line.dropRight(1).trim)
      if (columnNames.find(_ == "").isDefined) {
        errors += "Found empty column names."
      }
      numColumns = columnNames.length
      if (numColumns < 2) {
        errors += "The data must contain at least two columns."
      }
      if (classIndex >= numColumns) {
        errors += "There are too few columns for the class index specified."
      }
      if (errors.isEmpty) {
        val result = infer(columnNames, linesWithoutHeader)
        errors ++= result._1
        schema = result._2
        labeledPoint = result._3
      }
    }
    catch {
      case fne : FileNotFoundException =>  errors += "File not found."
      case mie : MalformedInputException =>  errors += "Invalid file. Please use a valid CSV file."
      case ex: Exception =>   errors += "Unknown error. Please use a valid CSV file."
    }
    finally{
      if(source.isDefined){
        source.get.close
      }
    }
    (errors, schema, labeledPoint)
  }

  /**
    * Given the path of a comma-separated file (CSV) file which contains data and a suggested schema, this method
    * attempts to create a list of LabeledPoint objects. The following assumptions are made:
    *
    * 1. There is NO header information; the schema contains all the information needed for generating the LabeledPoint
    * objects. The ordering of the columns correspond to the one specified in the schema.
    * 2. The data can contain missing information. This can either show up in the file as empty strings (or
    * whitespaces) or can be represented by a question mark ( ? ).
    * 3. The data can contain commas; however in such cases, that data item must be enclosed in double-quotes.
    * E.g.: "Hello, World". Moreover, escaped strings are not handled; in such cases, behaviour is undefined.
    * 4. The data can consist of both categorical (nominal) features as well as numerical data.
    *
    * A labeled point is returned for each row in the data with a the label value and a dense or sparse Vector. For a
    * categorical feature, the corresponding entry in the Vector is a zero-indexed integer that corresponds to the
    * index of that value in that categorical feature. For a numerical feature, the entry in the Vector will be the
    * number itself.
    *
    * @param path Path of the file representing the input file.
    * @param schema  An array of tuples representing the schema. Each element of this array should identify a feature
    *         at a corresponding column in the data. A feature is again represented by a 2-tuple. The first element
    *         of the feature tuple should denote the name of the feature. The second element of the feature tuple
    *         should be an array of categorical values for that feature as found in the data. If a feature is numerical,
    *         this corresponding array should be empty.
    * @param classIndex  The class or target index indicating which column represents the class. This must be
    *                    a zero-based indexed integer and must be lesser than the number of features suggested in the
    *                    schema.
    * @param caseSensitive If this is specified as "true" categorical values will be checked in a case-
    *                      sensitive manner, and case-insensitive otherwise.
    * @return A two-tuple value which represents the following:
    *         1. The first element of the tuple is an array containing a descriptive Strings of errors. Errors
    *         might be found due to inconsistency in the data. If no errors are found, this array buffer
    *         will be empty. Each error description starts with the line number at which the error was seen.
    *         The line numbers are 0-indexed.
    *         2. The second element of the tuple is the list of labeled points. They are wrapped in an Optional
    *         value to address the possibility of inconsistency of the data at the corresponding row. If a row has
    *         missing data or some elements of a row could not be inferred, a LabeledPoint is created with a
    *         sparse Vector for that row; else if a row has data that could be inferred correctly for all fields, a
    *         LabeledPoint is created with a dense Vector for that row.
    */

  def loadLabeledPoints(path : String, schema : Array[(String, Array[String])], classIndex : Int, caseSensitive: Boolean):
  (ArrayBuffer[String], Array[Option[LabeledPoint]]) ={
    val errors = ArrayBuffer[String]()
    var source: Option[BufferedSource] = None
    val errorInSchema = validateSchema(schema, classIndex)
    if(errorInSchema.isDefined){
      // Found an error in the specified schema
      errors += errorInSchema.get
      (errors, Array.empty)
    }
    else {
      var lineCount = 0
      var missingLineCount = 0
      val numColumns = schema.length
      val (allFields, fieldsAndTypes) = initializeFieldsAndTypes(schema, numColumns)

      val errsAndLabeledPoints: Array[(ArrayBuffer[String], Option[LabeledPoint])] =
        try {
          source = Some(Source.fromFile(path))
          val lines = source.get.getLines().toArray
          require(!lines.isEmpty, "File is empty, cannot infer the schema.")
          lines map { line =>
            val errsAndPoint = {
              val lineInFile = line + "\n"
              if (lineInFile.trim.length == 0) {
                missingLineCount += 1
                (ArrayBuffer[String](), None)
              }
              else {
                createLabeledPointFromLine(lineInFile, lineCount, numColumns, classIndex,
                  allFields, fieldsAndTypes, caseSensitive, false)
              }
            }
            lineCount += 1
            errsAndPoint
          }
        }
        catch {
          case mie: MalformedInputException =>
            errors += "Invalid file. Please use a valid CSV file."
            Array.empty
          case _: Throwable =>
            errors += "Unknown error."
            Array.empty
        }
        finally {
          if (source.isDefined) {
            source.get.close
          }
        }

      val result = errsAndLabeledPoints.unzip
      val allErrs = result._1.flatten
      val labeledPoints = result._2.toArray
      (allErrs.to[ArrayBuffer], labeledPoints)
    }
  }

  /**
    * Given RDD of String representing CSV (comma-separated values) data and a suggested schema, this method
    * attempts to create a RDD of LabeledPoint objects. The following assumptions are made:
    *
    * 1. The ordering of the columns correspond to the one specified in the schema.
    * 2. The data can contain missing information. This can either show up in the file as empty strings (or
    * whitespaces) or can be represented by a question mark ( ? ).
    * 3. The data can contain commas; however in such cases, that data item must be enclosed in double-quotes.
    * E.g.: "Hello, World". Moreover, escaped strings are not handled; in such cases, behaviour is undefined.
    * 4. The data can consist of both categorical (nominal) features as well as numerical data.
    *
    * A labeled point is returned for each row in the data with a the label value and a dense or sparse Vector. For a
    * categorical feature, the corresponding entry in the Vector is a zero-based indexed integer that corresponds to the
    * index of that value in that categorical feature. For a numerical feature, the entry in the Vector will be the
    * number itself.
    *
    * @param sc The SparkContext object
    * @param rdd The RDD of Strings which represents the data.
    * @param schema  An array of tuples representing the schema. Each element of this array should identify a feature
    *         at a corresponding column in the data. A feature is again represented by a 2-tuple. The first element
    *         of the feature tuple should denote the name of the feature. The second element of the feature tuple
    *         should be an array of categorical values for that feature as found in the data. If a feature is numerical,
    *         this corresponding array should be empty.
    * @param classIndex  The class or target index indicating which column represents the class. This must be
    *                    a zero-based indexed integer and must be lesser than the number of features suggested in the
    *                    schema.
    * @param caseSensitive If this is specified as "true" categorical values will be checked in a case-
    *                      sensitive manner, and case-insensitive otherwise.
    * @return A two-tuple value which represents the following:
    *         1. The first element of the tuple is a RDD of descriptive Strings of errors. Errors
    *         might be found due to inconsistency in the data. If no errors are found, this RDD
    *         will be empty. Each error description starts with the line number at which the error was seen.
    *         The line numbers are 0-indexed.
    *         2. The second element of the tuple is the resulting RDD of labeled points. They are wrapped in an Optional
    *         value to address the possibility of inconsistency of the data at the corresponding row. If a row has
    *         missing data or some elements of a row could not be inferred, a LabeledPoint is created with a
    *         sparse Vector for that row; else if a row has data that could be inferred correctly for all fields, a
    *         LabeledPoint is created with a dense Vector for that row.
    */

  def loadLabeledPointsRDD(sc: SparkContext, rdd: RDD[String], path : String, schema : Array[(String, Array[String])],
                           classIndex : Int, caseSensitive: Boolean):
  (ArrayBuffer[String], RDD[Option[LabeledPoint]]) ={
    //require(Files.exists(Paths.get(path)), "Cannot find input file.")
    val errors = ArrayBuffer[String]()
    val errorInSchema = validateSchema(schema, classIndex)
    if(errorInSchema.isDefined){
      // Found an error in the specified schema
      errors += errorInSchema.get
      (errors, sc.emptyRDD )
    }
    else {
      var lineCount = 0L
      var missingLineCount = 0
      val numColumns = schema.length
      val (allFields, fieldsAndTypes) = initializeFieldsAndTypes(schema, numColumns)

      val zipped = rdd.zipWithIndex()
      val errsAndlabeledPoints: RDD[(ArrayBuffer[String], Option[LabeledPoint])] = zipped map {tuple =>
        val line = tuple._1
        lineCount =  tuple._2
        val point = {
          val lineInFile = line + "\n"
          if (lineInFile.trim.length == 0) {
            missingLineCount += 1
            (new ArrayBuffer[String](), None)
          }
          else {
            createLabeledPointFromLine(lineInFile, lineCount, numColumns, classIndex,
              allFields, fieldsAndTypes, caseSensitive, false)
          }
        }
        point
      }
      val result = errsAndlabeledPoints.groupByKey()
      val allErrs = result.keys.flatMap(errs => errs)
      val labeledPoints = errsAndlabeledPoints.values

      val errsBuff = new ArrayBuffer[String]()
      if(allErrs.count() < 100000){
        // Not too many errors
        errsBuff ++= allErrs.collect()
      }
      (errsBuff, labeledPoints)
    }
  }

  private def createLabeledPointFromLine(line : String, lineCount : Long, numColumns: Int,
                                         classIndex: Int, allFields: mutable.HashMap[Int, ArrayBuffer[String]],
                                         fieldsAndInferredTypes: mutable.HashMap[Int, FeatureType.Value],
                                         caseSensitive: Boolean, isInferenceContext: Boolean): (ArrayBuffer[String], Option[LabeledPoint])  = {
    val tokens = pattern.findAllIn(line).toArray
    var errors = ArrayBuffer[String]()
    if (tokens.length > numColumns) {
      errors += ": Number of fields (" + tokens.length + ") is greater than the number of columns: (" + numColumns + ")"
      (errors, None)
    }
    else {
      val indexesWithMissingFields: ArrayBuffer[Integer] = new ArrayBuffer[Integer]()
      val (fatalError, errors, label, indexAndValues) = getLabelAndIndexFeatureValues(tokens, classIndex,
        lineCount, indexesWithMissingFields, numColumns, allFields, fieldsAndInferredTypes, caseSensitive, isInferenceContext)
      if(!fatalError) {
        if (label.isEmpty) {
          errors += lineCount + ": Unable to identify the label."
          (errors, None)
        }
        else {
          val numMissing = indexesWithMissingFields.length + (numColumns - tokens.length)
          (errors, Some(getLabeledPoint(label.get, numMissing, indexAndValues, numColumns - 1)))
        }
      }
      else{
        (errors, None)
      }
    }
  }

  private def initializeFieldsAndTypes(schema : Array[(String, Array[String])],  numColumns : Int):
  (mutable.HashMap[Int, ArrayBuffer[String]], mutable.HashMap[Int, FeatureType.Value]) ={
    val fieldsAndTypes = new mutable.HashMap[Int, FeatureType.Value]
    val allFields = new mutable.HashMap[Int, ArrayBuffer[String]]()
    Range(0, numColumns).foreach { idx =>
      val newBuffer = new ArrayBuffer[String]()
      schema(idx)._2 foreach {
        newBuffer += _
      }
      allFields(idx) = newBuffer
    }
    Range(0, numColumns).foreach(idx =>
      fieldsAndTypes(idx) = {
        if (schema(idx)._2.length > 0) FeatureType.Categorical else FeatureType.Numerical
      })

    (allFields, fieldsAndTypes)
  }

  private def getLabelAndIndexFeatureValues(tokens : Array[String], classIndex : Int,
                                            lineCount : Long, indexesWithMissingFields: ArrayBuffer[Integer],
                                            numColumns: Int, allFields: mutable.HashMap[Int, ArrayBuffer[String]],
                                            fieldsAndFeatureTypes: mutable.HashMap[Int, FeatureType.Value], caseSensitive: Boolean,
                                            isInferenceContext: Boolean)
  : (Boolean, ArrayBuffer[String], Option[Double], Array[Option[(Int, Option[Double])]]) = {
    var tokenIndex = 0
    var label: Option[Double] = None
    var fatalError : Boolean = false
    val numTokens = tokens.length
    val indexAndValues = new Array[Option[(Int, Option[Double])]](numTokens)

    val errors = ArrayBuffer[String]()

    while(tokenIndex < numTokens && !fatalError) {
      val field = tokens(tokenIndex)
      var numericValue: Option[Double] = None
      var errorGettingNumericValue: Option[String] = None
      val token = field.dropRight(1).trim

      if (token.length == 0 || token == "?") {
        if (tokenIndex == classIndex) {
          errors += lineCount + ": Number of fields (" + tokens.length + ") does not match the number of columns: (" + numColumns + ")"
        }
        else indexesWithMissingFields += tokenIndex
      }
      else {
        val result = getNumericValue(token, tokenIndex, lineCount, allFields, fieldsAndFeatureTypes, caseSensitive, isInferenceContext )
        errorGettingNumericValue = result._1
        numericValue = result._2
        if (tokenIndex == classIndex) {
          label = numericValue
        }
      }

      if (errorGettingNumericValue.isDefined) {
        if(isInferenceContext) {
          indexesWithMissingFields += tokenIndex
        }
        else{
          fatalError = true
        }
        errors += errorGettingNumericValue.get
        indexAndValues(tokenIndex) = None
      }
      else {
        var index = tokenIndex
        if (index == classIndex) {
          // No matter where the class index is, we treat it as the last column
          // to align feature indexes in order
          index = numColumns - 1
        }
        else if (index > classIndex) {
          index -= 1
        }
        indexAndValues(tokenIndex) =  Some((index, numericValue))
      }
      tokenIndex += 1
    }

    (fatalError, errors, label, indexAndValues)
  }

  private def getNumericValue(token: String, tokenIndex: Int, lineCount: Long,
                              allFields: mutable.HashMap[Int, ArrayBuffer[String]],
                              fieldsAndFeatureTypes: mutable.HashMap[Int, FeatureType.Value],
                              caseSensitive: Boolean, isInferenceContext: Boolean)
  : (Option[String], Option[Double]) = {

    var error: Option[String] = None
    val featureType = fieldsAndFeatureTypes.get(tokenIndex).get
    val isNumericField = isNumeric(token)
    if (isNumericField) {
      // Found a number.
      val v = token.toDouble
      if(featureType == FeatureType.Unknown){
        // Was hitherto not inferred; we now regard it as numerical
        fieldsAndFeatureTypes(tokenIndex) = FeatureType.Numerical
        allFields(tokenIndex) = new ArrayBuffer[String]()
        (error, Some(v))
      }
      else if(featureType == FeatureType.Categorical) {
        val categoryFields = allFields.get(tokenIndex).get
        if (!isInferenceContext) {
          val idx = categoryFields.indexOf(token)
          if (idx < 0) {
            error = Some(lineCount + ": Found a numerical value in a field that is actually categorical.")
            (error, None)
          }
          else{
            (error, Some(idx))
          }
        }
        else {
          // Was hitherto inferred as categorical; we will now have to regard this as categorical even
          // thought it is a number; we add this as a warning.
          // TODO Log this
          //error = Some(lineCount + ": Warning - found a numerical field where a categorical value is expected at column, now regarding "  + token + " as a categorical value.")

          val idx = categoryFields.indexOf(token)
          if (idx < 0) {
            categoryFields += token
            (error, Some(categoryFields.size - 1))
          }
          else {
            (error, Some(idx))
          }
        }
      }
      else
      {
        (error, Some(v))
      }
    }
    else {
      // Found a categorical field
      if(featureType == FeatureType.Unknown){
        // Was hitherto not inferred; we now regard it as categorical
        fieldsAndFeatureTypes(tokenIndex) = FeatureType.Categorical
        val newArray = new ArrayBuffer[String]()
        allFields(tokenIndex) = newArray
        newArray += token
        (error, Some(0))
      }
      else if(featureType == FeatureType.Numerical){
        // This was a numerical feature earlier; now we dont know how to handle this. Flag
        // this as an error
        error = Some(lineCount + ": Found a categorical value in a field that was regarded as numerical so far.")
        (error, None)
      }
      else {
        val values = allFields.get(tokenIndex)
        val valuesArray = values match {
          case Some(array) => array
          case None =>
            val newArray = new ArrayBuffer[String]()
            allFields(tokenIndex) = newArray
            newArray
        }

        var idx = -1
        if(caseSensitive) {
          idx = valuesArray.indexOf(token)
        }
        else{
          val lowercaseToken = token.toLowerCase
          idx = valuesArray.indexWhere(_.toLowerCase == lowercaseToken)
        }
        if (idx < 0) {
          if(!isInferenceContext){
            // Not part of the schema
            error = Some(lineCount + ": Cannot recognize token " + token + " at category index " + tokenIndex)
            (error, None)
          }
          else {
            valuesArray += token
            (error, Some(valuesArray.size - 1))
          }
        }
        else {
          (error, Some(idx))
        }
      }
    }
  }

  private def getIndexOfFirstNonEmptyLine(lines: Array[String]): Int = {
    var index = 0
    var foundFirstNonEmptyLines = false
    while (index < lines.length && !foundFirstNonEmptyLines) {
      val line = lines(index)
      if (line.trim.length > 0) {
        foundFirstNonEmptyLines = true
      }
      index += 1
    }
    index -1
  }

  private def getLabeledPoint(label: Double, numMissing: Int, indexAndValues: Array[Option[(Int, Option[Double])]], size: Int): LabeledPoint = {
    if (numMissing > 0) {
      val idxAndVals: Array[(Int, Double)] = indexAndValues
        .filter(idxAndVal => idxAndVal.isDefined && idxAndVal.get._1 != size && idxAndVal.get._2.isDefined)
        .map(idxAndVal => (idxAndVal.get._1, idxAndVal.get._2.get))
      new LabeledPoint(label, Vectors.sparse(size, idxAndVals))
    }
    else {
      val allValues = indexAndValues
        .filter(idxAndVal => idxAndVal.isDefined && idxAndVal.get._1 != size)
        .map(idxAndVal => idxAndVal.get._2.get)
      new LabeledPoint(label, Vectors.dense(allValues))
    }
  }

  private def validateSchema(schema : Array[(String, Array[String])], classIndex : Int) : Option[String]  = {
    val numColumns = schema.length

    val error: Option[String] = {
      if(numColumns < 2)
        Some("The schema must specify at least two columns.")
      else if(classIndex >= numColumns)
        Some ("The number of columns are too few.")
      else if(schema.map(_._1).exists(_ == "")){
        Some("Found empty column names.")
      }
      else if(schema.map(_._1).distinct.length < schema.length){
        Some("Found columns with duplicate names.")
      }
      else if(schema.find(fields => {fields._2.distinct.length < fields._2.length} ).isDefined) {
        Some("Found category values with duplicate names.")
      }
      else{
        None
      }
    }
    error
  }

  private def isNumeric(str: String): Boolean = {
    !throwsNumberFormatException(str.toLong) || !throwsNumberFormatException(str.toDouble)
  }

  private def throwsNumberFormatException(f: => Any): Boolean = {
    try {
      f; false
    } catch {
      case e: NumberFormatException => true
    }
  }
}