package org.inferneon.bayesnet.core

import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.rdd.RDD
import org.inferneon.bayesnet.hillclimber.ScoringType

import scala.collection.mutable.ArrayBuffer
import scala.tools.nsc.util._
import scala.language.postfixOps

/**
  * This trait denotes a bunch of useful methods that are needed to implement a Bayesian belief network algorithm. This
  * is inherited by algorithms that learn the network structure, like the HillClimber and the SimulatedAnnealing
  * algorithms.
  */
trait BayesianNetAlgorithm {

  var scoringType: ScoringType.Value


  /**
    * Initializes the network depending upon the "isCausal" flag. If this is set to true, we initialize the network
    * structure with all feature nodes having directed edges to the label node; else the network is intialized with
    * edges from the label node to all other feature nodes.
    *
    * @return Initial network
    */
  def initializeNetwork(schema : Array[(String, Array[String])], classIndex : Int, isCausal : Boolean): BayesianBeliefNetwork = {
    val network = BayesianBeliefNetwork.emptyNetwork()
    val numFeatures = schema.length
    validateSchema(schema, classIndex, numFeatures)
    var classNode: Option[Node] = None

    val allNodes = Range(0, numFeatures) map { featureIndex =>
      val node = {
        if (featureIndex != classIndex) {
          Node.emptyNode(featureIndex)
        }
        else {
          val labelNode = Node.emptyNode(featureIndex, isLabel = true)
          classNode = Some(labelNode)
          labelNode
        }
      }
      network.add(node)
      node
    }

    network.allNodes = allNodes.toList
    allNodes foreach { node =>
      if (!node.isLabel) {
        if (isCausal) {
          network.addEdge(node, classNode.get)
        }
        else {
          network.addEdge(classNode.get, node)
        }
      }
    }

    network
  }

  /**
    * Validates the schema for consistency of features and feature name spaces.
    * @param schema                            The schema to be validated.
    * @param classIndex                        The class or target index.
    * @param numFeatures                       The number of features.
    */
  private def validateSchema(schema : Array[(String, Array[String])], classIndex: Int, numFeatures: Int): Unit = {
    require(schema.length > 2)
    require(classIndex > 0 && classIndex < schema.length)
    val featureNames = HashSet[String]()
    schema foreach { feature =>
      val name = feature._1.trim

      require(!featureNames.contains(name), "Error in schema: Duplicate feature name: " + name)
      require(!name.isEmpty, "Error in schema: Feature must have a valid name")
      featureNames.addEntry(name)

      val featureValues = HashSet[String]()
      val values = feature._2
      require(values.length != 1)
      values foreach { valueOfFeature =>
        val value = valueOfFeature.trim
        require(!featureValues.contains(value), "Error in schema: Feature " + name + "= has duplicate value: " + value)
        require(!value.isEmpty, "Error in schema: Feature " + name +  "= has empty values")
        featureValues.addEntry(value)
      }
    }
  }

  /**
    * Counts the distribution of values of a categorical feature. Again, instead of multiple passes over the data for
    * each feature, we use mapPartitions to get the counts per partition and then aggregate them.
    *
    * @param nodes    All nodes in the network.
    * @param rdd      The data in the the form of a RDD of LabeledPoints.
    * @return         A mapping of each feature (node) index and corresponding counts of values.
    */

  def getCategoricalCountsPerFeature(schema : Array[(String, Array[String])], nodes:List[Node], rdd : RDD[LabeledPoint]) : Map[Int, Array[Int]] = {

    val numFeatures = nodes.length
    val countsPerPartion = rdd mapPartitions { labeledPoints =>
      val counts = new Array[Array[Int]](numFeatures)
      Range(0, numFeatures) foreach { featureIndex =>
        counts(featureIndex) = Array.fill(schema(featureIndex)._2.length)(0)
      }

      labeledPoints foreach  { labeledPoint =>
        val features = labeledPoint.features.toDense
        val labelIndex = labeledPoint.label.toInt
        Range(0, numFeatures) foreach { featureIndex =>
          val node = nodes(featureIndex)
          if(!node.isLabel) {
            val valueIndex = features(featureIndex).toInt
            counts(featureIndex)(valueIndex) = counts(featureIndex)(valueIndex) + 1
          }
          else{
            counts(featureIndex)(labelIndex) = counts(featureIndex)(labelIndex) + 1
          }
        }
      }

      val fIndexCounts = new ArrayBuffer[(Int, Array[Int])]()
      var featureIndex = 0
      while(featureIndex < counts.length){
        fIndexCounts += ((featureIndex, counts(featureIndex)))
        featureIndex = featureIndex + 1
      }

      fIndexCounts.iterator
    } collect()

    countsPerPartion.groupBy(_._1).map {ctsForFeature =>
      (ctsForFeature._1, ctsForFeature._2 map {ctsTuple =>  ctsTuple._2})
    } map ( cts =>  (cts._1, getColumnCounts(cts._2)) )
  }

  private def getColumnCounts(data : Array[Array[Int]]): Array[Int] ={
    data reduce {(row1, row2) =>
      var colIndex = 0
      row2 foreach {count =>
        row1(colIndex) = row1(colIndex) + count
        colIndex = colIndex + 1
      }
      row1
    }
  }

  def computeScore(rdd: RDD[LabeledPoint], network: BayesianBeliefNetwork, featureValuesCounts: Map[Int, Array[Int]], featureIndex : Int): Double = {
    val node = network.allNodes(featureIndex)
    if(network.getParents(node).isEmpty){ // Just return the score for this node with no parent
      scoreBasedOnCounts(Array[Array[Int]](featureValuesCounts(featureIndex)))
    }
    else{
      getScoreForNodeWithParents(rdd, network, node)
    }
  }

  def getScoreForNodeWithParents(rdd: RDD[LabeledPoint], network: BayesianBeliefNetwork, node : Node) : Double = {
    val parents = network.getParents(node).get
    val distribution = rdd map { labeledPoint =>
      val valuesCombination = getCombinationsFromLabeledPoint(node, parents, labeledPoint)
      (valuesCombination, 1) } reduceByKey { _ + _ } map { case (vc, counts) =>
      (vc.featureIndexAndValue,(vc.valueOfDependentFeature, counts))
    } groupByKey() collect

    scoreBasedOnCounts(distribution map {tuple => tuple._2.toArray map {vAndCs => vAndCs._2}})
  }

  /**
    * Computes the score based on the distribution (frequencies) of occurrences of feature values. The score is computed
    * depending upon the type of the scoring method specified.
    *
    * @param counts    Distribution or frequency of occurrences of feature values, typically across parent nodes.
    * @return          Computed score
    */
  def scoreBasedOnCounts(counts: Array[Array[Int]]): Double = {
    var score = 0.0
    if(scoringType == ScoringType.ENTROPY || scoringType == ScoringType.MDL || scoringType == ScoringType.AIC) {
      counts foreach { cts =>
        val total = cts.sum
        val tempScores = cts map { count => if (count == 0) 0 else {count.toDouble * Math.log(count.toDouble / total.toDouble)}}
        score = score + tempScores.sum
      }
    }
    else{
      // TODO Score for other scoring types.
    }
    score
  }

  private[bayesnet] def getCombinationsFromLabeledPoint(node: Node, parents: List[Node], labeledPoint: LabeledPoint) : ValuesCombination = {
    val featureIndexesAndValues = for {parent <- parents} yield  {if(parent.isLabel){
      (parent.id, labeledPoint.label.toInt)
    }
    else{
      (parent.id, labeledPoint.features(parent.id).toInt)
    }
    }

    val nodeValue = if(node.isLabel){
      labeledPoint.label.toInt
    }
    else{
      labeledPoint.features(node.id).toInt
    }

    val buffer = new ArrayBuffer[(Int, Int)]()
    featureIndexesAndValues.copyToBuffer(buffer)
    ValuesCombination(buffer, nodeValue)
  }

  /**
    * Computes the joint conditional probability distribution for each feature node in the DAG. Instead of multiple passes
    * over the data, we use mapPartitions to determine the distribution for all nodes in a given partition and aggregate
    * them to determine the final distribution.
    *
    * @param network   The network
    * @param rdd       Data in the form of RDD[LabeledPoint]
    * @return          A mapping of node id and corresponding conditional probability table.
    */
  private[bayesnet] def computeCPTs(schema : Array[(String, Array[String])],
                                    network: BayesianBeliefNetwork,
                                    rdd:RDD[LabeledPoint],
                                    prior: Double): Array[(Int, ConditionalProbabilityTable)] ={
    val distributionsForNodes = rdd mapPartitions { labeledPoints =>
      val labeledPointsArray = labeledPoints.toArray
      val allNodes = network.allNodes
      val distributionForNodeInPartition = allNodes map { node =>
        val parents = if(network.getParents(node).isDefined) network.getParents(node).get else List[Node]()
        val vcs = labeledPointsArray map { labeledPoint =>
          val valuesCombination = getCombinationsFromLabeledPoint(node, parents, labeledPoint)
          (valuesCombination, 1)
        }

        val groupedByVcs = vcs groupBy(value => value._1)
        val counts = groupedByVcs map {case (k, v) => (node, k, v.length)}
        counts
      }
      distributionForNodeInPartition.iterator
    }

    val distributionsGroupedByNodes = distributionsForNodes
      .flatMap(element => element map {item => item}) groupBy(element => element._1) collect()
    val distributionsGroupedByNodesAndCombinations = distributionsGroupedByNodes map
      {case (node, valueCombinations) =>  (node, valueCombinations groupBy(item => item._2)) }

    val nodeVCAndCounts = distributionsGroupedByNodesAndCombinations map { element => {element._2 map
      { case (vc, distsInPartitions) =>
        val countInPartitions = distsInPartitions map {case (node, vc, count) => count}
        val total = (0 /: countInPartitions) ((count1, count2) => count1 + count2)
        (element._1, vc, total)
      }
    }
    }

    // We have the overall distributions, lets compute the distributions per each feature (node) value.
    val nodeAndCPTs = nodeVCAndCounts map { distributionsForNode =>
      val node = distributionsForNode.head._1

      val parents = if(network.getParents(node).isDefined) network.getParents(node).get else List[Node]()
      val combinations = getCombinations(schema, parents)
      val allPossibleCombinations = combinations flatMap { featureCombination =>
        schema(node.id)._2.indices map { nodeVal =>
          new ValuesCombination(featureCombination, nodeVal)
        }
      }
      val combosForOperation = distributionsForNode.map(element => element._2).toArray
      val missingCombos = allPossibleCombinations filter { possibleCombo => !combosForOperation.contains(possibleCombo) }

      val zerosForMissing = missingCombos map { combo => (node, combo, 0) }
      val completeDistributionsForNode = distributionsForNode ++ zerosForMissing
      val allPossibleFeatureCombos = allPossibleCombinations map { combo => combo.getFeaturesCombination()} distinct

      val cpt =  if(allPossibleFeatureCombos.isEmpty){
        val counts = completeDistributionsForNode map { dist => (dist._2.valueOfDependentFeature, dist._3) }
        val cts = Array.fill[Double](counts.size)(0)
        counts foreach(c => cts(c._1) = c._2 + prior)
        new ConditionalProbabilityTable(node.id, None, Some(cts))
      }
      else {
        val countsForAllCombs = allPossibleFeatureCombos map { featureCombo =>
          val counts = completeDistributionsForNode.filter((value) => value._2.getFeaturesCombination() == featureCombo)
            .map(value => (value._2.valueOfDependentFeature, value._3))
          val cts = Array.fill[Double](counts.size)(0)
          counts foreach(c => cts(c._1) = c._2 + prior)
          (featureCombo.featureIndexAndValue, cts)
        }
        new ConditionalProbabilityTable(node.id, Some(countsForAllCombs.toMap), None)
      }
      (node.id, cpt)
    }

    nodeAndCPTs
  }

  /**
    * Given a list of nodes (could be parent nodes of a node in the network DAG), a collection of all possible combination
    * of values of all features is returned. For example, if categorical feature F1 can have values (f1, f2, f3) and
    * feature R can have values (r1, r2), the six possible combinations are: (f1, r1), (f1, r2), (f2, r1), (f2, r2),
    * (f3, r1), (f3, r2)
    *
    * @param parentNodes    List of nodes representing features
    * @return               All possible combinations of values
    */
   def getCombinations(schema : Array[(String, Array[String])], parentNodes : List[Node]) : ArrayBuffer[ArrayBuffer[(Int, Int)]] = {
    val numParents = parentNodes.size
    val valsOfParents = parentNodes map  { pNode =>
      val valuesBuffer = new ArrayBuffer[Int]
      schema(pNode.id)._2.indices map { index => valuesBuffer += index}
      valuesBuffer
    }

    var parentNodeIndex = 0
    var buffer = new ArrayBuffer[ArrayBuffer[Int]]()
    while(parentNodeIndex < numParents){
      val parentNodeValues = valsOfParents(parentNodeIndex)
      if(parentNodeIndex == 0){
        if(numParents == 1){
          buffer = for(value1 <- parentNodeValues) yield {ArrayBuffer(value1)}
          parentNodeIndex = parentNodeIndex + 1
        }
        else {
          val nextParentNodeValues = valsOfParents(parentNodeIndex + 1)
          buffer = for (value1 <- parentNodeValues; value2 <- nextParentNodeValues)
            yield {
              ArrayBuffer(value1, value2)
            }
          parentNodeIndex = parentNodeIndex + 1
        }
      }
      else{
        val updatedBuffer = for(value1 <- buffer; value2 <- parentNodeValues)
          yield {
            val updated = new ArrayBuffer[Int]()
            value1.copyToBuffer(updated)
            updated += value2
          }
        buffer = updatedBuffer
      }
      parentNodeIndex = parentNodeIndex + 1
    }

    buffer map { comb =>
      val buff = new ArrayBuffer[(Int, Int)]()
      ((parentNodes map {pn => pn.id}) zip comb).copyToBuffer(buff)
      buff
    }
  }
}
