package org.inferneon.bayesnet.hillclimber

import org.apache.spark.Logging
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.rdd.RDD
import org.inferneon.bayesnet.core.{BayesianBeliefNetwork, ConditionalProbabilityTable, Node, ValuesCombination}
import org.inferneon.bayesnet.utils.MathUtils

import scala.collection.mutable.ArrayBuffer
import scala.tools.nsc.util.HashSet
import scala.language.postfixOps

/**
  * This algorithm learns a Bayesian belief Network from data based on the hill climbing algorithm . Hill climbing is an
  * optimization technique that uses local searching. In the case of learning Bayesian Belief Networks, the technique
  * starts with an "initial guess" by assuming a particular configuration of the network then proceeds to by making
  * incremental changes, one minor step at a time. In this implementation, the changes include either removing an
  * an existing edge between source and target or adding a new one. The final network that is learnt consists of the DAG
  * (directed acyclic graph) that represents the Bayesian belief network and a mapping of each node with its corresponding
  * conditional probability table denoting the distribution for that feature.
  *
  * We first determine all possible changes that can be made. The constraints include : 1) Ensuring that adding an edge
  * does not result in a cycle in the graph and 2) Ensuring that the number of parents of a node is does not exceed
  * the maximum specified. The change that results in the best score is applied. This procedure is repeated until no further
  * improvements in scores are observed.
  *
  * The input data is assumed to be in the form of a RDD of LabeledPoints and only works with categorical data. The RDD
  * can be created from categorical data using the DataUtils utility.
  */

object HillClimber extends Serializable with Logging {

  /**
    * Main method to learn the network.
    *
    * @param input               Data represented as a RDD of LabeledPoints. It is assumed that the data is categorical.
    * @param maxNumberOfParents  The maximum number of parents a node can have in the graph.
    * @param prior               Prior on counts
    * @param isCausal            It this is set to true, the initial network will have edges from sources representing
    *                            the features to the label. If it is false, the initial edges is configured to start
    *                            with edges leading from the label to all other feature nodes.
    * @param classIndex          The class index in the data.
    * @param schema              The schema of the categorical data.
    * @return                    The Bayesian belief network learnt.
    */
  def learnNetwork(input: RDD[LabeledPoint],
                   maxNumberOfParents: Double = 2,
                   prior: Double,
                   isCausal: Boolean,
                   classIndex : Int,
                   schema : Array[(String, Array[String])],
                   scoringType: ScoringType.Value = ScoringType.ENTROPY) : BayesianBeliefNetwork = {
    val hillClimber = new HillClimber(maxNumberOfParents, prior, isCausal, classIndex, schema, scoringType)
    hillClimber.run(input)
  }
}

class HillClimber(private val maxNumberOfParents: Double,
                  private val prior: Double,
                  private val isCausal: Boolean,
                  private val classIndex: Int,
                  private val schema: Array[(String, Array[String])],
                  private val scoringType: ScoringType.Value) extends Serializable with Logging {

  private val scoreDeltasCache = new ScoreDeltasCache(schema.length)
  private var numRecords = 0L

  def run(rdd: RDD[LabeledPoint]): BayesianBeliefNetwork = {

    /**
      * Initializes the network depending upon the "isCausal" flag. If this is set to true, we initialize the network
      * structure with all feature nodes having directed edges to the label node; else the network is intialized with
      * edges from the label node to all other feature nodes.
      * @return Initial network
      */
    def initializeNetwork(): BayesianBeliefNetwork = {
      rdd.persist()
      numRecords = rdd.count()
      val network = BayesianBeliefNetwork.emptyNetwork()
      val numFeatures = schema.length
      validateSchema(numFeatures)
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

      rdd.unpersist()
      network
    }

    def validateSchema(numFeatures: Int): Unit = {
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

    def computeScore(network: BayesianBeliefNetwork, featureValuesCounts: Map[Int, Array[Int]], featureIndex : Int): Double = {
      val node = network.allNodes(featureIndex)
      if(network.getParents(node).isEmpty){ // Just return the score for this node with no parent
        scoreBasedOnCounts(Array[Array[Int]](featureValuesCounts(featureIndex)))
      }
      else{
        getScoreForNodeWithParents(network, node)
      }
    }

    def getScoreForNodeWithParents(network: BayesianBeliefNetwork, node : Node) : Double = {
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

    /**
      * Computes the score for each possible operation. Operations involve minor local changes - either adding a new
      * parent to a node or removing an existing parent.
      * Instead of computing a score for each partition separately, we use mapPartitions() to compute the distributions
      * in each partition and then merge the result to compute the score for all partitions in one pass of the data.
      *
      * @param rdd                     Complete data as RDD of LabeledPoints.
      * @param nw                      The current network
      * @param possibleOperations      A list of possible operations we want to try. The distributions of the data are
      *                                computed in each partition depending upon each of the possible operations.
      * @return                        A mapping of operation against the score.
      */
    def scoresByApplyingOperations(rdd: RDD[LabeledPoint], nw:BayesianBeliefNetwork,
                                   possibleOperations: List[Operation]) : Array[(Operation, Double)] = {

      val distributionsForOperations = rdd mapPartitions { labeledPoints =>
        val labeledPointsArray = labeledPoints.toArray
        val distributionForPossibleOperationsInPartition = possibleOperations map { operation =>
          val node = operation.target
          val proposedParents: List[Node] = operation.getProposedParents(nw.getParents(node))
          val vcs = labeledPointsArray map { labeledPoint =>
            val valuesCombination = getCombinationsFromLabeledPoint(node, proposedParents, labeledPoint)
            (valuesCombination, 1)
          }

          val groupedByVcs = vcs groupBy(value => value._1)
          val counts = groupedByVcs map {case (k, v) => (operation, k, v.length)}
          counts
        }
        distributionForPossibleOperationsInPartition.iterator
      }

      val distributionsGroupedByOperations = distributionsForOperations
        .flatMap(element => element map {item => item}) groupBy(element => element._1) collect()
      val distributionsGroupedByOperationsAndCombinations = distributionsGroupedByOperations map
        {case (operation, valueCombinations) =>  (operation, valueCombinations groupBy(item => item._2)) }
      val operationVCAndCounts = distributionsGroupedByOperationsAndCombinations map { element => {element._2 map
        { case (vc, distsInPartitions) =>
          val countInPartitions = distsInPartitions map {case (op, vc, count) => count}
          val total = (0 /: countInPartitions) ((count1, count2) => count1 + count2)
          (element._1, vc, total)
        }
      }
      }

      // We have the distributions, lets get the scores.
      val opsAndScores = operationVCAndCounts map { distributionsForOperation =>
        val op = distributionsForOperation.head._1
        val node = op.target
        val combinations = getCombinations(op.getProposedParents(nw.getParents(node)))
        val allPossiblecombinations = combinations flatMap { featureCombination =>
          schema(node.id)._2.indices map { nodeVal =>
            new ValuesCombination(featureCombination, nodeVal)
          }
        }
        val combosForOperation = distributionsForOperation.map(element => element._2).toArray
        val missingCombos = allPossiblecombinations filter { possibleCombo => !combosForOperation.contains(possibleCombo) }
        val zerosForMissing = missingCombos map { combo => (op, combo, 0) }
        val completeDistributionsForOperation = distributionsForOperation ++ zerosForMissing
        val allPossibleFeatureCombos = allPossiblecombinations map { combo => combo.getFeaturesCombination()} distinct

        if(allPossibleFeatureCombos.isEmpty){
          val counts = completeDistributionsForOperation map {dist => dist._3} toArray
          val scs = scoreBasedOnCounts(Array[Array[Int]](counts))
          (op, scs)
        }
        else {
          val counts =  allPossibleFeatureCombos map {featureCombo => completeDistributionsForOperation
            .filter(value => value._2.getFeaturesCombination() == featureCombo).toArray.map(value => value._3)}
          val scs = scoreBasedOnCounts(counts.toArray)
          (op, scs)
        }
      }
      opsAndScores
    }

    /**
      * Performs the operation (adding an edge or deleting an existing one) and updates the cache that holds the change
      * in score.
      * @param network                    The Bayesian network
      * @param operation                  The operation
      * @param featureValuesCounts        The distribution of data of the feature node to which either an incoming edge
      *                                   has been added or an incoming edge has been removed based on the operation.
      */
    def performOperation(network: BayesianBeliefNetwork, operation: Operation, featureValuesCounts: Map[Int, Array[Int]] ): Unit = {
      if(operation.operationType == OperationType.Add){
        network.addEdge(operation.source, operation.target)
      }
      else{
        network.removeEdge(operation.source, operation.target)
      }
      val score = computeScore(network, featureValuesCounts, operation.target.id)
      val possibleLocalOperations = getPossibleLocalOperations(network, Some(operation.target))
      val opsAndScores = scoresByApplyingOperations(rdd, network, possibleLocalOperations).toMap

      updateCache(possibleLocalOperations, Map((operation.target.id, score)), opsAndScores)
      scoreDeltasCache.printCache()
    }

    // The hill climber algorithm starts here

    // Initialize the network and compute base scores
    val network = initializeNetwork()
    val nodes = network.allNodes
    val featureValuesCounts: Map[Int, Array[Int]] = getCategoricalCountsPerFeature(nodes, rdd)
    // TODO: Use mapPartitions here instead of computing score for every feature?
    val baseScores = featureValuesCounts map { case (featureIndex, counts) =>
      (featureIndex, computeScore(network, featureValuesCounts, featureIndex))
    }

    // Compute change in scores by considering possible local operations
    var possibleLocalOperations = getPossibleLocalOperations(network, None)
    val opsAndScores = scoresByApplyingOperations(rdd, network, possibleLocalOperations).toMap
    updateCache(possibleLocalOperations, baseScores, opsAndScores)
    scoreDeltasCache.printCache()

    // Iterate until there is no improvement (converge to a solution).
    var stopSearching = false
    var score = 0.0
    while(!stopSearching){
      val scoreAndOption = getOptimalOperation(network, possibleLocalOperations)
      val opOption = scoreAndOption._2
      score = scoreAndOption._1
      if(opOption.isEmpty || MathUtils.approximatelyEqual(score, 0.0, 0.0000001)){
        stopSearching = true
      }
      else{
        val optimalOperation = opOption.get
        performOperation(network, optimalOperation, featureValuesCounts)
        possibleLocalOperations = getPossibleLocalOperations(network, None)
      }
    }

    // The network is determined, compute the conditional probability tables for each feature node.
    val cpts = computeCPTs(network, rdd)
    cpts foreach {case (nodeId, cpt) => network.cpts(nodeId) = cpt}
    network
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
    * Given a list of nodes (could be parent nodes of a node in the network DAG), a collection of all possible combination
    * of values of all features is returned. For example, if categorical feature F1 can have values (f1, f2, f3) and
    * feature R can have values (r1, r2), the six possible combinations are: (f1, r1), (f1, r2), (f2, r1), (f2, r2),
    * (f3, r1), (f3, r2)
    *
    * @param parentNodes    List of nodes representing features
    * @return               All possible combinations of values
    */
  private def getCombinations(parentNodes : List[Node]) : ArrayBuffer[ArrayBuffer[(Int, Int)]] = {
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

  /**
    * Computes the joint conditional probability distribution for each feature node in the DAG. Instead of multiple passes
    * over the data, we use mapPartitions to determine the distribution for all nodes in a given partition and aggregate
    * them to determine the final distribution.
    *
    * @param network   The network
    * @param rdd       Data in the form of RDD[LabeledPoint]
    * @return          A mapping of node id and corresponding conditional probability table.
    */
  private[bayesnet] def computeCPTs(network: BayesianBeliefNetwork, rdd:RDD[LabeledPoint]): Array[(Int, ConditionalProbabilityTable)] ={
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
      val combinations = getCombinations(parents)
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
    * Determines the optimal operation - the one with the highest change in score for an operation - by comparing the
    * changes in score stored in the cache. If an cycle can be introduced by adding an edge, we ignore that operation.
    *
    * @param nw                            The network
    * @param possibleLocalOperations       The possible local operations
    * @return                              A tuple containing the best change in score and the corresponding operation.
    */
  private[bayesnet] def getOptimalOperation(nw: BayesianBeliefNetwork, possibleLocalOperations: List[Operation]): (Double, Option[Operation]) = {
    var maxDelta = Double.MinValue
    var optimalOption: Option[Operation] = None
    possibleLocalOperations foreach { op =>
      val score: Double = scoreDeltasCache.get(op)
      val createsCycles = op.operationType == OperationType.Add && nw.addingEdgeCreatesCycle(op.source, op.target)
      if(!createsCycles && score > maxDelta){
        optimalOption = Some(op)
        maxDelta = score
      }
    }
    (maxDelta, optimalOption)
  }

  private[bayesnet] def updateCache(possibleLocalOperations: List[Operation], baseScores: Map[Int, Double],
                          opsAndScores: Map[Operation, Double]) {
    possibleLocalOperations foreach { op =>
      val delta = opsAndScores(op) - baseScores(op.target.id)
      scoreDeltasCache.put(op, delta)
    }
  }

  /**
    * Determines all possible local operations that can be performed on all feature nodes in the graph or on a specific
    * node in the graph. The choice is either to remove an existing edge between a source and target node or to insert
    * an edge.
    *
    * @param network              The network.
    * @param nodeOfInterest       If this is not empty, we look for operations only pertaining to this node. Else we
    *                             look in the entire graph.
    * @return                     A list of all possible operations.
    */
  private def getPossibleLocalOperations(network: BayesianBeliefNetwork, nodeOfInterest: Option[Node]): List[Operation] ={
    def filter(node: Node, otherNode: Node, nodeOfInterest: Option[Node]): Boolean ={
      if (nodeOfInterest.isEmpty) {
        node != otherNode
      }
      else {
        val ndOfInt = nodeOfInterest.get
        ndOfInt == node && node != otherNode
      }
    }

    val nodes = network.allNodes
    val possibleOps: List[Any] = for {node <- nodes ; otherNode <- nodes if filter(node, otherNode, nodeOfInterest)} yield {
      val prts = network.getParents(node)
      if(prts.isDefined){
        val parents = prts.get
        if(parents.contains(otherNode)){
          new Operation(otherNode, node, OperationType.Delete)
        }
        else{
          if(parents.length < maxNumberOfParents) {
            new Operation(otherNode, node, OperationType.Add)
          }
        }
      }
      else{
          new Operation(otherNode, node, OperationType.Add)
      }
    }
    possibleOps.filter(op => op.isInstanceOf[Operation]) map {op => op.asInstanceOf[Operation]}
  }

  /**
    * Counts the distribution of values of a categorical feature. Again, instead of multiple passes over the data for
    * each feature, we use mapPartitions to get the counts per partition and then aggregate them.
    * @param nodes    All nodes in the network.
    * @param rdd      The data in the the form of a RDD of LabeledPoints.
    * @return         A mapping of each feature (node) index and corresponding counts of values.
    */

  private def getCategoricalCountsPerFeature(nodes:List[Node], rdd : RDD[LabeledPoint]) : Map[Int, Array[Int]] = {

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
}