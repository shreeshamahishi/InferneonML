package org.inferneon.bayesnet.hillclimber

import org.apache.spark.Logging
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.rdd.RDD
import org.inferneon.bayesnet.core._
import org.inferneon.bayesnet.utils.MathUtils

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
    * @param scoringType         An enum indicating the method used for scoring.
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
                  private val sc: ScoringType.Value) extends Serializable with Logging with BayesianNetAlgorithm {

  private val scoreDeltasCache = new ScoreDeltasCache(schema.length)
  private var numRecords = 0L
  var scoringType = sc

  def run(rdd: RDD[LabeledPoint]): BayesianBeliefNetwork = {

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
        val combinations = getCombinations(schema, op.getProposedParents(nw.getParents(node)))
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
      val score = computeScore(rdd, network, featureValuesCounts, operation.target.id)
      val possibleLocalOperations = getPossibleLocalOperations(network, Some(operation.target))
      val opsAndScores = scoresByApplyingOperations(rdd, network, possibleLocalOperations).toMap

      updateCache(possibleLocalOperations, Map((operation.target.id, score)), opsAndScores)
      scoreDeltasCache.printCache()
    }

    // The hill climber algorithm starts here
    rdd.persist()
    val network = initializeNetwork(schema, classIndex, isCausal)
    rdd.unpersist()
    val nodes = network.allNodes
    val featureValuesCounts: Map[Int, Array[Int]] = getCategoricalCountsPerFeature(schema, nodes, rdd)
    // TODO: Use mapPartitions here instead of computing score for every feature?
    val baseScores = featureValuesCounts map { case (featureIndex, counts) =>
      (featureIndex, computeScore(rdd, network, featureValuesCounts, featureIndex))
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
    val cpts = computeCPTs(schema, network, rdd, prior)
    cpts foreach {case (nodeId, cpt) => network.cpts(nodeId) = cpt}
    network
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
}