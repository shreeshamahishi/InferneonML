package org.inferneon.bayesnet.simulatedannealing

import java.util.Random

import org.apache.spark.Logging
import org.apache.spark.api.java.JavaRDD
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.rdd.RDD
import org.inferneon.bayesnet.core.{BayesianBeliefNetwork, BayesianNetAlgorithm, Node}
import org.inferneon.bayesnet.hillclimber.ScoringType
import org.inferneon.bayesnet.DataUtils

import scala.collection.mutable.ArrayBuffer

/**
  * This algorithm learns a Bayesian belief Network from data based on the simulated annealing metaheuristic. Simulated
  * annealing is an optimization technique that uses Monte Carlo simulation. The idea for this technique has its roots
  * in a standard practice in the metallurgical industry where materials are heated to high temperatures and then
  * cooled gradually. The process of slow cooling can be viewed as equivalent to gradually reducing probability of
  * finding worse solutions in a large search space of (usually) discrete states.
  *
  * We start with some arbitrary temperature and after iteration reduce the temperature by a small amount. At each
  * iteration, minor changes are made between two randomly chosen nodes - adding an edge if does not exist between them
  * or removing an edge if it does exist. The difference in score is computed as the difference in the scores between
  * the new state and the earlier one. The difference is accepted, i.e., the change is retained if the score has
  * improved, and it if hasn't, it is probably accepted based on a random number.
  *
  * The algorithm thus results in a random walk over the search space and keeps reducing the probability of finding
  * bad solutions as the temperature decreases. This ensures that there is a good chance of a solution getting lodged
  * into a local minimum, thereby improving the chance of a good approximation of the global minimum.
  *
  */

  object SimulatedAnnealing extends Serializable with Logging {

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
      * @param initTemperature     The initial temperature at which annealing commences.
      * @param maxIterations       The maximum number of steps to be attempted in the random walk.
      * @param temperatureStep     The change in temperature after each iteration.
      * @param scoringType         An enum indicating the method used for scoring.
      * @return                    The Bayesian belief network learnt.
      */
    def learnNetwork(input: RDD[LabeledPoint],
                     maxNumberOfParents: Double = 2,
                     prior: Double,
                     isCausal: Boolean,
                     classIndex : Int,
                     schema : Array[(String, Array[String])],
                     initTemperature : Double,
                     maxIterations: Int,
                     temperatureStep : Double,
                     scoringType: ScoringType.Value = ScoringType.ENTROPY) : BayesianBeliefNetwork = {
      val annealer = new SimulatedAnnealing(maxNumberOfParents, prior, isCausal, classIndex, schema,
                                            initTemperature, maxIterations, temperatureStep,  scoringType)
      annealer.run(input)
    }

  /*  Java-friendly version of learnNetwork() */

    def learnNetwork(input: JavaRDD[LabeledPoint],
                   maxNumberOfParents: Double,
                   prior: Double,
                   isCausal: Boolean,
                   classIndex : Int,
                   format : java.util.List[java.util.Map[String, java.util.List[String]]],
                   initTemperature : Double,
                   maxIterations: Int,
                   temperatureStep : Double,
                   scoringType: ScoringType.Value) : BayesianBeliefNetwork = {
       val rdd = input.rdd
       val schema = DataUtils.schemaFromJava(format)
       val annealer = new SimulatedAnnealing(maxNumberOfParents, prior, isCausal, classIndex, schema,
                            initTemperature, maxIterations, temperatureStep,  scoringType)
       annealer.run(rdd)
  }


  /*  Another Java-friendly version of learnNetwork(). Needs only the raw data in RDD format */
  def learnNetwork(input: JavaRDD[String],
                   maxNumberOfParents: Double,
                   prior: Double,
                   isCausal: Boolean,
                   classIndex : Int,
                   caseSensitive: Boolean,
                   format : java.util.List[java.util.Map[String, java.util.List[String]]],
                   initTemperature : Double,
                   maxIterations: Int,
                   temperatureStep : Double,
                   scoringType: ScoringType.Value) : BayesianBeliefNetwork = {

    val rdd = input.rdd
    val schema = DataUtils.schemaFromJava(format)
    val (errors, labeledPointsRDD) = DataUtils.loadLabeledPointsRDD(input.sparkContext, rdd, schema,
      classIndex : Int, caseSensitive: Boolean)
    require(errors.isEmpty)
    val lbPointsRDD: RDD[LabeledPoint] = labeledPointsRDD filter {_.isDefined} map {_.get}

    val annealer = new SimulatedAnnealing(maxNumberOfParents, prior, isCausal, classIndex, schema,
      initTemperature, maxIterations, temperatureStep,  scoringType)

    annealer.run(lbPointsRDD)
  }
}


  class SimulatedAnnealing(private val maxNumberOfParents: Double,
                    private val prior: Double,
                    private val isCausal: Boolean,
                    private val classIndex: Int,
                    private val schema: Array[(String, Array[String])],
                    private val initTemperature : Double,
                    private val maxIterations: Int,
                    private val temperatureStep : Double,
                    private val sc: ScoringType.Value,
                    private val random : Random = new Random(1)) extends Serializable with Logging with BayesianNetAlgorithm {

    var scoringType = sc

    def run(rdd: RDD[LabeledPoint]): BayesianBeliefNetwork = {

      def getAnyValidSourceAndTarget(network: BayesianBeliefNetwork, iterationIndex: Int, numFeatures : Int): (Node, Node, Boolean) ={
        val srcId = random.nextInt(numFeatures)
        val source = Node.emptyNode(srcId, isLabel = if(srcId == classIndex){true} else{false})
        var targetId = random.nextInt(numFeatures)
        var target = Node.emptyNode(targetId, isLabel = if(targetId == classIndex){true} else{false})
        while(source == target){
          targetId = random.nextInt(numFeatures)
          target = Node.emptyNode(targetId, isLabel = if(targetId == classIndex){true} else{false})
        }

        (source, target, network.edgeExists(source, target))
      }

      // The algorithm starts here
      val numFeatures = schema.length
      rdd.persist()
      val network = initializeNetwork(schema, classIndex, isCausal)
      rdd.unpersist()
      val nodes = network.allNodes
      val featureValuesCounts: Map[Int, Array[Int]] = getCategoricalCountsPerFeature(schema, nodes, rdd)
      // TODO: Use mapPartitions here instead of computing score for every feature?
      val baseScoresMap = featureValuesCounts map { case (featureIndex, counts) =>
        (featureIndex, computeScore(rdd, network, featureValuesCounts, featureIndex))
      }

      val baseScores =  ArrayBuffer.fill(numFeatures)(0.0)
      baseScoresMap foreach {tuple =>
        baseScores(tuple._1) = tuple._2
      }
      var currScore  = baseScores.reduce(_ + _)
      val maxScore = currScore
      var currentTemperature = initTemperature

      var iterationIndex = 0
      while(iterationIndex < maxIterations){
        logInfo("Iteration number: " + iterationIndex)
        var iterationSuccessful = false
        while(!iterationSuccessful) {
          val (source, target, edgeExists) = getAnyValidSourceAndTarget(network, iterationIndex, numFeatures)
          if (edgeExists) {
            iterationSuccessful = true
            network.removeEdge(source, target)
            val score = computeScore(rdd, network, featureValuesCounts, target.id)
            val differenceInScore = score - baseScores(target.id)
            if (accept(currentTemperature, differenceInScore)) {
              baseScores(target.id) = score
              currScore = currScore + differenceInScore
              if (currScore < maxScore) {
                network.addEdge(source, target)
              }
            }
            else {   // Reset
              network.addEdge(source, target)
            }
          }
          else {
            if (!network.addingEdgeCreatesCycle(source, target)) {
              iterationSuccessful = true
              network.addEdge(source, target)
              val score = computeScore(rdd, network, featureValuesCounts, target.id)
              val differenceInScore = score - baseScores(target.id)
              if (accept(currentTemperature, differenceInScore)) {
                baseScores(target.id) = score
                currScore = currScore + differenceInScore
                if (currScore < maxScore) {
                  network.removeEdge(source, target)
                }
              }
              else {   // Reset
                network.removeEdge(source, target)
              }
            }
          }
        }

        iterationIndex = iterationIndex + 1
        currentTemperature = currentTemperature * temperatureStep
      }

      // The network is determined, compute the conditional probability tables for each feature node.
      val cpts = computeCPTs(schema, network, rdd, prior)
      cpts foreach {case (nodeId, cpt) => network.cpts(nodeId) = cpt}

      network
    }

    /**
      * Criteria for acceptance of the state. If the score is improved, the state is accepted, else is accepted with a
      * probability.
      *
      * @param currentTemperature       The current temperature in this iteration.
      * @param differenceInScore        The difference in score between the two states.
      * @return
      */
    def accept(currentTemperature: Double, differenceInScore: Double) : Boolean = {
      val randomInt = random.nextInt()
      val calc = currentTemperature * Math.log((Math.abs(randomInt) % 10000) / 10000.0 + 1e-100)
      val result = differenceInScore > calc
      result
    }
  }
