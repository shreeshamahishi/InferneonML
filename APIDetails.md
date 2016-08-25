# INFERNEON ML API DETAILS

This document has details of the important interfaces defined in the DataUtils and the BayesNet modules in the project. 

## DataUtils.inferSchema()

The signature of the function is defined as follows:

```
def inferSchema(file: String, classIndex: Int, caseSensitive: Boolean): (ArrayBuffer[String], 
						Array[(String, Array[String])],
					List[Option[LabeledPoint]])
```

Given the path of a comma-separated file (CSV) file which contains data with header information, this method attempts to infer the schema from the data. The schema is inferred based on the following assumptions:

- The first row of the file represents the header information. Each comma-delimited field in this header
     is considered to denote the name of the feature.
- The data can contain missing information. This can either show up in the file as empty strings (or
     whitespaces) or can be represented by a question mark ( ? ).

- The data can contain commas; however in such cases, that data item must be enclosed in double-quotes.
     E.g.: "Hello, World". Moreover, escaped strings are not handled; in such cases, behaviour is undefined.
- The data can consist of both categorical (nominal) features as well as numerical data.

If a schema is successfully inferred (or whatever best is inferred), the schema as well as a list of labeled points are returned. A labeled point is returned for each row in the data with a the label value and a dense or sparse Vector. For a categorical feature, the corresponding entry in the Vector is a zero-indexd integer that corresponds to the index of that value int that categorical feature. For a numerical feature, the entry in the Vector will be the number itself. If a schema cannot be inferred, empty values for both the schema as well as the list of labeled points will be returned. Please read the description of the return value for further information. 

The parameters are defined as follows:
- file: Path of the file representing the input file.
    
- classIndex:  The class or target index indicating which column represents the class. This must be
                        a zero-indexed integer and must be lesser than the number of columns in the header.
- caseSensitive: If this is specified as "true" categorical values will be checked in a case-sensitive manner, and case-insensitive otherwise.

The return value is a three-tuple value which represents the following:
- The first element of the tuple is an array containing a descriptive Strings of errors. Errors
             might be found due to inconsistency in the data. If no errors are found, this array buffer
             will be empty. Each error description starts with the line number at which the error was seen.
             The line numbers are 0-based indexed. It is also possible that even when a reasonable schema was inferred,
             this array may not be empty.

- The second element of the tuple is an array representing the schema. Each element of this
             array identifies a feature at a corresponding column in the data. A feature is again represented
             by a 2-tuple. The first element of the feature tuple is the name of the feature as found in the
             first row (header) of the data. The second element of the feature tuple is an array of categorical
             values for that feature as found in the data. If a feature is inferred to be numerical,
             this corresponding array will be empty.

- The third element of the tuple is the list of labeled points. They are wrapped in an Optional
             value to address the possibility of inconsistency of the data at the corresponding row. If a row has
             missing data or some elements of a row could not be inferred, a LabeledPoint is created with a
             sparse Vector for that row; else if a row has data that could be inferred correctly for all fields, a
             LabeledPoint is created with a sparse Vector for that row.
    
## DataUtils.loadLabeledPoints()

Given the path of a comma-separated file (CSV) file which contains data and a suggested schema, this method
     attempts to create a list of LabeledPoint objects. The signature of the method is defined as follows:
     
     ```
     def loadLabeledPoints(path : String, schema : Array[(String, Array[String])], classIndex : Int, caseSensitive: Boolean):
  			(ArrayBuffer[String], Array[Option[LabeledPoint]])
     ```
The following assumptions are made:

- There is NO header information; the schema contains all the information needed for generating the LabeledPoint
     objects. The ordering of the columns correspond to the one specified in the schema.
	 
- The data can contain missing information. This can either show up in the file as empty strings (or
     whitespaces) or can be represented by a question mark ( ? ).
	 
- The data can contain commas; however in such cases, that data item must be enclosed in double-quotes.	 
     E.g.: "Hello, World". Moreover, escaped strings are not handled; in such cases, behaviour is undefined.

- The data can consist of both categorical (nominal) features as well as numerical data.
	 
A labeled point is returned for each row in the data with a the label value and a dense or sparse Vector. For a
     categorical feature, the corresponding entry in the Vector is a zero-indexed integer that corresponds to the
     index of that value in that categorical feature. For a numerical feature, the entry in the Vector will be the
     number itself. The parameters are:

- path: Path of the file representing the input file.

- schema:  An array of tuples representing the schema. Each element of this array should identify a feature
             at a corresponding column in the data. A feature is again represented by a 2-tuple. The first element
             of the feature tuple should denote the name of the feature. The second element of the feature tuple
             should be an array of categorical values for that feature as found in the data. If a feature is numerical,
             this corresponding array should be empty.
- classIndex:  The class or target index indicating which column represents the class. This must be
                        a zero-based indexed integer and must be lesser than the number of features suggested in the
                        schema.
- caseSensitive: If this is specified as "true" categorical values will be checked in a case-sensitive manner, and case-insensitive otherwise.

The two-tuple value represents the following:
- The first element of the tuple is an array containing a descriptive Strings of errors. Errors
             might be found due to inconsistency in the data. If no errors are found, this array buffer
             will be empty. Each error description starts with the line number at which the error was seen.
             The line numbers are 0-indexed.

- The second element of the tuple is the list of labeled points. They are wrapped in an Optional
             value to address the possibility of inconsistency of the data at the corresponding row. If a row has
             missing data or some elements of a row could not be inferred, a LabeledPoint is created with a
             sparse Vector for that row; else if a row has data that could be inferred correctly for all fields, a
             LabeledPoint is created with a dense Vector for that row.

## DataUtils.loadLabeledPointsRDD()

Given RDD of String representing CSV (comma-separated values) data and a suggested schema, this method attempts to create a RDD of LabeledPoint objects. The signature of the method is as defined as follows:
```
     def loadLabeledPointsRDD(sc: SparkContext, rdd: RDD[String], schema : Array[(String, Array[String])],
                           classIndex : Int, caseSensitive: Boolean):
  			   (ArrayBuffer[String], RDD[Option[LabeledPoint]])
```
The following assumptions are made:

- The ordering of the columns correspond to the one specified in the schema.

- The data can contain missing information. This can either show up in the file as empty strings (or
     whitespaces) or can be represented by a question mark ( ? ).
- The data can contain commas; however in such cases, that data item must be enclosed in double-quotes.
     E.g.: "Hello, World". Moreover, escaped strings are not handled; in such cases, behaviour is undefined.
- The data can consist of both categorical (nominal) features as well as numerical data.

A labeled point is returned for each row in the data with a the label value and a dense or sparse Vector. For a
     categorical feature, the corresponding entry in the Vector is a zero-based indexed integer that corresponds to the
     index of that value in that categorical feature. For a numerical feature, the entry in the Vector will be the
     number itself. The parameters are:
- sc: The SparkContext object

- rdd: The RDD of Strings which represents the data.
     
- schema:  An array of tuples representing the schema. Each element of this array should identify a feature
             at a corresponding column in the data. A feature is again represented by a 2-tuple. The first element
             of the feature tuple should denote the name of the feature. The second element of the feature tuple
             should be an array of categorical values for that feature as found in the data. If a feature is numerical,
             this corresponding array should be empty.

- classIndex:  The class or target index indicating which column represents the class. This must be
                        a zero-based indexed integer and must be lesser than the number of features suggested in the

- caseSensitive:  If this is specified as "true" categorical values will be checked in a case-sensitive manner, and case-insensitive otherwise.
The two-tuple value represents the following:

- The first element of the tuple is a RDD of descriptive Strings of errors. Errors
             might be found due to inconsistency in the data. If no errors are found, this RDD
             will be empty. Each error description starts with the line number at which the error was seen.
             The line numbers are 0-indexed.
- The second element of the tuple is the resulting RDD of labeled points. They are wrapped in an Optional
             value to address the possibility of inconsistency of the data at the corresponding row. If a row has
             missing data or some elements of a row could not be inferred, a LabeledPoint is created with a
             sparse Vector for that row; else if a row has data that could be inferred correctly for all fields, a
             LabeledPoint is created with a dense Vector for that row.

	
## HillClimber.learnNetwork()

This algorithm learns a Bayesian belief Network from data based on the hill climbing algorithm . Hill climbing is an
   optimization technique that uses local searching. In the case of learning Bayesian Belief Networks, the technique
   starts with an "initial guess" by assuming a particular configuration of the network then proceeds to by making
   incremental changes, one minor step at a time. In this implementation, the changes include either removing an
   an existing edge between source and target or adding a new one. The final network that is learnt consists of the DAG
   (directed acyclic graph) that represents the Bayesian belief network and a mapping of each node with its corresponding
   conditional probability table denoting the distribution for that feature.
  
   We first determine all possible changes that can be made. The constraints include : 1) Ensuring that adding an edge
   does not result in a cycle in the graph and 2) Ensuring that the number of parents of a node is does not exceed
   the maximum specified. The change that results in the best score is applied. This procedure is repeated until no further
   improvements in scores are observed.
  
   The input data is assumed to be in the form of a RDD of LabeledPoints and only works with categorical data. The RDD
   can be created from categorical data using the DataUtils utility.
   The signature of the method is defined as follows:
   
   ```
   def learnNetwork(input: RDD[LabeledPoint],
                   maxNumberOfParents: Int,
                   prior: Double,
                   isCausal: Boolean,
                   classIndex : Int,
                   schema : Array[(String, Array[String])],
                   scoringType: ScoringType.Value = ScoringType.ENTROPY) : BayesianBeliefNetwork 
   ```
 
 The parameters are:
 
- input:               Data represented as a RDD of LabeledPoints. It is assumed that the data is categorical.

- maxNumberOfParents:  The maximum number of parents a node can have in the graph.
- prior:               Prior on counts

- isCausal:            It this is set to true, the initial network will have edges from sources representing
                              the features to the label. If it is false, the initial edges is configured to start
                              with edges leading from the label to all other feature nodes.
   
- classIndex:          The class index in the data.

- schema:              The schema of the categorical data.

- scoringType:         An enum indicating the method used for scoring.

The returned value is a the Bayesian belief network learnt.


## SimulatedAnnealing.learnNetwork()

This algorithm learns a Bayesian belief Network from data based on the simulated annealing metaheuristic. Simulated
   annealing is an optimization technique that uses Monte Carlo simulation. The idea for this technique has its roots
   in a standard practice in the metallurgical industry where materials are heated to high temperatures and then
   cooled gradually. The process of slow cooling can be viewed as equivalent to gradually reducing probability of
   finding worse solutions in a large search space of (usually) discrete states.
  
   We start with some arbitrary temperature and after iteration reduce the temperature by a small amount. At each
   iteration, minor changes are made between two randomly chosen nodes - adding an edge if does not exist between them
   or removing an edge if it does exist. The difference in score is computed as the difference in the scores between
   the new state and the earlier one. The difference is accepted, i.e., the change is retained if the score has
   improved, and it if hasn't, it is probably accepted based on a random number.
  
   The algorithm thus results in a random walk over the search space and keeps reducing the probability of finding
   bad solutions as the temperature decreases. This ensures that there is a good chance of a solution NOT getting
   lodged into a local minimum, thereby improving the chance of a good approximation of the global minimum.
   The signature of the method is defined as follows:
   
   ```
    def learnNetwork(input: RDD[LabeledPoint],
                     maxNumberOfParents: Double = 2,
                     prior: Double,
                     isCausal: Boolean,
                     classIndex : Int,
                     schema : Array[(String, Array[String])],
                     initTemperature : Double,
                     maxIterations: Int,
                     temperatureStep : Double,
                     scoringType: ScoringType.Value = ScoringType.ENTROPY) : BayesianBeliefNetwork
   ```
 
 The parameters are defined as follows:
 - input:               Data represented as a RDD of LabeledPoints. It is assumed that the data is categorical.

- maxNumberOfParents:  The maximum number of parents a node can have in the graph.
- prior:               Prior on counts
- isCausal:            It this is set to true, the initial network will have edges from sources representing
                              the features to the label. If it is false, the initial edges is configured to start
                              with edges leading from the label to all other feature nodes.
- classIndex:          The class index in the data.
- schema:              The schema of the categorical data.
- initTemperature:     The initial temperature at which annealing commences.
- maxIterations:       The maximum number of steps to be attempted in the random walk.
- temperatureStep:     The change in temperature after each iteration.
- scoringType:         An enum indicating the method used for scoring.

The returned value is the Bayesian belief network learnt.

  

