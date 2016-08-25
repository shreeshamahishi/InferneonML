# INFERNEON ML API DETAILS

This document has details of the important interfaces defined in the DataUtils and the BayesNet modules in the project. 

## DataUtils.inferSchema()

The signature of the function is defined as follows:
	
	```def inferSchema(file: String, classIndex: Int, caseSensitive: Boolean): (ArrayBuffer[String], Array[(String, Array[String])],
					List[Option[LabeledPoint]])```
	
	

     Given the path of a comma-separated file (CSV) file which contains data with header information, this method
     attempts to infer the schema from the data. The schema is inferred based on the following assumptions:
    
    -The first row of the file represents the header information. Each comma-delimited field in this header
     is considered to denote the name of the feature.
    -The data can contain missing information. This can either show up in the file as empty strings (or
     whitespaces) or can be represented by a question mark ( ? ).
    -The data can contain commas; however in such cases, that data item must be enclosed in double-quotes.
     E.g.: "Hello, World". Moreover, escaped strings are not handled; in such cases, behaviour is undefined.
    -The data can consist of both categorical (nominal) features as well as numerical data.
    
     If a schema is successfully inferred (or whatever best is inferred), the schema as well as a list of labeled
     points are returned. A labeled point is returned for each row in the data with a the label value and a
     dense or sparse Vector. For a categorical feature, the corresponding entry in the Vector is a zero-indexed
     integer that corresponds to the index of that value int that categorical feature. For a numerical feature,
     the entry in the Vector will be the number itself. If a schema cannot be inferred, empty values for both the
     schema as well as the list of labeled points will be returned. Please read the description of the return
     value for further information.
    
	The parameters are defined as follows:
	
    -file Path of the file representing the input file.
    -classIndex  The class or target index indicating which column represents the class. This must be
                        a zero-indexed integer and must be lesser than the number of columns in the header.
    -caseSensitive If this is specified as "true" categorical values will be checked in a case-
                          sensitive manner, and case-insensitive otherwise.
	
    The return value is a three-tuple value which represents the following:
             -The first element of the tuple is an array containing a descriptive Strings of errors. Errors
             might be found due to inconsistency in the data. If no errors are found, this array buffer
             will be empty. Each error description starts with the line number at which the error was seen.
             The line numbers are 0-based indexed. It is also possible that even when a reasonable schema was inferred,
             this array may not be empty.
             -The second element of the tuple is an array representing the schema. Each element of this
             array identifies a feature at a corresponding column in the data. A feature is again represented
             by a 2-tuple. The first element of the feature tuple is the name of the feature as found in the
             first row (header) of the data. The second element of the feature tuple is an array of categorical
             values for that feature as found in the data. If a feature is inferred to be numerical,
             this corresponding array will be empty.
             -The third element of the tuple is the list of labeled points. They are wrapped in an Optional
             value to address the possibility of inconsistency of the data at the corresponding row. If a row has
             missing data or some elements of a row could not be inferred, a LabeledPoint is created with a
             sparse Vector for that row; else if a row has data that could be inferred correctly for all fields, a
             LabeledPoint is created with a sparse Vector for that row.
    
	
