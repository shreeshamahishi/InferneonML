# INFERNEON ML
Inferneon ML is an open source machine learning library that can run on the distributed computing framework, Apache Spark. It contains algorithms that are not currently implemented in Spark and has other useful extensions and utilities. It consists of the following components:
- **DataUtils** - Tools and utilities to transform unstructured data into a format that works with Apache Spark and other distributed computing frameworks.
- **BayesNet** - A library for learning Bayesian belief network structures and joint probability distribution tables. Current implementations include search / optimization techniques like hill climbing and simulated annealing using local search methods. The current implementations support only categorical (or nominal) data. Future enhancements will extend the implementation to include global search techniques, support for continuous-valued data and other search / optimization procedures. 

**The project uses Scala as the implementing language. The BayesNet module also consists of Java-friendly APIs that Java programmers can use to integrate the library into their code.**

This document describes the details of the requirements of the project, how to build the project and an outline of the development environment. For information on the usage of the algorithm and specific API-level details, please refer to this document.

##Requirements##
- Scala: 2.10.5
- Spark: 1.6.1
- Spark MLLib:  1.6.1
- If this API needs to be used from Java code, Java 1.7 or above will suffice.

##Build and install the library##
Maven is used for building the project. Currently, the binaries are not available in a remote Maven repository yet and should be built from source. After cloning the repository, build the project using Maven with the following command:

`mvn -e clean install`

After the build completes successfully, the library jar files will be created under the DataUtils/target and BayesNet/target folders.
After building the project, the libraries should be installed in the developer’s local Maven repository. Both the libraries can be installed using Maven’s install commands: 

`mvn install:install-file -Dfile=<path_to_datautils_jar> -DartifactId=inferneon.ml-datautils_2.10 -Dversion=1.0-SNAPSHOT -Dpackaging=jar`

`mvn install:install-file -Dfile=<path_to_bayesnet_jar> -DartifactId=inferneon.ml-bayesnet_2.10 -Dversion=1.0-SNAPSHOT -Dpackaging=jar`

where the inputs \<path_to_datautils_jar\> and \<path_to_bayesnet_jar\> refer to the location of the DataUtils and BayesNet library jar files respectively. These commands install DataUtils and BayesNet libraries in the local Maven repository. Upon successful installation of these libraries, add the following dependencies in your POM file:

```
<dependency>
  <groupId>org.inferneon.ml</groupId>
	<artifactId>inferneon.ml-datautils_2.10</artifactId>
	<version>1.0-SNAPSHOT</version>
</dependency>
<dependency>
	<groupId>org.inferneon.ml</groupId>
	<artifactId>inferneon.ml-bayesnet_2.10</artifactId>
	<version>1.0-SNAPSHOT</version>
</dependency>
```

The libraries can now be used in your code. Please refer to the API documentation for information on how to use and integrate the algorithms.

##Development environment

The authors use IntelliJ IDEA as the development tool. However, choice of an IDE is subjective and preferences will vary. Users may also try Eclipse with suitable Scala plug-ins.
In the IntelliJ IDEA environment with Scala plug-ins installed, the project can be imported as a Maven project via the *File -> New -> Project from Existing Sources* menu path. The default settings in the Import Wizard should suffice to build the project.
