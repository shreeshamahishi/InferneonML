package org.inferneon.bayesnet;

import org.apache.spark.SparkConf;
import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.api.java.JavaSparkContext;
import org.inferneon.bayesnet.core.BayesianBeliefNetwork;
import org.inferneon.bayesnet.hillclimber.ScoringType;
import org.inferneon.bayesnet.simulatedannealing.SimulatedAnnealing;
import org.junit.After;
import org.junit.Before;
import org.junit.Test;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

/**
 *
 * Java test cases for the simulated annealing algorithm. For now, just for the purpose of illustration.
 */

// TODO : Implement some meaningful test cases.

public class SimulatedAnnealingTests {

    private static JavaSparkContext javaSparkContext;

    @Before
    public void setUp(){
        if(javaSparkContext == null){
            SparkConf sparkConf = new SparkConf().setMaster("local[2]").setAppName("Test");
            javaSparkContext = new JavaSparkContext(sparkConf);
        }
    }

    @After
    public void tearDown() {
        if(javaSparkContext != null){
            javaSparkContext.stop();
        }
        javaSparkContext = null;
    }

    @Test
    public void test1() {
        String file = "/SampleSalesNoHeader.csv";
        String filePath =  getClass().getResource(file).getFile();
        JavaRDD<String> rawInput = javaSparkContext.textFile(filePath);

        List<Map<String, List<String>>> schema = new ArrayList<>();
        Map<String, List<String>> feature1 = new HashMap<>();
        List<String> f1Vals = new ArrayList<String>(); f1Vals.add("y"); f1Vals.add("n"); feature1.put("isMetro", f1Vals);
        Map<String, List<String>> feature2 = new HashMap<>();
        List<String> f2Vals = new ArrayList<String>(); f2Vals.add("y"); f2Vals.add("n"); feature2.put("midAge", f2Vals);
        Map<String, List<String>> feature3 = new HashMap<>();
        List<String> f3Vals = new ArrayList<String>(); f3Vals.add("f"); f3Vals.add("m");  feature3.put("gender", f3Vals);
        Map<String, List<String>> feature4 = new HashMap<>();
        List<String> f4Vals = new ArrayList<String>(); f4Vals.add("y"); f4Vals.add("n");  feature4.put("sales", f4Vals);
        schema.add(feature1); schema.add(feature2); schema.add(feature3); schema.add(feature4);

        BayesianBeliefNetwork network = SimulatedAnnealing.learnNetwork(rawInput, 2, 0.5,false, 3, true, schema,
                10.0, 300, 0.999, ScoringType.ENTROPY());

        System.out.println("Bayesian belief network: ");
        System.out.println(network.treeDescription(schema));
    }
}
