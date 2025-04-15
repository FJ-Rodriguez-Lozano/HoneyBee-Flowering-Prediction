import weka.core.*;

import weka.classifiers.trees.*;
import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.FileReader;
import java.io.FileWriter;
import weka.core.Instance;
import weka.core.Attribute;

import java.io.*;

// Example of usage
// java -cp ".:weka.jar" c45  dataset/Flowering2016-train-1-of-10.arff dataset/Flowering2016-test-1-of-10.arff

public class c45 
{

  public c45() {
    super();
  }

  public static void main(String[] args) throws Exception 
  {
    double clsLabel; // variable para almacenar la probabilidad de la etiqueta de clase
    int instanceCounter = 0; // contador para bucles

    J48 clsJ48;      
    Instances testData, traindata; // variables for load test and train datasets


        
    traindata = new Instances(new BufferedReader(new FileReader(args[0]))); // load training set
    traindata.setClassIndex(traindata.numAttributes()-1); // set class last column

    clsJ48 = new J48(); // instanciate c45 classifier

    System.out.println("Training C45 ....");

    clsJ48.buildClassifier(traindata); // train classifier


    testData = new Instances(new BufferedReader(new FileReader(args[1]))); // load test set
       
    testData.setClassIndex(testData.numAttributes()-1); // set class last column

    System.out.println("Making predictions ....");

    // loop for iterate over all the test set to predict instancees
    for (instanceCounter = 0; instanceCounter < testData.numInstances(); ++instanceCounter) 
    {
      clsLabel = clsJ48.classifyInstance(testData.instance(instanceCounter)); // clasificar los datos

      // print information about current label of an instance and the predicted one
      System.out.println("Actual class: " + testData.classAttribute().value((int) testData.instance(instanceCounter).classValue()) + "... Predicted class: " + testData.classAttribute().value((int) clsLabel));      
    }
  }
}
