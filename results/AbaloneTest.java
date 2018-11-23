package opt.test;

import dist.*;
import opt.*;
import opt.example.*;
import opt.ga.*;
import shared.*;
import func.nn.backprop.*;
import src.opt.test.CSVUtils;

import java.util.*;
import java.io.*;
import java.text.*;

/**
 * Implementation of randomized hill climbing, simulated annealing, and genetic algorithm to
 * find optimal weights to a neural network that is classifying abalone as having either fewer 
 * or more than 15 rings. 
 *
 * @author Hannah Lau
 * @version 1.0
 */
public class AbaloneTest {
    private static Instance[] instances = initializeInstances("train");

    private static int inputLayer = 195, hiddenLayer = 5, outputLayer = 1;
    private static BackPropagationNetworkFactory factory = new BackPropagationNetworkFactory();
    
    private static ErrorMeasure measure = new SumOfSquaresError();

    private static DataSet set = new DataSet(instances);

    private static BackPropagationNetwork networks[] = new BackPropagationNetwork[3];
    private static NeuralNetworkOptimizationProblem[] nnop = new NeuralNetworkOptimizationProblem[3];

    private static OptimizationAlgorithm[] oa = new OptimizationAlgorithm[3];
    private static String[] oaNames = {"RHC", "SA", "GA"};
    private static String results = "";

    private static DecimalFormat df = new DecimalFormat("0.000");

    public static void main(String[] args) throws Exception {
        for(int i = 0; i < oa.length; i++) {
            networks[i] = factory.createClassificationNetwork(
                new int[] {inputLayer, hiddenLayer, outputLayer});
            nnop[i] = new NeuralNetworkOptimizationProblem(set, networks[i], measure);
        }

        oa[0] = new RandomizedHillClimbing(nnop[0]);
        oa[1] = new SimulatedAnnealing(1E11, .95, nnop[1]);
        oa[2] = new StandardGeneticAlgorithm(200, 100, 10, nnop[2]);

        List<String> iter_x = new ArrayList<String>();
        List<String> accuracy_y = new ArrayList<String>();
        List<String> time_y = new ArrayList<String>();

        for(int i = 0; i < oa.length; i++) {
            //String[] filenames = {"./results/nn-rhc-clinvar.csv", "./results/nn-sa-clinvar.csv", "./results/nn-ga-clinvar.csv"};
            //FileWriter writer = new FileWriter(filenames[i], true);
            double start = System.nanoTime(), end, trainingTime, testingTime, correct = 0, incorrect = 0;
            train(oa[i], networks[i], oaNames[i], 5000); //trainer.train();
            end = System.nanoTime();
            trainingTime = end - start;
            trainingTime /= Math.pow(10, 9);

            Instance optimalInstance = oa[i].getOptimal();
            networks[i].setWeights(optimalInstance.getData());

            double predicted, actual;
            start = System.nanoTime();

            instances = initializeInstances("test");
            for (int j = 0; j < instances.length; j++) {
                networks[i].setInputValues(instances[j].getData());
                networks[i].run();

                predicted = Double.parseDouble(instances[j].getLabel().toString());
                actual = Double.parseDouble(networks[i].getOutputValues().toString());

                double trash = Math.abs(predicted - actual) < 0.5 ? correct++ : incorrect++;
            }

            end = System.nanoTime();
            testingTime = end - start;
            testingTime /= Math.pow(10, 9);

            results += "\nResults for " + oaNames[i] + ": \nCorrectly classified " + correct + " instances." +
                    "\nIncorrectly classified " + incorrect + " instances.\nPercent correctly classified: "
                    + df.format(correct / (correct + incorrect) * 100) + "%\nTraining time: " + df.format(trainingTime)
                    + " seconds\nTesting time: " + df.format(testingTime) + " seconds\n";

            iter_x.add(Integer.toString(2000));
            accuracy_y.add(Double.toString(correct / (correct + incorrect)));
            time_y.add(Double.toString(testingTime));


            System.out.println(results);
            //CSVUtils.writeLine(writer, iter_x);
            //CSVUtils.writeLine(writer, accuracy_y);
            //CSVUtils.writeLine(writer, time_y);
            //writer.flush();
            //writer.close();
            iter_x.clear();
            accuracy_y.clear();
            time_y.clear();
        }
    }

    private static void train(OptimizationAlgorithm oa, BackPropagationNetwork network, String oaName, int trainingIterations) {
        //System.out.println("\nError results for " + oaName + "\n---------------------------");

        for(int i = 0; i < trainingIterations; i++) {
            oa.train();

            double error = 0;
            for(int j = 0; j < instances.length; j++) {
                network.setInputValues(instances[j].getData());
                network.run();

                Instance output = instances[j].getLabel(), example = new Instance(network.getOutputValues());
                example.setLabel(new Instance(Double.parseDouble(network.getOutputValues().toString())));
                error += measure.value(output, example);
            }

            //System.out.println(df.format(error));
        }
    }

    private static Instance[] initializeInstances(String purpose) {

        double[][][] attributes;

        if (purpose.equals("test")) {
            attributes = new double[19000][][];
        } else {
            attributes = new double[46000][][];
        }

        try {
            BufferedReader br = new BufferedReader(new FileReader(new File("src/opt/test/processed_2.txt")));
            int toSkip = 46000;
            int counter = 0;

            while (purpose.equals("test") && counter < toSkip) {
                br.readLine();
                counter++;
                System.out.println("Skipped line" + Integer.toString(counter));
            }

            for(int i = 0; i < attributes.length; i++) {
                Scanner scan = new Scanner(br.readLine());
                scan.useDelimiter(",");

                attributes[i] = new double[2][];
                attributes[i][0] = new double[195]; // 7 attributes
                attributes[i][1] = new double[1];

                for(int j = 0; j < 195; j++) {
                    String nxt = scan.next();
                    if (nxt.length() == 0) {
                        attributes[i][0][j] = Double.parseDouble("0");
                    } else {
                        attributes[i][0][j] = Double.parseDouble(nxt);
                    }
                }
                String clss = scan.next();
                if (clss.length() == 0) {
                    attributes[i][1][0] = Double.parseDouble("0");
                } else {
                    attributes[i][1][0] = Double.parseDouble(clss);
                }
            }
        }
        catch(Exception e) {
            e.printStackTrace();
        }

        Instance[] instances = new Instance[attributes.length];

        for(int i = 0; i < instances.length; i++) {
            instances[i] = new Instance(attributes[i][0]);
            // classifications range from 0 to 30; split into 0 - 14 and 15 - 30
            instances[i].setLabel(new Instance(attributes[i][1][0] == 0 ? 0 : 1));
        }

        return instances;
    }
}
