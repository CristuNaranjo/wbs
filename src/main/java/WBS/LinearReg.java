package WBS;

/**
 * Created by cristu on 10/08/16.
 */

import org.apache.spark.SparkConf;
import org.apache.spark.api.java.JavaSparkContext;
import org.apache.spark.ml.regression.LinearRegression;
import org.apache.spark.ml.regression.LinearRegressionModel;
import org.apache.spark.ml.regression.LinearRegressionTrainingSummary;
import org.apache.spark.ml.linalg.Vectors;
import org.apache.spark.sql.Dataset;
import org.apache.spark.sql.Row;
import org.apache.spark.sql.SparkSession;


public class LinearReg {

        public static void main(String[] args) {
            System.out.println("Hello Spark");

            SparkConf conf = new SparkConf().setAppName("WBS-LinearRg").setMaster("local[4]");
            JavaSparkContext sc = new JavaSparkContext(conf);

            SparkSession spark = SparkSession
                    .builder()
                    .appName("JavaLinearRegressionWithElasticNetExample")
                    .getOrCreate();

            // $example on$
            // Load training data.
            Dataset<Row> training = spark.read().format("libsvm")
                    .load("/home/cristu/Proyectos/BigData/src/main/resources/linear_data.txt");

            training.toJavaRDD().take(30).forEach(System.out::println);


            LinearRegression lr = new LinearRegression()
                    .setMaxIter(10)
                    .setRegParam(0.3)
                    .setElasticNetParam(0.8);

            // Fit the model.
            LinearRegressionModel lrModel = lr.fit(training);

            // Print the coefficients and intercept for linear regression.
            System.out.println("Coefficients: "
                    + lrModel.coefficients() + " Intercept: " + lrModel.intercept());

            // Summarize the model over the training set and print out some metrics.
            LinearRegressionTrainingSummary trainingSummary = lrModel.summary();
            System.out.println("numIterations: " + trainingSummary.totalIterations());
            System.out.println("objectiveHistory: " + Vectors.dense(trainingSummary.objectiveHistory()));
            trainingSummary.residuals().show();
            System.out.println("RMSE: " + trainingSummary.rootMeanSquaredError());
            System.out.println("r2: " + trainingSummary.r2());
            // $example off$

            spark.stop();
        }
    }
