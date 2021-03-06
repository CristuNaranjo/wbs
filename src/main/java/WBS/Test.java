package WBS;


/**
 * Created by cristu on 8/08/16.
 */

import org.apache.spark.SparkConf;
import org.apache.spark.api.java.JavaDoubleRDD;
import org.apache.spark.api.java.JavaPairRDD;
import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.api.java.JavaSparkContext;
import org.apache.spark.api.java.function.FilterFunction;
import org.apache.spark.api.java.function.Function;
import org.apache.spark.api.java.function.PairFunction;
import org.apache.spark.ml.feature.LabeledPoint;
import org.apache.spark.ml.linalg.Vectors;
import org.apache.spark.ml.regression.*;
import org.apache.spark.mllib.regression.LinearRegressionWithSGD;
import org.apache.spark.sql.Dataset;
import org.apache.spark.sql.Row;
import org.apache.spark.sql.SparkSession;
import org.apache.spark.util.LongAccumulator;
import scala.Tuple2;
import scala.Tuple3;

import java.util.Arrays;


public class Test {

    public static void main(String[] args) {

        System.out.println("Hello Spark");
        SparkConf conf = new SparkConf().setAppName("Test").setMaster("local[4]");
        JavaSparkContext sc = new JavaSparkContext(conf);
        SparkSession spark = SparkSession
                .builder()
                .appName("Test-SQL")
                .master("local[4]")
                .getOrCreate();

        Dataset<Row> df = spark.read().json("/home/cristu/Proyectos/BigData/src/main/resources/social_reviews.json");
//    HOME
//        Dataset<Row> df = spark.read().json("/Users/NaranjO/Documents/TFG/IntelliJ/wbs/src/main/resources/social_reviews.json");

        Dataset<Row> dfrows = df.select("id", "positive", "user_id")
                .filter(new FilterFunction<Row>() {
                    @Override
                    public boolean call(Row row) throws Exception {
                        Double x = 0.0;
                        if (!row.isNullAt(0)) {
                            x = new Double(row.getString(2));
                        }
                        return !row.isNullAt(0) && x == 36.0;
                    }
                });
//        System.out.println(dfrows.count());
        LongAccumulator accum = sc.sc().longAccumulator();
        JavaPairRDD<Double, Tuple2<Double, Double>> newpairrdd = dfrows.toJavaRDD().mapToPair(new PairFunction<Row, Double, Tuple2<Double, Double>>() {
            @Override
            public Tuple2<Double, Tuple2<Double, Double>> call(Row row) throws Exception {
                long count = 1;
                accum.add(count);
                return new Tuple2<>(new Double(row.getString(1)), new Tuple2<Double, Double>(new Double(row.getString(0)), new Double(accum.value())));
            }
        });
        final long count = newpairrdd.count();
        JavaPairRDD<Double, Tuple2<Double, Double>> newpairrdd2 = newpairrdd

                .filter(new Function<Tuple2<Double, Tuple2<Double, Double>>, Boolean>() {
            @Override
            public Boolean call(Tuple2<Double, Tuple2<Double, Double>> doubleTuple2Tuple2) throws Exception {
//                long i = 100;
//                long x = count - 1000;
                long x = 0;
                return doubleTuple2Tuple2._2()._2() >= x;
            }
        });
        JavaRDD<LabeledPoint> data = newpairrdd2.map(new Function<Tuple2<Double, Tuple2<Double, Double>>, LabeledPoint>() {
            @Override
            public LabeledPoint call(Tuple2<Double, Tuple2<Double, Double>> doubleTuple2Tuple2) throws Exception {
                return new LabeledPoint(doubleTuple2Tuple2._1(), Vectors.dense(doubleTuple2Tuple2._2()._2()));
            }
        });

//        data.take(20).forEach(System.out::println);
        Dataset<Row> datarow = spark.createDataFrame(data, LabeledPoint.class);
        System.out.println("Numero de datos: " + datarow.count());
        datarow.show();


        Dataset<Row>[] splits = datarow.randomSplit(new double[]{0.7, 0.3});
        Dataset<Row> trainingData = splits[0];
        Dataset<Row> testData = splits[1];


  /*      LinearRegression lr = new LinearRegression()
                .setMaxIter(1000)
                .setRegParam(0.3)
                .setElasticNetParam(0.8)
                .setFitIntercept(true)
                .setStandardization(true);

//        System.out.println("Columna de peswossosossosososso:  "  +  lr.getWeightCol());
//
    // Fit the model.
        LinearRegressionModel lrModel = lr.fit(trainingData);

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

        //Predictions

        LinearRegressionSummary modelSummary = lrModel.evaluate(testData);
        System.out.println("T-Values: " + modelSummary.tValues());
//        System.out.println("P-Values: " + Vectors.dense(modelSummary.pValues()));
        System.out.println("Residuos:  ");
        modelSummary.residuals().show();
        System.out.println("RMSE: " + modelSummary.rootMeanSquaredError());
        System.out.println("r2: " + modelSummary.r2());
        System.out.println("Predicciones: ");
        modelSummary.predictions().show();
        System.out.println("Numero de predicciones: " + modelSummary.predictions().count());;
*/




        GeneralizedLinearRegression glr = new GeneralizedLinearRegression()
                .setFamily("gaussian")
                .setLink("identity")
                .setMaxIter(100)
                .setRegParam(0);

        // Fit the model
        GeneralizedLinearRegressionModel model = glr.fit(trainingData);
        // Print the coefficients and intercept for generalized linear regression model
        System.out.println("Coefficients: " + model.coefficients());
        System.out.println("Intercept: " + model.intercept());

        // Summarize the model over the training set and print out some metrics
        GeneralizedLinearRegressionTrainingSummary summary = model.summary();
        System.out.println("Coefficient Standard Errors: "
                + Arrays.toString(summary.coefficientStandardErrors()));
        System.out.println("T Values: " + Arrays.toString(summary.tValues()));
        System.out.println("P Values: " + Arrays.toString(summary.pValues()));
        System.out.println("Dispersion: " + summary.dispersion());
        System.out.println("Null Deviance: " + summary.nullDeviance());
        System.out.println("Residual Degree Of Freedom Null: " + summary.residualDegreeOfFreedomNull());
        System.out.println("Deviance: " + summary.deviance());
        System.out.println("Residual Degree Of Freedom: " + summary.residualDegreeOfFreedom());
        System.out.println("AIC: " + summary.aic());
        System.out.println("Deviance Residuals: ");
        summary.residuals().show();

        Dataset<Row> results =  model.transform(testData);
        Dataset<Row> rows = results.select("features", "label", "prediction");
        for (Row r: rows.collectAsList()) {
            System.out.println("(" + r.get(0) + ", " + r.get(1) + "), prediction=" + r.get(2));
        }

       /* //([375.0], 0.551), prediction=0.2962022472445679
        ([377.0], 0.254), prediction=0.2962947889191027
        ([378.0], 0.416), prediction=0.29634105975637015*/


        /*60-40
        * ([371.0], 0.179), prediction=0.29018383827825733
([373.0], 0.171), prediction=0.29029305237162706
([376.0], 0.402), prediction=0.29045687351168165*/

    }
}
