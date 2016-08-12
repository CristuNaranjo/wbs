package WBS;

import org.apache.spark.SparkConf;
import org.apache.spark.api.java.JavaPairRDD;
import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.api.java.JavaSparkContext;
import org.apache.spark.api.java.function.FilterFunction;
import org.apache.spark.api.java.function.Function;
import org.apache.spark.api.java.function.PairFunction;
import org.apache.spark.ml.feature.LabeledPoint;
import org.apache.spark.ml.linalg.Vectors;
import org.apache.spark.sql.Dataset;
import org.apache.spark.sql.Row;
import org.apache.spark.sql.SparkSession;
import scala.Tuple2;
import org.apache.spark.ml.regression.GeneralizedLinearRegression;
import org.apache.spark.ml.regression.GeneralizedLinearRegressionModel;
import org.apache.spark.ml.regression.GeneralizedLinearRegressionTrainingSummary;

import java.util.Arrays;

/**
 * Created by NaranjO on 11/8/16.
 */
public class GenLinReg {
    public static void main(String[] args) {

        System.out.println("Hello Spark");

        SparkConf conf = new SparkConf().setAppName("WBS-SVMm").setMaster("local[4]");

        JavaSparkContext sc = new JavaSparkContext(conf);

        SparkSession spark = SparkSession
                .builder()
                .appName("WBS-SQL-SVMm")
                .master("local[4]")
                .getOrCreate();

        Dataset<Row> df = spark.read().json("/home/cristu/Proyectos/BigData/src/main/resources/social_reviews.json");
//    HOME
//        Dataset<Row> df = spark.read().json("/Users/NaranjO/Documents/TFG/IntelliJ/wbs/src/main/resources/social_reviews.json");

        Dataset<Row> dfrows = df.select("id", "positive", "user_id")
                .filter(new FilterFunction<Row>() {
                    @Override
                    public boolean call(Row row) throws Exception {
                        return !row.isNullAt(0) || row.getString(2)=="38";
                    }
                });

        JavaRDD<Row> rdd = dfrows.toJavaRDD();
//        System.out.println(dfrows.count());

        JavaPairRDD<Double, Tuple2<Double, Double>> newpairrdd = rdd.mapToPair(new PairFunction<Row, Double, Tuple2<Double, Double>>() {
            @Override
            public Tuple2<Double, Tuple2<Double, Double>> call(Row row) throws Exception {
                return new Tuple2<>(new Double(row.getString(1)), new Tuple2<Double, Double>(new Double(row.getString(2)), new Double(row.getString(0))));
            }
        });
//        System.out.println(newpairrdd.count());
//        JavaPairRDD<Double, Tuple2<Double, Double>> testdata = newpairrdd.filter(new Function<Tuple2<Double, Tuple2<Double, Double>>, Boolean>() {
//            @Override
//            public Boolean call(Tuple2<Double, Tuple2<Double, Double>> doubleTuple2Tuple2) throws Exception {
////                System.out.println(doubleTuple2Tuple2._1());
//                return doubleTuple2Tuple2._2()._1() == 38.0;
//            }
//        });
//        System.out.println(testdata.count());

        JavaRDD<LabeledPoint> data = newpairrdd.map(new Function<Tuple2<Double, Tuple2<Double, Double>>, LabeledPoint>() {
            @Override
            public LabeledPoint call(Tuple2<Double, Tuple2<Double, Double>> doubleTuple2Tuple2) throws Exception {
                return new LabeledPoint(doubleTuple2Tuple2._1(), Vectors.dense(doubleTuple2Tuple2._2()._2()));
            }
        });

        Dataset<Row> datarow = spark.createDataFrame(data, LabeledPoint.class);






        GeneralizedLinearRegression glr = new GeneralizedLinearRegression()
                .setFamily("gaussian")
                .setLink("identity")
                .setMaxIter(10)
                .setRegParam(0.3);

    // Fit the model
        GeneralizedLinearRegressionModel model = glr.fit(datarow);

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

//        datarow.show();
//        Dataset<Row> test = datarow.groupBy(datarow.col("label")).agg();
//        test.show();
//        data.take(10).forEach(System.out::println);
//        // Building the model
//        int numIterations = 10;
//        double stepSize = 0.00000001;
//        final LinearRegressionModel model =
//                LinearRegressionWithSGD.train(JavaRDD.toRDD(data), numIterations, stepSize);
//
//        // Evaluate model on training examples and compute training error
//        JavaRDD<Tuple2<Double, Double>> valuesAndPreds = data.map(
//                new Function<LabeledPoint, Tuple2<Double, Double>>() {
//                    public Tuple2<Double, Double> call(LabeledPoint point) {
//                        double prediction = model.predict(point.features());
//                        return new Tuple2<Double, Double>(prediction, point.label());
//                    }
//                }
//        );
//        valuesAndPreds.take(10).forEach(System.out::println);
//        double MSE = new JavaDoubleRDD(valuesAndPreds.map(
//                new Function<Tuple2<Double, Double>, Object>() {
//                    public Object call(Tuple2<Double, Double> pair) {
//                        return Math.pow(pair._1() - pair._2(), 2.0);
//                    }
//                }
//        ).rdd()).mean();
//        System.out.println("training Mean Squared Error = " + MSE);

                //1-MSE = 1.2856389
                //2-MSE = 1.2856389809004655E106
                //3-MSE = 1.2854953064758255E106
    }
}
