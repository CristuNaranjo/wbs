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
import org.apache.spark.ml.regression.GeneralizedLinearRegression;
import org.apache.spark.ml.regression.GeneralizedLinearRegressionModel;
import org.apache.spark.ml.regression.GeneralizedLinearRegressionTrainingSummary;
import org.apache.spark.sql.Dataset;
import org.apache.spark.sql.Row;
import org.apache.spark.sql.SparkSession;
import org.apache.spark.util.LongAccumulator;
import scala.Tuple2;

import java.util.Arrays;

/**
 * Created by cristu on 17/08/16.
 */
public class TestPosNeuNeg {

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

        Dataset<Row> dfrows = df.select("id", "positive", "user_id", "neutral", "negative")
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
        LongAccumulator accum = sc.sc().longAccumulator();

        JavaRDD<LabeledPoint> posrdd = dfrows.toJavaRDD().mapToPair(new PairFunction<Row, Double, Tuple2<Double, Double>>() {
            @Override
            public Tuple2<Double, Tuple2<Double, Double>> call(Row row) throws Exception {
                long count = 1;
                accum.add(count);
                return new Tuple2<>(new Double(row.getString(1)), new Tuple2<Double, Double>(new Double(row.getString(0)), new Double(accum.value())));
            }
        }).map(new Function<Tuple2<Double, Tuple2<Double, Double>>, LabeledPoint>() {
            @Override
            public LabeledPoint call(Tuple2<Double, Tuple2<Double, Double>> doubleTuple2Tuple2) throws Exception {
                return new LabeledPoint(doubleTuple2Tuple2._1(), Vectors.dense(doubleTuple2Tuple2._2()._2()));
            }
        });

        JavaRDD<LabeledPoint> neurdd = dfrows.toJavaRDD().mapToPair(new PairFunction<Row, Double, Tuple2<Double, Double>>() {
            @Override
            public Tuple2<Double, Tuple2<Double, Double>> call(Row row) throws Exception {
                long count = 1;
                accum.add(count);
                return new Tuple2<>(new Double(row.getString(3)), new Tuple2<Double, Double>(new Double(row.getString(0)), new Double(accum.value())));
            }
        }).map(new Function<Tuple2<Double, Tuple2<Double, Double>>, LabeledPoint>() {
            @Override
            public LabeledPoint call(Tuple2<Double, Tuple2<Double, Double>> doubleTuple2Tuple2) throws Exception {
                return new LabeledPoint(doubleTuple2Tuple2._1(), Vectors.dense(doubleTuple2Tuple2._2()._2()));
            }
        });

        JavaRDD<LabeledPoint> negrdd = dfrows.toJavaRDD().mapToPair(new PairFunction<Row, Double, Tuple2<Double, Double>>() {
            @Override
            public Tuple2<Double, Tuple2<Double, Double>> call(Row row) throws Exception {
                long count = 1;
                accum.add(count);
                return new Tuple2<>(new Double(row.getString(4)), new Tuple2<Double, Double>(new Double(row.getString(0)), new Double(accum.value())));
            }
        }).map(new Function<Tuple2<Double, Tuple2<Double, Double>>, LabeledPoint>() {
            @Override
            public LabeledPoint call(Tuple2<Double, Tuple2<Double, Double>> doubleTuple2Tuple2) throws Exception {
                return new LabeledPoint(doubleTuple2Tuple2._1(), Vectors.dense(doubleTuple2Tuple2._2()._2()));
            }
        });

        Dataset<Row> datapos = spark.createDataFrame(posrdd, LabeledPoint.class);
        Dataset<Row> dataneu = spark.createDataFrame(neurdd, LabeledPoint.class);
        Dataset<Row> dataneg = spark.createDataFrame(negrdd, LabeledPoint.class);

        System.out.println("Numero de datos positivos: " + datapos.count());
        System.out.println("Numero de datos neutrales: " + dataneu.count());
        System.out.println("Numero de datos negativos: " + dataneg.count());


        Dataset<Row>[] splits = datapos.randomSplit(new double[]{0.7, 0.3});
        Dataset<Row> trainingDataPos = splits[0];
        Dataset<Row> testDataPos = splits[1];

        Dataset<Row>[] splits1 = dataneu.randomSplit(new double[]{0.7, 0.3});
        Dataset<Row> trainingDataNeu = splits1[0];
        Dataset<Row> testDataNeu = splits1[1];
        Dataset<Row>[] splits2 = dataneg.randomSplit(new double[]{0.7, 0.3});
        Dataset<Row> trainingDataNeg = splits2[0];
        Dataset<Row> testDataNeg = splits2[1];


        GeneralizedLinearRegression glr = new GeneralizedLinearRegression()
                .setFamily("gaussian")
                .setLink("identity")
                .setMaxIter(100)
                .setRegParam(0);

        // Fit the model
        GeneralizedLinearRegressionModel modelPos = glr.fit(trainingDataPos);
        GeneralizedLinearRegressionModel modelNeu = glr.fit(trainingDataNeu);
        GeneralizedLinearRegressionModel modelNeg = glr.fit(trainingDataNeg);


//        // Print the coefficients and intercept for generalized linear regression model
//        System.out.println("Coefficients: " + model.coefficients());
//        System.out.println("Intercept: " + model.intercept());
//
//        // Summarize the model over the training set and print out some metrics
//        GeneralizedLinearRegressionTrainingSummary summary = model.summary();
//        System.out.println("Coefficient Standard Errors: "
//                + Arrays.toString(summary.coefficientStandardErrors()));
//        System.out.println("T Values: " + Arrays.toString(summary.tValues()));
//        System.out.println("P Values: " + Arrays.toString(summary.pValues()));
//        System.out.println("Dispersion: " + summary.dispersion());
//        System.out.println("Null Deviance: " + summary.nullDeviance());
//        System.out.println("Residual Degree Of Freedom Null: " + summary.residualDegreeOfFreedomNull());
//        System.out.println("Deviance: " + summary.deviance());
//        System.out.println("Residual Degree Of Freedom: " + summary.residualDegreeOfFreedom());
//        System.out.println("AIC: " + summary.aic());
//        System.out.println("Deviance Residuals: ");
//        summary.residuals().show();

        Dataset<Row> resultsPos = modelPos.transform(testDataPos).toDF("feautures", "labelPos", "predictionPos");
        Dataset<Row> resultsNeu = modelNeu.transform(testDataNeu).toDF("feautures", "labelNeu", "predictionNeu");
        Dataset<Row> resultsNeg = modelNeg.transform(testDataNeg).toDF("feautures", "labelNeg", "predictionNeg");

        resultsPos.createOrReplaceTempView("resultPos");
        resultsNeu.createOrReplaceTempView("resultNeu");
        resultsNeg.createOrReplaceTempView("resultNeg");

//        Dataset<Row> results = spark.sql("SELECT * FROM resultPos ");

//        JavaRDD<Row> results = resultsPos.toJavaRDD().union(resultsNeu.toJavaRDD());
//        results.take(10).forEach(System.out::println);


        resultsPos.show();
        resultsNeu.show();
        resultsNeg.show();
//
//        results.show();
//        results.printSchema();
//        Dataset<Row> rows = results.select("features", "label", "prediction");
//        for (Row r : rows.collectAsList()) {
//            System.out.println("(" + r.get(0) + ", " + r.get(1) + "), prediction=" + r.get(2));
//        }
    }
}
