package WBS;

import org.apache.spark.SparkConf;
import org.apache.spark.api.java.JavaDoubleRDD;
import org.apache.spark.api.java.JavaPairRDD;
import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.api.java.JavaSparkContext;
import org.apache.spark.api.java.function.FilterFunction;
import org.apache.spark.api.java.function.Function;
import org.apache.spark.api.java.function.PairFunction;
import org.apache.spark.mllib.linalg.Vectors;
import org.apache.spark.mllib.regression.LabeledPoint;
import org.apache.spark.mllib.regression.LinearRegressionModel;
import org.apache.spark.mllib.regression.LinearRegressionWithSGD;
import org.apache.spark.sql.Dataset;
import org.apache.spark.sql.Row;
import org.apache.spark.sql.SparkSession;
import scala.Tuple2;

/**
 * Created by NaranjO on 11/8/16.
 */
public class SVMmodel {
    public static void main(String[] args) {

        System.out.println("Hello Spark");

        SparkConf conf = new SparkConf().setAppName("WBS-SVMm").setMaster("local[4]");

        JavaSparkContext sc = new JavaSparkContext(conf);

        SparkSession spark = SparkSession
                .builder()
                .appName("WBS-SQL-SVMm")
                .master("local[4]")
                .getOrCreate();

//        Dataset<Row> df = spark.read().json("/home/cristu/Proyectos/BigData/src/main/resources/social_reviews.json");
//    HOME
        Dataset<Row> df = spark.read().json("/Users/NaranjO/Documents/TFG/IntelliJ/wbs/src/main/resources/social_reviews.json");

        Dataset<Row> dfrows = df.select("id", "positive", "user_id")
                .filter(new FilterFunction<Row>() {
                    @Override
                    public boolean call(Row row) throws Exception {
                        return !row.isNullAt(0);
                    }
                });
        JavaRDD<Row> rdd = dfrows.toJavaRDD();


        JavaPairRDD<Double, Tuple2<Double, Double>> newpairrdd = rdd.mapToPair(new PairFunction<Row, Double, Tuple2<Double, Double>>() {
            @Override
            public Tuple2<Double, Tuple2<Double, Double>> call(Row row) throws Exception {
                return new Tuple2<>(new Double(row.getString(1)), new Tuple2<Double, Double>(new Double(row.getString(2)), new Double(row.getString(0))));
            }
        });
        System.out.println(newpairrdd.count());
        JavaPairRDD<Double, Tuple2<Double, Double>> testdata = newpairrdd.filter(new Function<Tuple2<Double, Tuple2<Double, Double>>, Boolean>() {
            @Override
            public Boolean call(Tuple2<Double, Tuple2<Double, Double>> doubleTuple2Tuple2) throws Exception {
//                System.out.println(doubleTuple2Tuple2._1());
                return doubleTuple2Tuple2._2()._1() == 38.0;
            }
        });
        System.out.println(testdata.count());

        JavaRDD<LabeledPoint> data = testdata.map(new Function<Tuple2<Double, Tuple2<Double, Double>>, LabeledPoint>() {
            @Override
            public LabeledPoint call(Tuple2<Double, Tuple2<Double, Double>> doubleTuple2Tuple2) throws Exception {
                return new LabeledPoint(doubleTuple2Tuple2._1(), Vectors.dense(doubleTuple2Tuple2._2()._2()));
            }
        });
        data.take(10).forEach(System.out::println);
        // Building the model
        int numIterations = 10;
        double stepSize = 0.00000001;
        final LinearRegressionModel model =
                LinearRegressionWithSGD.train(JavaRDD.toRDD(data), numIterations, stepSize);

        // Evaluate model on training examples and compute training error
        JavaRDD<Tuple2<Double, Double>> valuesAndPreds = data.map(
                new Function<LabeledPoint, Tuple2<Double, Double>>() {
                    public Tuple2<Double, Double> call(LabeledPoint point) {
                        double prediction = model.predict(point.features());
                        return new Tuple2<Double, Double>(prediction, point.label());
                    }
                }
        );
        valuesAndPreds.take(10).forEach(System.out::println);
        double MSE = new JavaDoubleRDD(valuesAndPreds.map(
                new Function<Tuple2<Double, Double>, Object>() {
                    public Object call(Tuple2<Double, Double> pair) {
                        return Math.pow(pair._1() - pair._2(), 2.0);
                    }
                }
        ).rdd()).mean();
        System.out.println("training Mean Squared Error = " + MSE);

        //1-MSE = 1.2856389
        //2-MSE = 1.2856389809004655E106
        //3-MSE = 1.2854953064758255E106
    }
}
