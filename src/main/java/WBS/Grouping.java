package WBS;

import org.apache.spark.SparkConf;
import org.apache.spark.api.java.JavaDoubleRDD;
import org.apache.spark.api.java.JavaPairRDD;
import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.api.java.JavaSparkContext;
import org.apache.spark.api.java.function.Function;
import org.apache.spark.api.java.function.MapFunction;
import org.apache.spark.api.java.function.PairFunction;
import org.apache.spark.ml.linalg.Vector;
import org.apache.spark.mllib.linalg.Vectors;
import org.apache.spark.mllib.regression.IsotonicRegression;
import org.apache.spark.mllib.regression.IsotonicRegressionModel;
import org.apache.spark.mllib.regression.LabeledPoint;
import org.apache.spark.sql.Dataset;
import org.apache.spark.sql.RelationalGroupedDataset;
import org.apache.spark.sql.Row;
import org.apache.spark.sql.SparkSession;
import scala.Tuple2;
import scala.Tuple3;

/**
 * Created by cristu on 9/08/16.
 */
public class Grouping {
    public static void main(String[] args) {

        System.out.println("Hello Spark");

        SparkConf conf = new SparkConf().setAppName("WBS-grouping").setMaster("local[4]");
        JavaSparkContext sc = new JavaSparkContext(conf);

        SparkSession spark = SparkSession
                .builder()
                .appName("WBS-SQL-grouping")
                .master("local[4]")
                .getOrCreate();

        Dataset<Row> df = spark.read().json("/home/cristu/Proyectos/BigData/src/main/resources/social_reviews.json");
        Dataset<Row> dfrows = df.select("id", "positive", "user_id");
//        dfrows.toJavaRDD().take(30).forEach(System.out::println);
        JavaRDD<Row> posrdd = dfrows.toJavaRDD();

//        JavaRDD<LabeledPoint> parsedData = data.map(
//                new Function<String, LabeledPoint>() {
//                    public LabeledPoint call(String line) {
//                        String[] parts = line.split(",");
//                        String[] features = parts[1].split(" ");
//                        double[] v = new double[features.length];
//                        for (int i = 0; i < features.length - 1; i++) {
//                            v[i] = Double.parseDouble(features[i]);
//                        }
//                        return new LabeledPoint(Double.parseDouble(parts[0]), Vectors.dense(v));
//                    }
//                }
//        );
//        JavaRDD<LabeledPoint> labeldata = posrdd.map(new Function<Row, LabeledPoint>() {
//            @Override
//            public LabeledPoint call(Row row) throws Exception {
//                double[] v = new double[]
//
//
//
//                return null;
//            }
//        })





//        JavaPairRDD< Integer, Vectors> rddgrouped = posrdd.mapToPair(new PairFunction<Row, Integer, Vectors>() {
//            @Override
//            public Tuple2<Integer, Vectors> call(Row row) throws Exception {
//                if (row.isNullAt(0) || row.isNullAt(1)) {
//                    return new Tuple2<>(0, Vectors.dense(0.0,0.0));
//                } else {
//                    return new Tuple2<>(new Integer(row.getString(2),))
//                }
//            }
//        });

//
//        System.out.println(dfrows.count());
//
//        RelationalGroupedDataset testData =

//                Dataset<Row> olaqase = dfrows.filter(dfrows.col("user_id"));//groupBy(dfrows.col("user_id"));
//        Dataset<Row> olaqase = testData.count();
//        olaqase.toJavaRDD().take(30).forEach(System.out::println);

//        JavaRDD<Tuple3<Double, Double, Integer>> parsedData = posrdd.map(
//                new Function<Row, Tuple3<Double, Double, Integer>>() {
//                    @Override
//                    public Tuple3<Double, Double, Integer> call(Row row) throws Exception {
////                        System.out.println(row.isNullAt(0));
//                        if (row.isNullAt(0) || row.isNullAt(1)) {
//                            return new Tuple3<>(0.0, 0.0, 0);
//                        } else {
////                            System.out.println(row.getString(1));
////                            System.out.println(row.getString(0));
//                            return new Tuple3<>(new Double(row.getString(1)), new Double(row.getString(0)), new Integer(row.getString(2)));
//                        }
//                    }
//                });
//         JavaPairRDD<Tuple2<Integer, > testData =
//        olaqase.toJavaRDD().take(30).forEach(System.out::println);





//        parsedData.take(30).forEach(System.out::println);


//        // Split data into training (60%) and test (40%) sets.
//        JavaRDD<Tuple3<Double, Double, Double>>[] splits = parsedData.randomSplit(new double[]{0.6, 0.4}, 11L);
//        JavaRDD<Tuple3<Double, Double, Double>> training = splits[0];
//        JavaRDD<Tuple3<Double, Double, Double>> test = splits[1];
//
//        // Create isotonic regression model from training data.
//        // Isotonic parameter defaults to true so it is only shown for demonstration
//        final IsotonicRegressionModel model = new IsotonicRegression().setIsotonic(true).run(training);
//        // Create tuples of predicted and real labels.
//        JavaPairRDD<Double, Double> predictionAndLabel = test.mapToPair(
//                new PairFunction<Tuple3<Double, Double, Double>, Double, Double>() {
//                    @Override
//                    public Tuple2<Double, Double> call(Tuple3<Double, Double, Double> point) {
//                        Double predictedLabel = model.predict(point._2());
//                        return new Tuple2<Double, Double>(predictedLabel, point._1());
//                    }
//                }
//        );
//
//        // Calculate mean squared error between predicted and real labels.
//        Double meanSquaredError = new JavaDoubleRDD(predictionAndLabel.map(
//                new Function<Tuple2<Double, Double>, Object>() {
//                    @Override
//                    public Object call(Tuple2<Double, Double> pl) {
//                        return Math.pow(pl._1() - pl._2(), 2);
//                    }
//                }
//        ).rdd()).mean();
//        System.out.println("Mean Squared Error = " + meanSquaredError);


    }
}
