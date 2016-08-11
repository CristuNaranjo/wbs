package WBS;

import org.apache.spark.SparkConf;

import org.apache.spark.api.java.JavaDoubleRDD;
import org.apache.spark.api.java.JavaPairRDD;
import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.api.java.JavaSparkContext;
import org.apache.spark.api.java.function.FilterFunction;
import org.apache.spark.api.java.function.Function;
import org.apache.spark.api.java.function.PairFunction;
import org.apache.spark.ml.feature.VectorAssembler;
import org.apache.spark.ml.regression.LinearRegression;
import org.apache.spark.ml.regression.LinearRegressionTrainingSummary;
import org.apache.spark.mllib.linalg.DenseVector;
import org.apache.spark.mllib.regression.*;
import org.apache.spark.sql.Dataset;
import org.apache.spark.sql.Row;
import org.apache.spark.sql.SparkSession;
import org.apache.spark.ml.feature.VectorAssembler;
import org.apache.spark.mllib.linalg.VectorUDT;
import org.apache.spark.mllib.linalg.Vectors;
import org.apache.spark.sql.Dataset;
import org.apache.spark.sql.Row;
import org.apache.spark.sql.RowFactory;
import org.apache.spark.sql.types.*;
import scala.Tuple2;
import scala.Tuple3;

import static org.apache.spark.sql.types.DataTypes.*;

/**
 * Created by cristu on 10/08/16.
 */
public final class IsotonicReg {
//    public static class AvgCount implements Serializable {
//
//        public AvgCount(Double total, int num) {
////            total_ = total;
//            total_.add(total);
//            num_ = num;
//        }
////        public double[] total_;
//        public int num_;
//        public List<Double> total_ = new ArrayList<Double>();
////        public float avg() {
////            return (float) total_ / (float) num_; }
//    }
    public static void main(String[] args) {

        System.out.println("Hello Spark");

        SparkConf conf = new SparkConf().setAppName("WBS-isotonic").setMaster("local[4]");
        JavaSparkContext sc = new JavaSparkContext(conf);

        SparkSession spark = SparkSession
                .builder()
                .appName("WBS-SQL-isotonic")
                .master("local[4]")
                .getOrCreate();

        Dataset<Row> df = spark.read().json("/home/cristu/Proyectos/BigData/src/main/resources/social_reviews.json");


        Dataset<Row> dfrows = df.select("id", "positive", "user_id")
            .filter(new FilterFunction<Row>() {
                @Override
                public boolean call(Row row) throws Exception {
                    return !row.isNullAt(0);
                }
            });
         JavaRDD<Row> rdd = dfrows.toJavaRDD();


        JavaPairRDD<Double, Tuple2<Double,Double>> newpairrdd = rdd.mapToPair(new PairFunction<Row, Double, Tuple2<Double, Double>>() {
            @Override
            public Tuple2<Double, Tuple2<Double, Double>> call(Row row) throws Exception {
                return new Tuple2<>(new Double(row.getString(1)), new Tuple2<Double, Double>(new Double(row.getString(2)),new Double(row.getString(0))));
            }
        });

        JavaRDD<LabeledPoint> data = newpairrdd.map(new Function<Tuple2<Double, Tuple2<Double, Double>>, LabeledPoint>() {
            @Override
            public LabeledPoint call(Tuple2<Double, Tuple2<Double, Double>> doubleTuple2Tuple2) throws Exception {
                return new LabeledPoint(doubleTuple2Tuple2._1(),Vectors.dense(doubleTuple2Tuple2._2()._1(), doubleTuple2Tuple2._2()._2()));
            }
        });

        data.take(10).forEach(System.out::println);

        // ISOTONIC REGRESSION

        // Create label, feature, weight tuples from input data with weight set to default value 1.0.
        JavaRDD<Tuple3<Double, Double, Double>> parsedData = data.map(
                new Function<LabeledPoint, Tuple3<Double, Double, Double>>() {
                    public Tuple3<Double, Double, Double> call(LabeledPoint point) {
                        return new Tuple3<>(new Double(point.label()),
                                new Double(point.features().apply(1)), 1.0);
                    }
                }
        );


        // Split data into training (60%) and test (40%) sets.
        JavaRDD<Tuple3<Double, Double, Double>>[] splits =
                parsedData.randomSplit(new double[]{0.6, 0.4}, 11L);
        JavaRDD<Tuple3<Double, Double, Double>> training = splits[0];
        JavaRDD<Tuple3<Double, Double, Double>> test = splits[1];

        // Create isotonic regression model from training data.
        // Isotonic parameter defaults to true so it is only shown for demonstration
        final IsotonicRegressionModel model =
                new IsotonicRegression().setIsotonic(true).run(training);

        // Create tuples of predicted and real labels.
        JavaPairRDD<Double, Double> predictionAndLabel = test.mapToPair(
                new PairFunction<Tuple3<Double, Double, Double>, Double, Double>() {
                    @Override
                    public Tuple2<Double, Double> call(Tuple3<Double, Double, Double> point) {
                        Double predictedLabel = model.predict(point._2());
                        return new Tuple2<>(predictedLabel, point._1());
                    }
                }
        );

// Calculate mean squared error between predicted and real labels.
        Double meanSquaredError = new JavaDoubleRDD(predictionAndLabel.map(
                new Function<Tuple2<Double, Double>, Object>() {
                    @Override
                    public Object call(Tuple2<Double, Double> pl) {
                        return Math.pow(pl._1() - pl._2(), 2);
                    }
                }
        ).rdd()).mean();
        System.out.println("Mean Squared Error = " + meanSquaredError);










        //LINEAR REGRESSION

//        // Building the model
//        int numIterations = 100;
//        double stepSize = 0.00000001;
//        final LinearRegressionModel model =
//                LinearRegressionWithSGD.train(JavaRDD.toRDD(data), numIterations, stepSize);
//
//        // Evaluate model on training examples and compute training error
//        JavaRDD<Tuple2<Double, Double>> valuesAndPreds = data.map(
//                new Function<LabeledPoint, Tuple2<Double, Double>>() {
//                    public Tuple2<Double, Double> call(LabeledPoint point) {
//                        double prediction = model.predict(point.features());
//                        return new Tuple2<>(prediction, point.label());
//                    }
//                }
//        );
//        double MSE = new JavaDoubleRDD(valuesAndPreds.map(
//                new Function<Tuple2<Double, Double>, Object>() {
//                    public Object call(Tuple2<Double, Double> pair) {
//                        return Math.pow(pair._1() - pair._2(), 2.0);
//                    }
//                }
//        ).rdd()).mean();
//        System.out.println("training Mean Squared Error = " + MSE);

    }
}