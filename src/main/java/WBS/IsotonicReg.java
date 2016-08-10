package WBS;

import org.apache.spark.SparkConf;
import org.apache.spark.api.java.JavaDoubleRDD;
import org.apache.spark.api.java.JavaPairRDD;
import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.api.java.JavaSparkContext;
import org.apache.spark.api.java.function.Function;
import org.apache.spark.api.java.function.Function2;
import org.apache.spark.api.java.function.PairFunction;
import org.apache.spark.api.java.function.VoidFunction;
import org.apache.spark.ml.regression.IsotonicRegression;
import org.apache.spark.ml.regression.IsotonicRegressionModel;
import org.apache.spark.mllib.regression.LabeledPoint;
import org.apache.spark.sql.Dataset;
import org.apache.spark.sql.Row;
import org.apache.spark.sql.SparkSession;
import scala.Serializable;
import scala.Tuple2;
import scala.Tuple3;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.stream.Stream;

/**
 * Created by cristu on 10/08/16.
 */
public final class IsotonicReg {
    public static class AvgCount implements Serializable {

        public AvgCount(Double total, int num) {
//            total_ = total;
            total_.add(total);
            num_ = num;
        }
//        public double[] total_;
        public int num_;
        public List<Double> total_ = new ArrayList<Double>();
//        public float avg() {
//            return (float) total_ / (float) num_; }
    }
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
        Dataset<Row> dfrows = df.select("id", "positive", "user_id");
        JavaRDD<Row> dfrdd = dfrows.toJavaRDD();
        JavaPairRDD<Integer, Tuple2<Double, Double>> pairrdd = dfrdd.mapToPair(new PairFunction<Row, Integer, Tuple2<Double, Double>>() {
            @Override
            public Tuple2<Integer, Tuple2<Double, Double>> call(Row row) throws Exception {
                if (row.isNullAt(0) || row.isNullAt(1)) {
                    return new Tuple2<>(0, new Tuple2<>(0.0, 0.0));
                } else {
                    return new Tuple2<>(new Integer(row.getString(2)), new Tuple2<>(new Double(row.getString(1)), new Double(row.getString(0))));
                }
            }
        });
        pairrdd.take(10).forEach(System.out::println);

        JavaPairRDD<Integer, Iterable<Tuple2<Double, Double>>> data = pairrdd.groupByKey();
        System.out.println("Lineas: " + data.count());//10 lineas

        data.take(2).forEach(System.out::println);
       JavaPairRDD<Integer, Double[]> testdata = pairrdd.reduceByKey(
               new Function2 <Integer, Tuple2<Double, Double>, Tuple2<Integer, Double[]>>(){
                    @Override
                    public Tuple2<Integer, Double[]> call(Integer y, Tuple2<Double, Double> x) throws Exception{
                            //data, data,data, data
                        return ;
                    }
               });


//        Function<Double, AvgCount> createAcc = new Function<Double, AvgCount>() {
//            @Override
//            public AvgCount call(Double x) {
//                return new AvgCount(x, 1);
//            }
//        };
//        Function2<AvgCount, Double, AvgCount> addAndCount = new Function2<AvgCount, Double, AvgCount>() {
//            @Override
//            public AvgCount call(AvgCount a, Double x) {
//                a.total_.add(x);
//                a.num_ += 1;
//                return a;
//            }
//        };
//        Function2<AvgCount, AvgCount, AvgCount> combine =
//                new Function2<AvgCount, AvgCount, AvgCount>() {
//                    public AvgCount call(AvgCount a, AvgCount b) {
//                        public List<Double> mergetotal_ = new ArrayList<Double>();
//                        mergetotal_.addAll(a.total_,b.total_);
//                        a.total_ = mergetotal_;
////                        a.total_ += b.total_;
//                        a.num_ += b.num_;
//                        return a;
//                    }
//                };
//        AvgCount initial = new AvgCount(0,0);
//
//        JavaPairRDD<String, AvgCount> avgCounts =
//                data.combineByKey(createAcc, addAndCount, combine);
//        pairrdd.combineByKey(createAcc, addAndCount, combine);

//        JavaPairRDD<Double>
//
//        JavaRDD<LabeledPoint> datatest = data.map(new Function<Tuple2<Integer, Iterable<Tuple2<Double, Double>>>, LabeledPoint>() {
//            @Override
//            public LabeledPoint call(Tuple2<Integer, Iterable<Tuple2<Double, Double>>> integerIterableTuple2) throws Exception {
//                integerIterableTuple2
//            }
//        })

//        JavaRDD<Tuple3<Double, Double, Double>> parsedData = data.map(new Function<Tuple2<Integer, Iterable<Tuple2<Double, Double>>>, Tuple3<Double, Double, Double>>() {
//            @Override
//            public Tuple3<Double, Double, Double> call(Tuple2<Integer, Iterable<Tuple2<Double, Double>>> integerIterableTuple2) throws Exception {
//                return null;
//            }
//        })
        // Create label, feature, weight tuples from input data with weight set to default value 1.0.
//        JavaRDD<Tuple3<Double, Double, Double>> parsedData = data.map(
//                new Function<LabeledPoint, Tuple3<Double, Double, Double>>() {
//                    public Tuple3<Double, Double, Double> call(LabeledPoint point) {
//                        return new Tuple3<>(new Double(point.label()),
//                                new Double(point.features().apply(0)), 1.0);
//                    }
//                }
//        );
//        // Split data into training (60%) and test (40%) sets.
//        JavaRDD<Tuple3<Double, Double, Double>>[] splits = parseData.randomSplit(new double[]{0.6, 0.4}, 11L);
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