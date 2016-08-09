/**
 * Created by cristu on 5/08/16.
 */
package WBS;


import org.apache.spark.api.java.JavaSparkContext;
import org.apache.spark.SparkConf;
import org.apache.spark.sql.Dataset;
import org.apache.spark.sql.Row;
import org.apache.spark.sql.SparkSession;
import scala.Tuple2;
import scala.Tuple3;
import org.apache.spark.api.java.function.Function;
import org.apache.spark.api.java.function.PairFunction;
import org.apache.spark.api.java.JavaDoubleRDD;
import org.apache.spark.api.java.JavaPairRDD;
import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.mllib.regression.IsotonicRegression;
import org.apache.spark.mllib.regression.IsotonicRegressionModel;
import org.apache.spark.mllib.regression.LabeledPoint;
import org.apache.spark.mllib.util.MLUtils;



public class Main {
    public static void main(String[] args) {

        System.out.println("Hello Spark");

        SparkConf conf = new SparkConf().setAppName("WBS").setMaster("local[4]");
        JavaSparkContext sc = new JavaSparkContext(conf);

        SparkSession spark = SparkSession
                .builder()
                .appName("WBS-SQL")
                .master("local[4]")
                .getOrCreate();

        Dataset<Row> df = spark.read().json("/home/cristu/Proyectos/BigData/src/main/resources/social_reviews.json");
        Dataset<Row> pos = df.select("id", "positive", "user_id");
        JavaRDD<Row> posrdd = pos.toJavaRDD();


        JavaRDD<Tuple3<Double, Double, Double>> parsedData = posrdd.map(
                new Function<Row, Tuple3<Double, Double, Double>>() {
                    @Override
                    public Tuple3<Double, Double, Double> call(Row row) throws Exception {
//                        System.out.println(row.isNullAt(0));
                        if (row.isNullAt(0) || row.isNullAt(1)) {
                            return new Tuple3<>(0.0, 0.0, 0.0);
                        } else {
//                            System.out.println(row.getString(1));
//                            System.out.println(row.getString(0));
                            return new Tuple3<>(new Double(row.getString(1)), new Double(row.getString(0)), 1.0);
                        }
                    }
                });

        // Split data into training (60%) and test (40%) sets.
        JavaRDD<Tuple3<Double, Double, Double>>[] splits = parsedData.randomSplit(new double[]{0.6, 0.4}, 11L);
        JavaRDD<Tuple3<Double, Double, Double>> training = splits[0];
        JavaRDD<Tuple3<Double, Double, Double>> test = splits[1];

        // Create isotonic regression model from training data.
        // Isotonic parameter defaults to true so it is only shown for demonstration
        final IsotonicRegressionModel model = new IsotonicRegression().setIsotonic(true).run(training);
        // Create tuples of predicted and real labels.
        JavaPairRDD<Double, Double> predictionAndLabel = test.mapToPair(
                new PairFunction<Tuple3<Double, Double, Double>, Double, Double>() {
                    @Override
                    public Tuple2<Double, Double> call(Tuple3<Double, Double, Double> point) {
                        Double predictedLabel = model.predict(point._2());
                        return new Tuple2<Double, Double>(predictedLabel, point._1());
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





//
//        Dataset<Double> posdouble = pos.map(new MapFunction<Row, Double>() {
//            @Override
//            public Double call(Row row) throws Exception {
//                if(row.isNullAt(1)){
//                    return 0.0;
//                }else{
//                    return row.getDouble(1);
//                }
//
//            }
//        }, Encoders.DOUBLE());
//        Dataset<Double> iddouble = pos.map(new MapFunction<Row, Double>() {
//            @Override
//            public Double call(Row row) throws Exception {
//                if(row.isNullAt(0)){
//                    return 0.0;
//                }else{
//                    return row.getDouble(0);
//                }
//            }
//        }, Encoders.DOUBLE());

//        org.apache.spark.ml.linalg.Vector mlVec = pos.col("positive")
//        Dataset<Tuple3<Double,Double,Double>> tuple3test = pos.map(new MapFunction<Row, Tuple3<Double, Double, Double>>() {
//            @Override
//            public Tuple3<Double, Double, Double> call(Row row) throws Exception {
//                if (row.isNullAt(0) || row.isNullAt(1)) {
//                    return new Tuple3<>(0.0, 0.0, 0.0);
//                } else {
//                    return new Tuple3<>(row.getDouble(0), row.getDouble(1), 1.0);
//                }
//            }
//        }, Encoders.tuple(Encoders.DOUBLE(), Encoders.DOUBLE(),Encoders.DOUBLE()));
//        JavaRDD<Double> rddpos = posdouble.toJavaRDD();
//        JavaRDD<Double> rddid = iddouble.toJavaRDD();
//        JavaPairRDD.
//        Dataset<Tuple2<Double,Double>> testtuple = iddouble.join(posdouble);
//        JavaPairRDD<Double,Double> testpair = JavaPairRDD.fromJavaRDD(rddid,rddpos);
     /*   JavaRDD<Row> rdd = pos.toJavaRDD();
        JavaRDD<Tuple3<Double, Double, Double>> parsedData = rdd.map(
                new Function<Row, Tuple3<Double, Double, Double>>() {
                    public Tuple3<Double, Double, Double> call(Row row) {
//                        System.out.println(line.getAs("positive").toString());
//                        String[] parts = {line.getAs("positive").toString(),line.getAs("id").toString()};
//                        System.out.println(line);
//                        System.out.println(line.mkString(",").split(","));
//                        String[] parts = line.mkString(",").split(",");
                        if(row.isNullAt(0) || row.isNullAt(1)){
                            return new Tuple3<>(0.0, 0.0, 0.0);
                        }
                        return new Tuple3<>(new Double(row.getString(1)), new Double(row.getString(0)), 1.0);
//                        return new Tuple3<>(new Double(parts[0]), new Double(parts[1]), 1.0);
//                        return new Tuple3<>(new Double(1.0), new Double(2.0), 1.0);
                    }
                }
        );*/


//        LinearRegression lr = new LinearRegression()
//                .setMaxIter(10)
//                .setRegParam(0.3)
//                .setElasticNetParam(0.8);
//        // Fit the model.
//        LinearRegressionModel lrModel = lr.fit(pos);
//
//        // Print the coefficients and intercept for linear regression.
//        System.out.println("Coefficients: "
//                + lrModel.coefficients() + " Intercept: " + lrModel.intercept());


//        JavaRDD<Row> test = pos.javaRDD();
//        JavaRDD

//        JavaRDD<LabeledPoint> parsedData = pos.map(
//                new Function<Row, LabeledPoint>() {
//                    public LabeledPoint call(Row){
//
//                        return new LabeledPoint();
//                    }
//                }
//        );

//        pos.show();
//        df.show();


//        LogisticRegression lr = new LogisticRegression().setMaxIter(1000);

//        System.out.println("LogisticRegression parameters:\n" + lr.explainParams() + "\n");
//        LogisticRegressionModel model = lr.fit(df);
//        Vector weights = model.weightCol();
//
//        model.transform(df).show();


    }
}
