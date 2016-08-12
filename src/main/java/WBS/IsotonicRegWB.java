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
import org.apache.spark.ml.regression.IsotonicRegression;
import org.apache.spark.ml.regression.IsotonicRegressionModel;
import org.apache.spark.sql.Dataset;
import org.apache.spark.sql.Row;
import org.apache.spark.sql.SparkSession;
import org.apache.spark.util.LongAccumulator;
import scala.Tuple2;

import java.util.Arrays;

/**
 * Created by cristu on 12/08/16.
 */
public class IsotonicRegWB {
    public static void main(String[] args) {

        System.out.println("Hello Spark");

        SparkConf conf = new SparkConf().setAppName("WBS-IsotonicReg").setMaster("local[4]");

        JavaSparkContext sc = new JavaSparkContext(conf);

        SparkSession spark = SparkSession
                .builder()
                .appName("WBS-SQL-IsotonicReg")
                .master("local[4]")
                .getOrCreate();

        Dataset<Row> df = spark.read().json("/home/cristu/Proyectos/BigData/src/main/resources/social_reviews.json");
//    HOME
//        Dataset<Row> df = spark.read().json("/Users/NaranjO/Documents/TFG/IntelliJ/wbs/src/main/resources/social_reviews.json");

        Dataset<Row> dfrows = df.select("positive", "user_id")
                .filter(new FilterFunction<Row>() {
                    @Override
                    public boolean call(Row row) throws Exception {
                        return !row.isNullAt(0) || row.getString(1)=="36";
                    }
                });
        LongAccumulator accum = sc.sc().longAccumulator();

        JavaPairRDD<Double, Tuple2<Double, Double>> newpairrdd = dfrows.toJavaRDD().mapToPair(new PairFunction<Row, Double, Tuple2<Double, Double>>() {
            @Override
            public Tuple2<Double, Tuple2<Double, Double>> call(Row row) throws Exception {
                long count = 1;
                accum.add(count);
                return new Tuple2<>(new Double(row.getString(0)), new Tuple2<Double, Double>(new Double(row.getString(1)), new Double(accum.value())));
            }
        }).filter(new Function<Tuple2<Double, Tuple2<Double, Double>>, Boolean>() {
            @Override
            public Boolean call(Tuple2<Double, Tuple2<Double, Double>> doubleTuple2Tuple2) throws Exception {
                return doubleTuple2Tuple2._2()._2() <= 100.0;
            }
        });
        JavaRDD<LabeledPoint> data = newpairrdd.map(new Function<Tuple2<Double, Tuple2<Double, Double>>, LabeledPoint>() {
            @Override
            public LabeledPoint call(Tuple2<Double, Tuple2<Double, Double>> doubleTuple2Tuple2) throws Exception {
                return new LabeledPoint(doubleTuple2Tuple2._1(), Vectors.dense(doubleTuple2Tuple2._2()._2()));
            }
        });

        Dataset<Row> datarow = spark.createDataFrame(data, LabeledPoint.class);
        datarow.show();

        // Trains an isotonic regression model.
        IsotonicRegression ir = new IsotonicRegression();
        IsotonicRegressionModel model = ir.fit(datarow);



        System.out.println("Boundaries in increasing order: " + model.boundaries());
        System.out.println("Predictions associated with the boundaries: " + model.predictions());

        /*
        Boundaries in increasing order: [3080.0,10241.0,10243.0,22667.0,22668.0,22678.0,22679.0,27169.0,27170.0,27276.0,27277.0,70984.0,70985.0,71728.0,71729.0,73880.0,73881.0,73977.0,73978.0,74624.0,74625.0,75769.0,75770.0,81398.0,81399.0,82671.0,82672.0,83314.0,83315.0,83569.0]
        Predictions associated with the boundaries: [0.22998393112356685,0.22998393112356685,0.25205372410175264,0.25205372410175264,0.2531818181818182,0.2531818181818182,0.26272342594619674,0.26272342594619674,0.2644571428571475,0.2644571428571475,0.26821459633724404,0.26821459633724404,0.27513157894722434,0.27513157894722434,0.2770608736065247,0.2770608736065247,0.28413402061855836,0.28413402061855836,0.2922936630605219,0.2922936630605219,0.2929048034937499,0.2929048034937499,0.3058343033635063,0.3058343033635063,0.31628358209022395,0.31628358209022395,0.32495956454125047,0.32495956454125047,0.41894901960786146,0.41894901960786146]
         */

//         Makes predictions.
        System.out.println(ir.explainParams());
        model.transform(datarow).show();


    }
}
