package WBS;

import org.apache.spark.SparkConf;
import org.apache.spark.api.java.JavaPairRDD;
import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.api.java.JavaSparkContext;
import org.apache.spark.api.java.function.FilterFunction;
import org.apache.spark.api.java.function.ForeachFunction;
import org.apache.spark.api.java.function.Function;
import org.apache.spark.api.java.function.PairFunction;
import org.apache.spark.ml.feature.LabeledPoint;
import org.apache.spark.ml.linalg.Vectors;
import org.apache.spark.ml.regression.GeneralizedLinearRegression;
import org.apache.spark.ml.regression.GeneralizedLinearRegressionModel;
import org.apache.spark.ml.regression.GeneralizedLinearRegressionSummary;
import org.apache.spark.ml.regression.GeneralizedLinearRegressionTrainingSummary;
import org.apache.spark.sql.Dataset;
import org.apache.spark.sql.Row;
import org.apache.spark.sql.SparkSession;
import org.apache.spark.util.LongAccumulator;
import scala.Function1;
import scala.Tuple2;

import java.util.Arrays;
import java.util.List;

import org.math.plot.*;
import javax.swing.*;

/**
 * Created by cristu on 12/08/16.
 */
public class LinRegWB {

    public static void main(String[] args) {

        System.out.println("Hello Spark");

        SparkConf conf = new SparkConf().setAppName("WBS-LinReg").setMaster("local[4]");

        JavaSparkContext sc = new JavaSparkContext(conf);

        SparkSession spark = SparkSession
                .builder()
                .appName("WBS-SQL-LinReg")
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
                        if(!row.isNullAt(0)){
                            x = new Double (row.getString(2));
                        }
                        return !row.isNullAt(0) && x==44.0;
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
        JavaRDD<LabeledPoint> data = newpairrdd.map(new Function<Tuple2<Double, Tuple2<Double, Double>>, LabeledPoint>() {
            @Override
            public LabeledPoint call(Tuple2<Double, Tuple2<Double, Double>> doubleTuple2Tuple2) throws Exception {
                return new LabeledPoint(doubleTuple2Tuple2._1(), Vectors.dense(doubleTuple2Tuple2._2()._2()));
            }
        });
//
//        JavaPairRDD<Double, Tuple2<Double, Double>> newpairrdd = dfrows.toJavaRDD().mapToPair(new PairFunction<Row, Double, Tuple2<Double, Double>>() {
//            @Override
//            public Tuple2<Double, Tuple2<Double, Double>> call(Row row) throws Exception {
//                return new Tuple2<>(new Double(row.getString(1)), new Tuple2<Double, Double>(new Double(row.getString(2)), new Double(row.getString(0))));
//            }
//        });
//        JavaRDD<LabeledPoint> data = newpairrdd.map(new Function<Tuple2<Double, Tuple2<Double, Double>>, LabeledPoint>() {
//            @Override
//            public LabeledPoint call(Tuple2<Double, Tuple2<Double, Double>> doubleTuple2Tuple2) throws Exception {
//                return new LabeledPoint(doubleTuple2Tuple2._1(), Vectors.dense(doubleTuple2Tuple2._2()._2()));
//            }
//        });

        Dataset<Row> datarow = spark.createDataFrame(data, LabeledPoint.class);

        GeneralizedLinearRegression glr = new GeneralizedLinearRegression()
                .setFamily("gaussian")
                .setLink("identity")
                .setMaxIter(100)
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

        System.out.println(glr.explainParams());




//        evaluate

        GeneralizedLinearRegressionSummary sum = model.evaluate(datarow);
        Dataset<Row> test = sum.predictions();
        test.show();
        System.out.println("asdkjasdasjaskjdasdkjasdaskjasd:  " + summary.residuals().count());
        double[] x = new double[(int) summary.residuals().count()];
        double[] y = new double[(int) summary.residuals().count()];

        List<Row> resi = summary.residuals().collectAsList();
        for (int i = 0; i < resi.size(); i++) {
            y[i] = resi.get(i).getDouble(0);
            x[i] = i;
        }
//        resi.foreach(new ForeachFunction<Row>() {
//            int count = 0;
//            @Override
//            public void call(Row row) throws Exception {
////                System.out.println(row.getDouble(0));
//                x[count] = row.getDouble(0);
//                y[count] = (double) count;
//                count++;
//            }
//        });
//        System.out.println(x.length);
        System.out.println(y[4573]);
        // create your PlotPanel (you can use it as a JPanel)
        Plot2DPanel plot = new Plot2DPanel();

        // add a line plot to the PlotPanel
        plot.addLinePlot("my plot", x, y);

        // put the PlotPanel in a JFrame, as a JPanel
        JFrame frame = new JFrame("a plot panel");
        frame.setSize(600, 600);
        frame.setContentPane(plot);
        frame.setVisible(true);

        spark.stop();




    }
}
