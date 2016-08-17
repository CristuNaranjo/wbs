package WBS;

import org.apache.spark.SparkConf;
import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.api.java.JavaSparkContext;
import org.apache.spark.api.java.function.FilterFunction;
import org.apache.spark.api.java.function.Function;
import org.apache.spark.api.java.function.MapFunction;
import org.apache.spark.api.java.function.PairFunction;
import org.apache.spark.ml.feature.LabeledPoint;
import org.apache.spark.ml.linalg.Vectors;
import org.apache.spark.ml.param.ParamMap;
import org.apache.spark.ml.regression.GeneralizedLinearRegression;
import org.apache.spark.ml.regression.GeneralizedLinearRegressionModel;
import org.apache.spark.sql.Dataset;
import org.apache.spark.sql.Row;
import org.apache.spark.sql.RowFactory;
import org.apache.spark.sql.SparkSession;
import org.apache.spark.util.LongAccumulator;
import scala.Tuple2;

/**
 * Created by cristu on 17/08/16.
 */
public class MoreTests {

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

        Dataset<Row> data = dfrows.map(new MapFunction<Row, Row>() {
            @Override
            public Row call(Row row) throws Exception {
                long count = 1;
                accum.add(count);
                return RowFactory.create(accum.value(), new Double(row.getString(1)),);
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


        // We may alternatively specify parameters using a ParamMap.
        ParamMap paramMapPos = new ParamMap()
                .put(glr.labelCol().w("positive"))
                .put(glr.predictionCol().w("predictionPos"))
                .put(glr.featuresCol().w("id"));
        ParamMap paramMapNeu = new ParamMap()
                .put(glr.labelCol().w("neutral"))
                .put(glr.predictionCol().w("predictionNeu"))
                .put(glr.featuresCol().w("id"));
        ParamMap paramMapNeg = new ParamMap()
                .put(glr.labelCol().w("negative"))
                .put(glr.predictionCol().w("predictionNeg"))
                .put(glr.featuresCol().w("id"));

        // One can also combine ParamMaps.

        ParamMap paramMapCombined = paramMapPos.$plus$plus(paramMapNeu).$plus$plus(paramMapNeg);

        // Now learn a new model using the paramMapCombined parameters.
        // paramMapCombined overrides all parameters set earlier via lr.set* methods.
//        LogisticRegressionModel model2 = lr.fit(training, paramMapCombined);
//        System.out.println("Model 2 was fit using parameters: " + model2.parent().extractParamMap());

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




        resultsPos.show();
        resultsNeu.show();
        resultsNeg.show();

    }
}
