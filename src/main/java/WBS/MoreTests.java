package WBS;

import org.apache.commons.math3.util.Precision;
import org.apache.spark.SparkConf;
import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.api.java.JavaSparkContext;
import org.apache.spark.api.java.function.FilterFunction;
import org.apache.spark.api.java.function.Function;
import org.apache.spark.api.java.function.MapFunction;
import org.apache.spark.api.java.function.PairFunction;
import org.apache.spark.ml.feature.LabeledPoint;
import org.apache.spark.ml.linalg.VectorUDT;
import org.apache.spark.ml.linalg.Vectors;
import org.apache.spark.ml.param.ParamMap;
import org.apache.spark.ml.regression.GeneralizedLinearRegression;
import org.apache.spark.ml.regression.GeneralizedLinearRegressionModel;
import org.apache.spark.ml.regression.GeneralizedLinearRegressionTrainingSummary;
import org.apache.spark.sql.Dataset;
import org.apache.spark.sql.Row;
import org.apache.spark.sql.RowFactory;
import org.apache.spark.sql.SparkSession;
import org.apache.spark.sql.types.*;
import org.apache.spark.util.LongAccumulator;
import scala.Function1;
import scala.Tuple2;

import java.util.*;

    import static org.apache.spark.sql.types.DataTypes.*;

/**
 * Created by cristu on 17/08/16.
 */
public class MoreTests {

    private static final String MYSQL_USERNAME = "root";
    private static final String MYSQL_PWD = "1";
    private static final String MYSQL_CONNECTION_URL = "jdbc:mysql://localhost:3306/";
    private static final String BBDD_NAME = "reviews.predictions";




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

        Dataset<Row> dfrows = df.select("id", "user_id", "positive", "neutral", "negative")
                .filter(new FilterFunction<Row>() {
                    @Override
                    public boolean call(Row row) throws Exception {
                        Double x = 0.0;
                        if (!row.isNullAt(0)) {
                            x = new Double(row.getString(1));
                        }
                        return !row.isNullAt(0) && x == 36.0;
                    }
                });
        LongAccumulator accum = sc.sc().longAccumulator();

        List<Row> listdf = dfrows.collectAsList();
        List<Row> mylist = new ArrayList<Row>();
        int numComment = listdf.size()/2;
        for(int i=numComment; i< listdf.size(); i++){
//            Row temp;
//            if(i<listdf.size()){
//                temp = RowFactory.create(Vectors.dense(new Double(i)), new Integer(listdf.get(i).getString(1)), new Double(listdf.get(i).getString(2)),new Double(listdf.get(i).getString(3)),new Double(listdf.get(i).getString(4)));
//            }else{
//                temp = RowFactory.create(Vectors.dense(new Double(i)), new Integer(listdf.get(i-100).getString(1)),1.0,1.0,1.0);
//            }
//            mylist.add(temp);
            mylist.add(RowFactory.create(Vectors.dense(new Double(i)), new Integer(listdf.get(i).getString(1)), new Double(listdf.get(i).getString(2)),new Double(listdf.get(i).getString(3)),new Double(listdf.get(i).getString(4))));
        }
        StructType schema = new StructType(new StructField[]{
                createStructField("features", new VectorUDT(), false, Metadata.empty()),
                createStructField("user_id", IntegerType, false),
                createStructField("positive", DoubleType, false),
                createStructField("neutral", DoubleType, false),
                createStructField("negative", DoubleType, false)
        });
        Dataset<Row> data = spark.createDataFrame(mylist, schema);

        data.show();

        Dataset<Row>[] splits = data.randomSplit(new double[]{0.75, 0.25});
        Dataset<Row> trainingData = splits[0];
        Dataset<Row> testDatab = splits[1];

        long numData = dfrows.count();

        List<Row> testList = testDatab.collectAsList();
        List<Row> myTestList = new ArrayList<Row>();
        myTestList.addAll(testList);
        for(int i=0; i<10; i++){
            myTestList.add(RowFactory.create(Vectors.dense(new Double(numData+i)), myTestList.get(0).getInt(1),0.0,0.0,0.0));
        }
        Dataset<Row> testData = spark.createDataFrame(myTestList, schema);


        GeneralizedLinearRegression glr = new GeneralizedLinearRegression()
                .setFamily("gaussian")
                .setLink("identity")
                .setMaxIter(100)
                .setRegParam(0.1);

        ParamMap paramMapPos = new ParamMap()
                .put(glr.labelCol().w("positive"))
                .put(glr.predictionCol().w("predictionPos"));
        ParamMap paramMapNeu = new ParamMap()
                .put(glr.labelCol().w("neutral"))
                .put(glr.predictionCol().w("predictionNeu"));
        ParamMap paramMapNeg = new ParamMap()
                .put(glr.labelCol().w("negative"))
                .put(glr.predictionCol().w("predictionNeg"));

//        // Fit the model
        GeneralizedLinearRegressionModel modelPos = glr.fit(trainingData,paramMapPos);
        GeneralizedLinearRegressionModel modelNeu = glr.fit(trainingData,paramMapNeu);
        GeneralizedLinearRegressionModel modelNeg = glr.fit(trainingData,paramMapNeg);

//        // Print the coefficients and intercept for generalized linear regression model
//        System.out.println("Coefficients: " + model.coefficients());
//        System.out.println("Intercept: " + model.intercept());
////
////        // Summarize the model over the training set and print out some metrics
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

        Dataset<Row> resultsPos = modelPos.transform(testData);
        Dataset<Row> resultsNeu = modelNeu.transform(testData);
        Dataset<Row> resultsNeg = modelNeg.transform(testData);

        List<Row> listPos = resultsPos.collectAsList();
        List<Row> listNeu = resultsNeu.collectAsList();
        List<Row> listNeg = resultsNeg.collectAsList();

/*        List<Row> resultsList = new ArrayList<Row>();
        for(int i=0; i< listPos.size(); i++){
            double sumPred = Precision.round(listNeg.get(i).getDouble(5)+listNeu.get(i).getDouble(5)+listPos.get(i).getDouble(5),3);
            resultsList.add(RowFactory.create(listPos.get(i).get(0), listPos.get(i).get(1), listPos.get(i).get(2), listPos.get(i).get(3), listPos.get(i).get(4), listPos.get(i).get(5), listNeu.get(i).get(5), listNeg.get(i).get(5),sumPred));
        }
        StructType schemaResults = new StructType(new StructField[]{
                createStructField("features", new VectorUDT(), false, Metadata.empty()),
                createStructField("user_id", IntegerType, false),
                createStructField("positive", DoubleType, false),
                createStructField("neutral", DoubleType, false),
                createStructField("negative", DoubleType, false),
                createStructField("predPositive", DoubleType, false),
                createStructField("predNeutral", DoubleType, false),
                createStructField("predNegative", DoubleType, false),
                createStructField("SumPred", DoubleType, false)
        });
        Dataset<Row> results = spark.createDataFrame(resultsList, schemaResults);*/

        List<Row> resultsList = new ArrayList<Row>();
        for(int i=0; i< listPos.size(); i++){
            double sumPred = Precision.round(listNeg.get(i).getDouble(5)+listNeu.get(i).getDouble(5)+listPos.get(i).getDouble(5),3);
            resultsList.add(RowFactory.create(i, listPos.get(i).get(1), listPos.get(i).get(2), listPos.get(i).get(3), listPos.get(i).get(4), listPos.get(i).get(5), listNeu.get(i).get(5), listNeg.get(i).get(5),sumPred));
        }
        StructType schemaResults = new StructType(new StructField[]{
                createStructField("comment_id_spark", IntegerType,false),
                createStructField("user_id", IntegerType, false),
                createStructField("positive", DoubleType, false),
                createStructField("neutral", DoubleType, false),
                createStructField("negative", DoubleType, false),
                createStructField("predPositive", DoubleType, false),
                createStructField("predNeutral", DoubleType, false),
                createStructField("predNegative", DoubleType, false),
                createStructField("SumPred", DoubleType, false)
        });
        Dataset<Row> results = spark.createDataFrame(resultsList, schemaResults);
        Dataset<Row> resWrite = results.orderBy(results.col("comment_id_spark").desc()).filter(new FilterFunction<Row>() {
            @Override
            public boolean call(Row row) throws Exception {
                return row.getDouble(2)==0.0;
            }
        });
        resWrite.show();



//        results.write().json("/home/cristu/Proyectos/BigData/src/main/resources/social_predictions.json");


        Properties prop = new Properties();
        prop.setProperty("user", "root");
        prop.setProperty("password","1");
        prop.setProperty("driver", "com.mysql.jdbc.Driver");

//        results.select("predPositive","predNeutral", "predNegative").write().jdbc("jdbc:mysql://localhost:3306/","reviews.predictions",prop);
        resWrite.write().mode("overwrite").jdbc("jdbc:mysql://localhost:3306/","reviews.predictions",prop);

//        resultsPos.show();
//        resultsNeu.show();
//        resultsNeg.show();

        for (Row r: results.collectAsList()) {
            System.out.println("Comment-> " + r.get(0) + " , predicPositive = " + r.get(5) + " , predicNeutral = " + r.get(6) + " , predicNegative = " + r.get(7) + " , SumPredict = " + r.get(8) );
        }

        //Results regPar 0.1 -> Total coments
       /* Comment-> [31991.0] , predicPositive = 0.26754697386078247 , predicNeutral = 0.5981142334308499 , predicNegative = 0.1384910266861133 , SumPredict = 1.004
        Comment-> [31993.0] , predicPositive = 0.2675446555447972 , predicNeutral = 0.5981180397568597 , predicNegative = 0.13849005591142044 , SumPredict = 1.004
        Comment-> [31994.0] , predicPositive = 0.2675434963868046 , predicNeutral = 0.5981199429198647 , predicNegative = 0.13848957052407404 , SumPredict = 1.004
        Comment-> [31995.0] , predicPositive = 0.26754233722881193 , predicNeutral = 0.5981218460828696 , predicNegative = 0.1384890851367276 , SumPredict = 1.004
        Comment-> [31996.0] , predicPositive = 0.26754117807081934 , predicNeutral = 0.5981237492458745 , predicNegative = 0.13848859974938119 , SumPredict = 1.004
        Comment-> [31997.0] , predicPositive = 0.26754001891282675 , predicNeutral = 0.5981256524088794 , predicNegative = 0.1384881143620348 , SumPredict = 1.004
        Comment-> [31998.0] , predicPositive = 0.2675388597548341 , predicNeutral = 0.5981275555718844 , predicNegative = 0.13848762897468836 , SumPredict = 1.004
        Comment-> [31999.0] , predicPositive = 0.26753770059684145 , predicNeutral = 0.5981294587348892 , predicNegative = 0.13848714358734193 , SumPredict = 1.004
        Comment-> [32000.0] , predicPositive = 0.26753654143884886 , predicNeutral = 0.5981313618978942 , predicNegative = 0.1384866581999955 , SumPredict = 1.004
        Comment-> [32001.0] , predicPositive = 0.2675353822808562 , predicNeutral = 0.598133265060899 , predicNegative = 0.13848617281264908 , SumPredict = 1.004
        Comment-> [32002.0] , predicPositive = 0.2675342231228636 , predicNeutral = 0.598135168223904 , predicNegative = 0.13848568742530268 , SumPredict = 1.004*/
       //Results regPar 0.1 -> Total/2 coments
     /*   Comment-> [31992.0] , predicPositive = 0.25385320400066663 , predicNeutral = 0.6078502163707635 , predicNegative = 0.14046416356159439 , SumPredict = 1.002
        Comment-> [31993.0] , predicPositive = 0.2538509060303858 , predicNeutral = 0.6078527023780714 , predicNegative = 0.14046424455125994 , SumPredict = 1.002
        Comment-> [31994.0] , predicPositive = 0.25384860806010495 , predicNeutral = 0.6078551883853793 , predicNegative = 0.1404643255409255 , SumPredict = 1.002
        Comment-> [31995.0] , predicPositive = 0.2538463100898241 , predicNeutral = 0.6078576743926872 , predicNegative = 0.14046440653059103 , SumPredict = 1.002
        Comment-> [31996.0] , predicPositive = 0.2538440121195433 , predicNeutral = 0.6078601603999951 , predicNegative = 0.1404644875202566 , SumPredict = 1.002
        Comment-> [31997.0] , predicPositive = 0.2538417141492625 , predicNeutral = 0.607862646407303 , predicNegative = 0.14046456850992212 , SumPredict = 1.002
        Comment-> [31998.0] , predicPositive = 0.25383941617898165 , predicNeutral = 0.6078651324146109 , predicNegative = 0.14046464949958767 , SumPredict = 1.002
        Comment-> [31999.0] , predicPositive = 0.2538371182087008 , predicNeutral = 0.6078676184219189 , predicNegative = 0.1404647304892532 , SumPredict = 1.002
        Comment-> [32000.0] , predicPositive = 0.25383482023841997 , predicNeutral = 0.6078701044292268 , predicNegative = 0.14046481147891876 , SumPredict = 1.002
        Comment-> [32001.0] , predicPositive = 0.2538325222681391 , predicNeutral = 0.6078725904365347 , predicNegative = 0.14046489246858432 , SumPredict = 1.002
        Comment-> [32002.0] , predicPositive = 0.2538302242978583 , predicNeutral = 0.6078750764438426 , predicNegative = 0.14046497345824985 , SumPredict = 1.002*/

     //Results regPar 0.1 /-> last 100 coments
     /*   Comment-> [31992.0] , predicPositive = 0.26065041325860605 , predicNeutral = 0.6053759180880203 , predicNegative = 0.1360231442955763 , SumPredict = 1.002
        Comment-> [31993.0] , predicPositive = 0.2601899953050033 , predicNeutral = 0.6058668319007623 , predicNegative = 0.13603506170764276 , SumPredict = 1.002
        Comment-> [31994.0] , predicPositive = 0.25972957735140234 , predicNeutral = 0.6063577457135043 , predicNegative = 0.13604697911970923 , SumPredict = 1.002
        Comment-> [31995.0] , predicPositive = 0.2592691593977996 , predicNeutral = 0.6068486595262463 , predicNegative = 0.1360588965317757 , SumPredict = 1.002
        Comment-> [31996.0] , predicPositive = 0.25880874144419863 , predicNeutral = 0.6073395733389884 , predicNegative = 0.13607081394384216 , SumPredict = 1.002
        Comment-> [31997.0] , predicPositive = 0.2583483234905959 , predicNeutral = 0.6078304871517304 , predicNegative = 0.13608273135590862 , SumPredict = 1.002
        Comment-> [31998.0] , predicPositive = 0.25788790553699315 , predicNeutral = 0.6083214009644724 , predicNegative = 0.13609464876797508 , SumPredict = 1.002
        Comment-> [31999.0] , predicPositive = 0.2574274875833922 , predicNeutral = 0.6088123147772144 , predicNegative = 0.13610656618004155 , SumPredict = 1.002
        Comment-> [32000.0] , predicPositive = 0.25696706962978944 , predicNeutral = 0.6093032285899564 , predicNegative = 0.136118483592108 , SumPredict = 1.002
        Comment-> [32001.0] , predicPositive = 0.25650665167618847 , predicNeutral = 0.6097941424026985 , predicNegative = 0.13613040100417448 , SumPredict = 1.002
        Comment-> [32002.0] , predicPositive = 0.2560462337225857 , predicNeutral = 0.6102850562154405 , predicNegative = 0.13614231841624094 , SumPredict = 1.002
*/

     //Results Results regPar 0.1 /-> last 10 coments

       /* Comment-> [31992.0] , predicPositive = 0.31929181816263963 , predicNeutral = 0.477603042615101 , predicNegative = 0.19537155148326235 , SumPredict = 0.992
        Comment-> [31993.0] , predicPositive = 0.3284501817951764 , predicNeutral = 0.464890317804759 , predicNegative = 0.1973458617799153 , SumPredict = 0.991
        Comment-> [31994.0] , predicPositive = 0.3376085454277131 , predicNeutral = 0.45217759299441695 , predicNegative = 0.19932017207656827 , SumPredict = 0.989
        Comment-> [31995.0] , predicPositive = 0.34676690906019303 , predicNeutral = 0.4394648681841318 , predicNegative = 0.20129448237322123 , SumPredict = 0.988
        Comment-> [31996.0] , predicPositive = 0.3559252726927298 , predicNeutral = 0.42675214337378975 , predicNegative = 0.2032687926698742 , SumPredict = 0.986
        Comment-> [31997.0] , predicPositive = 0.3650836363252665 , predicNeutral = 0.41403941856350457 , predicNegative = 0.20524310296652715 , SumPredict = 0.984
        Comment-> [31998.0] , predicPositive = 0.3742419999578033 , predicNeutral = 0.40132669375316254 , predicNegative = 0.2072174132631801 , SumPredict = 0.983
        Comment-> [31999.0] , predicPositive = 0.38340036359034 , predicNeutral = 0.3886139689428205 , predicNegative = 0.20919172355982596 , SumPredict = 0.981
        Comment-> [32000.0] , predicPositive = 0.39255872722287677 , predicNeutral = 0.37590124413253534 , predicNegative = 0.21116603385647892 , SumPredict = 0.98
        Comment-> [32001.0] , predicPositive = 0.4017170908554135 , predicNeutral = 0.3631885193221933 , predicNegative = 0.21314034415313188 , SumPredict = 0.978
        Comment-> [32002.0] , predicPositive = 0.4108754544878934 , predicNeutral = 0.35047579451190813 , predicNegative = 0.21511465444978484 , SumPredict = 0.976*/
    }
}
