package WBS;

import org.apache.commons.math3.util.Precision;
import org.apache.spark.SparkConf;
import org.apache.spark.api.java.JavaSparkContext;
import org.apache.spark.api.java.function.FilterFunction;
import org.apache.spark.api.java.function.ForeachFunction;
import org.apache.spark.api.java.function.MapFunction;
import org.apache.spark.ml.linalg.VectorUDT;
import org.apache.spark.ml.linalg.Vectors;
import org.apache.spark.ml.param.ParamMap;
import org.apache.spark.ml.regression.GeneralizedLinearRegression;
import org.apache.spark.ml.regression.GeneralizedLinearRegressionModel;
import org.apache.spark.sql.Dataset;
import org.apache.spark.sql.Row;
import org.apache.spark.sql.RowFactory;
import org.apache.spark.sql.SparkSession;
import org.apache.spark.sql.types.Metadata;
import org.apache.spark.sql.types.StructField;
import org.apache.spark.sql.types.StructType;
import org.apache.spark.util.LongAccumulator;

import java.util.ArrayList;
import java.util.List;
import java.util.Properties;

import static org.apache.spark.sql.types.DataTypes.DoubleType;
import static org.apache.spark.sql.types.DataTypes.IntegerType;
import static org.apache.spark.sql.types.DataTypes.createStructField;

/**
 * Created by cristu on 22/08/16.
 */
public class RegEachId {
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
//                        return !row.isNullAt(0) && x == 36.0;
                        return !row.isNullAt(0);
                    }
                }).sort(df.col("user_id").asc());

        Dataset<Row> userCount = dfrows.groupBy(dfrows.col("user_id")).count();

        StructType schema = new StructType(new StructField[]{
                createStructField("features", new VectorUDT(), false, Metadata.empty()),
                createStructField("user_id", IntegerType, false),
                createStructField("positive", DoubleType, false),
                createStructField("neutral", DoubleType, false),
                createStructField("negative", DoubleType, false)
        });

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
        Properties prop = new Properties();
        prop.setProperty("user", "root");
        prop.setProperty("password","1");
        prop.setProperty("driver", "com.mysql.jdbc.Driver");



        userCount.foreach(new ForeachFunction<Row>() {
                  @Override
                  public void call(Row rowEach) throws Exception {
                     Dataset<Row> dataFiltered = dfrows.filter(new FilterFunction<Row>() {
                                                  @Override
                                                  public boolean call(Row rowFilter) throws Exception {
                                                      return new Double(rowFilter.getString(1)) == new Double(rowEach.getString(0));
                                                  }
                                              });

                      List<Row> listdf = dataFiltered.collectAsList();
                      List<Row> mylist = new ArrayList<Row>();

                      //Num of comments for regression...
                      int numComment = listdf.size()/2;
                      for(int j=numComment; j< listdf.size(); j++){
                          mylist.add(RowFactory.create(Vectors.dense(new Double(j)), new Integer(listdf.get(j).getString(1)), new Double(listdf.get(j).getString(2)),new Double(listdf.get(j).getString(3)),new Double(listdf.get(j).getString(4))));
                      }

                      Dataset<Row> data = spark.createDataFrame(mylist, schema);
                      Dataset<Row>[] splits = data.randomSplit(new double[]{0.75, 0.25});
                      Dataset<Row> trainingData = splits[0];
                      Dataset<Row> testDatab = splits[1];

                      long numData = dfrows.count();

                      List<Row> testList = testDatab.collectAsList();
                      List<Row> myTestList = new ArrayList<Row>();
                      myTestList.addAll(testList);
                      for(int i=0; i<3; i++){
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


                    Dataset<Row> resultsPos = modelPos.transform(testData);
                    Dataset<Row> resultsNeu = modelNeu.transform(testData);
                    Dataset<Row> resultsNeg = modelNeg.transform(testData);

                    List<Row> listPos = resultsPos.collectAsList();
                    List<Row> listNeu = resultsNeu.collectAsList();
                    List<Row> listNeg = resultsNeg.collectAsList();


                    List<Row> resultsList = new ArrayList<Row>();
                    for(int i=0; i< listPos.size(); i++){
                        double sumPred = Precision.round(listNeg.get(i).getDouble(5)+listNeu.get(i).getDouble(5)+listPos.get(i).getDouble(5),3);
                        resultsList.add(RowFactory.create(i, listPos.get(i).get(1), listPos.get(i).get(2), listPos.get(i).get(3), listPos.get(i).get(4), listPos.get(i).get(5), listNeu.get(i).get(5), listNeg.get(i).get(5),sumPred));
                    }

                    Dataset<Row> results = spark.createDataFrame(resultsList, schemaResults);
                    Dataset<Row> resWrite = results.orderBy(results.col("comment_id_spark").desc()).filter(new FilterFunction<Row>() {
                        @Override
                        public boolean call(Row row) throws Exception {
                            return row.getDouble(2)==0.0;
                        }
                    });
                    resWrite.show();
                    resWrite.write().mode("append").jdbc("jdbc:mysql://localhost:3306/","reviews.predictions",prop);

                  }
          });



//        System.out.println(test.count());
//        test.show();
        LongAccumulator accum = sc.sc().longAccumulator();
//
//        List<Row> listdf = dfrows.collectAsList();
//        List<Row> mylist = new ArrayList<Row>();
//        int numComment = listdf.size()/2;
//        for(int i=numComment; i< listdf.size(); i++){
////            Row temp;
////            if(i<listdf.size()){
////                temp = RowFactory.create(Vectors.dense(new Double(i)), new Integer(listdf.get(i).getString(1)), new Double(listdf.get(i).getString(2)),new Double(listdf.get(i).getString(3)),new Double(listdf.get(i).getString(4)));
////            }else{
////                temp = RowFactory.create(Vectors.dense(new Double(i)), new Integer(listdf.get(i-100).getString(1)),1.0,1.0,1.0);
////            }
////            mylist.add(temp);
//            mylist.add(RowFactory.create(Vectors.dense(new Double(i)), new Integer(listdf.get(i).getString(1)), new Double(listdf.get(i).getString(2)),new Double(listdf.get(i).getString(3)),new Double(listdf.get(i).getString(4))));
//        }
//        StructType schema = new StructType(new StructField[]{
//                createStructField("features", new VectorUDT(), false, Metadata.empty()),
//                createStructField("user_id", IntegerType, false),
//                createStructField("positive", DoubleType, false),
//                createStructField("neutral", DoubleType, false),
//                createStructField("negative", DoubleType, false)
//        });
//        Dataset<Row> data = spark.createDataFrame(mylist, schema);
//
//        data.show();
//
//        Dataset<Row>[] splits = data.randomSplit(new double[]{0.75, 0.25});
//        Dataset<Row> trainingData = splits[0];
//        Dataset<Row> testDatab = splits[1];
//
//        long numData = dfrows.count();
//
//        List<Row> testList = testDatab.collectAsList();
//        List<Row> myTestList = new ArrayList<Row>();
//        myTestList.addAll(testList);
//        for(int i=0; i<10; i++){
//            myTestList.add(RowFactory.create(Vectors.dense(new Double(numData+i)), myTestList.get(0).getInt(1),0.0,0.0,0.0));
//        }
//        Dataset<Row> testData = spark.createDataFrame(myTestList, schema);
//
//
//        GeneralizedLinearRegression glr = new GeneralizedLinearRegression()
//                .setFamily("gaussian")
//                .setLink("identity")
//                .setMaxIter(100)
//                .setRegParam(0.1);
//
//        ParamMap paramMapPos = new ParamMap()
//                .put(glr.labelCol().w("positive"))
//                .put(glr.predictionCol().w("predictionPos"));
//        ParamMap paramMapNeu = new ParamMap()
//                .put(glr.labelCol().w("neutral"))
//                .put(glr.predictionCol().w("predictionNeu"));
//        ParamMap paramMapNeg = new ParamMap()
//                .put(glr.labelCol().w("negative"))
//                .put(glr.predictionCol().w("predictionNeg"));
//
////        // Fit the model
//        GeneralizedLinearRegressionModel modelPos = glr.fit(trainingData,paramMapPos);
//        GeneralizedLinearRegressionModel modelNeu = glr.fit(trainingData,paramMapNeu);
//        GeneralizedLinearRegressionModel modelNeg = glr.fit(trainingData,paramMapNeg);
//
////        // Print the coefficients and intercept for generalized linear regression model
////        System.out.println("Coefficients: " + model.coefficients());
////        System.out.println("Intercept: " + model.intercept());
//////
//////        // Summarize the model over the training set and print out some metrics
////        GeneralizedLinearRegressionTrainingSummary summary = model.summary();
////        System.out.println("Coefficient Standard Errors: "
////                + Arrays.toString(summary.coefficientStandardErrors()));
////        System.out.println("T Values: " + Arrays.toString(summary.tValues()));
////        System.out.println("P Values: " + Arrays.toString(summary.pValues()));
////        System.out.println("Dispersion: " + summary.dispersion());
////        System.out.println("Null Deviance: " + summary.nullDeviance());
////        System.out.println("Residual Degree Of Freedom Null: " + summary.residualDegreeOfFreedomNull());
////        System.out.println("Deviance: " + summary.deviance());
////        System.out.println("Residual Degree Of Freedom: " + summary.residualDegreeOfFreedom());
////        System.out.println("AIC: " + summary.aic());
////        System.out.println("Deviance Residuals: ");
////        summary.residuals().show();
//
//        Dataset<Row> resultsPos = modelPos.transform(testData);
//        Dataset<Row> resultsNeu = modelNeu.transform(testData);
//        Dataset<Row> resultsNeg = modelNeg.transform(testData);
//
//        List<Row> listPos = resultsPos.collectAsList();
//        List<Row> listNeu = resultsNeu.collectAsList();
//        List<Row> listNeg = resultsNeg.collectAsList();
//
//
//        List<Row> resultsList = new ArrayList<Row>();
//        for(int i=0; i< listPos.size(); i++){
//            double sumPred = Precision.round(listNeg.get(i).getDouble(5)+listNeu.get(i).getDouble(5)+listPos.get(i).getDouble(5),3);
//            resultsList.add(RowFactory.create(i, listPos.get(i).get(1), listPos.get(i).get(2), listPos.get(i).get(3), listPos.get(i).get(4), listPos.get(i).get(5), listNeu.get(i).get(5), listNeg.get(i).get(5),sumPred));
//        }
//        StructType schemaResults = new StructType(new StructField[]{
//                createStructField("comment_id_spark", IntegerType,false),
//                createStructField("user_id", IntegerType, false),
//                createStructField("positive", DoubleType, false),
//                createStructField("neutral", DoubleType, false),
//                createStructField("negative", DoubleType, false),
//                createStructField("predPositive", DoubleType, false),
//                createStructField("predNeutral", DoubleType, false),
//                createStructField("predNegative", DoubleType, false),
//                createStructField("SumPred", DoubleType, false)
//        });
//        Dataset<Row> results = spark.createDataFrame(resultsList, schemaResults);
//        Dataset<Row> resWrite = results.orderBy(results.col("comment_id_spark").desc()).filter(new FilterFunction<Row>() {
//            @Override
//            public boolean call(Row row) throws Exception {
//                return row.getDouble(2)==0.0;
//            }
//        });
//        resWrite.show();
//
//
//
//
//        Properties prop = new Properties();
//        prop.setProperty("user", "root");
//        prop.setProperty("password","1");
//        prop.setProperty("driver", "com.mysql.jdbc.Driver");
//
//        resWrite.write().mode("overwrite").jdbc("jdbc:mysql://localhost:3306/","reviews.predictions",prop);
//
//
//        for (Row r: results.collectAsList()) {
//            System.out.println("Comment-> " + r.get(0) + " , predicPositive = " + r.get(5) + " , predicNeutral = " + r.get(6) + " , predicNegative = " + r.get(7) + " , SumPredict = " + r.get(8) );
//        }


    }
}
