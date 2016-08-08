/**
 * Created by cristu on 5/08/16.
 */
package WBS;


import org.apache.spark.api.java.JavaSparkContext;
import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.SparkConf;
import org.apache.spark.ml.classification.LogisticRegression;
import org.apache.spark.ml.classification.LogisticRegressionModel;
import org.apache.spark.ml.linalg.Vector;
import org.apache.spark.sql.SparkSession;
import org.apache.spark.sql.Dataset;
import org.apache.spark.sql.Row;

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
        Dataset<Row> pos = df.select("positive");
        pos.show();

//        LogisticRegression lr = new LogisticRegression().setMaxIter(1000);

//        System.out.println("LogisticRegression parameters:\n" + lr.explainParams() + "\n");
//        LogisticRegressionModel model = lr.fit(df);
//        Vector weights = model.weightCol();
//
//        model.transform(df).show();


    }
}
