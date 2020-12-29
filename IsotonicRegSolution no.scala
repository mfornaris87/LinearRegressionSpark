//////////////////////////////////////////////
// ISOTONIC REGRESSION PROJECT SOLUTIONS ////
////////////////////////////////////////////

//  In this project we will be working with a fake advertising data set, indicating whether or not a particular internet user clicked on an Advertisement. We will try to create a model that will predict whether or not they will click on an ad based off the features of that user.
//  This data set contains the following features:
//    'T': consumer time on site in minutes
//    'V': cutomer age in years
//    'AP': Avg. Income of geographical area of consumer
//    'RH': Avg. minutes a day consumer is on the internet
//    'EP': 0 or 1 indicated clicking on Ad

////////////////////////
/// GET THE DATA //////
//////////////////////

// Import SparkSession and Isotonic Regression
import org.apache.spark.sql.Dataset[]
import org.apache.spark.ml.evaluation.RegressionEvaluator
import org.apache.spark.ml.regression.IsotonicRegression
import org.apache.spark.sql.SparkSession

// Optional: Use the following code below to set the Error reporting
import org.apache.log4j._
Logger.getLogger("org").setLevel(Level.ERROR)

// Create a Spark Session
val spark = SparkSession.builder().getOrCreate()

// Use Spark to read in the Advertising csv file.
val data = spark.read.option("header","true").option("inferSchema","true").format("csv").load("Folds5x2_pp.csv")

// Print the Schema of the DataFrame
data.printSchema()

///////////////////////
/// Display Data /////
/////////////////////

// Print out a sample row of the data (multiple ways to do this)
val colnames = data.columns
val firstrow = data.head(1)(0)
println("\n")
println("Example Data Row")
for(ind <- Range(1,colnames.length)){
  println(colnames(ind))
  println(firstrow(ind))
  println("\n")
}

////////////////////////////////////////////////////
//// Setting Up DataFrame for Machine Learning ////
//////////////////////////////////////////////////

//   Do the Following:
//    - Rename the EP column to "label"
//    - Grab the following columns "T","V","AP","RH"

val irtdata = (data.select(data("EP").as("label"), $"T", $"V", $"AP", $"RH"))


// Import VectorAssembler
import org.apache.spark.ml.feature.VectorAssembler

// Create a new VectorAssembler object called assembler for the feature
// columns as the input Set the output column to be called features
val assembler = (new VectorAssembler()
  .setInputCols(Array("T", "V", "AP", "RH"))
  .setOutputCol("features") )


// Use randomSplit to create a train test split of 70/30
val Array(training, test) = irtdata.randomSplit(Array(0.7, 0.3))
///////////////////////////////
// Set Up the Pipeline ///////
/////////////////////////////

// Import Pipeline
import org.apache.spark.ml.Pipeline

// Entrenar el modelo de Arbol de Decision
val ir = new IsotonicRegression()

// Create a new pipeline with the stages: assembler, lr
val pipeline = new Pipeline().setStages(Array(assembler, ir))

// Fit the pipeline to training set.
val model = pipeline.fit(training)

// Get Results on Test Set with transform
val results = model.transform(test)
//val results = model.transform(test).show()

////////////////////////////////////
//// MODEL EVALUATION /////////////
//////////////////////////////////

// Seleccionar (prediccion, etiqueta verdadera) y calcular el error de prueba
val evaluator = new RegressionEvaluator().setLabelCol("label").setPredictionCol("prediction").setMetricName("rmse")

val rmse = evaluator.evaluate(results)
println("Root Mean Squared Error (RMSE) on CCP dataset = " + rmse)
println(s"Boundaries in increasing order: ${model.boundaries}\n")
println(s"Predictions associated with the boundaries: ${model.predictions}\n")
