//////////////////////////////////////////////////
// RANDOM FOREST REGRESSION PROJECT SOLUTION ////
////////////////////////////////////////////////

//  En este proyecto se analiza un dataset que contiene los datos recolectados en una Central de Ciclo Combinado durante 6 annos
//  para predecir la produccion neta de energia electrica por hora (EP) segun los rasgos:
//    'T': temperatura dada en un rango entre 1.81°C and 37.11°C.
//    'V': aspiracion de escape dada entre el rango 25.36-81.56 cm Hg
//    'AP': presion ambiental expresada en el rango 992.89-1033.30 milibares
//    'RH': humedad relativa dada en el rango 25.56% to 100.16%


// Descripcion del metodo de regresion:
// Este metodo combina varios arboles de decision 
// para impedir que el modelo se ajuste demasiado al ejemplo especifico 
// que se esta analizando (overfitting)

//////////////////////////////
/// CAPTURAR LOS DATOS //////
////////////////////////////

// Importar bibliotecas necesarias para esta regresion
import org.apache.spark.ml.evaluation.RegressionEvaluator
import org.apache.spark.ml.regression.{RandomForestRegressionModel, RandomForestRegressor}
import org.apache.spark.sql.SparkSession

// Codigo para registrar los errores
import org.apache.log4j._
Logger.getLogger("org").setLevel(Level.ERROR)

// Crear la sesion Spark
val spark = SparkSession.builder().getOrCreate()

// Utilizar Spark para leer el archivo CSV Folds5x2_pp
val data = spark.read.option("header","true").option("inferSchema","true").format("csv").load("Folds5x2_pp.csv")

// Mostrar el esquema del DataFrame
data.printSchema()

//////////////////////////////////////////////////
/// Comprobar la visualizacion de los datos /////
////////////////////////////////////////////////

// Mostrar la primera fila del dataset
val colnames = data.columns
val firstrow = data.head(1)(0)
println("\n")
println("Example Data Row")
for(ind <- Range(1,colnames.length)){
  println(colnames(ind))
  println(firstrow(ind))
  println("\n")
}

//////////////////////////////////////////////////////
//// Preparar el DataFrame para Machine Learning ////
////////////////////////////////////////////////////

//    - Se renombra la columna EP a "label"
//    - Se agrupan las columnas "T","V","AP","RH" en la columan "features" mediante un VectorAssembler

val rfrdata = (data.select(data("EP").as("label"), $"T", $"V", $"AP", $"RH"))


// Importar VectorAssembler and Vectors
import org.apache.spark.ml.feature.VectorAssembler
import org.apache.spark.ml.linalg.Vectors


// El objeto VectorAssembler permitira sacar todas las columnas de los rasgos en una sola denominada "features"
val assembler = (new VectorAssembler().setInputCols(Array("T", "V", "AP", "RH")).setOutputCol("features") )

// Dividir el dataset en dos conjuntos: 70% para entrenamiento y el resto para prueba
val Array(training, test) = rfrdata.randomSplit(Array(0.7, 0.3))


//////////////////////////////////
// Establecer la conexion ///////
////////////////////////////////

// Importar Pipeline
import org.apache.spark.ml.Pipeline

// Entrenar el GBTRegressionModel
val rf = new RandomForestRegressor()

// Crear una nueva conexion con los escenarios: assembler, rf
val pipeline = new Pipeline().setStages(Array(assembler, rf))

// Llenar el conjunto de entrenamiento con los datos procesados
val model = pipeline.fit(training)

// Cargar los resultados en el conjunto de prueba utilizando transformacion
val results = model.transform(test)

////////////////////////////////////
//// EVALUACION DEL MODELO ////////
//////////////////////////////////

// Seleccionar la prediccion y la etiqueta para calcular el error de computo
results.select("prediction", "label", "features").show(5)
val evaluator = new RegressionEvaluator().setLabelCol("label").setPredictionCol("prediction").setMetricName("rmse")

val rmse = evaluator.evaluate(results)

println("Root Mean Squared Error (RMSE) on test data = " + rmse)

val rfModel = model.stages(1).asInstanceOf[RandomForestRegressionModel]
println("Learned regression model forest:\n" + rfModel)


