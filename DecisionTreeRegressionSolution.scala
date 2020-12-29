//////////////////////////////////////////////////
// SOLUCION USANDO DECISION TREE REGRESSION /////
////////////////////////////////////////////////

//  En este proyecto se analiza un dataset que contiene los datos recolectados en una Central de Ciclo Combinado durante 6 annos
//  para predecir la produccion neta de energia electrica por hora (EP) segun los rasgos:
//    'T': temperatura dada en un rango entre 1.81°C and 37.11°C.
//    'V': aspiracion de escape dada entre el rango 25.36-81.56 cm Hg
//    'AP': presion ambiental expresada en el rango 992.89-1033.30 milibares
//    'RH': humedad relativa dada en el rango 25.56% to 100.16%

// Descripcion del algoritmo
// Metodo basado en arboles de decisiones que particiona los datos por columnas 
// permitiendo el entrenamiento distribuido de un gran numero de instancias.



//////////////////////////////
/// CAPTURAR LOS DATOS //////
////////////////////////////

// Importar bibliotecas necesarias para esta regresion
import org.apache.spark.ml.evaluation.RegressionEvaluator
import org.apache.spark.ml.regression.DecisionTreeRegressionModel
import org.apache.spark.ml.regression.DecisionTreeRegressor
import org.apache.spark.sql.SparkSession

// Codigo para registrar los errores
import org.apache.log4j._
Logger.getLogger("org").setLevel(Level.ERROR)

// Utilizar Spark para leer el archivo CSV Folds5x2_pp
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

val dstdata = (data.select(data("EP").as("label"), $"T", $"V", $"AP", $"RH"))


// Importar VectorAssembler and Vectors
import org.apache.spark.ml.feature.VectorAssembler
import org.apache.spark.ml.linalg.Vectors

// El objeto VectorAssembler permitira sacar todas las columnas de los rasgos en una sola denominada "features"
val assembler = (new VectorAssembler()
  .setInputCols(Array("T", "V", "AP", "RH"))
  .setOutputCol("features") )


// Se usa randomSplit para crear un conjunto de entrenamiento del 70% del dataset y el restante 30% sera usado para probar el modelo
val Array(training, test) = dstdata.randomSplit(Array(0.7, 0.3))

//////////////////////////////////
// Establecer la conexion ///////
////////////////////////////////

// Importar Pipeline
import org.apache.spark.ml.Pipeline

// Entrenar el modelo de Arbol de Decision
val dt = new DecisionTreeRegressor()

// Crear una nueva conexion con los escenarios: assembler, lr
val pipeline = new Pipeline().setStages(Array(assembler, dt))

// Llenar el conjunto de entrenamiento con los datos procesados
val model = pipeline.fit(training)

// Cargar los resultados en el conjunto de prueba utilizando transformacion
val results = model.transform(test)

////////////////////////////////////
//// EVALUACION DEL MODELO ////////
//////////////////////////////////

//Seleccionar las filas que se van a mostrar
results.select("prediction", "label", "features").show(5)

// Seleccionar (prediccion, etiqueta verdadera) y calcular el error de prueba
val evaluator = new RegressionEvaluator().setLabelCol("label").setPredictionCol("prediction").setMetricName("rmse")

val rmse = evaluator.evaluate(results)

println("Root Mean Squared Error (RMSE) on CCP dataset = " + rmse)

val treeModel = model.stages(1).asInstanceOf[DecisionTreeRegressionModel]
println("Learned regression tree model:\n" + treeModel)

