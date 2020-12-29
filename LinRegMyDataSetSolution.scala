//////////////////////////////////////////////////////////////////////////////////////////////////////
////Regresion lineal para el dataset de una Planta Electrica de Ciclo Combinado                 /////
////Rasgos: Temperatura(T), Presion ambiental(AP), Humedad Relativa(RH), Aspiracion de Escape(V)////
////para predecir la producción neta de energía eléctrica por hora (EP) de la planta           ////
//////////////////////////////////////////////////////////////////////////////////////////////////

// Importar la Regresion Lineal
import org.apache.spark.ml.regression.LinearRegression

// opcional: codigo para tratamiento de errores
import org.apache.log4j._
Logger.getLogger("org").setLevel(Level.ERROR)


// Iniciar una sesion de Spark
import org.apache.spark.sql.SparkSession
val spark = SparkSession.builder().getOrCreate()

// Utilizar Spark para leer del archivo Folds5x2_pp.csv donde estan los datos medidos en una planta durante 6 annos de funcionamiento a plena carga
val data = spark.read.option("header","true").option("inferSchema","true").format("csv").load("Folds5x2_pp.csv")

// Imprimir el esquema del DataFrame
data.printSchema()

// Se imprimira como ejemplo la primera fila del dataset
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
//// Preparando el DataFrame obtenido para Machine Learning        ////
//// Esto es preciso porque solo se aceptan 2 columnas o variables ///
//// una dependiente (features) y una independiente (label)        //
////////////////////////////////////////////////////////////////////

// Importar VectorAssembler y Vectors
import org.apache.spark.ml.feature.VectorAssembler
import org.apache.spark.ml.linalg.Vectors

// Se renombra la columna EP como "label"
// Todo esto se combina en un dataframe nuevo denominado df
val df = data.select(data("EP").as("label"),$"T",$"V",$"AP",$"RH")

// Un ensamblador convierte los valores de entrada en un unico vector
// porque los algoritmos de machine learning solamente leen un vector para entrenar un modelo

// Se utiliza VectorAssembler para convertir las columnas de entrada de df
// a una unica columna de salida con la forma de un array denominado "features"
// Se establecen las columnas de entrada desde las que se deben leer los valores
// El nuevo objeto se denominara assembler
val assembler = new VectorAssembler().setInputCols(Array("T","V","AP","RH")).setOutputCol("features")

// Se usa un ensamblador para convertir el DataFrame a las dos columnas necesarias para aplicar ML
val output = assembler.transform(df).select($"label",$"features")


// Se crea un objeto de tipo Linear Regression
val lr = new LinearRegression()

// Se ajusta el modelo a los datos y se le llamara a este modelo lrModel
val lrModel = lr.fit(output)

// Se muestran los coeficientes y la intercepccion para la regresion lineal obtenida
println(s"Coefficients: ${lrModel.coefficients} Intercept: ${lrModel.intercept}")

// Se resume el modelo sobre el conjunto de entrenamiento y se muestran las metricas seleccionadas
// Se usa el metodo .summary sobre el modelo para crear un nuevo objeto denominado trainingSummary
val trainingSummary = lrModel.summary

// Del objeto obtenido se mostraran los residuales, el Root-Mean Square Error(RMSE), el Mean Square Error (MSE) y los valores R^2 para determinar la exactitud del modelo
trainingSummary.residuals.show()
println(s"RMSE: ${trainingSummary.rootMeanSquaredError}")
println(s"MSE: ${trainingSummary.meanSquaredError}")
println(s"r2: ${trainingSummary.r2}")

// Great Job!
