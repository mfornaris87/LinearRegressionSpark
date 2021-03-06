<h1>Linear regressions with Scala in Spark</h1>
<h3>@Author: Maite Sánchez Fornaris</h3>

<h2>Description</h2>
<p>The task is set about selecting a dataset in UCI machine learning repository and apply different kinds of regression models using Scala in Spark and compare the results according to the RMSE.</p>
<p>The selected dataset was Combined Cycle Power Plant. It contains 9568 data points collected from a Combined Cycle Power Plant over 6 years (2006-2011), when the power plant was set to work with full load. Features consist of hourly average ambient variables Temperature (T), Ambient Pressure (AP), Relative Humidity (RH) and Exhaust Vacuum (V) to predict the net hourly electrical energy output (EP) of the plant.
A combined cycle power plant (CCPP) is composed of gas turbines (GT), steam turbines (ST) and heat recovery steam generators. In a CCPP, the electricity is generated by gas and steam turbines, which are combined in one cycle, and is transferred from one turbine to another. While the Vacuum is colected from and has effect on the Steam Turbine, he other three of the ambient variables effect the GT performance.
For comparability with our baseline studies, and to allow 5x2 fold statistical tests be carried out, we provide the data shuffled five times. For each shuffling 2-fold CV is carried out and the resulting 10 measurements are used for statistical testing.
We provide the data both in .ods and in .xlsx formats.</p>
<h3>Attribute Information:</h3>
<p>
Features consist of hourly average ambient variables
<ul>Temperature (T) in the range 1.81°C and 37.11°C,</ul>
<ul>Ambient Pressure (AP) in the range 992.89-1033.30 milibar,</ul>
<ul>Relative Humidity (RH) in the range 25.56% to 100.16%</ul>
<ul>Exhaust Vacuum (V) in teh range 25.36-81.56 cm Hg</ul>
<ul>Net hourly electrical energy output (EP) 420.26-495.76 MW</ul>
The averages are taken from various sensors located around the plant that record the ambient variables every second. The variables are given without normalization. </p>

<h2>Requirements</h2>
<ul>Apache Spark</ul>
<ul>Scala</ul>
<ul>org.apache.spark.ml.evaluation.RegressionEvaluator</ul>
<ul>org.apache.spark.ml.regression</ul>
<ul>org.apache.spark.sql.SparkSession</ul>


<h2>Results</h2>
<ul>Linear Regression RMSE: 4.557126016749488</ul>
<ul>Decision Tree Regression RMSE: 4.543657451965862</ul>
<ul>Random Forest Regression RMSE: 4.372567983200875</ul>
<ul>Gradient-boost Regression RMSE: 3.989604463319834</ul>

<p>The best performing model for predicting maximum power generation in the Combined Cycle Power Plant was Gradient-Boost Regression with the minimum RMSE of all applied methods</p>
