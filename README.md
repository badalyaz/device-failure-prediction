# Possibility of device failure

## About Project
In this project presented how to use predictive maintenance techniques to predict device failure and which maintenance use for each case. In this project used different ML algorithms and NN to find the best model for our data.

<p align="center">
  <img src="project_images/failure.jpg" style='width:40%; background-color: white'>
</p>


-----------------------
-----------------------
## Data Description
Data contains information about 7 devices and for each device 9 sensors resgister indication.

<p align="center">
  <img src="project_images/image1.png" style='width:40%; background-color: white'>
</p>


Dataset has 124K rows and 12 columns(Date, Device_ID, 9 sensors and Fail). 
Sensors registered indication during year. 

Some of sensors contain very spars values, some of them have high correlation, but all values are integers. 

The dataset is unbalanced.

---------------------
---------------------

## Related Works
After data examination one of the two correlated features was droped. Feature selection was done using Random Forest Classifier and the unimportant features were also dropped. Because the imbalanced dataset the failures in dataset were upsampled using Random Oversampling teqnique. Different classic ML models were trained for the task.

---------------------
---------------------

## Results
Results of classification using Random Forest Classifier:
<p align="center">
  <img src="project_images/results.png" style='width:50%; background-color: white'>
</p>
Results of classification using multiple ML algorithms:
<p align="center">
  <img src="project_images/results1.png" style='width:50%; background-color: white'>
</p>
GEO results using the same models:
<p align="center">
  <img src="project_images/results2.png" style='width:50%; background-color: white'>
</p>