# Credibility Classification of Credit Card Clients

#### Contributors

- Chester Wang
- HanChen Wang
- Qurat-ul-Ain Azim
- Renee Kwon

## About
In the field of risk management, one of the most common problems is default prediction. This allows companies to predict the credibility of each person, analyze the risk level and optimize decisions for better business economics. In this project, the main question we are asking is:

**Given a credit card holder's basic personal information (gender, education, age, history of past payment etc.), will the person default on next month's payment?**

The report of the analysis can be found [here](https://github.com/UBC-MDS/Credit_Card_Default_Prediction_Group13/blob/Makefile_report_hw/doc/report.md).

## Research Question
Through this project, we aim to answer the question: Which attributes are most important when we use machine learning models to predict the default? Specifically we would like to know if the `weight` of attributes would change when we employ different models. Answering this question is, from our perspective, of great importance because it allows to understand what attributes relate to credibility the most. We would also aim to a comparative study of the mainstream machine learning classification models to be able to identify how the best performing model assigns weights to the various model features.

### Data Summary
We use a dataset hosted by the UCI machine learning repository (1). Originally it is collected by researchers from Chung Hua University and Tamkang University (2). As the probability of default cannot be actually acquired, the targets are obtained through estimation as stated by the authors of this dataset. The dataset consists of 30000 instances, with each consists of 23 attributes and a target. The raw dataset is about 5.5 MB large, and we split it into the training set (80%) and testing set (20%) for further use. The data attributes range from client's gender, age, education, previous payment history, credit amount etc.

### Data Analysis Plan
Building machine learning models and score the performance of each model would be our major method to approach the dataset. Specifically to study the influence of each attribute, we will look into the models based on their performance and order the importance of attributes with weights.

We plan to explore a wide range of algorithms (both naive and advanced), including:
- Support Vector Machine
- Logistic Regression
- Decision Tree
- Ensemble Methods (e.g. Random Forest)
- KNN (both unweighted and weighted)

Most of our code for data processing and model building and will be in Python 3. For EDA and data visualization we plan to use both R and Python 3.


### Exploratory Data Analysis

Our preliminary EDA includes looking for naay. missing values and for unclean data. We then study and plot the distributions of all the numeric type features for our analysis. Further analysis involves looking for any correlated features. We further use the pandas profiling package to report in depth results for the analysis.


## Dependencies

The data download and analysis is dependent on the following
1. docopt 0.6.2
2. xlrd 2.0.1
3. Python 3.10.6 with the following packages
    - scikit-learn>=1.1.3
    - altair_data_server
    - pandas<1.5
    - pandas profiler 3.4.0
4. RStudio 2022.07.1+554 

The complete list of packages used can be found in the [environment file](https://github.com/UBC-MDS/Credit_Card_Default_Prediction_Group13/blob/makefile_chester/environment.yaml).

The steps to using the environment are given below:

Creating an environment
```
conda env create --file environment.yaml
```

Activate the environment 
```
conda activate credit_default_env
```

Deactivate the environment 
```
conda deactivate
```

## Usage

### With Makefile

You can reproduce the results in this GitHub repository by cloning and installing all necessary [dependencies](https://github.com/UBC-MDS/Credit_Card_Default_Prediction_Group13/tree/Makefile_report_hw#dependencies). 

Then, you can run the command below using terminal in the root directory of this project to automatically run the full analysis and generate the final report. 

```
make all 
```

To delete all intermediate and result files, run the following command using terminal in the root directory of this project:

```
make clean
```

### Running each step

You can reproduce the results in this GitHub repository by cloning and installing all necessary [dependencies](https://github.com/UBC-MDS/Credit_Card_Default_Prediction_Group13/tree/Makefile_report_hw#dependencies).

For reproducing the results in step-by-step manner, run the scripts below using terminal in a sequential manner in the root directory of this project:

```
python3 src/data_downloader.py https://archive.ics.uci.edu/ml/machine-learning-databases/00350/default%20of%20credit%20card%20clients.xls data/raw/
```

Some preprocessing, data cleaning and splitting can be performed via the following command:

```
python3 src/data_preprocessor.py data/raw/raw_data.xls data/processed/
```

The EDA can be performed as:

```
python3 src/data_eda.py data/processed/train_raw.csv data/eda_results/
```

And finally, the model building and predictive analysis can be done by running the following code in the terminal:

```
python3 src/data_analysis.py data/processed/train_cleaned.csv data/processed/test_cleaned.csv data/results
```

## Report

A report summarizing the results of our study can be found [here](https://github.com/UBC-MDS/Credit_Card_Default_Prediction_Group13/blob/Makefile_report_hw/doc/report.md).


## License
We use MIT license for this project.


## References

1. The original data set is located [here.](https://archive.ics.uci.edu/ml/datasets/default+of+credit+card+clients)
2. Yeh, I. C., & Lien, C. H. (2009). The comparisons of data mining techniques for the predictive accuracy of probability of default of credit card clients. Expert Systems with Applications, 36(2), 2473-2480.

