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

Our project is dependent on Python 3 with the following packages:
- altair==4.2.0
- vl-convert-python==0.5.0
- docopt=0.6.2
- numpy=1.23.3
- ipykernel
- scikit-learn==1.1.2
- pandas=1.4.4
- requests>=2.24.0
- scikit-learn=1.1.2
- scipy=1.9.2
- ipython>=7.15
- altair_saver
- selenium<4.3.0
- matplotlib>=3.5.3
- pandas-profiling
- pandoc
- joblib==1.1.0
- psutil>=5.7.2
- openpyxl>=3.0.0
- xlrd>=2.0.1
- xlwt>=1.3.0
And R, specifically we use RStudio of version:
- RStudio 2022.07.1+554 

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


*Note about the usage of python3:*  
*Python 2 is no longer in use since 2020, and because Python 3 and Python 2 do not share the exactly same language syntax, many popular packages are developed in Python 3 instead of Python 2. Hence in this project, please use Python3 so the dependent packages can be properly run.*


For reproducing the results in step-by-step manner, run the scripts below using terminal in a sequential manner in the root directory of this project:

```
python3 src/data_downloader.py --url=https://archive.ics.uci.edu/ml/machine-learning-databases/00350/default%20of%20credit%20card%20clients.xls --path=data/raw/
```

Some preprocessing, data cleaning and splitting can be performed via the following command:

```
python3 src/data_preprocessor.py --input=data/raw/raw_data.xls --output=data/processed/
```

The EDA can be performed as:

```
python3 src/data_eda.py --traindata=data/processed/train_raw.csv --output=results/eda_results/
```

And finally, the model building, model scoring, and predictive analysis can be done by running the following code in the terminal:

```
python3 src/data_analysis.py --traindata=data/processed/train_cleaned.csv --testdata=data/processed/test_cleaned.csv --output=results/model/
```

To render the final report:

```
Rscript -e "rmarkdown::render('doc/report.Rmd')"
```

## Report

A report summarizing the results of our study can be found [here](https://github.com/UBC-MDS/Credit_Card_Default_Prediction_Group13/blob/main/doc/report.md).


## License
We use MIT license for this project.


## References

1. The original data set is located [here.](https://archive.ics.uci.edu/ml/datasets/default+of+credit+card+clients)
2. Yeh, I. C., & Lien, C. H. (2009). The comparisons of data mining techniques for the predictive accuracy of probability of default of credit card clients. Expert Systems with Applications, 36(2), 2473-2480.

