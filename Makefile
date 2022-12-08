# Makefile
# Authors: Chester Wang, HanChen Wang, Qurat-ul-Ain Azim, Renee Kwon
# Date: 2022-11-30

# This driver script completes the supervised machine learning data analysis on
# a credit card default prediction data set and creates a final report on whether
# a credit card holder will likely default their bill next month. This script
# takes no arguments.

# example usage:
# make all
# make clean

# Run all to download & preprocess data, perform EDA, generate figures and build model. 
# It will also generate the report on the credit default prediction analysis. 
all: doc/report.md doc/report.html


# This line of code downloads the data from the UCI repository.
# (https://archive.ics.uci.edu/ml/datasets/default+of+credit+card+clients).
# raw_data.xls is saved to the raw folder under the data folder.
data/raw/raw_data.xls : src/data_downloader.py
	python3 src/data_downloader.py --url=https://archive.ics.uci.edu/ml/machine-learning-databases/00350/default%20of%20credit%20card%20clients.xls --path=data/raw/

# Preprocessing, data cleaning and splitting the raw data into test and train datasets:
data/processed/test_cleaned.csv data/processed/test_raw.csv data/processed/train_cleaned.csv data/processed/train_raw.csv : data/raw/raw_data.xls src/data_preprocessor.py
	python3 src/data_preprocessor.py --input=data/raw/raw_data.xls --output=data/processed/

# Perform exploratory data analysis on the cleaned train dataset and visualize the distribution
# of features. This generates figures and tables necessary for the final report. 
results/eda_results/binary_result.png results/eda_results/categorical_result.png results/eda_results/corr.csv results/eda_results/numeric_result.png : data/processed/train_raw.csv src/data_eda.py
	python3 src/data_eda.py --traindata=data/processed/train_raw.csv --output=results/eda_results/

# Build and compare different machine learning models, perform hyperparameter optimization,
# and report the final score based on the best model selected. Figures and tables are generate
# for the report:
results/model/confusion_matrix.png results/model/feature_coefficient.csv results/model/model_selection.csv results/model/precision_recall.png results/model/roc.png : data/processed/train_cleaned.csv data/processed/test_cleaned.csv src/data_analysis.py
	python3 src/data_analysis.py --traindata=data/processed/train_cleaned.csv --testdata=data/processed/test_cleaned.csv --output=results/model/

# Generate the report using an automated R markdown file. The Rmd file loads the previously 
# generated figures and tables, and presents them in the final file generated. 
doc/report.md doc/report.html: doc/report.Rmd doc/references.bib results/eda_results/categorical_result.png results/eda_results/binary_result.png results/eda_results/numeric_result.png results/model/confusion_matrix.png 
	Rscript -e "rmarkdown::render('doc/report.Rmd')"

# Clean data and report files.
clean :
	rm -f doc/report.md		# remove report markdown file
	rm -f doc/report.html		# remove report html file
	rm -rf data # remove the entire data folder
	rm -rf results # remove the entire results folder
  
