# Makefile
# Authors: Chester Wang, HanChen Wang, Qurat-ul-Ain Azim, Renee Kwon
# Date: 2022-11-30

# This driver script completes the supervised machine learning data analysis on
# a credit card default prediction data set and creates a final report on whether
# a credit card holder will likely pay their bill next month. This script
# takes no arguments.

# example usage:
# make all
# make clean

# Run all to download & preprocess data, perform EDA, generate figures and build model.
# It will also generate the report on the credit default prediction analysis.
all: data/raw/raw_data.xls data/processed/test_cleaned.csv data/processed/test_raw.csv data/processed/train_cleaned.csv data/processed/train_raw.csv data/eda_results/binary_result.png data/eda_results/categorical_result.png data/eda_results/corr.csv data/eda_results/numeric_result.png data/results/confusion_matrix.png data/results/feature_coefficient.csv data/results/model_selection.csv data/results/precision_recall.png data/results/roc.png doc/report.md doc/report.html

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
data/eda_results/binary_result.png data/eda_results/categorical_result.png data/eda_results/corr.csv data/eda_results/numeric_result.png : data/processed/train_raw.csv src/data_eda.py
	python3 src/data_eda.py --traindata=data/processed/train_raw.csv --output=data/eda_results/

# Build and compare different machine learning models, perform hyperparameter optimization,
# and report the final score based on the best model selected. Figures and tables are generate
# for the report:
data/results/confusion_matrix.png data/results/feature_coefficient.csv data/results/model_selection.csv data/results/precision_recall.png data/results/roc.png : data/processed/train_cleaned.csv data/processed/test_cleaned.csv src/data_analysis.py
	python3 src/data_analysis.py --traindata=data/processed/train_cleaned.csv --testdata=data/processed/test_cleaned.csv --output=data/results/

# Generate the report using an automated R markdown file. The Rmd file loads the previously
# generated figures and tables, and presents them in the final file generated.
doc/report.md doc/report.html: doc/report.Rmd doc/references.bib data/eda_results/categorical_result.png data/eda_results/binary_result.png data/eda_results/numeric_result.png data/results/confusion_matrix.png
	Rscript -e "rmarkdown::render('doc/report.Rmd')"

# Clean data and report files.
clean :
	rm -f data/eda_results/*.png	# remove all eda figures
	rm -f data/eda_results/*.csv	# remove eda table
	rm -f data/processed/*.csv	# remove preprocessed data
	rm -f data/raw/*.xls		# remove raw data
	rm -f data/results/*.png	# remove analysis figures
	rm -f data/results/*.csv	# remove analysis tables
	rm -f doc/report.md		# remove report markdown file
	rm -f doc/report.html		# remove report html file
