# Makefile
# Authors: Chester Wang, HanChen Wang, Qurat-ul-Ain Azim, Renee Kwon
# Date: 2022-11-30


# Necessary dependencies installation to run all the scripts in the project
install:
	python3 -m pip install numpy==1.23.3 docopt==0.6.2 scikit-learn==1.1.2 matplotlib==3.5.3 pandas==1.4.4 scipy==1.9.2 altair==4.2.0 vl-convert-python==0.5.0

# Run all to download & preprocess data, perform EDA, generate figures and build model
all: data/eda_results/binary_result.png data/results/confusion_matrix.png

# The automated data download can be performed via running the following command using terminal:
data/raw/raw_data.xls : src/data_downloader.py
	python3 src/data_downloader.py https://archive.ics.uci.edu/ml/machine-learning-databases/00350/default%20of%20credit%20card%20clients.xls data/raw/

# Some preprocessing, data cleaning and splitting performed with the following command:
data/processed/test_cleaned.csv data/processed/test_raw.csv data/processed/train_cleaned.csv data/processed/train_raw.csv : data/raw/raw_data.xls src/data_preprocessor.py
	python3 src/data_preprocessor.py data/raw/raw_data.xls data/processed/

# Perform EDA and generate figures from the training data
data/eda_results/binary_result.png data/eda_results/categorical_result.png data/eda_results/corr.csv data/eda_results/numeric_result.png : data/processed/train_raw.csv src/data_eda.py
	python3 src/data_eda.py data/processed/train_raw.csv data/eda_results/

# Finally, the model building and predictive analysis can be done by running the following code in the terminal:
data/results/confusion_matrix.png data/results/feature_coefficient.csv data/results/model_selection.csv data/results/precision_recall.png data/results/roc.png : data/processed/train_cleaned.csv data/processed/test_cleaned.csv src/data_analysis.py
	python3 src/data_analysis.py data/processed/train_cleaned.csv data/processed/test_cleaned.csv data/results/

# # write the report
doc/report.html : doc/report.Rmd doc/references.bib data/eda_results/categorical_result.png data/eda_results/binary_result.png data/eda_results/numeric_result.png data/results/confusion_matrix.png 
	Rscript -e "rmarkdown::render('doc/report.Rmd')"

# # clean :
# #	rm -f results/*.dat
# #	rm -f figures/*.png
# #	rm -f doc/*.html

