# Makefile
# Authors: Chester Wang, HanChen Wang, Qurat-ul-Ain Azim, Renee Kwon
# Date: 2022-11-29


install:
	python3 -m pip install numpy==1.23.3 docopt==0.6.2 scikit-learn==1.1.2 matplotlib==3.5.3 pandas==1.4.4 scipy==1.9.2 altair==4.2.0 vl-convert-python==0.5.0

# The automated data download can be performed via running the following command using terminal:
# data/raw/raw_data.xls : src/data_downloader.py
# 	python3 src/data_downloader.py https://archive.ics.uci.edu/ml/machine-learning-databases/00350/default%20of%20credit%20card%20clients.xls data/raw/

# Some preprocessing, data cleaning and splitting canbe performed via the following command:
# data/processed/test_cleaned.csv data/processed/test_raw.csv data/processed/train_cleaned.csv data/processed/train_raw.csv : data/raw/raw_data.xls utils/data_preprocessor.py
# 	python3 utils/data_preprocessor.py data/raw/raw_data.xls data/processed/

# [output] : data/processed/train_raw.csv data_eda.py
# 	python3 data_eda.py data/processed/train_raw.csv data/eda_results/

# Finally, the model building and predictive analysis can be done by running the following code in the terminal:
# [output] : data/processed/train_cleaned.csv data/processed/test_cleaned.csv data_analysis.py
# 	python3 data_analysis.py data/processed/train_cleaned.csv data/processed/test_cleaned.csv data/results



# # all : doc/report.html
#
# # count the words
# results/isles.dat : data/isles.txt src/countwords.py
# 	python src/countwords.py data/isles.txt results/isles.dat
#
# results/abyss.dat : data/abyss.txt src/countwords.py
# 	python src/countwords.py data/abyss.txt results/abyss.dat
#
# results/last.dat : data/last.txt src/countwords.py
# 	python src/countwords.py data/last.txt results/last.dat
#
# results/sierra.dat : data/sierra.txt src/countwords.py
# 	python src/countwords.py data/sierra.txt results/sierra.dat
#
#
#
# # create the plots
# figures/isles.png : results/isles.dat src/plotcount.py
# 	python src/plotcount.py results/isles.dat figures/isles.png
#
# figures/abyss.png : results/abyss.dat src/plotcount.py
# 	python src/plotcount.py results/abyss.dat figures/abyss.png
#
# figures/last.png : results/last.dat src/plotcount.py
# 	python src/plotcount.py results/last.dat figures/last.png
#
# figures/sierra.png : results/sierra.dat src/plotcount.py
# 	python src/plotcount.py results/sierra.dat figures/sierra.png
#
# # write the report
# doc/count_report.html : figures/isles.png figures/abyss.png figures/sierra.png figures/last.png
# 	Rscript -e "rmarkdown::render('doc/count_report.Rmd')"
#
#
# # clean :
# #	rm -f results/*.dat
# #	rm -f figures/*.png
# #	rm -f doc/*.html
#
#
#
