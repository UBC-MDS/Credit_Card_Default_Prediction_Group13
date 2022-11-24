python3 data_downloader.py https://archive.ics.uci.edu/ml/machine-learning-databases/00350/default%20of%20credit%20card%20clients.xls ../data/raw/
python3 data_preprocessor.py ../data/raw/raw_data.xls ../data/processed/
python3 data_eda.py ../data/processed/train_raw.csv ../data/eda_results/
python3 data_analysis.py ../data/processed/train_cleaned.csv ../data/processed/test_cleaned.csv ../data/results
