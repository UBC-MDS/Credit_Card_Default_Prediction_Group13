"""Usage: data_analysis.py [TRAINDATA] [TESTDATA] [OUTPUTFOLDER] ...

Arguments:
  TRAINDATA        path of the raw training data
  TESTDATA         path of the raw testing data
  OUTPUTFOLDER     folder that stores the generated plots


"""
from docopt import docopt
import numpy as np
import pandas as pd

def perform_ml_analysis(train_data_path, test_data_path, out_folder):
    if(not os.path.exists(out_folder)):
        os.mkdir(out_folder)
        
    headernames = [
        "ID",
        "LIMIT_BAL",
        "SEX",
        "EDUCATION",
        "MARRIAGE",
        "AGE",
        "PAY_0",
        "PAY_2",
        "PAY_3",
        "PAY_4",
        "PAY_5",
        "PAY_6",
        "BILL_AMT1",
        "BILL_AMT2",
        "BILL_AMT3",
        "BILL_AMT4",
        "BILL_AMT5",
        "BILL_AMT6",
        "PAY_AMT1",
        "PAY_AMT2",
        "PAY_AMT3",
        "PAY_AMT4",
        "PAY_AMT5",
        "PAY_AMT6",
        "default payment next month",
    ]
        
    train_df = pd.read_csv(train_data_path, index_col=0, names=headernames, skiprows=1, encoding="utf-8")
    test_df = pd.read_csv(test_data_path, index_col=0, names=headernames, skiprows=1, encoding="utf-8")
        
    


# Make sure you call this script in the utils folder
# Example: python3 data_eda.py ../data/processed/train_raw.csv ../data/eda_results/
if __name__ == '__main__':
    arguments = docopt(__doc__)

    train_data_path = arguments['TRAINDATA'] # Download 1 dataset at a time
    test_data_path = arguments['TESTDATA'] # Download 1 dataset at a time
    out_path = arguments['OUTPUTFOLDER'][0]
    ext_name = train_data_path.split('.')[-1]
    perform_ml_analysis(train_data_path, test_data_path, out_path)
