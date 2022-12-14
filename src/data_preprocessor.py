# authors: Chester Wang, HanChen Wang, Qurat-ul-Ain Azim, Renee Kwon
# date: 2022-11-24

"""This script read the raw data from the input file and splits it on a 80-20 train test split ratio. Some feature cleaning are performed to remove ambiguity. Eventually it saves the cleaned data in .csv files in the output folder.

Usage: data_preprocessor.py --input=<input> --output=<output> ...

Arguments:
  --input=<input>        path to raw data file
  --output=<output>      folder to save the preprocessed and split dataset (as train.csv and test.csv)

Example:
python src/data_preprocessor.py --input=data/raw/raw_data.xls --output=data/processed/
"""
from docopt import docopt
import numpy as np
import pandas as pd
import requests, os
from sklearn.model_selection import train_test_split
from sklearn.compose import make_column_transformer
from sklearn.preprocessing import (
    MinMaxScaler,
    OneHotEncoder,
    OrdinalEncoder,
    StandardScaler,
)
from sklearn.pipeline import Pipeline, make_pipeline


# Split the data into train.csv and test.csv
def preprocess_data(raw_data_path, out_folder):
    if not (os.path.exists(out_folder)):
        os.makedirs(out_folder)

    # Skipping 2nd row specifically for our credit default dataset...
    raw_df = pd.read_excel(raw_data_path, skiprows=[0])
    train_df, test_df = train_test_split(raw_df, test_size=0.2, random_state=522)
    train_df.to_csv(out_folder + 'train_raw.csv')
    test_df.to_csv(out_folder + 'test_raw.csv')

    # Data cleaning
    raw_df["EDUCATION"] = raw_df.loc[:,"EDUCATION"].replace([0,4,5,6],3).replace([1,2,3,4],[3,2,1,4])
    raw_df["MARRIAGE"] = raw_df.loc[:,"MARRIAGE"].replace([0],3)

    # Make sure not to change the random seed
    train_df_cleaned, test_df_cleaned = train_test_split(raw_df, test_size=0.2, random_state=522)
    train_df_cleaned.to_csv(out_folder + 'train_cleaned.csv')
    test_df_cleaned.to_csv(out_folder + 'test_cleaned.csv')




# Make sure you call this script in the repo's root path
# Example: python src/data_preprocessor.py --input=data/raw/raw_data.xls --output=data/processed/
if __name__ == '__main__':
    arguments = docopt(__doc__)

    input_path = arguments['--input'] # Points to the raw dataset
    out_folder = arguments['--output'][0]

    # Tests arguments
    assert input_path.endswith('.xls')
    assert 'data' in out_folder and 'processed' in out_folder

    preprocess_data(input_path, out_folder)

    # Tests that the files are generated as expected
    assert os.path.exists('./data/processed/test_cleaned.csv')
    assert os.path.exists('./data/processed/train_cleaned.csv')
    assert os.path.exists('./data/processed/test_raw.csv')
    assert os.path.exists('./data/processed/train_raw.csv')

    print("-- Cleaned data available at: {}".format(out_folder))
