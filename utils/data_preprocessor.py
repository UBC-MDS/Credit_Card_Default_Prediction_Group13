"""Usage: data_preprocessor.py [INPUTPATH] [OUTPUTFOLDER] ...

Arguments:
  INPUTPATH         path to raw data file
  OUTPUTFOLDER      folder to save the preprocessed and split dataset (as train.csv and test.csv)


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
def split_data(ext_name, raw_data_path, out_folder):
    if(not os.path.exists(out_folder)):
        os.mkdir(out_folder)

    # Skipping 2nd row specifically for our credit default dataset...
    raw_df = pd.read_excel(raw_data_path, skiprows=[0])

    # Make sure not to change the random seed
    train_df, test_df = train_test_split(raw_df, test_size=0.2, random_state=522)
    train_df.to_csv(out_folder + 'train_raw.csv')
    test_df.to_csv(out_folder + 'test_raw.csv')


# Cleaning data for transformation
def clean_data(csv_path):
    df = pd.read_csv(csv_path)
    # print(df.info()) # No missing value found.
    # print(df.describe()) # No significant outlier found
    df = df.drop(['Unnamed: 0', 'ID'], axis=1) # Drop irrelevant columns
    df = df.rename(columns={'default payment next month': 'TARGET'})
    # print(df.head())
    return df


# Transforming data to appropriate type and range
def transform_data(df, name):

    categorical_features = [
        "EDUCATION",
        "MARRIAGE",
        "PAY_0",
        "PAY_2",
        "PAY_3",
        "PAY_4",
        "PAY_5",
        "PAY_6",
    ]

    target = ['TARGET']

    binary_features = ["SEX"]

    numeric_features = [
        "LIMIT_BAL",
        "AGE",
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
    ]

    ct = make_column_transformer(
        (StandardScaler(), numeric_features),  # scaling on numeric features
        ("passthrough", target),  # no transformations on the binary features
        (OneHotEncoder(), categorical_features),  # OHE on categorical features
        (OneHotEncoder(drop='if_binary'), binary_features),
    )

    transformed_df = ct.fit_transform(df)
    column_names = ct.named_transformers_['standardscaler'].get_feature_names_out().tolist() +\
                   target +\
                   ct.named_transformers_["onehotencoder-1"].get_feature_names_out().tolist() +\
                   ct.named_transformers_["onehotencoder-2"].get_feature_names_out().tolist()
    transformed_df = pd.DataFrame(transformed_df.toarray(), columns=column_names)

    transformed_df.to_csv(out_folder + '{}_processed.csv'.format(name))




# Make sure you call this script in the utils folder
# Example: python3 data_preprocessor.py ../data/raw/raw_data.xls ../data/processed/
if __name__ == '__main__':
    arguments = docopt(__doc__)

    input_path = arguments['INPUTPATH'] # Points to the raw dataset
    out_folder = arguments['OUTPUTFOLDER'][0]
    ext_name = input_path.split('.')[-1]
    split_data(ext_name, input_path, out_folder)

    # Process Training data
    cleaned_df_train = clean_data(out_folder + 'train_raw.csv')
    transform_data(cleaned_df_train, 'train')
    print("-- Training data is ready at {}".format(out_folder))

    # Process Testing data
    cleaned_df_test = clean_data(out_folder + 'test_raw.csv')
    transform_data(cleaned_df_test, 'test')
    print("-- Testing data is ready at {}".format(out_folder))
