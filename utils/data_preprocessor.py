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
    # df = df.drop(['Unnamed: 0', 'ID'], axis=1) # Drop irrelevant columns
    # df = df.rename(columns={'default payment next month': 'TARGET'})
    # print(df.head())
    return df


# Transforming data to appropriate type and range
def transform_data(train_df, test_df):

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

    categorical_features = ["MARRIAGE", "SEX"]

    drop = ["ID"]

    passthrough_features = [
        "PAY_0",
        "PAY_2",
        "PAY_3",
        "PAY_4",
        "PAY_5",
        "PAY_6",
        "EDUCATION",
    ]

    # Create the column transformer
    preprocessor = make_column_transformer(
        (StandardScaler(), numeric_features),
        (
            "passthrough",
            passthrough_features,
        ),
        (
            OneHotEncoder(drop="if_binary", handle_unknown="ignore", sparse=False),
            categorical_features,
        ),
        ("drop", drop),
    )

    # 2. Fit and transform on the training data
    X_train = train_df.drop(columns=["default payment next month"])
    X_test = test_df.drop(columns=["default payment next month"])
    y_train = train_df["default payment next month"]
    y_test = test_df["default payment next month"]

    # This line nicely formats the feature names from `preprocessor.get_feature_names_out()`
    # so that we can more easily use them below
    preprocessor.verbose_feature_names_out = False
    # Create a dataframe with the transformed features and column names
    preprocessor.fit(X_train)

    # transformed data
    X_train_transformed = preprocessor.transform(X_train)
    X_test_transformed = preprocessor.transform(X_test)
    ohe_features = (
        preprocessor.named_transformers_["onehotencoder"].get_feature_names_out().tolist()
    )

    # Code to get all the feature names
    feature_names = numeric_features + passthrough_features + ohe_features

    X_train_enc = pd.DataFrame(X_train_transformed, columns=feature_names)
    X_test_enc = pd.DataFrame(X_test_transformed, columns=feature_names)

    X_train_enc.to_csv(out_folder + 'X_train_transformed.csv')
    X_test_enc.to_csv(out_folder + 'X_test_transformed.csv')
    y_train.to_csv(out_folder + 'y_train.csv')
    y_test.to_csv(out_folder + 'y_test.csv')


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
    cleaned_df_test = clean_data(out_folder + 'test_raw.csv')
    transform_data(cleaned_df_train, cleaned_df_test)
    print("-- Transformed data available at: {}".format(out_folder))
