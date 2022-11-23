"""Usage: data_analysis.py [TRAINDATA] [TESTDATA] [OUTPUTFOLDER] ...

Arguments:
  TRAINDATA        path of the raw training data
  TESTDATA         path of the raw testing data
  OUTPUTFOLDER     folder that stores the generated plots


"""
from docopt import docopt
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer, make_column_transformer
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler, OrdinalEncoder


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
        
    
    # 1. Create the column transformer / preprocessor
    # For the numeric features, we will Standardize the numbers. 
    # For the two categorical features, we will apply OneHotEncoder on them to convert
    # the column into multiple binary features. 
    # Many other features already have good encoding, so we don't
    # have to modify them (passthrough). 

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

    # Show the preprocessor
    preprocessor
    
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
    ohe_features = (
        preprocessor.named_transformers_["onehotencoder"].get_feature_names_out().tolist()
    )

    # Code to get all the feature names
    feature_names = numeric_features + passthrough_features + ohe_features

    X_train_enc = pd.DataFrame(X_train_transformed, columns=feature_names)

    # Show the transformed data
    X_train_enc
    
    
    
    
    
    
    
    
    
    
    
    

# Make sure you call this script in the utils folder
# Example: python3 data_eda.py ../data/processed/train_raw.csv ../data/eda_results/
if __name__ == '__main__':
    arguments = docopt(__doc__)

    train_data_path = arguments['TRAINDATA'] # Download 1 dataset at a time
    test_data_path = arguments['TESTDATA'] # Download 1 dataset at a time
    out_path = arguments['OUTPUTFOLDER'][0]
    ext_name = train_data_path.split('.')[-1]
    perform_ml_analysis(train_data_path, test_data_path, out_path)
