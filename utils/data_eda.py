"""Usage: data_eda.py [TRAINDATA] [OUTPUTFOLDER] ...

Arguments:
  TRAINDATA        path of the raw training data
  OUTPUTFOLDER     folder that stores the generated plots


"""
from docopt import docopt
import numpy as np
import pandas as pd
import altair as alt
import os
from pandas_profiling import ProfileReport
from altair_data_server import data_server
from altair_saver import save
import seaborn as sns
import matplotlib.pyplot as plt

# Save a vega-lite spec and a PNG blob for each plot in the notebook
alt.renderers.enable('mimetype')
alt.data_transformers.enable('data_server')
alt.renderers.enable('altair_saver', fmts=['vega-lite', 'png'])
# Handle large data sets without embedding them in the notebook


def perform_eda(train_data_path, out_folder):
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

    binary_features = ["SEX"]

    drop = ["ID", "default payment next month"]

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


    # Plotting categorical features
    categorical_chart = alt.Chart(train_df).mark_bar(opacity=0.7).encode(
        x=alt.X(alt.repeat()),
        y=alt.Y("count()", title="Count of records"),
        color=alt.Color("default payment next month:N"),
    ).properties(width=200, height=100).repeat(categorical_features, columns=3)

    save(categorical_chart, out_path + 'categorical_result.png')

    # Plotting the binary feature.
    binary_chart = alt.Chart(train_df).mark_bar(opacity=0.7).encode(
        x=alt.X("count()", title="Count of records"),
        y=alt.Y("SEX:N", title="SEX"),
        color=alt.Color("default payment next month:N"),
    )

    save(binary_chart, out_path + 'binary_result.png')

    # Plotting numeric features.
    numeric_chart = alt.Chart(train_df).mark_bar(opacity=0.7).encode(
        x=alt.X(alt.repeat(), bin=alt.Bin(maxbins=30)),
        y=alt.Y("count()", title="Count of records"),
        color=alt.Color("default payment next month:N")
    ).properties(width=300, height=300).repeat(numeric_features, columns=3)

    save(numeric_chart, out_path + 'numeric_result.png')

    matrix = train_df.corr('spearman')
    sns.set(rc={'figure.figsize':(25,9)})
    sns.heatmap(matrix, annot=True)
    plt.savefig(out_path + 'correlation_matrix.png')

    # profile = ProfileReport(train_df, title="Pandas Profiling Report")  # , minimal=True)
    # profile.to_file(out_path + "summary.html")



# Make sure you call this script in the utils folder
# Example: python3 data_eda.py ../data/processed/train_raw.csv ../data/eda_results/
if __name__ == '__main__':
    arguments = docopt(__doc__)

    train_data_path = arguments['TRAINDATA'] # Download 1 dataset at a time
    out_path = arguments['OUTPUTFOLDER'][0]
    ext_name = train_data_path.split('.')[-1]
    perform_eda(train_data_path, out_path)
