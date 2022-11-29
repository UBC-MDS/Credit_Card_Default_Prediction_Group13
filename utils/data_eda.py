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
import seaborn as sns
import matplotlib.pyplot as plt
import vl_convert as vlc

# Save a vega-lite spec and a PNG blob for each plot in the notebook
alt.renderers.enable('mimetype')
alt.data_transformers.enable('data_server')
# Handle large data sets without embedding them in the notebook


# Reference: 531 Slack Channel by Joel
def save_chart(chart, filename, scale_factor=1):
    '''
    Save an Altair chart using vl-convert

    Parameters
    ----------
    chart : altair.Chart
        Altair chart to save
    filename : str
        The path to save the chart to
    scale_factor: int or float
        The factor to scale the image resolution by.
        E.g. A value of `2` means two times the default resolution.
    '''
    with alt.data_transformers.enable("default") and alt.data_transformers.disable_max_rows():
        if filename.split('.')[-1] == 'svg':
            with open(filename, "w") as f:
                f.write(vlc.vegalite_to_svg(chart.to_dict()))
        elif filename.split('.')[-1] == 'png':
            with open(filename, "wb") as f:
                f.write(vlc.vegalite_to_png(chart.to_dict(), scale=scale_factor))
        else:
            raise ValueError("Only svg and png formats are supported")



def perform_eda(train_data_path, out_folder):
    if not (os.path.exists(out_folder)):
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

    save_chart(categorical_chart, out_path + 'categorical_result.png')

    # Plotting the binary feature.
    binary_chart = alt.Chart(train_df).mark_bar(opacity=0.7).encode(
        x=alt.X("count()", title="Count of records"),
        y=alt.Y("SEX:N", title="SEX"),
        color=alt.Color("default payment next month:N"),
    )

    save_chart(binary_chart, out_path + 'binary_result.png')

    # Plotting numeric features.
    numeric_chart = alt.Chart(train_df).mark_bar(opacity=0.7).encode(
        x=alt.X(alt.repeat(), bin=alt.Bin(maxbins=30)),
        y=alt.Y("count()", title="Count of records"),
        color=alt.Color("default payment next month:N")
    ).properties(width=300, height=300).repeat(numeric_features, columns=3)

    save_chart(numeric_chart, out_path + 'numeric_result.png')

    train_df.corr('spearman').style.background_gradient()
    train_df.to_csv(out_path + 'corr.csv')



# Make sure you call this script in the utils folder
# Example: python3 data_eda.py ../data/processed/train_raw.csv ../data/eda_results/
if __name__ == '__main__':
    arguments = docopt(__doc__)

    train_data_path = arguments['TRAINDATA'] # Download 1 dataset at a time
    out_path = arguments['OUTPUTFOLDER'][0]
    perform_eda(train_data_path, out_path)
