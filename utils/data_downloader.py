"""Usage: data_downloader.py [LINK] [PATH] ...

Arguments:
  LINK        optional input file
  PATH        local file path


"""
from docopt import docopt
import numpy as np
import pandas as pd
import requests, os
from sklearn.model_selection import train_test_split


def download_data(url, to='../data/raw/', ext_name='csv'):
    if(not os.path.exists(to)):
        os.mkdir('../data/')
        os.mkdir(to)

    file_data = requests.get(url).content
    with open(to + 'raw_data.' + ext_name, "wb") as file:
        file.write(file_data)


def split_data(ext_name, local_path='../data/raw/'):
    split_path = '../data/split/'
    if(not os.path.exists(split_path)):
        os.mkdir(split_path)

    raw_df = None
    if(ext_name in ['xls', 'xlsx', 'xlsm', 'xlsb']):
        raw_df = pd.read_excel(local_path + 'raw_data.' + ext_name, skiprows=[2])
        # Skipping 2nd row specifically for our credit default dataset...

    elif(ext_name in ['csv']):
        raw_df = pd.read_csv(local_path + 'raw_data.' + ext_name)

    else:
        raise Exception("Data Type Unverified. Check URL")

    # Make sure not to change the random seed
    train_df, test_df = train_test_split(raw_df, test_size=0.2, random_state=522)
    train_df.to_csv(split_path + 'train.csv')
    test_df.to_csv(split_path + 'test.csv')




# Make sure you call this script in the utils folder
# So like python3 data_downloader.py <url>
# NOT LIKE: python3 <some_path>/data_downloader.py <url>
if __name__ == '__main__':
    arguments = docopt(__doc__)
    print(arguments)

    data_url = arguments['LINK'] # Download 1 dataset at a time
    local_path = arguments['PATH'][0]
    ext_name = data_url.split('.')[-1]
    download_data(data_url, to=local_path, ext_name=ext_name)
    split_data(ext_name)
