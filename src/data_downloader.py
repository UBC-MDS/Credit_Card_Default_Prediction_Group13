# authors: Chester Wang, HanChen Wang, Qurat-ul-Ain Azim, Renee Kwon
# date: 2022-11-24

"""Downloads the data from the host url and saves it in the output folder

Usage: src/data_downloader.py --url=<url> --path=<path> ...

Arguments:
  --url=<url>        optional input file
  --path=<path>        local file path

Example:
python src/data_downloader.py --url=https://archive.ics.uci.edu/ml/machine-learning-databases/00350/default%20of%20credit%20card%20clients.xls --path=data/raw/
"""
from docopt import docopt
import numpy as np
import pandas as pd
import requests, os
from sklearn.model_selection import train_test_split

# Downloads raw data from the Internet
def download_data(url, to='./data/', ext_name='csv'):
    if not (os.path.exists(to)):
        os.makedirs(to)

    file_data = requests.get(url).content
    with open(to + 'raw_data.' + ext_name, "wb") as file:
        file.write(file_data)


# Make sure you call this script in the repo's root path
# Example: python src/data_downloader.py --url=https://archive.ics.uci.edu/ml/machine-learning-databases/00350/default%20of%20credit%20card%20clients.xls --path=data/raw/
if __name__ == '__main__':
    arguments = docopt(__doc__)

    data_url = arguments['--url'] # Download 1 dataset at a time
    local_path = arguments['--path'][0]
    ext_name = data_url.split('.')[-1]

    # Unit tests
    assert ext_name == 'xls'
    assert 'https' in data_url
    assert 'data' in local_path and 'raw' in local_path

    download_data(data_url, to=local_path, ext_name=ext_name)

    # test if the file is stored properly
    assert os.path.exists('./data/raw/raw_data.xls')

    print("-- Data downloaded to: {}".format(local_path))
