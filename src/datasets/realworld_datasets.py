import pandas as pd
from sklearn.datasets import fetch_california_housing
import requests
from io import BytesIO
from zipfile import ZipFile

def load_real_dataset(dataset_name):
    if dataset_name == "cal_housing":
        X, Y = fetch_california_housing(return_X_y=True)
    elif dataset_name == "bike_sharing":
        url = "http://archive.ics.uci.edu/ml/machine-learning-databases/00275/Bike-Sharing-Dataset.zip"
        response = requests.get(url)
        zipfile = ZipFile(BytesIO(response.content))
        with zipfile.open('day.csv') as f:
            df = pd.read_csv(f)
        Y = df['cnt'].values
        X = df.drop(['instant', 'dteday', 'cnt', 'casual', 'registered'], axis=1).values
    elif dataset_name == "higgs_boson":
        url = "https://archive.ics.uci.edu/ml/machine-learning-databases/00280/HIGGS.csv.gz"
        df = pd.read_csv(url, compression='gzip', header=None)
        Y = df.iloc[:, 0].values
        X = df.iloc[:, 1:].values
    elif dataset_name == "letter":
        url = "http://archive.ics.uci.edu/ml/machine-learning-databases/letter-recognition/letter-recognition.data"
        df = pd.read_csv(url, header=None)
        Y = df.iloc[:, 0].values
        X = df.iloc[:, 1:].values
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")
    return X, Y