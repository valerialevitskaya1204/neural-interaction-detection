import pandas as pd
import functools
import requests

from io import BytesIO
from zipfile import ZipFile
from sklearn.datasets import fetch_california_housing
from ucimlrepo import fetch_ucirepo


def digits_only(df):
    categorical_columns = df.select_dtypes(include=['object', 'category']).columns
    df_cleaned = df.drop(columns=categorical_columns)
    
    return df_cleaned

def handle_errors(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        try:
            print(f"Processing {args[0] if args else ''}")
            result = func(*args, **kwargs)
            print(f"Successfully Loaded {args[0] if args else ''}!")
            return result
        except Exception as e:
            print(f"Something went wrong: {e}")

    return wrapper


@handle_errors
def load_real_dataset(dataset_name):
    match dataset_name:
        case "cal_housing":
            X, Y = fetch_california_housing(return_X_y=True)
        case "bike_sharing":
            url = "http://archive.ics.uci.edu/ml/machine-learning-databases/00275/Bike-Sharing-Dataset.zip"
            response = requests.get(url)
            zipfile = ZipFile(BytesIO(response.content))
            with zipfile.open("day.csv") as f:
                df = pd.read_csv(f)
                df = digits_only(df)
            Y = df["cnt"].values
            X = df.drop(
                ["cnt"], axis=1
            ).values
        case "higgs_boson":
            url = "https://archive.ics.uci.edu/ml/machine-learning-databases/00280/HIGGS.csv.gz"
            df = pd.read_csv(url, compression="gzip", header=None)
            df = digits_only(df)
            Y = df.iloc[:, 0].values
            X = df.iloc[:, 1:].values
        case "letter":
            url = "http://archive.ics.uci.edu/ml/machine-learning-databases/letter-recognition/letter-recognition.data"
            df = pd.read_csv(url, header=None)
            df = digits_only(df)
            Y = df.iloc[:, 0].values
            X = df.iloc[:, 1:].values
        case _:
            print("Going to another func...")
            load_new_dataset(dataset_name)

    return X, Y


@handle_errors
def load_new_dataset(dataset_name):
    match dataset_name:
        case "parkinsons":
            parkinsons_telemonitoring = fetch_ucirepo(id=189)
            X1 = parkinsons_telemonitoring.data.features
            X = X.drop(
                ["age", "sex"], axis=1
            )
            X = digits_only(X)
            Y = parkinsons_telemonitoring.data.targets.drop(["motor_UPDRS"], axis=1)

        case "images":
            image_segmentation = fetch_ucirepo(id=50)

            X = image_segmentation.data.features
            X = digits_only(X)
            Y = image_segmentation.data.targets

        case "robots":
            url = "http://archive.ics.uci.edu/static/public/963/ur3+cobotops.zip"

            response = requests.get(url)
            zipfile = ZipFile(BytesIO(response.content))
            with zipfile.open("dataset_02052023.xlsx") as f:
                df = pd.read_excel(f)

            Y = df["grip_lost"].astype(int)
            X = df.drop(
                ["Num", "Timestamp", "cycle ", "grip_lost", "Robot_ProtectiveStop"],
                axis=1,
            )
            X = digits_only(X)
        case "seoul_bikes":
            seoul_bike_sharing_demand = fetch_ucirepo(id=560)

            X = seoul_bike_sharing_demand.data.features.drop(
                ["Date", "Seasons", "Holiday"], axis=1
            )
            X = digits_only(X)
            Y = seoul_bike_sharing_demand.data.targets
            
        case _:
            load_real_dataset(dataset_name)
    return X, Y


if __name__ == "__main__":
    load_new_dataset("robots")
