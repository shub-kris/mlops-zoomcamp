import argparse
import pickle
import sys

import mlflow
import pandas as pd
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import Lasso
from sklearn.metrics import mean_squared_error


def get_data_from_path(path: str):
    df = pd.read_parquet(path)

    # Preprocessing data
    df.lpep_dropoff_datetime = pd.to_datetime(df.lpep_dropoff_datetime)
    df.lpep_pickup_datetime = pd.to_datetime(df.lpep_pickup_datetime)

    df["duration"] = df.lpep_dropoff_datetime - df.lpep_pickup_datetime
    df.duration = df.duration.apply(lambda td: td.total_seconds() / 60)

    df = df[(df.duration >= 1) & (df.duration <= 60)]

    categorical = ["PULocationID", "DOLocationID"]
    df[categorical] = df[categorical].astype(str)
    df["PU_DO"] = df["PULocationID"] + "_" + df["DOLocationID"]

    return df


## Configure MLFlow
def configure_mlflow(tracking_uri: str):
    # set_tracking_uri points to the backend.
    mlflow.set_tracking_uri(tracking_uri)


def parse_arguments(argv):
    parser = argparse.ArgumentParser(description="mlflow-demo")

    parser.add_argument(
        "-t",
        "--tracking-uri",
        help="SQLite backend to save and retrieve experiment data",
        default="sqlite:///mlflow.db",
    )

    args, _ = parser.parse_known_args(args=argv)
    return args


if __name__ == "__main__":
    args = parse_arguments(sys.argv)

    configure_mlflow(args.tracking_uri)

    # Name of experiment
    mlflow.set_experiment("nyc-taxi-train-experiment")
    train_data, val_data = get_data_from_path(
        "./data/green_tripdata_2021-01.parquet"
    ), get_data_from_path("./data/green_tripdata_2021-02.parquet")

    ## Extract features from data

    categorical = ["PU_DO"]  #'PULocationID', 'DOLocationID']
    numerical = ["trip_distance"]

    dv = DictVectorizer()

    train_dicts = train_data[categorical + numerical].to_dict(orient="records")
    X_train = dv.fit_transform(train_dicts)

    val_dicts = val_data[categorical + numerical].to_dict(orient="records")
    X_val = dv.transform(val_dicts)

    # Extract labels
    target = "duration"
    y_train = train_data[target].values
    y_val = val_data[target].values

    for alpha in [0.1, 0.3]:
        # wrap your code with this
        with mlflow.start_run():

            # tags are optional. They are useful for large teams and organization purposes
            # first param is the key and the second is the value
            mlflow.set_tag("developer", "shubham-krishna")

            # log any param that may be significant for your experiment.
            mlflow.log_param("train-data-path", "./data/green_tripdata_2021-01.parquet")
            mlflow.log_param("valid-data-path", "./data/green_tripdata_2021-02.parquet")

            # logging hyperparams; alpha in this example
            mlflow.log_param("alpha", alpha)

            lr = Lasso(alpha)
            lr.fit(X_train, y_train)

            y_pred = lr.predict(X_val)
            rmse = mean_squared_error(y_val, y_pred, squared=False)

            # and logging metrics
            mlflow.log_metric("rmse", rmse)

            with open("./models/lasso.bin", "wb") as f_out:
                pickle.dump(lr, f_out)

            # Tracking our model
            mlflow.log_artifact(
                local_path="./models/lasso.bin", artifact_path="models_pickle"
            )

            # Tracking preprocessor
            with open("models/preprocessor.b", "wb") as f_out:
                pickle.dump(dv, f_out)
            mlflow.log_artifact("./models/preprocessor.b", artifact_path="preprocessor")
