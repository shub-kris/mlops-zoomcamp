import argparse
import pickle
import sys

import mlflow
import pandas as pd
import xgboost as xgb
from hyperopt import STATUS_OK, Trials, fmin, hp, tpe
from hyperopt.pyll import scope
from sklearn.feature_extraction import DictVectorizer
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


# params contains the hyperparameters for xgboost for a specific run
# Define the objective function to optimize
def objective(params):

    with mlflow.start_run():

        # set a tag for easier classification and log the hyperparameters
        mlflow.set_tag("model", "xgboost")
        mlflow.log_param("train-data-path", "./data/green_tripdata_2021-01.parquet")
        mlflow.log_param("valid-data-path", "./data/green_tripdata_2021-02.parquet")
        mlflow.log_params(params)

        # model definition and training
        booster = xgb.train(
            params=params,
            dtrain=train,
            num_boost_round=1000,
            evals=[(valid, "validation")],
            early_stopping_rounds=50,
        )

        # predicting with the validation set
        y_pred = booster.predict(valid)

        # rmse metric and logging
        rmse = mean_squared_error(y_val, y_pred, squared=False)
        mlflow.log_metric("rmse", rmse)

    # we return a dict with the metric and the OK signal
    return {"loss": rmse, "status": STATUS_OK}


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

    # Create the matrices for xgboost
    train = xgb.DMatrix(X_train, label=y_train)
    valid = xgb.DMatrix(X_val, label=y_val)

    ## Define the search space for hyperparameters
    search_space = {
        "max_depth": scope.int(hp.quniform("max_depth", 4, 100, 1)),
        "learning_rate": hp.loguniform("learning_rate", -3, 0),
        "reg_alpha": hp.loguniform("reg_alpha", -5, -1),
        "reg_lambda": hp.loguniform("reg_lambda", -6, -1),
        "min_child_weight": hp.loguniform("min_child_weight", -1, 3),
        "objective": "reg:linear",
        "seed": 42,
    }

    ## Minimize the objective function
    best_result = fmin(
        fn=objective, space=search_space, algo=tpe.suggest, max_evals=5, trials=Trials()
    )
