import pandas as pd

from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

from prefect import flow, task, get_run_logger
from prefect.task_runners import SequentialTaskRunner

from datetime import datetime, timedelta
from dateutil.relativedelta import relativedelta

import pickle

@task
def read_data(path):
    df = pd.read_parquet(path)
    return df

@task
def prepare_features(df, categorical, train=True):
    logger = get_run_logger()

    df['duration'] = df.dropOff_datetime - df.pickup_datetime
    df['duration'] = df.duration.dt.total_seconds() / 60
    df = df[(df.duration >= 1) & (df.duration <= 60)].copy()

    mean_duration = df.duration.mean()

    if train:
        logger.info(f"The mean duration of training is {mean_duration}")
    else:
        logger.info(f"The mean duration of validation is {mean_duration}")
    
    df[categorical] = df[categorical].fillna(-1).astype('int').astype('str')
    return df

@task
def train_model(df, categorical, date):
    logger = get_run_logger()
    
    train_dicts = df[categorical].to_dict(orient='records')
    dv = DictVectorizer()
    X_train = dv.fit_transform(train_dicts) 
    y_train = df.duration.values

    logger.info(f"The shape of X_train is {X_train.shape}")
    logger.info(f"The DictVectorizer has {len(dv.feature_names_)} features")

    lr = LinearRegression()
    lr.fit(X_train, y_train)
    y_pred = lr.predict(X_train)
    mse = mean_squared_error(y_train, y_pred, squared=False)
    logger.info(f"The MSE of training is: {mse}")
    # model-{date}.pkl
    model_file = "models/model-{}.pkl".format(date)
    with open(model_file, "wb") as f_out:
        pickle.dump(lr, f_out)

    val_file = "models/dv-{}.pkl".format(date)
    with open(val_file, "wb") as f_out:
        pickle.dump(dv, f_out)
        
    return lr, dv

@task
def run_model(df, categorical, dv, lr):
    logger = get_run_logger()

    val_dicts = df[categorical].to_dict(orient='records')
    X_val = dv.transform(val_dicts) 
    y_pred = lr.predict(X_val)
    y_val = df.duration.values

    mse = mean_squared_error(y_val, y_pred, squared=False)
    logger.info(f"The MSE of validation is: {mse}")
    return

def get_paths(curr_date):
    logger = get_run_logger()

    if curr_date is None:
        train_date = datetime.now() - relativedelta(months=2)
        val_date  = datetime.now() - relativedelta(months=1)
    else:
        train_date = datetime.strptime(curr_date, '%Y-%m-%d') - relativedelta(months=2)
        val_date = datetime.strptime(curr_date, '%Y-%m-%d') - relativedelta(months=1)

    if train_date.month < 10:
        train_date_yyyy_mm = "{}-0{}".format(train_date.year, train_date.month)
        val_date_yyyy_mm = "{}-0{}".format(val_date.year, val_date.month)
    else:
        train_date_yyyy_mm = "{}-{}".format(train_date.year, train_date.month)
        val_date_yyyy_mm = "{}-{}".format(val_date.year, val_date.month)

    train_path = "./data/fhv_tripdata_{}.parquet".format(str(train_date_yyyy_mm))
    val_path = "./data/fhv_tripdata_{}.parquet".format(str(val_date_yyyy_mm))

    return train_path, val_path


@flow(task_runner=SequentialTaskRunner())
def main(date="2021-08-15"):
    train_path, valid_path = get_paths(date)
    
    categorical = ['PUlocationID', 'DOlocationID']

    df_train = read_data(train_path)
    df_train_processed = prepare_features(df_train, categorical)

    df_val = read_data(valid_path)
    df_val_processed = prepare_features(df_val, categorical, False)

    # train the model
    lr, dv = train_model(df_train_processed, categorical, date).result()
    run_model(df_val_processed, categorical, dv, lr)

 
from prefect.deployments import DeploymentSpec
from prefect.orion.schemas.schedules import CronSchedule
from prefect.flow_runners import SubprocessFlowRunner
from datetime import timedelta

DeploymentSpec(
    flow=main,
    name="model_training_2021-08-15",
    schedule=CronSchedule(
        cron="0 9 15 * *",
        timezone="America/New_York"),
    flow_runner=SubprocessFlowRunner(),
    tags=["ml"]
)
# main("2021-08-15")
# Run
# $ prefect deployment create homework.py