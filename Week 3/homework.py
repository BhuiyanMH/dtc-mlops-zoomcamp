import datetime
import pickle
from datetime import date as dt

import pandas as pd
from prefect import flow, get_run_logger, task
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error


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
def train_model(df, categorical):

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

@task
def get_paths(date):
    
    # get the logger
    logger = get_run_logger()
    
    logger.info(f"Generating paths for: {date}")
    
    # validation month
    current = date.replace(day=1)
    last_month = current - datetime.timedelta(days=1)
    val_month = last_month.month
    val_year = last_month.year
    
    # train month
    last_month = last_month.replace(day=1)
    last_last_month = last_month - datetime.timedelta(days=1)
    train_month = last_last_month.month
    train_year = last_last_month.year
    
    # generate the paths
    train_path = f"./data/fhv_tripdata_{train_year}-{train_month:02d}.parquet"
    val_path = f"./data/fhv_tripdata_{val_year}-{val_month:02d}.parquet"
    
    logger.info(f"Train path: {train_path}")
    logger.info(f"Validation path: {val_path}")

    return train_path, val_path


@flow
def main(date = None):

    # process the None date or string date
    if date == None:
        date = dt.today()
    else:
        date = datetime.datetime.strptime(date, '%Y-%m-%d')
    
    # generate train and validation path
    train_path, val_path = get_paths(date).result()
    
    # preprocess categorical columns
    categorical = ['PUlocationID', 'DOlocationID']

    df_train = read_data(train_path)
    df_train_processed = prepare_features(df_train, categorical)

    df_val = read_data(val_path)
    df_val_processed = prepare_features(df_val, categorical, False)

    # train the model
    lr, dv = train_model(df_train_processed, categorical).result()
    
    # save model
    model_path = f"./models/model-{date:%y-%m-%d}.bin"
    with open(model_path, "wb") as f_out:
            pickle.dump(lr, f_out)
    
    # save the dictionary vectorizer
    dv_path = f"./models/dv-{date:%y-%m-%d}.b"
    with open(dv_path, "wb") as f_out:
            pickle.dump(dv, f_out)
    
    # run model
    run_model(df_val_processed, categorical, dv, lr)

# main(date="2021-08-15")

# Creating a Prefect deployment
from prefect.deployments import DeploymentSpec
from prefect.flow_runners import SubprocessFlowRunner
from prefect.orion.schemas.schedules import CronSchedule

DeploymentSpec(
    flow=main,
    name="model_training-h3",
    schedule=CronSchedule(
        cron="0 9 15 * *",
        timezone="Europe/Paris"),
    flow_runner=SubprocessFlowRunner(),
    tags=["ml-train"],
)
