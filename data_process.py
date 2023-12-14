import os
import warnings
from typing import Any, Optional

import mlflow
import pandas as pd
import yaml
from kaggle.api.kaggle_api_extended import KaggleApi
from loguru import logger
from sklearn.compose import make_column_transformer
from sklearn.metrics import mean_squared_error as mse
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsRegressor
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import (
    MinMaxScaler,
    OneHotEncoder,
    PolynomialFeatures,
    RobustScaler,
    StandardScaler,
)

warnings.simplefilter("ignore", UserWarning)

UNNAMED_COLUMN = "Unnamed: 0"
FILE_PATH = os.path.dirname(__file__)
DATA_PATH = os.path.join(FILE_PATH, "data")
CSV_PATH = os.path.join(DATA_PATH, "diamonds.csv")
PARAMS_PATH = os.path.join(FILE_PATH, "params.yaml")
PARAMS = dict(yaml.safe_load(open(PARAMS_PATH)))


def _handle_exception(e: Exception, message: Optional[str] = None) -> None:
    """Handles Exception, shows message if specified, raises Exception"""
    error_type = type(e).__name__
    logger.error(f"An error occurred: {error_type}. {e}")
    if message:
        logger.warning(message)
    raise type(e)


def _auth_kaggle() -> KaggleApi:
    """Authentication in KaggleApi"""
    try:
        api = KaggleApi()
        api.authenticate()
        logger.info("Kaggle authentication was succesfull")
    except Exception as e:
        _handle_exception(
            e,
            "You need to provide correct kaggle.json with authentication info",
        )
    return api


def _download_dataset(api: KaggleApi) -> None:
    """Download Diamonds.csv dataset"""
    logger.info("Start Diamonds dataset download")
    try:
        api.dataset_download_files(
            dataset="shivam2503/diamonds",
            path=DATA_PATH,
            unzip=True,
        )
        logger.success(
            f"Diamonds dataset was successfully downloaded into `{DATA_PATH}`"
        )
    except Exception as e:
        _handle_exception(
            e,
            "Something went wrong with kaggle dataset download",
        )


def _drop_na(df: pd.DataFrame) -> pd.DataFrame:
    """Gets df and retuns df without NaN row(s)"""
    nan_amount = df.isna().sum().sum()
    if nan_amount == 0:
        logger.success("There is no NaN to drop")
    else:
        logger.warning(f"There is/are {nan_amount} NaN(s), they will be dropped")
        df.dropna(inplace=True)
        logger.info(f"Result df shape: {df.shape}")
    return df


def _drop_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    """Gets df and retuns df without duplicates"""
    df_shape_before = df.shape
    logger.info(f"df shape before drop_duplicates(): {df_shape_before}")
    df.drop_duplicates(inplace=True, ignore_index=True)
    df_shape_after = df.shape
    logger.info(f"df shape after drop_duplicates(): {df_shape_after}")
    logger.success(f"Dropped {df_shape_before[0] - df_shape_after[0]} rows")
    return df


def _drop_outliers(df: pd.DataFrame, params: dict) -> pd.DataFrame:
    """Gets df and retuns df without outliers according to params-dict"""
    total_dropped = 0
    for feature, borders in params.items():
        _min = borders["min"]
        _max = borders["max"]
        logger.info(f"For feature `{feature}` drop outliers not in [{_min}, {_max}]")

        df_shape_before = df.shape
        logger.info(
            f"df shape before drop_outliers() from `{feature}`: {df_shape_before}"
        )

        question_index = df[(df[feature] < _min) | (df[feature] > _max)].index
        df = df.drop(question_index)
        df_shape_after = df.shape
        logger.info(
            f"df shape after drop_outliers() from `{feature}`: {df_shape_after}"
        )

        dropped = df_shape_before[0] - df_shape_after[0]
        total_dropped += dropped
        logger.success(f"Dropped {dropped} rows(s) for feature `{feature}`")
    logger.info(f"drop_outliers() dropped {total_dropped} row(s) in total")
    return df


def _get_typed_columns(df: pd.DataFrame) -> tuple:
    """Gets df, returns tuple with names of categorial and numerical columns"""
    cat_columns = []
    num_columns = []

    for column_name in df.columns:
        if (df[column_name].dtypes == object) or (df[column_name].dtypes == "category"):
            cat_columns += [column_name]
        else:
            num_columns += [column_name]

    logger.info(f"categorical columns: {cat_columns}, len = {len(cat_columns)}")
    logger.info(f"numerical columns: {num_columns}, len = {len(num_columns)}")

    return cat_columns, num_columns


def _preprocessors(df: pd.DataFrame, params: dict) -> pd.DataFrame:
    logger.info(f"Input DataFrame shape: {df.shape}")
    scalers = params["scalers"]
    if scalers == "minmax":
        scaler = MinMaxScaler()
    elif scalers == "standard":
        scaler = StandardScaler()
    else:
        scaler = RobustScaler()

    poly_features = PolynomialFeatures(
        degree=params["PF_degree"],
        include_bias=False,
        interaction_only=params["PF_interaction"],
    )

    num_pipe = make_pipeline(poly_features, scaler)

    cat_pipe = make_pipeline(
        OneHotEncoder(
            drop="if_binary",
            handle_unknown="ignore",
            sparse_output=False,
        )
    )
    column_name = params["target"]
    X = df.drop(columns=column_name)
    y = df[[column_name]]
    cat_columns, num_columns = _get_typed_columns(X)

    preprocessors = make_column_transformer(
        (num_pipe, num_columns),
        (cat_pipe, cat_columns),
    )
    logger.success("Preprocessors are ready")

    preprocessors.fit(X)
    logger.info("Preprocessors are fitted")

    X_preprocessors = preprocessors.transform(X)
    logger.success("DataFrame is transormed")

    df_preprocessors = pd.DataFrame(
        data=X_preprocessors,
        columns=preprocessors.get_feature_names_out(),
    )
    result_df = pd.concat([df_preprocessors, y], axis=1).dropna()
    logger.info(f"Result DataFrame shape: {result_df.shape}")
    return result_df


def _tt_split_df(
    df: pd.DataFrame,
    params: dict,
) -> tuple:
    """Splits input df into train test"""
    target_column = params["target"]
    logger.info(f"Start split_df() function with target column `{target_column}`")
    X: pd.DataFrame = df.drop(columns=[target_column])
    y: pd.DataFrame = df[[target_column]]

    split_ratio = params["split_ratio"]
    random_seed = params["seed"]
    logger.info(f"Split ratio = {split_ratio}")
    logger.info(f"Random seed = {random_seed}")

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=split_ratio,
        random_state=random_seed,
    )
    logger.info(f"Length of train df = {len(X_train)}")
    logger.info(f"Length of test df = {len(X_test)}")
    logger.success("Dataset was splitted")
    return X_train, X_test, y_train, y_test


def _prepare_knr(
    x_train: pd.DataFrame,
    y_train: pd.DataFrame,
    params: dict,
) -> Any:
    """Gets DataFrame and params and prepares KNeighborsRegressor with specified params"""
    model = KNeighborsRegressor(
        n_neighbors=params["n_neighbors"],
        weights=params["weights"],
        n_jobs=-1,
    )
    logger.success(f"Model {type(model).__name__} was initialized")
    logger.info("Model's parameters:")
    logger.info(f"n_neighbors = {params['n_neighbors']}")
    logger.info(f"weights = {params['weights']}")
    model.fit(x_train, y_train)
    logger.success(f"Model {type(model).__name__} was fitted")
    return model


def _get_extended_scores(y_test: list, y_prediction: list) -> dict:
    """Counts extended scores and returns dicts with scores"""
    MSE = round(mse(y_test, y_prediction), 2)
    RMSE = round(mse(y_test, y_prediction, squared=False), 2)
    R2 = r2_score(y_test, y_prediction)
    logger.info("Test data extended scores:")
    logger.info(f"MSE: {MSE}")
    logger.info(f"RMSE: {RMSE}")
    logger.info(f"R2: {R2}")
    result = {
        "MSE": MSE,
        "RMSE": RMSE,
        "R2": R2,
    }
    return result


# -------------------------------------------------------------------


def read_data(**kwargs):
    logger.debug("Data reading was started")
    _download_dataset(_auth_kaggle())
    df = pd.read_csv(CSV_PATH)
    if UNNAMED_COLUMN in df.columns:
        df.drop(columns=UNNAMED_COLUMN, inplace=True)
        logger.debug(f"Column `{UNNAMED_COLUMN}` was dropped")
    return df


def preprocess_data(**kwargs):
    logger.debug("Data preprocessing was started")
    try:
        kwargs["ti"]
        ti = kwargs["ti"]
        df = ti.xcom_pull(task_ids="read_data")
        logger.debug("XFlow")
    except KeyError:
        df = kwargs["df"]
        logger.debug("Usual use")
    df = _drop_duplicates(df)
    df = _drop_na(df)
    df = _drop_outliers(df, PARAMS["drop_outliers"])
    df = _preprocessors(df, PARAMS["preprocessors"])
    result = _tt_split_df(df, PARAMS["split"])
    return result


def prepare_model(**kwargs):
    logger.debug("Model preparation was started")
    try:
        kwargs["ti"]
        ti = kwargs["ti"]
        X_train, _, y_train, _ = ti.xcom_pull(task_ids="preprocess_data")
        type_of_use = "XFlow"
    except KeyError:
        X_train, _, y_train, _ = kwargs["data"]
        type_of_use = "Usual use"
    logger.debug(type_of_use)
    model = _prepare_knr(X_train, y_train, PARAMS["train"])

    if type_of_use == "XFlow":
        mlflow.set_tracking_uri("http://mlflow_server:5000")
        try:
            mlflow.create_experiment("demo_data_process_flow")
        except Exception as e:
            logger.info(f"Got exception when mlflow.create_experiment: {e}")
        experiment = mlflow.set_experiment("demo_data_process_flow")
        with mlflow.start_run(experiment_id=experiment.experiment_id) as run:
            mlflow.log_params(PARAMS["train"])
            result = mlflow.sklearn.log_model(
                sk_model=model,
                artifact_path="models",
                registered_model_name="KNeighborsRegressor",
            )
            return result.model_uri
    else:
        return model


def evaluate_model(**kwargs):
    logger.debug("Model evaluation was started")
    try:
        kwargs["ti"]
        ti = kwargs["ti"]
        _, X_test, _, y_test = ti.xcom_pull(task_ids="preprocess_data")
        model_uri = ti.xcom_pull(task_ids="prepare_model")
        logger.info(f"model_uri: {model_uri}")
        type_of_use = "XFlow"
    except KeyError:
        _, X_test, _, y_test = kwargs["data"]
        model: KNeighborsRegressor = kwargs["model"]
        type_of_use = "Usual use"
    logger.debug(type_of_use)
    y_prediction = model.predict(X_test)
    extended_scores = _get_extended_scores(y_test, y_prediction)

    if type_of_use == "XFlow":
        mlflow.set_tracking_uri("http://mlflow_server:5000")
        model = mlflow.pyfunc.load_model(model_uri=model_uri)
        experiment = mlflow.set_experiment("demo_data_process_flow_check_model")
        with mlflow.start_run(experiment_id=experiment.experiment_id) as run:
            mlflow.log_metrics(extended_scores)


if __name__ == "__main__":
    df = read_data()
    preprocessed_data = preprocess_data(df=df)
    model = prepare_model(data=preprocessed_data)
    evaluate_model(model=model, data=preprocessed_data)
