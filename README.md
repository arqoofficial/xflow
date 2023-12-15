# xflow

Diamonds' price prediction.

Track data processing (**diamonds.csv**) / model evaluation (**KNeighborsRegressor**) using **AirFlow** / **MLFlow**.

Data source: [Kaggle-API](https://www.kaggle.com/datasets/shivam2503/diamonds)

## Instalation

* To get raw diamonds.csv you need to provide your Kaggle-token (follow [Kaggle Public API instructions](https://www.kaggle.com/docs/api)). 

```
conda create "python>=3.8,<3.11" -n xflow
```
```
conda activate xflow
```
```
python -m pip install -r requirements.txt
```

* Install MLFlow / AirFlow

## Run

### AirFlow

```
airflow webserver -p 8080
```

```
airflow scheduler
```

### MLFlow

```
mlflow server --backend-store-uri postgresql://mlflow:mlflow@localhost/mlflow --default-artifact-root file:/home/‹user›/mlruns -h 0.0.0.0 -p 5000
```

### Start tracking

```
python ./src/airflow/dags/lab3/dags.py
```
## Results

### AirFlow

![ScreenShot](./img/airflow.png)

### MLFlow

![ScreenShot](./img/mlflow.png)