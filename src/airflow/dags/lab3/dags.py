import pendulum
from airflow import DAG
from airflow.operators.python import PythonOperator

from lab3.data_process import *

default_args = dict(
    start_date=pendulum.datetime(2023, 12, 15, tz="UTC"),
    catchup=False,
    is_paused_upon_creation=True,
    provide_context=True,
)

dag = DAG(
    "dataflow_process_dag",
    default_args=default_args,
    schedule="0 2 * * *",
    max_active_runs=1,
)

read_data = PythonOperator(
    task_id="read_data",
    python_callable=read_data,
    dag=dag,
)
preprocess_data = PythonOperator(
    task_id="preprocess_data",
    python_callable=preprocess_data,
    dag=dag,
)
prepare_model = PythonOperator(
    task_id="prepare_model",
    python_callable=prepare_model,
    dag=dag,
)
evaluate_model = PythonOperator(
    task_id="evaluate_model",
    python_callable=evaluate_model,
    dag=dag,
)

read_data.set_downstream(preprocess_data)
preprocess_data.set_downstream(prepare_model)
prepare_model.set_downstream(evaluate_model)
