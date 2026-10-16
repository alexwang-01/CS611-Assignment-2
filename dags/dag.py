from datetime import timedelta

import pendulum
from airflow import DAG
from airflow.operators.bash import BashOperator
from airflow.timetables.events import EventsTimetable


default_args = {
    'owner': 'airflow',
    'depends_on_past': False,
    'retries': 1,
    'retry_delay': timedelta(minutes=1),
}

# Application months included in this historical evaluation
snapshot_dates = [
    pendulum.datetime(2024, 7, 1, tz='UTC'),
    pendulum.datetime(2024, 8, 1, tz='UTC'),
    pendulum.datetime(2024, 9, 1, tz='UTC'),
    pendulum.datetime(2024, 10, 1, tz='UTC'),
    pendulum.datetime(2024, 11, 1, tz='UTC'),
    pendulum.datetime(2024, 12, 1, tz='UTC'),
    pendulum.datetime(2025, 1, 1, tz='UTC'),
]

with DAG(
    'loan_model_pipeline',
    default_args=default_args,
    description='Retrospective monthly evaluation of held-out loan applications',
    schedule=EventsTimetable(snapshot_dates, restrict_to_events=True,
                             description='Monthly test cohorts: July 2024-January 2025'),
    start_date=snapshot_dates[0],
    catchup=True,
    max_active_runs=1,
) as dag:
    model_training = BashOperator(
        task_id='model_training',
        bash_command='cd /app && python utils/model_training.py',
    )
    model_inference = BashOperator(
        task_id='model_inference',
        bash_command='cd /app && python utils/model_inference.py --snapshotdate "{{ data_interval_start | ds }}"',
    )
    model_monitoring = BashOperator(
        task_id='model_monitoring',
        bash_command='cd /app && python utils/model_monitoring.py --snapshotdate "{{ data_interval_start | ds }}"',
    )

    model_training >> model_inference >> model_monitoring
