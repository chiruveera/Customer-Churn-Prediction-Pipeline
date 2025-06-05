from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.utils.dates import days_ago
from datetime import timedelta
import sys
import os

# Add the project root directory to the Python path
# This allows Airflow to find your data_ingestion.py, data_cleaning_and_unification.py, etc.
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Import your pipeline functions
from data_ingestion import ingest_customer_csv, ingest_customer_json, ingest_billing_csv, ingest_usage_logs_json, get_website_activity_from_api_simulated
from data_cleaning_and_unification import unified_data_cleaning_and_initial_processing
from feature_engineering import perform_feature_engineering


with DAG(
    dag_id='customer_churn_prediction_pipeline',
    default_args={
        'owner': 'airflow',
        'depends_on_past': False,
        'email_on_failure': False,
        'email_on_retry': False,
        'retries': 1,
        'retry_delay': timedelta(minutes=5),
    },
    description='An intelligent data pipeline for customer churn prediction using GenAI.',
    start_date=days_ago(1),
    schedule_interval=timedelta(days=7), # Run weekly, adjust as needed
    catchup=False,
    tags=['churn_prediction', 'data_pipeline', 'genai', 'telecom'],
) as dag:
    # Task 1: Ingest Customer Data (CSV)
    ingest_customer_csv_task = PythonOperator(
        task_id='ingest_customer_csv',
        python_callable=ingest_customer_csv,
    )

    # Task 2: Ingest Customer Data (JSON)
    ingest_customer_json_task = PythonOperator(
        task_id='ingest_customer_json',
        python_callable=ingest_customer_json,
    )

    # Task 3: Ingest Billing Data (CSV)
    ingest_billing_csv_task = PythonOperator(
        task_id='ingest_billing_csv',
        python_callable=ingest_billing_csv,
    )

    # Task 4: Ingest Usage Logs (JSON)
    ingest_usage_logs_json_task = PythonOperator(
        task_id='ingest_usage_logs_json',
        python_callable=ingest_usage_logs_json,
    )

    # Task 5: Ingest Website Activity (simulated API)
    ingest_website_activity_task = PythonOperator(
        task_id='ingest_website_activity',
        python_callable=get_website_activity_from_api_simulated,
    )

    # Task 6: Initial Data Cleaning and Unification
    # This task depends on all ingestion tasks completing successfully
    clean_and_unify_data_task = PythonOperator(
        task_id='clean_and_unify_data',
        python_callable=unified_data_cleaning_and_initial_processing,
    )

    # Task 7: Feature Engineering
    # This task depends on the cleaning and unification task completing successfully
    feature_engineering_task = PythonOperator(
        task_id='feature_engineering',
        python_callable=perform_feature_engineering,
    )

    # Define task dependencies
    # All ingestion tasks can run in parallelcd 
    [ingest_customer_csv_task, ingest_customer_json_task, ingest_billing_csv_task,
     ingest_usage_logs_json_task, ingest_website_activity_task] >> clean_and_unify_data_task

    clean_and_unify_data_task >> feature_engineering_task