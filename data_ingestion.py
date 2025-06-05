import pandas as pd
import json
import os
from datetime import datetime
import random # For simulating API data if sample file is missing

# Define directory paths - These are relative to your project's root directory
# Make sure you run the script from the 'Customer_Churn_Prediction_Project' directory
RAW_DIR = 'data/raw'
STAGING_DIR = 'data/staging'
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__)) # Get script's dir
DATA_DIR = os.path.join(PROJECT_ROOT, 'data') # Assume 'data' is a sibling to the script

# Ensure the staging directory exists
os.makedirs(os.path.join(PROJECT_ROOT, STAGING_DIR), exist_ok=True)

def ingest_customer_csv(file_name='customers.csv'):
    """Ingests customer data from a CSV file into the staging area as Parquet."""
    file_path = os.path.join(PROJECT_ROOT, RAW_DIR, file_name)
    output_path = os.path.join(PROJECT_ROOT, STAGING_DIR, f'customers_csv_{datetime.now().strftime("%Y%m%d%H%M%S")}.parquet')
    print(f"Ingesting customer CSV from {file_path}...")
    try:
        df = pd.read_csv(file_path)
        df.to_parquet(output_path, index=False)
        print(f"Successfully ingested {file_name} to {output_path}")
    except FileNotFoundError:
        print(f"Error: Customer CSV file not found at {file_path}")
    except Exception as e:
        print(f"Error ingesting customer CSV: {e}")

def ingest_customer_json(file_name='customers.json'):
    """Ingests customer data from a JSON file into the staging area as Parquet."""
    file_path = os.path.join(PROJECT_ROOT, RAW_DIR, file_name)
    output_path = os.path.join(PROJECT_ROOT, STAGING_DIR, f'customers_json_{datetime.now().strftime("%Y%m%d%H%M%S")}.parquet')
    print(f"Ingesting customer JSON from {file_path}...")
    try:
        df = pd.read_json(file_path)
        df.to_parquet(output_path, index=False)
        print(f"Successfully ingested {file_name} to {output_path}")
    except FileNotFoundError:
        print(f"Error: Customer JSON file not found at {file_path}")
    except Exception as e:
        print(f"Error ingesting customer JSON: {e}")

def ingest_billing_csv(file_name='billing.csv'):
    """Ingests billing data from a CSV file into the staging area as Parquet."""
    file_path = os.path.join(PROJECT_ROOT, RAW_DIR, file_name)
    output_path = os.path.join(PROJECT_ROOT, STAGING_DIR, f'billing_{datetime.now().strftime("%Y%m%d%H%M%S")}.parquet')
    print(f"Ingesting billing CSV from {file_path}...")
    try:
        df = pd.read_csv(file_path)
        df.to_parquet(output_path, index=False)
        print(f"Successfully ingested {file_name} to {output_path}")
    except FileNotFoundError:
        print(f"Error: Billing CSV file not found at {file_path}")
    except Exception as e:
        print(f"Error ingesting billing CSV: {e}")

def ingest_usage_logs_json(file_name='usage_logs_v1.json'):
    """Ingests usage log data from a JSON file into the staging area as Parquet."""
    file_path = os.path.join(PROJECT_ROOT, RAW_DIR, file_name)
    output_path = os.path.join(PROJECT_ROOT, STAGING_DIR, f'usage_logs_{datetime.now().strftime("%Y%m%d%H%M%S")}.parquet')
    print(f"Ingesting usage logs from {file_path}...")
    try:
        with open(file_path, 'r') as f:
            data = json.load(f)
        df = pd.DataFrame(data)
        df.to_parquet(output_path, index=False)
        print(f"Successfully ingested {file_name} to {output_path}")
    except FileNotFoundError:
        print(f"Error: Usage logs JSON file not found at {file_path}")
    except Exception as e:
        print(f"Error ingesting usage logs: {e}")

def get_website_activity_from_api_simulated(file_name='sample_website_activity.json'):
    """
    Simulates fetching website activity data from an API.
    In a real scenario, this would make an actual HTTP request.
    For this project, it loads from a sample file or generates on the fly.
    """
    print("Simulating API call for website activity...")
    sample_file_path = os.path.join(PROJECT_ROOT, RAW_DIR, file_name)
    data = []

    if os.path.exists(sample_file_path):
        try:
            with open(sample_file_path, 'r') as f:
                data = json.load(f)
            print(f"  - Loaded sample API data from {sample_file_path}")
        except Exception as e:
            print(f"  - Error loading sample API data: {e}. Generating new data.")
            # Fallback to generating data if sample file is corrupt
            customer_ids = [f'C{i:04d}' for i in range(1000)]
            for _ in range(50):
                data.append({
                    'customer_id': random.choice(customer_ids),
                    'page_visited': random.choice(['home', 'pricing', 'support']),
                    'visit_timestamp': datetime.now().isoformat(),
                    'time_on_page_seconds': random.randint(10, 300)
                })
    else:
        print("  - Sample API data file not found. Generating new data.")
        customer_ids = [f'C{i:04d}' for i in range(1000)]
        for _ in range(50):
            data.append({
                'customer_id': random.choice(customer_ids),
                'page_visited': random.choice(['home', 'pricing', 'support']),
                'visit_timestamp': datetime.now().isoformat(),
                'time_on_page_seconds': random.randint(10, 300)
            })

    df = pd.DataFrame(data)
    output_path = os.path.join(PROJECT_ROOT, STAGING_DIR, f'website_activity_{datetime.now().strftime("%Y%m%d%H%M%S")}.parquet')
    df.to_parquet(output_path, index=False)
    print(f"Successfully ingested website activity data from simulated API to {output_path}")


if __name__ == "__main__":
    print("Starting data ingestion process directly...")
    try:
        # Call all your ingestion functions here
        ingest_customer_csv()
        ingest_customer_json()
        ingest_billing_csv()
        ingest_usage_logs_json()
        get_website_activity_from_api_simulated()

        print(f"Data ingestion completed. Please check the '{os.path.join(PROJECT_ROOT, STAGING_DIR)}' directory for new .parquet files.")
    except Exception as e:
        print(f"Error during data ingestion: {e}")
        import traceback
        traceback.print_exc()