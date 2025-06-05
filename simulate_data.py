import pandas as pd
import numpy as np
import random
from datetime import datetime, timedelta
import json
import os
from faker import Faker

fake = Faker()

def generate_customer_data(num_customers=1000):
    data = {
        'customer_id': [f'C{i:04d}' for i in range(num_customers)],
        'age': [random.randint(18, 70) if random.random() > 0.05 else fake.word() for _ in range(num_customers)], # Introduce some non-numeric ages
        'gender': random.choices(['Male', 'Female', 'Other'], k=num_customers),
        'location': random.choices(['New York', 'Los Angeles', 'Chicago', 'Houston', 'Miami'], k=num_customers),
        'signup_date': [(datetime.now() - timedelta(days=random.randint(30, 1000))).strftime('%Y-%m-%d') for _ in range(num_customers)],
        'contract_type': random.choices(['Monthly', 'Annual', 'Two Year'], k=num_customers),
        'service_plan': random.choices(['Basic', 'Standard', 'Premium'], k=num_customers),
        'monthly_charges': [round(random.uniform(20, 150), 2) for _ in range(num_customers)],
        'total_charges': [round(random.uniform(50, 5000), 2) for _ in range(num_customers)],
        'payment_method': random.choices(['Credit Card', 'Bank Transfer', 'E-Wallet', 'Mail'], k=num_customers),
        'churn': [random.choice([0, 1]) for _ in range(num_customers)] # Target variable for ML
    }
    df = pd.DataFrame(data)
    # Introduce some duplicate customer IDs for cleaning later
    df = pd.concat([df, df.sample(n=int(num_customers * 0.02), replace=True)], ignore_index=True)
    return df

def generate_billing_data(num_records=5000):
    customer_ids = [f'C{i:04d}' for i in range(1000)] # Assuming 1000 unique customers
    data = {
        'customer_id': random.choices(customer_ids, k=num_records),
        'bill_date': [(datetime.now() - timedelta(days=random.randint(1, 365))).strftime('%Y-%m-%d') for _ in range(num_records)],
        'amount_due': [round(random.uniform(10, 200), 2) for _ in range(num_records)],
        'payment_status': random.choices(['Paid', 'Due', 'Overdue'], k=num_records),
        'data_usage_gb': [round(random.uniform(1, 100), 2) for _ in range(num_records)],
        'call_minutes': [random.randint(0, 500) for _ in range(num_records)],
        'sms_count': [random.randint(0, 200) for _ in range(num_records)],
        'promo_applied': [random.choice([True, False]) for _ in range(num_records)]
    }
    return pd.DataFrame(data)

def generate_usage_logs_v1(num_records=10000):
    customer_ids = [f'C{i:04d}' for i in range(1000)]
    logs = []
    for _ in range(num_records):
        log = {
            'customer_id': random.choice(customer_ids),
            'timestamp': (datetime.now() - timedelta(minutes=random.randint(1, 1440))).isoformat() + 'Z',
            'activity_type': random.choice(['data_usage', 'call', 'sms_sent', 'app_login', 'video_stream']),
            'data_volume_mb': random.randint(1, 100) if random.random() < 0.7 else None,
            'duration_seconds': random.randint(10, 600) if random.random() < 0.3 else None
        }
        logs.append(log)
    return logs

def generate_website_activity_data(num_records=2000):
    customer_ids = [f'C{i:04d}' for i in range(1000)]
    activity = []
    for _ in range(num_records):
        act = {
            'customer_id': random.choice(customer_ids),
            'page_visited': random.choice(['homepage', 'support', 'billing', 'plans', 'contact_us', 'faq', 'forum']),
            'visit_timestamp': (datetime.now() - timedelta(minutes=random.randint(1, 60))).isoformat(),
            'time_on_page_seconds': random.randint(5, 300),
            'browser': random.choice(['Chrome', 'Firefox', 'Safari', 'Edge'])
        }
        activity.append(act)
    return activity


if __name__ == "__main__":
    RAW_DIR = 'data/raw'
    os.makedirs(RAW_DIR, exist_ok=True)

    print("Generating simulated customer data...")
    # Generate and save Customer Data (CSV and JSON for variety)
    customer_df = generate_customer_data()
    customer_df.to_csv(f'{RAW_DIR}/customers.csv', index=False)
    customer_df.to_json(f'{RAW_DIR}/customers.json', orient='records', indent=4)
    print("  - customers.csv and customers.json created.")

    print("Generating simulated billing data...")
    # Generate and save Billing Data
    billing_df = generate_billing_data()
    billing_df.to_csv(f'{RAW_DIR}/billing.csv', index=False)
    print("  - billing.csv created.")

    print("Generating simulated usage logs...")
    # Generate and save Usage Logs
    usage_logs_v1 = generate_usage_logs_v1()
    with open(f'{RAW_DIR}/usage_logs_v1.json', 'w') as f:
        json.dump(usage_logs_v1, f, indent=4)
    print("  - usage_logs_v1.json created.")

    print("Generating simulated website activity data (sample for API)..")
    # The website activity will be simulated via an API call within the Airflow DAG or a specific Python task.
    # For now, let's save a sample of what the API would return.
    sample_website_activity = generate_website_activity_data(num_records=50)
    with open(f'{RAW_DIR}/sample_website_activity.json', 'w') as f:
        json.dump(sample_website_activity, f, indent=4)
    print("  - sample_website_activity.json (representing API data) created.")

    print("\nSimulated data generation complete. Check the 'data/raw' directory.")