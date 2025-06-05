import pandas as pd
import numpy as np
import os
import glob
import json
from datetime import datetime
import re # For cleaning column names

# --- GenAI Configuration ---
try:
    import google.generativeai as genai
    # Ensure GOOGLE_API_KEY is set in your environment variables for production
    # For local testing, you can uncomment and set it directly, but it's not recommended for production
    # os.environ["GOOGLE_API_KEY"] = "YOUR_GOOGLE_API_KEY" # REMOVE IN PRODUCTION
    genai.configure(api_key=os.environ.get("GOOGLE_API_KEY")) # This will try to get it from env var
    llm_model = genai.GenerativeModel('gemini-1.5-flash') # Or 'gemini-1.5-flash', 'gemini-1.5-pro' etc.
    print("GenAI model initialized successfully for Feature Engineering.")
except Exception as e:
    print(f"Warning: Could not initialize GenAI model for Feature Engineering. Error: {e}")
    llm_model = None # Set to None if initialization fails

# Define directory paths
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
PROCESSED_DIR = os.path.join(PROJECT_ROOT, 'data/processed')
WAREHOUSE_DIR = os.path.join(PROJECT_ROOT, 'data/warehouse') # Final ML-ready data goes here

os.makedirs(WAREHOUSE_DIR, exist_ok=True)

def llm_suggest_features(df_schema: dict, business_context: str) -> str:
    """Uses LLM to suggest new features based on DataFrame schema and business context."""
    if llm_model is None:
        return "No AI feature suggestions due to LLM not being configured."

    prompt = f"""
    You are a data scientist helping a telecom company predict customer churn.
    Analyze the following DataFrame schema, which represents unified and cleaned customer data.
    Based on this schema and the business context provided, suggest creative and impactful new features that could be engineered from the existing columns.
    For each suggested feature, provide:
    - **Feature Name**: A concise name for the new feature.
    - **Description**: Explain what the feature represents and why it might be important for churn prediction.
    - **Derivation Logic/Formula**: How to calculate this feature using existing columns (provide clear Python Pandas-like logic if possible).
    - **Expected Impact**: Briefly explain its potential relationship with churn (e.g., "Higher values indicate more engagement, likely decreasing churn risk").

    DataFrame Schema (column name and its inferred type):
    {json.dumps(df_schema, indent=2)}

    Business Context: {business_context}

    Consider features related to:
    - Customer tenure
    - Usage patterns (data, calls, SMS)
    - Billing behavior (payment consistency, charges changes)
    - Engagement levels (website visits, app usage)
    - Contract type implications
    - Recency, Frequency, Monetary (RFM) concepts if applicable.

    Please format the output clearly, using markdown for structure.
    """
    try:
        response = llm_model.generate_content(prompt)
        return response.text
    except Exception as e:
        return f"Error calling LLM for feature suggestions: {e}. Skipping AI suggestions."


def perform_feature_engineering():
    """
    Loads processed data, performs feature engineering, and saves the ML-ready dataset.
    Includes GenAI assistance for feature ideation.
    """
    print("Starting Feature Engineering...")

    # Load the latest unified data from the processed directory
    processed_files = glob.glob(os.path.join(PROCESSED_DIR, 'unified_customers_*.parquet'))
    if not processed_files:
        print(f"No unified customer parquet files found in {PROCESSED_DIR}. Please ensure cleaning and unification ran successfully.")
        return

    # Get the most recently created file (assuming timestamp in filename)
    latest_processed_file = max(processed_files, key=os.path.getctime)
    print(f"Loading latest processed data from: {latest_processed_file}")
    df = pd.read_parquet(latest_processed_file)

    # --- GenAI Feature Ideation ---
    df_schema = {col: str(df[col].dtype) for col in df.columns}
    business_context = "This data is a cleaned and unified view of telecom customer information (CRM, billing, usage, web activity). Our goal is to predict customer churn."
    feature_suggestions = llm_suggest_features(df_schema, business_context)
    print("\n--- GenAI Feature Ideation Suggestions ---")
    print(feature_suggestions)
    print("------------------------------------------\n")

    # --- Feature Engineering Implementations ---
    # These implementations are based on common churn prediction features and
    # potentially informed by the kind of suggestions an LLM might provide.

    # 1. Tenure (customer_lifetime)
    if 'signup_date' in df.columns:
        df['signup_date'] = pd.to_datetime(df['signup_date'], errors='coerce', utc=True)
        # Calculate tenure in months. Use current time in the local timezone (not UTC) for `datetime.now()`
        # then convert to UTC for consistent comparison with `signup_date` if it's UTC.
        # Or, just ensure both are timezone-naive or both are timezone-aware and match.
        # For simplicity and given previous UTC conversion:
        df['tenure_months'] = ((pd.to_datetime(datetime.now().isoformat()).tz_localize('UTC') - df['signup_date']).dt.days / 30.4375).fillna(0).astype(int)
        df['tenure_years'] = (df['tenure_months'] / 12).astype(int)
        print("  - Engineered 'tenure_months' and 'tenure_years'.")

    # 2. Ratio Features (e.g., total_charges per tenure)
    if 'total_charges' in df.columns and 'tenure_months' in df.columns:
        df['charges_per_month'] = df['total_charges'] / df['tenure_months'].replace(0, np.nan) # Avoid division by zero
        df['charges_per_month'].fillna(df['monthly_charges'], inplace=True) # Fill inf/NaN for new customers with monthly_charges
        print("  - Engineered 'charges_per_month'.")

    # 3. Usage features (average data/call/sms per month)
    # Assuming 'data_usage_gb', 'call_minutes', 'sms_count' are present from billing or usage logs
    for usage_col in ['data_usage_gb', 'call_minutes', 'sms_count']:
        if usage_col in df.columns and 'tenure_months' in df.columns:
            new_col_name = f'avg_{usage_col.replace("_gb", "").replace("_minutes", "").replace("_count", "")}_per_month'
            df[new_col_name] = df[usage_col] / df['tenure_months'].replace(0, np.nan)
            df[new_col_name].fillna(0, inplace=True) # Fill NaNs for new customers
            print(f"  - Engineered '{new_col_name}'.")

    # 4. Payment Consistency (e.g., number of overdue payments)
    # This would typically require multiple billing records per customer.
    # Since we merged, 'payment_status' might represent the latest status.
    # For simplicity, let's create a binary feature for 'Overdue' status (if applicable)
    if 'payment_status' in df.columns:
        df['is_payment_overdue'] = df['payment_status'].apply(lambda x: 1 if 'Overdue' in str(x) else 0)
        print("  - Engineered 'is_payment_overdue'.")

    # 5. Engagement Score (derived from web activity, if available)
    if 'time_on_page_seconds' in df.columns:
        # Assuming 'time_on_page_seconds' is the sum or max from a previous aggregation.
        # If it's not aggregated yet (i.e. if it's still per activity), this would be complex.
        # For simplicity, let's create a proxy assuming it's already semi-aggregated or just use it directly.
        # A more robust approach would be to aggregate web activity *before* merging.
        # Ensure 'customer_id' is set as index or used in groupby
        if 'customer_id' in df.columns:
            df['avg_time_on_page'] = df.groupby('customer_id')['time_on_page_seconds'].transform('mean')
            df['total_pages_visited'] = df.groupby('customer_id')['page_visited'].transform('count') # Simple count of entries
            df['engagement_score'] = df['avg_time_on_page'] * np.log1p(df['total_pages_visited']) # Combine with log transform
            df['engagement_score'].fillna(0, inplace=True) # Fill for customers with no web activity
            print("  - Engineered 'avg_time_on_page', 'total_pages_visited', and 'engagement_score'.")
        else:
            print("  - Skipping engagement score: 'customer_id' not found for grouping.")
    else:
        print("  - Skipping engagement score: 'time_on_page_seconds' not found.")

    # 6. Contract Type numerical representation (one-hot encoding or mapping)
    if 'contract_type' in df.columns:
        contract_mapping = {'Monthly': 1, 'Annual': 12, 'Two Year': 24}
        df['contract_duration_months'] = df['contract_type'].map(contract_mapping).fillna(1) # Default to 1 for unknown
        print("  - Engineered 'contract_duration_months'.")

    # 7. Binary features for common categorical columns (simple examples)
    # This is a basic form of one-hot encoding for specific columns
    if 'gender' in df.columns:
        df['is_male'] = (df['gender'] == 'Male').astype(int)
        print("  - Engineered 'is_male'.")
    if 'payment_method' in df.columns:
        df['uses_credit_card'] = (df['payment_method'] == 'Credit Card').astype(int)
        print("  - Engineered 'uses_credit_card'.")

    # 8. Clean up column names for ML compatibility (remove special characters, standardize)
    df.columns = [re.sub(r'[^a-zA-Z0-9_]', '', col).lower() for col in df.columns]
    print("  - Cleaned up column names.")

    # Select final features for the ML model (and target variable 'churn')
    # You'd refine this list based on EDA, feature importance, and model needs
    final_columns = [col for col in df.columns if col not in ['signup_date', 'bill_date', 'timestamp', 'visit_timestamp',
                                                              'gender', 'location', 'contract_type', 'service_plan',
                                                              'payment_method', 'payment_status', 'activity_type', 'page_visited',
                                                              'browser' # Original categorical columns that might have new numerical features
                                                              ]] # Exclude original raw columns if new features derived

    # Specifically handle columns with '_y_' suffix from merge, if they exist and are not desired
    final_columns = [col for col in final_columns if not col.endswith('_y_')]
    # Remove raw usage/charges if new rates are preferred
    final_columns = [col for col in final_columns if col not in ['data_usage_gb', 'call_minutes', 'sms_count',
                                                                 'total_charges', 'monthly_charges', 'amount_due',
                                                                 'data_volume_mb', 'duration_seconds', 'time_on_page_seconds']]


    # Ensure 'churn' is always in the final columns
    if 'churn' not in final_columns and 'churn' in df.columns:
        final_columns.append('churn')
    if 'customer_id' not in final_columns and 'customer_id' in df.columns:
        final_columns.append('customer_id') # Keep ID for traceability

    # Handle cases where some engineered features might not have been created due to missing raw data
    final_columns_present = [col for col in final_columns if col in df.columns]

    ml_ready_df = df[final_columns_present].copy()

    # Fill any remaining NaNs in numeric columns with 0 or median (post-engineering)
    for col in ml_ready_df.select_dtypes(include=np.number).columns:
        if ml_ready_df[col].isnull().any():
            ml_ready_df[col].fillna(0, inplace=True) # Or median: ml_ready_df[col].median()

    # Save the final ML-ready dataset to the warehouse directory
    output_path = os.path.join(WAREHOUSE_DIR, f'ml_ready_customers_{datetime.now().strftime("%Y%m%d%H%M%S")}.parquet')
    ml_ready_df.to_parquet(output_path, index=False)
    print(f"\nFeature Engineering complete. ML-ready data saved to {output_path}")


    # Save the final ML-ready dataset to the warehouse directory
    output_path = os.path.join(WAREHOUSE_DIR, f'ml_ready_customers_{datetime.now().strftime("%Y%m%d%H%M%S")}.csv')
    ml_ready_df.to_csv(output_path, index=False)
    print(f"\nFeature Engineering complete. ML-ready data saved to {output_path}")

    # You might also want to save the final feature list for the model training script
    feature_list_path = os.path.join(WAREHOUSE_DIR, 'ml_ready_features.json')
    features_only = [col for col in ml_ready_df.columns if col not in ['customer_id', 'churn']]
    with open(feature_list_path, 'w') as f:
        json.dump(features_only, f, indent=4)
    print(f"List of ML features saved to {feature_list_path}")


if __name__ == "__main__":
    print("Starting feature engineering process directly...")
    try:
        # Call the main function that performs all feature engineering
        perform_feature_engineering()
    except Exception as e:
        print(f"Error during feature engineering: {e}")
        import traceback
        traceback.print_exc()