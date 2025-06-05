import pandas as pd
import os
import glob
import json
import numpy as np # For numeric operations like median
from datetime import datetime
import re # For cleaning column names (though mostly used in feature_engineering, good to have)

# --- GenAI Configuration ---
try:
    import google.generativeai as genai
    # Configure your GenAI API key (from environment variable or directly here for testing)
    # Ensure GOOGLE_API_KEY is set in your environment variables for production
    # For local testing, you can uncomment and set it directly, but it's not recommended for production
    # os.environ["GOOGLE_API_KEY"] = "YOUR_GOOGLE_API_KEY" # REMOVE IN PRODUCTION
    genai.configure(api_key=os.environ.get("GOOGLE_API_KEY")) # This will try to get it from env var
    llm_model = genai.GenerativeModel('gemini-1.5-flash') # Or 'gemini-1.5-flash', 'gemini-1.5-pro' etc.
    print("GenAI model initialized successfully for Data Cleaning & Unification.")
except Exception as e:
    print(f"Warning: Could not initialize GenAI model for Cleaning. Ensure 'google-generativeai' is installed and GOOGLE_API_KEY is set. Error: {e}")
    llm_model = None # Set to None if initialization fails

# Define directory paths
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
STAGING_DIR = os.path.join(PROJECT_ROOT, 'data/staging')
PROCESSED_DIR = os.path.join(PROJECT_ROOT, 'data/processed') # Where cleaned, unified data goes

os.makedirs(PROCESSED_DIR, exist_ok=True)


def generate_data_profile(df: pd.DataFrame, df_name: str) -> str:
    """Generates a text profile of a DataFrame for LLM analysis."""
    profile = f"Data Profile for {df_name}:\n"
    profile += f"Shape: {df.shape}\n"
    profile += "Columns and Data Types:\n"
    for col in df.columns:
        profile += f"- {col}: {df[col].dtype}\n"
    profile += "Missing Values (Column %):\n"
    for col in df.columns:
        missing_percent = df[col].isnull().sum() / len(df) * 100
        if missing_percent > 0:
            profile += f"- {col}: {missing_percent:.2f}%\n"
    profile += "Unique Values (first 5 for object/categorical, min/max for numeric, sample of text):\n"
    for col in df.columns:
        if df[col].dtype == 'object' or df[col].nunique() < 20: # For categorical or low cardinality
            sample_values = df[col].dropna().unique()
            profile += f"- {col}: {sample_values[:5].tolist()}...\n"
        elif pd.api.types.is_numeric_dtype(df[col]):
            profile += f"- {col}: Min={df[col].min()}, Max={df[col].max()}\n"
    profile += "Sample of problematic data (e.g., non-numeric in numeric column, inconsistent date formats):\n"
    # Example: finding non-numeric 'age'
    if 'age' in df.columns:
        non_numeric_ages = df[pd.to_numeric(df['age'], errors='coerce').isna() & df['age'].notna()]['age'].unique()
        if len(non_numeric_ages) > 0:
            profile += f"- age (non-numeric): {non_numeric_ages[:5].tolist()}...\n"
    if 'signup_date' in df.columns:
        # Check for non-standard date formats (simple check)
        sample_dates = df['signup_date'].dropna().astype(str).sample(min(5, len(df))).tolist()
        profile += f"- signup_date (sample formats): {sample_dates}...\n"
    return profile

def llm_suggest_cleaning_rules(profile_text: str) -> str:
    """Uses LLM to suggest data cleaning rules based on a data profile."""
    if llm_model is None:
        return "No AI cleaning suggestions due to LLM not being configured."

    prompt = f"""
    Analyze the following data profile for a telecom customer dataset.
    Identify potential data quality issues (e.g., incorrect data types, missing values, inconsistent formats, outliers, duplicates).
    Based on the issues, suggest specific, actionable data cleaning rules and provide Python Pandas code snippets to address them.
    Focus on steps like:
    - Handling missing values (e.g., imputation, dropping).
    - Correcting data types (e.g., converting to numeric, datetime).
    - Standardizing formats (e.g., text capitalization, date formats).
    - Identifying and removing duplicate records.
    - Basic outlier identification for numerical columns.

    Data Profile:
    {profile_text}

    Output should be structured clearly, with markdown code blocks for Python snippets.
    **Identified Issues:**
    - Issue 1: Description
    - Issue 2: Description
    ...

    **Suggested Cleaning Rules & Code:**
    1. Rule Description:
    ```python
    # code snippet
    ```
    2. Rule Description:
    ```python
    # code snippet
    ```
    ...
    """
    try:
        response = llm_model.generate_content(prompt)
        return response.text
    except Exception as e:
        return f"Error calling LLM for cleaning suggestions: {e}. Skipping AI suggestions."

def llm_generate_data_dictionary(df_schema: dict, business_context: str) -> str:
    """Uses LLM to generate a data dictionary based on DataFrame schema."""
    if llm_model is None:
        return "No AI data dictionary generated due to LLM not being configured."

    prompt = f"""
    Generate a comprehensive data dictionary and metadata for the following Pandas DataFrame schema.
    This dataset contains customer information for a telecom company, being prepared for churn prediction.

    Include for each column:
    - **Column Name**
    - **Description**: A clear explanation of the column's content in a telecom context.
    - **Data Type**: The inferred Pandas data type (e.g., 'object', 'int64', 'float64', 'datetime64[ns]').
    - **Source**: Where the data likely originated (e.g., CRM, Billing System, Usage Logs, Web Activity).
    - **Example Values**: A few representative, diverse examples.
    - **Potential Use Cases**: How this column might be used in ML models or business analysis.
    - **Initial Cleaning/Transformation Notes**: What initial steps (like type conversion, handling NaNs) might have been applied.

    DataFrame Schema (column name and its inferred type):
    {json.dumps(df_schema, indent=2)}

    Business Context: {business_context}

    Please format the output clearly, possibly using markdown tables or bullet points for readability.
    """
    try:
        response = llm_model.generate_content(prompt)
        return response.text
    except Exception as e:
        return f"Error calling LLM for data dictionary: {e}. Skipping AI documentation."


def unified_data_cleaning_and_initial_processing():
    """
    Performs initial data cleaning, unification, and profiles data with GenAI assistance.
    Saves the cleaned, unified data to the processed directory.
    """
    all_data_frames = []

    # 1. Load all ingested Parquet files from staging
    parquet_files = glob.glob(os.path.join(STAGING_DIR, '*.parquet'))
    if not parquet_files:
        print(f"No parquet files found in {STAGING_DIR}. Please ensure ingestion ran successfully.")
        return

    print(f"Found {len(parquet_files)} parquet files in staging. Starting initial cleaning and unification.")

    for file_path in parquet_files:
        df_name = os.path.basename(file_path).split('_')[0] # e.g., 'customers', 'billing'
        print(f"\n--- Processing {file_path} ({df_name}) ---")
        try:
            df = pd.read_parquet(file_path)

            # GenAI Data Profiling and Suggestion
            profile_text = generate_data_profile(df.copy(), df_name)
            cleaning_suggestions = llm_suggest_cleaning_rules(profile_text)
            print(f"\n--- GenAI Cleaning Suggestions for {df_name} ---")
            print(cleaning_suggestions)
            print("--------------------------------------------------\n")

            # Apply initial hardcoded cleaning rules (you'd refine these based on LLM suggestions)
            # These are common initial steps, which the LLM helps identify.

            # Handle customer_id consistency
            if 'customer_id' in df.columns:
                df['customer_id'] = df['customer_id'].astype(str).str.strip()
            if 'user_id' in df.columns and 'customer_id' not in df.columns:
                df.rename(columns={'user_id': 'customer_id'}, inplace=True)
            if 'customer_id' not in df.columns:
                print(f"Warning: '{df_name}' does not have a 'customer_id' or 'user_id' column. Skipping for unification.")
                continue # Skip this dataframe if no customer_id

            # Common cleaning for customer_id (remove duplicates if any are created by ingestion)
            if 'customer_id' in df.columns:
                # Keep the last record in case of duplicate IDs within a single source file
                df.drop_duplicates(subset=['customer_id'], keep='last', inplace=True)


            # Example: Cleaning 'age' column
            if 'age' in df.columns:
                initial_age_dtype = df['age'].dtype
                df['age'] = pd.to_numeric(df['age'], errors='coerce') # Convert to numeric, errors become NaN
                if df['age'].isnull().any():
                    print(f"  - Imputing missing/invalid 'age' values with median ({df['age'].median()}).")
                    df['age'].fillna(df['age'].median(), inplace=True)
                df['age'] = df['age'].astype(int, errors='ignore') # Convert to int if possible
                if df['age'].dtype != initial_age_dtype:
                    print(f"  - 'age' column converted from {initial_age_dtype} to {df['age'].dtype}.")


            # Example: Standardize 'signup_date' or 'bill_date'
            for date_col in ['signup_date', 'bill_date', 'timestamp', 'visit_timestamp']:
                if date_col in df.columns:
                    original_dtype = df[date_col].dtype
                    # Convert to datetime, coercing errors to NaT (Not a Time)
                    df[date_col] = pd.to_datetime(df[date_col], errors='coerce', utc=True) # Use UTC for consistency
                    if df[date_col].isnull().any():
                        missing_count = df[date_col].isnull().sum()
                        print(f"  - Dropped {missing_count} rows due to unparseable '{date_col}' values.")
                        df.dropna(subset=[date_col], inplace=True) # Drop rows where date couldn't be parsed
                    if original_dtype != df[date_col].dtype:
                        print(f"  - '{date_col}' column converted from {original_dtype} to {df[date_col].dtype}.")

            # Example: Standardize string columns (e.g., 'gender', 'location', 'contract_type')
            for col in ['gender', 'location', 'contract_type', 'service_plan', 'payment_method', 'payment_status', 'activity_type', 'page_visited', 'browser']:
                if col in df.columns and df[col].dtype == 'object':
                    df[col] = df[col].astype(str).str.strip().str.title() # Strip whitespace, title case
                    print(f"  - Standardized '{col}' to title case.")


            # Ensure numeric types for financial/usage columns
            for col in ['monthly_charges', 'total_charges', 'amount_due', 'data_usage_gb', 'call_minutes', 'sms_count', 'data_volume_mb', 'duration_seconds', 'time_on_page_seconds']:
                if col in df.columns:
                    initial_dtype = df[col].dtype
                    df[col] = pd.to_numeric(df[col], errors='coerce')
                    if df[col].isnull().any():
                        print(f"  - Imputing missing '{col}' values with 0.")
                        df[col].fillna(0, inplace=True)
                    if initial_dtype != df[col].dtype:
                        print(f"  - '{col}' converted from {initial_dtype} to {df[col].dtype}.")

            all_data_frames.append(df)

        except Exception as e:
            print(f"Failed to process {file_path}: {e}")
            # Decide whether to halt pipeline or continue. For now, we continue.

    # 2. Unify data into a single DataFrame (merge on customer_id)
    if not all_data_frames:
        print("No valid data frames to unify.")
        return

    # Assuming 'customers_csv' or 'customers_json' is the primary source of customer IDs
    # Find the dataframe that is most likely the 'master' customer list
    master_df = None
    for df_item in all_data_frames: # Renamed 'df' to 'df_item' to avoid conflict
        # Check if 'customer_id' is in the dataframe AND if its original file name implies it's a customer source
        # This part requires linking df_item back to its original file name, which is not trivial with current loop
        # A simpler approach: assume the first df with customer_id is the master, or try to merge all
        if 'customer_id' in df_item.columns:
            # A more robust check might be needed if you have multiple customer sources and need a specific one as master
            master_df = df_item.copy() # Make a copy to avoid modifying original in list
            break # Found a candidate master, break

    if master_df is None:
        print("Could not identify a master customer dataframe. Attempting to concatenate and then dedup.")
        # Fallback if no clear 'customer' source is found; might lead to more NaNs
        unified_df = pd.concat(all_data_frames, ignore_index=True, sort=False)
        if 'customer_id' in unified_df.columns:
            unified_df.drop_duplicates(subset=['customer_id'], keep='last', inplace=True)
            print(f"Concatenated and deduplicated on customer_id. Final shape: {unified_df.shape}")
        else:
            print("No 'customer_id' found even after concatenation. Unification may be problematic.")
            unified_df = unified_df # Just use the concatenated data as is
    else:
        print(f"Identified a master customer dataframe (shape: {master_df.shape}). Merging other data sources.")
        unified_df = master_df.copy()

        # Merge other dataframes onto the master using 'customer_id'
        for df_other in all_data_frames:
            # Ensure we don't merge the master with itself, and it has customer_id
            if not df_other.equals(master_df) and 'customer_id' in df_other.columns:
                # Merge based on customer_id, using a left join to keep all master customers
                # Suffixes help if columns with same name exist in both DFs
                unified_df = pd.merge(unified_df, df_other, on='customer_id', how='left', suffixes=('', '_y_'))
                # Drop duplicate columns that have been merged (e.g., _y_ suffix from merge)
                # This drops columns that ended with _y_ because they were duplicates
                unified_df = unified_df.loc[:,~unified_df.columns.duplicated()].copy()
                # Clean up columns that were directly duplicated without a suffix
                # This handles cases where merge didn't add a suffix (e.g., if columns were exactly the same)
                cols_to_drop_after_merge = [col for col in unified_df.columns if col.endswith('_y_')]
                if cols_to_drop_after_merge:
                    unified_df.drop(columns=cols_to_drop_after_merge, inplace=True)
                print(f"  - Merged a dataframe with customer_id. Unified shape: {unified_df.shape}")

    # Final cleanup of the unified DataFrame
    # Drop rows where customer_id might be NaN (shouldn't happen if primary source was good)
    if 'customer_id' in unified_df.columns:
        unified_df.dropna(subset=['customer_id'], inplace=True)
        print(f"Dropped rows with missing customer_id. Final shape: {unified_df.shape}")

    # Drop any remaining duplicate customer IDs after all merges (e.g., if different source had same customer with slightly different details)
    # Keep the last one for recency or apply specific business logic
    if 'customer_id' in unified_df.columns:
        initial_unified_rows = unified_df.shape[0]
        unified_df.drop_duplicates(subset=['customer_id'], keep='last', inplace=True)
        print(f"Removed {initial_unified_rows - unified_df.shape[0]} duplicate customer IDs from unified data.")

    # Client Request: Add 'Planning_to_leave' column based on 'churn'
    # 'churn' column contains [0, 1] and 'Planning_to_leave' should contain ['Yes', 'No']
    if 'churn' in unified_df.columns:
        print("\nAdding 'Planning_to_leave' column based on 'churn' status (client request).")
        # Ensure 'churn' is numeric (0 or 1) before applying logic
        unified_df['churn'] = pd.to_numeric(unified_df['churn'], errors='coerce').fillna(-1).astype(int) # Handle potential non-numeric
        unified_df['planning_to_leave'] = unified_df['churn'].apply(lambda x: 'Yes' if x == 0 else 'No')
        # Display sample for verification, ensuring column names match your df
        print(unified_df[['customer_id', 'churn', 'planning_to_leave']].sample(min(5, len(unified_df))))
    else:
        print("Warning: 'churn' column not found in DataFrame. Skipping 'planning_to_leave' column creation.")

    # Save the final cleaned and unified dataset to the processed directory
    output_path = os.path.join(PROCESSED_DIR, f'unified_customers_{datetime.now().strftime("%Y%m%d%H%M%S")}.parquet')
    unified_df.to_parquet(output_path, index=False)
    print(f"Initial cleaning and unification complete. Saved to {output_path}")

    # --- Save to CSV format ---
    csv_output_filename = f'unified_customers_{datetime.now().strftime("%Y%m%d%H%M%S")}.csv'
    csv_output_path = os.path.join(PROCESSED_DIR, csv_output_filename)
    unified_df.to_csv(csv_output_path, index=False)
    print(f"Initial cleaning and unification complete. Also saved to {csv_output_path}")

    # --- GenAI Data Dictionary & Metadata Generation for the final unified dataset ---
    if llm_model:
        df_schema = {col: str(unified_df[col].dtype) for col in unified_df.columns}
        business_context = "This dataset contains unified customer information from CRM, billing, usage logs, and website activity, after initial cleaning and standardization. It will be used for further feature engineering towards churn prediction."
        data_dict_response = llm_generate_data_dictionary(df_schema, business_context)
        print("\n--- GenAI Generated Data Dictionary for Unified Data ---")
        print(data_dict_response)
        # You can save this to a file for documentation
        data_dict_file_path = os.path.join(PROCESSED_DIR, 'unified_data_dictionary.md')
        with open(data_dict_file_path, 'w') as f:
            f.write(data_dict_response)
        print(f"Generated data dictionary saved to {data_dict_file_path}")
    else:
        print("\nSkipping GenAI Data Dictionary generation as LLM model is not configured.")


if __name__ == "__main__":
    print("Starting data cleaning and unification process directly...")
    try:
        # Call the main function that performs all cleaning and unification
        unified_data_cleaning_and_initial_processing()
    except Exception as e:
        print(f"Error during data cleaning and unification: {e}")
        import traceback
        traceback.print_exc()