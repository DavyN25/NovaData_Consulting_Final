import pandas as pd
from google.cloud import bigquery
import os

# ==========================================
# 1. PREPARE DENORMALIZED DATA
# ==========================================

def get_denormalized_data(uci_df, campaign_df, ecb_df):
    """
    Combines normalized DataFrames into a single Wide Table for BigQuery.
    """
    print("\n--- Starting Denormalization for BigQuery ---")

    # A. Prepare Dates & IDs (Re-using our existing logic)
    ecb_df['date'] = pd.to_datetime(ecb_df['date'])
    ecb_df['economics_id'] = ecb_df['date'].dt.strftime('%Y%m%d').astype(int)
    
    uci_df['call_date'] = pd.to_datetime(dict(year=uci_df.year, month=uci_df.month_num, day=uci_df.last_contact_day))
    uci_df['economics_id'] = uci_df['call_date'].dt.strftime('%Y%m%d').astype(int)
    
    campaign_df['campaign_id'] = range(1, len(campaign_df) + 1)
    campaign_df['campaign_start_date'] = pd.to_datetime(campaign_df['campaign_start_date'])
    campaign_df['campaign_end_date'] = pd.to_datetime(campaign_df['campaign_end_date'])

    # B. Map Campaign IDs to UCI Interactions
    uci_df['campaign_id'] = 1 
    for _, row in campaign_df.iterrows():
        mask = (uci_df['call_date'] >= row['campaign_start_date']) & (uci_df['call_date'] <= row['campaign_end_date'])
        uci_df.loc[mask, 'campaign_id'] = row['campaign_id']

    # C. Execute Merges (The Denormalization)
    # Join Interactions with Campaigns
    wide_df = uci_df.merge(campaign_df, on='campaign_id', how='left', suffixes=('', '_drop'))
    
    # Join with Economics (Interest Rates)
    wide_df = wide_df.merge(ecb_df, on='economics_id', how='left', suffixes=('', '_drop'))

    # D. Clean up duplicated columns from joins
    wide_df = wide_df.loc[:, ~wide_df.columns.str.contains('_drop')]
    
    print(f"Denormalized Table Created: {wide_df.shape[1]} columns and {len(wide_df)} rows.")
    return wide_df


# ==========================================
# 2. UPLOAD TO GOOGLE BIGQUERY
# ==========================================


def load_to_bigquery(df, project_id, dataset_id, table_name, key_filename):
    """
    Automates the upload of a denormalized DataFrame to Google BigQuery 
    with Partitioning and Clustering optimization.
    """
    # 1. Resolve pathing
    current_dir = os.getcwd()
    key_path = os.path.join(current_dir, key_filename)
    
    # 2. Set environment variable for authentication
    os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = key_path

    if not os.path.exists(key_path):
        return f"Error: {key_filename} not found at {key_path}"

    try:
        # 3. Initialize Client & Table ID
        client = bigquery.Client(project=project_id)
        table_id = f"{project_id}.{dataset_id}.{table_name}"

        # 4. CONFIGURE JOB WITH PARTITIONING AND CLUSTERING
        job_config = bigquery.LoadJobConfig(
            write_disposition="WRITE_TRUNCATE",
            
            # PARTITIONING: Optimizes costs when filtering by date
            time_partitioning=bigquery.TimePartitioning(
                type_=bigquery.TimePartitioningType.DAY,
                field="call_date" # Matches your wide_df date column
            ),
            
            # CLUSTERING: Speeds up "GROUP BY" and filter queries
            clustering_fields=["job_category", "education_level"]
        )

        print(f"Starting optimized upload to: {table_id}...")
        
        # 5. Run the job
        job = client.load_table_from_dataframe(df, table_id, job_config=job_config)
        job.result()  # Wait for completion

        return f"SUCCESS! Loaded {len(df)} rows with Partitioning & Clustering."

    except Exception as e:
        return f"BigQuery Error: {str(e)}"

        
# ==========================================
# 3. CLEAN TO CORRECT ERROR ON DATASET LOAD
# ==========================================

def perform_clean_upload(df):
    # 1. Setup the credentials using your JSON key file
    # Ensure this file is in the same folder as your notebook
    os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "key.json.json"

    # 2. Initialize the BigQuery Client
    client = bigquery.Client(project="rncp-bank-marketing")
    
    # Define your destination table
    table_id = "rncp-bank-marketing.bank_analytics.denormalized_marketing"

    # 3. CONFIGURE THE TRUNCATE SETTING
    # This is the "Undo" button that clears old data first
    job_config = bigquery.LoadJobConfig(
        write_disposition="WRITE_TRUNCATE", 
        source_format=bigquery.SourceFormat.PARQUET if hasattr(df, 'to_parquet') else None
    )

    print(f"🚀 Overwriting table {table_id} with wide_df data...")

    # 4. Start the load job
    job = client.load_table_from_dataframe(df, table_id, job_config=job_config)
    
    # Wait for the upload to finish
    job.result() 

    print(f"✅ SUCCESS! The table has been refreshed with {len(df)} rows.")

