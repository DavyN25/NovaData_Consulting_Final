import pandas as pd
import os
from sqlalchemy import create_engine, text
from dotenv import load_dotenv
from urllib.parse import quote_plus

# ==========================================
# 1. SETUP & CONFIGURATION
# ==========================================

# Load environment variables from .env file
load_dotenv()

# Securely retrieve credentials
DB_USER = os.getenv("DB_USER")
DB_PASS = os.getenv("DB_PASSWORD")
DB_HOST = os.getenv("DB_HOST")
DB_PORT = os.getenv("DB_PORT")
DB_NAME = os.getenv("DB_NAME")

# Check if credentials exist
if not all([DB_USER, DB_PASS, DB_HOST, DB_NAME]):
    raise ValueError("Error: Missing database credentials in .env file.")

# Encode password to handle special chars like @ or # safely
encoded_password = quote_plus(DB_PASS)

# Construct Connection String
connection_string = f"mysql+pymysql://{DB_USER}:{encoded_password}@{DB_HOST}:{DB_PORT}/{DB_NAME}"

# Define paths (Adjust "../data/clean/" if your script is in a 'scripts' folder)
CLEAN_DATA_PATH = "../data/clean/" 

def get_db_engine():
    try:
        engine = create_engine(connection_string)
        # Test connection
        with engine.connect() as conn:
            print("Connected to the MySQL database...")
        return engine
    except Exception as e:
        print(f" Connection Error: {e}")
        return None

# ==========================================
# 2. FUNCTION: CLEAN DATABASE (Idempotency)
# ==========================================
def clean_database(engine):
    """
    Truncates all tables to prevent Duplicate Entry errors when re-running the script.
    """
    print("\n Cleaning existing data (Truncating tables)...")
    
    with engine.connect() as conn:
        try:
            # Disable Foreign Key Checks to allow truncating in any order
            conn.execute(text("SET FOREIGN_KEY_CHECKS = 0;"))
            
            # Truncate tables (Removes data, keeps structure)
            conn.execute(text("TRUNCATE TABLE fact_interactions;"))
            conn.execute(text("TRUNCATE TABLE dim_client;"))
            conn.execute(text("TRUNCATE TABLE dim_campaign;"))
            conn.execute(text("TRUNCATE TABLE dim_economics;"))
            
            # Re-enable Foreign Key Checks
            conn.execute(text("SET FOREIGN_KEY_CHECKS = 1;"))
            conn.commit()
            print("Database is clean and ready for fresh data.")
            
        except Exception as e:
            print(f"Error cleaning database: {e}")

# ==========================================
# 3. MAIN ETL FUNCTION
# ==========================================
def load_data():
    engine = get_db_engine()
    if not engine:
        return

    # STEP A: Clean the Slate
    clean_database(engine)

    # STEP B: Read CSV Files
    print("\n--- Reading Cleaned CSV Files ---")
    try:
        uci_df = pd.read_csv(os.path.join(CLEAN_DATA_PATH, "uci_bank_marketing_cleaned.csv"))
        campaign_df = pd.read_csv(os.path.join(CLEAN_DATA_PATH, "campaign_dim_cleaned.csv"))
        ecb_df = pd.read_csv(os.path.join(CLEAN_DATA_PATH, "ecb_interest_rates_cleaned.csv"))
        print("Files loaded successfully.")
    except FileNotFoundError as e:
        print(f"File not found: {e}")
        return

    # STEP C: Transform & Load DIM_ECONOMICS
    print("\n--- Processing DIM_ECONOMICS ---")
    ecb_df['date'] = pd.to_datetime(ecb_df['date'])
    # Create Smart Key (YYYYMMDD)
    ecb_df['economics_id'] = ecb_df['date'].dt.strftime('%Y%m%d').astype(int)
    
    dim_economics = ecb_df.rename(columns={'date': 'economics_date'})
    dim_economics = dim_economics[['economics_id', 'economics_date', 'ecb_rate', 'rate_description', 'currency']]
    
    # Upload
    dim_economics.to_sql('dim_economics', engine, if_exists='append', index=False)
    print(f"DIM_ECONOMICS Loaded: {len(dim_economics)} rows.")

    # STEP D: Transform & Load DIM_CAMPAIGN
    print("\n--- Processing DIM_CAMPAIGN ---")
    # Generate explicit IDs
    campaign_df['campaign_id'] = range(1, len(campaign_df) + 1)
    campaign_df['campaign_start_date'] = pd.to_datetime(campaign_df['campaign_start_date'])
    campaign_df['campaign_end_date'] = pd.to_datetime(campaign_df['campaign_end_date'])
    
    dim_campaign = campaign_df[['campaign_id', 'campaign_name', 'campaign_start_date', 'campaign_end_date', 'campaign_channel']]
    
    # Upload
    dim_campaign.to_sql('dim_campaign', engine, if_exists='append', index=False)
    print(f"DIM_CAMPAIGN Loaded: {len(dim_campaign)} rows.")

    # STEP E: Transform & Load DIM_CLIENT
    print("\n--- Processing DIM_CLIENT ---")
    # Generate explicit IDs
    uci_df['client_id'] = range(1, len(uci_df) + 1)
    
    dim_client = uci_df[[
        'client_id', 'client_age', 'job_category', 'marital_status', 
        'education_level', 'has_credit_default', 'has_housing_loan', 
        'has_personal_loan', 'account_balance'
    ]].copy()
    
    # Upload
    dim_client.to_sql('dim_client', engine, if_exists='append', index=False)
    print(f"DIM_CLIENT Loaded: {len(dim_client)} rows.")

    # STEP F: Transform & Load FACT_INTERACTIONS
    print("\n--- Processing FACT_INTERACTIONS ---")
    
    # 1. Create a Call Date Object for mapping
    uci_df['call_date'] = pd.to_datetime(dict(year=uci_df.year, month=uci_df.month_num, day=uci_df.last_contact_day))

    # 2. Map Economics ID (YYYYMMDD)
    uci_df['economics_id'] = uci_df['call_date'].dt.strftime('%Y%m%d').astype(int)

    # 3. Map Campaign ID (Range Mapping)
    # Default to 1 (or Unknown)
    uci_df['campaign_id'] = 1 
    
    # Iterate campaigns to find matches (vectorized approach is better for huge data, but loop is fine for 30 campaigns)
    for index, row in campaign_df.iterrows():
        mask = (uci_df['call_date'] >= row['campaign_start_date']) & (uci_df['call_date'] <= row['campaign_end_date'])
        uci_df.loc[mask, 'campaign_id'] = row['campaign_id']

    # 4. Prepare Fact DataFrame
    fact_interactions = uci_df.rename(columns={
        'call_duration_sec': 'contact_duration_sec',
        'days_since_prev_contact': 'days_since_previous_contact'
    })

    # Select columns matching SQL Schema
    fact_cols = [
        'client_id', 'campaign_id', 'economics_id',
        'contact_type', 'last_contact_day', 'last_contact_month',
        'contact_duration_sec', 'contacts_this_campaign',
        'days_since_previous_contact', 'nb_previous_interactions',
        'prev_campaign_outcome', 'has_subscribed_target'
    ]
    
    fact_interactions = fact_interactions[fact_cols]

    # Upload
    fact_interactions.to_sql('fact_interactions', engine, if_exists='append', index=False)
    print(f"FACT_INTERACTIONS Loaded: {len(fact_interactions)} rows.")

    print("\nSUCCESS: Data Warehouse fully populated!")

if __name__ == "__main__":
    load_data()