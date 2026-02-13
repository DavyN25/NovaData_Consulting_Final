import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import calendar
import re
from difflib import get_close_matches
from bs4 import BeautifulSoup

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, ConfusionMatrixDisplay, RocCurveDisplay







    

def explore_dataset(df: pd.DataFrame, name: str = "Dataset") -> None:
    """
    Print basic information about a pandas DataFrame.

    Parameters
    ----------
    df : pd.DataFrame
        The dataset to explore.
    name : str, optional
        Name of the dataset (for display purposes).
    """
    print(f"=== {name} ===")
    print("\nShape:", df.shape)
    print("\nColumns:\n", df.columns.tolist())
    print("\nInfo:")
    print(df.info())
    print("\nMissing values per column:\n", df.isna().sum()*100/len(df))
    print("\nPercentage of duplicated rows:\n", df.duplicated().mean() * 100)
    print("\nFirst 5 rows:\n", df.head())





def add_year_from_month_sequence(df, month_col="month", start_year=2008):
    """
    Reconstructs a year column based on the chronological order of months
    in the UCI Bank Marketing dataset.

    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe containing a month column with abbreviated month names.
    month_col : str
        Name of the column containing month abbreviations (e.g., 'jan', 'feb').
    start_year : int
        The first year of the dataset (UCI dataset starts around 2008).

    Returns
    -------
    pd.DataFrame
        The same dataframe with two new columns:
        - 'month_num': numeric month (1–12)
        - 'year': reconstructed year based on month progression
    """

    # Month mapping
    month_map = {
        "jan": 1, "feb": 2, "mar": 3, "apr": 4,
        "may": 5, "jun": 6, "jul": 7, "aug": 8,
        "sep": 9, "oct": 10, "nov": 11, "dec": 12
    }

    # Validate input
    if month_col not in df.columns:
        raise ValueError(f"Column '{month_col}' not found in dataframe.")

    # Map month abbreviations to numbers
    df["month_num"] = df[month_col].map(month_map)

    if df["month_num"].isna().any():
        invalid = df[df["month_num"].isna()][month_col].unique()
        raise ValueError(f"Invalid month values found: {invalid}")

    # Reconstruct year sequence
    years = []
    current_year = start_year
    last_month = df["month_num"].iloc[0]

    for month in df["month_num"]:
        if month < last_month:  # Month sequence restarted → new year
            current_year += 1
        years.append(current_year)
        last_month = month

    df["year"] = years

    return df


def clean_uci_dataset(df):
    """
    Standardizes the UCI Bank dataset.
    Includes: Normalization, Imputation (Pre-merge cleaning), and Binary Encoding.
    """
    # Use the parameter 'df' throughout to avoid Scope Errors (NameErrors)
    df = df.copy()

    # 1. Column Renaming (Explicit Metadata)
    mapping = {
        'age': 'client_age',
        'job': 'job_category',
        'marital': 'marital_status',
        'education': 'education_level',
        'default': 'has_credit_default',
        'balance': 'account_balance',
        'housing': 'has_housing_loan',
        'loan': 'has_personal_loan',
        'contact': 'contact_type',
        'day': 'last_contact_day',
        'month': 'last_contact_month',
        'duration': 'call_duration_sec',
        'campaign': 'contacts_this_campaign',
        'pdays': 'days_since_prev_contact',
        'previous': 'nb_previous_interactions',
        'poutcome': 'prev_campaign_outcome',
        'y': 'has_subscribed_target'
    }
    df = df.rename(columns=mapping)

    # 2. Convert 'unknown' to real NaN 
    df = df.replace('unknown', np.nan)

    # 3. PRE-MERGE CLEANING (Imputation) to ensure data integrity before joining with external sources.
    
    # Fill Categorical NaNs with a standard label
    cat_cols = df.select_dtypes(include=['object']).columns
    df[cat_cols] = df[cat_cols].fillna('not_reported')

    # Fill Numerical NaNs with Median (Robust to outliers)
    num_cols = df.select_dtypes(include=['int64', 'float64']).columns
    for col in num_cols:
        df[col] = df[col].fillna(df[col].median())

    # 4. Normalization (Strings to lowercase)
    for col in cat_cols:
        df[col] = df[col].str.lower().str.strip()

    # 5. Binary Encoding
    binary_fields = [
        'has_credit_default', 'has_housing_loan', 
        'has_personal_loan', 'has_subscribed_target'
    ]
    
    for col in binary_fields:
        if col in df.columns:
            # Re-map after imputation/normalization to ensure 1/0
            df[col] = df[col].map({'yes': 1, 'no': 0, '1': 1, '0': 0, 1: 1, 0: 0})
            
            # Validation for the Jury
            nan_count = df[col].isna().sum()
            if nan_count > 0:
                print(f"Warning: {nan_count} values in {col} failed mapping. Check raw data.")
                
    # 6. Balance outlier capping (Only if column exists)
    if "balance" in df.columns:
        low, high = df["balance"].quantile([0.01, 0.99])
        df["balance"] = df["balance"].clip(low, high)
    else:
        print("Note: 'balance' column not found in this dataset version. Skipping capping.")
    
    # 7. Specific Feature Cleaning
    # Standard practice: -1 means 'never contacted', 999 is standard for distance-based ML
    if 'days_since_prev_contact' in df.columns:
        df['days_since_prev_contact'] = df['days_since_prev_contact'].replace(-1, 999)

    return df





def generate_campaign_table(df, year_col="year", month_col="last_contact_month", month_num_col="month_num"):
    """
    Generate a normalized Campaign table from the UCI Bank Marketing dataset.

    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe containing year, month, and month_num columns.
    year_col : str
        Column name containing the reconstructed year.
    month_col : str
        Column name containing month abbreviations (e.g., 'jan', 'feb').
    month_num_col : str
        Column name containing numeric month values (1–12).

    Returns
    -------
    pd.DataFrame
        A dataframe with the following columns:
        - campaign_id
        - campaign_name
        - campaign_start_date
        - campaign_end_date
        - campaign_channel
    """

    # Validate required columns
    required_cols = [year_col, month_col, month_num_col]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    # Extract unique campaigns
    campaigns = (
        df[[year_col, month_col, month_num_col]]
        .drop_duplicates()
        .sort_values([year_col, month_num_col])
        .reset_index(drop=True)
    )

    # Assign campaign_id
    campaigns["campaign_id"] = campaigns.index + 1

    # Campaign name
    campaigns["campaign_name"] = (
        "Term Deposit Campaign – "
        + campaigns[month_col].str.capitalize()
        + " "
        + campaigns[year_col].astype(str)
    )

    # Start date (vectorized)
    campaigns["campaign_start_date"] = campaigns.apply(
        lambda row: datetime(row[year_col], row[month_num_col], 1),
        axis=1
    )

    # End date (vectorized)
    campaigns["campaign_end_date"] = campaigns.apply(
        lambda row: datetime(
            row[year_col],
            row[month_num_col],
            calendar.monthrange(row[year_col], row[month_num_col])[1]
        ),
        axis=1
    )

    # Placeholder channel (to be replaced by scraping)
    campaigns["campaign_channel"] = "Phone / Call Center"

    # Final column order
    campaigns = campaigns[
        [
            "campaign_id",
            "campaign_name",
            "campaign_start_date",
            "campaign_end_date",
            "campaign_channel",
        ]
    ]

    return campaigns








def enrich_campaign_channels(campaigns_df, scraped_df):
    """
    Enriches the campaign table with scraped campaign_channel values.

    Parameters
    ----------
    campaigns_df : pd.DataFrame
        Campaign dimension table generated earlier.
    scraped_df : pd.DataFrame
        DataFrame containing scraped campaign metadata:
        - campaign_name
        - campaign_channel

    Returns
    -------
    pd.DataFrame
        Enriched campaign table with updated campaign_channel values.
    """

    # 1. Deduplicate scraped data to prevent merge explosions
    scraped_clean = scraped_df.drop_duplicates(subset=['campaign_name'])

    # 2. Perform the merge
    merged = campaigns_df.merge(
        scraped_clean[["campaign_name", "campaign_channel"]],
        on="campaign_name",
        how="left"
    )

    # 3. Initialize the source tracking column
    merged["enrichment_source"] = "exact_match"
    merged.loc[merged["campaign_channel"].isna(), "enrichment_source"] = "pending"

    # 4. Fuzzy Matching (Applied only to missing values)
    candidates = scraped_clean["campaign_name"].tolist()

    def find_fuzzy(name):
        match = get_close_matches(name, candidates, n=1, cutoff=0.8)
        return match[0] if match else None

    # Identify rows needing fuzzy match
    mask = merged["enrichment_source"] == "pending"

    # Map the fuzzy names to the channels
    if mask.any():
        # Create a mapping dictionary for speed
        fuzzy_map = {name: find_fuzzy(name) for name in merged.loc[mask, "campaign_name"].unique()}

        # Apply mapping
        merged.loc[mask, "matched_name"] = merged.loc[mask, "campaign_name"].map(fuzzy_map)

        # Pull channel from scraped_clean based on the matched name
        # We join back to get the channel for the fuzzy-matched name
        final_lookup = merged.loc[mask].merge(
            scraped_clean, 
            left_on="matched_name", 
            right_on="campaign_name", 
            how="left", 
            suffixes=('', '_scraped')
        )

        # Update original merged dataframe
        merged.loc[mask, "campaign_channel"] = final_lookup["campaign_channel_scraped"].values
        merged.loc[mask, "enrichment_source"] = "fuzzy_match"

    # 5. Handle remaining unknowns
    unknown_mask = merged["campaign_channel"].isna()
    merged.loc[unknown_mask, "campaign_channel"] = "Unknown"
    merged.loc[unknown_mask, "enrichment_source"] = "none/default"

    # Drop temporary helper column
    if "matched_name" in merged.columns:
        merged = merged.drop(columns=["matched_name"])

    return merged


def clean_campaign_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Cleaning and validation of campaign dimension table.
    """
    df = df.copy()

    # Ensure datetime
    df["campaign_start_date"] = pd.to_datetime(df["campaign_start_date"])
    df["campaign_end_date"] = pd.to_datetime(df["campaign_end_date"])

    # Normalize channel naming
    df["campaign_channel"] = df["campaign_channel"].str.strip()

    return df




def clean_ecb_market_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Cleans raw ECB interest rate data for the campaign enrichment project.

    Processing Steps:
    1. Standardize column names (uppercase → lowercase).
    2. Rename ECB technical fields to business-friendly names.
    3. Filter to retain only relevant analytical columns.
    4. Convert time period to datetime.
    5. Extract year and month for joins.
    6. Handle missing observations.
    7. Sort chronologically.
    """

    # 1. Defensive copy
    df_clean = df.copy()

    # 2. Normalize column names to uppercase
    df_clean.columns = [col.upper() for col in df_clean.columns]

    # 3. Column mapping (ECB → business names)
    column_mapping = {
        "OBS_VALUE": "ecb_rate",
        "TIME_PERIOD": "date",
        "TITLE": "rate_description",
        "CURRENCY": "currency",
    }

    # 4. Rename columns 
    df_clean = df_clean.rename(columns=column_mapping)

    # 5. KEEP only relevant columns 
    expected_cols = ["ecb_rate", "date", "rate_description", "currency"]
    df_clean = df_clean[[col for col in expected_cols if col in df_clean.columns]]

    # 6. Normalize column names to lowercase
    df_clean.columns = [col.lower() for col in df_clean.columns]

    # 7. Convert date to datetime
    df_clean["date"] = pd.to_datetime(df_clean["date"], errors="coerce")

    # 8. Extract year and month for merging
    df_clean["year"] = df_clean["date"].dt.year
    df_clean["month_num"] = df_clean["date"].dt.month

    # 9. Drop rows with missing ECB rate (business-critical field)
    df_clean = df_clean.dropna(subset=["ecb_rate"])

    # 10. Filter ECB data to the UCI campaign period (2008–2010)
    df_clean = df_clean[df_clean["year"].between(2008, 2010)]

    # 11. Sort chronologically
    df_clean = df_clean.sort_values("date").reset_index(drop=True)

    return df_clean







def perform_eda(df: pd.DataFrame):
    """
    Performs exploratory data analysis aligned with RNCP expectations.
    Focus on business insight and communication.
    """

    import matplotlib.pyplot as plt
    import seaborn as sns

    # Target distribution
    sns.countplot(x="subscribed", data=df)
    plt.title("Term Deposit Subscription Distribution")
    plt.show()

    # Subscription rate by campaign channel
    channel_rate = (
        df.groupby("campaign_channel")["subscribed"]
        .mean()
        .sort_values(ascending=False)
    )

    channel_rate.plot(kind="bar")
    plt.title("Subscription Rate by Campaign Channel")
    plt.ylabel("Subscription Rate")
    plt.show()

    # Call duration vs subscription
    sns.boxplot(x="subscribed", y="duration", data=df)
    plt.title("Call Duration vs Subscription")
    plt.show()



def create_age_groups(df, age_col='client_age'):
    """
    Creates categorical age groups (bins) from a numerical age column.
    Useful for segmentation, visualization, and behavioral analysis.

    Parameters
    ----------
    df : pandas.DataFrame
        The dataset containing the age column.
    age_col : str
        The name of the numerical age column.

    Returns
    -------
    pandas.DataFrame
        A dataframe with a new column 'age_group_bin'.
    """

    df['age_group_bin'] = pd.cut(
        df[age_col],
        bins=[0, 25, 35, 45, 55, 65, 120],
        labels=['18-25', '26-35', '36-45', '46-55', '56-65', '65+']
    )

    return df

def create_campaign_groups(df, campaign_col='num_contacts_current_campaign'):
    """
    Groups the number of contacts made during the current campaign into 
    meaningful categories. Helps analyze diminishing returns and customer fatigue.

    Parameters
    ----------
    df : pandas.DataFrame
        The dataset containing the campaign column.
    campaign_col : str
        The name of the numerical campaign column.

    Returns
    -------
    pandas.DataFrame
        A dataframe with a new column 'campaign_group_bin'.
    """

    df['campaign_group'] = pd.cut(
        df[campaign_col],
        bins=[0, 1, 3, 5, 10, 100],
        labels=['1 contact', '2–3 contacts', '4–5 contacts', '6–10 contacts', '10+ contacts']
    )

    return df


def create_contact_missing_flag(df, contact_col='contact_type'):
    """
    Creates a binary flag indicating whether the contact_type field 
    was missing in the original dataset.

    Parameters
    ----------
    df : pandas.DataFrame
        The dataset containing the contact column.
    contact_col : str
        The name of the contact column.

    Returns
    -------
    pandas.DataFrame
        A dataframe with a new column 'contact_missing_flag'.
    """

    df['contact_missing_flag'] = df[contact_col].isna().astype(int)
    return df




def detect_outliers_iqr(df, columns, method="flag"):
    """
    Detects outliers in numerical columns using the IQR (Interquartile Range) method.
    Can either flag, remove, or cap outliers depending on the selected method.

    Parameters
    ----------
    df : pandas.DataFrame
        The dataset containing numerical columns.
    columns : list
        List of numerical column names to check for outliers.
    method : str, optional
        How to handle outliers:
        - "flag": create a binary flag column for each variable
        - "remove": drop rows containing outliers
        - "cap": cap outliers to the IQR boundaries
        Default is "flag".

    Returns
    -------
    pandas.DataFrame
        A dataframe with outliers flagged, removed, or capped.
    """

    df = df.copy()

    for col in columns:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1

        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR

        if method == "flag":
            df[f"{col}_outlier_flag"] = (
                (df[col] < lower_bound) | (df[col] > upper_bound)
            ).astype(int)

        elif method == "remove":
            df = df[(df[col] >= lower_bound) & (df[col] <= upper_bound)]

        elif method == "cap":
            df[col] = df[col].clip(lower=lower_bound, upper=upper_bound)

        else:
            raise ValueError("method must be 'flag', 'remove', or 'cap'")

    return df







def handle_missing_values_uci(df: pd.DataFrame) -> pd.DataFrame:
    """
    Handle missing values for the UCI Bank Marketing dataset.

    Steps:
    - Drop 'poutcome' column
    - Drop rows where 'job' or 'education' is missing
    - Impute remaining missing values with 'missing'

    Parameters
    ----------
    df : pd.DataFrame

    Returns
    -------
    pd.DataFrame
    """
    df_mv = df.copy()

    # Drop 'poutcome' column
    if 'poutcome' in df_mv.columns:
        df_mv = df_mv.drop(columns=['poutcome'])

    # Drop rows missing job or education
    df_mv = df_mv.dropna(subset=['job', 'education'])

    # Impute remaining missing values
    df_mv = df_mv.fillna('missing')

    print("Missing-value handling complete. Shape:", df_mv.shape)
    return df_mv
  
    import pandas as pd
import uuid



def eda_uci_dataset(df, target_col='y'):
    """
    Perform basic EDA on the UCI bank marketing dataset with embedded explanations.
    
    What:
        - Univariate analysis of key numeric features
        - Target distribution
        - Selected bivariate relationships (numeric + categorical vs target)
        - Correlation matrix for numeric features
    
    Why:
        - To understand the data structure and distributions
        - To detect class imbalance and potential data quality issues
        - To identify variables that are likely to be predictive of the target
        - To inform feature engineering and model design in a business-relevant way
    """
    
    # 1. Target distribution
    print("\n[1] Target distribution")
    print("What: Distribution of the target variable.")
    print("Why: To check for class imbalance and understand baseline conversion rates.\n")
    
    plt.figure(figsize=(6, 4))
    sns.countplot(x=target_col, data=df)
    plt.title('Target Distribution')
    plt.xlabel(f'{target_col}')
    plt.ylabel('Count')
    plt.show()
    
    # 2. Numeric distributions
    print("\n[2] Numeric feature distributions")
    print("What: Histograms of numeric variables.")
    print("Why: To detect skewness, outliers, and typical value ranges,")
    print("     which influence scaling choices and model robustness.\n")
    
    numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns
    df[numeric_cols].hist(figsize=(12, 8), bins=30)
    plt.suptitle('Numeric Feature Distributions', y=1.02)
    plt.tight_layout()
    plt.show()
    
    # 3. Subscription rate by job
    if 'job' in df.columns:
        print("\n[3] Subscription rate by job")
        print("What: Mean target value by job category.")
        print("Why: To identify professional segments with higher conversion rates,")
        print("     which is key for marketing segmentation and prioritization.\n")
        
        job_y = (df
                 .groupby('job')[target_col]
                 .mean()
                 .sort_values(ascending=False))
        
        plt.figure(figsize=(10, 5))
        sns.barplot(x=job_y.index, y=job_y.values)
        plt.title('Subscription Rate by Job')
        plt.xlabel('Job')
        plt.ylabel('Mean Subscription Rate')
        plt.xticks(rotation=45, ha='right')
        plt.show()
    
    # 4. Balance vs target
    if 'balance' in df.columns:
        print("\n[4] Balance vs target")
        print("What: Boxplot of account balance by target.")
        print("Why: To test whether higher balances are associated with higher subscription rates,")
        print("     which supports targeting higher-value clients.\n")
        
        plt.figure(figsize=(8, 4))
        sns.boxplot(x=target_col, y='balance', data=df)
        plt.title('Balance by Target')
        plt.xlabel('Target')
        plt.ylabel('Balance')
        plt.ylim(df['balance'].quantile(0.01),
                 df['balance'].quantile(0.99))
        plt.show()
    
    # 5. Correlation matrix
    print("\n[5] Correlation matrix of numeric features")
    print("What: Correlation heatmap of numeric variables.")
    print("Why: To understand relationships between features and with the target,")
    print("     and to detect potential multicollinearity before modeling.\n")
    
    plt.figure(figsize=(10, 8))
    corr = df[numeric_cols].corr()
    sns.heatmap(corr, annot=False, cmap='coolwarm', center=0)
    plt.title('Correlation Matrix of Numeric Features')
    plt.show()
    

