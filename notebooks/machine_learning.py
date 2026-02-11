import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import chi2_contingency

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
from sklearn.metrics import classification_report, roc_auc_score, f1_score

# 1. PREPROCESSING: Clean and Split
def prepare_duel_datasets(df):
    # Outlier treatment (Capping at 1st and 99th percentile)
    for col in ['account_balance', 'nb_previous_interactions']:
        upper = df[col].quantile(0.99)
        lower = df[col].quantile(0.01)
        df[col] = df[col].clip(lower, upper)

    # TARGET
    y = df['has_subscribed_target']

    # FEATURES A: Client only
    client_cols = ['client_age', 'job_category', 'marital_status', 'education_level', 
                   'has_credit_default', 'account_balance', 'has_housing_loan', 'has_personal_loan']
    X_a = df[client_cols]

    # FEATURES B: Client + Macro + Campaign (Integrated)
    # We drop IDs, redundant dates, and metadata
    drop_cols = ['has_subscribed_target', 'economics_id', 'campaign_id', 'call_date', 
                 'campaign_start_date', 'campaign_end_date', 'date', 'currency', 
                 'rate_description', 'campaign_name', 'call_duration_sec']
    X_b = df.drop(columns=drop_cols)

    return X_a, X_b, y


# 2. FUNCTION: Train and Evaluate with SMOTE & Hyperparameters
def train_and_evaluate(X, y, model_name):
    # Automatic detection of categorical and numeric columns
    cat_cols = X.select_dtypes(include=['object']).columns.tolist()
    num_cols = X.select_dtypes(include=['int64', 'float64']).columns.tolist()

    # Preprocessing Pipeline
    preprocessor = ColumnTransformer([
        ('num', StandardScaler(), num_cols),
        ('cat', OneHotEncoder(handle_unknown='ignore'), cat_cols)
    ])

    # Model Pipeline (Including SMOTE to balance the data)
    # Hyperparameters: Random Forest is more complex and handles non-linear patterns better
    pipeline = ImbPipeline([
        ('pre', preprocessor),
        ('smote', SMOTE(random_state=42)),
        ('clf', RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42))
    ])

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
    pipeline.fit(X_train, y_train)
    
    y_pred = pipeline.predict(X_test)
    y_proba = pipeline.predict_proba(X_test)[:, 1]

    print(f"\n--- {model_name} Results ---")
    print(f"F1-Score: {f1_score(y_test, y_pred):.4f}")
    print(f"AUC-ROC:  {roc_auc_score(y_test, y_proba):.4f}")
    return f1_score(y_test, y_pred)



def test_stat_significance(df, feature_col, target_col='has_subscribed_target'):
    # 1. Create a contingency table (Cross-tabulation)
    contingency_table = pd.crosstab(df[feature_col], df[target_col])
    
    # 2. Run Chi-Square Test
    chi2, p, dof, expected = chi2_contingency(contingency_table)
    
    print(f"--- Statistical Test: {feature_col} vs Target ---")
    print(f"Chi-Square Statistic: {chi2:.2f}")
    print(f"P-Value: {p:.4e}") # Scientific notation for very small numbers
    
    # 3. Interpretation
    alpha = 0.05
    if p < alpha:
        print(f"✅ RESULT: Significant (p < {alpha}). \nWe reject the Null Hypothesis. The variables are dependent.")
    else:
        print(f"❌ RESULT: Not Significant (p >= {alpha}). \nWe cannot reject the Null Hypothesis.")
    print("-" * 30)


