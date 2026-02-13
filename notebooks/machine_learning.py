import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import chi2_contingency


from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
from sklearn.metrics import classification_report, roc_auc_score, f1_score, ConfusionMatrixDisplay, RocCurveDisplay

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
        print(f"RESULT: Significant (p < {alpha}). \nWe reject the Null Hypothesis. The variables are dependent.")
    else:
        print(f"RESULT: Not Significant (p >= {alpha}). \nWe cannot reject the Null Hypothesis.")
    print("-" * 30)





def visualize_smote(n_samples=500, weights=[0.95], colors=["#3274A1", "#C0392B"]):
    """
    Generates a visualization of SMOTE (Synthetic Minority Over-sampling Technique).
    
    Parameters:
    - n_samples: Total number of samples to generate (default: 500)
    - weights: Ratio of the majority class (default: [0.95] means 95% Majority / 5% Minority)
    - colors: List of two hex codes for plotting [Majority Color, Minority Color]
    """
    
    # 0. SETUP STYLE
    sns.set_palette(sns.color_palette(colors))
    sns.set_theme(style="whitegrid")

    # 1. GENERATE IMBALANCED DATA
    # Create a fake dataset with 2 features for easy 2D plotting
    X, y = make_classification(
        n_samples=n_samples, 
        n_features=2, 
        n_redundant=0, 
        n_clusters_per_class=1, 
        weights=weights, 
        flip_y=0, 
        random_state=42
    )

    # Convert to DataFrame
    df_original = pd.DataFrame(X, columns=['Feature 1', 'Feature 2'])
    df_original['Target'] = y

    # 2. APPLY SMOTE
    smote = SMOTE(random_state=42)
    X_res, y_res = smote.fit_resample(X, y)

    # Convert to DataFrame
    df_smote = pd.DataFrame(X_res, columns=['Feature 1', 'Feature 2'])
    df_smote['Target'] = y_res

    # 3. VISUALIZE
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))

    # Plot Before
    sns.scatterplot(
        data=df_original, x='Feature 1', y='Feature 2', hue='Target',
        palette=colors, style='Target', s=100, alpha=0.6, ax=axes[0]
    )
    axes[0].set_title(f'Before SMOTE: Imbalanced Dataset\n(Minority: {sum(y)} samples)', fontsize=14, fontweight='bold', color='#2C3E50')
    axes[0].legend(title='Class')

    # Plot After
    sns.scatterplot(
        data=df_smote, x='Feature 1', y='Feature 2', hue='Target',
        palette=colors, style='Target', s=100, alpha=0.6, ax=axes[1]
    )
    axes[1].set_title(f'After SMOTE: Balanced Dataset\n(Minority: {sum(y_res)} samples)', fontsize=14, fontweight='bold', color='#2C3E50')
    axes[1].legend(title='Class')

    plt.tight_layout()
    plt.show()


def visualize_hyperparameters():
    # Define the grid
    param_grid = {
        'n_estimators': [100, 200],
        'max_depth': [10, 20, "None"],
        'min_samples_split': [2, 5],
        'class_weight': ['balanced', 'None']
    }

    # Convert to a readable DataFrame for the report
    # We create a list of dictionaries to format it nicely
    rows = []
    for key, values in param_grid.items():
        rows.append({'Hyperparameter': key, 'Values Tested': str(values)})

    df_params = pd.DataFrame(rows)

    # Plotting as a Table
    fig, ax = plt.subplots(figsize=(8, 3)) # Small size
    ax.axis('tight')
    ax.axis('off')
    
    # Create the table
    table = ax.table(
        cellText=df_params.values, 
        colLabels=df_params.columns, 
        cellLoc='left', 
        loc='center',
        colColours=["#3274A1", "#3274A1"] # Your Brand Blue
    )
    
    # Styling
    table.auto_set_font_size(False)
    table.set_fontsize(12)
    table.scale(1.2, 1.5) # Make cells taller
    
    # Color text white for headers
    for (row, col), cell in table.get_celld().items():
        if row == 0:
            cell.get_text().set_color('white')
            cell.get_text().set_weight('bold')

    plt.title('Grid Search Search Space', fontsize=14, fontweight='bold', color='#2C3E50', y=1.1)
    plt.show()




def plot_model_evaluation(model, X_test, y_test, colors=["#3274A1", "#C0392B"]):
    """
    Generates a dual-plot with Confusion Matrix and ROC Curve.
    
    Parameters:
    - model: The trained model (e.g., Random Forest, XGBoost)
    - X_test: The test features
    - y_test: The true test labels
    - colors: List of [Majority Color, Minority/Action Color]
    """
    
    # 0. GET PREDICTIONS
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1] # Probability for Class 1

    # 1. SETUP PLOT STYLE
    sns.set_palette(sns.color_palette(colors))
    sns.set_theme(style="white") 

    # Create Figure
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))

    # ==============================================
    # PLOT 1: CONFUSION MATRIX
    # ==============================================
    cm = confusion_matrix(y_test, y_pred)

    # Dynamic Labels
    group_names = ['True Neg\n(Correct Rejection)', 'False Pos\n(Wasted Call)', 
                   'False Neg\n(Missed Opportunity)', 'True Pos\n(Successful Sale)']
    group_counts = ["{0:0.0f}".format(value) for value in cm.flatten()]
    group_percentages = ["{0:.2%}".format(value) for value in cm.flatten()/np.sum(cm)]
    
    labels = [f"{v1}\n{v2}\n{v3}" for v1, v2, v3 in zip(group_names, group_counts, group_percentages)]
    labels = np.asarray(labels).reshape(2,2)

    # Plot Heatmap
    sns.heatmap(
        cm, 
        annot=labels, 
        fmt='', 
        cmap='Blues', 
        cbar=False, 
        ax=axes[0], 
        annot_kws={"fontsize":12, "fontweight":"bold"}
    )

    axes[0].set_title('Confusion Matrix: Operational Impact', fontsize=16, fontweight='bold', color='#2C3E50')
    axes[0].set_ylabel('Actual Status', fontsize=12)
    axes[0].set_xlabel('Predicted Status', fontsize=12)

    # ==============================================
    # PLOT 2: ROC CURVE
    # ==============================================
    fpr, tpr, thresholds = roc_curve(y_test, y_proba)
    roc_auc = auc(fpr, tpr)

    # Plot Curve
    axes[1].plot(
        fpr, tpr, 
        color=colors[1], # Red/Orange for the line 
        lw=3, 
        label=f'ROC Curve (AUC = {roc_auc:.2f})'
    )

    # Plot Random Guess
    axes[1].plot([0, 1], [0, 1], color='#2C3E50', lw=2, linestyle='--', label='Random Guess')

    # Customization
    axes[1].set_xlim([0.0, 1.0])
    axes[1].set_ylim([0.0, 1.05])
    axes[1].set_xlabel('False Positive Rate (False Alarms)', fontsize=12)
    axes[1].set_ylabel('True Positive Rate (Recall)', fontsize=12)
    axes[1].set_title('ROC Curve: Predictive Power', fontsize=16, fontweight='bold', color='#2C3E50')
    axes[1].legend(loc="lower right", fontsize=12)
    axes[1].grid(True, alpha=0.3)
    axes[1].fill_between(fpr, tpr, color=colors[1], alpha=0.1)

    plt.tight_layout()
    plt.show()

