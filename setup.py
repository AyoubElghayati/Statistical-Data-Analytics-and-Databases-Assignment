# ==============================================================================
# Module: CSC-40080 - Statistical Data Analytics and Databases
# Assessment Title: Data-Driven Problem and Analysis
# Student ID: 25044662
# Script: Water Quality Compliance Modeling Pipeline
# ==============================================================================

import os
import sqlite3
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.impute import KNNImputer
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier, export_text
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, 
    confusion_matrix, classification_report
)

# Set random seed for reproducibility
SEED = 42


# 1. DATA ACCESS & SQL INTEGRATION 

print("=" * 70)
print("1. INGESTING RAW DATASET & SQL DATABASE QUERY")
print("=" * 70)

raw_csv_path = "water_potability.csv"

if not os.path.exists(raw_csv_path):
    raise FileNotFoundError(f"Missing file '{raw_csv_path}'. Please place it in the working directory.")

# Load CSV into Pandas
df_raw = pd.read_csv(raw_csv_path)

# Simulate Relational Database Querying via SQLite In-Memory Database
conn = sqlite3.connect(":memory:")
df_raw.to_sql("water_monitoring_logs", conn, index=False, if_exists="replace")


sql_query = """
SELECT 
    ph,
    Hardness,
    Solids,
    Chloramines,
    Sulfate,
    Conductivity,
    Organic_carbon,
    Trihalomethanes,
    Turbidity,
    Potability
FROM water_monitoring_logs;
"""

df_db = pd.read_sql_query(sql_query, conn)
conn.close()

print(f"Data successfully pulled via SQL query. Dataset Dimensions: {df_db.shape}")
print("\nInitial Missing Value Summary:")
print(df_db.isnull().sum()[df_db.isnull().sum() > 0])


# 2. DATA PREPROCESSING & KNN IMPUTATION 

print("\n" + "=" * 70)
print("2. DATA CLEANING & IMPUTATION (KNN k=5)")
print("=" * 70)

features = ['ph', 'Hardness', 'Solids', 'Chloramines', 'Sulfate', 
            'Conductivity', 'Organic_carbon', 'Trihalomethanes', 'Turbidity']
target = 'Potability'

X = df_db[features]
y = df_db[target]

# Preserve sample size via 5-NN Imputation rather than listwise deletion
knn_imputer = KNNImputer(n_neighbors=5)
X_imputed_array = knn_imputer.fit_transform(X)
X_cleaned = pd.DataFrame(X_imputed_array, columns=features)

# Combine and export cleaned dataset
df_cleaned = X_cleaned.copy()
df_cleaned[target] = y.values
cleaned_csv_path = "cleaned_water_potability.csv"
df_cleaned.to_csv(cleaned_csv_path, index=False)

print(f"KNN Imputation complete. Saved cleaned file as '{cleaned_csv_path}'.")
print(f"Remaining null values across features: {X_cleaned.isnull().sum().sum()}")


# 3. TRAIN / TEST SPLIT & FEATURE SCALING

print("\n" + "=" * 70)
print("3. TRAIN-TEST SPLIT & STANDARDIZATION")
print("=" * 70)

# Stratified 80/20 train-test split to preserve target distribution
X_train, X_test, y_train, y_test = train_test_split(
    X_cleaned, y, test_size=0.20, random_state=SEED, stratify=y
)

print(f"Training set size: {X_train.shape[0]} rows")
print(f"Testing set size:  {X_test.shape[0]} rows")

# Standardize feature variables (zero mean, unit variance)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


# 4. MODELING & EVALUATION 

print("\n" + "=" * 70)
print("4. MODEL FITTING & PERFORMANCE COMPARISON")
print("=" * 70)


def compute_metrics(model_name, y_true, y_pred):
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, pos_label=1, zero_division=0)
    rec = recall_score(y_true, y_pred, pos_label=1, zero_division=0)
    f1 = f1_score(y_true, y_pred, pos_label=1, zero_division=0)
    cm = confusion_matrix(y_true, y_pred)
    return {
        "Model": model_name,
        "Accuracy": f"{acc:.1%}",
        "Precision": f"{prec:.1%}",
        "Recall (Non-Compliant)": f"{rec:.1%}",
        "F1-Score": f"{f1:.3f}",
        "TN": cm[0, 0], "FP": cm[0, 1],
        "FN": cm[1, 0], "TP": cm[1, 1]
    }

results_list = []

#  4a. Benchmark Model (Majority Class) 
benchmark = DummyClassifier(strategy="most_frequent")
benchmark.fit(X_train_scaled, y_train)
y_pred_bm = benchmark.predict(X_test_scaled)
results_list.append(compute_metrics("Benchmark (Majority Class)", y_test, y_pred_bm))

#  4b. Model 1: Logistic Regression
log_reg = LogisticRegression(class_weight='balanced', random_state=SEED, max_iter=1000)
log_reg.fit(X_train_scaled, y_train)
y_pred_lr = log_reg.predict(X_test_scaled)
results_list.append(compute_metrics("Logistic Regression (Balanced)", y_test, y_pred_lr))

#  4c. Model 2: Decision Tree Classifier (Threshold 0.50) 
dt_model = DecisionTreeClassifier(class_weight='balanced', max_depth=5, random_state=SEED)
dt_model.fit(X_train_scaled, y_train)
y_pred_dt = dt_model.predict(X_test_scaled)
results_list.append(compute_metrics("Decision Tree (depth=5)", y_test, y_pred_dt))

#  4d. Model 2 Variation: Decision Tree with Decision Threshold Tuning (0.35) 
dt_probabilities = dt_model.predict_proba(X_test_scaled)[:, 1]
y_pred_dt_tuned = (dt_probabilities >= 0.35).astype(int)
results_list.append(compute_metrics("Decision Tree (Threshold = 0.35)", y_test, y_pred_dt_tuned))


# 5. SUMMARY OF RESULTS & COEFFICIENT INSPECTION

summary_df = pd.DataFrame(results_list)
print("\nFINAL EVALUATION SUMMARY ON TEST SET (656 samples):")
print(summary_df.to_string(index=False))

print("\n" + "=" * 70)
print("5. LOGISTIC REGRESSION COEFFICIENTS (ODDS RATIOS)")
print("=" * 70)
coef_df = pd.DataFrame({
    'Feature': features,
    'Beta Coefficient': log_reg.coef_[0],
    'Odds Ratio': np.exp(log_reg.coef_[0])
}).sort_values(by='Odds Ratio', ascending=False)
print(coef_df.to_string(index=False))

print("\n" + "=" * 70)
print("6. DECISION TREE FEATURE IMPORTANCE")
print("=" * 70)
tree_imp_df = pd.DataFrame({
    'Feature': features,
    'Importance': dt_model.feature_importances_
}).sort_values(by='Importance', ascending=False)
print(tree_imp_df.to_string(index=False))

print("\nPipeline execution complete. All outputs generated successfully.")