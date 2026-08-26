"""
AI-Based Dermatological Diagnosis - Standalone Analysis & Baseline Model Evaluation
"""

import os
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

BASE_DIR = Path(__file__).resolve().parent

# Check for sample dataset in data/ or root
data_path = BASE_DIR / "data" / "skin_conditions_sample.csv"
if not data_path.exists():
    data_path = BASE_DIR / "skin_conditions_sample.csv"

# If extended dataset is available, also use it
extended_path = BASE_DIR / "data" / "dermatology_extended.csv"

print(f"Loading dataset from: {data_path}")
df = pd.read_csv(data_path)

print("\n--- Dataset Preview ---")
print(df.head())

print("\n--- Summary Statistics ---")
print(df.describe())

# Encode target variable
le = LabelEncoder()
df["diagnosis_encoded"] = le.fit_transform(df["diagnosis"])

X = df.drop(columns=["diagnosis", "diagnosis_encoded"], errors="ignore")
y = df["diagnosis_encoded"]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Model training
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Predictions
pred = model.predict(X_test)

# Evaluation
acc = accuracy_score(y_test, pred)
print(f"\nModel Accuracy: {acc:.4f}")

print("\n--- Classification Report ---")
print(classification_report(y_test, pred, zero_division=0))

# Feature importance
importances = model.feature_importances_
plt.figure(figsize=(8, 4))
plt.bar(X.columns, importances, color="#06B6D4")
plt.title("Feature Importance for Skin Disease Prediction", fontsize=12, fontweight="bold")
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig(BASE_DIR / "feature_importance_baseline.png", dpi=150)
print(f"\nFeature importance chart saved to: {BASE_DIR / 'feature_importance_baseline.png'}")
