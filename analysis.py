
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

# Load dataset
df = pd.read_csv("data/skin_conditions_sample.csv")

print("Dataset Preview:")
print(df.head())

# Basic statistics
print("\nSummary Statistics:")
print(df.describe())

# Encode target variable
le = LabelEncoder()
df["diagnosis_encoded"] = le.fit_transform(df["diagnosis"])

X = df.drop(columns=["diagnosis","diagnosis_encoded"])
y = df["diagnosis_encoded"]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2,random_state=42)

# Model training
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Predictions
pred = model.predict(X_test)

# Evaluation
acc = accuracy_score(y_test, pred)
print("\nModel Accuracy:", acc)

print("\nClassification Report:")
print(classification_report(y_test, pred))

# Feature importance
importances = model.feature_importances_
plt.bar(X.columns, importances)
plt.title("Feature Importance for Skin Disease Prediction")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()
