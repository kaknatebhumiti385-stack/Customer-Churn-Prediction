
import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)

# Load dataset
df = pd.read_csv(
    "data/processed/final_customer_churn.csv"
)

# Separate features and target
X = df.drop("Churn", axis=1)
y = df["Churn"]

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# Load trained models
log_model = joblib.load(
    "models/logistic_regression.pkl"
)

rf_model = joblib.load(
    "models/random_forest.pkl"
)

# Scale data for Logistic Regression
scaler = joblib.load("models/scaler.pkl")

X_test_scaled = scaler.transform(X_test)

# Predictions
log_pred = log_model.predict(X_test_scaled)
rf_pred = rf_model.predict(X_test)

# Evaluation function
def evaluate_model(name, y_true, predictions):
    return {
        "Model": name,
        "Accuracy": accuracy_score(y_true, predictions),
        "Precision": precision_score(y_true, predictions),
        "Recall": recall_score(y_true, predictions),
        "F1 Score": f1_score(y_true, predictions)
    }

# Compare models
results = [
    evaluate_model("Logistic Regression", y_test, log_pred),
    evaluate_model("Random Forest", y_test, rf_pred)
]

comparison = pd.DataFrame(results)

print("\n--- MODEL PERFORMANCE COMPARISON ---")
print(comparison.to_string(index=False))

# Save comparison
comparison.to_csv(
    "reports/model_comparison.csv",
    index=False
)

print("\nModel comparison saved successfully!")
