
import pandas as pd
import joblib
from pathlib import Path

# Project folder
BASE_DIR = Path(__file__).resolve().parent.parent

# Load trained model and scaler
model = joblib.load(BASE_DIR / "models" / "logistic_regression.pkl")
scaler = joblib.load(BASE_DIR / "models" / "scaler.pkl")

# Load training dataset to get the exact feature columns
data = pd.read_csv(
    BASE_DIR / "data" / "processed" / "final_customer_churn.csv"
)

X = data.drop("Churn", axis=1)

print("\n--- CUSTOMER CHURN PREDICTION SYSTEM ---")
print("Enter the customer's details below.\n")

customer = {
    "gender": input("Gender (Male/Female): "),
    "SeniorCitizen": int(input("SeniorCitizen (0 or 1): ")),
    "Partner": input("Partner (Yes/No): "),
    "Dependents": input("Dependents (Yes/No): "),
    "tenure": int(input("Tenure (months): ")),
    "PhoneService": input("PhoneService (Yes/No): "),
    "MultipleLines": input(
        "MultipleLines (Yes/No/No phone service): "
    ),
    "InternetService": input("InternetService (DSL/Fiber optic/No): "),
    "OnlineSecurity": input(
        "OnlineSecurity (Yes/No/No internet service): "
    ),
    "OnlineBackup": input(
        "OnlineBackup (Yes/No/No internet service): "
    ),
    "DeviceProtection": input(
        "DeviceProtection (Yes/No/No internet service): "
    ),
    "TechSupport": input(
        "TechSupport (Yes/No/No internet service): "
    ),
    "StreamingTV": input(
        "StreamingTV (Yes/No/No internet service): "
    ),
    "StreamingMovies": input(
        "StreamingMovies (Yes/No/No internet service): "
    ),
    "Contract": input("Contract (Month-to-month/One year/Two year): "),
    "PaperlessBilling": input("PaperlessBilling (Yes/No): "),
    "PaymentMethod": input(
        "PaymentMethod (Electronic check/Mailed check/"
        "Bank transfer (automatic)/Credit card (automatic)): "
    ),
    "MonthlyCharges": float(input("MonthlyCharges: ")),
    "TotalCharges": float(input("TotalCharges: "))
}

# Convert customer details into the same encoded format
customer_df = pd.DataFrame([customer])

customer_encoded = pd.get_dummies(
    customer_df,
    drop_first=True,
    dtype=int
)

# Match the exact training columns
customer_encoded = customer_encoded.reindex(
    columns=X.columns,
    fill_value=0
)

# Apply the saved scaler
customer_scaled = scaler.transform(customer_encoded)

# Make prediction
prediction = model.predict(customer_scaled)
probability = model.predict_proba(customer_scaled)[0][1]

print("\n--- PREDICTION RESULT ---")

if prediction[0] == 1:
    print("Customer is likely to CHURN.")
else:
    print("Customer is likely to STAY.")

print(f"Churn probability: {probability * 100:.2f}%")
