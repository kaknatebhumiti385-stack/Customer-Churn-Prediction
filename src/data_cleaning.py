
import pandas as pd
import os

# Load dataset
df = pd.read_csv("data/raw/Telco-Customer-Churn.csv")

# Convert TotalCharges into numeric
df["TotalCharges"] = pd.to_numeric(
    df["TotalCharges"], errors="coerce"
)

# Remove duplicate records
df.drop_duplicates(inplace=True)

# Remove duplicate customer IDs
df.drop_duplicates(subset="customerID", inplace=True)

# Check missing TotalCharges
missing = df["TotalCharges"].isnull()

print("Missing TotalCharges:", missing.sum())

# Fill missing charges with 0 only for new customers
if missing.any():
    if (df.loc[missing, "tenure"] == 0).all():
        df.loc[missing, "TotalCharges"] = 0
    else:
        raise ValueError(
            "Check missing TotalCharges before filling."
        )

# Check remaining missing values
print("\nRemaining Missing Values:")
print(df.isnull().sum())

# Save cleaned dataset
os.makedirs("data/processed", exist_ok=True)

df.to_csv(
    "data/processed/cleaned_customer_churn.csv",
    index=False
)

print("\nFinal Dataset Shape:", df.shape)
print("\nDATA CLEANING COMPLETED SUCCESSFULLY!")

