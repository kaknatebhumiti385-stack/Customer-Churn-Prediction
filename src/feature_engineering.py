
import pandas as pd

# Load cleaned dataset
df = pd.read_csv(
    "data/processed/cleaned_customer_churn.csv"
)

# Remove customerID because it is not useful
# for predicting customer churn
df.drop("customerID", axis=1, inplace=True)

# Convert target column into numerical values
df["Churn"] = df["Churn"].map({
    "No": 0,
    "Yes": 1
})

# Convert categorical columns into numerical columns
df = pd.get_dummies(
    df,
    columns=df.select_dtypes(
        include=["object"]
    ).columns,
    drop_first=True,
    dtype=int
)

# Display processed dataset
print("\n--- FEATURE ENGINEERING COMPLETED ---")
print(df.head())

print("\nDataset Shape:", df.shape)

print("\nData Types:")
print(df.dtypes)

# Save processed dataset
df.to_csv(
    "data/processed/final_customer_churn.csv",
    index=False
)

print("\nFinal dataset saved successfully!")
