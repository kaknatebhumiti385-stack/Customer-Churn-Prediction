
import pandas as pd

# Load dataset
df = pd.read_csv("data/raw/Telco-Customer-Churn.csv")

# Display first 5 rows
print("\n--- FIRST 5 ROWS ---")
print(df.head())

# Dataset shape
print("\n--- DATASET SHAPE ---")
print(df.shape)

# Column information
print("\n--- DATASET INFORMATION ---")
print(df.info())

# Missing values
print("\n--- MISSING VALUES ---")
print(df.isnull().sum())

# Duplicate records
print("\n--- DUPLICATE RECORDS ---")
print(df.duplicated().sum())

# Statistical summary
print("\n--- STATISTICAL SUMMARY ---")
print(df.describe())

# Customer Churn Analysis

print("\n--- CUSTOMER CHURN COUNT ---")
print(df["Churn"].value_counts())

print("\n--- CUSTOMER CHURN PERCENTAGE ---")
print(df["Churn"].value_counts(normalize=True) * 100)

print("\n--- UNIQUE VALUES IN EACH COLUMN ---")
print(df.nunique())

print("\n--- DATA TYPES ---")
print(df.dtypes)

print("\n--- CHURN ANALYSIS COMPLETED ---")
