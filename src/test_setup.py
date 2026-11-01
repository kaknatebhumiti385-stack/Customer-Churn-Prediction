
import pandas as pd
import numpy as np
import sklearn
import matplotlib
import seaborn
import streamlit
import joblib

print("====================================")
print(" SAIKET SYSTEMS - MACHINE LEARNING ")
print("====================================")

print("\nAll libraries imported successfully!")

print("\nLibrary Versions:")
print("Pandas:", pd.__version__)
print("NumPy:", np.__version__)
print("Scikit-learn:", sklearn.__version__)

# Load customer churn dataset
file_path = "data/raw/Telco-Customer-Churn.csv"

df = pd.read_csv(file_path)

print("\nDataset loaded successfully!")

# Display dataset information
print("\nDataset Shape:")
print(df.shape)

print("\nFirst Five Records:")
print(df.head())

print("\nColumn Names:")
print(df.columns.tolist())

print("\nMissing Values:")
print(df.isnull().sum())

print("\nChurn Distribution:")
print(df["Churn"].value_counts())

print("\n====================================")
print(" PROJECT SETUP COMPLETED SUCCESSFULLY!")
print("====================================")
