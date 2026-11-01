
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Load cleaned dataset
df = pd.read_csv(
    "data/processed/cleaned_customer_churn.csv"
)

# Create reports folder
os.makedirs("reports", exist_ok=True)

sns.set_style("whitegrid")

# Create 2x2 visualization layout
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# 1. Customer Churn Count
sns.countplot(data=df, x="Churn", ax=axes[0, 0])
axes[0, 0].set_title("Customer Churn Count")
axes[0, 0].set_xlabel("Churn")
axes[0, 0].set_ylabel("Number of Customers")

# 2. Churn by Contract Type
contract_churn = (
    df.groupby("Contract")["Churn"]
    .apply(lambda x: (x == "Yes").mean() * 100)
    .reset_index(name="Churn_Percentage")
)

sns.barplot(
    data=contract_churn,
    x="Contract",
    y="Churn_Percentage",
    ax=axes[0, 1]
)

axes[0, 1].set_title("Churn by Contract Type")
axes[0, 1].set_ylabel("Churn Percentage (%)")

# 3. Monthly Charges vs Churn
sns.boxplot(
    data=df,
    x="Churn",
    y="MonthlyCharges",
    ax=axes[1, 0]
)

axes[1, 0].set_title("Monthly Charges vs Churn")

# 4. Customer Tenure vs Churn
sns.histplot(
    data=df,
    x="tenure",
    hue="Churn",
    bins=20,
    multiple="stack",
    ax=axes[1, 1]
)

axes[1, 1].set_title("Customer Tenure vs Churn")
axes[1, 1].set_xlabel("Tenure (Months)")

plt.tight_layout()

# Save visualization
plt.savefig(
    "reports/customer_churn_analysis.png",
    dpi=300
)

plt.show()

print("VISUALIZATION COMPLETED SUCCESSFULLY!")
print("Graph saved in reports folder.")
