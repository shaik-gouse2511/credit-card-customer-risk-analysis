import pandas as pd
import numpy as np
import os

print("Generating Power BI ready CSV dataset...")

# Load prediction dataset
df = pd.read_csv("outputs/customer_risk_predictions.csv")

# Create clean, human-readable categorical labels
gender_map = {1: "Male", 2: "Female"}
edu_map = {1: "Graduate School", 2: "University", 3: "High School", 4: "Others"}
mar_map = {1: "Married", 2: "Single", 3: "Others"}

def map_pay_status(p):
    if p <= -1:
        return "Paid in Full"
    elif p == 0:
        return "On Time"
    elif p == 1:
        return "Delayed 1 Month"
    elif p == 2:
        return "Delayed 2 Months"
    else:
        return "Delayed 3+ Months"

power_bi_df = pd.DataFrame()

power_bi_df["Customer_ID"] = df["CUSTOMER_ID"].astype(int)
power_bi_df["Credit_Limit"] = df["LIMIT_BAL"].astype(int)
power_bi_df["Gender"] = df["SEX"].map(gender_map).fillna("Others")
power_bi_df["Education"] = df["EDUCATION"].map(edu_map).fillna("Others")
power_bi_df["Marital_Status"] = df["MARRIAGE"].map(mar_map).fillna("Others")
power_bi_df["Age"] = df["AGE"].astype(int)
power_bi_df["Age_Group"] = pd.cut(
    df["AGE"],
    bins=[19, 29, 39, 49, 59, 100],
    labels=["20-29", "30-39", "40-49", "50-59", "60+"]
)

power_bi_df["Payment_Status"] = df["PAY_0"].apply(map_pay_status)
power_bi_df["Delayed_Months"] = df["DELAYED_MONTHS"].astype(int)
power_bi_df["Total_Billed"] = df["TOTAL_BILL_AMT"].round(0).astype(int)
power_bi_df["Total_Paid"] = df["TOTAL_PAYMENT_AMT"].round(0).astype(int)
power_bi_df["Credit_Utilization"] = df["CREDIT_UTILIZATION"].round(3)
power_bi_df["Credit_Utilization_Pct"] = (df["CREDIT_UTILIZATION"] * 100).round(1)

power_bi_df["Default_Probability"] = df["DEFAULT_PROBABILITY"].round(3)
power_bi_df["Default_Probability_Pct"] = (df["DEFAULT_PROBABILITY"] * 100).round(1)
power_bi_df["Risk_Level"] = df["RISK_LEVEL"]

power_bi_df["Is_Default"] = df["ACTUAL_DEFAULT"].astype(int)
power_bi_df["Actual_Default"] = df["ACTUAL_DEFAULT"].map({1: "Defaulted", 0: "Paid on Time"})
power_bi_df["Predicted_Default"] = df["PREDICTED_DEFAULT"].map({1: "Defaulted", 0: "Paid on Time"})

# Save Power BI Ready CSV
output_path = "outputs/power_bi_credit_card_risk.csv"
power_bi_df.to_csv(output_path, index=False)
print(f"Power BI dataset saved to {output_path} ({len(power_bi_df)} rows).")

# Also generate summary statistics for Power BI reference
print("\nPower BI Key Figures:")
print(f"Total Customers: {len(power_bi_df):,}")
print(f"Default Rate: {(power_bi_df['Is_Default'].mean() * 100):.1f}%")
print(f"High Risk Count: {(power_bi_df['Risk_Level'] == 'High Risk').sum():,} ({(power_bi_df['Risk_Level'] == 'High Risk').mean() * 100:.1f}%)")
print(f"Avg Credit Limit: NT$ {power_bi_df['Credit_Limit'].mean():,.0f}")

