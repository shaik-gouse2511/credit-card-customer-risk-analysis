
import pandas as pd
import numpy as np

# Load cleaned dataset
df = pd.read_csv("outputs/cleaned_credit_card.csv")

print("Feature Engineering Started")

# 1. Monthly bill columns
bill_columns = [
    "BILL_AMT1", "BILL_AMT2", "BILL_AMT3",
    "BILL_AMT4", "BILL_AMT5", "BILL_AMT6"
]

# 2. Monthly payment columns
payment_columns = [
    "PAY_AMT1", "PAY_AMT2", "PAY_AMT3",
    "PAY_AMT4", "PAY_AMT5", "PAY_AMT6"
]

# 3. Repayment status columns
pay_status_columns = [
    "PAY_0", "PAY_2", "PAY_3",
    "PAY_4", "PAY_5", "PAY_6"
]

# 4. Calculate total bill amount
df["TOTAL_BILL_AMT"] = df[bill_columns].sum(axis=1)

# 5. Calculate total payment amount
df["TOTAL_PAYMENT_AMT"] = df[payment_columns].sum(axis=1)

# 6. Calculate average bill amount
df["AVG_BILL_AMT"] = df[bill_columns].mean(axis=1)

# 7. Calculate average payment amount
df["AVG_PAYMENT_AMT"] = df[payment_columns].mean(axis=1)

# 8. Count delayed payment months
df["DELAYED_MONTHS"] = (
    df[pay_status_columns] > 0
).sum(axis=1)

# 9. Find maximum repayment delay status
df["MAX_DELAY_STATUS"] = df[pay_status_columns].max(axis=1)

# 10. Calculate latest credit utilization
df["CREDIT_UTILIZATION"] = (
    df["BILL_AMT1"] /
    df["LIMIT_BAL"].replace(0, np.nan)
)

# 11. Calculate payment-to-bill ratio
df["PAYMENT_BILL_RATIO"] = (
    df["TOTAL_PAYMENT_AMT"] /
    df["TOTAL_BILL_AMT"].abs().replace(0, np.nan)
)

# Replace infinite and missing engineered values
df.replace([np.inf, -np.inf], np.nan, inplace=True)

df.fillna(0, inplace=True)

# Display new features
new_features = [
    "TOTAL_BILL_AMT",
    "TOTAL_PAYMENT_AMT",
    "AVG_BILL_AMT",
    "AVG_PAYMENT_AMT",
    "DELAYED_MONTHS",
    "MAX_DELAY_STATUS",
    "CREDIT_UTILIZATION",
    "PAYMENT_BILL_RATIO"
]

print("\nNewly Created Features:")
print(df[new_features].head())

print("\nFinal Dataset Shape:", df.shape)

# Save feature-engineered dataset
df.to_csv(
    "outputs/credit_card_features.csv",
    index=False
)

print("\nFeature Engineering Completed Successfully!")
print("Feature-engineered dataset saved!")
