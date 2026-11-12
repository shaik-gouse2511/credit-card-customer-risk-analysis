
import pandas as pd
import numpy as np
import os

# Load dataset
df = pd.read_csv("dataset/UCI_Credit_Card.csv")

print("Original Dataset Shape:", df.shape)

# 1. Remove unnecessary ID column
df.drop(columns=["ID"], inplace=True)

# 2. Rename target column
df.rename(
    columns={
        "default.payment.next.month": "DEFAULT"
    },
    inplace=True
)

# 3. Check missing values
print("\nMissing Values:")
print(df.isnull().sum().sum())

# 4. Check duplicate records
print("\nDuplicate Records:", df.duplicated().sum())

# Remove duplicates if present
df.drop_duplicates(inplace=True)

# 5. Check categorical values
print("\nEducation Categories:")
print(df["EDUCATION"].value_counts().sort_index())

print("\nMarriage Categories:")
print(df["MARRIAGE"].value_counts().sort_index())

# 6. Handle unknown education categories
# 4 represents Others in the original dataset
df["EDUCATION"] = df["EDUCATION"].replace(
    [0, 5, 6], 4
)

# 7. Handle unknown marriage category
# 3 represents Others
df["MARRIAGE"] = df["MARRIAGE"].replace(
    0, 3
)

# 8. Check target values
print("\nDefault Status:")
print(df["DEFAULT"].value_counts())

# 9. Check final dataset
print("\nFinal Dataset Shape:", df.shape)

print("\nPreprocessing Completed Successfully!")

# 10. Save cleaned dataset
os.makedirs("outputs", exist_ok=True)

df.to_csv(
    "outputs/cleaned_credit_card.csv",
    index=False
)

print("\nCleaned dataset saved successfully!")
