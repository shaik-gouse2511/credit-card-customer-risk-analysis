
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load cleaned dataset
df = pd.read_csv("outputs/cleaned_credit_card.csv")

sns.set_theme(style="whitegrid")

print("EDA Started")
print("Dataset Shape:", df.shape)

#Default vs Non-default customers
plt.figure(figsize=(7,5))

sns.countplot(data=df, x="DEFAULT")

plt.title("Default vs Non-Default Customers")
plt.xlabel("Default Status (0 = No, 1 = Yes)")
plt.ylabel("Number of Customers")

plt.savefig("outputs/default_distribution.png")
plt.show()

#Age vs Default

plt.figure(figsize=(9,5))

sns.histplot(
    data=df,
    x="AGE",
    hue="DEFAULT",
    bins=20,
    multiple="stack"
)

plt.title("Age Distribution by Default Status")
plt.xlabel("Customer Age")
plt.ylabel("Number of Customers")

plt.savefig("outputs/age_default.png")
plt.show()

#Credit Limit vs Default

plt.figure(figsize=(8,5))

sns.boxplot(
    data=df,
    x="DEFAULT",
    y="LIMIT_BAL"
)

plt.title("Credit Limit vs Default Status")
plt.xlabel("Default Status")
plt.ylabel("Credit Limit")

plt.savefig("outputs/credit_limit_default.png")
plt.show()

#Repayment behaviour

plt.figure(figsize=(10,5))

sns.countplot(
    data=df,
    x="PAY_0",
    hue="DEFAULT"
)

plt.title("Repayment Status vs Default")
plt.xlabel("Repayment Status")
plt.ylabel("Number of Customers")

plt.savefig("outputs/repayment_behavior.png")
plt.show()

#Monthly bill amount analysis

bill_columns = [
    "BILL_AMT1",
    "BILL_AMT2",
    "BILL_AMT3",
    "BILL_AMT4",
    "BILL_AMT5",
    "BILL_AMT6"
]

df[bill_columns].mean().plot(
    kind="bar",
    figsize=(9,5)
)

plt.title("Average Monthly Bill Amount")
plt.xlabel("Month")
plt.ylabel("Average Bill Amount")

plt.savefig("outputs/monthly_bills.png")
plt.show()

#Monthly payment analysis

payment_columns = [
    "PAY_AMT1",
    "PAY_AMT2",
    "PAY_AMT3",
    "PAY_AMT4",
    "PAY_AMT5",
    "PAY_AMT6"
]

df[payment_columns].mean().plot(
    kind="bar",
    figsize=(9,5)
)

plt.title("Average Monthly Payment Amount")
plt.xlabel("Month")
plt.ylabel("Average Payment")

plt.savefig("outputs/monthly_payments.png")
plt.show()

print("EDA Completed Successfully!")