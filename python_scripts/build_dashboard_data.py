import pandas as pd
import numpy as np
import json
import os

print("Building dashboard data...")

# 1. Load predictions
df_preds = pd.read_csv("outputs/customer_risk_predictions.csv")
print(f"Loaded {len(df_preds)} customer risk predictions.")

# 2. Overall KPIs
total_customers = len(df_preds)
low_risk_count = int((df_preds['RISK_LEVEL'] == 'Low Risk').sum())
med_risk_count = int((df_preds['RISK_LEVEL'] == 'Medium Risk').sum())
high_risk_count = int((df_preds['RISK_LEVEL'] == 'High Risk').sum())

actual_defaults = int(df_preds['ACTUAL_DEFAULT'].sum())
actual_default_rate = round(actual_defaults / total_customers * 100, 2)
predicted_defaults = int(df_preds['PREDICTED_DEFAULT'].sum())
predicted_default_rate = round(predicted_defaults / total_customers * 100, 2)
avg_probability = round(df_preds['DEFAULT_PROBABILITY'].mean() * 100, 2)
avg_credit_limit = round(df_preds['LIMIT_BAL'].mean(), 0)
avg_utilization = round(df_preds['CREDIT_UTILIZATION'].mean() * 100, 1)

kpis = {
    "total_customers": total_customers,
    "low_risk_count": low_risk_count,
    "low_risk_pct": round(low_risk_count / total_customers * 100, 1),
    "med_risk_count": med_risk_count,
    "med_risk_pct": round(med_risk_count / total_customers * 100, 1),
    "high_risk_count": high_risk_count,
    "high_risk_pct": round(high_risk_count / total_customers * 100, 1),
    "actual_defaults": actual_defaults,
    "actual_default_rate": actual_default_rate,
    "predicted_defaults": predicted_defaults,
    "predicted_default_rate": predicted_default_rate,
    "avg_probability": avg_probability,
    "avg_credit_limit": avg_credit_limit,
    "avg_utilization": avg_utilization
}

# 3. Model comparison benchmarks
df_models = pd.read_csv("outputs/model_comparison.csv")
model_comparison = df_models.to_dict(orient="records")

# 4. Age groups aggregation
df_preds['AGE_GROUP'] = pd.cut(
    df_preds['AGE'],
    bins=[19, 29, 39, 49, 59, 100],
    labels=['20-29', '30-39', '40-49', '50-59', '60+']
)
age_group_stats = []
for grp, sub in df_preds.groupby('AGE_GROUP', observed=False):
    age_group_stats.append({
        "group": grp,
        "count": int(len(sub)),
        "avg_probability": round(sub['DEFAULT_PROBABILITY'].mean() * 100, 1),
        "actual_default_rate": round(sub['ACTUAL_DEFAULT'].mean() * 100, 1),
        "high_risk_pct": round((sub['RISK_LEVEL'] == 'High Risk').mean() * 100, 1)
    })

# 5. Education level aggregation
edu_labels = {1: "Graduate School", 2: "University", 3: "High School", 4: "Others"}
edu_stats = []
for edu_code, label in edu_labels.items():
    sub = df_preds[df_preds['EDUCATION'] == edu_code]
    if len(sub) > 0:
        edu_stats.append({
            "code": edu_code,
            "label": label,
            "count": int(len(sub)),
            "avg_probability": round(sub['DEFAULT_PROBABILITY'].mean() * 100, 1),
            "actual_default_rate": round(sub['ACTUAL_DEFAULT'].mean() * 100, 1),
            "avg_credit_limit": round(sub['LIMIT_BAL'].mean(), 0)
        })

# 6. Repayment status (PAY_0) aggregation
pay0_labels = {
    -2: "No Activity (-2)",
    -1: "Paid In Full (-1)",
    0: "Revolving / On Time (0)",
    1: "Delayed 1 Mo (1)",
    2: "Delayed 2 Mo (2)",
    3: "Delayed 3+ Mo (3+)"
}
df_preds['PAY_0_GROUP'] = df_preds['PAY_0'].apply(lambda x: 3 if x >= 3 else x)
repayment_stats = []
for p_val, label in pay0_labels.items():
    sub = df_preds[df_preds['PAY_0_GROUP'] == p_val]
    if len(sub) > 0:
        repayment_stats.append({
            "pay_code": p_val,
            "label": label,
            "count": int(len(sub)),
            "avg_probability": round(sub['DEFAULT_PROBABILITY'].mean() * 100, 1),
            "actual_default_rate": round(sub['ACTUAL_DEFAULT'].mean() * 100, 1)
        })

# 7. Monthly trend (Apr to Sep)
months = ["Apr (M6)", "May (M5)", "Jun (M4)", "Jul (M3)", "Aug (M2)", "Sep (M1)"]
bill_cols = ["BILL_AMT6", "BILL_AMT5", "BILL_AMT4", "BILL_AMT3", "BILL_AMT2", "BILL_AMT1"]
pay_cols = ["PAY_AMT6", "PAY_AMT5", "PAY_AMT4", "PAY_AMT3", "PAY_AMT2", "PAY_AMT1"]

monthly_trends = []
for i in range(6):
    monthly_trends.append({
        "month": months[i],
        "avg_bill": round(df_preds[bill_cols[i]].mean(), 0),
        "avg_payment": round(df_preds[pay_cols[i]].mean(), 0)
    })

# 8. Calibrated scoring weights for simulator (using logit regression for high accuracy)
features = [
    'LIMIT_BAL', 'AGE', 'EDUCATION', 'MARRIAGE', 'PAY_0', 'PAY_2', 'PAY_3', 'PAY_4', 'PAY_5', 'PAY_6', 
    'BILL_AMT1', 'PAY_AMT1', 'TOTAL_BILL_AMT', 'TOTAL_PAYMENT_AMT', 'DELAYED_MONTHS', 'MAX_DELAY_STATUS', 
    'CREDIT_UTILIZATION', 'PAYMENT_BILL_RATIO'
]
X = df_preds[features].copy()
X['CONST'] = 1.0
p = np.clip(df_preds['DEFAULT_PROBABILITY'].values, 0.01, 0.99)
logit_y = np.log(p / (1 - p))
w, _, _, _ = np.linalg.lstsq(X.values, logit_y, rcond=None)
weights_dict = {feat: float(weight) for feat, weight in zip(features + ['CONST'], w)}

# 9. Compact Customer Records for interactive table and search (5,993 records)
customers = []
for _, r in df_preds.iterrows():
    customers.append({
        "id": int(r['CUSTOMER_ID']),
        "bal": int(r['LIMIT_BAL']),
        "sex": int(r['SEX']),
        "edu": int(r['EDUCATION']),
        "mar": int(r['MARRIAGE']),
        "age": int(r['AGE']),
        "p0": int(r['PAY_0']),
        "p2": int(r['PAY_2']),
        "p3": int(r['PAY_3']),
        "p4": int(r['PAY_4']),
        "p5": int(r['PAY_5']),
        "p6": int(r['PAY_6']),
        "b1": int(r['BILL_AMT1']),
        "b2": int(r['BILL_AMT2']),
        "b3": int(r['BILL_AMT3']),
        "b4": int(r['BILL_AMT4']),
        "b5": int(r['BILL_AMT5']),
        "b6": int(r['BILL_AMT6']),
        "pm1": int(r['PAY_AMT1']),
        "pm2": int(r['PAY_AMT2']),
        "pm3": int(r['PAY_AMT3']),
        "pm4": int(r['PAY_AMT4']),
        "pm5": int(r['PAY_AMT5']),
        "pm6": int(r['PAY_AMT6']),
        "tot_b": round(float(r['TOTAL_BILL_AMT']), 0),
        "tot_p": round(float(r['TOTAL_PAYMENT_AMT']), 0),
        "del_m": int(r['DELAYED_MONTHS']),
        "max_del": int(r['MAX_DELAY_STATUS']),
        "util": round(float(r['CREDIT_UTILIZATION']), 3),
        "pb_rat": round(float(r['PAYMENT_BILL_RATIO']), 3),
        "act": int(r['ACTUAL_DEFAULT']),
        "pred": int(r['PREDICTED_DEFAULT']),
        "prob": round(float(r['DEFAULT_PROBABILITY']), 3),
        "risk": r['RISK_LEVEL']
    })

dashboard_data = {
    "kpis": kpis,
    "model_comparison": model_comparison,
    "age_group_stats": age_group_stats,
    "edu_stats": edu_stats,
    "repayment_stats": repayment_stats,
    "monthly_trends": monthly_trends,
    "simulator_weights": weights_dict,
    "customers": customers
}

# Save as JSON
json_path = "outputs/dashboard_data.json"
with open(json_path, "w", encoding="utf-8") as f:
    json.dump(dashboard_data, f)
print(f"Saved {json_path} ({os.path.getsize(json_path) / 1024:.1f} KB)")

# Save as JS variable for direct file:// browsing without CORS issues
js_path = "outputs/dashboard_data.js"
with open(js_path, "w", encoding="utf-8") as f:
    f.write("window.DASHBOARD_DATA = ")
    json.dump(dashboard_data, f)
    f.write(";\n")
print(f"Saved {js_path} ({os.path.getsize(js_path) / 1024:.1f} KB)")

print("Dashboard data built successfully!")

