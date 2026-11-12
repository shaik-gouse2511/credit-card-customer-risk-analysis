# Credit Card Customer Risk Analysis & Interactive Intelligence Dashboard

An end-to-end Financial Analytics & Machine Learning system to predict customer credit card default risk, segment borrowers by delinquency exposure, and provide institutional underwriting recommendations.

---

## 📌 Project Overview
* **Domain**: Financial Analytics & Decision-Driven Credit Risk Management (FADD)
* **Dataset**: UCI Default of Credit Card Clients Dataset (30,000 customers in Taiwan)
* **Objective**: Accurately predict whether a customer will default on credit card payments next month (`DEFAULT = 1`) and classify accounts into actionable risk tiers (**Low Risk**, **Medium Risk**, **High Risk**).

---

## 🏆 Model Performance Benchmark

Four supervised machine learning models were trained and evaluated on stratified test partitions:

| Machine Learning Model | Accuracy | Precision | Recall | F1-Score | ROC-AUC | Verdict |
| :--- | :---: | :---: | :---: | :---: | :---: | :--- |
| **Random Forest** | **76.87%** | **48.15%** | 58.75% | **52.92%** | **77.40%** | 🏆 **Champion Production Model** |
| **Decision Tree** | 75.30% | 45.62% | 60.48% | 52.01% | 76.05% | High recall, prone to variance |
| **SVM (Support Vector Machine)** | 74.92% | 45.03% | 60.41% | 51.59% | 75.14% | Strong margin, high compute |
| **Logistic Regression** | 74.42% | 44.30% | **60.71%** | 51.22% | 75.26% | Linear baseline model |

> **Key Finding**: Random Forest with balanced class weights achieved the highest Area Under the ROC Curve (**77.40%**), effectively separating high-risk defaulters while minimizing false alarms.

---

## 🔬 Feature Engineering Highlights
The following engineered features demonstrated the strongest predictive power:
* `DELAYED_MONTHS`: Number of months payment was delayed across 6 months (+0.782 correlation with default probability).
* `MAX_DELAY_STATUS`: Worst repayment delay status in recent history.
* `CREDIT_UTILIZATION`: Latest billed amount relative to credit limit (`BILL_AMT1 / LIMIT_BAL`).
* `PAYMENT_BILL_RATIO`: Total payments made relative to total billed amount.

---

## 📊 Interactive Dashboard & Power BI Integration

The project includes an interactive web dashboard and a clean Power BI-ready dataset:

1. **Web Dashboard (`index.html` / `dashboard.html`)**:
   * **Top KPI Cards**: Total Customers (`5,993`), Default Rate (`22.1%`), High Risk Accounts (`20.3%`), and Avg Credit Limit (`NT$ 168K`).
   * **Visuals**: Risk Tier Donut Chart, Delinquency vs Default Rate Bar Chart, and Age Group Column Chart.
   * **Customer Explorer**: Real-time multi-filter table with customer search and CSV export.
   * **Built-in Power BI Guide**: Click the top toggle to see exact DAX formulas and field mappings.

2. **Power BI Ready Dataset (`outputs/power_bi_credit_card_risk.csv`)**:
   * Pre-cleaned with plain English labels (`Graduate School`, `Delayed 2 Months`, `Male`/`Female`).
   * Import directly into Power BI Desktop via **Get Data &rarr; Text/CSV** with zero custom transformations.

---

## 🚀 How to Run Locally

### 1. View Dashboard in Browser
Double-click `index.html` in Windows Explorer, or run:
```powershell
python run_dashboard.py
```
This serves the dashboard locally at `http://localhost:8501/dashboard.html`.

### 2. Re-train Models & Run Pipelines
```powershell
# 1. Preprocess raw data
python main.py

# 2. Engineer features
python feature_engineering.py

# 3. Exploratory Data Analysis
python eda.py

# 4. Train Random Forest & Generate Predictions
python risk_prediction.py

# 5. Model comparison benchmarks
python model_comparison.py
```

---

## 📁 Repository Structure

```text
├── index.html                           # Production interactive dashboard
├── dashboard.html                       # Dashboard template
├── run_dashboard.py                     # Local web server launcher
├── prepare_power_bi_data.py             # Generates clean Power BI CSV
├── build_dashboard_data.py              # Compiles client-side dataset
├── requirements.txt                     # Python dependencies
├── main.py                              # Data cleaning and preprocessing
├── feature_engineering.py               # Feature derivation pipeline
├── eda.py                               # Exploratory data visualizations
├── random_forest.py                     # Random Forest training & evaluation
├── decision_tree.py                     # Decision Tree implementation
├── logistic_regression.py               # Logistic Regression baseline
├── svm.py                               # Support Vector Machine implementation
├── model_comparison.py                  # Multi-model evaluation script
├── risk_prediction.py                   # Customer risk classification
├── dataset/
│   └── UCI_Credit_Card.csv              # Raw UCI dataset (30,000 records)
├── models/
│   └── random_forest_model.pkl          # Saved trained Random Forest model
├── outputs/
│   ├── power_bi_credit_card_risk.csv    # Clean dataset for Power BI
│   ├── customer_risk_predictions.csv    # Evaluated predictions & probabilities
│   ├── credit_card_features.csv         # Feature-engineered dataset
│   ├── cleaned_credit_card.csv          # Cleaned dataset
│   ├── model_comparison.csv             # Metrics table
│   └── *.png                            # Visualizations
└── web_deploy/                          # Standalone package for Netlify/GitHub Pages
```

