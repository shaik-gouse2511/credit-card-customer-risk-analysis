# Credit Card Customer Risk Analysis - Streamlit Dashboard

An interactive, executive-grade Streamlit web application and Machine Learning risk intelligence dashboard for credit card default risk assessment (UCI Credit Card Dataset, 30,000 accounts).

---

## 🚀 Quick Start: How to Run Locally

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Launch the Streamlit Dashboard
```bash
streamlit run app.py
```
*(or run `python main.py`)*

The dashboard will automatically open in your default browser at `http://localhost:8501`.

---

## ☁️ 1-Click Free Deployment on Streamlit Cloud

You can deploy this dashboard online for free in 1 minute using **Streamlit Community Cloud**:

1. Go to **[https://share.streamlit.io](https://share.streamlit.io)** and sign in with your GitHub account.
2. Click **"New app"**.
3. Fill in:
   * **Repository**: `shaik-gouse2511/credit-card-customer-risk-analysis`
   * **Branch**: `main`
   * **Main file path**: `app.py`
4. Click **"Deploy!"**.
5. Your dashboard will be live with a permanent URL:
   `https://credit-card-customer-risk-analysis.streamlit.app`

---

## 🖥️ Streamlit Dashboard Features

1. **📊 Executive Visual Analytics**:
   * **KPIs**: Total Evaluated Accounts (5,993), Default Rate (22.1%), High Risk Share (20.3%), and Avg Credit Limit (NT$ 168K).
   * **Donut Chart**: Risk Tier Breakdown (Low Risk 40.6%, Medium Risk 39.1%, High Risk 20.3%).
   * **Delinquency Analysis**: Explores how 2-month repayment delays surge default rate to 68.3%.
   * **Age Group Demographics**: Cohort volume and default rate breakdown.

2. **🔍 Customer Risk Explorer & Profiler**:
   * Sidebar multi-filters (Risk Tier, Education, Recent Delinquency, Customer ID search).
   * Searchable interactive dataframe with CSV export.
   * Single-customer dossier inspector with automated credit recommendations.

3. **🧮 Live "What-If" Credit Risk Simulator**:
   * Real-time sliders for Credit Limit, Age, Recent Delay, Past Delayed Months, Bill, and Payment.
   * Instant Risk Score Meter (0% to 100%), color-coded risk badge, and automated underwriting action (Approve / Monitor / Decline).
   * 1-click preset scenarios (Prime Borrower, Average Revolver, Stressed Delinquent).

4. **🤖 Supervised ML Model Benchmarks**:
   * Evaluates Random Forest (**77.4% ROC-AUC Champion**), Decision Tree, SVM, and Logistic Regression.
   * Interactive grouped bar chart comparing Accuracy, Precision, Recall, F1-Score, and ROC-AUC.

5. **📘 Power BI Integration Guide**:
   * Step-by-step guide on how to import `outputs/power_bi_credit_card_risk.csv` into Power BI Desktop.
   * Ready-to-use DAX measures and field drag-and-drop mappings.

---

## 📁 Repository Structure

```text
├── app.py                               # Main Streamlit web application
├── main.py                              # Streamlit launcher script
├── requirements.txt                     # Dependencies (streamlit, pandas, numpy, plotly)
├── README.md                            # Documentation and deployment guide
├── outputs/
│   ├── power_bi_credit_card_risk.csv    # Clean dataset for Power BI & Streamlit
│   ├── customer_risk_predictions.csv    # Evaluated test predictions & probabilities
│   ├── credit_card_features.csv         # Feature-engineered dataset
│   ├── cleaned_credit_card.csv          # Cleaned dataset
│   └── model_comparison.csv             # ML benchmark metrics table
├── dataset/
│   └── UCI_Credit_Card.csv              # Raw UCI dataset (30,000 records)
├── models/
│   └── random_forest_model.pkl          # Trained Random Forest model
├── python_scripts/                      # Model training & preprocessing scripts
│   ├── main.py
│   ├── feature_engineering.py
│   ├── eda.py
│   ├── random_forest.py
│   ├── decision_tree.py
│   ├── logistic_regression.py
│   ├── svm.py
│   └── risk_prediction.py
└── index.html                           # Standalone HTML dashboard alternative
```
