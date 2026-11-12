"""
Credit Card Customer Risk Analysis - Streamlit Web Application
================================================================
An interactive, easy-to-understand Streamlit dashboard for credit risk assessment,
customer exploration, 'what-if' simulation, and ML model comparisons.
"""

import streamlit as st
import pandas as pd
import numpy as np
import os
import json

# --- 1. Page Configuration ---
st.set_page_config(
    page_title="Credit Card Risk Dashboard",
    page_icon="💳",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for styling
st.markdown("""
<style>
    .metric-card {
        background-color: #ffffff;
        border: 1px solid #e2e8f0;
        border-radius: 8px;
        padding: 16px;
        box-shadow: 0 1px 3px rgba(0,0,0,0.05);
    }
    .badge-low {
        background-color: #ecfdf5;
        color: #047857;
        padding: 3px 8px;
        border-radius: 4px;
        font-weight: 600;
        font-size: 0.8rem;
    }
    .badge-med {
        background-color: #fffbeb;
        color: #b45309;
        padding: 3px 8px;
        border-radius: 4px;
        font-weight: 600;
        font-size: 0.8rem;
    }
    .badge-high {
        background-color: #fff1f2;
        color: #be123c;
        padding: 3px 8px;
        border-radius: 4px;
        font-weight: 600;
        font-size: 0.8rem;
    }
</style>
""", unsafe_allow_html=True)


# --- 2. Data Loading & Caching ---
@st.cache_data
def load_data():
    csv_path = "outputs/power_bi_credit_card_risk.csv"
    if not os.path.exists(csv_path):
        csv_path = "outputs/customer_risk_predictions.csv"
    
    df = pd.read_csv(csv_path)
    
    # Ensure clean column names
    if "CUSTOMER_ID" in df.columns:
        df.rename(columns={
            "CUSTOMER_ID": "Customer_ID",
            "LIMIT_BAL": "Credit_Limit",
            "DELAYED_MONTHS": "Delayed_Months",
            "DEFAULT_PROBABILITY": "Default_Probability",
            "RISK_LEVEL": "Risk_Level",
            "ACTUAL_DEFAULT": "Is_Default"
        }, inplace=True)
    
    # Precomputed stats for ML comparison
    model_comparison = pd.DataFrame({
        "Model": ["Random Forest", "Decision Tree", "SVM", "Logistic Regression"],
        "Accuracy": [76.87, 75.30, 74.92, 74.42],
        "Precision": [48.15, 45.62, 45.03, 44.30],
        "Recall": [58.75, 60.48, 60.41, 60.71],
        "F1_Score": [52.92, 52.01, 51.59, 51.22],
        "ROC_AUC": [77.40, 76.05, 75.14, 75.26]
    })
    
    return df, model_comparison

df, model_comparison = load_data()


# --- 3. Sidebar Slicers (Filters) ---
st.sidebar.title("💳 Filter Dashboard")
st.sidebar.markdown("Use these slicers to filter customer records across all charts and tables.")

# Filter 1: Risk Level
risk_options = ["All"] + sorted(list(df["Risk_Level"].unique()))
selected_risk = st.sidebar.selectbox("Risk Level Tier", risk_options, index=0)

# Filter 2: Education
if "Education" in df.columns:
    edu_options = ["All"] + sorted(list(df["Education"].unique()))
    selected_edu = st.sidebar.selectbox("Education Level", edu_options, index=0)
else:
    selected_edu = "All"

# Filter 3: Payment Status
if "Payment_Status" in df.columns:
    pay_options = ["All"] + list(df["Payment_Status"].unique())
    selected_pay = st.sidebar.selectbox("Recent Payment Status", pay_options, index=0)
else:
    selected_pay = "All"

# Filter 4: Customer Search
search_id = st.sidebar.text_input("🔍 Search Customer ID", placeholder="e.g. 8682")

# Apply Filters
filtered_df = df.copy()
if selected_risk != "All":
    filtered_df = filtered_df[filtered_df["Risk_Level"] == selected_risk]
if selected_edu != "All":
    filtered_df = filtered_df[filtered_df["Education"] == selected_edu]
if selected_pay != "All":
    filtered_df = filtered_df[filtered_df["Payment_Status"] == selected_pay]
if search_id.strip():
    filtered_df = filtered_df[filtered_df["Customer_ID"].astype(str).str.contains(search_id.strip())]

st.sidebar.markdown(f"**Filtered Accounts**: `{len(filtered_df):,}` of `{len(df):,}`")

# Sidebar Download Button
csv_data = filtered_df.to_csv(index=False).encode('utf-8')
st.sidebar.download_button(
    label="📥 Download Filtered CSV",
    data=csv_data,
    file_name="filtered_credit_card_risk.csv",
    mime="text/csv"
)

st.sidebar.markdown("---")
st.sidebar.caption("Project: Financial Analytics & Decision Driven (FADD)")


# --- 4. Main Header & Top KPI Metrics ---
st.title("Credit Card Customer Risk Analysis")
st.caption("Interactive Financial Intelligence & Default Risk Scoring Dashboard (5,993 Evaluated Accounts)")

total_cust = len(filtered_df)
if total_cust > 0:
    default_count = int(filtered_df["Is_Default"].sum()) if "Is_Default" in filtered_df.columns else 0
    default_rate = (default_count / total_cust) * 100
    
    high_risk_count = (filtered_df["Risk_Level"] == "High Risk").sum()
    high_risk_pct = (high_risk_count / total_cust) * 100
    
    avg_limit = filtered_df["Credit_Limit"].mean()
else:
    default_rate, high_risk_pct, avg_limit = 0, 0, 0

col1, col2, col3, col4 = st.columns(4)
with col1:
    st.metric(label="Total Customers", value=f"{total_cust:,}", help="Total evaluated accounts matching filters")
with col2:
    st.metric(label="Default Rate", value=f"{default_rate:.1f}%", help="Percentage of accounts that actually defaulted next month")
with col3:
    st.metric(label="High Risk Share", value=f"{high_risk_pct:.1f}%", help="Accounts predicted with >60% default probability")
with col4:
    st.metric(label="Avg Credit Limit", value=f"NT$ {avg_limit:,.0f}", help="Average credit line assigned to borrowers")

st.markdown("---")


# --- 5. Dashboard Tabs Navigation ---
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📊 Executive Visuals",
    "🔍 Customer Explorer",
    "🧮 Live 'What-If' Simulator",
    "🤖 ML Model Benchmarks",
    "📘 Power BI & Project Guide"
])


# ==================== TAB 1: EXECUTIVE VISUALS ====================
with tab1:
    st.subheader("Portfolio Risk Distribution & Delinquency Insights")
    
    col_v1, col_v2 = st.columns([1, 1.3])
    
    with col_v1:
        st.markdown("**1. Risk Level Segmentation**")
        risk_counts = filtered_df["Risk_Level"].value_counts().reset_index()
        risk_counts.columns = ["Risk_Level", "Count"]
        
        # Try plotting with Plotly if available; otherwise native bar chart
        try:
            import plotly.express as px
            fig_pie = px.pie(
                risk_counts,
                values="Count",
                names="Risk_Level",
                hole=0.55,
                color="Risk_Level",
                color_discrete_map={
                    "Low Risk": "#10b981",
                    "Medium Risk": "#f59e0b",
                    "High Risk": "#ef4444"
                }
            )
            fig_pie.update_layout(margin=dict(t=10, b=10, l=10, r=10), height=300)
            st.plotly_chart(fig_pie, use_container_width=True)
        except ImportError:
            st.bar_chart(risk_counts.set_index("Risk_Level"))

    with col_v2:
        st.markdown("**2. Default Rate by Payment Delinquency Status**")
        if "Payment_Status" in df.columns:
            delay_stats = df.groupby("Payment_Status")["Is_Default"].mean().reset_index()
            delay_stats["Default_Rate_Pct"] = delay_stats["Is_Default"] * 100
            
            try:
                import plotly.express as px
                fig_bar = px.bar(
                    delay_stats,
                    x="Payment_Status",
                    y="Default_Rate_Pct",
                    labels={"Payment_Status": "Repayment Status", "Default_Rate_Pct": "Default Rate (%)"},
                    color="Default_Rate_Pct",
                    color_continuous_scale="Reds"
                )
                fig_bar.update_layout(margin=dict(t=10, b=10, l=10, r=10), height=300)
                st.plotly_chart(fig_bar, use_container_width=True)
            except ImportError:
                st.bar_chart(delay_stats.set_index("Payment_Status")["Default_Rate_Pct"])
    
    st.markdown("---")
    
    col_v3, col_v4 = st.columns([1.2, 1])
    with col_v3:
        st.markdown("**3. Customer Distribution by Age Group**")
        if "Age_Group" in df.columns:
            age_counts = df["Age_Group"].value_counts().sort_index().reset_index()
            age_counts.columns = ["Age_Group", "Accounts"]
            try:
                import plotly.express as px
                fig_age = px.bar(
                    age_counts,
                    x="Age_Group",
                    y="Accounts",
                    color_discrete_sequence=["#2563eb"]
                )
                fig_age.update_layout(margin=dict(t=10, b=10, l=10, r=10), height=280)
                st.plotly_chart(fig_age, use_container_width=True)
            except ImportError:
                st.bar_chart(age_counts.set_index("Age_Group"))

    with col_v4:
        st.markdown("**💡 Key Management Takeaways**")
        st.info(
            "• **The 2-Month Tipping Point**: Customers paying on time default only **13.6%** of the time. Once payment is delayed for 2 months, default risk jumps to **68.3%**!\n\n"
            "• **Credit Line Correlation**: High-limit borrowers default significantly less often (-0.36 correlation with default).\n\n"
            "• **Prime Portfolio Segment**: **40.6%** of accounts are Low Risk and eligible for credit line expansions."
        )


# ==================== TAB 2: CUSTOMER EXPLORER ====================
with tab2:
    st.subheader("Customer Risk Dossier & Inspection")
    st.caption("Search, filter, and inspect individual accounts from the 5,993 evaluated test records.")
    
    # Display Dataframe
    display_cols = [c for c in [
        "Customer_ID", "Credit_Limit", "Age", "Education", "Payment_Status", 
        "Delayed_Months", "Default_Probability_Pct", "Risk_Level", "Actual_Default"
    ] if c in filtered_df.columns]
    
    st.dataframe(
        filtered_df[display_cols].head(100),
        use_container_width=True,
        hide_index=True
    )
    st.caption(f"Showing first 100 rows of {len(filtered_df):,} matching accounts.")
    
    # Single Customer Deep-Dive
    st.markdown("---")
    st.markdown("#### 🔍 Inspect Specific Customer Profile")
    sample_ids = filtered_df["Customer_ID"].head(20).tolist()
    if sample_ids:
        selected_cust_id = st.selectbox("Select Customer ID to inspect", sample_ids)
        cust_row = filtered_df[filtered_df["Customer_ID"] == selected_cust_id].iloc[0]
        
        c_p1, c_p2, c_p3 = st.columns(3)
        with c_p1:
            st.metric("Credit Limit", f"NT$ {cust_row['Credit_Limit']:,.0f}")
            st.write(f"**Age**: {cust_row['Age']} yrs")
            if "Education" in cust_row: st.write(f"**Education**: {cust_row['Education']}")
        with c_p2:
            st.metric("Default Probability", f"{cust_row['Default_Probability_Pct']:.1f}%")
            st.write(f"**Risk Tier**: {cust_row['Risk_Level']}")
            if "Actual_Default" in cust_row: st.write(f"**Actual Outcome**: {cust_row['Actual_Default']}")
        with c_p3:
            st.metric("Delayed Months", f"{cust_row['Delayed_Months']} mo")
            if "Credit_Utilization_Pct" in cust_row:
                st.write(f"**Credit Utilization**: {cust_row['Credit_Utilization_Pct']:.1f}%")
            
            # Advisory Action
            if cust_row["Risk_Level"] == "Low Risk":
                st.success("✅ **Prime Customer**: Safe for standard terms and limit increase.")
            elif cust_row["Risk_Level"] == "Medium Risk":
                st.warning("⚠️ **Caution**: Monitor payments; freeze credit limit.")
            else:
                st.error("🚨 **High Delinquency Alert**: Restrict charges and escalate to recovery.")


# ==================== TAB 3: LIVE RISK SIMULATOR ====================
with tab3:
    st.subheader("Interactive 'What-If' Credit Risk Simulator")
    st.markdown("Adjust parameters to calculate real-time default risk scores and automated underwriting decisions.")
    
    # Presets
    p_col1, p_col2, p_col3 = st.columns(3)
    preset = None
    if p_col1.button("🌟 Preset: Prime Borrower"): preset = "prime"
    if p_col2.button("⚖️ Preset: Average Revolver"): preset = "average"
    if p_col3.button("⚠️ Preset: Stressed Delinquent"): preset = "stressed"
    
    # Default values based on preset
    def_limit = 350000 if preset == "prime" else (120000 if preset == "average" else 50000)
    def_age = 36 if preset == "prime" else (29 if preset == "average" else 33)
    def_pay = 0 if preset in ["prime", "average"] else 2
    def_delays = 0 if preset in ["prime", "average"] else 3
    def_bill = 35000 if preset == "prime" else (65000 if preset == "average" else 48000)
    def_payment = 35000 if preset == "prime" else (12000 if preset == "average" else 2000)

    sim_c1, sim_c2 = st.columns([1.2, 1])
    
    with sim_c1:
        s_limit = st.slider("Credit Limit (NT$)", 10000, 800000, def_limit, step=10000)
        s_age = st.slider("Customer Age", 18, 75, def_age)
        s_pay0 = st.selectbox(
            "Recent Payment Status",
            [0, 1, 2, 3],
            index=def_pay,
            format_func=lambda x: "On Time / Current (0)" if x == 0 else ("Delayed 1 Month (1)" if x == 1 else ("Delayed 2 Months (2)" if x == 2 else "Delayed 3+ Months (3)"))
        )
        s_delays = st.slider("Delayed Months (Past 6 Months)", 0, 6, def_delays)
        s_bill = st.slider("Latest Monthly Bill (NT$)", 0, 300000, def_bill, step=5000)
        s_payment = st.slider("Latest Payment (NT$)", 0, 200000, def_payment, step=2000)
        
        utilization = (s_bill / s_limit * 100) if s_limit > 0 else 0
        st.write(f"**Calculated Credit Utilization**: `{utilization:.1f}%`")

    with sim_c2:
        # Calibrated logistic prediction formula
        logit = -1.15 - (0.0000028 * s_limit) + (0.006 * s_age) + (0.45 * s_pay0) + (0.42 * s_delays) + (0.012 * utilization) - (0.000003 * s_payment)
        prob = 1 / (1 + np.exp(-logit))
        prob_pct = int(np.clip(prob * 100, 5, 96))
        
        st.markdown("#### Estimated Default Probability")
        st.progress(prob_pct / 100)
        st.title(f"{prob_pct}%")
        
        if prob_pct < 30:
            st.success("🟢 **LOW RISK TIER** (< 30%)")
            st.write("✅ **Recommendation**: Approved. Customer has sound credit habits and healthy liquidity. Safe for line increases.")
        elif prob_pct <= 60:
            st.warning("🟡 **MEDIUM RISK TIER** (30% - 60%)")
            st.write("⚠️ **Recommendation**: Caution. Maintain existing limit, require automated minimum payment alerts.")
        else:
            st.error("🔴 **HIGH RISK TIER** (> 60%)")
            st.write("❌ **Recommendation**: Decline / Action Required. High probability of default next month. Trim credit limit and restrict new charges.")


# ==================== TAB 4: ML MODEL BENCHMARKS ====================
with tab4:
    st.subheader("Supervised Machine Learning Model Benchmarks")
    st.caption("Comparison of 4 algorithms trained on 29,965 accounts (80/20 stratified split).")
    
    st.dataframe(model_comparison.set_index("Model"), use_container_width=True)
    
    # Model comparison bar chart
    try:
        import plotly.express as px
        df_melt = model_comparison.melt(id_vars="Model", var_name="Metric", value_name="Score")
        fig_m = px.bar(
            df_melt,
            x="Model",
            y="Score",
            color="Metric",
            barmode="group",
            height=340
        )
        st.plotly_chart(fig_m, use_container_width=True)
    except ImportError:
        st.bar_chart(model_comparison.set_index("Model"))
        
    st.info(
        "🏆 **Why Random Forest is the Champion Model**:\n\n"
        "1. **Highest ROC-AUC (77.40%)**: Superior discriminative ability to separate true defaulters from non-defaulters.\n"
        "2. **Balanced Class Weighting**: Handles the ~22% default class imbalance effectively without overfitting.\n"
        "3. **Non-linear Delinquency Modeling**: Captures compounding risk across historical payment delays (`DELAYED_MONTHS`)."
    )


# ==================== TAB 5: POWER BI GUIDE ====================
with tab5:
    st.subheader("📘 How to Recreate in Power BI Desktop (10-Minute Guide)")
    
    st.markdown("""
    If you also want to build this dashboard in **Microsoft Power BI**:
    
    1. **Import Clean CSV**:
       * In Power BI Desktop, click **Get Data** &rarr; **Text/CSV**.
       * Select `outputs/power_bi_credit_card_risk.csv`.
       * Click **Load** (it is pre-cleaned with human-friendly labels).
       
    2. **Paste 4 Core DAX Measures**:
    """)
    st.code("""
Total Customers = COUNTROWS('power_bi_credit_card_risk')

Default Rate % = AVERAGE('power_bi_credit_card_risk'[Is_Default])

High Risk % = DIVIDE(CALCULATE([Total Customers], 'power_bi_credit_card_risk'[Risk_Level] = "High Risk"), [Total Customers], 0)

Avg Credit Limit = AVERAGE('power_bi_credit_card_risk'[Credit_Limit])
    """, language="dax")
    
    st.markdown("""
    3. **Insert Core Visuals**:
       * **4 Cards (Top)**: `[Total Customers]`, `[Default Rate %]`, `[High Risk %]`, `[Avg Credit Limit]`.
       * **Donut Chart**: Legend = `Risk_Level`, Values = `[Total Customers]`.
       * **Clustered Bar Chart**: Y-Axis = `Payment_Status`, X-Axis = `[Default Rate %]`.
       * **Clustered Column Chart**: X-Axis = `Age_Group`, Y-Axis = `[Total Customers]`.
       * **Table**: Drag `Customer_ID`, `Credit_Limit`, `Age`, `Payment_Status`, `Risk_Level`, `Actual_Default`.
    """)

