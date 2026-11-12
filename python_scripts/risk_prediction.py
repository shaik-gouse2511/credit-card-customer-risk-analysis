
import pandas as pd
import numpy as np
import joblib
import os

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

# Create folders
os.makedirs("outputs", exist_ok=True)
os.makedirs("models", exist_ok=True)

# Load feature-engineered dataset
df = pd.read_csv("outputs/credit_card_features.csv")

# Create customer identifier for dashboard
customer_ids = pd.Series(df.index + 1, index=df.index)

# Separate input and target
X = df.drop("DEFAULT", axis=1)
y = df["DEFAULT"]

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# Create Random Forest model
model = RandomForestClassifier(
    n_estimators=100,
    max_depth=12,
    min_samples_split=20,
    class_weight="balanced",
    random_state=42,
    n_jobs=-1
)

# Train model
model.fit(X_train, y_train)

print("Random Forest model trained!")

# Predict default status
predictions = model.predict(X_test)

# Predict default probability
probabilities = model.predict_proba(X_test)[:, 1]

# Create prediction dataset
risk_df = X_test.copy()

risk_df.insert(
    0,
    "CUSTOMER_ID",
    customer_ids.loc[X_test.index]
)

risk_df["ACTUAL_DEFAULT"] = y_test
risk_df["PREDICTED_DEFAULT"] = predictions
risk_df["DEFAULT_PROBABILITY"] = probabilities

# Assign risk categories
risk_df["RISK_LEVEL"] = pd.cut(
    probabilities,
    bins=[-0.001, 0.30, 0.60, 1.0],
    labels=["Low Risk", "Medium Risk", "High Risk"]
)

# Save predictions
risk_df.to_csv(
    "outputs/customer_risk_predictions.csv",
    index=False
)

# Save trained model
joblib.dump(
    model,
    "models/random_forest_model.pkl"
)

print("\nCustomer Risk Prediction Completed!")

print("\nRisk Distribution:")
print(risk_df["RISK_LEVEL"].value_counts())

print("\nSample Predictions:")
print(
    risk_df[
        [
            "CUSTOMER_ID",
            "ACTUAL_DEFAULT",
            "PREDICTED_DEFAULT",
            "DEFAULT_PROBABILITY",
            "RISK_LEVEL"
        ]
    ].head(10)
)

print("\nPrediction file saved successfully!")
print("Model saved successfully!")
