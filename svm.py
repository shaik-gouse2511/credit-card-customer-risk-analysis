
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    classification_report,
    confusion_matrix
)

# Load dataset
df = pd.read_csv("outputs/credit_card_features.csv")

# Separate features and target
X = df.drop("DEFAULT", axis=1)
y = df["DEFAULT"]

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# Feature scaling
scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Create SVM model
model = SVC(
    kernel="rbf",
    C=1.0,
    class_weight="balanced",
    random_state=42
)

# Train model
model.fit(X_train_scaled, y_train)

print("SVM Training Completed!")

# Predictions
y_pred = model.predict(X_test_scaled)

# Decision scores for ROC-AUC
y_scores = model.decision_function(X_test_scaled)

# Evaluation
print("\n--- SVM RESULTS ---")

print("Accuracy:", accuracy_score(y_test, y_pred))
print("Precision:", precision_score(y_test, y_pred))
print("Recall:", recall_score(y_test, y_pred))
print("F1 Score:", f1_score(y_test, y_pred))
print("ROC-AUC:", roc_auc_score(y_test, y_scores))

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))
