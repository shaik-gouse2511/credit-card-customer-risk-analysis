
import pandas as pd
import matplotlib.pyplot as plt
import os

# Create output folder
os.makedirs("outputs", exist_ok=True)

# Store actual model evaluation results
results = {
    "Model": [
        "Logistic Regression",
        "Decision Tree",
        "Random Forest",
        "SVM"
    ],
    "Accuracy": [
        0.7442015685,
        0.7530452194,
        0.7687301852,
        0.7492074086
    ],
    "Precision": [
        0.4430379747,
        0.4562002275,
        0.4814585909,
        0.4502529511
    ],
    "Recall": [
        0.6070889844,
        0.6048265460,
        0.5874811463,
        0.6040723982
    ],
    "F1_Score": [
        0.5122494432,
        0.5201003763,
        0.5292119565,
        0.5159420289
    ],
    "ROC_AUC": [
        0.7526429108,
        0.7605234888,
        0.7739710577,
        0.7513731243
    ]
}

# Create DataFrame
df = pd.DataFrame(results)

# Convert metrics into percentages for visualization
metrics = [
    "Accuracy", "Precision", "Recall",
    "F1_Score", "ROC_AUC"
]

print("\nMODEL COMPARISON")
print(df.round(4).to_string(index=False))

# Save comparison CSV
df.to_csv(
    "outputs/model_comparison.csv",
    index=False
)

# Create comparison chart
df_plot = df.set_index("Model")[metrics] * 100

df_plot.plot(
    kind="bar",
    figsize=(12, 6)
)

plt.title("Machine Learning Model Performance Comparison")
plt.ylabel("Performance (%)")
plt.xlabel("ML Algorithms")
plt.xticks(rotation=15)
plt.legend(loc="lower right")
plt.tight_layout()

plt.savefig("outputs/model_comparison.png")
plt.show()

print("\nModel comparison completed!")
print("CSV and chart saved successfully!")
