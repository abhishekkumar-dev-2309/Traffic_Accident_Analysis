
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

from pathlib import Path
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score
)

# --------------------------------------------------
# STEP 1: Define file paths
# --------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = BASE_DIR / "output"

test_file = OUTPUT_DIR / "testing_dataset.csv"
model_file = OUTPUT_DIR / "accident_severity_random_forest.pkl"

# --------------------------------------------------
# STEP 2: Load testing dataset and trained model
# --------------------------------------------------

print("Loading testing dataset...")

test_data = pd.read_csv(test_file)

print("Loading trained Random Forest model...")

model = joblib.load(model_file)

# --------------------------------------------------
# STEP 3: Separate input and actual output
# --------------------------------------------------

X_test = test_data.drop(columns=["Severity"])
y_test = test_data["Severity"]

# Convert Boolean features into integers
boolean_features = [
    "Traffic_Signal",
    "Junction",
    "Crossing",
    "Railway",
    "Stop",
    "Bump",
    "Amenity"
]

for column in boolean_features:
    X_test[column] = X_test[column].astype("int8")

print("Testing records:", len(X_test))

# --------------------------------------------------
# STEP 4: Predict accident severity
# --------------------------------------------------

print("\nPredicting accident severity...")

y_pred = model.predict(X_test)

print("Prediction completed!")

# --------------------------------------------------
# STEP 5: Calculate evaluation metrics
# --------------------------------------------------

accuracy = accuracy_score(y_test, y_pred)

balanced_accuracy = balanced_accuracy_score(
    y_test, y_pred
)

macro_f1 = f1_score(
    y_test, y_pred, average="macro"
)

weighted_f1 = f1_score(
    y_test, y_pred, average="weighted"
)

print("\n========== MODEL PERFORMANCE ==========")

print(f"Accuracy: {accuracy:.4f}")
print(f"Balanced Accuracy: {balanced_accuracy:.4f}")
print(f"Macro F1-Score: {macro_f1:.4f}")
print(f"Weighted F1-Score: {weighted_f1:.4f}")

# --------------------------------------------------
# STEP 6: Classification report
# --------------------------------------------------

report = classification_report(
    y_test,
    y_pred,
    labels=[1, 2, 3, 4],
    zero_division=0
)

print("\n========== CLASSIFICATION REPORT ==========")
print(report)

# Save classification report
report_dict = classification_report(
    y_test,
    y_pred,
    labels=[1, 2, 3, 4],
    output_dict=True,
    zero_division=0
)

pd.DataFrame(report_dict).transpose().to_csv(
    OUTPUT_DIR / "classification_report.csv"
)

# --------------------------------------------------
# STEP 7: Generate confusion matrix
# --------------------------------------------------

cm = confusion_matrix(
    y_test,
    y_pred,
    labels=[1, 2, 3, 4]
)

print("\nConfusion Matrix:")
print(cm)

# --------------------------------------------------
# STEP 8: Plot confusion matrix
# --------------------------------------------------

plt.figure(figsize=(8, 6))

sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=[1, 2, 3, 4],
    yticklabels=[1, 2, 3, 4]
)

plt.title("Random Forest - Confusion Matrix")
plt.xlabel("Predicted Severity")
plt.ylabel("Actual Severity")

plt.tight_layout()

plt.savefig(
    OUTPUT_DIR / "confusion_matrix.png",
    dpi=300
)

plt.close()

# --------------------------------------------------
# STEP 9: Save predictions
# --------------------------------------------------

prediction_results = pd.DataFrame({
    "Actual_Severity": y_test.to_numpy(),
    "Predicted_Severity": y_pred
})

prediction_results.to_csv(
    OUTPUT_DIR / "severity_predictions.csv",
    index=False
)

print("\nEvaluation completed successfully!")
print("All results saved in output folder.")
