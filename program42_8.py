
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

from pathlib import Path
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    classification_report,
    confusion_matrix
)

# --------------------------------------------------
# STEP 1: Define paths
# --------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = BASE_DIR / "output"

test_file = OUTPUT_DIR / "testing_dataset.csv"

original_file = OUTPUT_DIR / "accident_severity_random_forest.pkl"

improved_file = OUTPUT_DIR / "improved_random_forest_model.pkl"

# --------------------------------------------------
# STEP 2: Load testing dataset
# --------------------------------------------------

print("Loading original testing dataset...")

test_data = pd.read_csv(test_file)

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
# STEP 3: Load both models
# --------------------------------------------------

print("\nLoading original model...")
original_model = joblib.load(original_file)

print("Loading improved model...")
improved_model = joblib.load(improved_file)

# --------------------------------------------------
# STEP 4: Define evaluation function
# --------------------------------------------------

def evaluate_model(model, model_name):

    print("\nEvaluating:", model_name)

    predictions = model.predict(X_test)

    accuracy = accuracy_score(y_test, predictions)

    balanced_accuracy = balanced_accuracy_score(
        y_test, predictions
    )

    macro_f1 = f1_score(
        y_test, predictions, average="macro"
    )

    weighted_f1 = f1_score(
        y_test, predictions, average="weighted"
    )

    print("Accuracy:", round(accuracy, 4))
    print("Balanced Accuracy:", round(balanced_accuracy, 4))
    print("Macro F1:", round(macro_f1, 4))
    print("Weighted F1:", round(weighted_f1, 4))

    print("\nClassification Report:")
    print(classification_report(
        y_test,
        predictions,
        labels=[1, 2, 3, 4],
        zero_division=0
    ))

    return {
        "Model": model_name,
        "Accuracy": accuracy,
        "Balanced_Accuracy": balanced_accuracy,
        "Macro_F1": macro_f1,
        "Weighted_F1": weighted_f1
    }, predictions

# --------------------------------------------------
# STEP 5: Evaluate original model
# --------------------------------------------------

original_results, original_predictions = evaluate_model(
    original_model,
    "Original Random Forest"
)

# --------------------------------------------------
# STEP 6: Evaluate improved model
# --------------------------------------------------

improved_results, improved_predictions = evaluate_model(
    improved_model,
    "Improved Random Forest"
)

# --------------------------------------------------
# STEP 7: Compare both models
# --------------------------------------------------

comparison = pd.DataFrame([
    original_results,
    improved_results
])

print("\n========== FINAL MODEL COMPARISON ==========")
print(comparison.to_string(index=False))

comparison.to_csv(
    OUTPUT_DIR / "final_model_comparison.csv",
    index=False
)

# --------------------------------------------------
# STEP 8: Plot metric comparison
# --------------------------------------------------

metrics = [
    "Accuracy",
    "Balanced_Accuracy",
    "Macro_F1",
    "Weighted_F1"
]

comparison.set_index("Model")[metrics].plot(
    kind="bar",
    figsize=(11, 6)
)

plt.title("Original vs Improved Random Forest")
plt.ylabel("Score")
plt.ylim(0, 1)
plt.xticks(rotation=0)
plt.legend(title="Metrics")
plt.tight_layout()

plt.savefig(
    OUTPUT_DIR / "final_model_comparison.png",
    dpi=300
)

plt.close()

# --------------------------------------------------
# STEP 9: Generate confusion matrices
# --------------------------------------------------

fig, axes = plt.subplots(1, 2, figsize=(13, 5))

models_data = [
    (original_predictions, "Original Random Forest"),
    (improved_predictions, "Improved Random Forest")
]

for ax, (predictions, title) in zip(axes, models_data):

    cm = confusion_matrix(
        y_test,
        predictions,
        labels=[1, 2, 3, 4]
    )

    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=[1, 2, 3, 4],
        yticklabels=[1, 2, 3, 4],
        ax=ax
    )

    ax.set_title(title)
    ax.set_xlabel("Predicted Severity")
    ax.set_ylabel("Actual Severity")

plt.tight_layout()

plt.savefig(
    OUTPUT_DIR / "final_confusion_matrix_comparison.png",
    dpi=300
)

plt.close()

# --------------------------------------------------
# STEP 10: Save predictions
# --------------------------------------------------

prediction_results = pd.DataFrame({
    "Actual_Severity": y_test.to_numpy(),
    "Original_Prediction": original_predictions,
    "Improved_Prediction": improved_predictions
})

prediction_results.to_csv(
    OUTPUT_DIR / "final_model_predictions.csv",
    index=False
)

print("\nFinal evaluation completed successfully!")
print("All comparison files saved in output folder.")
