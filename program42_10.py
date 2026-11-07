
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
# STEP 1: Define project paths
# --------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = BASE_DIR / "output"

test_file = OUTPUT_DIR / "testing_dataset.csv"

original_file = OUTPUT_DIR / "accident_severity_random_forest.pkl"

improved_file = OUTPUT_DIR / "improved_random_forest_model.pkl"

tuned_file = OUTPUT_DIR / "tuned_random_forest_model.pkl"

# --------------------------------------------------
# STEP 2: Load testing dataset
# --------------------------------------------------

print("Loading testing dataset...")

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
# STEP 3: Load all three models
# --------------------------------------------------

print("\nLoading three trained models...")

models = {
    "Original Random Forest": joblib.load(original_file),
    "Improved Random Forest": joblib.load(improved_file),
    "Tuned Random Forest": joblib.load(tuned_file)
}

# --------------------------------------------------
# STEP 4: Evaluate all models
# --------------------------------------------------

results = []
all_predictions = {}

for model_name, model in models.items():

    print("\n================================")
    print("Evaluating:", model_name)
    print("================================")

    predictions = model.predict(X_test)

    all_predictions[model_name] = predictions

    accuracy = accuracy_score(y_test, predictions)

    balanced_acc = balanced_accuracy_score(
        y_test, predictions
    )

    macro_f1 = f1_score(
        y_test, predictions, average="macro"
    )

    weighted_f1 = f1_score(
        y_test, predictions, average="weighted"
    )

    print("Accuracy:", round(accuracy, 4))
    print("Balanced Accuracy:", round(balanced_acc, 4))
    print("Macro F1:", round(macro_f1, 4))
    print("Weighted F1:", round(weighted_f1, 4))

    results.append({
        "Model": model_name,
        "Accuracy": accuracy,
        "Balanced_Accuracy": balanced_acc,
        "Macro_F1": macro_f1,
        "Weighted_F1": weighted_f1
    })

    # Save class-wise report
    report = classification_report(
        y_test,
        predictions,
        labels=[1, 2, 3, 4],
        output_dict=True,
        zero_division=0
    )

    report_df = pd.DataFrame(report).transpose()

    safe_name = model_name.lower().replace(" ", "_")

    report_df.to_csv(
        OUTPUT_DIR / f"{safe_name}_report.csv"
    )

# --------------------------------------------------
# STEP 5: Compare all models
# --------------------------------------------------

comparison = pd.DataFrame(results)

comparison = comparison.sort_values(
    by="Macro_F1",
    ascending=False
)

print("\n========== FINAL MODEL COMPARISON ==========")
print(comparison.to_string(index=False))

comparison.to_csv(
    OUTPUT_DIR / "three_model_comparison.csv",
    index=False
)

# --------------------------------------------------
# STEP 6: Plot metric comparison
# --------------------------------------------------

metrics = [
    "Accuracy",
    "Balanced_Accuracy",
    "Macro_F1",
    "Weighted_F1"
]

comparison.set_index("Model")[metrics].plot(
    kind="bar",
    figsize=(12, 6)
)

plt.title("Comparison of Three Random Forest Models")
plt.ylabel("Performance Score")
plt.ylim(0, 1)
plt.xticks(rotation=15)
plt.tight_layout()

plt.savefig(
    OUTPUT_DIR / "three_model_comparison.png",
    dpi=300
)

plt.close()

# --------------------------------------------------
# STEP 7: Compare confusion matrices
# --------------------------------------------------

fig, axes = plt.subplots(1, 3, figsize=(17, 5))

for ax, (model_name, predictions) in zip(
    axes, all_predictions.items()
):

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

    ax.set_title(model_name)
    ax.set_xlabel("Predicted Severity")
    ax.set_ylabel("Actual Severity")

plt.tight_layout()

plt.savefig(
    OUTPUT_DIR / "three_model_confusion_matrices.png",
    dpi=300
)

plt.close()

# --------------------------------------------------
# STEP 8: Compare class-wise recall
# --------------------------------------------------

recall_data = []

for model_name, predictions in all_predictions.items():

    report = classification_report(
        y_test,
        predictions,
        labels=[1, 2, 3, 4],
        output_dict=True,
        zero_division=0
    )

    for severity in ["1", "2", "3", "4"]:

        recall_data.append({
            "Model": model_name,
            "Severity": severity,
            "Recall": report[severity]["recall"]
        })

recall_df = pd.DataFrame(recall_data)

recall_df.to_csv(
    OUTPUT_DIR / "three_model_class_recall.csv",
    index=False
)

plt.figure(figsize=(10, 6))

sns.barplot(
    data=recall_df,
    x="Severity",
    y="Recall",
    hue="Model"
)

plt.title("Class-wise Recall Comparison")
plt.xlabel("Accident Severity")
plt.ylabel("Recall")
plt.tight_layout()

plt.savefig(
    OUTPUT_DIR / "three_model_recall_comparison.png",
    dpi=300
)

plt.close()

# --------------------------------------------------
# STEP 9: Save all model predictions
# --------------------------------------------------

prediction_results = pd.DataFrame({
    "Actual_Severity": y_test.to_numpy()
})

for model_name, predictions in all_predictions.items():

    column_name = model_name.replace(" ", "_")

    prediction_results[column_name] = predictions

prediction_results.to_csv(
    OUTPUT_DIR / "three_model_predictions.csv",
    index=False
)

# --------------------------------------------------
# STEP 10: Final summary
# --------------------------------------------------

best_model_name = comparison.iloc[0]["Model"]

print("\nBest model by Macro F1:", best_model_name)

print("\nProgram 42.10 completed successfully!")

print("All comparison files saved in output folder.")
