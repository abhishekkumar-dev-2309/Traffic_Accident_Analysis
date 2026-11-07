
# PROGRAM 42.15: TEMPORAL PREDICTION ERROR ANALYSIS

import pandas as pd
import numpy as np

from pathlib import Path
from sklearn.metrics import confusion_matrix

# -----------------------------------
# STEP 1: DEFINE PROJECT PATHS
# -----------------------------------

BASE_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = BASE_DIR / "output"

# -----------------------------------
# STEP 2: LOAD PREDICTIONS
# -----------------------------------

print("Loading temporal prediction results...")

predictions = pd.read_csv(
    OUTPUT_DIR / "temporal_predictions.csv"
)

actual = predictions["Actual_Severity"]
predicted = predictions["Predicted_Severity"]

# -----------------------------------
# STEP 3: COMPARE ACTUAL AND PREDICTED
# -----------------------------------

predictions["Correct_Prediction"] = (
    actual == predicted
)

predictions["Prediction_Status"] = np.where(
    predictions["Correct_Prediction"],
    "Correct",
    "Incorrect"
)

print("\n========== PREDICTION SUMMARY ==========")

print(predictions["Prediction_Status"].value_counts())

# -----------------------------------
# STEP 4: CLASS-WISE ERROR ANALYSIS
# -----------------------------------

print("\n========== CLASS-WISE ERROR ANALYSIS ==========")

for severity in [1, 2, 3, 4]:

    actual_class = actual == severity

    correct = (
        (actual == severity) &
        (predicted == severity)
    ).sum()

    total = actual_class.sum()

    incorrect = total - correct

    recall = correct / total if total else 0

    print(f"\nSeverity {severity}")

    print("Actual records:", total)

    print("Correct predictions:", correct)

    print("Incorrect predictions:", incorrect)

    print("Recall:", round(recall, 4))

# -----------------------------------
# STEP 5: CONFUSION MATRIX
# -----------------------------------

cm = confusion_matrix(
    actual,
    predicted,
    labels=[1, 2, 3, 4]
)

confusion_df = pd.DataFrame(
    cm,
    index=["Actual_1", "Actual_2", "Actual_3", "Actual_4"],
    columns=[
        "Predicted_1",
        "Predicted_2",
        "Predicted_3",
        "Predicted_4"
    ]
)

print("\n========== CONFUSION MATRIX ==========")

print(confusion_df)

# -----------------------------------
# STEP 6: ACTUAL DISTRIBUTION
# -----------------------------------

print("\n========== ACTUAL DISTRIBUTION ==========")

print(actual.value_counts().sort_index())

# -----------------------------------
# STEP 7: PREDICTED DISTRIBUTION
# -----------------------------------

print("\n========== PREDICTED DISTRIBUTION ==========")

print(predicted.value_counts().sort_index())

# -----------------------------------
# STEP 8: SAVE ERROR ANALYSIS
# -----------------------------------

predictions.to_csv(
    OUTPUT_DIR / "temporal_error_analysis.csv",
    index=False
)

confusion_df.to_csv(
    OUTPUT_DIR / "temporal_error_confusion_matrix.csv"
)

print("\nError analysis files saved successfully!")

print("\nProgram 42.15 completed successfully!")
