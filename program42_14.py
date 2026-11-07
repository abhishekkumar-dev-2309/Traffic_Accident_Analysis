
# PROGRAM 42.14: TEMPORAL RANDOM FOREST MODEL

import pandas as pd
import numpy as np
import joblib

from pathlib import Path

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    classification_report,
    confusion_matrix
)

# --------------------------------------
# STEP 1: DEFINE PROJECT PATHS
# --------------------------------------

BASE_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = BASE_DIR / "output"

# --------------------------------------
# STEP 2: LOAD TRAINING AND TESTING DATA
# --------------------------------------

print("Loading temporal datasets...")

train_data = pd.read_csv(
    OUTPUT_DIR / "temporal_training_dataset.csv"
)

test_data = pd.read_csv(
    OUTPUT_DIR / "temporal_testing_dataset.csv"
)

print("Training records:", len(train_data))
print("Testing records:", len(test_data))

# --------------------------------------
# STEP 3: SEPARATE FEATURES AND TARGET
# --------------------------------------

X_train = train_data.drop(columns=["Severity"])
y_train = train_data["Severity"]

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
    X_train[column] = X_train[column].astype("int8")
    X_test[column] = X_test[column].astype("int8")

# --------------------------------------
# STEP 4: DEFINE FEATURE COLUMNS
# --------------------------------------

numerical_features = [
    "Start_Lat",
    "Start_Lng",
    "Temperature(F)",
    "Humidity(%)",
    "Pressure(in)",
    "Visibility(mi)",
    "Wind_Speed(mph)",
    "Precipitation(in)",
    "Hour",
    "Day_of_Week",
    "Month",
    "Year",
    "Is_Weekend",
    "Traffic_Signal",
    "Junction",
    "Crossing",
    "Railway",
    "Stop",
    "Bump",
    "Amenity"
]

categorical_features = [
    "Weather_Condition",
    "Sunrise_Sunset"
]

# --------------------------------------
# STEP 5: CALCULATE CLASS WEIGHTS
# --------------------------------------

class_counts = y_train.value_counts().sort_index()

alpha = 0.75

weights = (
    len(y_train) /
    (len(class_counts) * class_counts)
) ** alpha

weights = weights / weights.mean()

custom_weights = {
    int(label): float(weight)
    for label, weight in weights.items()
}

print("\nTraining class weights:")
print(custom_weights)

# --------------------------------------
# STEP 6: CREATE PREPROCESSING PIPELINE
# --------------------------------------

numerical_pipeline = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="median"))
])

categorical_pipeline = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OneHotEncoder(handle_unknown="ignore"))
])

preprocessor = ColumnTransformer(transformers=[
    ("numerical", numerical_pipeline, numerical_features),
    ("categorical", categorical_pipeline, categorical_features)
])

# --------------------------------------
# STEP 7: CREATE RANDOM FOREST MODEL
# --------------------------------------

classifier = RandomForestClassifier(
    n_estimators=100,
    max_depth=25,
    min_samples_leaf=4,
    max_features="sqrt",
    class_weight=custom_weights,
    random_state=42,
    n_jobs=-1
)

model = Pipeline(steps=[
    ("preprocessing", preprocessor),
    ("classifier", classifier)
])

# --------------------------------------
# STEP 8: TRAIN THE MODEL
# --------------------------------------

print("\nTraining temporal Random Forest model...")
print("Please wait...")

model.fit(X_train, y_train)

print("Training completed!")

# --------------------------------------
# STEP 9: MAKE PREDICTIONS
# --------------------------------------

print("\nPredicting 2022-2023 records...")

predictions = model.predict(X_test)

# --------------------------------------
# STEP 10: EVALUATE MODEL
# --------------------------------------

accuracy = accuracy_score(y_test, predictions)

balanced_acc = balanced_accuracy_score(
    y_test, predictions
)

macro_f1 = f1_score(
    y_test, predictions,
    average="macro",
    zero_division=0
)

weighted_f1 = f1_score(
    y_test, predictions,
    average="weighted",
    zero_division=0
)

print("\n========== TEMPORAL MODEL RESULTS ==========")

print("Accuracy:", round(accuracy, 4))
print("Balanced Accuracy:", round(balanced_acc, 4))
print("Macro F1:", round(macro_f1, 4))
print("Weighted F1:", round(weighted_f1, 4))

# --------------------------------------
# STEP 11: CLASSIFICATION REPORT
# --------------------------------------

report = classification_report(
    y_test,
    predictions,
    labels=[1, 2, 3, 4],
    output_dict=True,
    zero_division=0
)

print("\nClassification Report:")

print(classification_report(
    y_test,
    predictions,
    labels=[1, 2, 3, 4],
    zero_division=0
))

pd.DataFrame(report).transpose().to_csv(
    OUTPUT_DIR / "temporal_classification_report.csv"
)

# --------------------------------------
# STEP 12: CONFUSION MATRIX
# --------------------------------------

cm = confusion_matrix(
    y_test,
    predictions,
    labels=[1, 2, 3, 4]
)

pd.DataFrame(
    cm,
    index=["Actual_1", "Actual_2", "Actual_3", "Actual_4"],
    columns=[
        "Predicted_1",
        "Predicted_2",
        "Predicted_3",
        "Predicted_4"
    ]
).to_csv(
    OUTPUT_DIR / "temporal_confusion_matrix.csv"
)

# --------------------------------------
# STEP 13: SAVE EVALUATION METRICS
# --------------------------------------

results = pd.DataFrame([{
    "Accuracy": accuracy,
    "Balanced_Accuracy": balanced_acc,
    "Macro_F1": macro_f1,
    "Weighted_F1": weighted_f1
}])

results.to_csv(
    OUTPUT_DIR / "temporal_model_evaluation.csv",
    index=False
)

# --------------------------------------
# STEP 14: SAVE PREDICTIONS
# --------------------------------------

prediction_results = pd.DataFrame({
    "Actual_Severity": y_test.values,
    "Predicted_Severity": predictions
})

prediction_results.to_csv(
    OUTPUT_DIR / "temporal_predictions.csv",
    index=False
)

# --------------------------------------
# STEP 15: SAVE TRAINED MODEL
# --------------------------------------

joblib.dump(
    model,
    OUTPUT_DIR / "temporal_random_forest_model.pkl"
)

print("\nAll temporal evaluation files saved!")

print("\nProgram 42.14 completed successfully!")
