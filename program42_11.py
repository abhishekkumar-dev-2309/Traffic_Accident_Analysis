
import pandas as pd
import numpy as np
import joblib

from pathlib import Path
from sklearn.model_selection import GroupShuffleSplit
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

# --------------------------------------------------
# STEP 1: Define paths
# --------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = BASE_DIR / "output"

input_file = OUTPUT_DIR / "accident_prediction_dataset.csv"

print("Loading prediction dataset...")

data = pd.read_csv(input_file)

print("Total records:", len(data))

# --------------------------------------------------
# STEP 2: Separate features and target
# --------------------------------------------------

X = data.drop(columns=["Severity"]).copy()
y = data["Severity"].copy()

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
    X[column] = X[column].astype("int8")

# --------------------------------------------------
# STEP 3: Create geographical groups
# --------------------------------------------------

# Divide locations into approximately 1-degree blocks

latitude_block = np.floor(X["Start_Lat"]).astype(int)
longitude_block = np.floor(X["Start_Lng"]).astype(int)

groups = (
    latitude_block.astype(str)
    + "_"
    + longitude_block.astype(str)
)

print("\nUnique geographical blocks:", groups.nunique())

# --------------------------------------------------
# STEP 4: Split by geographical groups
# --------------------------------------------------

splitter = GroupShuffleSplit(
    n_splits=1,
    test_size=0.20,
    random_state=42
)

train_indices, test_indices = next(
    splitter.split(X, y, groups=groups)
)

X_train = X.iloc[train_indices].copy()
X_test = X.iloc[test_indices].copy()

y_train = y.iloc[train_indices].copy()
y_test = y.iloc[test_indices].copy()

train_groups = set(groups.iloc[train_indices])
test_groups = set(groups.iloc[test_indices])

print("\nGeographic training records:", len(X_train))
print("Geographic testing records:", len(X_test))

print("Training geographical blocks:", len(train_groups))
print("Testing geographical blocks:", len(test_groups))

print(
    "Overlapping geographical blocks:",
    len(train_groups.intersection(test_groups))
)

# --------------------------------------------------
# STEP 5: Define features
# --------------------------------------------------

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

# --------------------------------------------------
# STEP 6: Calculate class weights
# --------------------------------------------------

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

print("\nGeographic training class distribution:")
print(class_counts)

print("\nClass weights:")
print(custom_weights)

# --------------------------------------------------
# STEP 7: Create preprocessing pipeline
# --------------------------------------------------

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

# --------------------------------------------------
# STEP 8: Create tuned Random Forest model
# --------------------------------------------------

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

# --------------------------------------------------
# STEP 9: Train geographic model
# --------------------------------------------------

print("\nTraining geographic Random Forest model...")
print("Please wait...")

model.fit(X_train, y_train)

print("\nGeographic model training completed!")

# --------------------------------------------------
# STEP 10: Predict unseen geographical records
# --------------------------------------------------

print("\nPredicting unseen geographical records...")

predictions = model.predict(X_test)

# --------------------------------------------------
# STEP 11: Evaluate performance
# --------------------------------------------------

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

print("\n========== GEOGRAPHIC EVALUATION ==========")

print("Accuracy:", round(accuracy, 4))
print("Balanced Accuracy:", round(balanced_acc, 4))
print("Macro F1:", round(macro_f1, 4))
print("Weighted F1:", round(weighted_f1, 4))

print("\nClassification Report:")

print(classification_report(
    y_test,
    predictions,
    labels=[1, 2, 3, 4],
    zero_division=0
))

# --------------------------------------------------
# STEP 12: Save evaluation results
# --------------------------------------------------

results = pd.DataFrame([{
    "Accuracy": accuracy,
    "Balanced_Accuracy": balanced_acc,
    "Macro_F1": macro_f1,
    "Weighted_F1": weighted_f1,
    "Training_Records": len(X_train),
    "Testing_Records": len(X_test),
    "Training_Blocks": len(train_groups),
    "Testing_Blocks": len(test_groups)
}])

results.to_csv(
    OUTPUT_DIR / "geographic_evaluation.csv",
    index=False
)

# Save confusion matrix
cm = confusion_matrix(
    y_test,
    predictions,
    labels=[1, 2, 3, 4]
)

pd.DataFrame(
    cm,
    index=["Actual_1", "Actual_2", "Actual_3", "Actual_4"],
    columns=["Predicted_1", "Predicted_2", "Predicted_3", "Predicted_4"]
).to_csv(
    OUTPUT_DIR / "geographic_confusion_matrix.csv"
)

# Save geographic model
joblib.dump(
    model,
    OUTPUT_DIR / "geographic_random_forest_model.pkl"
)

print("\nGeographic evaluation completed!")
print("All results saved successfully.")
