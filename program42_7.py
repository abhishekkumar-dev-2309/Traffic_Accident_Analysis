
import pandas as pd
import numpy as np
import joblib

from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    classification_report
)

# --------------------------------------------------
# STEP 1: Define project paths
# --------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = BASE_DIR / "output"

train_file = OUTPUT_DIR / "training_dataset.csv"

print("Loading training dataset...")

data = pd.read_csv(train_file)

X = data.drop(columns=["Severity"])
y = data["Severity"]

# --------------------------------------------------
# STEP 2: Convert Boolean features into integers
# --------------------------------------------------

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
# STEP 3: Create validation dataset
# --------------------------------------------------

X_train, X_val, y_train, y_val = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("\nTraining records:", len(X_train))
print("Validation records:", len(X_val))

# --------------------------------------------------
# STEP 4: Define feature categories
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
# STEP 5: Create preprocessing pipeline
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
# STEP 6: Calculate custom class weights
# --------------------------------------------------

class_counts = y_train.value_counts().sort_index()

print("\nClass distribution in training data:")
print(class_counts)

def calculate_class_weights(alpha):

    weights = (
        len(y_train) / (len(class_counts) * class_counts)
    ) ** alpha

    # Normalize weights so their average is 1
    weights = weights / weights.mean()

    return {
        int(class_label): float(weight)
        for class_label, weight in weights.items()
    }

# --------------------------------------------------
# STEP 7: Define three experiments
# --------------------------------------------------

experiments = {
    "Balanced_Subsample": "balanced_subsample",
    "Custom_Alpha_0.5": calculate_class_weights(0.5),
    "Custom_Alpha_0.75": calculate_class_weights(0.75)
}

results = []
best_score = -1
best_model = None
best_name = None

# --------------------------------------------------
# STEP 8: Train and compare models
# --------------------------------------------------

for experiment_name, class_weight in experiments.items():

    print("\n----------------------------------")
    print("Training:", experiment_name)
    print("----------------------------------")

    classifier = RandomForestClassifier(
        n_estimators=80,
        max_depth=20,
        min_samples_leaf=2,
        class_weight=class_weight,
        random_state=42,
        n_jobs=-1
    )

    model = Pipeline(steps=[
        ("preprocessing", preprocessor),
        ("classifier", classifier)
    ])

    model.fit(X_train, y_train)

    # Predict validation data
    y_pred = model.predict(X_val)

    # Calculate performance
    accuracy = accuracy_score(y_val, y_pred)

    balanced_acc = balanced_accuracy_score(
        y_val, y_pred
    )

    macro_f1 = f1_score(
        y_val, y_pred, average="macro"
    )

    weighted_f1 = f1_score(
        y_val, y_pred, average="weighted"
    )

    print("Accuracy:", round(accuracy, 4))
    print("Balanced Accuracy:", round(balanced_acc, 4))
    print("Macro F1:", round(macro_f1, 4))
    print("Weighted F1:", round(weighted_f1, 4))

    results.append({
        "Experiment": experiment_name,
        "Accuracy": accuracy,
        "Balanced_Accuracy": balanced_acc,
        "Macro_F1": macro_f1,
        "Weighted_F1": weighted_f1
    })

    # Select model using Macro F1
    if macro_f1 > best_score:

        best_score = macro_f1
        best_model = model
        best_name = experiment_name

# --------------------------------------------------
# STEP 9: Compare all experiments
# --------------------------------------------------

results_df = pd.DataFrame(results)

results_df = results_df.sort_values(
    by="Macro_F1",
    ascending=False
)

print("\n========== EXPERIMENT COMPARISON ==========")
print(results_df.to_string(index=False))

results_df.to_csv(
    OUTPUT_DIR / "class_imbalance_comparison.csv",
    index=False
)

# --------------------------------------------------
# STEP 10: Save best model
# --------------------------------------------------

best_model_file = OUTPUT_DIR / "improved_random_forest_model.pkl"

joblib.dump(best_model, best_model_file)

print("\nBest experiment:", best_name)
print("Best validation Macro F1:", round(best_score, 4))

print("\nImproved model saved successfully!")
print(best_model_file)

# --------------------------------------------------
# STEP 11: Display best model classification report
# --------------------------------------------------

best_predictions = best_model.predict(X_val)

print("\n========== BEST MODEL CLASSIFICATION REPORT ==========")

print(classification_report(
    y_val,
    best_predictions,
    labels=[1, 2, 3, 4],
    zero_division=0
))

print("\nProgram 42.7 completed successfully!")
