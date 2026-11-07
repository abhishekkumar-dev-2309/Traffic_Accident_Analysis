
import pandas as pd
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
# STEP 3: Create training and validation datasets
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
# STEP 4: Define numerical and categorical features
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
# STEP 5: Calculate custom class weights
# --------------------------------------------------

class_counts = y_train.value_counts().sort_index()

alpha = 0.75

weights = (
    len(y_train) / (len(class_counts) * class_counts)
) ** alpha

weights = weights / weights.mean()

custom_weights = {
    int(label): float(weight)
    for label, weight in weights.items()
}

print("\nCustom class weights:")
print(custom_weights)

# --------------------------------------------------
# STEP 6: Define hyperparameter configurations
# --------------------------------------------------

experiments = [
    {
        "Name": "Experiment_1",
        "n_estimators": 80,
        "max_depth": 20,
        "min_samples_leaf": 2,
        "max_features": "sqrt",
        "class_weight": custom_weights
    },
    {
        "Name": "Experiment_2",
        "n_estimators": 100,
        "max_depth": 15,
        "min_samples_leaf": 2,
        "max_features": "sqrt",
        "class_weight": custom_weights
    },
    {
        "Name": "Experiment_3",
        "n_estimators": 100,
        "max_depth": 25,
        "min_samples_leaf": 4,
        "max_features": "sqrt",
        "class_weight": custom_weights
    },
    {
        "Name": "Experiment_4",
        "n_estimators": 120,
        "max_depth": 20,
        "min_samples_leaf": 4,
        "max_features": "sqrt",
        "class_weight": "balanced_subsample"
    }
]

# --------------------------------------------------
# STEP 7: Create preprocessing function
# --------------------------------------------------

def create_preprocessor():

    numerical_pipeline = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median"))
    ])

    categorical_pipeline = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore"))
    ])

    return ColumnTransformer(transformers=[
        ("numerical", numerical_pipeline, numerical_features),
        ("categorical", categorical_pipeline, categorical_features)
    ])

# --------------------------------------------------
# STEP 8: Train and evaluate configurations
# --------------------------------------------------

results = []

best_score = -1
best_model = None
best_name = None

for config in experiments:

    print("\n================================")
    print("Training:", config["Name"])
    print("================================")

    classifier = RandomForestClassifier(
        n_estimators=config["n_estimators"],
        max_depth=config["max_depth"],
        min_samples_leaf=config["min_samples_leaf"],
        max_features=config["max_features"],
        class_weight=config["class_weight"],
        random_state=42,
        n_jobs=-1
    )

    model = Pipeline(steps=[
        ("preprocessing", create_preprocessor()),
        ("classifier", classifier)
    ])

    model.fit(X_train, y_train)

    predictions = model.predict(X_val)

    accuracy = accuracy_score(y_val, predictions)

    balanced_acc = balanced_accuracy_score(
        y_val, predictions
    )

    macro_f1 = f1_score(
        y_val, predictions, average="macro"
    )

    weighted_f1 = f1_score(
        y_val, predictions, average="weighted"
    )

    print("Accuracy:", round(accuracy, 4))
    print("Balanced Accuracy:", round(balanced_acc, 4))
    print("Macro F1:", round(macro_f1, 4))
    print("Weighted F1:", round(weighted_f1, 4))

    results.append({
        "Experiment": config["Name"],
        "Trees": config["n_estimators"],
        "Max_Depth": config["max_depth"],
        "Min_Samples_Leaf": config["min_samples_leaf"],
        "Max_Features": config["max_features"],
        "Accuracy": accuracy,
        "Balanced_Accuracy": balanced_acc,
        "Macro_F1": macro_f1,
        "Weighted_F1": weighted_f1
    })

    if macro_f1 > best_score:

        best_score = macro_f1
        best_model = model
        best_name = config["Name"]

# --------------------------------------------------
# STEP 9: Compare all experiments
# --------------------------------------------------

results_df = pd.DataFrame(results)

results_df = results_df.sort_values(
    by="Macro_F1",
    ascending=False
)

print("\n========== HYPERPARAMETER COMPARISON ==========")
print(results_df.to_string(index=False))

results_df.to_csv(
    OUTPUT_DIR / "hyperparameter_tuning_results.csv",
    index=False
)

# --------------------------------------------------
# STEP 10: Save best model
# --------------------------------------------------

best_model_file = (
    OUTPUT_DIR / "tuned_random_forest_model.pkl"
)

joblib.dump(best_model, best_model_file)

print("\nBest configuration:", best_name)
print("Best validation Macro F1:", round(best_score, 4))

# --------------------------------------------------
# STEP 11: Display best model classification report
# --------------------------------------------------

best_predictions = best_model.predict(X_val)

print("\n========== BEST MODEL REPORT ==========")

print(classification_report(
    y_val,
    best_predictions,
    labels=[1, 2, 3, 4],
    zero_division=0
))

print("\nHyperparameter tuning completed!")
print("Best model saved successfully.")
