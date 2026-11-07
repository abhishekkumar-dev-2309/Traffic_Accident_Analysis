
import pandas as pd
from pathlib import Path

# -----------------------------------
# STEP 1: Define paths
# -----------------------------------

BASE_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = BASE_DIR / "output"

INPUT_FILE = OUTPUT_DIR / "temporal_prediction_dataset.csv"

# -----------------------------------
# STEP 2: Load temporal dataset
# -----------------------------------

print("Loading temporal dataset...")

data = pd.read_csv(INPUT_FILE)

data["Start_Time"] = pd.to_datetime(
    data["Start_Time"],
    utc=True,
    errors="coerce"
)

data = data.dropna(
    subset=["Start_Time", "Severity"]
).copy()

# -----------------------------------
# STEP 3: Sort chronologically
# -----------------------------------

data = data.sort_values(
    "Start_Time"
).reset_index(drop=True)

# -----------------------------------
# STEP 4: Define cutoff year
# -----------------------------------

cutoff_year = 2022

train_data = data[
    data["Start_Time"].dt.year < cutoff_year
].copy()

test_data = data[
    data["Start_Time"].dt.year >= cutoff_year
].copy()

# -----------------------------------
# STEP 5: Remove unnecessary columns
# -----------------------------------

# Keep Year as a feature.
# Remove original timestamp and temporary
# helper columns if present.

train_data = train_data.drop(
    columns=["Start_Time"]
)

test_data = test_data.drop(
    columns=["Start_Time"]
)

# -----------------------------------
# STEP 6: Separate features and target
# -----------------------------------

X_train = train_data.drop(
    columns=["Severity"]
)

y_train = train_data["Severity"]

X_test = test_data.drop(
    columns=["Severity"]
)

y_test = test_data["Severity"]

# -----------------------------------
# STEP 7: Display split information
# -----------------------------------

print("\n========== TEMPORAL SPLIT ==========")

print("Training records:", len(X_train))
print("Testing records:", len(X_test))

print("\nTraining year distribution:")
print(train_data["Year"].value_counts().sort_index())

print("\nTesting year distribution:")
print(test_data["Year"].value_counts().sort_index())

print("\nTraining severity distribution:")
print(y_train.value_counts().sort_index())

print("\nTesting severity distribution:")
print(y_test.value_counts().sort_index())

# -----------------------------------
# STEP 8: Save datasets
# -----------------------------------

train_data.to_csv(
    OUTPUT_DIR / "temporal_training_dataset.csv",
    index=False
)

test_data.to_csv(
    OUTPUT_DIR / "temporal_testing_dataset.csv",
    index=False
)

print("\nTemporal train-test split completed!")

print("Training and testing datasets saved.")
