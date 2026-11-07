

import pandas as pd
from sklearn.model_selection import train_test_split
from pathlib import Path

# Step 1: Define file paths
BASE_DIR = Path(__file__).resolve().parent

input_file = BASE_DIR / "output" / "accident_prediction_dataset.csv"
output_dir = BASE_DIR / "output"

# Step 2: Load prepared dataset
print("Loading dataset...")

data = pd.read_csv(input_file)

print("Dataset loaded successfully!")
print("Total records:", len(data))
print("Total columns:", len(data.columns))

# Step 3: Separate input features and target variable
X = data.drop(columns=["Severity"])
y = data["Severity"]

print("\nInput features:", X.shape)
print("Target variable:", y.shape)

# Step 4: Divide data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

# Step 5: Combine features and target for saving
train_data = X_train.copy()
train_data["Severity"] = y_train

test_data = X_test.copy()
test_data["Severity"] = y_test

# Step 6: Save training and testing datasets
train_data.to_csv(output_dir / "training_dataset.csv", index=False)
test_data.to_csv(output_dir / "testing_dataset.csv", index=False)

# Step 7: Display results
print("\nTrain-Test Split Completed!")
print("Training records:", len(train_data))
print("Testing records:", len(test_data))

print("\nTraining severity distribution:")
print(train_data["Severity"].value_counts().sort_index())

print("\nTesting severity distribution:")
print(test_data["Severity"].value_counts().sort_index())

print("\nFiles saved successfully in output folder.")
