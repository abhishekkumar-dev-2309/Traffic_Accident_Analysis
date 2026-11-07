
# PROGRAM 3: DESCRIPTIVE STATISTICAL ANALYSIS

import pandas as pd
from pathlib import Path

# Step 1: Define dataset path
BASE_DIR = Path(__file__).resolve().parent

DATASET_PATH = (
    BASE_DIR / "dataset" / "US_Accidents_March23.csv"
)

# Step 2: Load dataset
data = pd.read_csv(
    DATASET_PATH,
    nrows=1000,
    low_memory=False
)

print("========== DESCRIPTIVE STATISTICS ==========")

# Step 3: Numerical columns
numerical_data = data.select_dtypes(
    include=["number"]
)

print("\nNumerical Columns:")
print(numerical_data.columns.tolist())

# Step 4: Calculate statistical measures
print("\n========== STATISTICAL SUMMARY ==========")

print(numerical_data.describe())

# Step 5: Calculate mean
print("\n========== MEAN ==========")

print(numerical_data.mean())

# Step 6: Calculate median
print("\n========== MEDIAN ==========")

print(numerical_data.median())

# Step 7: Calculate standard deviation
print("\n========== STANDARD DEVIATION ==========")

print(numerical_data.std())

# Step 8: Calculate minimum and maximum
print("\n========== MINIMUM VALUES ==========")

print(numerical_data.min())

print("\n========== MAXIMUM VALUES ==========")

print(numerical_data.max())

# Step 9: Severity distribution
print("\n========== ACCIDENT SEVERITY DISTRIBUTION ==========")

print(data["Severity"].value_counts().sort_index())

print("\nProgram 3 completed successfully!")
